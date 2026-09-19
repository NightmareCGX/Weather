"""Staging area and aggregate pass for ensemble stores.

Why a staging area exists
-------------------------
An aggregate is a function of the *whole* member set: the bins are defined on
``(x - mean) / std``, so they need the final mean and standard deviation, and a quantile
function needs every member value. Members therefore cannot be aggregated as they arrive.

They also cannot be held in memory until the set is complete: one lead of one variable is
``30 * 721 * 1440 * 4 B ~ 125 MB``, and a lead's 14 variables would be ``~1.7 GB`` against a
4 GB container budget that already runs near its limit. The pipeline's existing memory bound
is its staging semaphore, which assumes one decoded member dataset per in-flight item.

So members are written to a **staging prefix** under the same store first -- reusing the
existing member-shard container and write path unchanged -- and a separate pass per
``(variable, lead)`` reads that variable's 30 staging shards, computes the aggregate, writes
one aggregate object, and drops the staging shards. Peak residency is one variable's member
stack plus working copies, measured at ~459 MB.

The staging namespace
---------------------
``__staging__/v1/<variable>/mem<member>_L<lead>.shard`` under the store root. It is a sibling
of the commit namespace (``__commit__/``), not a child, so it can never be mistaken for
committed data: the inventory, the marker evidence validator and the API readers all address
objects by key, and none of them enumerates the store root for data. The version segment lets
a staging layout change coexist with in-flight waves instead of requiring a quiesce.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import xarray as xr
from domain.aggregate import (
    MEMBER_COUNT_SCALE,
    AggregateSpec,
    compute_aggregate,
    finite_member_count,
)
from domain.shard_format import (
    SHARD_V1_MAGIC,
    TRAILER_SIZE,
    ShardFormatError,
    parse_index,
    parse_trailer,
)
from numcodecs import Zstd  # type: ignore[import-untyped]

from ingestion.core.aggregate_writer import (
    AggregateShardLayout,
    AggregateWriterError,
    aggregate_store_relative_key,
    encode_aggregate_shard,
    layout_for_spec,
)
from ingestion.core.store_io import StoreAccessError, StoreIO, StoreRef

logger = logging.getLogger(__name__)

#: Root of the staging namespace. Deliberately NOT under ``__commit__/``.
STAGING_ROOT: str = "__staging__"

#: Version segment of the staging namespace.
STAGING_VERSION: str = "v1"

#: Staging object name: ``mem<member>_L<lead>.shard``.
_STAGING_NAME_RE = re.compile(r"^mem(?P<member>\d{3})_L(?P<lead>\d{4})\.shard$")

#: Decoder for member shards. A zstd frame is self-describing, so no level is negotiated.
_DECODER = Zstd()

# ``StoreRef`` is re-exported from the store layer, which owns the union of accepted store
#: shapes (an ``s3://`` URL, a local path, or an in-memory mapping).
_STOREREF_IS_REEXPORTED: bool = StoreRef is not None


class StagingError(RuntimeError):
    """Raised when the staging area cannot satisfy a requested aggregate."""


@dataclass(frozen=True)
class StagedMember:
    """One member's plane as recovered from a staging shard.

    Attributes:
        member: Ensemble member index.
        lead_time_hours: Forecast lead the object belongs to.
        plane: ``(lat, lon)`` float32 field.
    """

    member: int
    lead_time_hours: int
    plane: npt.NDArray[np.float32]


def staging_relative_key(variable_code: str, member: int, lead_time_hours: int) -> str:
    """Store-relative key of one member's staging object."""
    return (
        f"{STAGING_ROOT}/{STAGING_VERSION}/{variable_code}/"
        f"mem{int(member):03d}_L{int(lead_time_hours):04d}.shard"
    )


def staging_prefix(variable_code: str) -> str:
    """Store-relative prefix holding one variable's staging objects across all leads."""
    return f"{STAGING_ROOT}/{STAGING_VERSION}/{variable_code}/"


def parse_staging_name(name: str) -> tuple[int, int]:
    """Parse ``(member, lead_time_hours)`` from a staging object name.

    Raises:
        StagingError: if the name is not a staging object name. A committed shard is
            addressed by a derived key, whereas the staging area is *enumerated*, so an
            unparseable object means the area holds something unexpected and must not be
            skipped silently.
    """
    base = name.rsplit("/", 1)[-1]
    match = _STAGING_NAME_RE.match(base)
    if match is None:
        raise StagingError(f"not a staging object name: {name!r}")
    return int(match.group("member")), int(match.group("lead"))


def member_plane_from_shard(
    shard_bytes: bytes,
    *,
    grid_lat: int,
    grid_lon: int,
    chunk_lat: int = 100,
    chunk_lon: int = 100,
) -> npt.NDArray[np.float32]:
    """Reassemble one member's 2-D plane from a ``sharded_v1`` container.

    Deliberately mirrors the ingestion reader's reassembly, because the equivalence the
    aggregate pass is verified against is "the plane stored, reassembled the way the store
    reassembles it" -- a different reassembly would make the comparison meaningless.

    Args:
        shard_bytes: Full container bytes.
        grid_lat: Grid latitude extent.
        grid_lon: Grid longitude extent.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.

    Raises:
        StagingError: if the container is malformed or its chunk count disagrees with the
            supplied geometry.
    """
    if len(shard_bytes) < TRAILER_SIZE:
        raise StagingError(f"staging shard too short: {len(shard_bytes)} bytes")
    try:
        trailer = parse_trailer(shard_bytes[-TRAILER_SIZE:])
    except ShardFormatError as exc:
        raise StagingError(str(exc)) from exc
    if trailer.magic != SHARD_V1_MAGIC:
        raise StagingError(
            "staging shards are expected to be sharded_v1 member containers; "
            f"found magic 0x{trailer.magic:08x}"
        )
    index_size = trailer.index_byte_size
    if len(shard_bytes) < TRAILER_SIZE + index_size:
        raise StagingError(
            f"staging shard declares a {index_size}-byte index but is only "
            f"{len(shard_bytes)} bytes"
        )
    index_bytes = shard_bytes[-(TRAILER_SIZE + index_size) : -TRAILER_SIZE]
    entries = parse_index(index_bytes, trailer.num_chunks)

    lat_chunks = -(-grid_lat // chunk_lat)
    lon_chunks = -(-grid_lon // chunk_lon)
    expected_chunks = lat_chunks * lon_chunks
    if trailer.num_chunks != expected_chunks:
        raise StagingError(
            f"staging shard holds {trailer.num_chunks} chunks but the grid implies "
            f"{expected_chunks}"
        )

    plane = np.full((grid_lat, grid_lon), np.nan, dtype=np.float32)
    ordinal = 0
    for row in range(lat_chunks):
        r0 = row * chunk_lat
        r1 = min(r0 + chunk_lat, grid_lat)
        for col in range(lon_chunks):
            c0 = col * chunk_lon
            c1 = min(c0 + chunk_lon, grid_lon)
            offset, length = entries[ordinal]
            ordinal += 1
            if length == 0:
                continue
            raw = _DECODER.decode(shard_bytes[offset : offset + length])
            chunk = np.frombuffer(raw, dtype=np.float32).reshape(chunk_lat, chunk_lon)
            plane[r0:r1, c0:c1] = chunk[: r1 - r0, : c1 - c0]
    return plane


def staged_objects(store: StoreRef, variable_code: str) -> dict[tuple[int, int], str]:
    """Map ``(member, lead)`` to staging key for one variable.

    Raises:
        StagingError: if a staged object's name cannot be parsed, so a corrupt staging area
            is reported rather than silently partially aggregated.
    """
    io = StoreIO(store)
    prefix = staging_prefix(variable_code)
    found: dict[tuple[int, int], str] = {}
    for key in io.list_under(prefix):
        if key.endswith(".shard"):
            found[parse_staging_name(key)] = key
    return found


def staged_members_for_lead(store: StoreRef, variable_code: str, lead_time_hours: int) -> list[int]:
    """Member indices staged for one ``(variable, lead)``, sorted."""
    return sorted(
        member
        for (member, lead) in staged_objects(store, variable_code)
        if lead == lead_time_hours
    )


def collect_member_planes(
    store: StoreRef,
    variable_code: str,
    lead_time_hours: int,
    *,
    grid_lat: int,
    grid_lon: int,
    chunk_lat: int = 100,
    chunk_lon: int = 100,
) -> list[StagedMember]:
    """Read every staged member plane for one ``(variable, lead)``.

    Returns:
        Planes sorted by member index, so the aggregate does not depend on listing order.

    Raises:
        StagingError: if no member is staged for the requested lead.
    """
    io = StoreIO(store)
    staged = staged_objects(store, variable_code)
    members = sorted(member for (member, lead) in staged if lead == lead_time_hours)
    if not members:
        raise StagingError(
            f"no staged members for {variable_code!r} at lead {lead_time_hours}h"
        )

    planes: list[StagedMember] = []
    for member in members:
        try:
            raw = io.read(staged[(member, lead_time_hours)])
        except StoreAccessError as exc:
            raise StagingError(f"cannot read staged member {member}: {exc}") from exc
        planes.append(
            StagedMember(
                member=member,
                lead_time_hours=lead_time_hours,
                plane=member_plane_from_shard(
                    raw,
                    grid_lat=grid_lat,
                    grid_lon=grid_lon,
                    chunk_lat=chunk_lat,
                    chunk_lon=chunk_lon,
                ),
            )
        )
    return planes


def aggregate_from_planes(
    planes: Iterable[npt.NDArray[np.float32]],
    *,
    spec: AggregateSpec,
    grid_lat: int,
    grid_lon: int,
    expected_members: int | None = None,
) -> bytes:
    """Compute and encode the aggregate container from member planes.

    The stored field vector is the encoding's own fields **preceded by the per-cell finite-member
    count** (``domain.field_layout``). The container records that count per cell rather than per
    container because one member can be missing at one cell and present at its neighbour, and the
    count is what the serving coverage rule and the reported ``member_count`` are read from.

    Fields are quantised to int16 at their own scales before encoding. That is not an
    optimisation that can be deferred: measured on real GEFS members it is 17.73 MB -> 13.54 MB
    for a 34-field container, and the domain has carried the quantiser and its clipping guard all
    along. Each field's step differs -- a 314 K mean at a bin probability's 0.001 would need
    314000 and overflow -- which is why the descriptor stores the per-field marker rather than a
    single scale.

    The scales come from the spec being encoded, and the reader derives them from the variable's
    approved spec. That pair agrees because :func:`aggregate_staged_lead` -- the only production
    caller -- obtains its spec from ``spec_for(variable)`` and checks it against the variable's
    stored layout before calling this.

    Args:
        planes: One plane per member. Order does not affect the result.
        spec: Which statistics to store.
        grid_lat: Grid latitude extent.
        grid_lon: Grid longitude extent.
        expected_members: The contract's member count, which the per-cell coverage floor is
            measured against. Defaults to the number of planes supplied.

    Returns:
        The ``sharded_v2`` container bytes.

    Raises:
        StagingError: if no plane is supplied, the shapes disagree with the grid, or a field
            would exceed its quantisation scale.
    """
    materialised = [np.asarray(plane, dtype=np.float32) for plane in planes]
    if not materialised:
        raise StagingError("at least one member plane is required to aggregate")
    stack = np.stack(materialised)
    if stack.shape[1:] != (grid_lat, grid_lon):
        raise StagingError(
            f"member planes have shape {stack.shape[1:]}, expected {(grid_lat, grid_lon)}"
        )

    fields = [finite_member_count(stack)]
    fields.extend(
        compute_aggregate(
            stack,
            spec,
            expected_members=expected_members or stack.shape[0],
        )
    )
    scales = (MEMBER_COUNT_SCALE, *spec.field_scales)

    layout = layout_for_spec(spec, grid_lat=grid_lat, grid_lon=grid_lon, n_fields=len(fields))
    try:
        return encode_aggregate_shard(
            fields,
            layout,
            member_count=int(stack.shape[0]),
            field_scales=scales,
        )
    except AggregateWriterError as exc:
        raise StagingError(f"cannot encode the aggregate for {spec.kind}: {exc}") from exc


def aggregate_staged_lead(
    store: StoreRef,
    variable_code: str,
    lead_time_hours: int,
    *,
    spec: AggregateSpec,
    grid_lat: int,
    grid_lon: int,
    chunk_lat: int = 100,
    chunk_lon: int = 100,
    drop_staging: bool = True,
    expected_members: int | None = None,
) -> tuple[str, int]:
    """Aggregate one ``(variable, lead)``'s staged members and write the aggregate object.

    Args:
        store: Store root.
        variable_code: Variable to aggregate.
        lead_time_hours: Forecast lead.
        spec: Statistics to store.
        grid_lat: Grid latitude extent.
        grid_lon: Grid longitude extent.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.
        drop_staging: Remove the member staging objects once the aggregate is written.
        expected_members: When given, refuse to publish a partial aggregate. Defaults to
            aggregating whatever is staged, which is what progressive publication wants.

    Returns:
        ``(aggregate_relative_key, member_count)``.

    Raises:
        StagingError: if nothing is staged, or ``expected_members`` is not met.
    """
    io = StoreIO(store)
    staged = collect_member_planes(
        store,
        variable_code,
        lead_time_hours,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        chunk_lat=chunk_lat,
        chunk_lon=chunk_lon,
    )
    if expected_members is not None and len(staged) != expected_members:
        raise StagingError(
            f"{variable_code!r} lead {lead_time_hours}h has {len(staged)} staged members, "
            f"expected {expected_members}"
        )

    container = aggregate_from_planes(
        [item.plane for item in staged],
        spec=spec,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        expected_members=expected_members or len(staged),
    )
    key = aggregate_store_relative_key(variable_code, lead_time_hours)
    io.write(key, container)

    if drop_staging:
        keys_by_member = staged_objects(store, variable_code)
        io.delete_many(
            [
                keys_by_member[(item.member, lead_time_hours)]
                for item in staged
                if (item.member, lead_time_hours) in keys_by_member
            ]
        )
    return key, len(staged)


def aggregate_layout_for(
    spec: AggregateSpec,
    *,
    grid_lat: int,
    grid_lon: int,
    chunk_lat: int = 100,
    chunk_lon: int = 100,
    n_fields: int | None = None,
) -> AggregateShardLayout:
    """Layout an aggregate shard will have, for callers that need it before reading.

    ``n_fields`` defaults to the spec's own count plus one, because every stored container
    carries the per-cell member count ahead of the encoding's fields.
    """
    return layout_for_spec(
        spec,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        chunk_lat=chunk_lat,
        chunk_lon=chunk_lon,
        n_fields=spec.n_fields + 1 if n_fields is None else n_fields,
    )


def staging_object_is_v1_container(
    store: StoreRef,
    variable_code: str,
    lead_time_hours: int,
    member: int,
) -> bool:
    """Whether a specific staging object exists and declares a v1 member container.

    Lets the aggregate pass fail early on a corrupt staging object rather than part-way
    through a 30-member read.
    """
    io = StoreIO(store)
    key = staging_relative_key(variable_code, member, lead_time_hours)
    try:
        raw = io.read(key)
    except StoreAccessError:
        return False
    if len(raw) < TRAILER_SIZE:
        return False
    try:
        trailer = parse_trailer(raw[-TRAILER_SIZE:])
    except ShardFormatError:
        return False
    return trailer.magic == SHARD_V1_MAGIC


def staged_lead_keys(store: StoreRef, variable_code: str, leads: Sequence[int]) -> list[str]:
    """Staging keys for the given leads, for a caller that wants to inspect or clear them."""
    staged = staged_objects(store, variable_code)
    wanted = set(leads)
    return sorted(key for (_member, lead), key in staged.items() if lead in wanted)


__all__ = [
    "STAGING_ROOT",
    "STAGING_VERSION",
    "StagedMember",
    "StagingError",
    "aggregate_from_planes",
    "aggregate_layout_for",
    "aggregate_lead_all_variables",
    "aggregate_staged_lead",
    "collect_member_planes",
    "member_plane_from_shard",
    "parse_staging_name",
    "staged_lead_keys",
    "staged_members_for_lead",
    "stage_region",
    "staged_objects",
    "staged_objects_by_variable",
    "staging_object_is_v1_container",
    "staging_prefix",
    "staging_relative_key",
]


def stage_region(
    dataset: xr.Dataset,
    store: StoreRef,
    *,
    member: int,
    lead_time_hours: int,
    data_vars: Sequence[str] | None = None,
) -> list[str]:
    """Write one member region's variables into the staging area.

    Reuses the member-shard encoder unchanged, so a staged object is byte-identical to the
    serving shard for the same data. That equality is what makes the aggregate pass's
    equivalence property meaningful: it computes from the same bytes a reader would see.

    Must be called for every committed ensemble member region. The aggregate pass assumes the
    invariant "a committed member has a staged member", and without it a lead could be
    aggregated from a partial member set without anyone noticing.

    Args:
        dataset: The normalized single-lead, single-member dataset.
        store: Store root.
        member: Upstream member identity (``1..30``).
        lead_time_hours: Forecast lead.
        data_vars: Variables to stage; defaults to every data variable present.

    Returns:
        The staging keys written, in encoder order.

    Raises:
        StagingError: if the encoder produces no object, which would mean the dataset does
            not carry the expected 2-D variables.
    """
    from ingestion.core.zarr_writer import encode_region_sharded_v1

    encoded = encode_region_sharded_v1(
        dataset, member=member, lead_time_hours=lead_time_hours, data_vars=data_vars
    )
    if not encoded:
        raise StagingError(
            f"nothing encoded for member {member} lead {lead_time_hours}h; "
            "the dataset carries no 2-D variables"
        )

    io = StoreIO(store)
    written: list[str] = []
    for key, blob in encoded:
        variable_code = key.split("/", 1)[0]
        target = staging_relative_key(variable_code, member, lead_time_hours)
        io.write(target, blob)
        written.append(target)
    return written


def aggregate_lead_all_variables(
    store: StoreRef,
    lead_time_hours: int,
    *,
    variables: Sequence[str] | None = None,
    grid_lat: int | None = None,
    grid_lon: int | None = None,
    chunk_lat: int = 100,
    chunk_lon: int = 100,
    drop_staging: bool = True,
) -> list[tuple[str, str, int]]:
    """Aggregate every classifiable variable staged for one lead.

    The pass is per ``(variable, lead)`` and deliberately sequential: a lead's variables
    aggregated concurrently would each hold a member stack, and 14 of those do not fit in the
    container's memory budget. One at a time bounds residency at a single variable's stack.

    Variables are aggregated in the order given, so the pass is reproducible and a partial
    failure leaves an inspectable prefix rather than an arbitrary subset.

    Args:
        store: Store root.
        lead_time_hours: Forecast lead to aggregate.
        variables: Variables to consider; defaults to every variable with staging objects.
        grid_lat: Grid latitude extent; defaults to the configured platform grid.
        grid_lon: Grid longitude extent; defaults to the configured platform grid.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.
        drop_staging: Remove each variable's staging objects once its aggregate is written.

    Returns:
        ``(variable, aggregate_key, member_count)`` per variable aggregated.

    Raises:
        StagingError: if nothing is staged for the lead at all, or a variable's geometry does
            not match the configured grid. A variable with no staging is skipped: not every
            lead carries every variable (the GEFS product omits the instantaneous
            precipitation rate), and that is normal rather than an error.
    """
    from domain.variable_class import VariableClassError, spec_for
    from ingestion.core.aggregate_writer import DEFAULT_CHUNK_LAT, DEFAULT_CHUNK_LON

    resolved_grid_lat = grid_lat or _grid("ENSEMBLE_AGGREGATE_GRID_LAT", 721)
    resolved_grid_lon = grid_lon or _grid("ENSEMBLE_AGGREGATE_GRID_LON", 1440)
    resolved_chunk_lat = chunk_lat or _grid("ENSEMBLE_AGGREGATE_CHUNK_LAT", DEFAULT_CHUNK_LAT)
    resolved_chunk_lon = chunk_lon or _grid("ENSEMBLE_AGGREGATE_CHUNK_LON", DEFAULT_CHUNK_LON)

    staged = staged_objects_by_variable(store)
    candidates = list(variables) if variables is not None else sorted(staged)
    leads_present = {lead for by_lead in staged.values() for (_member, lead) in by_lead}
    if lead_time_hours not in leads_present:
        raise StagingError(f"nothing staged for lead {lead_time_hours}h")

    results: list[tuple[str, str, int]] = []
    for variable in candidates:
        if lead_time_hours not in {lead for (_m, lead) in staged.get(variable, {})}:
            continue
        try:
            spec = spec_for(variable)
        except VariableClassError:
            # A flag is staged like any member variable and has no ``AggregateSpec``: its
            # approved representation is a per-cell exceedance fraction, which is a separate
            # pass. Leaving its staging in place is what keeps the fraction computable later;
            # it is also why every aggregate pass re-reads this variable and skips it again
            # rather than the staging quietly disappearing.
            logger.debug(
                "no aggregate spec for %s at lead %d (a flag, or unclassified); "
                "its staging is retained",
                variable,
                lead_time_hours,
            )
            continue
        key, count = aggregate_staged_lead(
            store,
            variable,
            lead_time_hours,
            spec=spec,
            grid_lat=resolved_grid_lat,
            grid_lon=resolved_grid_lon,
            chunk_lat=resolved_chunk_lat,
            chunk_lon=resolved_chunk_lon,
            drop_staging=drop_staging,
        )
        results.append((variable, key, count))
    return results


def staged_objects_by_variable(store: StoreRef) -> dict[str, dict[tuple[int, int], str]]:
    """Map every variable present in the staging area to its ``(member, lead) -> key`` map.

    Enumerates the staging root once. Staging is a namespace the platform owns entirely, so
    an object under it that is not a staging key is an error rather than something to skip.
    """
    io = StoreIO(store)
    prefix = f"{STAGING_ROOT}/{STAGING_VERSION}/"
    by_variable: dict[str, dict[tuple[int, int], str]] = {}
    for key in io.list_under(prefix):
        if not key.endswith(".shard"):
            continue
        relative = key[len(prefix) :]
        variable, separator, _name = relative.partition("/")
        if not separator or not variable:
            raise StagingError(
                f"staging object {key!r} is not under a variable segment; the staging "
                "namespace holds only objects this module wrote"
            )
        by_variable.setdefault(variable, {})[parse_staging_name(key)] = key
    return by_variable


def _grid(name: str, default: int) -> int:
    """Read a geometry setting, falling back to the stored default."""
    try:
        from ingestion.core.config import settings

        return int(getattr(settings, name, default))
    except Exception:  # noqa: BLE001 - configuration must not break an offline call
        return default
