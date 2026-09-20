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

import numpy as np
import numpy.typing as npt
import xarray as xr
from domain.field_layout import aggregate_fields_for
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
    aggregate_store_relative_key,
    encode_aggregate_shard,
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


def aggregate_staged_lead(
    store: StoreRef,
    variable_code: str,
    lead_time_hours: int,
    *,
    grid_lat: int,
    grid_lon: int,
    chunk_lat: int = 100,
    chunk_lon: int = 100,
    drop_staging: bool = True,
    expected_members: int | None = None,
    wave_leads: Sequence[int] | None = None,
) -> tuple[str, int]:
    """Build and write one ``(variable, lead)``'s container from its staged member sets.

    The container is the whole field vector -- the per-cell member count, the variable's
    distribution, and the supplementary groups its products read -- assembled by
    :func:`ingestion.core.aggregate_fields.build_container_fields`. Which fields those are is
    ``domain.field_layout``'s answer, and the same function supplies the per-field fixed-point
    steps the container is quantised at, so the write and read sides take them from one authority
    rather than from two that have to agree by inspection.

    Args:
        store: Store root.
        variable_code: Variable to aggregate.
        lead_time_hours: Forecast lead.
        grid_lat: Grid latitude extent.
        grid_lon: Grid longitude extent.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.
        drop_staging: Remove this variable's staging objects once the container is written. A
            caller aggregating several variables at one lead passes ``False`` and releases them
            together: the variables a container reads are not the variables it is written as.
        expected_members: The contract's member count. When given, refuse to publish a partial
            container. Defaults to the members actually staged, which is what progressive
            publication wants.
        wave_leads: The leads this wave is filling, so a predecessor that has not landed yet is
            refused rather than encoded as absent.

    Returns:
        ``(aggregate_relative_key, member_count)``.

    Raises:
        StagingError: if an input is not staged, or the container cannot be built or encoded.
    """
    from ingestion.core.aggregate_fields import (
        AggregateBuildError,
        MemberReader,
        build_container_fields,
    )

    reader = MemberReader(
        store,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        chunk_lat=chunk_lat,
        chunk_lon=chunk_lon,
    )
    own_members = reader.members(variable_code, lead_time_hours)
    if expected_members is not None:
        # The variable's own members, not its inputs': a derived variable has none of its own and
        # takes its member set from a component, so counting there would compare unlike things.
        if own_members and len(own_members) != expected_members:
            raise StagingError(
                f"{variable_code!r} lead {lead_time_hours}h has {len(own_members)} staged "
                f"members, expected {expected_members}"
            )

    try:
        fields, member_count = build_container_fields(
            reader,
            variable_code,
            lead_time_hours,
            expected_members=expected_members or len(own_members) or 1,
            wave_leads=wave_leads,
        )
    except AggregateBuildError as exc:
        raise StagingError(str(exc)) from exc

    layout = aggregate_fields_for(variable_code)
    container = encode_aggregate_shard(
        fields,
        AggregateShardLayout(
            n_fields=layout.n_fields,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            chunk_lat=chunk_lat,
            chunk_lon=chunk_lon,
        ),
        member_count=member_count,
        field_scales=layout.field_scales,
    )
    key = aggregate_store_relative_key(variable_code, lead_time_hours)
    StoreIO(store).write(key, container)

    if drop_staging:
        # This is the single-variable form, so it has no pass-wide listing to share and lists just
        # its own variable -- one LIST for one container.
        _release_staging(
            store,
            _staged_keys_for_lead(
                staged_objects_by_variable(store), [variable_code], lead_time_hours
            ),
        )
    return key, member_count


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
    "StagingError",
    "aggregate_lead_all_variables",
    "aggregate_staged_lead",
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
    expected_members: int | None = None,
    wave_leads: Sequence[int] | None = None,
) -> list[tuple[str, str, int]]:
    """Aggregate every classifiable variable staged for one lead.

    The pass is per ``(variable, lead)`` and deliberately sequential: a lead's variables
    aggregated concurrently would each hold a member stack, and 14 of those do not fit in the
    container's memory budget. One at a time bounds residency at a single variable's stack --
    measured at 1.5 GB for precipitation, the most expensive of the four group-bearing variables.

    Variables are aggregated in the order given, so the pass is reproducible and a partial
    failure leaves an inspectable prefix rather than an arbitrary subset. That ordering is also
    what makes the staging release correct, and it is why the release is deferred to the end of
    the pass: the variables a container *reads* are not the variables it is written as, so
    dropping a variable's staging as soon as its own container is written would destroy an input
    the next variable in the pass still needs. ``wind_10m`` reads a component's staging and the
    phase groups read the four flags, so with the release inline the wind container would lose
    ``wind_u_10m`` and a later precipitation container would find no flags at all -- and a
    publication that ran with a subset of the inputs is exactly what every refusal in
    :mod:`ingestion.core.aggregate_fields` exists to prevent.

    A **derived** variable -- one the platform serves but stores no members for -- is added to the
    pass whenever its inputs are staged, even when ``variables`` names an explicit set: the wind
    pair is a fifth of a cycle's member bytes and no caller's variable list contains ``wind_10m``,
    which is synthesised rather than stored. Which names those are is
    ``domain.variable_class.DERIVED_VARIABLES``, not an inference.

    Args:
        store: Store root.
        lead_time_hours: Forecast lead to aggregate.
        variables: Variables to consider; defaults to every variable with staging objects. A
            derived variable is added regardless, if its inputs are staged.
        grid_lat: Grid latitude extent; defaults to the configured platform grid.
        grid_lon: Grid longitude extent; defaults to the configured platform grid.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.
        drop_staging: Remove each variable's staging objects once the whole pass has written
            its containers, and only for the variables whose container was written.
        expected_members: The contract's member count, which the per-cell coverage floor is
            measured against. Defaults to the members actually staged, which is right only when
            the set is complete.
        wave_leads: The leads the wave is filling, so a predecessor that has not landed yet is
            refused rather than encoded as absent -- see
            :func:`ingestion.core.aggregate_fields.build_container_fields`.

    Returns:
        ``(variable, aggregate_key, member_count)`` per variable aggregated.

    Raises:
        StagingError: if nothing is staged for the lead at all, a variable's geometry does not
            match the configured grid, or a container cannot be built from the staged inputs. A
            variable with no staging is skipped: not every lead carries every variable (the GEFS
            product omits the instantaneous precipitation rate), and that is normal rather than an
            error. A variable the platform does not classify is skipped with its staging retained;
            a *classified* variable whose container cannot be built is reported.
    """
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
    candidates.extend(
        _derived_candidates(candidates, staged, lead_time_hours)
    )

    results: list[tuple[str, str, int]] = []
    for variable in candidates:
        # A derived variable has no staging of its own: whether it belongs in this pass is decided
        # by its inputs, which ``_derived_candidates`` has already checked.
        staged_here = {lead for (_m, lead) in staged.get(variable, {})}
        if variable in staged and lead_time_hours not in staged_here:
            continue
        try:
            key, count = aggregate_staged_lead(
                store,
                variable,
                lead_time_hours,
                grid_lat=resolved_grid_lat,
                grid_lon=resolved_grid_lon,
                chunk_lat=resolved_chunk_lat,
                chunk_lon=resolved_chunk_lon,
                drop_staging=False,
                expected_members=expected_members,
                wave_leads=wave_leads,
            )
        except StagingError as exc:
            if _is_unclassified(variable):
                # A variable the platform does not classify is staged like any other and has no
                # container to build. Skipping it leaves its staging in place for a human to
                # resolve, rather than dropping bytes nothing can rebuild from.
                logger.warning(
                    "no aggregate encoding for %s at lead %d; its staging is retained: %s",
                    variable,
                    lead_time_hours,
                    exc,
                )
                continue
            raise
        results.append((variable, key, count))

    if drop_staging and results:
        # Only the variables whose container was written. A variable the pass skipped keeps its
        # staging -- its members exist nowhere else, so deleting them would make a later pass
        # unable to build the container rather than merely late.
        _release_staging(
            store,
            _staged_keys_for_lead(
                staged, [variable for variable, _k, _c in results], lead_time_hours
            ),
        )
    return results


def _is_unclassified(variable: str) -> bool:
    """Whether the platform has no approved encoding for a variable at all.

    The distinction the pass needs: a variable the registry does not know is one whose container
    was never designed, and is skipped -- but a *known* variable whose container could not be
    built from the staged inputs is a failure of the pass and is reported. Treating the two the
    same would let a missing input vanish into a warning.
    """
    from domain.variable_class import VariableClassError, encoding_for

    try:
        encoding_for(variable)
    except VariableClassError:
        return True
    return False


def _derived_candidates(
    candidates: Sequence[str],
    staged: dict[str, dict[tuple[int, int], str]],
    lead_time_hours: int,
) -> list[str]:
    """Derived variables to add to the pass, given that their inputs are staged at this lead.

    ``wind_10m`` is the case: the platform serves it, the API synthesises it from two stored
    components, and it has no member shards at all -- so it never appears in the staging area and
    a caller listing the staged variables would never name it. Its container is as much of the
    cycle's storage decision as any other variable's (it is a fifth of a cycle's member bytes),
    so the pass adds it rather than waiting to be told.

    The set is ``domain.variable_class.DERIVED_VARIABLES`` -- declared, not inferred. A *stored*
    variable that simply has not been staged at this lead must not be added: its container is
    computed from its own members, and there are none.
    """
    from domain.field_layout import FieldLayoutError, required_member_variables
    from domain.variable_class import DERIVED_VARIABLES

    staged_leads: dict[str, set[int]] = {}
    for variable, by_member in staged.items():
        staged_leads.setdefault(variable, set()).update(
            lead for (_member, lead) in by_member
        )
    known = set(candidates)
    extra: list[str] = []
    for variable in sorted(DERIVED_VARIABLES):
        if variable in known or variable in staged:
            continue
        try:
            needed = required_member_variables(variable)
        except FieldLayoutError:  # pragma: no cover - a registered name has a layout
            continue
        inputs = [name for name in needed if name != variable]
        if inputs and all(lead_time_hours in staged_leads.get(name, set()) for name in inputs):
            extra.append(variable)
    return extra


def _release_staging(store: StoreRef, keys: Iterable[str]) -> int:
    """Delete staged objects by key, returning how many were removed.

    The caller decides *which* keys: an aggregate pass that has already listed the whole staging
    root passes the subset it wrote containers for, and the single-variable form lists just its own
    variable. Taking keys rather than variable names keeps the two listings from having to agree
    on a shape.
    """
    doomed = sorted(set(keys))
    if not doomed:
        return 0
    return StoreIO(store).delete_many(doomed)


def _staged_keys_for_lead(
    staged: dict[str, dict[tuple[int, int], str]],
    variables: Iterable[str],
    lead_time_hours: int,
) -> list[str]:
    """The staged keys of ``variables`` at one lead, from a listing the caller already has."""
    keys: list[str] = []
    for variable in variables:
        keys.extend(
            key
            for (member, lead), key in staged.get(variable, {}).items()
            if lead == lead_time_hours
        )
    return keys


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
