"""Assembling one variable's aggregate container from the staged member set.

The container is a field vector: the per-cell member count, then the variable's *distribution*
(``domain.aggregate``'s ordering), then the *supplementary groups* its products need
(``domain.product_fields``' arithmetic). Which fields those are, in what order, and at what
fixed-point step is ``domain.field_layout``'s answer. This module only decides what to read and
in what order to compute it.

Three things about the reading are load-bearing.

**A container is not a function of one variable's members.** The phase and transition groups read
the four flag variables, the rose reads the two wind components, and the phase groups read the
predecessor interval as well: ``domain.field_layout.required_member_variables`` names the set for
a variable, so a publication has to wait for all of them. ``wind_10m`` is the extreme case -- it
has no staged members at all, because the API synthesises it from its two components, so its
member count comes from a component's staging.

**A missing input may be late rather than absent, and the two are not interchangeable.** The
predecessor lead of a reset lead is another work item of the same wave, so at a lead's first
publication it may simply not have landed yet; the classifier reads an absent predecessor and a
dry one differently, so a container built before it arrives would carry a definite but wrong
transition. A caller that knows the wave's target leads passes them, and the builder then refuses
the container rather than encoding the absent reading. Every other missing input is refused
unconditionally: it is a member set the container is a function of, and building without it means
publishing a statistic of the wrong set.

**The precipitation groups cannot be built from a stack.** They read ten member planes -- the
amount and four flags for the interval, and the same five for its predecessor -- which stacked at
721x1440 for thirty members is 1.25 GB, over the container's budget. They are therefore fed to
:class:`~domain.product_fields.PrecipitationGroupBuilder` one member at a time. The reads are the
same either way; only the residency differs. The rose does need its two components stacked,
because its bucket edges are quantiles of the whole member set, and two stacks are affordable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from domain.aggregate import (
    AggregateError,
    AggregateSpec,
    compute_aggregate,
    finite_member_count,
)
from domain.field_layout import (
    FieldLayout,
    FieldLayoutError,
    aggregate_fields_for,
    group_field_names,
    required_member_variables,
)
from domain.product_fields import (
    FLAG_NAMES,
    PrecipitationGroupBuilder,
    cloud_censoring_fields,
    fraction_of_members,
    rose_fields,
)
from domain.variable_class import VariableClassError, spec_for

from ingestion.core.aggregate_staging import (
    StagingError,
    member_plane_from_shard,
    staged_objects,
)
from ingestion.core.store_io import StoreAccessError, StoreIO, StoreRef

#: How a cell's aggregate is computed when some members are non-finite. The one policy
#: ``domain.field_layout`` defines: compute from the finite members, refuse the cell only below
#: the coverage floor. Asserted against the layout rather than assumed, so a second policy
#: cannot be added there without this module being updated.
_SKIP_AT_COVERAGE = "skip_at_coverage"

#: A flag counts as set at or above this value, matching the serving path's threshold.
_FLAG_SET_THRESHOLD = 0.5

#: The predecessor interval is three hours back, at a six-hour reset lead.
_PREDECESSOR_INTERVAL_HOURS = 3
_RESET_PERIOD_HOURS = 6


class AggregateBuildError(RuntimeError):
    """Raised when a container cannot be built from the member sets available."""


class MemberReader:
    """Reads staged member planes one at a time, listing each variable once.

    The listing is cached because a member-major walk asks for the same variable's keys thirty
    times, and re-listing an S3 prefix per member would turn one LIST per variable into thirty.
    """

    def __init__(
        self,
        store: StoreRef,
        *,
        grid_lat: int,
        grid_lon: int,
        chunk_lat: int = 100,
        chunk_lon: int = 100,
    ) -> None:
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.chunk_lat = chunk_lat
        self.chunk_lon = chunk_lon
        # ``staged_objects`` takes a store reference, not a StoreIO, so the reference is kept as
        # well; the StoreIO is reused for the reads.
        self._store = store
        self._io = StoreIO(store)
        self._keys: dict[str, dict[tuple[int, int], str]] = {}

    def keys(self, variable: str) -> dict[tuple[int, int], str]:
        """The variable's staged keys by ``(member, lead)``, listed once per reader."""
        if variable not in self._keys:
            self._keys[variable] = staged_objects(self._store, variable)
        return self._keys[variable]

    def members(self, variable: str, lead_time_hours: int) -> list[int]:
        """The members staged for a variable at a lead, sorted."""
        return sorted(
            member for (member, lead) in self.keys(variable) if lead == lead_time_hours
        )

    def plane(
        self, variable: str, lead_time_hours: int, member: int
    ) -> npt.NDArray[np.float32]:
        """One member's plane.

        Raises:
            AggregateBuildError: if that member is not staged, its object cannot be read, or its
                container does not reassemble to the reader's grid.
        """
        try:
            key = self.keys(variable)[(member, lead_time_hours)]
        except KeyError as exc:
            raise AggregateBuildError(
                f"{variable!r} has no staged member {member} at lead {lead_time_hours}h"
            ) from exc
        try:
            raw = self._io.read(key)
        except StoreAccessError as exc:
            raise AggregateBuildError(f"cannot read staged {key!r}: {exc}") from exc
        try:
            plane = member_plane_from_shard(
                raw,
                grid_lat=self.grid_lat,
                grid_lon=self.grid_lon,
                chunk_lat=self.chunk_lat,
                chunk_lon=self.chunk_lon,
            )
        except StagingError as exc:
            raise AggregateBuildError(f"cannot reassemble {key!r}: {exc}") from exc
        if plane.shape != (self.grid_lat, self.grid_lon):
            raise AggregateBuildError(
                f"{key!r} reassembled to {plane.shape}, expected "
                f"{(self.grid_lat, self.grid_lon)}"
            )
        return plane

    def stack(
        self, variable: str, lead_time_hours: int
    ) -> tuple[list[int], npt.NDArray[np.float32]]:
        """Every staged member plane of one variable, as ``(n_members, lat, lon)``.

        Raises:
            AggregateBuildError: if the variable has no staged member at the lead.
        """
        members = self.members(variable, lead_time_hours)
        if not members:
            raise AggregateBuildError(
                f"{variable!r} has no staged members at lead {lead_time_hours}h"
            )
        return members, np.stack(
            [self.plane(variable, lead_time_hours, member) for member in members]
        )


def build_container_fields(
    reader: MemberReader,
    variable: str,
    lead_time_hours: int,
    *,
    expected_members: int,
    wave_leads: Sequence[int] | None = None,
) -> tuple[list[npt.NDArray[np.float32]], int]:
    """Every field of a variable's container, in storage order, and its member count.

    Args:
        reader: A member reader over the cycle's store.
        variable: The variable whose container is being built.
        lead_time_hours: The forecast lead.
        expected_members: The contract's member count. Recorded in the container's descriptor;
            the per-cell coverage floor inside the distribution uses the members actually
            supplied, so a partial publication says what it is over.
        wave_leads: The leads this wave is filling, or ``None`` for a caller that does not know
            (a repair, a backfill, a test). Only the precipitation groups use it, and only to
            decide whether a missing predecessor is *absent* or *late* -- see
            :func:`_predecessor_lead`.

    Returns:
        ``(fields, member_count)``: the count field, the distribution and the groups concatenated
        exactly as :func:`domain.field_layout.aggregate_fields_for` lays them out, and how many
        members the container was computed from.

    Raises:
        AggregateBuildError: for an unclassified variable, a policy this module does not
            implement, or any member set the container needs that is not staged. All of these are
            refusals rather than partial containers: one published from a subset of its inputs is
            a statistic of the wrong member set, and nothing downstream could tell.
    """
    name = variable.strip()
    try:
        layout = aggregate_fields_for(name)
    except FieldLayoutError as exc:
        raise AggregateBuildError(str(exc)) from exc
    if layout.missing_policy != _SKIP_AT_COVERAGE:  # pragma: no cover - one policy today
        raise AggregateBuildError(
            f"{name!r} declares the {layout.missing_policy!r} missing-member policy, which this "
            "builder does not implement"
        )
    spec = _distribution_spec(name, layout)

    own_members, own_stack = _own_members(reader, name, lead_time_hours)
    member_count = len(own_members)
    count_field = (
        finite_member_count(own_stack)
        if own_stack is not None
        else np.full((reader.grid_lat, reader.grid_lon), float(member_count), dtype=np.float32)
    )

    fields: list[npt.NDArray[np.float32]] = [count_field]
    if spec is not None:
        if own_stack is None:  # pragma: no cover - a spec-bearing variable always has members
            raise AggregateBuildError(
                f"{name!r} is encoded as {spec.kind} but has no staged members"
            )
        try:
            fields.extend(compute_aggregate(own_stack, spec, expected_members=member_count))
        except AggregateError as exc:
            raise AggregateBuildError(f"cannot compute the distribution: {exc}") from exc

    # In the layout's own group order, so the vector this returns is the vector the descriptor
    # describes.
    for group in layout.groups:
        # Each group is returned as one block of planes; the field vector is the planes, so the
        # block is iterated. Appending the block as a single element would count a group as one
        # field and produce a container the descriptor disagrees with.
        for block in _group_fields(
            reader,
            name,
            lead_time_hours,
            group,
            own_stack=own_stack,
            wave_leads=wave_leads,
        ):
            fields.extend(list(block))

    expected_fields = layout.n_fields
    if len(fields) != expected_fields:
        raise AggregateBuildError(
            f"{name!r} produced {len(fields)} fields but its layout declares "
            f"{expected_fields}; the group producers and the layout disagree"
        )
    return fields, member_count


def _distribution_spec(variable: str, layout: FieldLayout) -> AggregateSpec | None:
    """The variable's aggregate spec, or ``None`` when its container holds no distribution.

    A missing spec is normal for two classes: a 0/1 flag's shape carries too little to store, and
    a product-fields variable's products *are* its distribution. Both are distinguished from an
    unclassified variable by having reached a layout at all.
    """
    if not layout.distribution_slice.stop > layout.distribution_slice.start:
        return None
    try:
        return spec_for(variable)
    except VariableClassError as exc:  # pragma: no cover - a layout implies a registration
        raise AggregateBuildError(str(exc)) from exc


def _own_members(
    reader: MemberReader, variable: str, lead_time_hours: int
) -> tuple[list[int], npt.NDArray[np.float32] | None]:
    """The variable's own member set, or its first input's when it has none of its own.

    A derived variable -- ``wind_10m`` -- is stored as no member shards at all; the API builds it
    from two components. Its container is still a function of a member set, namely those
    components', and that set is what names the members its products describe.
    """
    own = reader.members(variable, lead_time_hours)
    if own:
        return own, np.stack([reader.plane(variable, lead_time_hours, m) for m in own])

    inputs = [v for v in required_member_variables(variable) if v != variable]
    for candidate in inputs:
        derived = reader.members(candidate, lead_time_hours)
        if derived:
            return derived, None
    raise AggregateBuildError(
        f"{variable!r} has no staged members at lead {lead_time_hours}h"
        + (f", and neither do its inputs {inputs}" if inputs else "")
    )


def _group_fields(
    reader: MemberReader,
    variable: str,
    lead_time_hours: int,
    group: str,
    *,
    own_stack: npt.NDArray[np.float32] | None,
    wave_leads: Sequence[int] | None = None,
) -> list[npt.NDArray[np.float32]]:
    """One group's fields, from whichever member stacks it reads.

    The censoring and conditional groups are one computation, so the shared payload is built once
    per group asked for rather than recomputed per field: a group's producer returns its whole
    block of planes, and the caller concatenates them.
    """
    if group == "rose":
        return [_rose_block(reader, variable, lead_time_hours)]
    if group in ("phase", "transition"):
        return [
            _precipitation_block(
                reader, variable, lead_time_hours, group, wave_leads=wave_leads
            )
        ]
    needs_own = _needs_own_stack(variable, group, own_stack)
    if group == "fraction":
        finite = np.isfinite(needs_own)
        # A flag counts as set at the serving path's threshold, and a non-finite member neither
        # satisfies it nor joins the denominator: a NaN flag is an unknown, not a "no".
        return [
            fraction_of_members((needs_own >= _FLAG_SET_THRESHOLD) & finite, finite=finite)[None]
        ]
    if group in ("censoring", "conditional"):
        payload = cloud_censoring_fields(needs_own, variable=variable)
        width = len(group_field_names("censoring"))
        return [payload[:width]] if group == "censoring" else [payload[width:]]
    raise AggregateBuildError(  # pragma: no cover - every declared group has a producer
        f"{variable!r} declares the {group!r} group, but no producer is registered for it"
    )


def _needs_own_stack(
    variable: str, group: str, own_stack: npt.NDArray[np.float32] | None
) -> npt.NDArray[np.float32]:
    """The variable's own member stack for a group that reads it."""
    if own_stack is None:
        raise AggregateBuildError(
            f"the {group!r} group of {variable!r} reads its own members, which are not staged"
        )
    return own_stack


def _rose_block(
    reader: MemberReader, variable: str, lead_time_hours: int
) -> npt.NDArray[np.float32]:
    """The rose group: 64 bins, four consensus scalars and the nine bucket edges."""
    u_members = reader.members("wind_u_10m", lead_time_hours)
    v_members = reader.members("wind_v_10m", lead_time_hours)
    if not u_members or not v_members:
        raise AggregateBuildError(
            f"the rose group of {variable!r} reads wind_u_10m and wind_v_10m, and at least one "
            f"has no staged member at lead {lead_time_hours}h"
        )
    if set(u_members) != set(v_members):
        raise AggregateBuildError(
            f"wind_u_10m has members {u_members} but wind_v_10m has {v_members} at lead "
            f"{lead_time_hours}h; a rose needs both components of the same members"
        )
    u_stack = np.stack(
        [reader.plane("wind_u_10m", lead_time_hours, m) for m in u_members]
    )
    v_stack = np.stack(
        [reader.plane("wind_v_10m", lead_time_hours, m) for m in u_members]
    )
    rose, scalars, edges = rose_fields(u_stack, v_stack)
    edge_planes = np.broadcast_to(
        edges[:, None, None], (edges.size, *rose.shape[1:])
    ).astype(np.float32)
    return np.concatenate([rose, scalars, edge_planes], axis=0)


def _precipitation_block(
    reader: MemberReader,
    variable: str,
    lead_time_hours: int,
    group: str,
    *,
    wave_leads: Sequence[int] | None = None,
) -> npt.NDArray[np.float32]:
    """One precipitation group, streamed one member at a time.

    Both groups read the same ten planes per member -- the amount and four flags for the interval
    and for its predecessor -- so whichever is asked for, the walk is the same; only the
    accumulator differs.

    **The predecessor is read when it has been staged, not when it exists.** A lead's predecessor
    arrives in the same wave but as a different work item, so at the first publication of a lead
    the predecessor may simply be later than this publication rather than absent. The two are not
    the same statement -- the classifier reports ``persistent_rain`` for an absent predecessor and
    ``dry_to_rain`` for a dry one -- so a container built before the predecessor has landed would
    encode a definite but wrong transition, and the next patch would silently correct it.
    ``wave_leads`` is how the builder tells the two apart: when the predecessor lead is one this
    wave is filling, its staging is *expected*, and the group refuses to be built without it.
    ``None`` means the caller cannot tell (a repair, a backfill, a test), and the predecessor is
    read when present.
    """
    own_members = reader.members(variable, lead_time_hours)
    flag_members = {name: reader.members(name, lead_time_hours) for name in FLAG_NAMES}
    if any(not members for members in flag_members.values()) or not own_members:
        missing = [name for name, members in flag_members.items() if not members]
        raise AggregateBuildError(
            f"the {group!r} group of {variable!r} reads {variable} and the four flags, and at "
            f"least one is not staged at lead {lead_time_hours}h (missing: "
            f"{missing if missing else [variable]})"
        )
    # The flags must describe the same members, or the phases would be computed for a different
    # ensemble than the distribution beside them.
    for name, members in flag_members.items():
        if set(members) != set(own_members):
            raise AggregateBuildError(
                f"{variable!r} has staged members {own_members} at lead {lead_time_hours}h but "
                f"{name!r} has {members}; the phases cannot be computed for a different member set"
            )

    predecessor_lead = _predecessor_lead(lead_time_hours)
    predecessor_members: list[int] = []
    if predecessor_lead is not None:
        assert predecessor_lead is not None  # narrowed for the reads below
        predecessor_members = reader.members(variable, predecessor_lead)
        if not predecessor_members and _predecessor_is_expected(predecessor_lead, wave_leads):
            raise AggregateBuildError(
                f"{variable!r} lead {lead_time_hours}h has no staged predecessor at lead "
                f"{predecessor_lead}h, which this wave is filling; the transitions cannot be "
                "classified yet, and an absent predecessor would be encoded as a definite one"
            )
        if predecessor_members and set(predecessor_members) != set(own_members):
            raise AggregateBuildError(
                f"{variable!r} lead {lead_time_hours}h has members {own_members} but its "
                f"predecessor lead {predecessor_lead}h has {predecessor_members}; a transition "
                "between different member sets is not a transition"
            )
        if predecessor_members and any(
            not reader.members(name, predecessor_lead) for name in FLAG_NAMES
        ):
            raise AggregateBuildError(
                f"{variable!r} lead {lead_time_hours}h has a staged predecessor amount at lead "
                f"{predecessor_lead}h but its flags are not staged; an interval is its amount and "
                "its flags, and half of one would classify against members nobody described"
            )

    # Both groups come from one walk of the members, and the builder returns them concatenated in
    # the layout's order; the caller asked for one group, so only that part is returned. The walk
    # is not repeated because the phases and the transitions are computed from the same signature.
    #
    # ``expected_members`` is passed so the phases describe the whole set rather than the subset
    # added so far -- a patch is computed from every staged member, but the record of which
    # member set the fields describe is what a reader sanity-checks the count field against -- so
    # one builder serves the distribution, the phase group and the transition group in one pass.
    builder = PrecipitationGroupBuilder(
        expected_members=len(own_members), with_transitions=True
    )
    for member in own_members:
        flags = np.stack(
            [reader.plane(name, lead_time_hours, member) for name in FLAG_NAMES]
        )
        if predecessor_members and predecessor_lead is not None:
            builder.add_member(
                amounts=reader.plane(variable, lead_time_hours, member),
                flags=flags,
                amounts_prev=reader.plane(variable, predecessor_lead, member),
                flags_prev=np.stack(
                    [reader.plane(name, predecessor_lead, member) for name in FLAG_NAMES]
                ),
            )
        else:
            builder.add_member(
                amounts=reader.plane(variable, lead_time_hours, member), flags=flags
            )
    payload = builder.finish()
    phase_width = len(group_field_names("phase"))
    if group == "phase":
        return payload[:phase_width]
    return payload[phase_width:]


def _predecessor_lead(lead_time_hours: int) -> int | None:
    """The predecessor interval's lead, or ``None`` when the lead has no predecessor.

    A six-hour reset lead is differenced against the interval three hours before it; every other
    lead is read directly. Lead 0 is the analysis, and has no predecessor by definition.
    """
    if lead_time_hours <= 0 or lead_time_hours % _RESET_PERIOD_HOURS:
        return None
    return lead_time_hours - _PREDECESSOR_INTERVAL_HOURS


def _predecessor_is_expected(
    predecessor_lead: int, wave_leads: Sequence[int] | None
) -> bool:
    """Whether a wave is filling the predecessor lead, so its staging is on its way.

    ``None`` for the leads means the caller does not know the wave's target set, and the only
    honest answer is "cannot tell" -- which the caller reads as "not expected", the behaviour
    before this distinction existed.
    """
    if wave_leads is None:
        return False
    return predecessor_lead in set(wave_leads)


__all__ = [
    "AggregateBuildError",
    "MemberReader",
    "build_container_fields",
]
