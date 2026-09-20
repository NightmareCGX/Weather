"""How a variable's aggregate fields are laid out in its container.

One authority for three questions that must not be answered independently on the write and read
sides, because a disagreement between them is a silently mis-decoded statistic rather than an
error:

* **Which fields the container holds, and in what order.** A reader that assumed yesterday's
  field order would read today's mean as a bin probability.
* **The fixed-point scale of each field.** They differ within one container -- a 314 K mean needs
  0.01 to fit int16 while a 0-1 bin probability wants 0.001 -- so a single container-wide scale
  cannot describe them, and guessing "one scale for all fields" would clip the mean by 10x.
* **What each field means.** The statistics reader needs to know which index is the mean and
  which are levels or bins, rather than counting from a hardcoded offset.

This module answers those from the *variable*, not from the container: a container is data, and
data that describes its own interpretation is how the two sides drift apart. The container
carries only what it must (a count, a geometry, an encoding family); everything else is derived.

See ``docs/investigations/numeric-encoding/REPORT.md`` §15 for the class decisions and
``domain.variable_class`` for the variable → class mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from domain.aggregate import (
    KIND_MEAN_STD_BINS,
    KIND_QUANTILE_FUNCTION,
    MEMBER_COUNT_FIELD_NAME,
    MEMBER_COUNT_SCALE,
)
from domain.variable_class import VariableClassError, encoding_for

#: What a field vector's fields mean, so a reader addresses them by role.
ROLE_MEMBER_COUNT: Final[str] = "member_count"
ROLE_MEAN: Final[str] = "mean"
ROLE_STD: Final[str] = "std"
ROLE_BIN: Final[str] = "bin"
ROLE_LEVEL: Final[str] = "level"
#: Supplementary roles: the fields a variable's *products* need beyond its distribution.
ROLE_ROSE: Final[str] = "rose"
ROLE_CONSENSUS: Final[str] = "consensus"
ROLE_PHASE_CURRENT: Final[str] = "phase_current"
ROLE_PHASE_PREVIOUS: Final[str] = "phase_previous"
ROLE_TRANSITION: Final[str] = "transition"
ROLE_CENSORING: Final[str] = "censoring"
ROLE_CONDITIONAL: Final[str] = "conditional"
ROLE_FRACTION: Final[str] = "fraction"
ROLE_ROSE_EDGES: Final[str] = "rose_edges"

#: The four categorical flag variables, which the precipitation phase classification reads. Named
#: once because three places depend on the set: the phase group, the transition group, and the
#: registry that classifies them.
FLAG_VARIABLE_NAMES: Final[tuple[str, ...]] = ("crain", "csnow", "cfrzr", "cicep")

#: Field groups stored alongside a variable's distribution, with the observation that forced each.
#:
#: Each exists because a product the platform serves is a function of the *per-member values* and
#: not of the distribution the aggregate collapses (``docs/investigations/numeric-encoding``
#: §19.3). The group's name states which product, so a reader that cannot interpret a group can
#: still say what it is for.
#:
#: ``rose`` -- 8 direction sectors x 8 variable speed buckets, plus consensus scalars and a
#: direction stored as sin/cos. Direction is an angle, and an angle near 0/360 does not fit a
#: fixed-point field at any useful resolution without wrapping (360/0.1 = 3600 exceeds
#: int16's 3277), so the pair is stored and ``atan2`` taken on read.
#:
#: ``phase`` -- six physical-phase support planes for the current interval and six for the
#: predecessor. Phase support is a pure function of ``(amount, four flags)`` for each interval,
#: so these two groups hold every input the phase products need.
#:
#: ``transition`` -- twenty transition-category frequencies. Not derivable from the phase groups:
#: the transition branches on whether the *set* of active phases changed, and two members with
#: identical current support can differ there (rain+snow over rain is ``rain_to_snow``, over
#: rain+snow is ``mixed_transition``), so collapsing to weights loses the distinction.
#:
#: ``censoring`` and ``conditional`` -- for the two cloud variables, the counts taken before
#: summarising (how many members were in range, finite, unlimited) and the statistics computed
#: over the finite subset. The conditional percentiles are not recoverable from a mixture: the
#: unlimited members form a point mass at the top of the range, so a mixture percentile says how
#: much mass lies below it and nothing about how the finite mass is spread below that.
#:
#: ``fraction`` -- one per-cell fraction for a 0/1 flag. A shape over two values carries nothing,
#: so a flag is stored as its fraction rather than as a distribution.
_FIELD_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "rose": (
        *(f"ROSE_{sector}_{bucket}" for sector in
          ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
          for bucket in range(8)),
        "CONSENSUS_SPEED",
        "CONSENSUS_COHERENCE",
        "CONSENSUS_DIR_SIN",
        "CONSENSUS_DIR_COS",
        # The bucket edges travel with the fields, as constant planes. They are per
        # ``(variable, lead)`` -- quantile edges of that member set -- so a reader cannot label a
        # bucket, or add the rose up into a speed histogram, without them; and they belong in the
        # container rather than a sidecar object so one read answers the whole product.
        *(f"ROSE_EDGE_{index:02d}" for index in range(9)),
    ),
    "phase": (
        *(f"PHASE_{index}" for index in range(6)),
        *(f"PHASE_PREV_{index}" for index in range(6)),
    ),
    "transition": tuple(f"TRANSITION_{index:02d}" for index in range(20)),
    "censoring": ("VALID_COUNT", "FINITE_COUNT", "UNLIMITED_COUNT"),
    "conditional": (
        "COND_MEAN",
        "COND_SPREAD",
        # The percentile names are spelled the way the response spells them, so the reader's
        # "strip the prefix and lower-case" rule yields the contract's own keys rather than a
        # second mapping that could drift from it.
        "COND_P0.1",
        "COND_P10",
        "COND_P25",
        "COND_P50",
        "COND_P75",
        "COND_P90",
        "COND_P99.9",
    ),
    "fraction": ("FRACTION",),
}

#: Fixed-point step per group. Three kinds of quantity, and the step has to match the kind:
#:
#: * **probabilities**, bounded in [0, 1]: a step of 0.001 bounds the error at half a thousandth,
#:   which is below what a 30-member sample can resolve.
#: * **member counts** (``censoring``): the step is one member, because a count *is* an integer
#:   and a step of 1.0 stores it exactly. It also means a fraction of the member set must not be
#:   stored here: 19/30 at that step quantises to 0.
#: * **speeds in m/s** (the consensus scalars and the rose's bucket edges): 0.01 covers +-327 m/s,
#:   which is past any wind the platform stores.
_GROUP_SCALES: Final[dict[str, float]] = {
    "rose": 0.001,
    "rose_speed": 0.01,
    "phase": 0.001,
    "transition": 0.001,
    "censoring": 1.0,
    "conditional": 0.01,
    "fraction": 0.001,
}

#: Which groups each variable's container carries. A variable absent here has none.
#:
#: ``wind_10m`` is the reason the grouping exists as a first-class input: it is synthesised by
#: the API from ``wind_u_10m`` and ``wind_v_10m`` rather than stored itself, its products are
#: functions of the *pair*, and the rose is also its distribution -- the speed histogram is the
#: rose summed over its sectors, so no separate distribution fields are needed.
_VARIABLE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "wind_10m": ("rose",),
    "precipitation_amount_3h": ("phase", "transition"),
    "precipitation_rate": ("phase", "transition"),
    "cloud_ceiling": ("censoring", "conditional"),
    "cloud_cover_3h": ("censoring", "conditional"),
    "crain": ("fraction",),
    "csnow": ("fraction",),
    "cfrzr": ("fraction",),
    "cicep": ("fraction",),
}

#: Which member variables a group reads, beyond the variable's own field, and whether it also
#: needs the predecessor interval.
#:
#: Declared here rather than in the writer because it is the group's own definition: a group that
#: reads a second variable is a function of *that* variable's members, and the writer, the reader
#: and the reclamation rule all have to agree on it. A group whose inputs are missing cannot be
#: computed, and the variable whose container holds it cannot be published.
#:
#: ``fraction`` reads the variable itself, so its entry names no extra variable.
_GROUP_INPUTS: Final[dict[str, tuple[tuple[str, ...], bool]]] = {
    # (extra member variables, needs the predecessor interval)
    "rose": (("wind_u_10m", "wind_v_10m"), False),
    "phase": ((*FLAG_VARIABLE_NAMES,), True),
    "transition": ((*FLAG_VARIABLE_NAMES,), True),
    "censoring": ((), False),
    "conditional": ((), False),
    "fraction": ((), False),
}

#: Variables whose container holds supplementary fields and no distribution at all.
#:
#: A 0/1 flag has no shape worth storing, so its fraction *is* its representation.
_GROUPS_ONLY_VARIABLES: Final[frozenset[str]] = frozenset(
    {"crain", "csnow", "cfrzr", "cicep"}
)

#: How a cell's aggregate is computed when some members are non-finite.
#:
#: ``skip_at_coverage`` computes from the finite members and refuses the cell only when their
#: count falls below the platform's coverage floor. This is what the serving paths do for every
#: variable -- the member reader filters to finite members and then applies
#: ``is_cell_statistically_valid`` -- so the aggregate reproduces the member answer rather than
#: being stricter than it. The count of participating members is carried per cell so the
#: distinction between "30 members" and "26 members" survives the collapse.
MISSING_SKIP_AT_COVERAGE: Final[str] = "skip_at_coverage"
VALID_MISSING_POLICIES: Final[frozenset[str]] = frozenset({MISSING_SKIP_AT_COVERAGE})


class FieldLayoutError(ValueError):
    """Raised when a variable's stored field layout cannot be resolved."""


@dataclass(frozen=True)
class FieldLayout:
    """The fields of one variable's aggregate container, in storage order.

    Field 0 is always the per-cell finite-member count (:data:`ROLE_MEMBER_COUNT`), followed by
    the variable's distribution fields and then its supplementary groups. The count leads because
    it is read *before* anything else can be interpreted: it answers "is this cell aggregated at
    all" and "how many members is that statistic over", and both questions come before a mean or
    a percentile means anything.

    A variable with no distribution carries only the count and its groups -- a 0/1 flag's
    fraction *is* its representation, since a shape over two values carries nothing.

    Attributes:
        variable: Variable code the layout belongs to.
        kind: Aggregate encoding family (:data:`domain.aggregate.VALID_KINDS`), or the empty
            string for a variable with no distribution fields.
        field_names: Human-readable name per field, in storage order. Used by the writer's
            metadata and by error messages; never parsed to recover a role.
        field_scales: Fixed-point scale per field, in the same order.
        roles: What each field means, in the same order.
        groups: The supplementary field groups carried, in storage order.
        missing_policy: How a partially observed cell is aggregated.
    """

    variable: str
    kind: str
    field_names: tuple[str, ...]
    field_scales: tuple[float, ...]
    roles: tuple[str, ...]
    groups: tuple[str, ...] = ()
    missing_policy: str = MISSING_SKIP_AT_COVERAGE

    def __post_init__(self) -> None:
        counts = {len(self.field_names), len(self.field_scales), len(self.roles)}
        if len(counts) != 1:
            raise FieldLayoutError(
                f"{self.variable!r}: field_names/field_scales/roles disagree in length: {counts}"
            )
        if not self.field_names:
            raise FieldLayoutError(f"{self.variable!r}: a layout must declare at least one field")
        if self.missing_policy not in VALID_MISSING_POLICIES:
            raise FieldLayoutError(
                f"{self.variable!r}: unknown missing-member policy {self.missing_policy!r}"
            )

    @property
    def n_fields(self) -> int:
        """Number of stored lat/lon planes."""
        return len(self.field_names)

    def index_of_role(self, role: str) -> int | None:
        """Index of the single field carrying ``role``, or ``None`` when there is none.

        Raises:
            FieldLayoutError: if more than one field claims the role. A reader asking for "the
                mean" must not be handed the first of several.
        """
        matches = [index for index, value in enumerate(self.roles) if value == role]
        if len(matches) > 1:
            raise FieldLayoutError(
                f"{self.variable!r}: {len(matches)} fields carry role {role!r}"
            )
        return matches[0] if matches else None

    def indices_of_role(self, role: str) -> tuple[int, ...]:
        """Indices of every field carrying ``role``, in storage order."""
        return tuple(index for index, value in enumerate(self.roles) if value == role)

    @property
    def distribution_slice(self) -> slice:
        """Where the variable's distribution fields sit: after the count, before the groups.

        Derived from the group sizes rather than hardcoded, so adding a group cannot move it, and
        unlike :meth:`index_of_role` it works for a role a level-per-field encoding repeats --
        ``quantile_function`` has nineteen fields carrying ``ROLE_LEVEL``, so asking for "the
        field with that role" is a question with nineteen answers.

        Empty for a variable with no distribution: a flag's fraction is its representation, and a
        product-fields variable's products are its rose.
        """
        group_size = sum(len(_FIELD_GROUPS[group]) for group in self.groups)
        return slice(1, self.n_fields - group_size)

    def group_slice(self, group: str) -> slice:
        """Where a supplementary group's fields sit in the vector.

        Raises:
            FieldLayoutError: if the layout does not carry that group. A reader asking for fields
                the container does not hold is a specification mismatch, not a missing value.
        """
        if group not in self.groups:
            raise FieldLayoutError(
                f"{self.variable!r} carries no {group!r} fields; it carries {self.groups}"
            )
        return self._group_slices[group]

    @property
    def _group_slices(self) -> dict[str, slice]:
        """Offset of each carried group, derived from the group sizes."""
        out: dict[str, slice] = {}
        cursor = self.n_fields - sum(len(_FIELD_GROUPS[g]) for g in self.groups)
        for group in self.groups:
            size = len(_FIELD_GROUPS[group])
            out[group] = slice(cursor, cursor + size)
            cursor += size
        return out


def group_inputs(group: str) -> tuple[tuple[str, ...], bool]:
    """Which member variables a group reads, and whether it also needs the predecessor.

    Returns:
        ``(extra_variables, needs_predecessor)``. The variable's own members are always a
        further input; this names only what is read *besides* them.

    Raises:
        FieldLayoutError: for an unknown group name.
    """
    try:
        return _GROUP_INPUTS[group]
    except KeyError as exc:
        raise FieldLayoutError(
            f"unknown field group {group!r}; known: {sorted(_GROUP_INPUTS)}"
        ) from exc


def required_member_variables(variable: str) -> tuple[str, ...]:
    """Every member variable needed to build a variable's container, its own included.

    A publication needs all of these committed before it can build the container. For most
    variables that is the variable itself; for the phase and transition groups it is the four
    flags, and for the rose it is the two wind components -- so a partial member set can block a
    container even when the variable's own members have all arrived.
    """
    name = variable.strip()
    needed = [name]
    for group in _VARIABLE_GROUPS.get(name, ()):
        extra, _needs_predecessor = group_inputs(group)
        for extra_variable in extra:
            if extra_variable not in needed:
                needed.append(extra_variable)
    return tuple(needed)


def needs_predecessor(variable: str) -> bool:
    """Whether a variable's container is a function of the predecessor interval as well."""
    return any(
        group_inputs(group)[1] for group in _VARIABLE_GROUPS.get(variable.strip(), ())
    )


def group_field_names(group: str) -> tuple[str, ...]:
    """Field names of a supplementary group, in storage order.

    Raises:
        FieldLayoutError: for an unknown group name.
    """
    try:
        return _FIELD_GROUPS[group]
    except KeyError as exc:
        raise FieldLayoutError(
            f"unknown field group {group!r}; known: {sorted(_FIELD_GROUPS)}"
        ) from exc


#: Which fields of a group need a different step from the group's default. Named by field name so
#: the exception is visible where it is made, rather than as an index offset a reader has to
#: recount whenever the group changes.
_FIELD_SCALE_OVERRIDES: Final[dict[str, float]] = {
    "CONSENSUS_SPEED": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_00": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_01": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_02": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_03": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_04": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_05": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_06": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_07": _GROUP_SCALES["rose_speed"],
    "ROSE_EDGE_08": _GROUP_SCALES["rose_speed"],
}


def group_field_scales(group: str) -> tuple[float, ...]:
    """Fixed-point step of each field in a supplementary group."""
    default = _GROUP_SCALES[group]
    overridden = tuple(
        _FIELD_SCALE_OVERRIDES.get(name, default) for name in group_field_names(group)
    )
    return overridden


def aggregate_fields_for(variable: str) -> FieldLayout:
    """Resolve the stored field layout for a variable.

    The vector is: the per-cell member count, then the variable's distribution fields (none for a
    0/1 flag, whose fraction is its representation), then its supplementary groups in the order
    :data:`_VARIABLE_GROUPS` lists them.

    Raises:
        FieldLayoutError: for a variable with no approved encoding or no supplementary groups and
            no distribution spec. There is deliberately no default: a fallback layout would
            silently write a field vector under the wrong names.
    """
    name = variable.strip()
    try:
        encoding = encoding_for(name)
    except VariableClassError as exc:
        raise FieldLayoutError(str(exc)) from exc

    groups = _VARIABLE_GROUPS.get(name, ())
    spec = encoding.spec
    if spec is None and not groups:
        raise FieldLayoutError(
            f"{name!r} is a 0/1 flag with no fraction group registered; it would have no fields"
        )

    names: list[str] = []
    scales: list[float] = []
    roles: list[str] = []
    if spec is not None:
        names = list(spec.field_names)
        scales = list(spec.field_scales)
        if spec.kind == KIND_MEAN_STD_BINS:
            roles = [ROLE_MEAN, ROLE_STD, *([ROLE_BIN] * spec.n_bins)]
        elif spec.kind == KIND_QUANTILE_FUNCTION:
            roles = [ROLE_LEVEL] * len(spec.levels)
        else:  # pragma: no cover - AggregateSpec validates its own kind
            raise FieldLayoutError(f"{name!r} has unknown aggregate kind {spec.kind!r}")

    for group in groups:
        names.extend(group_field_names(group))
        scales.extend(group_field_scales(group))
        # A group's role is its own name, except the phase groups which say which interval they
        # describe: a consumer asking for "the current phases" must not be handed the previous.
        roles.extend(_group_roles(group))

    # Field 0 is the per-cell finite-member count, prepended to everything else. It leads rather
    # than trails so the two readers that need it before interpreting anything -- the coverage
    # decision and the reported member count -- find it at a fixed index.
    return FieldLayout(
        variable=name,
        kind=spec.kind if spec is not None else "",
        field_names=(MEMBER_COUNT_FIELD_NAME, *names),
        field_scales=(MEMBER_COUNT_SCALE, *scales),
        roles=(ROLE_MEMBER_COUNT, *roles),
        groups=groups,
    )


def _group_roles(group: str) -> list[str]:
    """The role of each field in a group."""
    if group == "rose":
        return [ROLE_ROSE] * 64 + [ROLE_CONSENSUS] * 4 + [ROLE_ROSE_EDGES] * 9
    if group == "phase":
        return [ROLE_PHASE_CURRENT] * 6 + [ROLE_PHASE_PREVIOUS] * 6
    if group == "transition":
        return [ROLE_TRANSITION] * 20
    if group == "censoring":
        return [ROLE_CENSORING] * 3
    if group == "conditional":
        return [ROLE_CONDITIONAL] * len(_FIELD_GROUPS["conditional"])
    if group == "fraction":
        return [ROLE_FRACTION]
    raise FieldLayoutError(f"unknown field group {group!r}")


__all__ = [
    "MISSING_SKIP_AT_COVERAGE",
    "ROLE_BIN",
    "ROLE_CENSORING",
    "ROLE_CONDITIONAL",
    "ROLE_CONSENSUS",
    "ROLE_FRACTION",
    "ROLE_LEVEL",
    "ROLE_MEAN",
    "ROLE_MEMBER_COUNT",
    "ROLE_PHASE_CURRENT",
    "ROLE_PHASE_PREVIOUS",
    "ROLE_ROSE",
    "ROLE_ROSE_EDGES",
    "ROLE_STD",
    "ROLE_TRANSITION",
    "VALID_MISSING_POLICIES",
    "FieldLayout",
    "FieldLayoutError",
    "aggregate_fields_for",
    "group_field_names",
    "group_field_scales",
    "group_inputs",
    "needs_predecessor",
    "required_member_variables",
]
