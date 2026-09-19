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

from domain.aggregate import KIND_MEAN_STD_BINS, KIND_QUANTILE_FUNCTION
from domain.variable_class import VariableClassError, encoding_for

#: What a field vector's fields mean, so a reader addresses them by role.
ROLE_MEAN: Final[str] = "mean"
ROLE_STD: Final[str] = "std"
ROLE_BIN: Final[str] = "bin"
ROLE_LEVEL: Final[str] = "level"

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

    Attributes:
        variable: Variable code the layout belongs to.
        kind: Aggregate encoding family (:data:`domain.aggregate.VALID_KINDS`).
        field_names: Human-readable name per field, in storage order. Used by the writer's
            metadata and by error messages; never parsed to recover a role.
        field_scales: Fixed-point scale per field, in the same order.
        roles: What each field means, in the same order.
        missing_policy: How a partially observed cell is aggregated.
    """

    variable: str
    kind: str
    field_names: tuple[str, ...]
    field_scales: tuple[float, ...]
    roles: tuple[str, ...]
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


def aggregate_fields_for(variable: str) -> FieldLayout:
    """Resolve the stored field layout for a variable.

    Raises:
        FieldLayoutError: for a variable with no approved aggregate encoding, or one whose
            class carries no ``AggregateSpec`` (a flag). There is deliberately no default: a
            fallback layout would silently write a field vector under the wrong names.
    """
    name = variable.strip()
    try:
        encoding = encoding_for(name)
    except VariableClassError as exc:
        raise FieldLayoutError(str(exc)) from exc
    spec = encoding.spec
    if spec is None:
        raise FieldLayoutError(
            f"{name!r} is a 0/1 flag; it has no aggregate spec and therefore no field vector"
        )

    names = spec.field_names
    if spec.kind == KIND_MEAN_STD_BINS:
        roles = (ROLE_MEAN, ROLE_STD, *(ROLE_BIN,) * spec.n_bins)
    elif spec.kind == KIND_QUANTILE_FUNCTION:
        roles = (ROLE_LEVEL,) * len(spec.levels)
    else:  # pragma: no cover - AggregateSpec validates its own kind
        raise FieldLayoutError(f"{name!r} has unknown aggregate kind {spec.kind!r}")

    return FieldLayout(
        variable=name,
        kind=spec.kind,
        field_names=names,
        field_scales=spec.field_scales,
        roles=roles,
    )


__all__ = [
    "MISSING_SKIP_AT_COVERAGE",
    "ROLE_BIN",
    "ROLE_LEVEL",
    "ROLE_MEAN",
    "ROLE_STD",
    "VALID_MISSING_POLICIES",
    "FieldLayout",
    "FieldLayoutError",
    "aggregate_fields_for",
]
