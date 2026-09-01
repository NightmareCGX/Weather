"""Authoritative model expected-member contract and shared coverage logic.

This module centralizes the platform's serving availability rules:
1. Expected member counts are fixed per model under the configured product
   contract (e.g. GFS=1, GEFS=30) and never dynamically inferred from runtime
   occupancy or catalog presence.
2. Lead-level serving eligibility requires at least the configured minimum
   member coverage ratio (default 85%):
   ``available_members * 10000 >= expected_members * threshold_scaled``.
3. Point/cell-level statistical validity requires at least the configured
   minimum finite-value coverage against expected members:
   ``finite_cell_count * 10000 >= expected_members * threshold_scaled``.
4. Once cell validity holds, statistical calculation operates strictly on the
   participating finite sample.
5. The minimum coverage threshold defaults to 0.85 (85%) and can be configured
   via environment variable ``ENSEMBLE_MIN_COVERAGE_RATIO`` with strict range
   validation (0 < threshold <= 1.0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

#: Default fallback serving coverage threshold ratio (0.85 = 85%).
DEFAULT_MIN_COVERAGE_RATIO: float = 0.85

#: Authoritative fixed expected-member counts per model.
MODEL_EXPECTED_MEMBERS: dict[str, int] = {
    "gfs": 1,
    "gefs": 30,
}


def _validate_coverage_ratio(ratio: float) -> float:
    """Validate that a coverage ratio is within (0.0, 1.0]."""
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(
            f"ENSEMBLE_MIN_COVERAGE_RATIO must be in (0, 1], got {ratio}"
        )
    return float(ratio)


def _init_min_coverage_ratio_from_env() -> float:
    """Read and validate the environment-configured coverage ratio, or default."""
    raw = os.getenv("ENSEMBLE_MIN_COVERAGE_RATIO") or os.getenv(
        "WEATHER_ENSEMBLE_MIN_COVERAGE_RATIO"
    )
    if raw is not None:
        try:
            val = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ENSEMBLE_MIN_COVERAGE_RATIO (not a number): {raw!r}"
            ) from exc
        return _validate_coverage_ratio(val)
    return DEFAULT_MIN_COVERAGE_RATIO


_current_min_coverage_ratio: float = _init_min_coverage_ratio_from_env()


def get_min_coverage_ratio() -> float:
    """Return the active minimum member coverage ratio for serving eligibility."""
    return _current_min_coverage_ratio


def set_min_coverage_ratio(ratio: float) -> None:
    """Set the minimum member coverage ratio for serving eligibility.

    Args:
        ratio: A float in (0.0, 1.0] (e.g. 0.85).

    Raises:
        ValueError: If ratio is not within (0.0, 1.0].
    """
    global _current_min_coverage_ratio
    _current_min_coverage_ratio = _validate_coverage_ratio(ratio)


def reset_min_coverage_ratio() -> None:
    """Reset the minimum member coverage ratio to the default or env value."""
    global _current_min_coverage_ratio
    _current_min_coverage_ratio = _init_min_coverage_ratio_from_env()


def register_expected_members(model_id: str, count: int) -> None:
    """Register or override the expected member count for a model (test/startup).

    Args:
        model_id: Platform model identifier (e.g. "gfs", "gefs").
        count: Fixed expected member count.

    Raises:
        ValueError: If count is <= 0.
    """
    if count <= 0:
        raise ValueError(f"expected_members must be positive, got {count}")
    MODEL_EXPECTED_MEMBERS[model_id.lower().strip()] = count


def get_expected_members(
    model_id: str, default_if_unknown: int | None = None
) -> int:
    """Return the authoritative fixed expected member count for a model.

    The expected member count is a fixed specification of our ingestion and
    serving contract. It is NOT dynamically inferred from the current run,
    database rows, or store occupancy.

    Args:
        model_id: Platform model identifier (e.g. "gfs", "gefs").
        default_if_unknown: Optional fallback value if the model is not in the
            authoritative registry. When ``None`` (the default), an unrecognized
            model raises ``ValueError``.

    Returns:
        The configured expected member count.

    Raises:
        ValueError: If ``model_id`` is not registered and ``default_if_unknown``
            is ``None``.
    """
    normalized = model_id.lower().strip()
    if normalized in MODEL_EXPECTED_MEMBERS:
        return MODEL_EXPECTED_MEMBERS[normalized]
    if default_if_unknown is not None:
        return default_if_unknown
    raise ValueError(
        f"Unknown model identifier {model_id!r}; registered models: "
        f"{sorted(MODEL_EXPECTED_MEMBERS)}"
    )


def is_lead_servable(
    available_members: int,
    expected_members: int,
    *,
    min_coverage_ratio: float | None = None,
) -> bool:
    """Return whether a lead satisfies the member-coverage requirement.

    Uses integer-safe arithmetic (``available * 10000 >= expected * threshold_scaled``).

    Args:
        available_members: Count of successfully committed logical members.
        expected_members: Fixed authoritative expected member count.
        min_coverage_ratio: Optional explicit ratio override in (0, 1]. Defaults
            to the active configured policy (default 0.85).

    Returns:
        True when lead member coverage is at or above the threshold; False otherwise.
    """
    if expected_members <= 0 or available_members < 0:
        return False
    threshold = (
        min_coverage_ratio
        if min_coverage_ratio is not None
        else get_min_coverage_ratio()
    )
    threshold_scaled = int(round(threshold * 10000))
    return (available_members * 10000) >= (expected_members * threshold_scaled)


def is_cell_statistically_valid(
    finite_cell_count: Any,
    expected_members: int,
    *,
    min_coverage_ratio: float | None = None,
) -> Any:
    """Return whether a spatial cell satisfies the finite-sample requirement.

    Evaluated independently at each spatial cell or point after unavailable
    members have been excluded using catalog evidence. Supports both scalar
    integer counts and NumPy integer count arrays.

    Args:
        finite_cell_count: Count (or 2-D array of counts) of participating
            members with finite values at this cell.
        expected_members: Fixed authoritative expected member count.
        min_coverage_ratio: Optional explicit ratio override in (0, 1]. Defaults
            to the active configured policy (default 0.85).

    Returns:
        True (or boolean mask array) when finite cell coverage is at or above
        threshold; False otherwise.
    """
    if expected_members <= 0:
        return False
    threshold = (
        min_coverage_ratio
        if min_coverage_ratio is not None
        else get_min_coverage_ratio()
    )
    threshold_scaled = int(round(threshold * 10000))
    if isinstance(finite_cell_count, (int, float)):
        if finite_cell_count < 0:
            return False
        return (int(finite_cell_count) * 10000) >= (expected_members * threshold_scaled)
    import numpy as np

    arr = np.asarray(finite_cell_count)
    return (arr * 10000) >= (expected_members * threshold_scaled)


def compute_coverage_ratio(available_members: int, expected_members: int) -> float:
    """Compute the fractional coverage ratio, rounded to 4 decimal places.

    Args:
        available_members: Available/committed count.
        expected_members: Expected total count.

    Returns:
        Fractional coverage ratio in [0.0, 1.0] (e.g. 0.9667), or 0.0 if
        ``expected_members <= 0``.
    """
    if expected_members <= 0 or available_members <= 0:
        return 0.0
    ratio = available_members / expected_members
    return round(min(1.0, max(0.0, ratio)), 4)


@dataclass(frozen=True)
class LeadCoverage:
    """The computed availability and member-coverage state of a forecast lead.

    Attributes:
        lead_time_hours: The forecast lead offset in hours.
        available_members: Count of committed members.
        expected_members: Authoritative expected member count.
        coverage_ratio: Computed fractional ratio (available / expected).
        servable: Whether coverage meets the serving threshold.
        available_member_indices: Sorted tuple of available member numbers.
    """

    lead_time_hours: int
    available_members: int
    expected_members: int
    coverage_ratio: float
    servable: bool
    available_member_indices: tuple[int, ...] = ()


def build_lead_coverage(
    lead_time_hours: int,
    available_member_indices: tuple[int, ...] | list[int] | set[int],
    expected_members: int,
    *,
    min_coverage_ratio: float | None = None,
) -> LeadCoverage:
    """Construct a LeadCoverage value from committed member indices.

    Args:
        lead_time_hours: Forecast lead offset in hours.
        available_member_indices: Collection of committed member indices.
        expected_members: Authoritative expected member count.
        min_coverage_ratio: Optional explicit threshold override in (0, 1].

    Returns:
        An immutable :class:`LeadCoverage` instance.
    """
    sorted_indices = tuple(sorted(available_member_indices))
    avail_count = len(sorted_indices)
    ratio = compute_coverage_ratio(avail_count, expected_members)
    servable = is_lead_servable(
        avail_count, expected_members, min_coverage_ratio=min_coverage_ratio
    )
    return LeadCoverage(
        lead_time_hours=lead_time_hours,
        available_members=avail_count,
        expected_members=expected_members,
        coverage_ratio=ratio,
        servable=servable,
        available_member_indices=sorted_indices,
    )
