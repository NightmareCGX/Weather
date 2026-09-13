"""Temporal semantics and lead-0 display fallback classification for forecast variables.

Defines the authoritative set of interval forecast variables that require
positive-lead display fallback when resolved to lead_time_hours == 0 under
Data Lifecycle V2, and their source-coupled companion variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from domain.horizon import CANONICAL_LEAD_CADENCE_HOURS

#: Public variables representing backward-looking interval accumulations or averages
#: ending at valid_time ([t - interval, t]). At lead 0 (cycle analysis time), no preceding
#: forecast interval exists within the cycle, so the Zarr store contains NaN and display
#: resolution must fall back to the newest serveable positive lead for the same valid_time.
INTERVAL_LEAD0_FALLBACK_VARIABLES: frozenset[str] = frozenset(
    {
        "precipitation_amount_3h",
        "cloud_cover_3h",
    }
)

#: Internal precipitation diagnostic companion variables that provide interval-averaged
#: categorical flags. When precipitation_amount_3h is sampled from a fallback source cycle,
#: these companion fields must be sampled from the exact same source cycle, lead, run,
#: and store to preserve physical phase and transition consistency.
PRECIPITATION_COMPANION_VARIABLES: frozenset[str] = frozenset(
    {
        "crain",
        "csnow",
        "cfrzr",
        "cicep",
    }
)


def requires_lead0_display_fallback(variable: str | None) -> bool:
    """Return True if the variable represents an interval quantity requiring lead-0 fallback."""
    if variable is None:
        return False
    return variable.strip().lower() in INTERVAL_LEAD0_FALLBACK_VARIABLES


def is_precipitation_companion(variable: str | None) -> bool:
    """Return True if the variable is an interval-averaged precipitation categorical flag."""
    if variable is None:
        return False
    return variable.strip().lower() in PRECIPITATION_COMPANION_VARIABLES


@dataclass(frozen=True)
class VariableTemporalMetadata:
    """Independent temporal semantics metadata for one interval variable (I19).

    ``interval_width_hours`` (W) and ``reset_period_hours`` (R) are deliberately
    independent registered values. Reconstruction and predecessor-hold logic must
    be expressed only in terms of (W, R); the historically coincident pairing
    ``W == R / 2`` (3h interval in a 6h reset period) is a configuration fact,
    never a formula assumption.
    """

    interval_width_hours: int
    reset_period_hours: int


#: Authoritative temporal metadata per interval variable. New interval products
#: must register here; unregistered variables have no reset/reconstruction semantics.
VARIABLE_TEMPORAL_METADATA: dict[str, VariableTemporalMetadata] = {
    "precipitation_amount_3h": VariableTemporalMetadata(
        interval_width_hours=3, reset_period_hours=6
    ),
    "cloud_cover_3h": VariableTemporalMetadata(
        interval_width_hours=3, reset_period_hours=6
    ),
}


def get_variable_temporal_metadata(variable: str) -> VariableTemporalMetadata:
    """Return the registered temporal metadata for an interval variable.

    Raises:
        ValueError: If the variable is unknown or carries no interval semantics.
    """
    if variable is None:
        raise ValueError("variable must not be None")
    key = variable.strip().lower()
    if key not in VARIABLE_TEMPORAL_METADATA:
        raise ValueError(
            f"No temporal metadata registered for {variable!r}; registered: "
            f"{sorted(VARIABLE_TEMPORAL_METADATA)}"
        )
    return VARIABLE_TEMPORAL_METADATA[key]


def serving_start_valid_time(
    now_utc: datetime,
    cadence_hours: int = CANONICAL_LEAD_CADENCE_HOURS,
) -> datetime:
    """Calculate the authoritative forecast serving-window left boundary.

    The authoritative serving-window start is defined as:
        serving_start_valid_time = latest model valid time <= now_utc

    For the canonical forecast valid-time cadence (3 hours by default),
    this is equivalent to flooring UTC wall-clock time to the cadence grid.
    Exact cadence boundaries immediately advance the anchor.

    Examples:
        now = 2026-09-10 05:59:59Z -> serving_start = 2026-09-10 03:00:00Z
        now = 2026-09-10 06:00:00Z -> serving_start = 2026-09-10 06:00:00Z
        now = 2026-09-10 07:00:00Z -> serving_start = 2026-09-10 06:00:00Z
        now = 2026-09-10 08:59:59Z -> serving_start = 2026-09-10 06:00:00Z
        now = 2026-09-10 09:00:00Z -> serving_start = 2026-09-10 09:00:00Z

    Args:
        now_utc: The reference current UTC datetime. Must be timezone-aware.
        cadence_hours: The valid-time cadence in hours (default: 3).

    Returns:
        The floored UTC datetime boundary (with tzinfo=timezone.utc,
        minute=0, second=0, microsecond=0).

    Raises:
        ValueError: If ``now_utc`` is naive or ``cadence_hours`` is not positive.
    """
    if now_utc.tzinfo is None:
        raise ValueError(
            f"now_utc must be a timezone-aware datetime in UTC, got naive datetime: {now_utc!r}"
        )
    if cadence_hours <= 0:
        raise ValueError(
            f"cadence_hours must be strictly positive, got {cadence_hours}"
        )

    utc_dt = now_utc.astimezone(timezone.utc)
    floored_hour = (utc_dt.hour // cadence_hours) * cadence_hours
    return datetime(
        utc_dt.year,
        utc_dt.month,
        utc_dt.day,
        floored_hour,
        0,
        0,
        tzinfo=timezone.utc,
    )


def is_valid_time_protected(
    valid_time: datetime,
    now_utc: datetime,
    *,
    cadence_hours: int = CANONICAL_LEAD_CADENCE_HOURS,
) -> bool:
    """Return whether a valid_time is inside the active serving window (I3/I20).

    This is the membership test of the authoritative
    ``protected_valid_times(model, now)`` primitive:

        protected = {vt on the model horizon grid : vt >= serving_start_valid_time(now)}

    UI visibility, API serving eligibility, and GC canonical protection must all
    consume this primitive (directly or via the serving boundary it exposes);
    none of them may compute an independent boundary.

    The boundary is a deterministic, monotonically non-decreasing function of
    UTC, so an exit from the protected set is permanent — this is what makes
    window-exit GC possible without any replacement publication (architecture
    doc §7.5 class B).
    """
    if valid_time.tzinfo is None:
        raise ValueError(
            f"valid_time must be timezone-aware, got naive datetime: {valid_time!r}"
        )
    vt_utc = valid_time.astimezone(timezone.utc)
    return vt_utc >= serving_start_valid_time(now_utc, cadence_hours)
