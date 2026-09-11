"""Temporal semantics and lead-0 display fallback classification for forecast variables.

Defines the authoritative set of interval forecast variables that require
positive-lead display fallback when resolved to lead_time_hours == 0 under
Data Lifecycle V2, and their source-coupled companion variables.
"""

from __future__ import annotations

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
