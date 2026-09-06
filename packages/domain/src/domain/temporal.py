"""Temporal semantics and lead-0 display fallback classification for forecast variables.

Defines the authoritative set of interval forecast variables that require
positive-lead display fallback when resolved to lead_time_hours == 0 under
Data Lifecycle V2, and their source-coupled companion variables.
"""

from __future__ import annotations

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
