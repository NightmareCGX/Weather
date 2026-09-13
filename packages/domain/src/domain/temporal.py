"""Temporal semantics and lead-0 display fallback classification for forecast variables.

Defines the authoritative set of interval forecast variables that require
positive-lead display fallback when resolved to lead_time_hours == 0 under
Data Lifecycle V2, and their source-coupled companion variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from domain.horizon import (
    CANONICAL_LEAD_CADENCE_HOURS,
    canonical_lead_time_hours,
)

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


def model_serving_start_valid_time(model_id: str, now_utc: datetime) -> datetime:
    """Serving-window left boundary taken on the model's own horizon grid.

    This is the boundary component of the model's protected set — exactly the
    boundary :func:`is_valid_time_protected` applies when given ``model_id``:
    the UTC wall-clock floor is computed on the cadence of the model's
    registered canonical horizon, never on a caller-chosen default. GC
    consumers (planner / worker / finalizer / inventory) must derive their
    boundary through this helper: a bare global-cadence
    ``serving_start_valid_time(now)`` call silently diverges from the
    per-model membership test the moment a model's horizon cadence differs
    from the canonical 3h grid.

    Args:
        model_id: Platform model identifier (e.g. ``gfs``, ``gefs``).
        now_utc: The reference current UTC datetime. Must be timezone-aware.

    Returns:
        The floored UTC datetime boundary (with tzinfo=timezone.utc).

    Raises:
        ValueError: If ``now_utc`` is naive or the model is unknown (same
            registry as :func:`domain.horizon.canonical_lead_time_hours`).
    """
    leads = canonical_lead_time_hours(model_id)
    cadence = leads[1] - leads[0] if len(leads) > 1 else CANONICAL_LEAD_CADENCE_HOURS
    return serving_start_valid_time(now_utc, cadence)


def is_valid_time_on_horizon_grid(
    valid_time: datetime,
    *,
    model_id: str | None = None,
    cadence_hours: int | None = None,
) -> bool:
    """Return whether a valid_time lies on the model's canonical horizon grid.

    The grid is the cadence of the model's canonical horizon lead sequence
    (``domain.horizon``). A valid_time on the grid is exactly one the model can
    produce as ``cycle_time + lead``: its wall-clock offset from the cycle
    cadence grid always coincides with a registered lead (for the current
    GFS/GEFS registry: 6h cycles + the 0..240h/3h lead set → every 3h-aligned
    instant).

    Args:
        valid_time: The candidate valid time. Must be timezone-aware.
        model_id: Optional platform model identifier. When given, the grid
            cadence is derived from the model's registered canonical horizon
            (per-model authoritative grid, Lifecycle V3 §11-11 / I20).
        cadence_hours: Explicit grid cadence override used when ``model_id``
            is None. Defaults to the canonical lead cadence.

    Returns:
        True if the valid_time is grid-aligned for the resolved cadence.

    Raises:
        ValueError: If ``valid_time`` is naive, or the model is unknown while
            ``model_id`` is given.
    """
    if valid_time.tzinfo is None:
        raise ValueError(
            f"valid_time must be timezone-aware, got naive datetime: {valid_time!r}"
        )
    if cadence_hours is None:
        if model_id is not None:
            leads = canonical_lead_time_hours(model_id)
            cadence_hours = (
                leads[1] - leads[0] if len(leads) > 1 else CANONICAL_LEAD_CADENCE_HOURS
            )
        else:
            cadence_hours = CANONICAL_LEAD_CADENCE_HOURS
    vt_utc = valid_time.astimezone(timezone.utc)
    return (
        vt_utc.minute == 0
        and vt_utc.second == 0
        and vt_utc.microsecond == 0
        and vt_utc.hour % cadence_hours == 0
    )


def is_valid_time_protected(
    valid_time: datetime,
    now_utc: datetime,
    *,
    model_id: str | None = None,
    cadence_hours: int = CANONICAL_LEAD_CADENCE_HOURS,
) -> bool:
    """Return whether a valid_time is inside the active serving window (I3/I20).

    This is the membership test of the authoritative
    ``protected_valid_times(model, now)`` primitive:

        protected = {vt on the model horizon grid : vt >= serving_start_valid_time(now)}

    UI visibility, API serving eligibility, and GC canonical protection must all
    consume this primitive (directly or via the serving boundary it exposes);
    none of them may compute an independent boundary.

    When ``model_id`` is given, the full membership test applies: the boundary
    comparison AND horizon-grid alignment derived from that model's registered
    canonical horizon. Without a model, only the boundary component is tested
    (the caller asserts grid alignment upstream).

    The boundary is a deterministic, monotonically non-decreasing function of
    UTC, so an exit from the protected set is permanent — this is what makes
    window-exit GC possible without any replacement publication (architecture
    doc §7.5 class B).

    Args:
        valid_time: The candidate valid time. Must be timezone-aware.
        now_utc: The reference current UTC datetime. Must be timezone-aware.
        model_id: Optional platform model identifier enabling the per-model
            horizon-grid membership check.
        cadence_hours: The valid-time cadence in hours (default: 3), used for
            the boundary when no model is given.

    Returns:
        True if the valid_time is protected for the (optionally) given model.

    Raises:
        ValueError: If either datetime is naive, or the model is unknown while
            ``model_id`` is given.
    """
    if valid_time.tzinfo is None:
        raise ValueError(
            f"valid_time must be timezone-aware, got naive datetime: {valid_time!r}"
        )
    if model_id is not None and not is_valid_time_on_horizon_grid(
        valid_time, model_id=model_id
    ):
        return False
    vt_utc = valid_time.astimezone(timezone.utc)
    return vt_utc >= serving_start_valid_time(now_utc, cadence_hours)


def protected_valid_times(
    model_id: str,
    now_utc: datetime,
    *,
    version_string: str | None = None,
) -> tuple[datetime, ...]:
    """Enumerate the model's protected valid times inside one producible horizon.

    The authoritative protected set is defined by :func:`is_valid_time_protected`
    (boundary ∩ horizon grid). It is bounded above by the model's canonical
    horizon: no cycle can produce a valid_time more than ``max_lead`` hours past
    the newest cycle, so the enumerable protected window is

        [serving_start, serving_start + max_lead]

    restricted to the model's horizon grid.

    Args:
        model_id: Platform model identifier (e.g. ``gfs``, ``gefs``).
        now_utc: The reference current UTC datetime. Must be timezone-aware.
        version_string: Optional version identifier selecting a registered
            per-version horizon.

    Returns:
        The ascending tuple of protected valid times (UTC, grid-aligned).

    Raises:
        ValueError: If ``now_utc`` is naive or the model is unknown.
    """
    leads = canonical_lead_time_hours(model_id, version_string=version_string)
    cadence = leads[1] - leads[0] if len(leads) > 1 else CANONICAL_LEAD_CADENCE_HOURS
    start = serving_start_valid_time(now_utc, cadence)
    max_lead = leads[-1]
    return tuple(
        start + timedelta(hours=lead) for lead in range(0, max_lead + 1, cadence)
    )
