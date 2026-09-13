"""Centralized lifecycle and serviceability authority for API serving paths.

This service is the single source of truth for forecast cycle visibility in the
serving tier (Data Lifecycle V3).

Serviceability Invariant:
-------------------------
A forecast cycle C for model M is SERVICEABLE (visible) iff it has NOT been
claimed for deletion or deleted in the durable lifecycle table:

    forecast_cycle_lifecycle.deletion_started_at IS NULL
    AND
    forecast_cycle_lifecycle.deleted_at IS NULL
    AND
    forecast_cycle_lifecycle.model_id = M

Cycles without a row in ``forecast_cycle_lifecycle`` are lazily created and
treated as VISIBLE (unfenced).

Once a model's cycle is deletion-fenced or deleted, it must become completely
inaccessible across all user-facing serving paths for that model:
- /v1/forecast/availability
- /v1/points (cross-cycle min-lead winner selection)
- /v1/ensembles (implicit newest and explicit valid_time/initial_time)
- /v1/probabilities (implicit newest and explicit valid_time/initial_time)
- /v1/maps (metadata availability)
- /v1/maps/.../{z}/{x}/{y}.png (raster tile rendering and cache)
- /v1/maps/.../vector-field (flow field rendering)
- /v1/verifications (run discovery)
- /v1/runs (public catalog listing filters fenced runs from serving views)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

from fastapi import HTTPException
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from api.models.entities import ForecastCycleLifecycle, ModelRun, ModelVersion
from domain.temporal import serving_start_valid_time

logger = logging.getLogger(__name__)

TSelect = TypeVar("TSelect", bound=Select[Any])


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_cycle_time(cycle_time: datetime | str) -> datetime:
    """Parse an ISO 8601 string or datetime to a UTC timezone-aware datetime."""
    if isinstance(cycle_time, datetime):
        return _ensure_utc(cycle_time)
    s = cycle_time.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid ISO 8601 cycle time format: '{cycle_time}'",
        ) from exc
    return _ensure_utc(dt)


def filter_visible_runs(stmt: TSelect, model_id: str | None = None) -> TSelect:
    """Apply the centralized visibility filter to a SQLAlchemy query selecting ModelRun.

    Excludes all model_runs whose (model_id, cycle_time) has a durable lifecycle row with
    deletion_started_at IS NOT NULL or deleted_at IS NOT NULL (Lifecycle V3).

    Args:
        stmt: The SQLAlchemy select statement to decorate.
        model_id: Optional model_id filter. When omitted, correlates with ModelRun's
            model_version_id -> models.model_id.

    Returns:
        The decorated statement with the physical deletion fence applied.
    """
    return filter_fenced_runs(stmt, model_id=model_id)


def filter_fenced_runs(stmt: TSelect, model_id: str | None = None) -> TSelect:
    """Apply the physical safety fence filter to a SQLAlchemy query selecting ModelRun.

    Excludes all model_runs whose (model_id, cycle_time) has a durable lifecycle row with
    deletion_started_at IS NOT NULL or deleted_at IS NOT NULL (Lifecycle V3 per-valid-time canonical ownership).

    Args:
        stmt: The SQLAlchemy select statement to decorate.
        model_id: Optional model_id filter. When omitted, correlates with ModelRun's
            model_version_id -> models.model_id.

    Returns:
        The decorated statement with the physical deletion fence applied.
    """
    if model_id is not None:
        fenced_subq = (
            select(1)
            .select_from(ForecastCycleLifecycle)
            .where(
                ForecastCycleLifecycle.model_id == model_id.lower().strip(),
                ForecastCycleLifecycle.cycle_time == ModelRun.cycle_time,
                or_(
                    ForecastCycleLifecycle.deletion_started_at.isnot(None),
                    ForecastCycleLifecycle.deleted_at.isnot(None),
                ),
            )
            .correlate(ModelRun)
        )
    else:
        fenced_subq = (
            select(1)
            .select_from(ForecastCycleLifecycle)
            .join(
                ModelVersion,
                ModelVersion.model_id == ForecastCycleLifecycle.model_id,
            )
            .where(
                ModelVersion.id == ModelRun.model_version_id,
                ForecastCycleLifecycle.cycle_time == ModelRun.cycle_time,
                or_(
                    ForecastCycleLifecycle.deletion_started_at.isnot(None),
                    ForecastCycleLifecycle.deleted_at.isnot(None),
                ),
            )
            .correlate(ModelRun)
        )
    return stmt.where(~fenced_subq.exists())


def is_cycle_fenced(
    db: Session,
    cycle_time: datetime | str,
    model_id: str | None = None,
) -> bool:
    """Return True if cycle_time has a physical deletion fence (deletion_started_at or deleted_at).

    Args:
        db: Database session.
        cycle_time: Cycle datetime or ISO 8601 string.
        model_id: Optional model identifier ('gfs', 'gefs').

    Returns:
        True if the cycle has an active deletion fence or is deleted, False otherwise.
    """
    dt_utc = parse_cycle_time(cycle_time)
    dt_naive = dt_utc.replace(tzinfo=None)
    time_pred = or_(
        ForecastCycleLifecycle.cycle_time == dt_utc,
        ForecastCycleLifecycle.cycle_time == dt_naive,
    )
    query = select(
        ForecastCycleLifecycle.deletion_started_at,
        ForecastCycleLifecycle.deleted_at,
    ).where(time_pred)

    if model_id is not None:
        query = query.where(ForecastCycleLifecycle.model_id == model_id.lower().strip())

    rows = db.execute(query).all()
    if not rows:
        return False

    return any(
        deletion_started_at is not None or deleted_at is not None
        for deletion_started_at, deleted_at in rows
    )


def is_cycle_visible(
    db: Session,
    cycle_time: datetime | str,
    model_id: str | None = None,
) -> bool:
    """Return True if cycle_time is currently visible / serviceable for model_id.

    Cycles with no row in forecast_cycle_lifecycle are visible by default.
    Under Lifecycle V3, a cycle is visible iff it is not physically fenced
    (deletion_started_at is None and deleted_at is None).

    Args:
        db: Database session.
        cycle_time: Cycle datetime or ISO 8601 string.
        model_id: Optional model identifier ('gfs', 'gefs').

    Returns:
        True if the cycle is visible, False if claimed for deletion or deleted.
    """
    return not is_cycle_fenced(db, cycle_time, model_id=model_id)


def require_cycle_visible(
    db: Session,
    cycle_time: datetime | str,
    model_id: str | None = None,
) -> datetime:
    """Validate that an explicit cycle_time is visible, raising HTTP 404 if deleted or claimed.

    Args:
        db: Database session.
        cycle_time: Explicit cycle datetime or ISO 8601 string.
        model_id: Optional model identifier.

    Returns:
        The normalized UTC cycle datetime.

    Raises:
        HTTPException: 404 when the cycle is claimed for deletion, deleted, or unserviceable.
    """
    dt_utc = parse_cycle_time(cycle_time)
    if not is_cycle_visible(db, dt_utc, model_id=model_id):
        logger.info(
            "fenced_cycle_access_denied: cycle_time=%s model_id=%s",
            dt_utc.isoformat(),
            model_id,
            extra={
                "event": "fenced_cycle_access_denied",
                "cycle_time": dt_utc.isoformat(),
                "model_id": model_id,
            },
        )
        raise HTTPException(
            status_code=404,
            detail=f"Forecast cycle '{cycle_time}' is not available"
            + (f" for model '{model_id}'." if model_id else "."),
        )
    return dt_utc


def assert_valid_time_in_serving_window(
    cycle_time: datetime | str,
    lead_time_hours: int,
    *,
    now: datetime,
) -> None:
    """Enforce the API serving-window left boundary for a (cycle, lead) pair.

    Consumer of the shared ``serving_start_valid_time`` primitive (Lifecycle V3
    §5): the valid time implied by ``cycle_time + lead_time_hours`` must lie in
    the active serving window; a pinned request below the boundary targets a
    valid time no serving path can guarantee.

    Args:
        cycle_time: The resolved run's cycle time (datetime or ISO 8601 string).
        lead_time_hours: The forecast lead offset hours.
        now: The current UTC time anchoring the serving window.

    Raises:
        HTTPException: 404 when the implied valid time is before the window start.
    """
    c_utc = parse_cycle_time(cycle_time)
    computed_valid_time = c_utc + timedelta(hours=lead_time_hours)
    start_vt = serving_start_valid_time(now)
    if computed_valid_time < start_vt:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Valid time '{computed_valid_time.isoformat()}' is before the "
                f"active serving window ({start_vt.isoformat()})."
            ),
        )


def assert_pinned_lead_available(
    db: Session,
    model: str,
    cycle_time: datetime | str,
    lead_time_hours: int,
) -> None:
    """Validate physical availability of a pinned (cycle, lead) BEFORE cache lookup.

    The lead is servable when, after fencing out units queued for reclamation
    (``deleting``/``deleted``/``failed`` member rows), the committed member
    count still passes the model's coverage threshold. This is a pre-cache fast
    rejection only: callers wrap it in a best-effort ``except Exception: pass``
    guard and the serving path re-validates physical availability authoritatively.

    Args:
        db: Database session.
        model: The model identifier.
        cycle_time: The pinned cycle time (datetime or ISO 8601 string).
        lead_time_hours: The pinned forecast lead offset hours.

    Raises:
        HTTPException: 404 when the lead fails the servability threshold.
    """
    from api.models.entities import ReclamationQueue
    from domain.coverage import get_expected_members, is_lead_servable

    expected_members = get_expected_members(model, default_if_unknown=30)
    c_p = parse_cycle_time(cycle_time)
    fenced_count = (
        db.execute(
            select(func.count(ReclamationQueue.id)).where(
                ReclamationQueue.model_id == model.lower().strip(),
                ReclamationQueue.cycle_time == c_p,
                ReclamationQueue.lead_time_hours == lead_time_hours,
                ReclamationQueue.target_kind == "mem",
                ReclamationQueue.status.in_(("deleting", "deleted", "failed")),
            )
        ).scalar()
        or 0
    )
    if not is_lead_servable(expected_members - fenced_count, expected_members):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Forecast lead {lead_time_hours}h for model '{model}' at cycle "
                f"'{cycle_time}' is not available."
            ),
        )
