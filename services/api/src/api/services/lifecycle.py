"""Centralized lifecycle and serviceability authority for API serving paths.

This service is the single source of truth for forecast cycle visibility in the
serving tier (Data Lifecycle V2).

Serviceability Invariant:
-------------------------
A forecast cycle C for model M is SERVICEABLE (visible) iff it has NOT been
retired or deleted in the durable lifecycle table:

    forecast_cycle_lifecycle.retired_at IS NULL
    AND
    forecast_cycle_lifecycle.deleted_at IS NULL
    AND
    forecast_cycle_lifecycle.model_id = M

Cycles without a row in ``forecast_cycle_lifecycle`` are lazily created and
treated as VISIBLE (not retired).

Once a model's cycle is marked retired (or deleted), it must become completely
inaccessible across all user-facing serving paths for that model:
- /v1/forecast/availability
- /v1/points (cross-cycle min-lead winner selection)
- /v1/ensembles (implicit newest and explicit valid_time/initial_time)
- /v1/probabilities (implicit newest and explicit valid_time/initial_time)
- /v1/maps (metadata availability)
- /v1/maps/.../{z}/{x}/{y}.png (raster tile rendering and cache)
- /v1/maps/.../vector-field (flow field rendering)
- /v1/verifications (run discovery)
- /v1/runs (public catalog listing)

Explicit historical requests for retired cycles raise HTTP 404 Not Found.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypeVar

from fastapi import HTTPException
from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from api.models.entities import ForecastCycleLifecycle, ModelRun, ModelVersion

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


def retired_cycle_times_subquery(model_id: str | None = None) -> Select[tuple[datetime]]:
    """Return a subquery selecting retired or deleted cycle_time values."""
    stmt = select(ForecastCycleLifecycle.cycle_time).where(
        or_(
            ForecastCycleLifecycle.retired_at.isnot(None),
            ForecastCycleLifecycle.deleted_at.isnot(None),
        )
    )
    if model_id is not None:
        stmt = stmt.where(ForecastCycleLifecycle.model_id == model_id.lower().strip())
    return stmt


def filter_visible_runs(stmt: TSelect, model_id: str | None = None) -> TSelect:
    """Apply the centralized visibility filter to a SQLAlchemy query selecting ModelRun.

    Excludes all model_runs whose (model_id, cycle_time) has a durable lifecycle row with
    retired_at IS NOT NULL or deleted_at IS NOT NULL.

    Args:
        stmt: The SQLAlchemy select statement to decorate.
        model_id: Optional model_id filter. When omitted, correlates with ModelRun's
            model_version_id -> models.model_id.

    Returns:
        The decorated statement with the lifecycle visibility predicate applied.
    """
    if model_id is not None:
        retired_subq = (
            select(1)
            .select_from(ForecastCycleLifecycle)
            .where(
                ForecastCycleLifecycle.model_id == model_id.lower().strip(),
                ForecastCycleLifecycle.cycle_time == ModelRun.cycle_time,
                or_(
                    ForecastCycleLifecycle.retired_at.isnot(None),
                    ForecastCycleLifecycle.deleted_at.isnot(None),
                ),
            )
            .correlate(ModelRun)
        )
    else:
        retired_subq = (
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
                    ForecastCycleLifecycle.retired_at.isnot(None),
                    ForecastCycleLifecycle.deleted_at.isnot(None),
                ),
            )
            .correlate(ModelRun)
        )
    return stmt.where(~retired_subq.exists())


def is_cycle_visible(
    db: Session,
    cycle_time: datetime | str,
    model_id: str | None = None,
) -> bool:
    """Return True if cycle_time is currently visible / serviceable for model_id.

    Cycles with no row in forecast_cycle_lifecycle are visible by default.

    Args:
        db: Database session.
        cycle_time: Cycle datetime or ISO 8601 string.
        model_id: Optional model identifier ('gfs', 'gefs').

    Returns:
        True if the cycle is visible, False if retired or deleted.
    """
    dt_utc = parse_cycle_time(cycle_time)
    dt_naive = dt_utc.replace(tzinfo=None)
    time_pred = or_(
        ForecastCycleLifecycle.cycle_time == dt_utc,
        ForecastCycleLifecycle.cycle_time == dt_naive,
    )
    query = select(
        ForecastCycleLifecycle.retired_at,
        ForecastCycleLifecycle.deleted_at,
    ).where(time_pred)

    if model_id is not None:
        query = query.where(ForecastCycleLifecycle.model_id == model_id.lower().strip())

    rows = db.execute(query).all()
    if not rows:
        return True

    # If any matching row is visible, return True; if all matching rows are retired/deleted, False
    return any(retired_at is None and deleted_at is None for retired_at, deleted_at in rows)


def require_cycle_visible(
    db: Session,
    cycle_time: datetime | str,
    model_id: str | None = None,
) -> datetime:
    """Validate that an explicit cycle_time is visible, raising HTTP 404 if retired.

    Args:
        db: Database session.
        cycle_time: Explicit cycle datetime or ISO 8601 string.
        model_id: Optional model identifier.

    Returns:
        The normalized UTC cycle datetime.

    Raises:
        HTTPException: 404 when the cycle is retired, deleted, or unserviceable.
    """
    dt_utc = parse_cycle_time(cycle_time)
    if not is_cycle_visible(db, dt_utc, model_id=model_id):
        logger.info(
            "retired_cycle_access_denied: cycle_time=%s model_id=%s",
            dt_utc.isoformat(),
            model_id,
            extra={
                "event": "retired_cycle_access_denied",
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
