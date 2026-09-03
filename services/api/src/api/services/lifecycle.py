"""Centralized lifecycle and serviceability authority for API serving paths.

This service is the single source of truth for forecast cycle visibility in the
serving tier (Phase 6C).

Serviceability Invariant:
-------------------------
A forecast cycle C is SERVICEABLE (visible) iff it has NOT been retired or
deleted in the durable lifecycle table:

    forecast_cycle_lifecycle.retired_at IS NULL
    AND
    forecast_cycle_lifecycle.deleted_at IS NULL

Cycles without a row in ``forecast_cycle_lifecycle`` are lazily created and
treated as VISIBLE (not retired).

Once a cycle is marked retired (or deleted), it must become completely
inaccessible across all user-facing serving paths:
- /v1/forecast/availability
- /v1/points (cross-cycle min-lead winner selection)
- /v1/ensembles (implicit newest and explicit initial_time)
- /v1/probabilities (implicit newest and explicit initial_time)
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

from api.models.entities import ForecastCycleLifecycle, ModelRun

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


def retired_cycle_times_subquery() -> Select[tuple[datetime]]:
    """Return a subquery selecting all retired or deleted cycle_time values."""
    return select(ForecastCycleLifecycle.cycle_time).where(
        or_(
            ForecastCycleLifecycle.retired_at.isnot(None),
            ForecastCycleLifecycle.deleted_at.isnot(None),
        )
    )


def filter_visible_runs(stmt: TSelect) -> TSelect:
    """Apply the centralized visibility filter to a SQLAlchemy query selecting ModelRun.

    Excludes all model_runs whose cycle_time has a durable lifecycle row with
    retired_at IS NOT NULL or deleted_at IS NOT NULL.

    Args:
        stmt: The SQLAlchemy select statement to decorate.

    Returns:
        The decorated statement with the lifecycle visibility predicate applied.
    """
    return stmt.where(ModelRun.cycle_time.not_in(retired_cycle_times_subquery()))


def is_cycle_visible(db: Session, cycle_time: datetime | str) -> bool:
    """Return True if cycle_time is currently visible / serviceable.

    Cycles with no row in forecast_cycle_lifecycle are visible by default.

    Args:
        db: Database session.
        cycle_time: Cycle datetime or ISO 8601 string.

    Returns:
        True if the cycle is visible, False if retired or deleted.
    """
    dt_utc = parse_cycle_time(cycle_time)
    dt_naive = dt_utc.replace(tzinfo=None)
    row = db.execute(
        select(
            ForecastCycleLifecycle.retired_at,
            ForecastCycleLifecycle.deleted_at,
        ).where(
            or_(
                ForecastCycleLifecycle.cycle_time == dt_utc,
                ForecastCycleLifecycle.cycle_time == dt_naive,
            )
        )
    ).first()

    if row is None:
        return True

    retired_at, deleted_at = row
    return retired_at is None and deleted_at is None


def require_cycle_visible(db: Session, cycle_time: datetime | str) -> datetime:
    """Validate that an explicit cycle_time is visible, raising HTTP 404 if retired.

    Args:
        db: Database session.
        cycle_time: Explicit cycle datetime or ISO 8601 string.

    Returns:
        The normalized UTC cycle datetime.

    Raises:
        HTTPException: 404 when the cycle is retired, deleted, or unserviceable.
    """
    dt_utc = parse_cycle_time(cycle_time)
    if not is_cycle_visible(db, dt_utc):
        logger.info(
            "retired_cycle_access_denied: cycle_time=%s",
            dt_utc.isoformat(),
            extra={
                "event": "retired_cycle_access_denied",
                "cycle_time": dt_utc.isoformat(),
            },
        )
        raise HTTPException(
            status_code=404,
            detail=f"Forecast cycle '{cycle_time}' is not available.",
        )
    return dt_utc
