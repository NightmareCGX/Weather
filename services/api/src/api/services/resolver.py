"""Authoritative Valid-Time Resolver for forecast serving paths (Lifecycle V2).

This service generalizes the cross-cycle minimum-lead winner selection into a
shared, reusable resolver used across all forecast serving endpoint families:
- /v1/points
- /v1/ensembles
- /v1/probabilities
- /v1/maps (spatial layer metadata and raster tiles)

Mathematical Rule:
------------------
For each requested ``valid_time`` and model M:
    source_cycle(valid_time) =
        MAX(cycle_time)
        WHERE cycle_time + lead_time_hours == valid_time
          AND requested forecast data is committed and serveable

Because cycle times are multiples of the model's cadence C (e.g. 6h) and leads
are multiples of the lead cadence (e.g. 3h), maximizing cycle_time is mathematically
equivalent to minimizing lead_time_hours:
    cycle_time = valid_time - lead_time_hours

Guarantees:
-----------
1. Realtime Promotion: A safely committed lead from a newer partial/processing run
   promotes immediately for its covered valid times without waiting for the full
   cycle to become ready.
2. Graceful Fallback: If a newer cycle has not yet committed a lead for a valid
   time, the resolver seamlessly falls back to the newest older committed cycle
   that covers it.
3. GEFS Serveability: Ensemble candidates are checked against the authoritative
   member coverage threshold (>= 85%). An under-covered newer lead falls back to
   an older cycle with sufficient coverage.
4. Performance: All resolution queries are resolved entirely from PostgreSQL
   catalog metadata (no object-store prefix scanning).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from domain.coverage import get_expected_members, is_lead_servable
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.models.entities import (
    EnsembleMemberProduct,
    ForecastProduct,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.lifecycle import filter_visible_runs, parse_cycle_time

logger = logging.getLogger(__name__)

#: ModelRun statuses eligible for serving candidate discovery.
SERVING_ELIGIBLE_STATUSES: tuple[str, ...] = ("ready", "processing", "partial")


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class ResolvedForecastSource:
    """Authoritative source provenance resolved for a requested valid_time.

    Attributes:
        model: Platform model identifier ('gfs', 'gefs').
        valid_time: The requested UTC valid datetime.
        cycle_time: The winning source cycle datetime.
        lead_time_hours: The forecast lead offset hours (valid_time - cycle_time).
        run_id: The database model_runs.id of the winning run.
        store_path: Canonical Zarr store path of the winning run.
        serving_generation: The store's manifest serving generation, if available.
    """

    model: str
    valid_time: datetime
    cycle_time: datetime
    lead_time_hours: int
    run_id: str
    store_path: str
    serving_generation: str | None = None


def resolve_valid_time_candidates(
    db: Session,
    model: str,
    *,
    target_valid_time: datetime | None = None,
    variable: str | None = None,
    start_lead_time_hours: int | None = None,
    end_lead_time_hours: int | None = None,
) -> dict[datetime, list[tuple[datetime, int, str, str]]]:
    """Discover all servable (cycle_time, lead_time_hours, run_id, store_path) candidates per valid_time.

    Returns:
        Mapping of valid_time -> list of (cycle_time, lead, run_id, store_path) tuples,
        sorted by ascending lead (newest cycle first).
    """
    m_id = model.lower().strip()
    expected_members = get_expected_members(m_id, default_if_unknown=1)
    is_ensemble = expected_members > 1

    stmt = (
        select(
            ModelRun.id,
            ModelRun.cycle_time,
            ModelRun.zarr_store_path,
            ForecastProduct.lead_time_hours,
        )
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .join(ForecastProduct, ForecastProduct.run_id == ModelRun.id)
        .where(Model.model_id == m_id)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if variable is not None:
        stmt = stmt.where(ForecastProduct.variable_id == variable)

    stmt = filter_visible_runs(stmt, model_id=m_id)
    rows = db.execute(stmt).all()

    # If ensemble, pre-query member counts per (run_id, lead_time_hours)
    emp_counts: dict[tuple[str, int], int] = {}
    if is_ensemble:
        emp_rows = db.execute(
            select(
                EnsembleMemberProduct.run_id,
                EnsembleMemberProduct.lead_time_hours,
                func.count(EnsembleMemberProduct.member_index),
            )
            .group_by(
                EnsembleMemberProduct.run_id,
                EnsembleMemberProduct.lead_time_hours,
            )
        ).all()
        emp_counts = {(str(r_id), int(lead)): int(cnt) for r_id, lead, cnt in emp_rows}

    candidates_by_valid: dict[datetime, set[tuple[datetime, int, str, str]]] = {}

    for run_id, cycle_time, store_path, lead in rows:
        if store_path is None:
            continue
        c_utc = _ensure_utc(cycle_time)
        lead_num = int(lead)

        if start_lead_time_hours is not None and lead_num < start_lead_time_hours:
            continue
        if end_lead_time_hours is not None and lead_num > end_lead_time_hours:
            continue

        v_time = c_utc + timedelta(hours=lead_num)
        if target_valid_time is not None and v_time != _ensure_utc(target_valid_time):
            continue

        # Check ensemble coverage if ensemble
        if is_ensemble:
            count = emp_counts.get((str(run_id), lead_num), 0)
            if not is_lead_servable(count, expected_members):
                continue

        candidates_by_valid.setdefault(v_time, set()).add(
            (c_utc, lead_num, str(run_id), str(store_path))
        )

    product_cycles = {c_utc for pairs in candidates_by_valid.values() for (c_utc, _, _, _) in pairs}

    # Fallback discovery: READY runs with a store but no forecast_products rows
    # still serve via their Zarr lead coordinate. Read the store's lead axis
    # once per distinct cycle (metadata only; cheap) to enumerate candidates.
    from api.services.point_forecast import gated_cycle_metadata

    runs_stmt = (
        select(ModelRun.id, ModelRun.cycle_time, ModelRun.zarr_store_path)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == m_id)
        .where(ModelRun.status == "ready")
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    runs_stmt = filter_visible_runs(runs_stmt, model_id=m_id)
    for run_id, cycle_time, store_path in db.execute(runs_stmt).all():
        if store_path is None:
            continue
        c_utc = _ensure_utc(cycle_time)
        if c_utc in product_cycles:
            continue
        try:
            metadata = gated_cycle_metadata(str(store_path))
        except Exception:
            continue
        leads = sorted(metadata.lead_times)
        for lead_num in leads:
            if start_lead_time_hours is not None and lead_num < start_lead_time_hours:
                continue
            if end_lead_time_hours is not None and lead_num > end_lead_time_hours:
                continue
            v_time = c_utc + timedelta(hours=lead_num)
            if target_valid_time is not None and v_time != _ensure_utc(target_valid_time):
                continue
            candidates_by_valid.setdefault(v_time, set()).add(
                (c_utc, lead_num, str(run_id), str(store_path))
            )

    # Sort each valid_time's candidates by ascending lead (newest cycle first)
    return {
        v_time: sorted(pairs, key=lambda p: p[1])
        for v_time, pairs in candidates_by_valid.items()
    }


def resolve_valid_time_source(
    db: Session,
    model: str,
    valid_time: datetime | str,
    *,
    variable: str | None = None,
) -> ResolvedForecastSource:
    """Resolve the single newest committed source cycle and lead for a valid_time.

    Args:
        db: Database session.
        model: Platform model identifier ('gfs', 'gefs').
        valid_time: Requested valid datetime or ISO 8601 string.
        variable: Optional variable code constraint.

    Returns:
        ResolvedForecastSource with complete provenance.

    Raises:
        HTTPException: 404 if no eligible source cycle can serve the valid_time.
    """
    from api.services.point_forecast import resolve_serving_generation_for_store

    v_utc = parse_cycle_time(valid_time) if isinstance(valid_time, str) else _ensure_utc(valid_time)
    m_id = model.lower().strip()

    candidates = resolve_valid_time_candidates(
        db,
        m_id,
        target_valid_time=v_utc,
        variable=variable,
    )

    pairs = candidates.get(v_utc)
    if not pairs:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast data is available for model '{model}' at valid time '{v_utc.isoformat()}'.",
        )

    # First candidate is newest cycle (minimum lead)
    cycle_time, lead_time_hours, run_id, store_path = pairs[0]

    generation = resolve_serving_generation_for_store(store_path)

    return ResolvedForecastSource(
        model=m_id,
        valid_time=v_utc,
        cycle_time=cycle_time,
        lead_time_hours=lead_time_hours,
        run_id=run_id,
        store_path=store_path,
        serving_generation=generation,
    )
