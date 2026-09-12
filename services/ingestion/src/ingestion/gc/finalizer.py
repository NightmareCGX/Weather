"""V3 Whole-Cycle End-of-Life Physical Finalizer (Data Lifecycle V3 Milestone 2).

This module implements the authoritative whole-cycle physical end-of-life
finalizer for Data Lifecycle V3. It manages the two candidate classes
(recovery vs fresh), enforces conservative multi-version horizon eligibility,
guarantees atomic PostgreSQL finalization, and normalizes reclamation_queue
rows while preserving detailed catalog metadata for Milestone 3.

Invariants:
-----------
1. Fresh candidate eligibility is strictly conservative across all versions:
   cycle_time + max(model_max_lead_hours(model_id, version)) < serving_start_valid_time(now).
   Exact equality means NOT expired.
2. Recovery candidates (deletion_started_at != NULL and deleted_at IS NULL)
   resume monotonically without re-evaluating fresh horizon eligibility.
3. Every distinct physical store belonging to (model_id, cycle_time) across all
   versions/runs must be absent before deleted_at can commit.
4. Physical deletion runs sequentially under individual EXCLUSIVE store gates
   without holding any open database transactions.
5. Queue normalization (status='deleted') and deleted_at tombstone commit
   atomically in the same transaction.
6. Detailed catalog metadata (model_runs, forecast_products, etc.) is retained
   intact for the 14-day retention window (owned by Milestone 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from domain.horizon import model_max_lead_hours
from domain.lifecycle import canonical_cycle_store_path, is_cycle_horizon_expired
from domain.temporal import serving_start_valid_time
from ingestion.core.catalog import (
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ReclamationQueueRecord,
    _ensure_utc_datetime,
    _utcnow,
    ensure_lifecycle_row,
)
from ingestion.gc.reconciler import delete_physical_store_gated

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalizerCandidate:
    """A candidate cycle evaluated by the whole-cycle finalizer."""

    model_id: str
    cycle_time: datetime
    is_recovery: bool


@dataclass(frozen=True)
class FinalizerPassResult:
    """Authoritative structured diagnostic result of one finalizer pass."""

    dry_run: bool
    evaluated_at: datetime
    recovery_candidates: tuple[FinalizerCandidate, ...]
    fresh_candidates: tuple[FinalizerCandidate, ...]
    claimed_cycles: tuple[tuple[str, datetime], ...]
    finalized_cycles: tuple[tuple[str, datetime], ...]
    blocked_cycles: tuple[tuple[str, datetime], ...]
    failed_cycles: tuple[tuple[str, datetime], ...]


def discover_finalizer_candidates(
    session: Session,
    *,
    models: Sequence[str] = ("gfs", "gefs"),
    batch_size: int = 50,
) -> tuple[list[FinalizerCandidate], list[FinalizerCandidate]]:
    """Discover recovery and fresh candidate cycles for whole-cycle finalization.

    Returns:
        (recovery_candidates, fresh_candidates)
    """
    target_models = tuple(m.lower().strip() for m in models)

    # 1. Recovery candidates: claimed for deletion but not yet finalized
    rec_stmt = (
        select(
            ForecastCycleLifecycleRecord.model_id,
            ForecastCycleLifecycleRecord.cycle_time,
        )
        .where(
            ForecastCycleLifecycleRecord.model_id.in_(target_models),
            ForecastCycleLifecycleRecord.deletion_started_at.isnot(None),
            ForecastCycleLifecycleRecord.deleted_at.is_(None),
        )
        .order_by(ForecastCycleLifecycleRecord.cycle_time.asc())
        .limit(batch_size)
    )
    rec_rows = session.execute(rec_stmt).all()
    recovery_candidates = [
        FinalizerCandidate(
            model_id=str(m_id),
            cycle_time=_ensure_utc_datetime(c_time),
            is_recovery=True,
        )
        for m_id, c_time in rec_rows
    ]

    recovery_keys = {(c.model_id, c.cycle_time) for c in recovery_candidates}

    # 2. Fresh candidates: active cycles with model_runs, unfenced in lifecycle
    fresh_stmt = (
        select(
            ModelVersionRecord.model_id,
            ModelRunRecord.cycle_time,
        )
        .distinct()
        .join(
            ModelVersionRecord,
            ModelRunRecord.model_version_id == ModelVersionRecord.id,
        )
        .outerjoin(
            ForecastCycleLifecycleRecord,
            (ForecastCycleLifecycleRecord.model_id == ModelVersionRecord.model_id)
            & (ForecastCycleLifecycleRecord.cycle_time == ModelRunRecord.cycle_time),
        )
        .where(
            ModelVersionRecord.model_id.in_(target_models),
            (
                ForecastCycleLifecycleRecord.deleted_at.is_(None)
                | (ForecastCycleLifecycleRecord.model_id.is_(None))
            ),
            (
                ForecastCycleLifecycleRecord.deletion_started_at.is_(None)
                | (ForecastCycleLifecycleRecord.model_id.is_(None))
            ),
        )
        .order_by(ModelRunRecord.cycle_time.asc())
        .limit(batch_size)
    )
    fresh_rows = session.execute(fresh_stmt).all()

    fresh_candidates: list[FinalizerCandidate] = []
    for m_id, c_time in fresh_rows:
        c_utc = _ensure_utc_datetime(c_time)
        m_str = str(m_id)
        if (m_str, c_utc) not in recovery_keys:
            fresh_candidates.append(
                FinalizerCandidate(
                    model_id=m_str,
                    cycle_time=c_utc,
                    is_recovery=False,
                )
            )

    return recovery_candidates, fresh_candidates


def claim_fresh_candidate(
    session: Session,
    model_id: str,
    cycle_time: datetime,
    *,
    serving_start: datetime,
    claim_time: datetime | None = None,
) -> bool:
    """Atomically evaluate fresh horizon eligibility and commit deletion_started_at claim.

    Guarantee B implementation:
    1. Ensures lifecycle row exists via race-safe upsert.
    2. Locks row with SELECT ... FOR UPDATE.
    3. Resolves all distinct model versions attached to ModelRuns for this cycle.
    4. Evaluates conservative horizon: cycle_time + max(leads) < serving_start.
    5. Stamps deletion_started_at and commits.

    Returns True if claimed, False if ineligible or fail-closed.
    """
    m_id = model_id.lower().strip()
    c_utc = _ensure_utc_datetime(cycle_time)

    # 1. Race-safely ensure row exists
    ensure_lifecycle_row(session, m_id, c_utc)

    # 2. Lock lifecycle row FOR UPDATE
    lc = session.execute(
        select(ForecastCycleLifecycleRecord)
        .where(
            ForecastCycleLifecycleRecord.model_id == m_id,
            ForecastCycleLifecycleRecord.cycle_time == c_utc,
        )
        .with_for_update()
    ).scalar_one()

    # Recheck under lock: abort if already claimed or deleted
    if lc.deletion_started_at is not None or lc.deleted_at is not None:
        return False

    # 3. Query all distinct versions attached to model_runs for this cycle
    stmt = (
        select(ModelVersionRecord.version_string)
        .distinct()
        .join(
            ModelRunRecord,
            ModelRunRecord.model_version_id == ModelVersionRecord.id,
        )
        .where(
            ModelVersionRecord.model_id == m_id,
            ModelRunRecord.cycle_time == c_utc,
        )
    )
    versions = list(session.execute(stmt).scalars().all())
    if not versions:
        logger.warning(
            "fresh_candidate_missing_versions: model=%s cycle_time=%s; failing closed",
            m_id,
            c_utc.isoformat(),
        )
        return False

    # 4. Conservative multi-version max lead
    try:
        max_lead = max(model_max_lead_hours(m_id, v) for v in versions)
    except ValueError as exc:
        logger.error(
            "fresh_candidate_unknown_version: model=%s cycle_time=%s error=%s; failing closed",
            m_id,
            c_utc.isoformat(),
            exc,
        )
        return False

    # 5. Strict horizon expiry check (strict < serving_start; == is NOT expired)
    if not is_cycle_horizon_expired(
        c_utc, max_lead_hours=max_lead, serving_start=serving_start
    ):
        return False

    # 6. Stamp claim
    now_utc = _ensure_utc_datetime(claim_time) if claim_time is not None else _utcnow()
    setattr(lc, "deletion_started_at", now_utc)
    setattr(lc, "updated_at", now_utc)
    session.commit()
    logger.info(
        "finalizer_cycle_claimed: model=%s cycle_time=%s versions=%s max_lead=%dh",
        m_id,
        c_utc.isoformat(),
        versions,
        max_lead,
        extra={
            "event": "finalizer_cycle_claimed",
            "model": m_id,
            "cycle_time": c_utc.isoformat(),
            "versions": versions,
            "max_lead_hours": max_lead,
        },
    )
    return True


def enumerate_cycle_store_paths(
    session: Session,
    model_id: str,
    cycle_time: datetime,
    *,
    base_bucket: str = "weather-data",
) -> list[str]:
    """Enumerate all distinct physical Zarr store paths for a lifecycle cycle.

    Queries all recorded non-null zarr_store_path values from model_runs for this
    (model_id, cycle_time) and includes the canonical template fallback.
    """
    m_id = model_id.lower().strip()
    c_utc = _ensure_utc_datetime(cycle_time)

    stmt = (
        select(ModelRunRecord.zarr_store_path)
        .distinct()
        .join(
            ModelVersionRecord,
            ModelRunRecord.model_version_id == ModelVersionRecord.id,
        )
        .where(
            ModelVersionRecord.model_id == m_id,
            ModelRunRecord.cycle_time == c_utc,
            ModelRunRecord.zarr_store_path.isnot(None),
        )
    )
    recorded_paths = set(session.execute(stmt).scalars().all())

    # Include canonical fallback (guarantees orphaned canonical storage is deleted)
    canonical = canonical_cycle_store_path(m_id, c_utc, base_bucket=base_bucket)
    recorded_paths.add(canonical)

    return sorted(p for p in recorded_paths if p)


def finalize_cycle_physical_and_queue(
    engine: Engine,
    model_id: str,
    cycle_time: datetime,
    *,
    finalization_time: datetime | None = None,
) -> None:
    """Atomically commit deleted_at tombstone and normalize all queue rows to 'deleted'.

    Invariants:
    1. Both mutations commit in the SAME database transaction.
    2. Updates ALL existing queue rows for (model_id, cycle_time).
    3. Preserves attempt_count, last_error, created_at.
    4. Preserves existing reclaimed_at for already-deleted rows via COALESCE.
    5. Zero synthetic rows inserted.
    6. Detailed metadata (model_runs, forecast_products, etc.) remains untouched.
    """
    m_id = model_id.lower().strip()
    c_utc = _ensure_utc_datetime(cycle_time)
    now_utc = (
        _ensure_utc_datetime(finalization_time)
        if finalization_time is not None
        else _utcnow()
    )

    with Session(engine) as session:
        ensure_lifecycle_row(session, m_id, c_utc)
        lc = session.execute(
            select(ForecastCycleLifecycleRecord)
            .where(
                ForecastCycleLifecycleRecord.model_id == m_id,
                ForecastCycleLifecycleRecord.cycle_time == c_utc,
            )
            .with_for_update()
        ).scalar_one()

        setattr(lc, "deleted_at", now_utc)
        setattr(lc, "updated_at", now_utc)

        # Set-based normalization across ALL existing reclamation_queue rows
        stmt = (
            update(ReclamationQueueRecord)
            .where(
                ReclamationQueueRecord.model_id == m_id,
                ReclamationQueueRecord.cycle_time == c_utc,
            )
            .values(
                status="deleted",
                updated_at=now_utc,
                reclaimed_at=func.coalesce(
                    ReclamationQueueRecord.reclaimed_at, now_utc
                ),
            )
        )
        session.execute(stmt)
        session.commit()

    logger.info(
        "finalizer_cycle_completed: model=%s cycle_time=%s deleted_at=%s",
        m_id,
        c_utc.isoformat(),
        now_utc.isoformat(),
        extra={
            "event": "finalizer_cycle_completed",
            "model": m_id,
            "cycle_time": c_utc.isoformat(),
            "deleted_at": now_utc.isoformat(),
        },
    )


def finalize_cycle_eol(
    engine: Engine,
    model_id: str,
    cycle_time: datetime,
    *,
    is_recovery: bool,
    serving_start: datetime,
    base_bucket: str = "weather-data",
    timeout_seconds: float = 5.0,
    now: datetime | None = None,
) -> bool:
    """Execute the full whole-cycle physical end-of-life finalizer for one candidate.

    Returns True if successfully finalized (or already deleted), False if skipped/blocked.
    """
    m_id = model_id.lower().strip()
    c_utc = _ensure_utc_datetime(cycle_time)
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()

    # Step 1: Fresh eligibility claim (skipped for recovery candidates)
    if not is_recovery:
        with Session(engine) as session:
            claimed = claim_fresh_candidate(
                session,
                m_id,
                c_utc,
                serving_start=serving_start,
                claim_time=now_utc,
            )
            if not claimed:
                return False

    # Step 2: Enumerate all distinct physical stores belonging to the cycle
    with Session(engine) as session:
        store_paths = enumerate_cycle_store_paths(
            session, m_id, c_utc, base_bucket=base_bucket
        )

    # Step 3: Sequentially delete every physical store prefix under EXCLUSIVE gate
    for store_path in store_paths:
        deleted = delete_physical_store_gated(
            engine, store_path, timeout_seconds=timeout_seconds
        )
        if not deleted:
            logger.warning(
                "finalizer_store_gate_blocked: model=%s cycle_time=%s store_path=%s; deferring",
                m_id,
                c_utc.isoformat(),
                store_path,
            )
            return False

    # Step 4: Atomic DB finalization (deleted_at + queue normalization)
    finalize_cycle_physical_and_queue(
        engine, m_id, c_utc, finalization_time=now_utc
    )
    return True


def run_finalizer_pass(
    engine: Engine,
    *,
    models: Sequence[str] = ("gfs", "gefs"),
    dry_run: bool = False,
    base_bucket: str = "weather-data",
    timeout_seconds: float = 5.0,
    batch_size: int = 50,
    now: datetime | None = None,
) -> FinalizerPassResult:
    """Execute a single bounded V3 whole-cycle finalizer pass across specified models."""
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    serving_start = serving_start_valid_time(now_utc)

    logger.info(
        "finalizer_pass_started: dry_run=%s now=%s serving_start=%s models=%s",
        dry_run,
        now_utc.isoformat(),
        serving_start.isoformat(),
        models,
    )

    with Session(engine) as session:
        rec_cands, fresh_cands = discover_finalizer_candidates(
            session, models=models, batch_size=batch_size
        )

    if dry_run:
        logger.info(
            "finalizer_dry_run: recovery_candidates=%d fresh_candidates=%d",
            len(rec_cands),
            len(fresh_cands),
        )
        return FinalizerPassResult(
            dry_run=True,
            evaluated_at=now_utc,
            recovery_candidates=tuple(rec_cands),
            fresh_candidates=tuple(fresh_cands),
            claimed_cycles=(),
            finalized_cycles=(),
            blocked_cycles=(),
            failed_cycles=(),
        )

    claimed: list[tuple[str, datetime]] = []
    finalized: list[tuple[str, datetime]] = []
    blocked: list[tuple[str, datetime]] = []
    failed: list[tuple[str, datetime]] = []

    # 1. Process recovery candidates first (monotonicity priority)
    for cand in rec_cands:
        try:
            ok = finalize_cycle_eol(
                engine,
                cand.model_id,
                cand.cycle_time,
                is_recovery=True,
                serving_start=serving_start,
                base_bucket=base_bucket,
                timeout_seconds=timeout_seconds,
                now=now_utc,
            )
            if ok:
                finalized.append((cand.model_id, cand.cycle_time))
            else:
                blocked.append((cand.model_id, cand.cycle_time))
        except Exception as exc:
            logger.error(
                "finalizer_candidate_failed: model=%s cycle_time=%s error=%s",
                cand.model_id,
                cand.cycle_time.isoformat(),
                exc,
            )
            failed.append((cand.model_id, cand.cycle_time))

    # 2. Process fresh candidates
    for cand in fresh_cands:
        try:
            ok = finalize_cycle_eol(
                engine,
                cand.model_id,
                cand.cycle_time,
                is_recovery=False,
                serving_start=serving_start,
                base_bucket=base_bucket,
                timeout_seconds=timeout_seconds,
                now=now_utc,
            )
            if ok:
                claimed.append((cand.model_id, cand.cycle_time))
                finalized.append((cand.model_id, cand.cycle_time))
            else:
                blocked.append((cand.model_id, cand.cycle_time))
        except Exception as exc:
            logger.error(
                "finalizer_candidate_failed: model=%s cycle_time=%s error=%s",
                cand.model_id,
                cand.cycle_time.isoformat(),
                exc,
            )
            failed.append((cand.model_id, cand.cycle_time))

    return FinalizerPassResult(
        dry_run=False,
        evaluated_at=now_utc,
        recovery_candidates=tuple(rec_cands),
        fresh_candidates=tuple(fresh_cands),
        claimed_cycles=tuple(claimed),
        finalized_cycles=tuple(finalized),
        blocked_cycles=tuple(blocked),
        failed_cycles=tuple(failed),
    )
