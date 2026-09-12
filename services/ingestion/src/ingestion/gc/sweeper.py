"""14-Day Detailed Metadata Retention Sweeper (Data Lifecycle V3 Milestone 3).

This module implements the bounded PostgreSQL-only detailed metadata retention
sweeper for Data Lifecycle V3. It removes detailed catalog and reclamation
history (model_runs, forecast_products, ensemble_members,
ensemble_member_products, reclamation_queue) for forecast cycles whose physical
storage was finalized at least 14 days ago.

Invariants:
-----------
1. Retention Policy:
   A cycle is eligible for detailed metadata sweep if and only if:
       deleted_at <= now_utc - 14 days (inclusive)
   Comparison is strictly inclusive: exact equality means eligible.
   Retention period is 14 days (METADATA_RETENTION_DAYS), locked product policy.
2. Permanent Tombstone Preservation:
   The `forecast_cycle_lifecycle` row is the permanent anti-resurrection
   tombstone and is NEVER modified or deleted by the sweeper.
3. No Storage Work:
   The sweeper performs ZERO S3/MinIO operations, zero local filesystem Zarr
   deletions, zero StoreLockCoordinator acquisitions, and zero store-gate checks.
   It is a PostgreSQL-only retention task.
4. Non-Starvation Bounded Candidate Discovery:
   Candidate discovery requires BOTH `deleted_at <= cutoff` AND detailed metadata
   existence (`EXISTS(model_runs)`). Permanent tombstones that have already been
   swept do not qualify, preventing starvation of future candidate batches.
5. Bounded Batching:
   Every sweeper pass is bounded by a finite positive integer batch size.
   No unbounded or unlimited delete mode.
6. Per-Cycle Transaction Isolation:
   Purge operations execute in exactly one database transaction per cycle.
   Failure on cycle N does not roll back successfully committed purges of
   cycles 1..N-1.
7. FK-Safe Child-First Deletion Order:
   1. reclamation_queue
   2. ensemble_member_products
   3. ensemble_members
   4. forecast_products
   5. model_runs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from domain.lifecycle import METADATA_RETENTION_DAYS, is_metadata_purge_eligible
from ingestion.core.catalog import (
    EnsembleMemberProductRecord,
    EnsembleMemberRecord,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    ReclamationQueueRecord,
    _ensure_utc_datetime,
    _utcnow,
)

logger = logging.getLogger(__name__)

#: Default bounded batch capacity for one metadata sweeper pass.
DEFAULT_SWEEPER_BATCH_SIZE: int = 50


@dataclass(frozen=True)
class SweeperCandidate:
    """A candidate lifecycle cycle eligible for detailed metadata sweep."""

    model_id: str
    cycle_time: datetime
    deleted_at: datetime


@dataclass(frozen=True)
class CyclePurgeResult:
    """Diagnostic outcome of purging detailed metadata for one cycle."""

    model_id: str
    cycle_time: datetime
    success: bool
    model_runs_deleted: int = 0
    forecast_products_deleted: int = 0
    ensemble_members_deleted: int = 0
    ensemble_member_products_deleted: int = 0
    reclamation_queue_deleted: int = 0
    error: str | None = None


@dataclass(frozen=True)
class SweeperPassResult:
    """Authoritative structured diagnostic result of one metadata sweeper pass."""

    dry_run: bool
    evaluated_at: datetime
    cutoff: datetime
    candidates: tuple[SweeperCandidate, ...]
    swept_cycles: tuple[tuple[str, datetime], ...]
    failed_cycles: tuple[tuple[str, datetime], ...]
    total_model_runs_deleted: int = 0


def _normalize_batch_size(batch_size: int | None) -> int:
    """Ensure batch_size resolves to a finite positive integer.

    Raises:
        ValueError: If batch_size is less than or equal to 0.
    """
    if batch_size is None:
        return DEFAULT_SWEEPER_BATCH_SIZE
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
    return batch_size


def _get_deleted_count(res: object) -> int:
    """Safely extract positive deleted row count from execution result."""
    cnt = getattr(res, "rowcount", 0)
    return int(cnt) if cnt and cnt > 0 else 0


def discover_sweeper_candidates(
    session: Session,
    *,
    cutoff: datetime,
    models: Sequence[str] | None = None,
    batch_size: int = DEFAULT_SWEEPER_BATCH_SIZE,
) -> list[SweeperCandidate]:
    """Discover lifecycle cycles eligible for metadata sweep.

    A cycle is eligible if and only if:
    1. It has been physically finalized (deleted_at IS NOT NULL)
    2. Its retention duration is expired (deleted_at <= cutoff)
    3. Detailed metadata (model_runs) still exists for the cycle.

    Already-swept tombstones (where model_runs no longer exist) are excluded,
    guaranteeing non-starvation of subsequent batches.
    """
    cutoff_utc = _ensure_utc_datetime(cutoff)
    effective_batch_size = _normalize_batch_size(batch_size)

    # Relational existence predicate: model_runs exist for this (model_id, cycle_time)
    model_run_exists = (
        select(1)
        .select_from(ModelRunRecord)
        .join(
            ModelVersionRecord,
            ModelRunRecord.model_version_id == ModelVersionRecord.id,
        )
        .where(
            ModelVersionRecord.model_id == ForecastCycleLifecycleRecord.model_id,
            ModelRunRecord.cycle_time == ForecastCycleLifecycleRecord.cycle_time,
        )
        .exists()
    )

    stmt = (
        select(
            ForecastCycleLifecycleRecord.model_id,
            ForecastCycleLifecycleRecord.cycle_time,
            ForecastCycleLifecycleRecord.deleted_at,
        )
        .where(
            ForecastCycleLifecycleRecord.deleted_at.isnot(None),
            ForecastCycleLifecycleRecord.deleted_at <= cutoff_utc,
            model_run_exists,
        )
    )

    if models is not None:
        target_models = tuple(m.lower().strip() for m in models)
        stmt = stmt.where(ForecastCycleLifecycleRecord.model_id.in_(target_models))

    stmt = stmt.order_by(
        ForecastCycleLifecycleRecord.deleted_at.asc(),
        ForecastCycleLifecycleRecord.model_id.asc(),
        ForecastCycleLifecycleRecord.cycle_time.asc(),
    ).limit(effective_batch_size)

    rows = session.execute(stmt).all()
    return [
        SweeperCandidate(
            model_id=str(r[0]),
            cycle_time=_ensure_utc_datetime(r[1]),
            deleted_at=_ensure_utc_datetime(r[2]),
        )
        for r in rows
    ]


def purge_cycle_metadata(
    engine: Engine,
    model_id: str,
    cycle_time: datetime,
) -> CyclePurgeResult:
    """Purge detailed catalog and queue metadata for one cycle in an atomic transaction.

    Child-first FK-safe deletion order:
    1. reclamation_queue
    2. ensemble_member_products
    3. ensemble_members
    4. forecast_products
    5. model_runs

    The permanent tombstone `forecast_cycle_lifecycle` is untouched.
    """
    m_id = model_id.lower().strip()
    c_utc = _ensure_utc_datetime(cycle_time)

    with Session(engine) as session:
        try:
            with session.begin():
                # Step 1: Identify all model_run IDs for this (model_id, cycle_time)
                run_ids_stmt = (
                    select(ModelRunRecord.id)
                    .join(
                        ModelVersionRecord,
                        ModelRunRecord.model_version_id == ModelVersionRecord.id,
                    )
                    .where(
                        ModelVersionRecord.model_id == m_id,
                        ModelRunRecord.cycle_time == c_utc,
                    )
                )
                run_ids = list(session.execute(run_ids_stmt).scalars().all())

                # Step 2: Delete reclamation_queue
                # Check both run_ids and (model_id, cycle_time)
                del_q_stmt = (
                    delete(ReclamationQueueRecord).where(
                        (ReclamationQueueRecord.run_id.in_(run_ids))
                        | (
                            (ReclamationQueueRecord.model_id == m_id)
                            & (ReclamationQueueRecord.cycle_time == c_utc)
                        )
                    )
                    if run_ids
                    else delete(ReclamationQueueRecord).where(
                        (ReclamationQueueRecord.model_id == m_id)
                        & (ReclamationQueueRecord.cycle_time == c_utc)
                    )
                )
                q_cnt = _get_deleted_count(session.execute(del_q_stmt))

                emp_cnt = 0
                em_cnt = 0
                prod_cnt = 0
                mr_cnt = 0

                if run_ids:
                    # Step 3: Delete ensemble_member_products
                    emp_cnt = _get_deleted_count(
                        session.execute(
                            delete(EnsembleMemberProductRecord).where(
                                EnsembleMemberProductRecord.run_id.in_(run_ids)
                            )
                        )
                    )

                    # Step 4: Delete ensemble_members
                    em_cnt = _get_deleted_count(
                        session.execute(
                            delete(EnsembleMemberRecord).where(
                                EnsembleMemberRecord.run_id.in_(run_ids)
                            )
                        )
                    )

                    # Step 5: Delete forecast_products
                    prod_cnt = _get_deleted_count(
                        session.execute(
                            delete(ProductRecord).where(
                                ProductRecord.run_id.in_(run_ids)
                            )
                        )
                    )

                    # Step 6: Delete model_runs
                    mr_cnt = _get_deleted_count(
                        session.execute(
                            delete(ModelRunRecord).where(
                                ModelRunRecord.id.in_(run_ids)
                            )
                        )
                    )

            # Committed automatically on exit of with session.begin()
            logger.info(
                "sweeper_cycle_purged: model=%s cycle_time=%s model_runs=%d "
                "products=%d members=%d member_products=%d queue=%d",
                m_id,
                c_utc.isoformat(),
                mr_cnt,
                prod_cnt,
                em_cnt,
                emp_cnt,
                q_cnt,
                extra={
                    "event": "sweeper_cycle_purged",
                    "model": m_id,
                    "cycle_time": c_utc.isoformat(),
                    "model_runs_deleted": mr_cnt,
                    "products_deleted": prod_cnt,
                    "members_deleted": em_cnt,
                    "member_products_deleted": emp_cnt,
                    "queue_deleted": q_cnt,
                },
            )
            return CyclePurgeResult(
                model_id=m_id,
                cycle_time=c_utc,
                success=True,
                model_runs_deleted=mr_cnt,
                forecast_products_deleted=prod_cnt,
                ensemble_members_deleted=em_cnt,
                ensemble_member_products_deleted=emp_cnt,
                reclamation_queue_deleted=q_cnt,
            )
        except Exception as exc:
            logger.error(
                "sweeper_cycle_purge_failed: model=%s cycle_time=%s error=%s",
                m_id,
                c_utc.isoformat(),
                exc,
            )
            return CyclePurgeResult(
                model_id=m_id,
                cycle_time=c_utc,
                success=False,
                error=str(exc),
            )


def run_metadata_sweeper_pass(
    engine: Engine,
    *,
    models: Sequence[str] | None = None,
    dry_run: bool = False,
    batch_size: int = DEFAULT_SWEEPER_BATCH_SIZE,
    now: datetime | None = None,
    cutoff: datetime | None = None,
) -> SweeperPassResult:
    """Execute a single bounded metadata retention sweeper pass.

    Evaluates eligible cycles where deleted_at <= cutoff (14 days ago) and
    detailed metadata still exists, then purges detailed metadata per cycle
    in isolated transactions.
    """
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    effective_cutoff = (
        _ensure_utc_datetime(cutoff)
        if cutoff is not None
        else now_utc - timedelta(days=METADATA_RETENTION_DAYS)
    )
    effective_batch_size = _normalize_batch_size(batch_size)

    logger.info(
        "sweeper_pass_started: dry_run=%s now=%s cutoff=%s models=%s batch_size=%d",
        dry_run,
        now_utc.isoformat(),
        effective_cutoff.isoformat(),
        models,
        effective_batch_size,
    )

    with Session(engine) as session:
        candidates = discover_sweeper_candidates(
            session,
            cutoff=effective_cutoff,
            models=models,
            batch_size=effective_batch_size,
        )

    # Cross-check candidates against M1 domain eligibility primitive when using default policy
    if cutoff is None:
        for cand in candidates:
            if not is_metadata_purge_eligible(cand.deleted_at, now_utc=now_utc):
                logger.warning(
                    "candidate_ineligible_under_domain_contract: model=%s cycle=%s deleted_at=%s",
                    cand.model_id,
                    cand.cycle_time.isoformat(),
                    cand.deleted_at.isoformat(),
                )

    if dry_run:
        logger.info(
            "sweeper_pass_dry_run: candidates=%d",
            len(candidates),
        )
        return SweeperPassResult(
            dry_run=True,
            evaluated_at=now_utc,
            cutoff=effective_cutoff,
            candidates=tuple(candidates),
            swept_cycles=(),
            failed_cycles=(),
            total_model_runs_deleted=0,
        )

    swept: list[tuple[str, datetime]] = []
    failed: list[tuple[str, datetime]] = []
    total_mr_deleted = 0

    for cand in candidates:
        try:
            res = purge_cycle_metadata(engine, cand.model_id, cand.cycle_time)
            if res.success:
                swept.append((cand.model_id, cand.cycle_time))
                total_mr_deleted += res.model_runs_deleted
            else:
                failed.append((cand.model_id, cand.cycle_time))
        except Exception as exc:
            logger.error(
                "sweeper_candidate_failed: model=%s cycle_time=%s error=%s",
                cand.model_id,
                cand.cycle_time.isoformat(),
                exc,
            )
            failed.append((cand.model_id, cand.cycle_time))

    logger.info(
        "sweeper_pass_completed: evaluated=%d swept=%d failed=%d total_mr_deleted=%d",
        len(candidates),
        len(swept),
        len(failed),
        total_mr_deleted,
    )
    return SweeperPassResult(
        dry_run=False,
        evaluated_at=now_utc,
        cutoff=effective_cutoff,
        candidates=tuple(candidates),
        swept_cycles=tuple(swept),
        failed_cycles=tuple(failed),
        total_model_runs_deleted=total_mr_deleted,
    )
