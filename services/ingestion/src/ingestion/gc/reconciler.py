"""Physical GC engine, sequential store deleter, and atomic catalog cleaner (Lifecycle V2).

This module owns physical storage reclamation for retired, GC-eligible forecast
cycles under Data Lifecycle V2.

Execution Pipeline (per candidate model cycle M, C):
----------------------------------------------------
1. Re-check GC eligibility against durable PostgreSQL catalog state:
   Cycle C must be older than model M's cutoff (T - C_M).
2. Acquire EXCLUSIVE store gate on model M's store -> recursive delete S3 prefix -> release gate.
3. Execute atomic PostgreSQL transaction:
   - Delete cycle's ensemble_member_products rows for model M
   - Delete cycle's ensemble_members rows for model M
   - Delete cycle's forecast_products rows for model M
   - Delete cycle's model_runs rows for model M
   - Update forecast_cycle_lifecycle.deleted_at = NOW() for (model_id, cycle_time) (tombstone)
   - Commit transaction

Invariants:
-----------
- Models are completely independent: GFS cleanup never locks, touches, or deletes GEFS data.
- If store gate cannot be acquired due to active readers/writers, GC skips the
  cycle for the current pass and retries on the next pass.
- Missing S3/MinIO prefixes are treated as idempotent success (crash-safe).
- Dry-run mode makes ZERO mutations (no locks, no deletes, no DB writes).
- Lifecycle row survives indefinitely as a tombstone for (model_id, cycle_time).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from domain.cadence import canonical_cycle_cadence
from domain.lifecycle import (
    LifecycleDecision,
    canonical_cycle_store_path,
    compute_lifecycle_cutoff,
    plan_model_lifecycle,
)
from ingestion.core.catalog import (
    EnsembleMemberProductRecord,
    EnsembleMemberRecord,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    _ensure_utc_datetime,
    _utcnow,
    ensure_lifecycle_row,
    list_cycle_lifecycle_snapshots,
    list_model_ready_cycle_times,
    reconcile_cycle_lifecycle,
)
from ingestion.core.config import settings
from ingestion.core.locks import LockTimeoutError, StoreLockCoordinator
from ingestion.core.s3 import get_control_s3_fs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GcCandidateInfo:
    """Diagnostic info for a GC candidate."""

    model_id: str
    cycle_time: datetime
    retired_by_cycle_time: datetime | None
    cutoff: datetime | None
    store_path: str
    # Optional legacy compatibility fields
    gc_eligible_by_cycle_time: datetime | None = None
    gfs_store_path: str | None = None
    gefs_store_path: str | None = None


@dataclass(frozen=True)
class GcPassResult:
    """Authoritative result of one GC reconciliation pass."""

    dry_run: bool
    evaluated_at: datetime
    would_retire: tuple[LifecycleDecision, ...]
    would_gc: tuple[GcCandidateInfo, ...]
    processed_gc: tuple[datetime, ...]
    blocked_gc: tuple[datetime, ...]
    locked_gc: tuple[datetime, ...]
    failed_gc: tuple[datetime, ...]


def _delete_store_prefix(store_path: str) -> None:
    """Delete a store prefix recursively (idempotent, ignores missing paths)."""
    if store_path.startswith("s3://"):
        fs = get_control_s3_fs(settings)
        # s3fs path format: bucket/prefix
        raw_path = store_path[len("s3://") :].rstrip("/")
        try:
            if fs.exists(raw_path):
                fs.rm(raw_path, recursive=True)
        except Exception as exc:
            # Missing prefix or already deleted is success
            logger.debug("S3 prefix rm error on %s: %s", raw_path, exc)
    else:
        path = store_path[len("file://") :] if store_path.startswith("file://") else store_path
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def delete_physical_store_gated(
    engine: Engine,
    store_path: str,
    *,
    timeout_seconds: float = 5.0,
) -> bool:
    """Acquire EXCLUSIVE store gate and delete physical store prefix.

    Returns True if deletion succeeded (or store already absent), False if
    lock was blocked by an active reader or writer.
    """
    if engine.dialect.name != "postgresql":
        # SQLite / offline test harness without advisory lock support
        _delete_store_prefix(store_path)
        return True

    with engine.connect() as conn:
        coord = StoreLockCoordinator(
            conn,
            store_path=store_path,
            timeout_seconds=timeout_seconds,
        )
        try:
            coord.acquire_exclusive_gate()
        except LockTimeoutError:
            logger.warning(
                "gc_lock_blocked: store_path=%s timeout=%.1fs",
                store_path,
                timeout_seconds,
                extra={
                    "event": "gc_lock_blocked",
                    "store_path": store_path,
                    "timeout_seconds": timeout_seconds,
                },
            )
            return False

        try:
            logger.info("gc_store_delete_started: store_path=%s", store_path)
            start_t = time.monotonic()
            _delete_store_prefix(store_path)
            duration_ms = round((time.monotonic() - start_t) * 1000, 2)
            logger.info(
                "gc_store_deleted: store_path=%s duration_ms=%.1f",
                store_path,
                duration_ms,
                extra={
                    "event": "gc_store_deleted",
                    "store_path": store_path,
                    "duration_ms": duration_ms,
                },
            )
            return True
        finally:
            coord.release_exclusive_gate()


def recheck_gc_eligibility(
    session: Session,
    model_id: str,
    cycle_time: datetime,
    *,
    version_string: str = "v1.0",
) -> tuple[bool, str, datetime | None]:
    """Re-read PostgreSQL state and re-verify that (model_id, cycle_time) is GC eligible.

    Monotonic Recovery Rule:
    - If deletion_started_at IS NOT NULL and deleted_at IS NULL:
      The cycle was ALREADY authorized and claimed by a prior GC pass.
      It is valid for deletion resume even if historical successor runs were later cleaned up.
    - If deletion_started_at IS NULL:
      Requires retired_at IS NOT NULL and cycle_time < cutoff (T - C).

    Returns:
        (is_eligible, reason, latest_ready_T)
    """
    c_utc = _ensure_utc_datetime(cycle_time)
    m_id = model_id.lower().strip()
    lc = session.get(ForecastCycleLifecycleRecord, (m_id, c_utc))
    if lc is None:
        return False, "no_lifecycle_record", None
    if lc.deleted_at is not None:
        return False, "cycle_already_deleted", None

    # Monotonic recovery: if deletion fence was already committed, deletion is authorized to resume
    if lc.deletion_started_at is not None:
        return True, "gc_claimed_resumable", lc.retired_by_cycle_time

    if lc.retired_at is None:
        return False, "cycle_not_retired", None

    ready_cycles = list_model_ready_cycle_times(session, m_id, version_string=version_string)
    cadence = canonical_cycle_cadence(m_id)
    t_ready, cutoff = compute_lifecycle_cutoff(ready_cycles, cadence)

    if cutoff is None or c_utc >= cutoff:
        return False, "retained_at_or_above_cutoff", None

    return True, "gc_eligible", t_ready


def claim_cycle_for_deletion(
    session: Session,
    model_id: str,
    cycle_time: datetime,
    *,
    now: datetime | None = None,
) -> bool:
    """Atomically stamp deletion_started_at on forecast_cycle_lifecycle for (model_id, cycle_time).

    Establishes the durable cycle deletion fence before physical store deletion.
    If already claimed (deletion_started_at is not None), returns True (idempotent resume).
    If already deleted (deleted_at is not None), returns False.
    """
    c_utc = _ensure_utc_datetime(cycle_time)
    m_id = model_id.lower().strip()
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()

    lc = ensure_lifecycle_row(session, m_id, c_utc)
    if lc.deleted_at is not None:
        return False
    if lc.deletion_started_at is not None:
        return True

    setattr(lc, "deletion_started_at", now_utc)
    setattr(lc, "updated_at", now_utc)
    session.commit()
    logger.info(
        "gc_cycle_claimed: model=%s cycle_time=%s deletion_started_at=%s",
        m_id,
        c_utc.isoformat(),
        now_utc.isoformat(),
        extra={
            "event": "gc_cycle_claimed",
            "model": m_id,
            "cycle_time": c_utc.isoformat(),
            "deletion_started_at": now_utc.isoformat(),
        },
    )
    return True


def cleanup_cycle_catalog_and_tombstone(
    session: Session,
    model_id: str,
    cycle_time: datetime,
    *,
    now: datetime | None = None,
) -> None:
    """Atomically delete model cycle-scoped catalog rows and stamp deleted_at tombstone."""
    c_utc = _ensure_utc_datetime(cycle_time)
    m_id = model_id.lower().strip()
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()

    # Discover run IDs for this model and cycle_time
    run_ids = list(
        session.execute(
            select(ModelRunRecord.id)
            .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
            .where(
                (ModelVersionRecord.model_id == m_id)
                & (ModelRunRecord.cycle_time == c_utc)
            )
        ).scalars().all()
    )

    if run_ids:
        # 1. Delete child ensemble_member_products rows
        session.execute(
            delete(EnsembleMemberProductRecord).where(
                EnsembleMemberProductRecord.run_id.in_(run_ids)
            )
        )

        # 2. Delete child ensemble_members rows
        session.execute(
            delete(EnsembleMemberRecord).where(
                EnsembleMemberRecord.run_id.in_(run_ids)
            )
        )

        # 3. Delete child forecast_products rows
        session.execute(
            delete(ProductRecord).where(
                ProductRecord.run_id.in_(run_ids)
            )
        )

        # 4. Delete model_runs rows for this model and cycle_time
        session.execute(
            delete(ModelRunRecord).where(ModelRunRecord.id.in_(run_ids))
        )

    # 5. Update tombstone on forecast_cycle_lifecycle
    lc = ensure_lifecycle_row(session, m_id, c_utc)
    setattr(lc, "deleted_at", now_utc)
    setattr(lc, "updated_at", now_utc)
    session.commit()
    logger.info(
        "gc_catalog_cleanup_completed: model=%s cycle_time=%s",
        m_id,
        c_utc.isoformat(),
        extra={
            "event": "gc_catalog_cleanup_completed",
            "model": m_id,
            "cycle_time": c_utc.isoformat(),
        },
    )


def process_gc_candidate(
    engine: Engine,
    candidate: GcCandidateInfo,
    *,
    base_bucket: str = "weather-data",
    timeout_seconds: float = 5.0,
    version_string: str = "v1.0",
    now: datetime | None = None,
) -> bool:
    """Execute the full crash-safe GC deletion workflow for a single cycle candidate.

    Returns True if deletion and catalog cleanup succeeded, False if skipped/blocked.
    """
    c_utc = candidate.cycle_time
    m_id = candidate.model_id.lower().strip() if candidate.model_id else "gfs"

    # 1. Re-check eligibility against fresh catalog state
    with Session(engine) as session:
        is_eligible, reason, t_ready = recheck_gc_eligibility(
            session, m_id, c_utc, version_string=version_string
        )
    if not is_eligible:
        logger.warning(
            "gc_candidate_skipped: model=%s cycle_time=%s reason=%s",
            m_id,
            c_utc.isoformat(),
            reason,
            extra={
                "event": "gc_candidate_skipped",
                "model": m_id,
                "cycle_time": c_utc.isoformat(),
                "reason": reason,
            },
        )
        return False

    # 2. Atomically establish durable deletion fence before touching storage
    with Session(engine) as session:
        claimed = claim_cycle_for_deletion(session, m_id, c_utc, now=now)
    if not claimed:
        return False

    # 3. Gated physical store deletion
    stores_to_delete: list[str] = []
    if candidate.store_path:
        stores_to_delete.append(candidate.store_path)
    elif candidate.gfs_store_path and candidate.gefs_store_path:
        stores_to_delete.extend([candidate.gfs_store_path, candidate.gefs_store_path])

    for s_path in stores_to_delete:
        deleted = delete_physical_store_gated(
            engine,
            s_path,
            timeout_seconds=timeout_seconds,
        )
        if not deleted:
            return False

    # 4. Atomic catalog cleanup and tombstone commit
    with Session(engine) as session:
        cleanup_cycle_catalog_and_tombstone(session, m_id, c_utc, now=now)

    logger.info(
        "gc_completed: model=%s cycle_time=%s",
        m_id,
        c_utc.isoformat(),
        extra={"event": "gc_completed", "model": m_id, "cycle_time": c_utc.isoformat()},
    )
    return True


def run_gc_pass(
    engine: Engine,
    *,
    dry_run: bool = False,
    models: tuple[str, ...] = ("gfs", "gefs"),
    version_string: str = "v1.0",
    base_bucket: str = "weather-data",
    timeout_seconds: float = 5.0,
    now: datetime | None = None,
) -> GcPassResult:
    """Execute one complete GC pass across managed models under Lifecycle V2.

    In dry_run mode: plans without taking exclusive locks or mutating S3/PostgreSQL.
    In real mode: persists retirements and deletes GC-eligible stores sequentially per model.

    Args:
        engine: SQLAlchemy Engine for catalog and advisory lock coordination.
        dry_run: If True, execute observational planning only.
        models: Tuple of model identifiers to reconcile and GC.
        version_string: Target model version string.
        base_bucket: S3/MinIO bucket name.
        timeout_seconds: Timeout for exclusive store gate acquisition.
        now: Optional datetime override for testing.

    Returns:
        GcPassResult with structured diagnostics.
    """
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    logger.info("gc_pass_started: dry_run=%s now=%s models=%s", dry_run, now_utc.isoformat(), models)

    with Session(engine) as session:
        if not dry_run:
            # 1. Reconcile newly eligible retirements per model
            reconcile_cycle_lifecycle(session, models=models, now=now_utc, version_string=version_string)

        # 2. Discover recorded store paths from model_runs for accurate deletion
        runs = session.execute(
            select(
                ModelVersionRecord.model_id,
                ModelRunRecord.cycle_time,
                ModelRunRecord.zarr_store_path,
            )
            .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
            .where(
                (ModelVersionRecord.version_string == version_string)
                & (ModelVersionRecord.model_id.in_(models))
            )
        ).all()
        store_paths_by_cycle: dict[tuple[str, datetime], str] = {}
        for m_id, c_time, z_path in runs:
            if z_path:
                store_paths_by_cycle[(str(m_id), _ensure_utc_datetime(c_time))] = str(z_path)

        all_would_retire: list[LifecycleDecision] = []
        all_would_gc_candidates: list[GcCandidateInfo] = []
        all_blocked: list[datetime] = []

        for m_id in models:
            ready = list_model_ready_cycle_times(session, m_id, version_string=version_string)
            snapshots = list_cycle_lifecycle_snapshots(session, model_id=m_id, version_string=version_string)
            plan = plan_model_lifecycle(m_id, snapshots, ready)

            all_would_retire.extend(plan.would_retire)
            for g in plan.would_gc:
                store_path = store_paths_by_cycle.get(
                    (m_id, g.cycle_time),
                    canonical_cycle_store_path(m_id, g.cycle_time, base_bucket=base_bucket),
                )
                all_would_gc_candidates.append(
                    GcCandidateInfo(
                        model_id=m_id,
                        cycle_time=g.cycle_time,
                        retired_by_cycle_time=g.retired_by_cycle_time,
                        cutoff=g.cutoff,
                        store_path=store_path,
                        gc_eligible_by_cycle_time=g.retired_by_cycle_time,
                        gfs_store_path=store_path if m_id == "gfs" else None,
                        gefs_store_path=store_path if m_id == "gefs" else None,
                    )
                )
            all_blocked.extend(b.cycle_time for b in plan.blocked)

    if dry_run:
        logger.info(
            "gc_dry_run_plan: would_retire=%d would_gc=%d",
            len(all_would_retire),
            len(all_would_gc_candidates),
            extra={
                "event": "gc_dry_run_plan",
                "would_retire": [f"{r.model_id}:{r.cycle_time.isoformat()}" for r in all_would_retire],
                "would_gc": [f"{g.model_id}:{g.cycle_time.isoformat()}" for g in all_would_gc_candidates],
            },
        )
        return GcPassResult(
            dry_run=True,
            evaluated_at=now_utc,
            would_retire=tuple(all_would_retire),
            would_gc=tuple(all_would_gc_candidates),
            processed_gc=(),
            blocked_gc=(),
            locked_gc=(),
            failed_gc=(),
        )

    # Real execution mode: process GC candidates oldest-first
    processed: list[datetime] = []
    locked: list[datetime] = []
    failed: list[datetime] = []

    all_would_gc_candidates.sort(key=lambda c: c.cycle_time)

    for candidate in all_would_gc_candidates:
        try:
            success = process_gc_candidate(
                engine,
                candidate,
                base_bucket=base_bucket,
                timeout_seconds=timeout_seconds,
                version_string=version_string,
                now=now_utc,
            )
            if success:
                processed.append(candidate.cycle_time)
            else:
                locked.append(candidate.cycle_time)
        except Exception as exc:
            logger.error(
                "gc_candidate_failed: model=%s cycle_time=%s error=%s",
                candidate.model_id,
                candidate.cycle_time.isoformat(),
                exc,
                extra={
                    "event": "gc_candidate_failed",
                    "model": candidate.model_id,
                    "cycle_time": candidate.cycle_time.isoformat(),
                    "error": str(exc),
                },
            )
            failed.append(candidate.cycle_time)

    logger.info(
        "gc_pass_completed: processed=%d locked=%d failed=%d",
        len(processed),
        len(locked),
        len(failed),
    )
    return GcPassResult(
        dry_run=False,
        evaluated_at=now_utc,
        would_retire=tuple(all_would_retire),
        would_gc=tuple(all_would_gc_candidates),
        processed_gc=tuple(processed),
        blocked_gc=tuple(all_blocked),
        locked_gc=tuple(locked),
        failed_gc=tuple(failed),
    )
