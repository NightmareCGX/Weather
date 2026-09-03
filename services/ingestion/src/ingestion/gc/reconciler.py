"""Physical GC engine, sequential store deleter, and atomic catalog cleaner.

This module owns physical storage reclamation for retired, GC-eligible forecast
cycles (Phase 6D).

Execution Pipeline (per candidate cycle C):
------------------------------------------
1. Re-check GC eligibility against durable PostgreSQL catalog state.
2. Acquire EXCLUSIVE store gate on GFS store -> recursive delete GFS -> release gate.
3. Acquire EXCLUSIVE store gate on GEFS store -> recursive delete GEFS -> release gate.
4. Execute atomic PostgreSQL transaction:
   - Delete cycle's ensemble_member_products rows
   - Delete cycle's ensemble_members rows
   - Delete cycle's forecast_products rows
   - Delete cycle's model_runs rows
   - Update forecast_cycle_lifecycle.deleted_at = NOW() (tombstone)
   - Commit transaction

Invariants:
-----------
- GFS and GEFS store gates are NEVER held concurrently (deadlock-free).
- If store gate cannot be acquired due to active readers/writers, GC skips the
  cycle for the current pass and retries on the next pass.
- Missing S3/MinIO prefixes are treated as idempotent success (crash-safe).
- Dry-run mode makes ZERO mutations (no locks, no deletes, no DB writes).
- Lifecycle row survives indefinitely as a tombstone.
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

from domain.lifecycle import (
    RetirementDecision,
    canonical_cycle_store_path,
    find_r2,
    plan_lifecycle,
)
from ingestion.core.catalog import (
    EnsembleMemberProductRecord,
    EnsembleMemberRecord,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ProductRecord,
    _ensure_utc_datetime,
    _utcnow,
    ensure_lifecycle_row,
    list_cycle_lifecycle_snapshots,
    list_paired_ready_cycle_times,
    reconcile_cycle_lifecycle,
)
from ingestion.core.config import settings
from ingestion.core.locks import LockTimeoutError, StoreLockCoordinator
from ingestion.core.s3 import get_control_s3_fs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GcCandidateInfo:
    """Diagnostic info for a GC candidate."""

    cycle_time: datetime
    retired_by_cycle_time: datetime | None
    gc_eligible_by_cycle_time: datetime | None
    gfs_store_path: str
    gefs_store_path: str


@dataclass(frozen=True)
class GcPassResult:
    """Authoritative result of one GC reconciliation pass."""

    dry_run: bool
    evaluated_at: datetime
    would_retire: tuple[RetirementDecision, ...]
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
    cycle_time: datetime,
    *,
    version_string: str = "v1.0",
) -> tuple[bool, str, datetime | None]:
    """Re-read PostgreSQL state and re-verify that cycle_time is currently GC eligible.

    Monotonic Recovery Rule:
    - If deletion_started_at IS NOT NULL and deleted_at IS NULL:
      The cycle was ALREADY authorized and claimed by a prior GC pass.
      It is valid for deletion resume even if historical successor runs were later cleaned up.
    - If deletion_started_at IS NULL:
      Requires retired_at IS NOT NULL, retired_by_cycle_time IS NOT NULL, and
      earliest R2 >= R1 + 6h in current paired-ready catalog state.

    Returns:
        (is_eligible, reason, r2_time)
    """
    c_utc = _ensure_utc_datetime(cycle_time)
    lc = session.get(ForecastCycleLifecycleRecord, c_utc)
    if lc is None:
        return False, "no_lifecycle_record", None
    if lc.deleted_at is not None:
        return False, "cycle_already_deleted", None
    if lc.retired_at is None or lc.retired_by_cycle_time is None:
        return False, "cycle_not_retired", None

    r1 = _ensure_utc_datetime(lc.retired_by_cycle_time)

    # Monotonic recovery: if deletion fence was already committed, deletion is authorized to resume
    if lc.deletion_started_at is not None:
        return True, "gc_claimed_resumable", r1

    paired_ready = list_paired_ready_cycle_times(session, version_string=version_string)
    r2 = find_r2(r1, paired_ready)
    if r2 is None:
        return False, "no_qualifying_r2_paired_ready", None
    return True, "gc_eligible", r2


def claim_cycle_for_deletion(
    session: Session,
    cycle_time: datetime,
    *,
    now: datetime | None = None,
) -> bool:
    """Atomically stamp deletion_started_at on forecast_cycle_lifecycle.

    Establishes the durable cycle deletion fence before physical store deletion.
    If already claimed (deletion_started_at is not None), returns True (idempotent resume).
    If already deleted (deleted_at is not None), returns False.
    """
    c_utc = _ensure_utc_datetime(cycle_time)
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()

    lc = ensure_lifecycle_row(session, c_utc)
    if lc.deleted_at is not None:
        return False
    if lc.deletion_started_at is not None:
        return True

    setattr(lc, "deletion_started_at", now_utc)
    setattr(lc, "updated_at", now_utc)
    session.commit()
    logger.info(
        "gc_cycle_claimed: cycle_time=%s deletion_started_at=%s",
        c_utc.isoformat(),
        now_utc.isoformat(),
        extra={
            "event": "gc_cycle_claimed",
            "cycle_time": c_utc.isoformat(),
            "deletion_started_at": now_utc.isoformat(),
        },
    )
    return True


def cleanup_cycle_catalog_and_tombstone(
    session: Session,
    cycle_time: datetime,
    *,
    now: datetime | None = None,
) -> None:
    """Atomically delete cycle-scoped catalog rows and stamp deleted_at tombstone."""
    c_utc = _ensure_utc_datetime(cycle_time)
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()

    # 1. Delete child ensemble_member_products rows
    session.execute(
        delete(EnsembleMemberProductRecord).where(
            EnsembleMemberProductRecord.run_id.in_(
                select(ModelRunRecord.id).where(ModelRunRecord.cycle_time == c_utc)
            )
        )
    )

    # 2. Delete child ensemble_members rows
    session.execute(
        delete(EnsembleMemberRecord).where(
            EnsembleMemberRecord.run_id.in_(
                select(ModelRunRecord.id).where(ModelRunRecord.cycle_time == c_utc)
            )
        )
    )

    # 3. Delete child forecast_products rows
    session.execute(
        delete(ProductRecord).where(
            ProductRecord.run_id.in_(
                select(ModelRunRecord.id).where(ModelRunRecord.cycle_time == c_utc)
            )
        )
    )

    # 4. Delete model_runs rows for cycle_time
    session.execute(
        delete(ModelRunRecord).where(ModelRunRecord.cycle_time == c_utc)
    )

    # 5. Update tombstone on forecast_cycle_lifecycle
    lc = ensure_lifecycle_row(session, c_utc)
    setattr(lc, "deleted_at", now_utc)
    setattr(lc, "updated_at", now_utc)
    session.commit()
    logger.info(
        "gc_catalog_cleanup_completed: cycle_time=%s",
        c_utc.isoformat(),
        extra={
            "event": "gc_catalog_cleanup_completed",
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
    # 1. Re-check eligibility against fresh catalog state
    with Session(engine) as session:
        is_eligible, reason, r2 = recheck_gc_eligibility(
            session, c_utc, version_string=version_string
        )
    if not is_eligible:
        logger.warning(
            "gc_candidate_skipped: cycle_time=%s reason=%s",
            c_utc.isoformat(),
            reason,
            extra={
                "event": "gc_candidate_skipped",
                "cycle_time": c_utc.isoformat(),
                "reason": reason,
            },
        )
        return False

    # 2. Atomically establish durable deletion fence before touching storage
    with Session(engine) as session:
        claimed = claim_cycle_for_deletion(session, c_utc, now=now)
    if not claimed:
        return False

    # 3. Sequential gated physical store deletion (GFS first)
    gfs_deleted = delete_physical_store_gated(
        engine,
        candidate.gfs_store_path,
        timeout_seconds=timeout_seconds,
    )
    if not gfs_deleted:
        return False

    # 4. Sequential gated physical store deletion (GEFS second)
    gefs_deleted = delete_physical_store_gated(
        engine,
        candidate.gefs_store_path,
        timeout_seconds=timeout_seconds,
    )
    if not gefs_deleted:
        return False

    # 5. Atomic catalog cleanup and tombstone commit
    with Session(engine) as session:
        cleanup_cycle_catalog_and_tombstone(session, c_utc, now=now)

    logger.info(
        "gc_completed: cycle_time=%s",
        c_utc.isoformat(),
        extra={"event": "gc_completed", "cycle_time": c_utc.isoformat()},
    )
    return True


def run_gc_pass(
    engine: Engine,
    *,
    dry_run: bool = False,
    version_string: str = "v1.0",
    base_bucket: str = "weather-data",
    timeout_seconds: float = 5.0,
    now: datetime | None = None,
) -> GcPassResult:
    """Execute one complete GC pass (discovery, planning, retirement, deletion).

    In dry_run mode: plans without taking exclusive locks or mutating S3/PostgreSQL.
    In real mode: persists retirements and deletes GC-eligible stores sequentially.

    Args:
        engine: SQLAlchemy Engine for catalog and advisory lock coordination.
        dry_run: If True, execute observational planning only.
        version_string: Target model version string.
        base_bucket: S3/MinIO bucket name.
        timeout_seconds: Timeout for exclusive store gate acquisition.
        now: Optional datetime override for testing.

    Returns:
        GcPassResult with structured diagnostics.
    """
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    logger.info("gc_pass_started: dry_run=%s now=%s", dry_run, now_utc.isoformat())

    with Session(engine) as session:
        if not dry_run:
            # 1. Reconcile newly eligible retirements
            reconcile_cycle_lifecycle(session, now=now_utc, version_string=version_string)

        # 2. Query snapshots and paired-ready state
        snapshots = list_cycle_lifecycle_snapshots(session, version_string=version_string)
        paired_ready = list_paired_ready_cycle_times(session, version_string=version_string)

        # 3. Discover recorded store paths from model_runs for accurate deletion
        from ingestion.core.catalog import ModelVersionRecord

        runs = session.execute(
            select(
                ModelVersionRecord.model_id,
                ModelRunRecord.cycle_time,
                ModelRunRecord.zarr_store_path,
            )
            .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
            .where(
                (ModelVersionRecord.version_string == version_string)
                & (ModelVersionRecord.model_id.in_(["gfs", "gefs"]))
            )
        ).all()
        store_paths_by_cycle: dict[tuple[datetime, str], str] = {}
        for m_id, c_time, z_path in runs:
            if z_path:
                store_paths_by_cycle[(_ensure_utc_datetime(c_time), str(m_id))] = str(z_path)

    # 4. Plan lifecycle transitions deterministically
    plan = plan_lifecycle(snapshots, paired_ready)

    would_gc_candidates = tuple(
        GcCandidateInfo(
            cycle_time=g.cycle_time,
            retired_by_cycle_time=g.retired_by_cycle_time,
            gc_eligible_by_cycle_time=g.gc_eligible_by_cycle_time,
            gfs_store_path=store_paths_by_cycle.get(
                (g.cycle_time, "gfs"),
                canonical_cycle_store_path("gfs", g.cycle_time, base_bucket=base_bucket),
            ),
            gefs_store_path=store_paths_by_cycle.get(
                (g.cycle_time, "gefs"),
                canonical_cycle_store_path("gefs", g.cycle_time, base_bucket=base_bucket),
            ),
        )
        for g in plan.would_gc
    )

    if dry_run:
        logger.info(
            "gc_dry_run_plan: would_retire=%d would_gc=%d active_visible=%d",
            len(plan.would_retire),
            len(would_gc_candidates),
            len(plan.active_visible_cycles),
            extra={
                "event": "gc_dry_run_plan",
                "would_retire": [r.cycle_time.isoformat() for r in plan.would_retire],
                "would_gc": [g.cycle_time.isoformat() for g in would_gc_candidates],
                "active_visible": [c.isoformat() for c in plan.active_visible_cycles],
                "already_deleted": [d.isoformat() for d in plan.deleted_cycles],
            },
        )
        return GcPassResult(
            dry_run=True,
            evaluated_at=now_utc,
            would_retire=plan.would_retire,
            would_gc=would_gc_candidates,
            processed_gc=(),
            blocked_gc=(),
            locked_gc=(),
            failed_gc=(),
        )

    # Real execution mode: process GC candidates oldest-first
    processed: list[datetime] = []
    locked: list[datetime] = []
    failed: list[datetime] = []

    for candidate in would_gc_candidates:
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
                "gc_candidate_failed: cycle_time=%s error=%s",
                candidate.cycle_time.isoformat(),
                exc,
                extra={
                    "event": "gc_candidate_failed",
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
        would_retire=plan.would_retire,
        would_gc=would_gc_candidates,
        processed_gc=tuple(processed),
        blocked_gc=tuple(g.cycle_time for g in plan.gc_eligibilities if not g.is_gc_eligible),
        locked_gc=tuple(locked),
        failed_gc=tuple(failed),
    )
