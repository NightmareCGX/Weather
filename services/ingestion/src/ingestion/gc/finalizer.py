"""Lifecycle bookkeeping / metadata reconciliation (Data Lifecycle V3, converged).

This module owns the **derived** cycle-level bookkeeping of the V3 GC model.
There is no cycle-level GC authority: physical deletion happens exclusively
through the Reclamation Planner + Reclamation Worker at (variable, valid_time)
granularity. This module only:

1. Claims retirement (``deletion_started_at`` serving & mutation fence) for
   horizon-expired cycles — a conservative derived fast-path; claiming is what
   guarantees no *new* canonical holds can form on the cycle while its remaining
   units are reclaimed.
2. Observes whether all committed reclamation units of a cycle are terminal
   (every committed unit has a ``reclamation_queue`` row in status ``deleted``
   and no in-flight rows remain).
3. Derives ``deleted_at`` (permanent tombstone) once all units are terminal and
   starts the 14-day metadata retention clock (owned by the sweeper).

Invariants (architecture doc §7/§8, I7/I14/I16):
------------------------------------------------
- Zero physical storage operations: no DeleteObject, no store prefix deletion,
  no store gates. ``deleted_at`` is a derived bookkeeping fact, never a
  deletion authorization.
- Once a cycle is claimed (serving fence), the planner treats all of its
  remaining units as unprotected and the worker reclaims them; this pass never
  blocks granular reclamation.
- Tombstone is only committed when every committed unit is terminal. Units with
  no catalog evidence (orphans) are the store-catalog reconciler's concern.
- Models are independent: everything is keyed by (model_id, cycle_time).
- Detailed catalog metadata (model_runs, forecast_products, ...) is retained
  intact for the 14-day retention window (owned by the sweeper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from domain.coverage import get_expected_members
from domain.horizon import model_max_lead_hours
from domain.lifecycle import is_cycle_horizon_expired
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
from ingestion.gc.planner import enumerate_committed_unit_tuples

logger = logging.getLogger(__name__)

#: Queue statuses that mean a unit still needs physical work (or failed and
#: needs operator requeue). Any of these blocks the derived tombstone.
NON_TERMINAL_QUEUE_STATUSES: tuple[str, ...] = ("queued", "deleting", "failed")


@dataclass(frozen=True)
class FinalizerCandidate:
    """A candidate cycle evaluated by the lifecycle bookkeeping pass."""

    model_id: str
    cycle_time: datetime
    is_recovery: bool


@dataclass(frozen=True)
class FinalizerPassResult:
    """Structured diagnostic result of one lifecycle bookkeeping pass."""

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
    """Discover recovery and fresh candidate cycles for lifecycle bookkeeping.

    Returns:
        (recovery_candidates, fresh_candidates)
    """
    target_models = tuple(m.lower().strip() for m in models)

    # 1. Recovery candidates: claimed for retirement but not yet tombstoned
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

    The claim is the serving & mutation fence (Guarantee B): once committed, the
    API stops selecting the cycle, the planner treats its remaining units as
    unprotected, and no new canonical holds can form. The horizon check is a
    conservative derived fast-path (architecture doc §7.1) — it never
    authorizes physical deletion by itself.

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
        "bookkeeping_cycle_claimed: model=%s cycle_time=%s versions=%s max_lead=%dh",
        m_id,
        c_utc.isoformat(),
        versions,
        max_lead,
        extra={
            "event": "bookkeeping_cycle_claimed",
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
    (model_id, cycle_time) and includes the canonical template fallback. Used by
    the store-catalog reconciliation (orphan inventory), NOT by this module.
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

    # Include canonical fallback (guarantees orphaned canonical storage is detected)
    from domain.lifecycle import canonical_cycle_store_path

    canonical = canonical_cycle_store_path(m_id, c_utc, base_bucket=base_bucket)
    recorded_paths.add(canonical)

    return sorted(p for p in recorded_paths if p)


def cycle_reclamation_units_terminal(
    session: Session,
    model_id: str,
    cycle_time: datetime,
) -> bool:
    """Return True if every committed reclamation unit of the cycle is terminal.

    Terminal means (architecture doc §0 invariant, I7/I16):
    - every committed unit (from forecast_products / ensemble_member_products,
      enumerated with the same construction the planner uses) has a
      ``reclamation_queue`` row in status ``deleted``;
    - no queue row for the cycle remains in a non-terminal status
      (queued / deleting / failed).

    Cycles with no committed catalog units are trivially terminal; physical
    stores without catalog evidence are the store-catalog reconciler's concern.
    """
    m_id = model_id.lower().strip()
    c_utc = _ensure_utc_datetime(cycle_time)

    run_ids = list(
        session.execute(
            select(ModelRunRecord.id)
            .join(
                ModelVersionRecord,
                ModelRunRecord.model_version_id == ModelVersionRecord.id,
            )
            .where(
                ModelVersionRecord.model_id == m_id,
                ModelRunRecord.cycle_time == c_utc,
            )
        ).scalars().all()
    )
    if not run_ids:
        return True

    status_by_unit: dict[tuple[str, int, str, str, int], str] = {
        (str(r), int(ld), str(v), str(k), int(m)): str(st)
        for r, ld, v, k, m, st in session.execute(
            select(
                ReclamationQueueRecord.run_id,
                ReclamationQueueRecord.lead_time_hours,
                ReclamationQueueRecord.variable_code,
                ReclamationQueueRecord.target_kind,
                ReclamationQueueRecord.member_index,
                ReclamationQueueRecord.status,
            ).where(ReclamationQueueRecord.run_id.in_(run_ids))
        ).all()
    }

    # Any in-flight or failed unit blocks the derived tombstone.
    for status in status_by_unit.values():
        if status in NON_TERMINAL_QUEUE_STATUSES:
            return False

    is_ensemble = get_expected_members(m_id, default_if_unknown=1) > 1
    units = enumerate_committed_unit_tuples(
        session, run_ids=run_ids, is_ensemble=is_ensemble
    )
    for unit in units:
        if status_by_unit.get(unit) != "deleted":
            return False
    return True


def finalize_cycle_physical_and_queue(
    engine: Engine,
    model_id: str,
    cycle_time: datetime,
    *,
    finalization_time: datetime | None = None,
) -> None:
    """Atomically commit deleted_at tombstone and normalize all queue rows to 'deleted'.

    Only callable once :func:`cycle_reclamation_units_terminal` holds — the
    tombstone is a derived bookkeeping fact (architecture doc §8). Callers must
    not use this to authorize deletion.

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
        "bookkeeping_cycle_tombstoned: model=%s cycle_time=%s deleted_at=%s",
        m_id,
        c_utc.isoformat(),
        now_utc.isoformat(),
        extra={
            "event": "bookkeeping_cycle_tombstoned",
            "model": m_id,
            "cycle_time": c_utc.isoformat(),
            "deleted_at": now_utc.isoformat(),
        },
    )


def finalize_cycle_bookkeeping(
    engine: Engine,
    model_id: str,
    cycle_time: datetime,
    *,
    is_recovery: bool,
    serving_start: datetime,
    now: datetime | None = None,
) -> bool:
    """Execute the lifecycle bookkeeping flow for one candidate cycle.

    1. Fresh eligibility claim (skipped for recovery candidates) — the serving
       & mutation fence.
    2. Verify all committed reclamation units are terminal.
    3. Derive ``deleted_at`` (tombstone) atomically.

    Returns True if tombstoned (or already deleted), False if blocked.
    Performs ZERO physical storage operations.
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

    # Step 2: Derived tombstone requires all units terminal (I7/I16)
    with Session(engine) as session:
        if not cycle_reclamation_units_terminal(session, m_id, c_utc):
            logger.info(
                "bookkeeping_units_not_terminal: model=%s cycle_time=%s; deferring",
                m_id,
                c_utc.isoformat(),
            )
            return False

    # Step 3: Atomic DB tombstone (deleted_at + queue normalization)
    finalize_cycle_physical_and_queue(
        engine, m_id, c_utc, finalization_time=now_utc
    )
    return True


def run_lifecycle_bookkeeping_pass(
    engine: Engine,
    *,
    models: Sequence[str] = ("gfs", "gefs"),
    dry_run: bool = False,
    batch_size: int = 50,
    now: datetime | None = None,
) -> FinalizerPassResult:
    """Execute a single bounded lifecycle bookkeeping pass across specified models.

    Performs ZERO physical storage operations: claims retirement fences and
    derives tombstones for cycles whose reclamation units are all terminal.
    """
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    serving_start = serving_start_valid_time(now_utc)

    logger.info(
        "bookkeeping_pass_started: dry_run=%s now=%s serving_start=%s models=%s",
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
            "bookkeeping_dry_run: recovery_candidates=%d fresh_candidates=%d",
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
            ok = finalize_cycle_bookkeeping(
                engine,
                cand.model_id,
                cand.cycle_time,
                is_recovery=True,
                serving_start=serving_start,
                now=now_utc,
            )
            if ok:
                finalized.append((cand.model_id, cand.cycle_time))
            else:
                blocked.append((cand.model_id, cand.cycle_time))
        except Exception as exc:
            logger.error(
                "bookkeeping_candidate_failed: model=%s cycle_time=%s error=%s",
                cand.model_id,
                cand.cycle_time.isoformat(),
                exc,
            )
            failed.append((cand.model_id, cand.cycle_time))

    # 2. Process fresh candidates
    for cand in fresh_cands:
        try:
            ok = finalize_cycle_bookkeeping(
                engine,
                cand.model_id,
                cand.cycle_time,
                is_recovery=False,
                serving_start=serving_start,
                now=now_utc,
            )
            if ok:
                claimed.append((cand.model_id, cand.cycle_time))
                finalized.append((cand.model_id, cand.cycle_time))
            else:
                blocked.append((cand.model_id, cand.cycle_time))
        except Exception as exc:
            logger.error(
                "bookkeeping_candidate_failed: model=%s cycle_time=%s error=%s",
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
