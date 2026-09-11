"""Granular physical reclamation worker for Data Lifecycle V3 Phase 4.

Leases and claims batches from ``reclamation_queue``, acquires SHARED store gates,
performs counterfactual semantic revalidation, executes direct S3/disk DeleteObject,
deletes completed region markers, and provides quarantine/retry semantics.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.canonical import (
    CanonicalCandidate,
    filter_candidates_by_physical_fence,
    select_canonical_sources_bulk,
)
from domain.coverage import get_expected_members, is_lead_servable
from domain.horizon import model_max_lead_hours
from domain.reclamation import (
    PREDECESSOR_VARIABLES,
    RECLAMATION_STATUS_DELETED,
    RECLAMATION_STATUS_DELETING,
    RECLAMATION_STATUS_FAILED,
    RECLAMATION_STATUS_QUEUED,
    TARGET_KIND_DET,
    TARGET_KIND_MEAN,
    TARGET_KIND_MEM,
    can_delete_region_marker,
    get_expected_region_variables,
    get_predecessor_lead,
    make_region_marker_relative_key,
)
from domain.temporal import (
    is_precipitation_companion,
    requires_lead0_display_fallback,
    serving_start_valid_time,
)
from ingestion.core.catalog import (
    EnsembleMemberProductRecord,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    ReclamationQueueRecord,
    _ensure_utc_datetime,
    _utcnow,
)
from ingestion.core.config import settings
from ingestion.core.locks import LockTimeoutError, StoreLockCoordinator
from ingestion.core.s3 import get_control_s3_fs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReclamationWorkerResult:
    """Result of one reclamation worker processing pass."""

    claimed_count: int
    deleted_count: int
    revalidated_held_count: int
    failed_count: int
    markers_cleaned_count: int
    evaluated_at: datetime


def _delete_physical_object(store_path: str, physical_key: str) -> None:
    """Delete a single physical object idempotently (S3 or local file)."""
    if store_path.startswith("s3://"):
        fs = get_control_s3_fs(settings)
        clean_store = store_path[len("s3://") :].rstrip("/")
        clean_key = physical_key.lstrip("/")
        raw_path = f"{clean_store}/{clean_key}"
        try:
            if fs.exists(raw_path):
                fs.rm(raw_path)
        except Exception as exc:
            logger.debug("S3 object rm on %s (ignoring error): %s", raw_path, exc)
    else:
        base = store_path[len("file://") :] if store_path.startswith("file://") else store_path
        full_path = os.path.join(base, physical_key)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
        except OSError:
            pass


def _delete_region_marker_object(store_path: str, marker_key: str) -> None:
    """Delete a region commit marker object idempotently."""
    _delete_physical_object(store_path, marker_key)


def _physical_object_exists(store_path: str, physical_key: str) -> bool:
    """Return True if the physical object positively exists on S3 or disk."""
    if store_path.startswith("s3://"):
        fs = get_control_s3_fs(settings)
        clean_store = store_path[len("s3://") :].rstrip("/")
        clean_key = physical_key.lstrip("/")
        raw_path = f"{clean_store}/{clean_key}"
        try:
            return bool(fs.exists(raw_path))
        except Exception:
            return False
    base = store_path[len("file://") :] if store_path.startswith("file://") else store_path
    full_path = os.path.join(base, physical_key)
    return os.path.exists(full_path)


def run_reclamation_worker_pass(
    session: Session,
    *,
    batch_size: int | None = None,
    lease_seconds: float | None = None,
    delete_enabled: bool | None = None,
    max_retries: int | None = None,
    base_backoff_seconds: float | None = None,
    now: datetime | None = None,
    version_string: str = "v1.0",
) -> ReclamationWorkerResult:
    """Execute a single worker reclamation pass.

    Step 1: Claim bounded batch using SELECT FOR UPDATE SKIP LOCKED (including expired leases).
    Step 2: Group claimed targets by store_path.
    Step 3: For each store, acquire SHARED store gate.
    Step 4: Under store gate, re-read whole-cycle lifecycle: abort if whole-cycle GC claimed cycle.
    Step 5: Perform Counterfactual Semantic Revalidation (claimed batch evaluated as physically present).
    Step 6: If target required, revert to 'queued' and clear lease.
    Step 7: If DeleteObject authorized (delete_enabled=True), delete physical shard and mark 'deleted'.
    Step 8: Check and delete region markers dynamically derived from catalog commit records.
    """
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    b_size = batch_size if batch_size is not None else settings.RECLAMATION_BATCH_SIZE
    l_secs = lease_seconds if lease_seconds is not None else settings.RECLAMATION_LEASE_SECONDS
    del_en = delete_enabled if delete_enabled is not None else settings.RECLAMATION_DELETE_ENABLED
    m_retries = max_retries if max_retries is not None else settings.RECLAMATION_MAX_RETRIES
    b_backoff = (
        base_backoff_seconds
        if base_backoff_seconds is not None
        else settings.RECLAMATION_BASE_BACKOFF_SECONDS
    )
    lease_until = now_utc + timedelta(seconds=l_secs)

    is_postgres = bool(session.bind and session.bind.dialect.name == "postgresql")

    # 1. Claim batch
    claim_query = (
        select(ReclamationQueueRecord)
        .where(
            (ReclamationQueueRecord.status == RECLAMATION_STATUS_QUEUED)
            | (
                (ReclamationQueueRecord.status == RECLAMATION_STATUS_DELETING)
                & (ReclamationQueueRecord.lease_expires_at < now_utc)
            )
        )
        .where(
            (ReclamationQueueRecord.next_retry_at.is_(None))
            | (ReclamationQueueRecord.next_retry_at <= now_utc)
        )
        .order_by(ReclamationQueueRecord.cycle_time.asc(), ReclamationQueueRecord.created_at.asc())
        .limit(b_size)
    )
    if is_postgres:
        claim_query = claim_query.with_for_update(skip_locked=True)

    claimed_rows = list(session.execute(claim_query).scalars().all())
    if not claimed_rows:
        return ReclamationWorkerResult(
            claimed_count=0,
            deleted_count=0,
            revalidated_held_count=0,
            failed_count=0,
            markers_cleaned_count=0,
            evaluated_at=now_utc,
        )

    # Stamp 'deleting' status and set lease
    claimed_ids = [row.id for row in claimed_rows]
    for row in claimed_rows:
        row.status = RECLAMATION_STATUS_DELETING
        row.lease_expires_at = lease_until
        row.attempt_count += 1
        row.updated_at = now_utc
    session.commit()

    claimed_by_store: dict[str, list[ReclamationQueueRecord]] = {}
    for row in claimed_rows:
        claimed_by_store.setdefault(row.store_path, []).append(row)

    total_deleted = 0
    total_revalidated = 0
    total_failed = 0
    total_markers = 0

    # 2. Process each store under its SHARED store gate
    for store_path, targets in claimed_by_store.items():
        # Advisory lock coordinator if postgres
        coord: StoreLockCoordinator | None = None
        if is_postgres and session.bind:
            conn = session.connection()
            coord = StoreLockCoordinator(
                conn,
                store_path=store_path,
                timeout_seconds=5.0,
            )
            try:
                coord.acquire_shared_gate()
            except LockTimeoutError:
                # Store gate blocked (e.g. V2 whole-cycle GC or writer holding exclusive gate)
                logger.warning("Reclamation worker store gate blocked on %s, skipping batch", store_path)
                for t in targets:
                    t.status = RECLAMATION_STATUS_QUEUED
                    t.lease_expires_at = None
                    t.last_error = "store_gate_timeout"
                session.commit()
                continue

        try:
            # Re-read whole-cycle lifecycle under the gate
            first_target = targets[0]
            m_id = first_target.model_id
            c_utc = _ensure_utc_datetime(first_target.cycle_time)
            lc = session.get(ForecastCycleLifecycleRecord, (m_id, c_utc))
            if lc is not None and (lc.deletion_started_at is not None or lc.deleted_at is not None):
                # V2 whole-cycle GC claimed or deleted this cycle: abort granular reclamation
                logger.info("Cycle %s %s claimed by whole-cycle GC, aborting granular GC", m_id, c_utc)
                for t in targets:
                    t.status = RECLAMATION_STATUS_QUEUED
                    t.lease_expires_at = None
                    t.last_error = "cycle_claimed_by_whole_cycle_gc"
                session.commit()
                continue

            # 3. Counterfactual Semantic Revalidation
            # Evaluate reachability where targets in current claimed batch are NOT fenced
            fenced_query = select(
                ReclamationQueueRecord.run_id,
                ReclamationQueueRecord.lead_time_hours,
                ReclamationQueueRecord.variable_code,
                ReclamationQueueRecord.target_kind,
                ReclamationQueueRecord.member_index,
            ).where(
                ReclamationQueueRecord.status.in_(
                    [
                        RECLAMATION_STATUS_DELETING,
                        RECLAMATION_STATUS_DELETED,
                        RECLAMATION_STATUS_FAILED,
                    ]
                ),
                ReclamationQueueRecord.id.not_in(claimed_ids),
            )
            other_fenced = {
                (str(r), int(ld), str(v), str(k), int(m))
                for r, ld, v, k, m in session.execute(fenced_query).all()
            }

            # Discover candidates for this model
            serving_start = serving_start_valid_time(now_utc)
            m_max_lead = model_max_lead_hours(m_id)
            min_recovery_cycle = serving_start - timedelta(hours=m_max_lead)
            expected_members = get_expected_members(m_id, default_if_unknown=1)
            is_ensemble = expected_members > 1

            runs_stmt = (
                select(
                    ModelRunRecord.id,
                    ModelRunRecord.cycle_time,
                    ModelRunRecord.status,
                    ModelRunRecord.zarr_store_path,
                )
                .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
                .where(
                    ModelVersionRecord.model_id == m_id,
                    ModelVersionRecord.version_string == version_string,
                    ModelRunRecord.status.in_(("ready", "processing", "partial")),
                    ModelRunRecord.zarr_store_path.isnot(None),
                )
            )
            run_rows = session.execute(runs_stmt).all()
            eligible_runs = {str(r[0]): (_ensure_utc_datetime(r[1]), str(r[2]), str(r[3])) for r in run_rows}

            prod_rows = session.execute(
                select(
                    ProductRecord.run_id,
                    ProductRecord.lead_time_hours,
                    ProductRecord.variable_id,
                    ProductRecord.product_type,
                ).where(ProductRecord.run_id.in_(list(eligible_runs.keys())))
            ).all()

            emp_mems: dict[tuple[str, int], list[int]] = {}
            if is_ensemble:
                emp_rows = session.execute(
                    select(
                        EnsembleMemberProductRecord.run_id,
                        EnsembleMemberProductRecord.lead_time_hours,
                        EnsembleMemberProductRecord.member_index,
                    ).where(EnsembleMemberProductRecord.run_id.in_(list(eligible_runs.keys())))
                ).all()
                for r_id, lead, mem in emp_rows:
                    emp_mems.setdefault((str(r_id), int(lead)), []).append(int(mem))

            # Build candidates
            by_rl: dict[tuple[str, int], dict[str, Any]] = {}
            for r_id, lead, var, p_type in prod_rows:
                r_str = str(r_id)
                lead_num = int(lead)
                rec = by_rl.setdefault(
                    (r_str, lead_num),
                    {
                        "run_id": r_str,
                        "cycle_time": eligible_runs[r_str][0],
                        "lead_time_hours": lead_num,
                        "store_path": eligible_runs[r_str][2],
                        "status": eligible_runs[r_str][1],
                        "product_types": set(),
                        "variables": set(),
                    },
                )
                rec["product_types"].add(str(p_type))
                rec["variables"].add(str(var))

            cands_by_vt: dict[datetime, list[CanonicalCandidate]] = {}
            for (r_str, lead_num), rec in by_rl.items():
                c_utc_cand = rec["cycle_time"]
                v_time = c_utc_cand + timedelta(hours=lead_num)
                m_indices = None
                if is_ensemble:
                    raw_m = emp_mems.get((r_str, lead_num), [])
                    if not is_lead_servable(len(raw_m), expected_members):
                        continue
                    if not rec["product_types"]:
                        continue
                    m_indices = tuple(sorted(raw_m))
                cands_by_vt.setdefault(v_time, []).append(
                    CanonicalCandidate(
                        cycle_time=c_utc_cand,
                        lead_time_hours=lead_num,
                        run_id=r_str,
                        store_path=rec["store_path"],
                        product_types=frozenset(rec["product_types"]),
                        variables=frozenset(rec["variables"]),
                        member_indices=m_indices,
                        status=rec["status"],
                    )
                )

            for vt in cands_by_vt:
                cands_by_vt[vt].sort(key=lambda c: c.lead_time_hours)

            # Filter candidates using counterfactual fence (other_fenced)
            filtered_cands = filter_candidates_by_physical_fence(
                cands_by_vt, other_fenced, is_ensemble=is_ensemble, expected_members=expected_members
            )

            distinct_vars = sorted({t.variable_code for t in targets})
            if "wind_u_10m" in distinct_vars and "wind_v_10m" in distinct_vars:
                distinct_vars.append("wind_10m")

            canon_res = select_canonical_sources_bulk(
                filtered_cands,
                variables=distinct_vars,
                start_valid_time=serving_start,
            )

            # Build set of counterfactually necessary target tuples
            necessary_tuples: dict[tuple[str, int, str, str, int], str] = {}

            # Anchors
            for vt, anchor_cand in canon_res.anchors.items():
                if anchor_cand.lead_time_hours > 0:
                    for v in anchor_cand.variables:
                        if is_ensemble:
                            necessary_tuples[(anchor_cand.run_id, anchor_cand.lead_time_hours, v, TARGET_KIND_MEAN, -1)] = "anchor"
                            for m in anchor_cand.member_indices or ():
                                necessary_tuples[(anchor_cand.run_id, anchor_cand.lead_time_hours, v, TARGET_KIND_MEM, m)] = "anchor"
                        else:
                            necessary_tuples[(anchor_cand.run_id, anchor_cand.lead_time_hours, v, TARGET_KIND_DET, 0)] = "anchor"
                else:
                    for v in anchor_cand.variables:
                        if not (requires_lead0_display_fallback(v) or is_precipitation_companion(v)):
                            if is_ensemble:
                                necessary_tuples[(anchor_cand.run_id, 0, v, TARGET_KIND_MEAN, -1)] = "anchor"
                                for m in anchor_cand.member_indices or ():
                                    necessary_tuples[(anchor_cand.run_id, 0, v, TARGET_KIND_MEM, m)] = "anchor"
                            else:
                                necessary_tuples[(anchor_cand.run_id, 0, v, TARGET_KIND_DET, 0)] = "anchor"

            # Fallbacks and companions
            for (v, vt), src_cand in canon_res.variable_sources.items():
                if src_cand is not None:
                    if is_ensemble:
                        necessary_tuples[(src_cand.run_id, src_cand.lead_time_hours, v, TARGET_KIND_MEAN, -1)] = "variable_source"
                        for m in src_cand.member_indices or ():
                            necessary_tuples[(src_cand.run_id, src_cand.lead_time_hours, v, TARGET_KIND_MEM, m)] = "variable_source"
                    else:
                        necessary_tuples[(src_cand.run_id, src_cand.lead_time_hours, v, TARGET_KIND_DET, 0)] = "variable_source"

            # Predecessors
            for r_id_str, (c_utc_r, st_r, p_r) in eligible_runs.items():
                if c_utc_r >= min_recovery_cycle and st_r != "ready":
                    committed_leads = {ld for (r, ld) in by_rl.keys() if r == r_id_str}
                    for r_lead in range(6, m_max_lead + 1, 6):
                        if r_lead not in committed_leads:
                            p_lead = get_predecessor_lead(r_lead)
                            if p_lead in committed_leads:
                                for p_var in PREDECESSOR_VARIABLES:
                                    if is_ensemble:
                                        necessary_tuples[(r_id_str, p_lead, p_var, TARGET_KIND_MEAN, -1)] = "predecessor"
                                        for m in range(1, expected_members + 1):
                                            necessary_tuples[(r_id_str, p_lead, p_var, TARGET_KIND_MEM, m)] = "predecessor"
                                    else:
                                        necessary_tuples[(r_id_str, p_lead, p_var, TARGET_KIND_DET, 0)] = "predecessor"

            # 4. Process each target in the store's claimed batch
            deleted_targets_for_store: list[ReclamationQueueRecord] = []
            for target in targets:
                t_tuple = (
                    target.run_id,
                    target.lead_time_hours,
                    target.variable_code,
                    target.target_kind,
                    target.member_index,
                )
                if t_tuple in necessary_tuples:
                    reason = necessary_tuples[t_tuple]
                    logger.info(
                        "Revalidation abort: target %s became required (%s)",
                        t_tuple,
                        reason,
                    )
                    target.status = RECLAMATION_STATUS_QUEUED
                    target.lease_expires_at = None
                    target.last_error = f"counterfactual_revalidation_abort: {reason}"
                    target.updated_at = now_utc
                    total_revalidated += 1
                    continue

                # Safe to delete physically
                if not del_en:
                    # Deletion not enabled: revert to queued and log
                    target.status = RECLAMATION_STATUS_QUEUED
                    target.lease_expires_at = None
                    target.last_error = "deletion_disabled"
                    target.updated_at = now_utc
                    continue

                try:
                    _delete_physical_object(target.store_path, target.physical_key)
                    target.status = RECLAMATION_STATUS_DELETED
                    target.reclaimed_at = now_utc
                    target.lease_expires_at = None
                    target.last_error = None
                    target.updated_at = now_utc
                    deleted_targets_for_store.append(target)
                    total_deleted += 1
                except Exception as exc:
                    logger.error("Physical delete failed on %s: %s", target.physical_key, exc)
                    if target.attempt_count >= m_retries:
                        target.status = RECLAMATION_STATUS_FAILED
                        target.lease_expires_at = None
                        target.last_error = f"max_retries_exceeded: {exc}"
                        total_failed += 1
                    else:
                        backoff = b_backoff * (2 ** target.attempt_count)
                        target.status = RECLAMATION_STATUS_QUEUED
                        target.lease_expires_at = None
                        target.next_retry_at = now_utc + timedelta(seconds=backoff)
                        target.last_error = f"delete_error: {exc}"
                    target.updated_at = now_utc

            session.commit()

            # 5. Region marker cleanup for deleted targets
            if deleted_targets_for_store and del_en:
                distinct_regions = {
                    (
                        t.run_id,
                        t.lead_time_hours,
                        t.target_kind,
                        t.member_index,
                    )
                    for t in deleted_targets_for_store
                }
                for r_id, lead_h, t_kind, mem_idx in distinct_regions:
                    # Authoritative model/version schema for this model and region kind (no truncated catalog rows!)
                    try:
                        run_rec = session.get(ModelRunRecord, r_id)
                        run_ver = version_string
                        if run_rec and run_rec.model_version_id:
                            mv = session.get(ModelVersionRecord, run_rec.model_version_id)
                            if mv and mv.version_string:
                                run_ver = str(mv.version_string)
                        expected_vars = get_expected_region_variables(
                            str(m_id), t_kind, version_string=run_ver
                        )
                    except ValueError:
                        continue

                    # Query deleted variables in reclamation_queue for this region
                    deleted_vars = set(
                        session.execute(
                            select(ReclamationQueueRecord.variable_code).where(
                                ReclamationQueueRecord.run_id == r_id,
                                ReclamationQueueRecord.lead_time_hours == lead_h,
                                ReclamationQueueRecord.target_kind == t_kind,
                                ReclamationQueueRecord.member_index == mem_idx,
                                ReclamationQueueRecord.status == RECLAMATION_STATUS_DELETED,
                            )
                        ).scalars().all()
                    )

                    if can_delete_region_marker(expected_vars, deleted_vars):
                        marker_key = make_region_marker_relative_key(t_kind, lead_h, mem_idx)
                        _delete_region_marker_object(store_path, marker_key)
                        total_markers += 1
                        logger.info("Deleted region marker: %s in %s", marker_key, store_path)

        finally:
            if coord is not None:
                coord.release_shared_gate()

    return ReclamationWorkerResult(
        claimed_count=len(claimed_rows),
        deleted_count=total_deleted,
        revalidated_held_count=total_revalidated,
        failed_count=total_failed,
        markers_cleaned_count=total_markers,
        evaluated_at=now_utc,
    )


def requeue_failed_reclamation_targets(
    session: Session,
    *,
    model_id: str | None = None,
    run_id: str | None = None,
) -> int:
    """Operator CLI helper: reset 'failed' quarantined reclamation targets.

    Safe Requeue Semantics:
    - If the physical object is confirmed already absent on storage (prior delete succeeded
      remotely before network drop): promote immediately to 'deleted' with reclaimed_at timestamp!
      Prevents accidental un-fencing or serving of missing files.
    - If the physical object is positively confirmed still present on storage:
      reset attempt_count to 0 and requeue for worker retry.
    """
    stmt = select(ReclamationQueueRecord).where(
        ReclamationQueueRecord.status == RECLAMATION_STATUS_FAILED
    )
    if model_id is not None:
        stmt = stmt.where(ReclamationQueueRecord.model_id == model_id.lower().strip())
    if run_id is not None:
        stmt = stmt.where(ReclamationQueueRecord.run_id == run_id)

    rows = list(session.execute(stmt).scalars().all())
    now_utc = _utcnow()
    for row in rows:
        if not _physical_object_exists(str(row.store_path), str(row.physical_key)):
            # Object already absent on storage: finalize as deleted
            row.status = RECLAMATION_STATUS_DELETED
            row.reclaimed_at = now_utc
            row.lease_expires_at = None
            row.next_retry_at = None
            row.last_error = None
            row.updated_at = now_utc
        else:
            # Positively confirmed present: safe to reset for worker retry
            row.status = RECLAMATION_STATUS_QUEUED
            row.attempt_count = 0
            row.lease_expires_at = None
            row.next_retry_at = None
            row.last_error = None
            row.updated_at = now_utc
    session.commit()
    return len(rows)
