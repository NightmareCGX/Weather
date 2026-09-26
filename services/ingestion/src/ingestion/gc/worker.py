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
from typing import Any, Sequence

from sqlalchemy import or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from domain.canonical import (
    CanonicalCandidate,
    build_fence_index,
    filter_candidates_by_fence_index,
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
    get_variable_temporal_metadata,
    is_precipitation_companion,
    model_serving_start_valid_time,
    requires_lead0_display_fallback,
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
from ingestion.core.markers import read_store_generation
from ingestion.core.s3 import get_control_s3_fs

logger = logging.getLogger(__name__)

#: Rows the counterfactual-fence query fetches per server-side cursor round trip.
#: The fence is streamed rather than buffered, so this bounds the transient row
#: buffer (1e3 rows, well under 1 MB) while keeping the round-trip count
#: tolerable (~1.5e3 for the ~1.5e6-row steady-state fence). Measured against a
#: 1.5e6-row fence, smaller batches win: 1e3 -> 3.5s, 1e4 -> 4.1s, 5e4 -> 4.3s,
#: versus 49.8s and ~1.1 GB for the buffered ``.all()`` + set-comprehension it
#: replaces.
FENCE_FETCH_SIZE = 1_000


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


#: DeleteObjects accepts at most 1000 keys in a single request.
_S3_BULK_DELETE_MAX_KEYS = 1000


def _delete_physical_objects_batch(store_path: str, physical_keys: Sequence[str]) -> int:
    """Delete physical objects idempotently, one request per bounded batch.

    S3/MinIO uses DeleteObjects (<=1000 keys per request). A key that is
    already absent is a success for that API, so no per-object existence probe
    is issued — the single-object path's ``exists()`` HEAD is pure request
    overhead, and a worker pass removes thousands of objects.

    A failing chunk is logged and treated as done: that is the same outcome the
    single-object helper produced by swallowing its own error, so the caller's
    reclaimed marking is unchanged. DeleteObjects cannot attribute failures per
    key (s3fs issues it with ``Quiet: True`` and drops the error entries), so
    failures are reported at chunk granularity; the S3 layer's
    ``weather_storage_operation_failures_total{operation="delete"}`` counter
    records each failed request.

    Returns the number of keys submitted for deletion.
    """
    if not physical_keys:
        return 0

    if store_path.startswith("s3://"):
        fs = get_control_s3_fs(settings)
        clean_store = store_path[len("s3://") :].rstrip("/")
        raw_paths = [f"{clean_store}/{key.lstrip('/')}" for key in physical_keys]
        for start in range(0, len(raw_paths), _S3_BULK_DELETE_MAX_KEYS):
            chunk = raw_paths[start : start + _S3_BULK_DELETE_MAX_KEYS]
            try:
                fs.rm(chunk)
            except Exception as exc:
                logger.warning(
                    "S3 batch rm of %d objects under %s failed (ignoring error, "
                    "matching the single-object path): %s",
                    len(chunk),
                    store_path,
                    exc,
                )
        return len(raw_paths)

    base = store_path[len("file://") :] if store_path.startswith("file://") else store_path
    for physical_key in physical_keys:
        full_path = os.path.join(base, physical_key)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
        except OSError:
            pass
    return len(physical_keys)


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


def _apply_generation_evidence_gate(
    targets: list[ReclamationQueueRecord],
    store_path: str,
    now_utc: datetime,
) -> set[str]:
    """I14 mechanical replacement-evidence gate for one store's claimed batch.

    A committed-manifest generation change since the last observation is
    physical evidence that the store was rewritten (every EXCLUSIVE finalizer
    commit bumps the generation, including same-set same-cycle replacements),
    which invalidates the enqueue-time necessity judgement. Protocol:

    * no baseline on the row (legacy / pre-migration) -> backfill the
      baseline from the current read and proceed (first observation);
    * baseline == current -> proceed (no replacement since observation);
    * baseline set, current missing -> defer, keep the baseline (evidence
      disappeared; fail closed);
    * baseline != current -> re-baseline and defer (fail closed, but not
      sticky: a store that is not replaced again deletes on the next pass,
      while a store under repeated replacement never deletes).

    Must be called under the store's SHARED gate so the manifest read is
    consistent with the physical deletes that follow (writers need the
    EXCLUSIVE gate to commit a new generation).

    Returns the set of deferred target ids (reverted to QUEUED).
    """
    current = read_store_generation(store_path)
    deferred: set[str] = set()
    for t in targets:
        if t.store_generation is None:
            # No baseline yet (pre-migration row): start observing now.
            t.store_generation = current
            t.updated_at = now_utc
            continue
        if current is None:
            logger.warning(
                "generation_gate_defer: store manifest unavailable for %s "
                "(previously observed generation); target %s deferred",
                store_path,
                t.id,
            )
            t.last_error = "generation_gate_manifest_unavailable"
            t.status = RECLAMATION_STATUS_QUEUED
            t.lease_expires_at = None
            t.updated_at = now_utc
            deferred.add(t.id)
            continue
        if t.store_generation != current:
            logger.info(
                "generation_gate_defer: store %s replaced since observation "
                "(observed=%s current=%s); target %s deferred for re-evaluation",
                store_path,
                t.store_generation,
                current,
                t.id,
            )
            t.store_generation = current
            t.last_error = "generation_gate_replaced_since_observation"
            t.status = RECLAMATION_STATUS_QUEUED
            t.lease_expires_at = None
            t.updated_at = now_utc
            deferred.add(t.id)
    return deferred


def run_reclamation_worker_pass(
    session: Session,
    *,
    batch_size: int | None = None,
    lease_seconds: float | None = None,
    delete_enabled: bool | None = None,
    now: datetime | None = None,
    version_string: str = "v1.0",
) -> ReclamationWorkerResult:
    """Execute a single worker reclamation pass.

    Step 1: Claim bounded batch using SELECT FOR UPDATE SKIP LOCKED (including expired leases).
    Step 2: Group claimed targets by store_path.
    Step 3: For each store, acquire SHARED store gate.
    Step 4: Under store gate, re-read whole-cycle lifecycle: abort if whole-cycle GC claimed cycle.
    Step 4.5: I14 replacement-evidence gate: defer (re-baseline + revert to
    'queued') any target whose store's committed-manifest generation changed
    since the recorded baseline — physical evidence of replacement.
    Step 5: Perform Counterfactual Semantic Revalidation (claimed batch evaluated as physically present).
    Step 6: If target required, revert to 'queued' and clear lease.
    Step 7: If DeleteObject authorized (delete_enabled=True), remove the store's
    authorized shards in bounded DeleteObjects batches and mark them 'deleted'.
    Step 8: Check and delete region markers dynamically derived from catalog commit records.
    """
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    b_size = batch_size if batch_size is not None else settings.RECLAMATION_BATCH_SIZE
    l_secs = lease_seconds if lease_seconds is not None else settings.RECLAMATION_LEASE_SECONDS
    del_en = delete_enabled if delete_enabled is not None else settings.RECLAMATION_DELETE_ENABLED
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
        # Advisory lock coordinator if postgres. The gate is held on a DEDICATED
        # connection: the session's own connection may already carry an open
        # transaction (autobegin) and cannot host the advisory-lock transaction.
        coord: StoreLockCoordinator | None = None
        gate_conn = None
        catalog_bind = session.bind if isinstance(session.bind, Engine) else None
        if is_postgres and catalog_bind is not None:
            gate_conn = catalog_bind.connect()
            coord = StoreLockCoordinator(
                gate_conn,
                store_path=store_path,
                timeout_seconds=5.0,
            )
            try:
                coord.acquire_shared_gate()
            except LockTimeoutError:
                # Store gate blocked (e.g. writer holding exclusive gate)
                logger.warning("Reclamation worker store gate blocked on %s, skipping batch", store_path)
                gate_conn.close()
                for t in targets:
                    t.status = RECLAMATION_STATUS_QUEUED
                    t.lease_expires_at = None
                    t.last_error = "store_gate_timeout"
                session.commit()
                continue

        try:
            # Re-read whole-cycle lifecycle under the gate.
            # A claimed cycle (deletion_started_at) is the serving & mutation
            # fence: the resolver no longer selects it, so granular reclamation
            # of its remaining units proceeds — the V3 bookkeeping pass never
            # deletes store prefixes, so there is nothing to coordinate against.
            first_target = targets[0]
            m_id = first_target.model_id
            c_utc = _ensure_utc_datetime(first_target.cycle_time)
            lc = session.get(ForecastCycleLifecycleRecord, (m_id, c_utc))
            if lc is not None and lc.deleted_at is not None:
                # Tombstoned: all queue rows are terminal by construction —
                # nothing left to reclaim for this cycle.
                logger.info(
                    "Cycle %s %s already tombstoned, skipping granular GC", m_id, c_utc
                )
                for t in targets:
                    t.status = RECLAMATION_STATUS_QUEUED
                    t.lease_expires_at = None
                    t.last_error = "cycle_already_tombstoned"
                session.commit()
                continue

            # 2.5 I14 mechanical replacement-evidence gate: a committed-manifest
            # generation change since the last observation proves the store was
            # replaced, invalidating the enqueue-time necessity judgement.
            # Deferred targets revert to QUEUED (re-baselined) and are excluded
            # from this pass's revalidation and deletion.
            generation_deferred = _apply_generation_evidence_gate(
                targets, store_path, now_utc
            )
            if generation_deferred:
                targets = [t for t in targets if t.id not in generation_deferred]
                total_revalidated += len(generation_deferred)
                session.commit()
                if not targets:
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
            # Streamed straight into the projection: the fence never materialises
            # the row set, and the index is sized by distinct fence units rather
            # than by rows in the queue. The cursor is fully consumed inside
            # build_fence_index, before this store's next write.
            fence_index = build_fence_index(
                (str(r), int(ld), str(v), str(k), int(m))
                for r, ld, v, k, m in session.execute(fenced_query).yield_per(
                    FENCE_FETCH_SIZE
                )
            )

            # Discover candidates for this model
            serving_start = model_serving_start_valid_time(m_id, now_utc)
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

            # Filter candidates using the counterfactual fence (fence_index)
            filtered_cands = filter_candidates_by_fence_index(
                cands_by_vt, fence_index, is_ensemble=is_ensemble, expected_members=expected_members
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

            # Predecessors (per-variable (W, R) metadata, architecture doc I19)
            for r_id_str, (c_utc_r, st_r, p_r) in eligible_runs.items():
                if c_utc_r >= min_recovery_cycle and st_r != "ready":
                    committed_leads = {ld for (r, ld) in by_rl.keys() if r == r_id_str}
                    for p_var in PREDECESSOR_VARIABLES:
                        var_meta = get_variable_temporal_metadata(p_var)
                        reset_period = var_meta.reset_period_hours
                        if var_meta.interval_width_hours >= reset_period:
                            # W == R: no predecessor dependency exists.
                            continue
                        for r_lead in range(reset_period, m_max_lead + 1, reset_period):
                            if r_lead in committed_leads:
                                continue
                            p_lead = get_predecessor_lead(
                                r_lead,
                                interval_width_hours=var_meta.interval_width_hours,
                                reset_period_hours=reset_period,
                            )
                            if p_lead in committed_leads:
                                if is_ensemble:
                                    necessary_tuples[(r_id_str, p_lead, p_var, TARGET_KIND_MEAN, -1)] = "predecessor"
                                    for m in range(1, expected_members + 1):
                                        necessary_tuples[(r_id_str, p_lead, p_var, TARGET_KIND_MEM, m)] = "predecessor"
                                else:
                                    necessary_tuples[(r_id_str, p_lead, p_var, TARGET_KIND_DET, 0)] = "predecessor"

            # 4a. Wind U/V semantic atomicity (architecture doc I8):
            # eligibility atomic — the pair must be jointly judged unnecessary
            # before either component may enter deletion. Serving atomicity is
            # provided by the physical fence (a fenced component makes the pair
            # unselectable for wind_10m); physical DeleteObjects may proceed
            # sequentially and crash-induced one-deleted states remain fenced.
            deferred_pair_ids: set[str] = set()
            pair_groups: dict[
                tuple[str, int, str, int], dict[str, ReclamationQueueRecord]
            ] = {}
            for t in targets:
                if t.variable_code in ("wind_u_10m", "wind_v_10m"):
                    pair_groups.setdefault(
                        (t.run_id, t.lead_time_hours, t.target_kind, t.member_index),
                        {},
                    )[t.variable_code] = t
            for (r_id, lead, kind, mem), comps in pair_groups.items():
                # Any claimed component counterfactually necessary → revert both.
                if any(
                    (t.run_id, t.lead_time_hours, t.variable_code, t.target_kind, t.member_index)
                    in necessary_tuples
                    for t in comps.values()
                ):
                    deferred_pair_ids.update(t.id for t in comps.values())
                    continue
                counterpart_var = {
                    "wind_u_10m": "wind_v_10m",
                    "wind_v_10m": "wind_u_10m",
                }
                for var_code, target in comps.items():
                    other_var = counterpart_var[var_code]
                    if other_var in comps:
                        continue  # both in batch: evaluated jointly above
                    other_status = session.execute(
                        select(ReclamationQueueRecord.status).where(
                            ReclamationQueueRecord.run_id == r_id,
                            ReclamationQueueRecord.lead_time_hours == lead,
                            ReclamationQueueRecord.variable_code == other_var,
                            ReclamationQueueRecord.target_kind == kind,
                            ReclamationQueueRecord.member_index == mem,
                        )
                    ).scalar_one_or_none()
                    if other_status != RECLAMATION_STATUS_DELETED:
                        # Counterpart not terminal → pair not jointly evaluated yet.
                        deferred_pair_ids.add(target.id)

            # 4. Authorize every target in the store's claimed batch. Each
            # eligibility decision below (wind-pair atomicity, counterfactual
            # fence, delete authorization) is still evaluated per target, in
            # the same order as before; only the physical removal is deferred
            # to the batched pass that follows.
            deleted_targets_for_store: list[ReclamationQueueRecord] = []
            authorized_targets: list[ReclamationQueueRecord] = []
            for target in targets:
                if target.id in deferred_pair_ids:
                    target.status = RECLAMATION_STATUS_QUEUED
                    target.lease_expires_at = None
                    target.last_error = "wind_pair_eligibility_atomicity_deferred"
                    target.updated_at = now_utc
                    total_revalidated += 1
                    continue

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

                authorized_targets.append(target)

            # 5. Batched physical removal. Every target in this group shares the
            # group's store_path, so the whole authorized set is removed with
            # one DeleteObjects request per <=1000 keys instead of one
            # exists() HEAD + one single-key request per object.
            _delete_physical_objects_batch(
                store_path, [t.physical_key for t in authorized_targets]
            )

            for target in authorized_targets:
                target.status = RECLAMATION_STATUS_DELETED
                target.reclaimed_at = now_utc
                target.lease_expires_at = None
                target.last_error = None
                target.updated_at = now_utc
                deleted_targets_for_store.append(target)
                total_deleted += 1

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
            if gate_conn is not None:
                gate_conn.close()

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
    now: datetime | None = None,
) -> int:
    """Operator CLI helper: recover stuck non-terminal reclamation targets.

    Covers every non-terminal queue status — not only ``failed`` quarantine:

    - ``failed``: quarantined after retry exhaustion.
    - ``deleting`` with an EXPIRED lease: a worker pass crashed or stalled
      mid-deletion; without this path the stale claim blocks cycle terminality
      (tombstone) forever.
    - ``queued``: legacy V2-finalizer-era rows whose physical prefix was
      already removed externally. A queued/deleting row over a missing object
      otherwise blocks the tombstone permanently
      (``NON_TERMINAL_QUEUE_STATUSES`` includes queued and deleting).

    Safe Requeue Semantics (per row):
    - Rows holding an ACTIVE lease (``lease_expires_at > now``) are skipped:
      another worker may be mid-flight on that claim.
    - If the physical object is confirmed already absent on storage (prior delete
      succeeded remotely before network drop): promote immediately to 'deleted' with
      reclaimed_at timestamp! Prevents accidental un-fencing or serving of missing
      files and unblocks cycle terminality.
    - If the physical object is positively confirmed still present on storage:
      reset attempt_count to 0 and requeue for worker retry.

    Returns the number of rows actually recovered (skipped lease-active rows are
    not counted).
    """
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    stmt = (
        select(ReclamationQueueRecord)
        .where(
            ReclamationQueueRecord.status.in_(
                [
                    RECLAMATION_STATUS_QUEUED,
                    RECLAMATION_STATUS_DELETING,
                    RECLAMATION_STATUS_FAILED,
                ]
            )
        )
        .where(
            or_(
                ReclamationQueueRecord.lease_expires_at.is_(None),
                ReclamationQueueRecord.lease_expires_at <= now_utc,
            )
        )
    )
    if model_id is not None:
        stmt = stmt.where(ReclamationQueueRecord.model_id == model_id.lower().strip())
    if run_id is not None:
        stmt = stmt.where(ReclamationQueueRecord.run_id == run_id)

    rows = list(session.execute(stmt).scalars().all())
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


@dataclass(frozen=True)
class QueuePurgeResult:
    """Outcome of one terminal-row purge pass.

    Attributes:
        deleted_rows: Total rows removed across all batches.
        batches: Number of batches executed (0 when nothing was eligible).
        oldest_remaining: ``reclaimed_at`` of the oldest terminal row still in
            the queue afterwards, or ``None`` when none remain.
    """

    deleted_rows: int
    batches: int
    oldest_remaining: datetime | None


def purge_reclaimed_queue_rows(
    session: Session,
    *,
    older_than_days: float,
    batch_size: int | None = None,
    max_batches: int | None = None,
    now: datetime | None = None,
) -> QueuePurgeResult:
    """Delete terminal reclamation rows older than the retention window.

    The queue is append-mostly: the planner enqueues roughly 1.75x faster than
    the worker deletes, and terminal rows were previously removed only as a side
    effect of the per-cycle metadata sweeper. The table therefore grows without
    bound (382k ``deleted`` rows / 424 MB / 63k dead tuples observed in
    production), and that backlog is exactly what the sweeper's own VACUUM and
    the workers' claim scans have to fight.

    A ``deleted`` row whose ``reclaimed_at`` is older than the retention window
    is pure audit residue: the physical object is gone and the cycle's catalog
    metadata is either still needed or handled by the metadata sweeper. Deleting
    it in bounded batches keeps every transaction short (the caller sees many
    small commits rather than one long row-locking DELETE), and the resulting
    dead tuples stay within what autovacuum can absorb.

    Rows in ``failed`` are deliberately left alone: they are operator-visible
    quarantine evidence with their own requeue path
    (:func:`requeue_failed_reclamation_targets`), and ``queued``/``deleting``
    rows are live work.

    Args:
        session: Catalog session (the caller owns its lifecycle).
        older_than_days: Retention window for terminal rows. Must be >= 0.
        batch_size: Rows per batch. Defaults to
            ``settings.RECLAMATION_PURGE_BATCH_SIZE``.
        max_batches: Optional cap on batches (``None`` drains every eligible
            row). Must be >= 1 when provided.
        now: Optional injected current UTC time.

    Returns:
        The :class:`QueuePurgeResult`.

    Raises:
        ValueError: If ``older_than_days`` is negative or ``max_batches`` < 1.
    """
    from sqlalchemy import delete, func

    if older_than_days < 0:
        raise ValueError(
            f"older_than_days must be >= 0, got {older_than_days}"
        )
    if max_batches is not None and max_batches < 1:
        raise ValueError(f"max_batches must be >= 1, got {max_batches}")
    b_size = (
        int(batch_size)
        if batch_size is not None
        else int(settings.RECLAMATION_PURGE_BATCH_SIZE)
    )
    if b_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {b_size}")

    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    cutoff = now_utc - timedelta(days=float(older_than_days))

    deleted_rows = 0
    batches = 0
    while max_batches is None or batches < max_batches:
        ids = list(
            session.execute(
                select(ReclamationQueueRecord.id)
                .where(
                    ReclamationQueueRecord.status == RECLAMATION_STATUS_DELETED,
                    ReclamationQueueRecord.reclaimed_at.is_not(None),
                    ReclamationQueueRecord.reclaimed_at < cutoff,
                )
                .order_by(ReclamationQueueRecord.reclaimed_at.asc())
                .limit(b_size)
            ).scalars()
        )
        if not ids:
            break
        session.execute(
            delete(ReclamationQueueRecord).where(
                ReclamationQueueRecord.id.in_(ids)
            )
        )
        session.commit()
        deleted_rows += len(ids)
        batches += 1
        logger.info(
            "reclamation queue purge: removed %d terminal rows (%d batch(es), "
            "cutoff=%s)",
            deleted_rows,
            batches,
            cutoff.isoformat(),
        )

    oldest_remaining = session.execute(
        select(func.min(ReclamationQueueRecord.reclaimed_at)).where(
            ReclamationQueueRecord.status == RECLAMATION_STATUS_DELETED
        )
    ).scalar()
    return QueuePurgeResult(
        deleted_rows=deleted_rows,
        batches=batches,
        oldest_remaining=(
            _ensure_utc_datetime(oldest_remaining)
            if oldest_remaining is not None
            else None
        ),
    )
