"""Bounded bulk reclamation planner for Data Lifecycle V3 (converged model).

Canonical source resolution is the **sole deletion authority**: a physical
reclamation unit is reclaimable iff canonical serving no longer selects it for
any protected valid_time and no dependency hold (interval fallback, companion
coupling, wind U/V atomicity, predecessor/recovery, member integrity) applies.
The cycle cutoffs (``T - C`` / ``T - 2C`` / horizon) are only derived
consequences used here as batch-pruning fast-paths — never as a parallel
deletion authority.

Identifies physically reclaimable variable shards across model runs based on
shared canonical reachability, serving-fallback holds, and ingestion-predecessor
holds. Enqueues reclaimable shards into PostgreSQL ``reclamation_queue``
idempotently.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from domain.canonical import (
    CanonicalCandidate,
    select_canonical_sources_bulk,
)
from domain.coverage import get_expected_members, is_lead_servable
from domain.horizon import is_canonical_lead_horizon_registered, model_max_lead_hours
from domain.reclamation import (
    PhysicalShardTarget,
    PREDECESSOR_VARIABLES,
    TARGET_KIND_DET,
    TARGET_KIND_MEAN,
    TARGET_KIND_MEM,
    get_predecessor_lead,
    make_shard_relative_key,
)
from domain.temporal import (
    get_variable_temporal_metadata,
    is_precipitation_companion,
    is_valid_time_protected,
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
from ingestion.core.markers import read_store_generation

logger = logging.getLogger(__name__)

SERVING_ELIGIBLE_STATUSES: tuple[str, ...] = ("ready", "processing", "partial")


@dataclass(frozen=True)
class ReclamationPlanResult:
    """Diagnostic result of a reclamation planner pass."""

    dry_run: bool
    evaluated_at: datetime
    total_committed_shards: int
    active_held_shards: int
    reclaimable_shards: int
    enqueued_count: int
    would_enqueue: tuple[PhysicalShardTarget, ...]
    #: Models this pass deliberately did not plan, because their canonical
    #: horizon is unregistered and their serving boundary is therefore
    #: underivable. Reported rather than raised so one unregistered model cannot
    #: stop reclamation for the rest.
    skipped_models: tuple[str, ...] = ()


def enumerate_committed_unit_tuples(
    session: Session,
    *,
    run_ids: Sequence[str],
    is_ensemble: bool,
) -> set[tuple[str, int, str, str, int]]:
    """Enumerate the committed reclamation unit tuples for the given runs.

    Uses exactly the same shard construction as the planner pass (one unit per
    deterministic/mean product row, one unit per ensemble member product), so
    the lifecycle bookkeeping pass can verify terminality 1:1 against
    ``reclamation_queue`` rows.
    """
    if not run_ids:
        return set()

    emp_members: dict[tuple[str, int], list[int]] = {}
    if is_ensemble:
        emp_stmt = (
            select(
                EnsembleMemberProductRecord.run_id,
                EnsembleMemberProductRecord.lead_time_hours,
                EnsembleMemberProductRecord.member_index,
            ).where(EnsembleMemberProductRecord.run_id.in_(list(run_ids)))
        )
        for r_id, lead, mem_idx in session.execute(emp_stmt).all():
            emp_members.setdefault((str(r_id), int(lead)), []).append(int(mem_idx))

    prod_stmt = select(
        ProductRecord.run_id,
        ProductRecord.lead_time_hours,
        ProductRecord.variable_id,
        ProductRecord.product_type,
    ).where(ProductRecord.run_id.in_(list(run_ids)))

    units: set[tuple[str, int, str, str, int]] = set()
    for r_id, lead, var, p_type in session.execute(prod_stmt).all():
        r_str = str(r_id)
        lead_num = int(lead)
        v_code = str(var)
        if is_ensemble:
            if str(p_type) == "ensemble_mean":
                units.add((r_str, lead_num, v_code, TARGET_KIND_MEAN, -1))
            for mem in emp_members.get((r_str, lead_num), []):
                units.add((r_str, lead_num, v_code, TARGET_KIND_MEM, int(mem)))
        else:
            units.add((r_str, lead_num, v_code, TARGET_KIND_DET, 0))
    return units


def _is_cycle_tombstoned(
    lc_map: dict[tuple[str, datetime], ForecastCycleLifecycleRecord],
    model_id: str,
    cycle_time: datetime,
) -> bool:
    """Return True if the cycle tombstone (deleted_at) is committed."""
    lc = lc_map.get((model_id.lower().strip(), cycle_time))
    return lc is not None and lc.deleted_at is not None


def _is_cycle_claimed(
    lc_map: dict[tuple[str, datetime], ForecastCycleLifecycleRecord],
    model_id: str,
    cycle_time: datetime,
) -> bool:
    """Return True if the cycle is claimed (deletion_started_at serving fence)."""
    lc = lc_map.get((model_id.lower().strip(), cycle_time))
    return lc is not None and lc.deletion_started_at is not None


def plan_reclamation_pass(
    session: Session,
    *,
    models: Sequence[str] = ("gfs", "gefs"),
    dry_run: bool = True,
    now: datetime | None = None,
    version_string: str = "v1.0",
    batch_size: int = 1000,
) -> ReclamationPlanResult:
    """Execute a bounded bulk reclamation planning pass across specified models.

    Invariants:
    1. Bounded SQL queries independent of shard count (no N+1 per shard).
    2. Zero S3 ListObjects calls (the only object reads are bounded fail-open
       committed-manifest generation reads, one per distinct store at enqueue).
    3. Respects three truth layers: preserves catalog commit evidence, uses shared
       canonical reachability for serving truth, and writes physical availability truth.
    4. Safe against newer partial cycles: older cycles continue serving farther valid times.
    5. Retains active serving fallbacks (precipitation_amount_3h, cloud_cover_3h, companions).
    6. Retains active ingestion predecessors (L - W for uncommitted reset leads L % R == 0).
    7. Claimed cycles (deletion_started_at serving fence) contribute no canonical
       candidates and have ALL remaining units directly reclaimable.
    """
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()

    all_would_enqueue: list[PhysicalShardTarget] = []
    total_committed = 0
    total_held = 0
    total_reclaimable = 0
    total_enqueued = 0
    skipped_models: list[str] = []

    # 1. Pre-query lifecycle records for whole-cycle fences
    lc_rows = session.execute(select(ForecastCycleLifecycleRecord)).scalars().all()
    lc_map: dict[tuple[str, datetime], ForecastCycleLifecycleRecord] = {
        (row.model_id.lower().strip(), _ensure_utc_datetime(row.cycle_time)): row
        for row in lc_rows
    }

    for model in models:
        m_id = model.lower().strip()
        # A model whose canonical horizon is unregistered cannot have its serving
        # boundary derived: model_serving_start_valid_time, model_max_lead_hours
        # and is_valid_time_protected all resolve the horizon without a default
        # and raise. Those calls sit inside this loop, so letting the error
        # escape aborts planning for EVERY model rather than just this one —
        # reclamation would stop platform-wide, and only as a log line. Skip the
        # model explicitly instead: the gap is still reported, its blast radius
        # is not.
        if not is_canonical_lead_horizon_registered(m_id):
            logger.error(
                "reclamation planning skipped model=%s: no canonical lead horizon "
                "is registered for it, so its serving boundary cannot be derived. "
                "Register the model in domain.horizon before it can be reclaimed; "
                "all other models are unaffected.",
                m_id,
            )
            skipped_models.append(m_id)
            continue
        expected_members = get_expected_members(m_id, default_if_unknown=1)
        is_ensemble = expected_members > 1
        # Per-model serving boundary (I20): derived from this model's
        # registered canonical horizon cadence — never a caller-chosen
        # global default that could diverge from the per-model
        # is_valid_time_protected membership test below.
        serving_start = model_serving_start_valid_time(m_id, now_utc)

        # 2. Bulk query 1: eligible runs for model
        runs_stmt = (
            select(
                ModelRunRecord.id,
                ModelRunRecord.cycle_time,
                ModelRunRecord.status,
                ModelRunRecord.zarr_store_path,
                ModelVersionRecord.version_string,
            )
            .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
            .where(
                ModelVersionRecord.model_id == m_id,
                ModelRunRecord.status.in_(SERVING_ELIGIBLE_STATUSES),
                ModelRunRecord.zarr_store_path.isnot(None),
            )
        )
        run_rows = session.execute(runs_stmt).all()
        if not run_rows:
            continue

        runs_by_id: dict[str, dict[str, Any]] = {}
        retired_run_ids: list[str] = []
        for r_id, c_time, status, store_path, ver_str in run_rows:
            c_utc = _ensure_utc_datetime(c_time)
            if _is_cycle_tombstoned(lc_map, m_id, c_utc):
                # Tombstoned: all units terminal by construction — nothing to plan.
                continue
            if _is_cycle_claimed(lc_map, m_id, c_utc):
                # Claimed = serving & mutation fence: the resolver no longer
                # selects this cycle, so NONE of its units can be protected.
                # They are directly reclaimable (architecture doc §7.1/§7.5).
                retired_run_ids.append(str(r_id))
                continue
            runs_by_id[str(r_id)] = {
                "cycle_time": c_utc,
                "status": str(status),
                "store_path": str(store_path),
                "version_string": str(ver_str),
            }

        planable_run_ids = list(runs_by_id.keys())
        catalog_run_ids = planable_run_ids + retired_run_ids
        if not catalog_run_ids:
            continue

        # 3. Bulk query 2: forecast_products for all cataloged runs
        prod_stmt = select(
            ProductRecord.run_id,
            ProductRecord.lead_time_hours,
            ProductRecord.variable_id,
            ProductRecord.product_type,
        ).where(ProductRecord.run_id.in_(catalog_run_ids))
        prod_rows = session.execute(prod_stmt).all()

        # 4. Bulk query 3: ensemble_member_products if ensemble
        emp_members: dict[tuple[str, int], list[int]] = {}
        if is_ensemble:
            emp_stmt = select(
                EnsembleMemberProductRecord.run_id,
                EnsembleMemberProductRecord.lead_time_hours,
                EnsembleMemberProductRecord.member_index,
            ).where(EnsembleMemberProductRecord.run_id.in_(catalog_run_ids))
            for r_id, lead, mem_idx in session.execute(emp_stmt).all():
                emp_members.setdefault((str(r_id), int(lead)), []).append(int(mem_idx))

        # 5. Bulk query 4: already queued/fenced/deleted targets from reclamation_queue
        queue_stmt = select(
            ReclamationQueueRecord.run_id,
            ReclamationQueueRecord.lead_time_hours,
            ReclamationQueueRecord.variable_code,
            ReclamationQueueRecord.target_kind,
            ReclamationQueueRecord.member_index,
            ReclamationQueueRecord.status,
        ).where(ReclamationQueueRecord.run_id.in_(catalog_run_ids))
        queue_rows = session.execute(queue_stmt).all()
        existing_queue_targets = {
            (str(r_id), int(lead), str(var), str(kind), int(mem)): str(status)
            for r_id, lead, var, kind, mem, status in queue_rows
        }

        # Build candidate structures for canonical reachability evaluation
        by_run_lead: dict[tuple[str, int], dict[str, Any]] = {}
        all_shards_for_model: list[PhysicalShardTarget] = []

        for r_id, lead, var_id, prod_type in prod_rows:
            r_str = str(r_id)
            if r_str in retired_run_ids:
                continue  # retired units handled directly below
            lead_num = int(lead)
            v_code = str(var_id)
            p_type = str(prod_type)
            run_meta = runs_by_id[r_str]
            c_utc = run_meta["cycle_time"]
            store_path = run_meta["store_path"]
            v_time = c_utc + timedelta(hours=lead_num)

            rec = by_run_lead.setdefault(
                (r_str, lead_num),
                {
                    "run_id": r_str,
                    "cycle_time": c_utc,
                    "lead_time_hours": lead_num,
                    "store_path": store_path,
                    "status": run_meta["status"],
                    "product_types": set(),
                    "variables": set(),
                },
            )
            rec["product_types"].add(p_type)
            rec["variables"].add(v_code)

            # Build physical shard targets
            if is_ensemble:
                if p_type == "ensemble_mean":
                    all_shards_for_model.append(
                        PhysicalShardTarget(
                            run_id=r_str,
                            model_id=m_id,
                            cycle_time=c_utc,
                            lead_time_hours=lead_num,
                            variable_code=v_code,
                            target_kind=TARGET_KIND_MEAN,
                            member_index=-1,
                            valid_time=v_time,
                            store_path=store_path,
                            physical_key=make_shard_relative_key(v_code, TARGET_KIND_MEAN, lead_num, -1),
                        )
                    )
                # Perturbation members for this variable and lead
                mems = emp_members.get((r_str, lead_num), [])
                for mem in mems:
                    all_shards_for_model.append(
                        PhysicalShardTarget(
                            run_id=r_str,
                            model_id=m_id,
                            cycle_time=c_utc,
                            lead_time_hours=lead_num,
                            variable_code=v_code,
                            target_kind=TARGET_KIND_MEM,
                            member_index=mem,
                            valid_time=v_time,
                            store_path=store_path,
                            physical_key=make_shard_relative_key(v_code, TARGET_KIND_MEM, lead_num, mem),
                        )
                    )
            else:
                all_shards_for_model.append(
                    PhysicalShardTarget(
                        run_id=r_str,
                        model_id=m_id,
                        cycle_time=c_utc,
                        lead_time_hours=lead_num,
                        variable_code=v_code,
                        target_kind=TARGET_KIND_DET,
                        member_index=0,
                        valid_time=v_time,
                        store_path=store_path,
                        physical_key=make_shard_relative_key(v_code, TARGET_KIND_DET, lead_num, 0),
                    )
                )

        # Build candidates by valid time
        candidates_by_valid: dict[datetime, list[CanonicalCandidate]] = {}
        for (r_str, lead_num), rec in by_run_lead.items():
            c_utc = rec["cycle_time"]
            v_time = c_utc + timedelta(hours=lead_num)
            m_indices: tuple[int, ...] | None = None
            if is_ensemble:
                raw_mems = emp_members.get((r_str, lead_num), [])
                if not is_lead_servable(len(raw_mems), expected_members):
                    continue
                if not rec["product_types"]:
                    continue
                m_indices = tuple(sorted(raw_mems))

            cand = CanonicalCandidate(
                cycle_time=c_utc,
                lead_time_hours=lead_num,
                run_id=r_str,
                store_path=rec["store_path"],
                product_types=frozenset(rec["product_types"]),
                variables=frozenset(rec["variables"]),
                member_indices=m_indices,
                status=rec["status"],
            )
            candidates_by_valid.setdefault(v_time, []).append(cand)

        for vt in candidates_by_valid:
            candidates_by_valid[vt].sort(key=lambda c: c.lead_time_hours)

        # Collect distinct variables present in this model's candidates
        distinct_vars = sorted({s.variable_code for s in all_shards_for_model})
        if "wind_u_10m" in distinct_vars and "wind_v_10m" in distinct_vars:
            distinct_vars.append("wind_10m")

        # Active canonical held targets:
        held_target_tuples: set[tuple[str, int, str, str, int]] = set()

        # 6. Ingestion-predecessor holds (evaluated BEFORE canonical resolution so
        #    the window-exit fast-path below can run: predecessor shards may carry
        #    valid times that have already exited the active serving window while
        #    still being required to reconstruct uncommitted reset leads).
        # For incomplete recovery-eligible cycles (cycle_time >= min_recovery_cycle),
        # if reset lead L (L > 0, L % R == 0) is uncommitted, hold predecessor L - W
        # for predecessor variables (precipitation_amount_3h, cloud_cover_3h).
        committed_leads_by_run: dict[str, set[int]] = {}
        for (r_str, lead_num) in by_run_lead.keys():
            committed_leads_by_run.setdefault(r_str, set()).add(lead_num)

        for r_id, meta in runs_by_id.items():
            run_version = meta["version_string"]
            run_max_lead = model_max_lead_hours(m_id, version_string=run_version)
            run_min_recovery = serving_start - timedelta(hours=run_max_lead)
            c_utc = meta["cycle_time"]
            if c_utc < run_min_recovery:
                continue
            # If run is already 'ready' or has no uncommitted reset leads, no predecessor hold needed
            if meta["status"] == "ready":
                continue

            committed_leads = committed_leads_by_run.get(r_id, set())
            # Check all possible reset leads up to run's authoritative max lead,
            # gated per-variable by (W, R) metadata (architecture doc I19).
            for pred_var in PREDECESSOR_VARIABLES:
                var_meta = get_variable_temporal_metadata(pred_var)
                reset_period = var_meta.reset_period_hours
                if var_meta.interval_width_hours >= reset_period:
                    # W == R: the reset-lead sample already covers the interval;
                    # no predecessor dependency exists for this variable.
                    continue
                for reset_lead in range(reset_period, run_max_lead + 1, reset_period):
                    if reset_lead in committed_leads:
                        continue
                    pred_lead = get_predecessor_lead(
                        reset_lead,
                        interval_width_hours=var_meta.interval_width_hours,
                        reset_period_hours=reset_period,
                    )
                    if pred_lead in committed_leads:
                        if is_ensemble:
                            held_target_tuples.add(
                                (r_id, pred_lead, pred_var, TARGET_KIND_MEAN, -1)
                            )
                            for m in range(1, expected_members + 1):
                                held_target_tuples.add(
                                    (r_id, pred_lead, pred_var, TARGET_KIND_MEM, m)
                                )
                        else:
                            held_target_tuples.add(
                                (r_id, pred_lead, pred_var, TARGET_KIND_DET, 0)
                            )

        # 7. Window-exit fast-path (derived cutoff, architecture doc §7.5 class B):
        #    units whose valid_time has exited the active serving window can never
        #    be canonical-selected again (is_valid_time_protected is the shared
        #    boundary primitive); only the predecessor holds above may still
        #    protect them. Canonical resolution below therefore runs exclusively
        #    over protected valid times.
        protected_candidates: dict[datetime, list[CanonicalCandidate]] = {
            vt: cands
            for vt, cands in candidates_by_valid.items()
            if is_valid_time_protected(vt, now_utc, model_id=m_id)
        }

        # 8. Evaluate shared canonical resolution across the active serving window
        canon_res = select_canonical_sources_bulk(
            protected_candidates,
            variables=distinct_vars,
            start_valid_time=serving_start,
        )

        # Anchors hold all their ordinary committed variables (unless lead 0)
        for vt, anchor_cand in canon_res.anchors.items():
            if anchor_cand.lead_time_hours > 0:
                for var_code in anchor_cand.variables:
                    if is_ensemble:
                        # Mean shard
                        held_target_tuples.add(
                            (anchor_cand.run_id, anchor_cand.lead_time_hours, var_code, TARGET_KIND_MEAN, -1)
                        )
                        # Member shards — ALL committed members of the canonical
                        # source are retained: the coverage threshold is a
                        # candidate-eligibility bound, never a retention target.
                        for m in anchor_cand.member_indices or ():
                            held_target_tuples.add(
                                (anchor_cand.run_id, anchor_cand.lead_time_hours, var_code, TARGET_KIND_MEM, m)
                            )
                    else:
                        held_target_tuples.add(
                            (anchor_cand.run_id, anchor_cand.lead_time_hours, var_code, TARGET_KIND_DET, 0)
                        )
            else:
                # Lead 0 anchor holds ordinary variables (non-interval, non-companion)
                for var_code in anchor_cand.variables:
                    if not (
                        requires_lead0_display_fallback(var_code)
                        or is_precipitation_companion(var_code)
                    ):
                        if is_ensemble:
                            held_target_tuples.add(
                                (anchor_cand.run_id, 0, var_code, TARGET_KIND_MEAN, -1)
                            )
                            for m in anchor_cand.member_indices or ():
                                held_target_tuples.add(
                                    (anchor_cand.run_id, 0, var_code, TARGET_KIND_MEM, m)
                                )
                        else:
                            held_target_tuples.add(
                                (anchor_cand.run_id, 0, var_code, TARGET_KIND_DET, 0)
                            )

        # Hold active variable fallbacks and companions
        for (var_code, vt), src_cand in canon_res.variable_sources.items():
            if src_cand is None:
                continue
            if is_ensemble:
                held_target_tuples.add(
                    (src_cand.run_id, src_cand.lead_time_hours, var_code, TARGET_KIND_MEAN, -1)
                )
                for m in src_cand.member_indices or ():
                    held_target_tuples.add(
                        (src_cand.run_id, src_cand.lead_time_hours, var_code, TARGET_KIND_MEM, m)
                    )
            else:
                held_target_tuples.add(
                    (src_cand.run_id, src_cand.lead_time_hours, var_code, TARGET_KIND_DET, 0)
                )

        # Coherent wind components: hold wind_u_10m and wind_v_10m if wind_10m is resolved
        for (var_code, vt), src_cand in canon_res.variable_sources.items():
            if var_code == "wind_10m" and src_cand is not None:
                for w_comp in ("wind_u_10m", "wind_v_10m"):
                    if is_ensemble:
                        held_target_tuples.add(
                            (src_cand.run_id, src_cand.lead_time_hours, w_comp, TARGET_KIND_MEAN, -1)
                        )
                        for m in src_cand.member_indices or ():
                            held_target_tuples.add(
                                (src_cand.run_id, src_cand.lead_time_hours, w_comp, TARGET_KIND_MEM, m)
                            )
                    else:
                        held_target_tuples.add(
                            (src_cand.run_id, src_cand.lead_time_hours, w_comp, TARGET_KIND_DET, 0)
                        )

        # 9. Reconcile all physical shards against held targets
        dedup_shards = {s.target_tuple: s for s in all_shards_for_model}
        for t_tuple, shard_target in dedup_shards.items():
            total_committed += 1
            if t_tuple in held_target_tuples:
                total_held += 1
            else:
                total_reclaimable += 1
                # Check if already in queue
                if t_tuple not in existing_queue_targets:
                    all_would_enqueue.append(shard_target)

        # 10. Claimed (retired) runs: every committed unit is directly reclaimable.
        if retired_run_ids:
            retired_units = enumerate_committed_unit_tuples(
                session, run_ids=retired_run_ids, is_ensemble=is_ensemble
            )
            store_by_run: dict[str, str] = {}
            cycle_by_run: dict[str, datetime] = {}
            for r_id, c_time, z_path in session.execute(
                select(
                    ModelRunRecord.id,
                    ModelRunRecord.cycle_time,
                    ModelRunRecord.zarr_store_path,
                ).where(ModelRunRecord.id.in_(retired_run_ids))
            ).all():
                cycle_by_run[str(r_id)] = _ensure_utc_datetime(c_time)
                if z_path:
                    store_by_run[str(r_id)] = str(z_path)

            for t_tuple in sorted(retired_units):
                total_committed += 1
                if t_tuple in existing_queue_targets:
                    continue
                total_reclaimable += 1
                r_str, lead_num, v_code, kind, mem = t_tuple
                store_path = store_by_run.get(r_str, "")
                c_utc_opt: datetime | None = cycle_by_run.get(r_str)
                if not store_path or c_utc_opt is None:
                    continue
                all_would_enqueue.append(
                    PhysicalShardTarget(
                        run_id=r_str,
                        model_id=m_id,
                        cycle_time=c_utc,
                        lead_time_hours=lead_num,
                        variable_code=v_code,
                        target_kind=kind,
                        member_index=mem,
                        valid_time=c_utc_opt + timedelta(hours=lead_num),
                        store_path=store_path,
                        physical_key=make_shard_relative_key(v_code, kind, lead_num, mem),
                    )
                )

    # 11. Idempotent enqueue if not dry-run
    if not dry_run and all_would_enqueue:
        # I14 replacement-evidence baseline: snapshot each store's
        # committed-manifest generation at enqueue time so the worker can
        # detect a replacement between enqueue and deletion. One fail-open
        # manifest read per distinct store (cached across the pass) — this
        # is a bounded single-object read, not a listing.
        store_generation_cache: dict[str, str | None] = {}

        def _snapshot_generation(store_path: str) -> str | None:
            if store_path not in store_generation_cache:
                store_generation_cache[store_path] = read_store_generation(store_path)
            return store_generation_cache[store_path]

        # Enqueue in bounded batches
        for i in range(0, len(all_would_enqueue), batch_size):
            chunk = all_would_enqueue[i : i + batch_size]
            records_to_insert = [
                {
                    "id": f"rec_{uuid.uuid4().hex[:24]}",
                    "run_id": target.run_id,
                    "model_id": target.model_id,
                    "cycle_time": target.cycle_time,
                    "lead_time_hours": target.lead_time_hours,
                    "variable_code": target.variable_code,
                    "target_kind": target.target_kind,
                    "member_index": target.member_index,
                    "valid_time": target.valid_time,
                    "store_path": target.store_path,
                    "physical_key": target.physical_key,
                    "store_generation": _snapshot_generation(target.store_path),
                    "status": "queued",
                    "attempt_count": 0,
                    "lease_expires_at": None,
                    "next_retry_at": None,
                    "last_error": None,
                    "reclaimed_at": None,
                    "created_at": now_utc,
                    "updated_at": now_utc,
                }
                for target in chunk
            ]
            if session.bind and session.bind.dialect.name == "postgresql":
                stmt = pg_insert(ReclamationQueueRecord).values(records_to_insert)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=[
                        "run_id",
                        "lead_time_hours",
                        "variable_code",
                        "target_kind",
                        "member_index",
                    ]
                )
                session.execute(stmt)
            else:
                # SQLite / test fallback: insert one-by-one ignoring integrity error
                for rec in records_to_insert:
                    try:
                        session.add(ReclamationQueueRecord(**rec))
                        session.flush()
                    except Exception:
                        session.rollback()
            session.commit()
            total_enqueued += len(chunk)

    return ReclamationPlanResult(
        dry_run=dry_run,
        evaluated_at=now_utc,
        total_committed_shards=total_committed,
        active_held_shards=total_held,
        reclaimable_shards=total_reclaimable,
        enqueued_count=total_enqueued if not dry_run else len(all_would_enqueue),
        would_enqueue=tuple(all_would_enqueue),
        skipped_models=tuple(sorted(skipped_models)),
    )
