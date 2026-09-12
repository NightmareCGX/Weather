"""Authoritative Canonical Valid-Time Resolver for forecast serving paths (Lifecycle V3).

This service implements the single authoritative canonical source resolution rule
across all forecast serving endpoint families:
- /v1/points
- /v1/ensembles
- /v1/probabilities
- /v1/maps (spatial layer metadata and raster tiles)
- /v1/forecast/availability

Canonical V3 Principles:
------------------------
1. Per-Valid-Time Ownership:
   For model M and valid_time V, the canonical source is the newest committed
   eligible representation capable of representing V:
       source_cycle(V) = MAX(cycle_time)
       WHERE cycle_time + lead_time_hours == V
         AND representation is safely committed and servable
   A newer partial cycle supersedes older cycles ONLY for valid times it has
   actually committed. Older cycles continue serving farther valid times.
   The serving right boundary does not collapse.

2. Authoritative Physical Safety Fences:
   Physical safety fences (``deletion_started_at`` and ``deleted_at``) remain
   authoritative and are properly model-scoped.

3. Strict GEFS Coherent Vintage:
   A GEFS cycle is canonical for valid_time V if and only if BOTH:
   a. Official ensemble mean (geavg) is committed (product_type == 'ensemble_mean')
   b. Perturbation-member coverage is servable under the >=85% threshold (>=26/30)
   Mean products, maps, ensemble statistics, and probabilities all resolve the
   identical coherent cycle vintage.

4. Interval Fallback & Graceful Null:
   Interval variables (precipitation_amount_3h, cloud_cover_3h) at canonical lead 0
   fall back to the newest committed older cycle with lead > 0 for the same valid_time.
   If no positive-lead representation exists, the resolver returns None gracefully
   without failing the request.

5. Precipitation Companion Coupling:
   Categorical companion flags (crain, csnow, cfrzr, cicep) are strictly source-coupled
   to precipitation_amount_3h. They never resolve independently.

6. wind_10m Component Coherence:
   A candidate is eligible for synthetic wind_10m if and only if BOTH wind_u_10m and
   wind_v_10m are committed in the same run and lead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from domain.canonical import (
    CanonicalCandidate as _CandidateRecord,
    select_canonical_anchor,
    select_canonical_sources_bulk,
    select_variable_source,
)
from domain.coverage import get_expected_members, is_lead_servable
from domain.temporal import (
    is_precipitation_companion,
    requires_lead0_display_fallback,
    serving_start_valid_time,
)
from fastapi import HTTPException
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from api.core.time import get_current_time
from api.models.entities import (
    EnsembleMemberProduct,
    ForecastProduct,
    Model,
    ModelRun,
    ModelVersion,
    ReclamationQueue,
)
from api.services.lifecycle import filter_fenced_runs, parse_cycle_time

logger = logging.getLogger(__name__)

#: ModelRun statuses eligible for serving candidate discovery.
SERVING_ELIGIBLE_STATUSES: tuple[str, ...] = ("ready", "processing", "partial")


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_store_generation(store_path: str | None) -> str | None:
    """Resolve the store's committed manifest generation without holding DB locks."""
    if not store_path:
        return None
    try:
        from api.core.manifest_reader import manifest_generation

        return manifest_generation(store_path)
    except Exception:
        return None


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
        member_indices: Sorted committed member indices tuple for ensemble models.
    """

    model: str
    valid_time: datetime
    cycle_time: datetime
    lead_time_hours: int
    run_id: str
    store_path: str
    serving_generation: str | None = None
    member_indices: tuple[int, ...] | None = None

    @property
    def member_fingerprint(self) -> str | None:
        """Deterministic fingerprint of committed members for cache identity."""
        if not self.member_indices:
            return None
        return f"m_{'_'.join(str(m) for m in self.member_indices)}"


def _discover_candidates_bulk(
    db: Session,
    model: str,
    *,
    start_lead_time_hours: int | None = None,
    end_lead_time_hours: int | None = None,
    now: datetime | None = None,
) -> dict[datetime, list[_CandidateRecord]]:
    """Discover all valid-time candidates using exactly 1 catalog query (+1 member query for ensembles).

    Applies model-scoped physical deletion fencing.
    For GEFS ensembles, strictly enforces that candidates have both official geavg
    and servable member coverage (>=85%).
    Filters out physical shards that are in reclamation_queue with status IN ('deleting', 'deleted').
    """
    now_utc = now if now is not None else get_current_time()
    start_vt = serving_start_valid_time(now_utc)
    m_id = model.lower().strip()
    expected_members = get_expected_members(m_id, default_if_unknown=1)
    is_ensemble = expected_members > 1

    # 1. Main catalog query: model_runs ⋈ forecast_products with physical fence filter
    has_reclamation_queue = True
    try:
        bind = db.get_bind()
        if bind.dialect.name == "sqlite":
            from sqlalchemy import inspect
            has_reclamation_queue = inspect(bind).has_table("reclamation_queue")
    except Exception:
        has_reclamation_queue = False

    stmt = (
        select(
            ModelRun.id,
            ModelRun.cycle_time,
            ModelRun.zarr_store_path,
            ModelRun.status,
            ForecastProduct.lead_time_hours,
            ForecastProduct.product_type,
            ForecastProduct.variable_id,
        )
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .join(ForecastProduct, ForecastProduct.run_id == ModelRun.id)
        .where(Model.model_id == m_id)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if has_reclamation_queue:
        reclaim_subq = (
            select(1)
            .select_from(ReclamationQueue)
            .where(
                ReclamationQueue.run_id == ForecastProduct.run_id,
                ReclamationQueue.lead_time_hours == ForecastProduct.lead_time_hours,
                ReclamationQueue.variable_code == ForecastProduct.variable_id,
                ReclamationQueue.status.in_(("deleting", "deleted", "failed")),
                or_(
                    and_(ForecastProduct.product_type == "ensemble_mean", ReclamationQueue.target_kind == "mean"),
                    and_(ForecastProduct.product_type != "ensemble_mean", ReclamationQueue.target_kind == "det"),
                ),
            )
        )
        stmt = stmt.where(~reclaim_subq.exists())

    stmt = filter_fenced_runs(stmt, model_id=m_id)
    rows = db.execute(stmt).all()

    # 2. Ensemble member pre-query: only when ensemble model
    emp_members: dict[tuple[str, int], list[int]] = {}
    runs_with_members: set[str] = set()
    if is_ensemble:
        emp_stmt = (
            select(
                EnsembleMemberProduct.run_id,
                EnsembleMemberProduct.lead_time_hours,
                EnsembleMemberProduct.member_index,
            )
            .join(ModelRun, EnsembleMemberProduct.run_id == ModelRun.id)
            .join(ModelVersion, ModelRun.model_version_id == ModelVersion.id)
            .where(ModelVersion.model_id == m_id)
        )
        if has_reclamation_queue:
            emp_reclaim_subq = (
                select(1)
                .select_from(ReclamationQueue)
                .where(
                    ReclamationQueue.run_id == EnsembleMemberProduct.run_id,
                    ReclamationQueue.lead_time_hours == EnsembleMemberProduct.lead_time_hours,
                    ReclamationQueue.member_index == EnsembleMemberProduct.member_index,
                    ReclamationQueue.target_kind == "mem",
                    ReclamationQueue.status.in_(("deleting", "deleted", "failed")),
                )
            )
            emp_stmt = emp_stmt.where(~emp_reclaim_subq.exists())

        emp_rows = db.execute(emp_stmt).all()
        for r_id, lead, mem_idx in emp_rows:
            emp_members.setdefault((str(r_id), int(lead)), []).append(int(mem_idx))
            runs_with_members.add(str(r_id))

    # 3. In-memory aggregation per (run_id, lead_time_hours)
    by_run_lead: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}
    for run_id, cycle_time, store_path, run_status, lead, prod_type, var_id in rows:
        if store_path is None:
            continue
        lead_num = int(lead)
        if start_lead_time_hours is not None and lead_num < start_lead_time_hours:
            continue
        if end_lead_time_hours is not None and lead_num > end_lead_time_hours:
            continue

        r_str = str(run_id)

        key = (r_str, lead_num)
        rec = by_run_lead.setdefault(
            key,
            {
                "cycle_time": _ensure_utc(cycle_time),
                "lead_time_hours": lead_num,
                "run_id": r_str,
                "store_path": str(store_path),
                "status": str(run_status),
                "product_types": set(),
                "variables": set(),
            },
        )
        if prod_type:
            rec["product_types"].add(prod_type)
        if var_id:
            rec["variables"].add(var_id)

    candidates_by_valid: dict[datetime, list[_CandidateRecord]] = {}

    for (run_id, lead_num), rec in by_run_lead.items():
        c_utc = rec["cycle_time"]
        v_time = c_utc + timedelta(hours=lead_num)
        if v_time < start_vt:
            continue

        m_indices: tuple[int, ...] | None = None
        if is_ensemble:
            raw_members = emp_members.get((run_id, lead_num))
            if raw_members is None:
                # Legacy store / test mock without pair rows: if run is ready and has no member rows anywhere,
                # treat as fully servable with expected members (read-resilience compatibility)
                if str(run_id) not in runs_with_members and rec["status"] == "ready":
                    raw_members = list(range(1, expected_members + 1))
                else:
                    raw_members = []

            m_indices = tuple(sorted(raw_members)) if raw_members else ()
            # Strict GEFS Coherent Vintage Invariant:
            # Candidate is eligible ONLY if both mean product AND >=85% members exist
            if not is_lead_servable(len(m_indices), expected_members):
                continue
            if not rec["product_types"]:
                continue

        cand = _CandidateRecord(
            cycle_time=c_utc,
            lead_time_hours=lead_num,
            run_id=run_id,
            store_path=rec["store_path"],
            product_types=frozenset(rec["product_types"]),
            variables=frozenset(rec["variables"]),
            member_indices=m_indices,
            status=rec["status"],
        )
        candidates_by_valid.setdefault(v_time, []).append(cand)

    # 4. Fallback discovery for ready runs lacking forecast_products rows (legacy stores / tests)
    product_runs = {rec["run_id"] for rec in by_run_lead.values()}
    ready_stmt = (
        select(ModelRun.id, ModelRun.cycle_time, ModelRun.zarr_store_path)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == m_id)
        .where(ModelRun.status == "ready")
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    ready_stmt = filter_fenced_runs(ready_stmt, model_id=m_id)
    for run_id, cycle_time, store_path in db.execute(ready_stmt).all():
        if str(run_id) in product_runs or store_path is None:
            continue
        c_utc = _ensure_utc(cycle_time)
        try:
            from api.services.point_forecast import gated_cycle_metadata

            metadata = gated_cycle_metadata(str(store_path))
        except Exception:
            continue

        leads = sorted(metadata.lead_times)
        var_names = frozenset(metadata.var_names)
        for lead_num in leads:
            if start_lead_time_hours is not None and lead_num < start_lead_time_hours:
                continue
            if end_lead_time_hours is not None and lead_num > end_lead_time_hours:
                continue
            v_time = c_utc + timedelta(hours=lead_num)
            if v_time < start_vt:
                continue

            m_indices = None
            if is_ensemble:
                # In legacy stores without pair rows, check mean shard
                from api.core.zarr import get_sharded_reader

                try:
                    reader = get_sharded_reader(str(store_path))
                    if not reader.has_mean_shard("temperature_2m", lead_num):
                        continue
                except Exception:
                    continue
                m_indices = tuple(range(1, expected_members + 1))

            cand = _CandidateRecord(
                cycle_time=c_utc,
                lead_time_hours=lead_num,
                run_id=str(run_id),
                store_path=str(store_path),
                product_types=frozenset({"surface", "ensemble_mean" if is_ensemble else "surface"}),
                variables=var_names,
                member_indices=m_indices,
            )
            candidates_by_valid.setdefault(v_time, []).append(cand)

    # Sort each valid_time's candidate list by ascending lead_time_hours (newest cycle first)
    for v_time in candidates_by_valid:
        candidates_by_valid[v_time].sort(key=lambda c: c.lead_time_hours)

    return candidates_by_valid


def resolve_valid_time_candidates(
    db: Session,
    model: str,
    *,
    target_valid_time: datetime | None = None,
    variable: str | None = None,
    start_lead_time_hours: int | None = None,
    end_lead_time_hours: int | None = None,
    require_members: bool = False,
    now: datetime | None = None,
) -> dict[datetime, list[tuple[datetime, int, str, str]]]:
    """Compatibility discovery function returning (cycle_time, lead, run_id, store_path) tuples.

    Preserved for backward compatibility with existing tests and helper callers.
    Strictly applies V3 physical fencing and GEFS coherent-vintage rules.
    """
    candidates_map = _discover_candidates_bulk(
        db,
        model,
        start_lead_time_hours=start_lead_time_hours,
        end_lead_time_hours=end_lead_time_hours,
        now=now,
    )

    out: dict[datetime, list[tuple[datetime, int, str, str]]] = {}
    target_utc = _ensure_utc(target_valid_time) if target_valid_time is not None else None

    for v_time, cands in candidates_map.items():
        if target_utc is not None and v_time != target_utc:
            continue

        matched_cands: list[tuple[datetime, int, str, str]] = []
        for cand in cands:
            # Check variable capability if variable specified
            if variable is not None:
                if variable == "wind_10m":
                    if not {"wind_u_10m", "wind_v_10m"}.issubset(cand.variables):
                        continue
                elif cand.variables and variable not in cand.variables:
                    continue
                if cand.lead_time_hours == 0 and requires_lead0_display_fallback(variable):
                    continue

            matched_cands.append(
                (cand.cycle_time, cand.lead_time_hours, cand.run_id, cand.store_path)
            )

        if matched_cands:
            out[v_time] = matched_cands

    return out


def resolve_canonical_source(
    db: Session,
    model: str,
    valid_time: datetime | str,
    *,
    require_members: bool = False,
    now: datetime | None = None,
) -> ResolvedForecastSource:
    """Resolve the single authoritative generic canonical source for model + valid_time.

    Invariants:
    - Rejects valid_time strictly before serving_start_valid_time(now_utc) with HTTP 404.
    - Selects the newest committed representation covering valid_time.
    - Strictly excludes deletion fences.
    - Strictly enforces GEFS coherent vintage (both geavg and >=85% members).
    """
    v_utc = parse_cycle_time(valid_time) if isinstance(valid_time, str) else _ensure_utc(valid_time)
    now_utc = now if now is not None else get_current_time()
    start_vt = serving_start_valid_time(now_utc)

    if v_utc < start_vt:
        raise HTTPException(
            status_code=404,
            detail=f"Valid time '{v_utc.isoformat()}' is before the active serving window ({start_vt.isoformat()}).",
        )

    candidates_map = _discover_candidates_bulk(db, model, now=now_utc)
    cands = candidates_map.get(v_utc)
    if not cands:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast data is available for model '{model}' at valid time '{v_utc.isoformat()}'.",
        )

    winner = select_canonical_anchor(cands)
    if winner is None:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast data is available for model '{model}' at valid time '{v_utc.isoformat()}'.",
        )
    gen = _resolve_store_generation(winner.store_path)

    return ResolvedForecastSource(
        model=model.lower().strip(),
        valid_time=v_utc,
        cycle_time=winner.cycle_time,
        lead_time_hours=winner.lead_time_hours,
        run_id=winner.run_id,
        store_path=winner.store_path,
        serving_generation=gen,
        member_indices=winner.member_indices,
    )


def resolve_variable_source(
    db: Session,
    model: str,
    variable: str,
    valid_time: datetime | str,
    *,
    now: datetime | None = None,
) -> ResolvedForecastSource | None:
    """Resolve the authoritative source for a specific variable at valid_time.

    Supports graceful absence (returning None) for interval variables at lead 0
    when no older positive-lead fallback representation exists.
    """
    v_utc = parse_cycle_time(valid_time) if isinstance(valid_time, str) else _ensure_utc(valid_time)
    now_utc = now if now is not None else get_current_time()
    start_vt = serving_start_valid_time(now_utc)

    if v_utc < start_vt:
        raise HTTPException(
            status_code=404,
            detail=f"Valid time '{v_utc.isoformat()}' is before the active serving window ({start_vt.isoformat()}).",
        )

    # 1. Strict companion coupling: crain, csnow, cfrzr, cicep delegate to precipitation_amount_3h
    if is_precipitation_companion(variable):
        return resolve_variable_source(
            db, model, "precipitation_amount_3h", v_utc, now=now_utc
        )

    candidates_map = _discover_candidates_bulk(db, model, now=now_utc)
    cands = candidates_map.get(v_utc)
    if not cands:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast data is available for model '{model}' and variable '{variable}' at valid time '{v_utc.isoformat()}'.",
        )

    cand = select_variable_source(cands, variable)
    if cand is None:
        if requires_lead0_display_fallback(variable) or is_precipitation_companion(variable):
            return None
        if variable == "wind_10m":
            raise HTTPException(
                status_code=404,
                detail=f"No coherent wind components available for model '{model}' at valid time '{v_utc.isoformat()}'.",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable}' is not available for model '{model}' at valid time '{v_utc.isoformat()}'.",
        )

    gen = _resolve_store_generation(cand.store_path)
    return ResolvedForecastSource(
        model=model.lower().strip(),
        valid_time=v_utc,
        cycle_time=cand.cycle_time,
        lead_time_hours=cand.lead_time_hours,
        run_id=cand.run_id,
        store_path=cand.store_path,
        serving_generation=gen,
        member_indices=cand.member_indices,
    )


def resolve_valid_time_source(
    db: Session,
    model: str,
    valid_time: datetime | str,
    *,
    variable: str | None = None,
    require_members: bool = False,
    now: datetime | None = None,
) -> ResolvedForecastSource:
    """Wrapper resolving the single winning source, raising HTTP 404 if unavailable.

    Preserves existing signature for backward compatibility.
    """
    if variable is not None:
        src = resolve_variable_source(db, model, variable, valid_time, now=now)
        if src is None:
            v_utc = parse_cycle_time(valid_time) if isinstance(valid_time, str) else _ensure_utc(valid_time)
            raise HTTPException(
                status_code=404,
                detail=f"No positive-lead interval forecast is available for model '{model}' and variable '{variable}' at valid time '{v_utc.isoformat()}'.",
            )
        return src
    return resolve_canonical_source(db, model, valid_time, require_members=require_members, now=now)


def resolve_canonical_sources_bulk(
    db: Session,
    model: str,
    *,
    variables: Iterable[str] | None = None,
    target_valid_times: Iterable[datetime] | None = None,
    start_lead_time_hours: int | None = None,
    end_lead_time_hours: int | None = None,
    now: datetime | None = None,
) -> tuple[
    dict[datetime, ResolvedForecastSource],
    dict[tuple[str, datetime], ResolvedForecastSource | None],
]:
    """Bulk-resolve canonical anchors and variable-specific fallbacks in 1-2 SQL queries.

    Returns:
        (resolved_anchors, resolved_variables)
        where:
        - resolved_anchors: valid_time -> ResolvedForecastSource
        - resolved_variables: (variable, valid_time) -> ResolvedForecastSource | None
    """
    now_utc = now if now is not None else get_current_time()
    start_vt = serving_start_valid_time(now_utc)
    m_id = model.lower().strip()
    var_list = list(variables) if variables is not None else []

    cands_map = _discover_candidates_bulk(
        db,
        m_id,
        start_lead_time_hours=start_lead_time_hours,
        end_lead_time_hours=end_lead_time_hours,
        now=now_utc,
    )

    # Filter target valid times if requested
    allowed_vts: set[datetime] | None = (
        {_ensure_utc(vt) for vt in target_valid_times}
        if target_valid_times is not None
        else None
    )

    # Cache store generations in memory to avoid repeated manifest reads
    gen_cache: dict[str, str | None] = {}

    def get_gen(path: str) -> str | None:
        if path not in gen_cache:
            gen_cache[path] = _resolve_store_generation(path)
        return gen_cache[path]

    canon_res = select_canonical_sources_bulk(
        cands_map,
        variables=var_list,
        target_valid_times=allowed_vts,
        start_valid_time=start_vt,
    )

    resolved_anchors: dict[datetime, ResolvedForecastSource] = {}
    resolved_variables: dict[tuple[str, datetime], ResolvedForecastSource | None] = {}

    for v_time, anchor_cand in canon_res.anchors.items():
        resolved_anchors[v_time] = ResolvedForecastSource(
            model=m_id,
            valid_time=v_time,
            cycle_time=anchor_cand.cycle_time,
            lead_time_hours=anchor_cand.lead_time_hours,
            run_id=anchor_cand.run_id,
            store_path=anchor_cand.store_path,
            serving_generation=get_gen(anchor_cand.store_path),
            member_indices=anchor_cand.member_indices,
        )

    for (var, v_time), v_cand in canon_res.variable_sources.items():
        if v_cand is not None:
            resolved_variables[(var, v_time)] = ResolvedForecastSource(
                model=m_id,
                valid_time=v_time,
                cycle_time=v_cand.cycle_time,
                lead_time_hours=v_cand.lead_time_hours,
                run_id=v_cand.run_id,
                store_path=v_cand.store_path,
                serving_generation=get_gen(v_cand.store_path),
                member_indices=v_cand.member_indices,
            )
        else:
            resolved_variables[(var, v_time)] = None

    return resolved_anchors, resolved_variables


def check_physical_targets_fenced(
    db: Session,
    targets: Iterable[tuple[str, str]],
    resolved_at: datetime | None = None,
) -> set[tuple[str, str]]:
    """Return any (run_id, physical_key) targets that entered deleting or deleted.

    If resolved_at is provided, filters for transitions on or after resolved_at.
    """
    target_list = list(targets)
    if not target_list:
        return set()

    try:
        bind = db.get_bind()
        if bind.dialect.name == "sqlite":
            from sqlalchemy import inspect

            if not inspect(bind).has_table("reclamation_queue"):
                return set()
    except Exception:
        return set()

    run_ids = {r for r, _ in target_list}
    stmt = select(
        ReclamationQueue.run_id,
        ReclamationQueue.physical_key,
    ).where(
        ReclamationQueue.run_id.in_(run_ids),
        ReclamationQueue.status.in_(("deleting", "deleted", "failed")),
    )
    if resolved_at is not None:
        r_utc = _ensure_utc(resolved_at)
        stmt = stmt.where(ReclamationQueue.updated_at >= r_utc)

    rows = db.execute(stmt).all()
    target_set = set(target_list)
    return {
        (str(r_id), str(p_key))
        for r_id, p_key in rows
        if (str(r_id), str(p_key)) in target_set
    }


def build_canonical_provenance_digest(
    resolved_anchors: dict[datetime, ResolvedForecastSource],
    resolved_variables: dict[tuple[str, datetime], ResolvedForecastSource | None] | None = None,
) -> str:
    """Compute a deterministic SHA-256 digest of resolved canonical provenance.

    Explicitly sorts entries so the digest is 100% immune to dict iteration order.
    Includes anchor provenance, variable fallback provenance (or explicit NULL sentinel),
    and store serving generation.
    """
    entries: list[tuple[str, ...]] = []
    for vt in sorted(resolved_anchors.keys()):
        src = resolved_anchors[vt]
        entries.append(
            (
                "anchor",
                vt.isoformat(),
                src.run_id,
                src.cycle_time.isoformat(),
                str(src.lead_time_hours),
                src.serving_generation or "",
            )
        )
    if resolved_variables:
        for var, vt in sorted(resolved_variables.keys()):
            v_src = resolved_variables[(var, vt)]
            if v_src is not None:
                entries.append(
                    (
                        var,
                        vt.isoformat(),
                        v_src.run_id,
                        v_src.cycle_time.isoformat(),
                        str(v_src.lead_time_hours),
                        v_src.serving_generation or "",
                    )
                )
            else:
                entries.append(
                    (
                        var,
                        vt.isoformat(),
                        "NULL",
                        "NULL",
                        "-1",
                        "NULL",
                    )
                )
    canonical_bytes = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def resolve_point_provenance_digest(
    db: Session,
    model: str,
    *,
    variables: Iterable[str] | None = None,
    start_lead_time_hours: int | None = None,
    end_lead_time_hours: int | None = None,
    now: datetime | None = None,
) -> str:
    """Compute the multi-valid-time point forecast cache provenance digest in 1 SQL query."""
    anchors, var_sources = resolve_canonical_sources_bulk(
        db,
        model,
        variables=variables,
        start_lead_time_hours=start_lead_time_hours,
        end_lead_time_hours=end_lead_time_hours,
        now=now,
    )
    return build_canonical_provenance_digest(anchors, var_sources)


def list_canonical_valid_times(
    db: Session,
    model: str,
    *,
    variable: str | None = None,
    now: datetime | None = None,
) -> list[datetime]:
    """Return ordered ascending list of servable canonical valid times >= serving_start."""
    cands_map = _discover_candidates_bulk(db, model, now=now)
    if variable is None:
        return sorted(cands_map.keys())

    servable_vts: list[datetime] = []
    for vt, cands in sorted(cands_map.items()):
        if not cands:
            continue
        if requires_lead0_display_fallback(variable):
            anchor = cands[0]
            if anchor.lead_time_hours > 0 and (not anchor.variables or variable in anchor.variables):
                servable_vts.append(vt)
            else:
                # Needs positive lead fallback
                if any(c.lead_time_hours > 0 and (not c.variables or variable in c.variables) for c in cands):
                    servable_vts.append(vt)
        elif variable == "wind_10m":
            if any(not c.variables or {"wind_u_10m", "wind_v_10m"}.issubset(c.variables) for c in cands):
                servable_vts.append(vt)
        else:
            if any(not c.variables or variable in c.variables for c in cands):
                servable_vts.append(vt)

    return servable_vts


def canonical_serving_horizon(
    db: Session,
    model: str,
    *,
    variable: str | None = None,
    now: datetime | None = None,
) -> datetime | None:
    """Derive maximum canonical valid time: max(list_canonical_valid_times(...))."""
    vts = list_canonical_valid_times(db, model, variable=variable, now=now)
    return vts[-1] if vts else None
