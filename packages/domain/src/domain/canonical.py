"""Pure canonical valid-time source resolution and reachability semantics (Lifecycle V3).

This module contains the pure, service-independent canonical resolution engine
shared across API serving and GC reclamation planning. It contains no
SQLAlchemy, database connections, or HTTP logic.

Canonical V3 Principles:
------------------------
1. Per-Valid-Time Ownership:
   For model M and valid_time V, the canonical source is the newest committed
   eligible representation capable of representing V.
   A newer partial cycle supersedes older cycles ONLY for valid times it has
   actually committed. Older cycles continue serving farther valid times.

2. Two Reachability Perspectives:
   - Effective Serving Availability:
     Excludes all targets in reclamation_queue with status IN ('deleting', 'deleted').
   - Counterfactual Semantic Necessity (Worker Revalidation):
     Evaluates claimed batch B as physically present while all other deleting/deleted
     targets remain fenced.

3. Dependencies & Invariants:
   - Interval variables (precipitation_amount_3h, cloud_cover_3h) at lead 0 fall back
     to the newest committed older positive lead covering valid_time.
   - Precipitation companion variables (crain, csnow, cfrzr, cicep) are strictly
     coupled to precipitation_amount_3h.
   - Synthetic wind_10m requires both wind_u_10m and wind_v_10m from the same source.
   - Strict GEFS coherent vintage: requires both official mean and >=85% members.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from domain.coverage import is_lead_servable
from domain.temporal import (
    is_precipitation_companion,
    requires_lead0_display_fallback,
)


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class CanonicalCandidate:
    """A single forecast cycle lead candidate covering a valid time.

    Attributes:
        cycle_time: The run's cycle time (UTC).
        lead_time_hours: Forecast offset hours from cycle_time.
        run_id: Database identifier of the model run.
        store_path: Canonical store path/URL.
        product_types: Set of product types committed for this lead (e.g. 'surface',
            'ensemble_mean').
        variables: Set of data variables committed for this lead.
        member_indices: Tuple of committed ensemble member indices, or None for deterministic.
        status: Model run status ('ready', 'processing', 'partial').
    """

    cycle_time: datetime
    lead_time_hours: int
    run_id: str
    store_path: str
    product_types: frozenset[str]
    variables: frozenset[str]
    member_indices: tuple[int, ...] | None = None
    status: str = "ready"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_time", _ensure_utc(self.cycle_time))

    @property
    def valid_time(self) -> datetime:
        """UTC valid time represented by this candidate."""
        return self.cycle_time + timedelta(hours=self.lead_time_hours)


@dataclass(frozen=True)
class CanonicalResolutionResult:
    """Pure resolution result mapping valid times and variables to winning candidates."""

    anchors: dict[datetime, CanonicalCandidate]
    variable_sources: dict[tuple[str, datetime], CanonicalCandidate | None]
    interval_fallbacks: dict[tuple[str, datetime], CanonicalCandidate]
    companion_dependencies: dict[tuple[str, datetime], CanonicalCandidate]


def select_canonical_anchor(
    candidates: Sequence[CanonicalCandidate],
) -> CanonicalCandidate | None:
    """Select the newest committed eligible anchor candidate for a valid time.

    Assumes candidates are sorted by ascending lead_time_hours (newest cycle first).
    """
    if not candidates:
        return None
    return candidates[0]


def select_variable_source(
    candidates: Sequence[CanonicalCandidate],
    variable: str,
    *,
    anchor: CanonicalCandidate | None = None,
) -> CanonicalCandidate | None:
    """Select the winning candidate for a specific variable at one valid time.

    Implements:
    1. Interval fallback for lead 0 (precipitation_amount_3h, cloud_cover_3h).
    2. Companion coupling (crain, csnow, cfrzr, cicep) -> precipitation_amount_3h.
    3. Coherent wind components for wind_10m (requires both wind_u_10m and wind_v_10m).
    4. Ordinary variables: first candidate carrying variable in its variables set.
    """
    if not candidates:
        return None

    if is_precipitation_companion(variable):
        return select_variable_source(
            candidates, "precipitation_amount_3h", anchor=anchor
        )

    effective_anchor = anchor if anchor is not None else candidates[0]

    # Interval variables requiring positive-lead fallback when anchor is lead 0
    if requires_lead0_display_fallback(variable):
        if effective_anchor.lead_time_hours > 0:
            for cand in candidates:
                if not cand.variables or variable in cand.variables:
                    return cand
            return None
        # Anchor is lead 0 -> search for newest positive-lead candidate
        for cand in candidates:
            if cand.lead_time_hours > 0 and (
                not cand.variables or variable in cand.variables
            ):
                return cand
        return None

    # Synthetic wind_10m: requires both U and V components in same candidate
    if variable == "wind_10m":
        for cand in candidates:
            if not cand.variables or {"wind_u_10m", "wind_v_10m"}.issubset(
                cand.variables
            ):
                return cand
        return None

    # Ordinary variables
    for cand in candidates:
        if not cand.variables or variable in cand.variables:
            return cand
    return None


def select_canonical_sources_bulk(
    candidates_by_valid: dict[datetime, list[CanonicalCandidate]],
    *,
    variables: Iterable[str] | None = None,
    target_valid_times: Iterable[datetime] | None = None,
    start_valid_time: datetime | None = None,
) -> CanonicalResolutionResult:
    """Bulk-resolve canonical anchors and variable sources for a set of valid times.

    Pure deterministic implementation with zero database or filesystem I/O.
    """
    var_list = list(variables) if variables is not None else []
    allowed_vts: set[datetime] | None = (
        {_ensure_utc(vt) for vt in target_valid_times}
        if target_valid_times is not None
        else None
    )

    resolved_anchors: dict[datetime, CanonicalCandidate] = {}
    resolved_variables: dict[tuple[str, datetime], CanonicalCandidate | None] = {}
    interval_fallbacks: dict[tuple[str, datetime], CanonicalCandidate] = {}
    companion_dependencies: dict[tuple[str, datetime], CanonicalCandidate] = {}

    for v_time, cands in sorted(candidates_by_valid.items()):
        if start_valid_time is not None and v_time < start_valid_time:
            continue
        if allowed_vts is not None and v_time not in allowed_vts:
            continue
        if not cands:
            continue

        anchor = cands[0]
        resolved_anchors[v_time] = anchor

        # Check precipitation and companion resolution
        has_precip = "precipitation_amount_3h" in var_list or any(
            is_precipitation_companion(v) for v in var_list
        )
        if has_precip:
            p_src = select_variable_source(
                cands, "precipitation_amount_3h", anchor=anchor
            )
            resolved_variables[("precipitation_amount_3h", v_time)] = p_src
            if p_src is not None and anchor.lead_time_hours == 0:
                interval_fallbacks[("precipitation_amount_3h", v_time)] = p_src
            for comp in ("crain", "csnow", "cfrzr", "cicep"):
                if comp in var_list:
                    resolved_variables[(comp, v_time)] = p_src
                    if p_src is not None:
                        companion_dependencies[(comp, v_time)] = p_src

        for var in var_list:
            if var == "precipitation_amount_3h" or is_precipitation_companion(var):
                continue
            src = select_variable_source(cands, var, anchor=anchor)
            resolved_variables[(var, v_time)] = src
            if (
                src is not None
                and var == "cloud_cover_3h"
                and anchor.lead_time_hours == 0
            ):
                interval_fallbacks[(var, v_time)] = src

    return CanonicalResolutionResult(
        anchors=resolved_anchors,
        variable_sources=resolved_variables,
        interval_fallbacks=interval_fallbacks,
        companion_dependencies=companion_dependencies,
    )


def filter_candidates_by_physical_fence(
    candidates_by_valid: dict[datetime, list[CanonicalCandidate]],
    fenced_keys: set[tuple[str, int, str, str, int]],
    is_ensemble: bool = False,
    expected_members: int = 30,
) -> dict[datetime, list[CanonicalCandidate]]:
    """Filter out or narrow candidates whose physical shards are fenced.

    fenced_keys contains (run_id, lead_time_hours, variable_code, target_kind, member_index).
    """
    if not fenced_keys:
        return candidates_by_valid

    out: dict[datetime, list[CanonicalCandidate]] = {}
    for v_time, cands in candidates_by_valid.items():
        filtered_cands: list[CanonicalCandidate] = []
        for cand in cands:
            r_id = cand.run_id
            lead = cand.lead_time_hours

            active_vars = {
                v
                for v in cand.variables
                if not any(
                    fk[0] == r_id
                    and fk[1] == lead
                    and fk[2] == v
                    and fk[3] in ("det", "mean")
                    for fk in fenced_keys
                )
            }

            if is_ensemble and cand.member_indices is not None:
                m_indices = cand.member_indices
                avail_members = [
                    m
                    for m in m_indices
                    if not any(
                        fk[0] == r_id
                        and fk[1] == lead
                        and fk[3] == "mem"
                        and fk[4] == m
                        for fk in fenced_keys
                    )
                ]
                if not is_lead_servable(len(avail_members), expected_members):
                    continue
                mean_fenced = any(
                    fk[0] == r_id and fk[1] == lead and fk[3] == "mean"
                    for fk in fenced_keys
                )
                prod_types = (
                    cand.product_types - {"ensemble_mean"}
                    if mean_fenced
                    else cand.product_types
                )
                cand = CanonicalCandidate(
                    cycle_time=cand.cycle_time,
                    lead_time_hours=cand.lead_time_hours,
                    run_id=cand.run_id,
                    store_path=cand.store_path,
                    product_types=prod_types,
                    variables=frozenset(active_vars),
                    member_indices=tuple(avail_members),
                    status=cand.status,
                )
            else:
                if cand.variables and not active_vars:
                    continue
                cand = CanonicalCandidate(
                    cycle_time=cand.cycle_time,
                    lead_time_hours=cand.lead_time_hours,
                    run_id=cand.run_id,
                    store_path=cand.store_path,
                    product_types=cand.product_types,
                    variables=frozenset(active_vars),
                    member_indices=cand.member_indices,
                    status=cand.status,
                )
            filtered_cands.append(cand)
        if filtered_cands:
            out[v_time] = filtered_cands
    return out
