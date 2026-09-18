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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from domain.coverage import is_lead_servable
from domain.temporal import (
    is_precipitation_companion,
    requires_lead0_display_fallback,
)


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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


@dataclass(frozen=True)
class FenceIndex:
    """A physical fence key set projected into the membership shapes the
    candidate filter tests against.

    ``build_fence_index`` is the only implementation of that projection. Each
    set is an exact transliteration of the per-key predicate the filter used to
    evaluate by scanning the raw key set:

    * ``variable_keys`` <- the ``fk[3] in ("det", "mean")`` guard on the
      per-variable test (the member axis is deliberately absent from it);
    * ``member_keys``   <- the per-member test, which matches on
      ``(run_id, lead, member)`` with no variable component;
    * ``mean_run_leads`` <- the ``fk[3] == "mean"`` guard on the whole-lead
      mean test.

    ``has_keys`` records whether the input held any key at all, including keys
    whose ``target_kind`` matches none of the three projections. The filter's
    empty-input early return keys on that fact rather than on the projections
    being empty: an input of exclusively unrecognised kinds still takes the
    rebuild path and yields freshly constructed candidates.
    """

    variable_keys: frozenset[tuple[str, int, str]]
    member_keys: frozenset[tuple[str, int, int]]
    mean_run_leads: frozenset[tuple[str, int]]
    has_keys: bool


def build_fence_index(
    fenced_keys: Iterable[tuple[str, int, str, str, int]],
) -> FenceIndex:
    """Project raw fence keys into the three membership sets, consuming a stream.

    ``fenced_keys`` is iterated exactly once, so a caller holding a database
    cursor can pass it straight in: only the projections are retained, and they
    are sized by the number of distinct fence units rather than by the number of
    rows read.

    Database drivers hand back a fresh ``str`` per row, so the projections would
    otherwise retain one copy of ``run_id`` / ``variable_code`` per distinct key.
    ``memo`` collapses those onto one object per distinct value. It is scoped to
    this call so the retained values die with the index, unlike ``sys.intern``,
    which would mutate process-global state from a pure module and keep whatever
    any caller ever passed in.
    """
    variable_keys: set[tuple[str, int, str]] = set()
    member_keys: set[tuple[str, int, int]] = set()
    mean_run_leads: set[tuple[str, int]] = set()
    memo: dict[str, str] = {}
    has_keys = False
    for fk in fenced_keys:
        has_keys = True
        r_id = memo.setdefault(fk[0], fk[0])
        kind = fk[3]
        if kind == "mean":
            variable_keys.add((r_id, fk[1], memo.setdefault(fk[2], fk[2])))
            mean_run_leads.add((r_id, fk[1]))
        elif kind == "det":
            variable_keys.add((r_id, fk[1], memo.setdefault(fk[2], fk[2])))
        elif kind == "mem":
            member_keys.add((r_id, fk[1], fk[4]))
    return FenceIndex(
        variable_keys=frozenset(variable_keys),
        member_keys=frozenset(member_keys),
        mean_run_leads=frozenset(mean_run_leads),
        has_keys=has_keys,
    )


def filter_candidates_by_physical_fence(
    candidates_by_valid: dict[datetime, list[CanonicalCandidate]],
    fenced_keys: set[tuple[str, int, str, str, int]],
    is_ensemble: bool = False,
    expected_members: int = 30,
) -> dict[datetime, list[CanonicalCandidate]]:
    """Filter out or narrow candidates whose physical shards are fenced.

    fenced_keys contains (run_id, lead_time_hours, variable_code, target_kind, member_index).

    Scanning instead of indexing made the caller O(candidates x variables x
    |fenced_keys|): with the production fenced set at ~2.8e5 entries and ~7.8e2
    candidates that is ~1e10 Python iterations per worker revalidation pass, so
    the set is indexed once into three projections (``FenceIndex``) and every
    membership test below is a single hash lookup.

    Callers that read the fence out of a database should stream it through
    ``build_fence_index`` and call ``filter_candidates_by_fence_index``, which
    never materialises the row set.
    """
    if not fenced_keys:
        return candidates_by_valid
    return filter_candidates_by_fence_index(
        candidates_by_valid,
        build_fence_index(fenced_keys),
        is_ensemble=is_ensemble,
        expected_members=expected_members,
    )


def filter_candidates_by_fence_index(
    candidates_by_valid: dict[datetime, list[CanonicalCandidate]],
    fence_index: FenceIndex,
    is_ensemble: bool = False,
    expected_members: int = 30,
) -> dict[datetime, list[CanonicalCandidate]]:
    """Filter candidates against a pre-built :class:`FenceIndex`.

    Semantically identical to ``filter_candidates_by_physical_fence``; this is
    the entry point for callers that build the index from a stream. The empty
    test reads ``fence_index.has_keys``, which is the fact the raw key set's own
    empty test expresses.
    """
    if not fence_index.has_keys:
        return candidates_by_valid

    fenced_variable_keys = fence_index.variable_keys
    fenced_member_keys = fence_index.member_keys
    fenced_mean_run_leads = fence_index.mean_run_leads

    out: dict[datetime, list[CanonicalCandidate]] = {}
    for v_time, cands in candidates_by_valid.items():
        filtered_cands: list[CanonicalCandidate] = []
        for cand in cands:
            r_id = cand.run_id
            lead = cand.lead_time_hours

            active_vars = {
                v
                for v in cand.variables
                if (r_id, lead, v) not in fenced_variable_keys
            }

            if is_ensemble and cand.member_indices is not None:
                m_indices = cand.member_indices
                avail_members = [
                    m for m in m_indices if (r_id, lead, m) not in fenced_member_keys
                ]
                if not is_lead_servable(len(avail_members), expected_members):
                    continue
                mean_fenced = (r_id, lead) in fenced_mean_run_leads
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
