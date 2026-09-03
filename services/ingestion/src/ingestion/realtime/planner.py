"""Realtime lead-wave planner (pure, offline-testable).

Consumes one GFS and one GEFS :class:`~ingestion.providers.noaa.discovery.CycleSnapshot`
for the SAME cycle identity, the durable per-model committed state, the
canonical horizon, the bounded wave policy, and pending-timing metadata, and
returns a structured :class:`FrontierPlan` — never a bare lead list.

Semantics (Phase 5C, accepted):

* **Shared barrier (scheduler policy, v1):** a platform lead is complete only
  when GFS has data + ``.idx`` AND every required GEFS perturbation member
  (gep01..gep30, from the domain member contract) has data + ``.idx``. The
  barrier lives ONLY here — discovery reports per-model reality; storage,
  markers, catalog, and serving know nothing about it, so GFS/GEFS can be
  decoupled later by replacing this policy.
* **Three distinct frontiers:** ``observed_frontier`` (any upstream artifact —
  data or ``.idx``, any member, including upstream-only leads) ≠
  ``complete_frontier`` (contiguous shared-barrier-complete prefix of the
  canonical horizon) ≠ ``committed_frontier`` (contiguous prefix committed for
  BOTH models). A partially publishing lead can advance the observed frontier
  and trigger fast polling, but never enters a wave.
* **No jumping:** pending complete leads form a contiguous run after the
  committed frontier; the first lead that fails the barrier (or the
  predecessor dependency) blocks everything after it.
* **Per-model reconciliation:** wave targets are the shared candidate minus
  what each model has durably committed, so one model committing ahead (or a
  big-batch commit between polls) never duplicates work, and a failed model
  wave is retried alone on the next reconciliation.
* **Bounded batching:** a wave is due when ``len(pending) >= max_leads`` OR
  the oldest pending lead has waited ``max_wait`` seconds — whichever first.
  The candidate is the pending prefix truncated to ``max_leads``.
  ``first_seen_complete_at`` is timing-only scheduler memory; after a restart
  wait timers reset and pending work is reconstructed from upstream + durable
  state, which is correctness-safe.
* **Predecessor dependency:** a 6-hour-reset lead (``lead % 6 == 0, lead > 0``)
  requires its lead−3 predecessor committed for a model (or in the same
  candidate ahead of it) before that model may ingest it, otherwise ingestion
  would predictably fail with ``MissingPredecessorLeadError``. Given the
  contiguous walk this invariant holds structurally; the check is kept
  explicit so future policy changes cannot silently violate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from domain.coverage import get_expected_members
from domain.horizon import canonical_lead_time_hours

#: Canonical diagnostic reasons for the blocking lead.
BLOCK_HORIZON_COMPLETE = "horizon-complete"
BLOCK_GFS_INCOMPLETE = "gfs-incomplete"
BLOCK_GEFS_INCOMPLETE = "gefs-incomplete"
BLOCK_PREDECESSOR = "predecessor"
BLOCK_NOT_OBSERVED = "not-observed"


@dataclass(frozen=True)
class ModelCommittedState:
    """Durable committed state of ONE model's cycle, read from the catalog.

    ``leads`` is the committed lead set (deterministic products). ``pairs``
    carries the committed ``(member, lead)`` rows for ensemble products and is
    empty for deterministic models. A lead counts as committed for an ensemble
    model only when EVERY required member has a committed pair — partial
    member ingestion is a non-goal, so an under-covered lead stays pending and
    the whole lead is re-ingested idempotently.
    """

    leads: frozenset[int] = frozenset()
    pairs: frozenset[tuple[int, int]] = frozenset()

    def is_lead_committed(self, lead: int, *, ensemble: bool, expected_members: tuple[int, ...]) -> bool:
        """Whether ``lead`` is fully committed for this model."""
        if ensemble:
            committed_members = {
                member for (member, pair_lead) in self.pairs if pair_lead == lead
            }
            return all(m in committed_members for m in expected_members)
        return lead in self.leads


@dataclass(frozen=True)
class WavePolicy:
    """Bounded-batching policy for wave emission."""

    max_leads: int = 8
    max_wait_seconds: float = 1200.0


@dataclass(frozen=True)
class FrontierPlan:
    """Structured result of one planning pass (diagnostics + wave candidate).

    Attributes:
        observed_frontier: Highest lead with ANY upstream artifact across both
            models (any member, any state) — upstream reality, possibly ahead
            of completeness and including non-platform leads.
        complete_frontier: Highest lead of the contiguous shared-barrier-
            complete prefix of the canonical horizon (``None`` when the first
            horizon lead is incomplete).
        committed_frontier: Highest lead of the contiguous prefix committed
            for BOTH models (``None`` when the first horizon lead is not
            committed for both).
        pending_complete_leads: Shared-complete, uncommitted, contiguous and
            predecessor-eligible leads after the committed frontier (full,
            untruncated).
        next_blocked_lead: First lead that stops the walk, or ``None``.
        blocked_reason: Why ``next_blocked_lead`` is blocked (one of the
            ``BLOCK_*`` reasons), or ``None``.
        missing_gfs_artifacts: For the blocked lead, which GFS artifact kinds
            are absent (subset of ``("data", "idx")``); empty when GFS is fine.
        missing_gefs_members: For the blocked lead, the sorted required members
            lacking data + ``.idx``; empty when GEFS is fine.
        gfs_present / gefs_present: Whether each model's snapshot showed any
            artifact for the cycle (a model that has not started publishing is
            distinguishable from an empty-but-started snapshot only by this).
        wave_due: Whether the batching policy emits a wave now.
        wave_candidate: The emitted wave (pending prefix truncated to
            ``max_leads``); empty when no wave is due.
        wave_targets_gfs / wave_targets_gefs: Per-model dispatch targets — the
            candidate minus that model's durably committed leads.
        oldest_pending_age_seconds: Age of the oldest pending lead (0.0 for
            leads without timing metadata, e.g. after a restart); ``None``
            when nothing is pending.
    """

    cycle_label: str
    horizon: tuple[int, ...]
    observed_frontier: int | None
    complete_frontier: int | None
    committed_frontier: int | None
    pending_complete_leads: tuple[int, ...]
    next_blocked_lead: int | None
    blocked_reason: str | None
    missing_gfs_artifacts: tuple[str, ...] = ()
    missing_gefs_members: tuple[int, ...] = ()
    gfs_present: bool = False
    gefs_present: bool = False
    wave_due: bool = False
    wave_candidate: tuple[int, ...] = ()
    wave_targets_gfs: tuple[int, ...] = ()
    wave_targets_gefs: tuple[int, ...] = ()
    oldest_pending_age_seconds: float | None = None


def predecessor_satisfied(
    lead: int,
    *,
    committed: ModelCommittedState,
    ensemble: bool,
    expected_members: tuple[int, ...],
    candidate_so_far: tuple[int, ...],
) -> bool:
    """Whether ingesting ``lead`` for one model cannot hit a missing predecessor.

    A 6-hour-reset lead (precipitation de-accumulation / cloud reconstruction)
    reads its lead−3 slice from the model's own store; an all-NaN predecessor
    raises ``MissingPredecessorLeadError``. The predecessor must therefore be
    durably committed for the model OR be part of the same candidate ahead of
    this lead (the wave runner waits for in-wave predecessor decodes).

    Args:
        lead: The candidate lead.
        committed: The model's durable committed state.
        ensemble: Whether the model is an ensemble.
        expected_members: Required member identities (used for the ensemble
            committed check).
        candidate_so_far: Leads already accepted into the current candidate
            (ascending).

    Returns:
        True when the predecessor dependency is satisfied (trivially true for
        lead 0 and leads where ``lead % 6 != 0``).
    """
    if lead == 0 or lead % 6 != 0:
        return True
    predecessor = lead - 3
    if committed.is_lead_committed(predecessor, ensemble=ensemble, expected_members=expected_members):
        return True
    return predecessor in candidate_so_far


def plan_wave(
    *,
    cycle_label: str,
    gfs_snapshot: object | None,
    gefs_snapshot: object | None,
    committed_gfs: ModelCommittedState,
    committed_gefs: ModelCommittedState,
    policy: WavePolicy,
    first_seen_complete_at: Mapping[int, float] | None = None,
    now: float = 0.0,
    horizon: tuple[int, ...] | None = None,
    expected_gefs_members: tuple[int, ...] | None = None,
) -> FrontierPlan:
    """Plan one bounded lead wave from upstream + durable state.

    Args:
        cycle_label: Human-readable cycle identity for diagnostics (both models
            share this identity by construction).
        gfs_snapshot: The GFS cycle snapshot (``None`` when the model has not
            been observed at all for this cycle).
        gefs_snapshot: The GEFS cycle snapshot (``None`` when unobserved).
        committed_gfs: Durable GFS committed state.
        committed_gefs: Durable GEFS committed state.
        policy: Bounded-batching policy.
        first_seen_complete_at: Timing-only map of pending lead → first-seen-
            complete timestamp (scheduler memory; may be empty after restart).
        now: Current time (injected clock, same base as the timing map).
        horizon: Canonical lead sequence override; defaults to the domain
            contract for GFS/GEFS (identical 81-lead sequence).
        expected_gefs_members: Required GEFS members; defaults to the domain
            member contract (gep01..gep30).

    Returns:
        The structured :class:`FrontierPlan`.
    """
    from ingestion.providers.noaa.discovery import CycleSnapshot

    seq = horizon if horizon is not None else canonical_lead_time_hours("gfs")
    members = (
        expected_gefs_members
        if expected_gefs_members is not None
        else tuple(range(1, get_expected_members("gefs") + 1))
    )

    gfs: CycleSnapshot | None = gfs_snapshot  # type: ignore[assignment]
    gefs: CycleSnapshot | None = gefs_snapshot  # type: ignore[assignment]
    gfs_present = gfs is not None and len(gfs.regions) > 0
    gefs_present = gefs is not None and len(gefs.regions) > 0

    observed_values: list[int] = []
    if gfs is not None:
        gfs_observed = gfs.highest_observed_lead()
        if gfs_observed is not None:
            observed_values.append(gfs_observed)
    if gefs is not None:
        gefs_observed = gefs.highest_observed_lead()
        if gefs_observed is not None:
            observed_values.append(gefs_observed)
    observed_frontier = max(observed_values) if observed_values else None

    def gfs_complete(lead: int) -> bool:
        return gfs is not None and gfs.is_artifact_complete(None, lead)

    def gefs_complete(lead: int) -> bool:
        return gefs is not None and all(
            gefs.is_artifact_complete(m, lead) for m in members
        )

    def shared_complete(lead: int) -> bool:
        return gfs_complete(lead) and gefs_complete(lead)

    # Committed frontier: contiguous prefix committed for BOTH models.
    committed_frontier: int | None = None
    for lead in seq:
        gfs_ok = committed_gfs.is_lead_committed(lead, ensemble=False, expected_members=())
        gefs_ok = committed_gefs.is_lead_committed(
            lead, ensemble=True, expected_members=members
        )
        if gfs_ok and gefs_ok:
            committed_frontier = lead
        else:
            break

    # Complete frontier: contiguous shared-complete prefix from horizon start.
    complete_frontier: int | None = None
    for lead in seq:
        if shared_complete(lead):
            complete_frontier = lead
        else:
            break

    # Pending walk: after the committed frontier, accept contiguous
    # shared-complete, predecessor-eligible leads; stop at the first blocker.
    pending: list[int] = []
    next_blocked_lead: int | None = None
    blocked_reason: str | None = None
    missing_gfs_artifacts: tuple[str, ...] = ()
    missing_gefs_members: tuple[int, ...] = ()
    for lead in seq:
        gfs_committed = committed_gfs.is_lead_committed(
            lead, ensemble=False, expected_members=()
        )
        gefs_committed = committed_gefs.is_lead_committed(
            lead, ensemble=True, expected_members=members
        )
        if gfs_committed and gefs_committed:
            continue  # durably done for both models; nothing to plan here
        if not shared_complete(lead):
            next_blocked_lead = lead
            if not (gfs_present or gefs_present):
                blocked_reason = BLOCK_NOT_OBSERVED
            elif not gfs_complete(lead):
                blocked_reason = BLOCK_GFS_INCOMPLETE
            else:
                blocked_reason = BLOCK_GEFS_INCOMPLETE
            if gfs is not None:
                region = gfs.region(None, lead)
                missing = []
                if region.data is None:
                    missing.append("data")
                if region.idx is None:
                    missing.append("idx")
                missing_gfs_artifacts = tuple(missing)
            if gefs is not None:
                missing_gefs_members = gefs.missing_members(lead, expected=members)
            break
        gfs_ok = predecessor_satisfied(
            lead,
            committed=committed_gfs,
            ensemble=False,
            expected_members=(),
            candidate_so_far=tuple(pending),
        )
        gefs_ok = predecessor_satisfied(
            lead,
            committed=committed_gefs,
            ensemble=True,
            expected_members=members,
            candidate_so_far=tuple(pending),
        )
        if not (gfs_ok and gefs_ok):
            next_blocked_lead = lead
            blocked_reason = BLOCK_PREDECESSOR
            break
        pending.append(lead)

    # Bounded batching.
    timing = dict(first_seen_complete_at) if first_seen_complete_at else {}
    oldest_pending_age: float | None = None
    wave_due = False
    wave_candidate: tuple[int, ...] = ()
    if pending:
        oldest_seen = min(timing.get(lead, now) for lead in pending)
        oldest_pending_age = max(0.0, now - oldest_seen)
        wave_due = (
            len(pending) >= policy.max_leads
            or oldest_pending_age >= policy.max_wait_seconds
        )
        if wave_due:
            wave_candidate = tuple(pending[: policy.max_leads])

    targets_gfs = tuple(lead for lead in wave_candidate if not committed_gfs.is_lead_committed(
        lead, ensemble=False, expected_members=()
    ))
    targets_gefs = tuple(
        lead
        for lead in wave_candidate
        if not committed_gefs.is_lead_committed(
            lead, ensemble=True, expected_members=members
        )
    )

    if next_blocked_lead is None and pending:
        blocked_reason = None
    elif next_blocked_lead is None:
        blocked_reason = BLOCK_HORIZON_COMPLETE

    return FrontierPlan(
        cycle_label=cycle_label,
        horizon=seq,
        observed_frontier=observed_frontier,
        complete_frontier=complete_frontier,
        committed_frontier=committed_frontier,
        pending_complete_leads=tuple(pending),
        next_blocked_lead=next_blocked_lead,
        blocked_reason=blocked_reason,
        missing_gfs_artifacts=missing_gfs_artifacts,
        missing_gefs_members=missing_gefs_members,
        gfs_present=gfs_present,
        gefs_present=gefs_present,
        wave_due=wave_due,
        wave_candidate=wave_candidate,
        wave_targets_gfs=targets_gfs,
        wave_targets_gefs=targets_gefs,
        oldest_pending_age_seconds=oldest_pending_age,
    )
