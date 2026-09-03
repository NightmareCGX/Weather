"""Deterministic offline tests for the realtime lead-wave planner (Phase 5C).

Snapshots are constructed directly as immutable ``CycleSnapshot`` values (no
I/O, no live upstream). The canonical horizon is reduced via the domain
registry fixture contract for readability; production defaults to the
81-lead horizon (asserted once below).
"""

from __future__ import annotations

from datetime import date

import pytest

from ingestion.providers.noaa.discovery import (
    ArtifactObservation,
    CycleSnapshot,
    RegionArtifacts,
)
from ingestion.realtime.planner import (
    BLOCK_GFS_INCOMPLETE,
    BLOCK_GEFS_INCOMPLETE,
    BLOCK_HORIZON_COMPLETE,
    BLOCK_NOT_OBSERVED,
    FrontierPlan,
    ModelCommittedState,
    WavePolicy,
    plan_wave,
    predecessor_satisfied,
)

CYCLE = date(2026, 7, 21)
HORIZON = tuple(range(0, 24, 3))  # reduced injected horizon: 8 leads
MEMBERS = tuple(range(1, 31))


@pytest.fixture(autouse=True)
def reduced_horizon():
    """Inject a reduced canonical horizon (test fixture contract) and restore."""
    from domain.horizon import MODEL_CANONICAL_HORIZONS, register_canonical_lead_horizon

    saved = dict(MODEL_CANONICAL_HORIZONS)
    register_canonical_lead_horizon("gfs", HORIZON)
    register_canonical_lead_horizon("gefs", HORIZON)
    yield HORIZON
    MODEL_CANONICAL_HORIZONS.clear()
    MODEL_CANONICAL_HORIZONS.update(saved)


def _obs(key: str, size: int = 100) -> ArtifactObservation:
    return ArtifactObservation(key=key, size=size, etag="e", last_modified=None)


def gfs_snapshot(
    complete: tuple[int, ...] = (),
    *,
    data_only: tuple[int, ...] = (),
    idx_only: tuple[int, ...] = (),
    observed_only: tuple[int, ...] = (),
) -> CycleSnapshot:
    """A GFS snapshot: complete = data+idx; observed_only = e.g. hourly leads."""
    regions: dict[tuple[int | None, int], RegionArtifacts] = {}
    for lead in complete:
        regions[(None, lead)] = RegionArtifacts(
            data=_obs(f"gfs...f{lead:03d}"), idx=_obs(f"gfs...f{lead:03d}.idx")
        )
    for lead in data_only:
        regions.setdefault(
            (None, lead), RegionArtifacts(data=_obs(f"gfs...f{lead:03d}"), idx=None)
        )
    for lead in idx_only:
        regions.setdefault(
            (None, lead), RegionArtifacts(data=None, idx=_obs(f"gfs...f{lead:03d}.idx"))
        )
    for lead in observed_only:
        regions.setdefault(
            (None, lead), RegionArtifacts(data=_obs(f"gfs...f{lead:03d}"), idx=None)
        )
    return CycleSnapshot(
        model="gfs",
        cycle_date=CYCLE,
        cycle_hour=0,
        prefix="gfs-prefix",
        regions=regions,
    )


def gefs_snapshot(
    complete_members: dict[int, tuple[int, ...]] | None = None,
    *,
    unobserved: tuple[int, ...] = (),
) -> CycleSnapshot:
    """A GEFS snapshot: complete_members maps lead -> members with data+idx."""
    regions: dict[tuple[int | None, int], RegionArtifacts] = {}
    for lead, members in (complete_members or {}).items():
        for member in members:
            regions[(member, lead)] = RegionArtifacts(
                data=_obs(f"gep{member:02d}...f{lead:03d}"),
                idx=_obs(f"gep{member:02d}...f{lead:03d}.idx"),
            )
    for lead in unobserved:
        regions.setdefault((1, lead), RegionArtifacts(data=None, idx=None))
    return CycleSnapshot(
        model="gefs",
        cycle_date=CYCLE,
        cycle_hour=0,
        prefix="gefs-prefix",
        regions=regions,
    )


def committed(
    leads: tuple[int, ...] = (), *, pairs: tuple[tuple[int, int], ...] = ()
) -> ModelCommittedState:
    return ModelCommittedState(
        leads=frozenset(leads), pairs=frozenset(pairs)
    )


def gefs_committed(leads: tuple[int, ...]) -> ModelCommittedState:
    """GEFS committed state expressed as the full (member, lead) pair set."""
    return ModelCommittedState(
        leads=frozenset(leads),
        pairs=frozenset((member, lead) for lead in leads for member in MEMBERS),
    )


def plan(
    gfs: CycleSnapshot | None,
    gefs: CycleSnapshot | None,
    *,
    committed_gfs: ModelCommittedState | None = None,
    committed_gefs: ModelCommittedState | None = None,
    max_leads: int = 8,
    max_wait: float = 1200.0,
    first_seen: dict[int, float] | None = None,
    now: float = 0.0,
) -> FrontierPlan:
    return plan_wave(
        cycle_label="2026-07-21T00Z",
        gfs_snapshot=gfs,
        gefs_snapshot=gefs,
        committed_gfs=committed_gfs or committed(),
        committed_gefs=committed_gefs or committed(),
        policy=WavePolicy(max_leads=max_leads, max_wait_seconds=max_wait),
        first_seen_complete_at=first_seen,
        now=now,
    )


def full_gefs(leads: tuple[int, ...]) -> CycleSnapshot:
    return gefs_snapshot({lead: MEMBERS for lead in leads})


# ---------------------------------------------------------------------------
# Frontier / barrier scenarios
# ---------------------------------------------------------------------------


def test_nothing_observed_blocks_at_first_lead() -> None:
    result = plan(gfs_snapshot(), gefs_snapshot())
    assert result.observed_frontier is None
    assert result.complete_frontier is None
    assert result.committed_frontier is None
    assert result.pending_complete_leads == ()
    assert result.next_blocked_lead == 0
    assert result.blocked_reason == BLOCK_NOT_OBSERVED
    assert result.wave_candidate == ()
    assert not result.wave_due
    assert not result.gfs_present and not result.gefs_present


def test_gfs_only_blocks_until_gefs_publishes() -> None:
    result = plan(gfs_snapshot(complete=(0, 3)), gefs_snapshot())
    assert result.observed_frontier == 3
    assert result.gfs_present and not result.gefs_present
    assert result.pending_complete_leads == ()
    assert result.next_blocked_lead == 0
    assert result.blocked_reason == BLOCK_GEFS_INCOMPLETE
    # Exact missing-member diagnostics: the full required set is missing.
    assert result.missing_gefs_members == MEMBERS
    assert result.missing_gfs_artifacts == ()
    assert result.wave_targets_gefs == ()


def test_gefs_partial_publication_blocks_and_reports_missing_members() -> None:
    gefs = gefs_snapshot({0: MEMBERS[:8]})  # f000: 8/30 members complete
    result = plan(gfs_snapshot(complete=(0,)), gefs)
    assert result.observed_frontier == 0
    assert result.next_blocked_lead == 0
    assert result.blocked_reason == BLOCK_GEFS_INCOMPLETE
    assert result.missing_gefs_members == MEMBERS[8:]
    assert result.pending_complete_leads == ()
    # A partially publishing lead may inform polling but never enters a wave.
    assert result.wave_candidate == ()


def test_gefs_30_of_30_but_gfs_incomplete_blocks() -> None:
    gefs = full_gefs((0,))
    result = plan(gfs_snapshot(data_only=(0,)), gefs)  # data without .idx
    assert result.next_blocked_lead == 0
    assert result.blocked_reason == BLOCK_GFS_INCOMPLETE
    assert result.missing_gfs_artifacts == ("idx",)
    assert result.missing_gefs_members == ()  # GEFS side is fine
    assert result.pending_complete_leads == ()


def test_complete_shared_lead_becomes_pending() -> None:
    result = plan(
        gfs_snapshot(complete=(0,)),
        full_gefs((0,)),
    )
    assert result.complete_frontier == 0
    assert result.pending_complete_leads == (0,)
    assert result.next_blocked_lead == 3
    assert result.blocked_reason in (BLOCK_GFS_INCOMPLETE, BLOCK_GEFS_INCOMPLETE)
    # Single pending lead under default policy (max 8): not due yet.
    assert not result.wave_due
    assert result.wave_candidate == ()


def test_contiguous_progression_after_committed_frontier() -> None:
    result = plan(
        gfs_snapshot(complete=(0, 3, 6, 9)),
        full_gefs((0, 3, 6, 9)),
        committed_gfs=committed((0,)),
        committed_gefs=gefs_committed((0,)),
    )
    assert result.committed_frontier == 0
    assert result.pending_complete_leads == (3, 6, 9)
    assert result.next_blocked_lead == 12


def test_incomplete_gap_blocks_later_complete_leads() -> None:
    # f003 missing one GEFS member; f006..f015 fully complete upstream.
    gefs = gefs_snapshot({0: MEMBERS, 3: MEMBERS[:29], 6: MEMBERS, 9: MEMBERS, 12: MEMBERS, 15: MEMBERS})
    result = plan(
        gfs_snapshot(complete=(0, 3, 6, 9, 12, 15)),
        gefs,
        committed_gfs=committed((0,)),
        committed_gefs=gefs_committed((0,)),
    )
    assert result.next_blocked_lead == 3
    assert result.blocked_reason == BLOCK_GEFS_INCOMPLETE
    assert result.missing_gefs_members == (30,)
    assert result.pending_complete_leads == ()
    # Later leads must NOT be jumped over.
    assert 6 not in result.pending_complete_leads
    assert result.wave_candidate == ()


def test_observed_frontier_ahead_of_complete_frontier() -> None:
    # f015 is partially publishing (8/30) while f003..f012 are complete.
    gefs = gefs_snapshot(
        {0: MEMBERS, 3: MEMBERS, 6: MEMBERS, 9: MEMBERS, 12: MEMBERS, 15: MEMBERS[:8]}
    )
    result = plan(
        gfs_snapshot(complete=(0, 3, 6, 9, 12, 15), observed_only=(18,)),
        gefs,
    )
    assert result.observed_frontier == 18
    assert result.complete_frontier == 12
    assert result.pending_complete_leads == (0, 3, 6, 9, 12)
    assert result.next_blocked_lead == 15
    assert result.blocked_reason == BLOCK_GEFS_INCOMPLETE
    assert result.missing_gefs_members == MEMBERS[8:]


def test_already_committed_leads_leave_no_pending_work() -> None:
    result = plan(
        gfs_snapshot(complete=HORIZON),
        full_gefs(HORIZON),
        committed_gfs=committed(HORIZON),
        committed_gefs=gefs_committed(HORIZON),
    )
    assert result.committed_frontier == HORIZON[-1]
    assert result.pending_complete_leads == ()
    assert result.next_blocked_lead is None
    assert result.blocked_reason == BLOCK_HORIZON_COMPLETE
    assert not result.wave_due


def test_committed_prefix_with_upstream_gap_blocks_after_frontier() -> None:
    """Committed {0,3,6} with only f000..f006 upstream-complete: the walk has
    no pending work but still blocks (and reports) at the next lead."""
    result = plan(
        gfs_snapshot(complete=(0, 3, 6)),
        full_gefs((0, 3, 6)),
        committed_gfs=committed((0, 3, 6)),
        committed_gefs=gefs_committed((0, 3, 6)),
    )
    assert result.committed_frontier == 6
    assert result.pending_complete_leads == ()
    assert result.next_blocked_lead == 9
    assert result.blocked_reason == BLOCK_GFS_INCOMPLETE
    assert not result.wave_due


def test_one_model_committed_ahead_retargets_only_the_other() -> None:
    # GFS (big-batch) committed {0,3,6}; GEFS committed only {0}.
    result = plan(
        gfs_snapshot(complete=(0, 3, 6, 9)),
        full_gefs((0, 3, 6, 9)),
        committed_gfs=committed((0, 3, 6, 9)),
        committed_gefs=gefs_committed((0,)),
        max_leads=8,
        max_wait=0.0,  # force due
        first_seen={3: 0.0, 6: 0.0, 9: 0.0},
        now=1000.0,
    )
    assert result.pending_complete_leads == (3, 6, 9)
    assert result.wave_due
    assert result.wave_candidate == (3, 6, 9)
    # GFS already has the work; only GEFS is retargeted.
    assert result.wave_targets_gfs == ()
    assert result.wave_targets_gefs == (3, 6, 9)


def test_predecessor_dependency_helper() -> None:
    # Leads without a 6h reset never need a predecessor.
    assert predecessor_satisfied(0, committed=committed(), ensemble=False, expected_members=(), candidate_so_far=())
    assert predecessor_satisfied(3, committed=committed(), ensemble=False, expected_members=(), candidate_so_far=())
    # Lead 6 requires lead 3 committed or already in the candidate.
    assert not predecessor_satisfied(6, committed=committed((0,)), ensemble=False, expected_members=(), candidate_so_far=())
    assert predecessor_satisfied(6, committed=committed((0, 3)), ensemble=False, expected_members=(), candidate_so_far=())
    assert predecessor_satisfied(6, committed=committed((0,)), ensemble=False, expected_members=(), candidate_so_far=(3,))
    # Ensemble variant checks the model's own committed pairs.
    ens_committed = committed((), pairs=tuple((m, 3) for m in MEMBERS))
    assert predecessor_satisfied(6, committed=ens_committed, ensemble=True, expected_members=MEMBERS, candidate_so_far=())


def test_predecessor_blocks_uncommitted_gap_dependents() -> None:
    """A lead whose predecessor is neither committed nor in-candidate is blocked.

    Constructed via an artificial non-contiguous committed state (a defensive
    invariant: the contiguous walk normally satisfies this structurally).
    Lead 3 is already committed for both, but lead 6's predecessor chain is
    exercised by removing lead 3's GFS commitment while making upstream f003
    incomplete so it cannot re-enter the candidate.
    """
    gefs = gefs_snapshot({0: MEMBERS, 6: MEMBERS, 9: MEMBERS})
    result = plan(
        gfs_snapshot(complete=(0, 6, 9), data_only=(3,)),
        gefs,
        committed_gfs=committed((0, 6)),
        committed_gefs=gefs_committed((0, 6)),
    )
    # f003 is upstream-incomplete for GFS (data without idx): the walk blocks
    # there — later complete leads (9) are never jumped into.
    assert result.next_blocked_lead == 3
    assert result.pending_complete_leads == ()
    assert 9 not in result.wave_candidate


def test_canonical_horizon_filtering_excludes_upstream_only_leads() -> None:
    # Upstream GFS publishes hourly f001/f002; they are observed reality but
    # never platform targets.
    gfs = gfs_snapshot(complete=(0, 3), observed_only=(1, 2))
    result = plan(gfs, full_gefs((0, 3)))
    assert result.observed_frontier == 3
    assert 1 not in result.pending_complete_leads
    assert 2 not in result.pending_complete_leads
    assert result.pending_complete_leads == (0, 3)
    assert all(lead in HORIZON for lead in result.pending_complete_leads)


def test_exact_missing_member_diagnostics() -> None:
    gefs = gefs_snapshot({0: tuple(range(1, 31))})
    # Remove member 17's idx: members 1..16,18..30 complete → missing (17,).
    regions = dict(gefs.regions)
    regions[(17, 0)] = RegionArtifacts(
        data=_obs("gep17...f000"), idx=None
    )
    gefs_broken = CycleSnapshot(
        model="gefs",
        cycle_date=CYCLE,
        cycle_hour=0,
        prefix="gefs-prefix",
        regions=regions,
    )
    result = plan(gfs_snapshot(complete=(0,)), gefs_broken)
    assert result.missing_gefs_members == (17,)


def test_production_default_horizon_is_the_81_lead_contract() -> None:
    """Without an injected horizon, the planner uses the canonical contract."""
    from domain.horizon import register_canonical_lead_horizon

    # Re-register the production contract (the autouse fixture reduced it).
    production = tuple(range(0, 241, 3))
    register_canonical_lead_horizon("gfs", production)
    register_canonical_lead_horizon("gefs", production)
    result = plan_wave(
        cycle_label="c",
        gfs_snapshot=gfs_snapshot(),
        gefs_snapshot=gefs_snapshot(),
        committed_gfs=committed(),
        committed_gefs=committed(),
        policy=WavePolicy(),
    )
    assert result.horizon == production
    assert len(result.horizon) == 81


def test_expected_members_default_from_domain_contract() -> None:
    """The GEFS member contract comes from domain.coverage (gep01..gep30)."""
    from domain.coverage import get_expected_members

    result = plan(gfs_snapshot(complete=(0,)), gefs_snapshot({0: MEMBERS[:29]}))
    assert result.missing_gefs_members == (30,)
    assert len(MEMBERS) == get_expected_members("gefs")


# ---------------------------------------------------------------------------
# Bounded batching
# ---------------------------------------------------------------------------


def test_batching_count_threshold_triggers_truncated_wave() -> None:
    gfs = gfs_snapshot(complete=(0, 3, 6, 9, 12, 15))
    result = plan(
        gfs,
        full_gefs((0, 3, 6, 9, 12, 15)),
        max_leads=4,
        first_seen={lead: 0.0 for lead in (0, 3, 6, 9, 12, 15)},
        now=10.0,
    )
    assert result.pending_complete_leads == (0, 3, 6, 9, 12, 15)
    assert result.wave_due
    # Max-leads truncation: first four pending leads only.
    assert result.wave_candidate == (0, 3, 6, 9)


def test_batching_max_wait_threshold_triggers_wave() -> None:
    result = plan(
        gfs_snapshot(complete=(0, 3)),
        full_gefs((0, 3)),
        max_leads=8,
        max_wait=100.0,
        first_seen={0: 0.0, 3: 50.0},
        now=200.0,
    )
    assert result.pending_complete_leads == (0, 3)
    assert not (len(result.pending_complete_leads) >= 8)
    assert result.oldest_pending_age_seconds == 200.0
    assert result.wave_due  # oldest pending lead waited >= max_wait
    assert result.wave_candidate == (0, 3)


def test_batching_neither_threshold_holds_the_wave() -> None:
    result = plan(
        gfs_snapshot(complete=(0, 3)),
        full_gefs((0, 3)),
        max_leads=8,
        max_wait=1000.0,
        first_seen={0: 0.0, 3: 0.0},
        now=10.0,
    )
    assert result.pending_complete_leads == (0, 3)
    assert not result.wave_due
    assert result.wave_candidate == ()
    assert result.oldest_pending_age_seconds == 10.0


def test_batching_missing_timing_after_restart_resets_timers_only() -> None:
    """Restart loses wait timers (age 0) but not correctness: pending work is
    reconstructed from upstream + durable state and stays pending."""
    result = plan(
        gfs_snapshot(complete=(0, 3)),
        full_gefs((0, 3)),
        max_leads=8,
        max_wait=100.0,
        first_seen=None,  # scheduler memory lost
        now=99999.0,
    )
    assert result.pending_complete_leads == (0, 3)
    assert result.oldest_pending_age_seconds == 0.0
    assert not result.wave_due  # timers reset; count threshold still applies
    assert result.wave_candidate == ()
