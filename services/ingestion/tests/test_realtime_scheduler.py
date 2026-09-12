"""Deterministic offline tests for the realtime scheduler (Phase 5C).

All external effects are faked (discovery, committed-state reads, wave
dispatch, leadership, clock, sleep) — no network, no real sleeps, no NOAA
dependency. The committed-state reader itself is exercised against SQLite in
``test_realtime_committed.py``.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone

import pytest


from ingestion.core.config import IngestionSettings
from ingestion.providers.noaa.discovery import (
    ArtifactObservation,
    CycleSnapshot,
    DiscoveryUnavailableError,
    RegionArtifacts,
)
from ingestion.realtime.planner import ModelCommittedState
from ingestion.realtime.scheduler import (
    CycleIdentity,
    RealtimeScheduler,
    WaveDispatchResult,
    newest_eligible_cycle,
)

CYCLE = CycleIdentity(cycle_date=date(2026, 7, 21), cycle_hour=0)


@pytest.fixture()
def _restore_horizons():
    from domain.horizon import MODEL_CANONICAL_HORIZONS, MODEL_VERSION_HORIZONS

    saved = dict(MODEL_CANONICAL_HORIZONS)
    saved_versions = dict(MODEL_VERSION_HORIZONS)
    yield
    MODEL_CANONICAL_HORIZONS.clear()
    MODEL_CANONICAL_HORIZONS.update(saved)
    MODEL_VERSION_HORIZONS.clear()
    MODEL_VERSION_HORIZONS.update(saved_versions)

MEMBERS = tuple(range(1, 31))


def _obs(key: str) -> ArtifactObservation:
    return ArtifactObservation(key=key, size=1, etag=None, last_modified=None)


def gfs_snapshot(complete: tuple[int, ...], cycle: CycleIdentity = CYCLE) -> CycleSnapshot:
    regions = {
        (None, lead): RegionArtifacts(data=_obs(f"g{lead}"), idx=_obs(f"g{lead}i"))
        for lead in complete
    }
    return CycleSnapshot(
        model="gfs", cycle_date=cycle.cycle_date, cycle_hour=cycle.cycle_hour,
        prefix="p", regions=regions,
    )


def gefs_snapshot(complete: tuple[int, ...], cycle: CycleIdentity = CYCLE) -> CycleSnapshot:
    regions = {
        (member, lead): RegionArtifacts(data=_obs(f"m{member}l{lead}"), idx=_obs(f"m{member}l{lead}i"))
        for lead in complete
        for member in MEMBERS
    }
    return CycleSnapshot(
        model="gefs", cycle_date=cycle.cycle_date, cycle_hour=cycle.cycle_hour,
        prefix="p", regions=regions,
    )


def _settings(**overrides) -> IngestionSettings:
    defaults: dict[str, object] = {
        "REALTIME_ACTIVE_POLL_SECONDS": 600.0,
        "REALTIME_PUBLICATION_POLL_SECONDS": 120.0,
        "REALTIME_IDLE_BACKOFF_INITIAL_SECONDS": 1800.0,
        "REALTIME_IDLE_BACKOFF_MAX_SECONDS": 3600.0,
        "REALTIME_POLL_JITTER_FRACTION": 0.10,
        "REALTIME_WAVE_MAX_LEADS": 1,
        "REALTIME_WAVE_MAX_WAIT_SECONDS": 1200.0,
    }
    defaults.update(overrides)
    return IngestionSettings(**defaults)


class FakeWorld:
    """Configurable fakes for discovery, committed state, and dispatch."""

    def __init__(self) -> None:
        self.clock_time: float = 1000.0
        self.snapshots: dict[str, tuple[CycleSnapshot, CycleSnapshot]] = {}
        self.discover_responses: list = []  # explicit queue if set
        self.committed_state: tuple[ModelCommittedState, ModelCommittedState] = (
            ModelCommittedState(),
            ModelCommittedState(),
        )
        self.committed_states: dict[str, tuple[ModelCommittedState, ModelCommittedState]] = {}
        self.candidates: list[CycleIdentity] = []
        self.complete_cycles: set[str] = set()
        self.dispatch_failures: set[str] = set()
        self.dispatch_block_on_cancel = False
        self.dispatch_started = threading.Event()
        self.dispatch_calls: list[tuple[str, tuple[int, ...], str, bool]] = []
        self.discover_calls: list[str] = []

    def discover(self, cycle: CycleIdentity):
        self.discover_calls.append(cycle.label)
        if self.discover_responses:
            response = self.discover_responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return self.snapshots.get(cycle.label, (gfs_snapshot(()), gefs_snapshot(())))

    def read_committed(self, cycle: CycleIdentity):
        if cycle.label in self.committed_states:
            return self.committed_states[cycle.label]
        return self.committed_state

    def discover_candidates(self, active_cycle, now_utc):
        return [c for c in self.candidates if active_cycle is None or c != active_cycle]

    def is_complete(self, cycle: CycleIdentity):
        return cycle.label in self.complete_cycles

    def dispatch_wave(
        self, model: str, targets: tuple[int, ...], cycle: CycleIdentity, cancel_event: threading.Event
    ) -> WaveDispatchResult:
        self.dispatch_started.set()
        if self.dispatch_block_on_cancel:
            # Block until shutdown requests the non-abandoning drain.
            cancel_event.wait(timeout=5.0)
        self.dispatch_calls.append((model, targets, cycle.label, cancel_event.is_set()))
        if model in self.dispatch_failures:
            return WaveDispatchResult(model=model, targets=targets, error="simulated failure")
        return WaveDispatchResult(model=model, targets=targets, status="partial")


def _scheduler(
    world: FakeWorld,
    *,
    settings: IngestionSettings | None = None,
    stop_event: threading.Event | None = None,
    sleeps: list[float] | None = None,
    cycle_override: CycleIdentity | None = CYCLE,
    leadership=None,
    version_string: str = "v1.0",
) -> RealtimeScheduler:
    rng_seed = {"rng": __import__("random").Random(42)}

    def _sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return RealtimeScheduler(
        conn_settings=settings or _settings(),
        discover=world.discover,
        read_committed=world.read_committed,
        dispatch_wave=world.dispatch_wave,
        discover_candidates=world.discover_candidates,
        is_complete=world.is_complete,
        leadership=leadership,
        clock=lambda: world.clock_time,
        sleep=_sleep,
        stop_event=stop_event,
        cycle_override=cycle_override,
        version_string=version_string,
        **rng_seed,
    )


# ---------------------------------------------------------------------------
# Discover → plan → dispatch → reconcile
# ---------------------------------------------------------------------------


def test_discover_plan_dispatch_shared_wave() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0, 3)), gefs_snapshot((0, 3)))
    scheduler = _scheduler(world)
    outcome = scheduler.poll_once()

    assert outcome.kind == "planned"
    assert outcome.plan is not None
    assert outcome.plan.pending_complete_leads == (0, 3)
    # max_leads=1 → the wave is the first pending lead only.
    assert [d.model for d in outcome.dispatches] == ["gfs", "gefs"]
    assert outcome.dispatches[0].targets == (0,)
    assert outcome.dispatches[1].targets == (0,)
    # Same cycle identity dispatched to both models.
    assert {d[2] for d in world.dispatch_calls} == {CYCLE.label}


def test_successful_commit_removes_work_from_next_reconciliation() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    scheduler = _scheduler(world)
    scheduler.poll_once()
    assert world.dispatch_calls

    # The wave committed durably for both models.
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(
            leads=frozenset({0}),
            pairs=frozenset((m, 0) for m in MEMBERS),
        ),
    )
    scheduler.poll_once()
    assert len(world.dispatch_calls) == 2  # only the first poll dispatched


def test_gfs_success_gefs_failure_retries_only_gefs() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    world.dispatch_failures.add("gefs")
    scheduler = _scheduler(world)
    outcome = scheduler.poll_once()

    assert [d.model for d in outcome.dispatches] == ["gfs", "gefs"]
    assert outcome.dispatches[0].ok
    assert not outcome.dispatches[1].ok
    # GFS committed (simulated durable result); GEFS failed.
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(),
    )
    # Advance clock past active failure backoff
    world.clock_time += 60.0
    outcome2 = scheduler.poll_once()
    # Next reconciliation retries ONLY the missing GEFS work.
    assert [d.model for d in outcome2.dispatches] == ["gefs"]
    assert outcome2.dispatches[0].targets == (0,)


def test_restart_reconciliation_plans_only_missing_work() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0, 3)), gefs_snapshot((0, 3)))
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(
            leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)
        ),
    )
    # A brand-new scheduler instance (restart): no timing map, no tracked cycle.
    scheduler = _scheduler(world)
    outcome = scheduler.poll_once()
    assert outcome.plan is not None
    assert outcome.plan.pending_complete_leads == (3,)
    assert outcome.dispatches[0].targets == (3,)


def test_big_batch_commit_between_polls_is_not_duplicated() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0, 3)), gefs_snapshot((0, 3)))
    scheduler = _scheduler(world)
    scheduler.poll_once()  # dispatches lead 0

    # Big-batch commits leads 0 AND 3 externally between polls.
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0, 3})),
        ModelCommittedState(
            leads=frozenset({0, 3}),
            pairs=frozenset((m, lead) for lead in (0, 3) for m in MEMBERS),
        ),
    )
    outcome = scheduler.poll_once()
    assert outcome.plan is not None
    assert outcome.plan.pending_complete_leads == ()
    assert outcome.dispatches == []


# ---------------------------------------------------------------------------
# Cycle selection
# ---------------------------------------------------------------------------


def test_explicit_cycle_mode_snapshots_both_models_for_one_identity() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot(()))
    scheduler = _scheduler(world)
    scheduler.poll_once()
    # One discover call per poll covers BOTH models with the same identity.
    assert world.discover_calls == [CYCLE.label]
    assert scheduler._tracked_cycle == CYCLE


def test_staggered_cycle_appearance_adopts_and_blocks_until_both_publish() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot(()))
    scheduler = _scheduler(world)
    first = scheduler.poll_once()
    assert first.plan is not None
    assert first.plan.gfs_present and not first.plan.gefs_present
    assert first.plan.next_blocked_lead == 0
    assert first.dispatches == []  # barrier blocked: GEFS absent

    # GEFS starts publishing later.
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    second = scheduler.poll_once()
    assert second.plan is not None
    assert second.plan.pending_complete_leads == (0,)
    assert second.dispatches  # wave emerges once the barrier completes


def test_newest_eligible_cycle_respects_publication_delay() -> None:
    now = datetime(2026, 7, 21, 2, 0, tzinfo=timezone.utc)  # 2h after 00Z
    identity = newest_eligible_cycle(now, first_publication_delay_seconds=10800.0)
    # 00Z + 3h > 02Z → the newest eligible cycle is yesterday's 18Z.
    assert (identity.cycle_date, identity.cycle_hour) == (date(2026, 7, 20), 18)

    now3 = datetime(2026, 7, 21, 3, 0, tzinfo=timezone.utc)
    identity3 = newest_eligible_cycle(now3, first_publication_delay_seconds=10800.0)
    assert (identity3.cycle_date, identity3.cycle_hour) == (date(2026, 7, 21), 0)


# ---------------------------------------------------------------------------
# Poll state, discovery failure semantics, jittered sleeps
# ---------------------------------------------------------------------------


def test_unchanged_success_is_idle_not_failure_and_backs_off() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    scheduler = _scheduler(world)
    first = scheduler.poll_once()
    assert first.activity  # first sight of the cycle is activity
    second = scheduler.poll_once()  # unchanged snapshot, nothing new
    assert not second.activity
    assert scheduler._machine.state.value == "backoff"


def test_member_growth_is_publication_activity() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    scheduler = _scheduler(world)
    scheduler.poll_once()

    # GEFS member count grows 30 → more observed regions (8/30 → 22/30 shape:
    # any snapshot change counts as activity).
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    gefs_more = gefs_snapshot((0,))
    extra_regions = dict(gefs_more.regions)
    extra_regions[(31, 3)] = RegionArtifacts(data=_obs("x"), idx=_obs("xi"))
    world.snapshots[CYCLE.label] = (
        gfs_snapshot((0,)),
        CycleSnapshot(model="gefs", cycle_date=CYCLE.cycle_date, cycle_hour=0, prefix="p", regions=extra_regions),
    )
    outcome = scheduler.poll_once()
    assert outcome.activity
    assert scheduler._machine.state.value == "publishing"


def test_discovery_failure_preserves_state_and_last_good_snapshot() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    scheduler = _scheduler(world)
    scheduler.poll_once()
    last_good = scheduler._last_good
    state_before = scheduler._machine.state.value

    world.discover_responses = [DiscoveryUnavailableError("upstream down")]
    outcome = scheduler.poll_once()
    assert outcome.kind == "discovery-failed"
    assert outcome.plan is None
    # Last good snapshot and poll state are preserved — failure is NOT idle.
    assert scheduler._last_good is last_good
    assert scheduler._machine.state.value == state_before


def test_committed_state_read_failure_skips_planning() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    calls = {"n": 0}

    def failing_read(cycle):
        calls["n"] += 1
        raise RuntimeError("catalog unavailable")

    scheduler = _scheduler(world)
    scheduler.read_committed = failing_read
    outcome = scheduler.poll_once()
    assert outcome.kind == "state-read-failed"
    assert outcome.plan is None
    assert world.dispatch_calls == []
    assert calls["n"] == 1


def test_sleep_intervals_are_jittered_within_bounds() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot(()), gefs_snapshot(()))
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    sleeps: list[float] = []
    scheduler = _scheduler(world, sleeps=sleeps)
    scheduler.run(once=True)  # one poll, no sleep in once mode

    # Drive the loop manually (run with once returns before sleeping). The
    # jittered interval always lies within base*(1±fraction) for whatever
    # state the machine is in (idle polls legitimately back off).
    for _ in range(5):
        scheduler.poll_once()
        machine = scheduler._machine
        interval = machine.next_interval(scheduler._rng)
        frac = machine.config.jitter_fraction
        assert machine.base_interval() * (1 - frac) <= interval <= (
            machine.base_interval() * (1 + frac)
        )
    del sleeps


# ---------------------------------------------------------------------------
# Leadership, shutdown, dry-run
# ---------------------------------------------------------------------------


def test_double_start_second_instance_exits_passively() -> None:
    world = FakeWorld()

    class BusyLeadership:
        is_leader = False

        def acquire(self) -> bool:
            return False  # another instance holds the advisory lock

        def release(self) -> None:
            pass

    scheduler = _scheduler(world, leadership=BusyLeadership())
    code = scheduler.run(once=True)
    assert code == 0
    assert world.dispatch_calls == []
    assert world.discover_calls == []


def test_leader_flag_exposed_in_diagnostics() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot(()), gefs_snapshot(()))

    class Leader:
        def __init__(self) -> None:
            self.is_leader = True

        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            self.is_leader = False

    scheduler = _scheduler(world, leadership=Leader())
    outcome = scheduler.poll_once()
    assert outcome.diagnostics["leader"] is True


def test_graceful_shutdown_while_sleeping_is_prompt() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot(()), gefs_snapshot(()))
    stop_event = threading.Event()
    scheduler = _scheduler(world, stop_event=stop_event)
    scheduler._cycle_override = None  # auto mode; nothing published → idle polls

    scheduler.discover_responses = []
    done = threading.Event()
    results: list[int] = []

    def _run() -> None:
        results.append(scheduler.run())
        done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    # The first poll completed; the scheduler is sleeping on the stop event.
    stop_event.set()
    scheduler.request_stop()
    assert done.wait(timeout=5.0)
    assert results == [0]


def test_shutdown_during_wave_triggers_non_abandoning_cancel() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    world.dispatch_block_on_cancel = True
    stop_event = threading.Event()
    scheduler = _scheduler(world, stop_event=stop_event)

    done = threading.Event()

    def _poll() -> None:
        scheduler.poll_once()
        done.set()

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    assert world.dispatch_started.wait(timeout=5.0)
    # Shutdown during the active wave: request_stop triggers the wave
    # runner's non-abandoning cancellation/drain via the external event.
    scheduler.request_stop()
    assert done.wait(timeout=5.0)
    assert any(cancelled for (_, _, _, cancelled) in world.dispatch_calls)


def test_once_with_dry_run_plans_without_dispatching() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    scheduler = _scheduler(world)
    outcome = scheduler.poll_once(dry_run=True)
    assert outcome.plan is not None
    assert outcome.plan.wave_due
    assert outcome.dispatches == []
    assert world.dispatch_calls == []


def test_run_once_returns_after_single_iteration() -> None:
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    scheduler = _scheduler(world)
    assert scheduler.run(once=True) == 0
    assert len(world.discover_calls) == 1
    assert len(world.dispatch_calls) == 2


# ---------------------------------------------------------------------------
# Data Lifecycle V3 Phase 2 — Backlog recovery & bounded dispatch tests
# ---------------------------------------------------------------------------

CYCLE_00 = CycleIdentity(cycle_date=date(2026, 7, 21), cycle_hour=0)
CYCLE_06 = CycleIdentity(cycle_date=date(2026, 7, 21), cycle_hour=6)
CYCLE_12 = CycleIdentity(cycle_date=date(2026, 7, 21), cycle_hour=12)
CYCLE_18 = CycleIdentity(cycle_date=date(2026, 7, 21), cycle_hour=18)


def test_original_abandonment_regression_cycle_resumes_from_committed_frontier() -> None:
    """Cycle N (06Z) was interrupted at L21.

    When N+1 (12Z) appears and is active, N remains a recoverable candidate.
    While 12Z waits on upstream publication, 06Z dispatches L24.. and resumes
    progress towards L240 without restarting from L0.
    """
    world = FakeWorld()
    # 15:30 UTC -> 12Z is newest eligible cycle
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # Active cycle 12Z has only published lead 0 (shared wave not due under max_leads=4)
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0,), CYCLE_12),
        gefs_snapshot((0,), CYCLE_12),
    )

    # Incomplete historical candidate 06Z has leads 0..21 committed
    committed_06_leads = frozenset(range(0, 24, 3))  # 0, 3, 6, 9, 12, 15, 18, 21
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=committed_06_leads),
        ModelCommittedState(
            leads=committed_06_leads,
            pairs=frozenset((m, lead) for lead in committed_06_leads for m in MEMBERS),
        ),
    )
    # 06Z has all leads 0..48 published upstream
    leads_06_upstream = tuple(range(0, 51, 3))
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot(leads_06_upstream, CYCLE_06),
        gefs_snapshot(leads_06_upstream, CYCLE_06),
    )
    world.candidates = [CYCLE_06]

    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    outcome = scheduler.poll_once()

    # Priority 2: 12Z is waiting, so 06Z backlog wave is dispatched!
    assert outcome.kind == "planned"
    assert outcome.cycle == CYCLE_06
    assert [d.model for d in outcome.dispatches] == ["gfs", "gefs"]
    # Verify targets start at L24 (resuming from committed L21), NOT restarting at L0!
    assert outcome.dispatches[0].targets == (24, 27, 30, 33)
    assert outcome.dispatches[1].targets == (24, 27, 30, 33)
    assert {d[2] for d in world.dispatch_calls} == {CYCLE_06.label}


def test_multiple_historical_incomplete_candidates_bounded_dispatch() -> None:
    """Catalog contains multiple partial cycles (06Z, 00Z).

    Runtime dispatch remains bounded: active 12Z + at most one selected backlog (06Z).
    00Z remains in persistent candidate state and rotates in once 06Z completes.
    """
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # 12Z active has only L0 (no wave due)
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0,), CYCLE_12),
        gefs_snapshot((0,), CYCLE_12),
    )

    # 06Z has L0 committed; L3 ready upstream
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((0, 3), CYCLE_06),
        gefs_snapshot((0, 3), CYCLE_06),
    )

    # 00Z has L0 committed; L3 ready upstream
    world.committed_states[CYCLE_00.label] = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    world.snapshots[CYCLE_00.label] = (
        gfs_snapshot((0, 3), CYCLE_00),
        gefs_snapshot((0, 3), CYCLE_00),
    )

    world.candidates = [CYCLE_06, CYCLE_00]
    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    # First poll: 06Z (newest incomplete) is selected as backlog
    outcome = scheduler.poll_once()
    assert outcome.cycle == CYCLE_06
    assert [d.model for d in outcome.dispatches] == ["gfs", "gefs"]
    assert outcome.dispatches[0].targets == (3,)

    # Mark 06Z complete
    world.complete_cycles.add(CYCLE_06.label)

    # Second poll: 00Z rotates in as selected backlog cycle!
    outcome2 = scheduler.poll_once()
    assert outcome2.cycle == CYCLE_00
    assert outcome2.dispatches[0].targets == (3,)


def test_backlog_rotation_on_upstream_block() -> None:
    """Preferred backlog candidate 06Z is blocked upstream (missing lead 3).

    Scheduler puts 06Z in temporary backoff and rotates to 00Z, which has ready work.
    """
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # Active 12Z has no wave due
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0,), CYCLE_12),
        gefs_snapshot((0,), CYCLE_12),
    )

    # 06Z is committed at L0, but upstream only has L0 (missing L3)
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((0,), CYCLE_06),
        gefs_snapshot((0,), CYCLE_06),
    )

    # 00Z is committed at L0, and upstream has L0, L3
    world.committed_states[CYCLE_00.label] = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    world.snapshots[CYCLE_00.label] = (
        gfs_snapshot((0, 3), CYCLE_00),
        gefs_snapshot((0, 3), CYCLE_00),
    )

    world.candidates = [CYCLE_06, CYCLE_00]
    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    outcome = scheduler.poll_once()
    # Rotates past blocked 06Z and dispatches 00Z!
    assert outcome.cycle == CYCLE_00
    assert outcome.dispatches[0].targets == (3,)
    # 06Z is in temporary backoff
    assert CYCLE_06.label in scheduler._backlog_blocked_until


def test_active_strict_priority_over_backlog() -> None:
    """Both active cycle (12Z) and backlog candidate (06Z) have ready waves due.

    Active cycle strictly takes Priority 1; backlog does not dispatch this iteration.
    """
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # Active 12Z has wave ready (max_leads=1)
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0,), CYCLE_12),
        gefs_snapshot((0,), CYCLE_12),
    )

    # Backlog 06Z also has wave ready
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((3,), CYCLE_06),
        gefs_snapshot((3,), CYCLE_06),
    )
    world.candidates = [CYCLE_06]

    scheduler = _scheduler(world, cycle_override=None)
    outcome = scheduler.poll_once()

    # Priority 1: Active 12Z wins
    assert outcome.cycle == CYCLE_12
    assert outcome.dispatches[0].targets == (0,)


def test_active_idle_backlog_catchup() -> None:
    """Active cycle 12Z is waiting on upstream publication.

    Backlog candidate 06Z dispatches immediately without artificial delay.
    """
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # Active 12Z has nothing published yet (empty snapshots -> waiting)
    world.snapshots[CYCLE_12.label] = (gfs_snapshot(()), gefs_snapshot(()))

    # Backlog 06Z has leads 0, 3 ready
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((0, 3), CYCLE_06),
        gefs_snapshot((0, 3), CYCLE_06),
    )
    world.candidates = [CYCLE_06]

    scheduler = _scheduler(world, cycle_override=None)
    outcome = scheduler.poll_once()

    assert outcome.cycle == CYCLE_06
    assert outcome.dispatches[0].targets == (0,)


def test_backlog_failure_isolation() -> None:
    """Backlog wave dispatch raises an exception.

    Active cycle remains healthy and unaffected; backlog enters backoff.
    """
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # Active 12Z has no wave due (1 lead < 4)
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0,), CYCLE_12),
        gefs_snapshot((0,), CYCLE_12),
    )

    # Backlog 06Z has lead 0 committed, lead 3 ready, but simulated failure is injected
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((0, 3), CYCLE_06),
        gefs_snapshot((0, 3), CYCLE_06),
    )
    world.dispatch_failures.add("gfs")
    world.candidates = [CYCLE_06]

    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)
    outcome = scheduler.poll_once()

    assert outcome.cycle == CYCLE_06
    assert not outcome.dispatches[0].ok
    # Backlog failure recorded and backoff active
    assert scheduler._backlog_failures[CYCLE_06.label] == 1
    assert scheduler._backlog_blocked_until[CYCLE_06.label] > world.clock_time

    # Next poll: active cycle 12Z publishes full batch of 4 leads and is ready to dispatch
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0, 3, 6, 9), CYCLE_12),
        gefs_snapshot((0, 3, 6, 9), CYCLE_12),
    )
    world.dispatch_failures.clear()
    outcome2 = scheduler.poll_once()

    # Active cycle dispatches normally
    assert outcome2.cycle == CYCLE_12
    assert outcome2.dispatches[0].ok


def test_backlog_capped_retry_never_permanently_abandoned() -> None:
    """Repeated failures on backlog candidate clamp at max_backoff without dropping candidate."""
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()
    world.snapshots[CYCLE_12.label] = (gfs_snapshot((0,), CYCLE_12), gefs_snapshot((0,), CYCLE_12))
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    world.snapshots[CYCLE_06.label] = (gfs_snapshot((0, 3), CYCLE_06), gefs_snapshot((0, 3), CYCLE_06))
    world.dispatch_failures.add("gfs")
    world.candidates = [CYCLE_06]

    settings = _settings(
        REALTIME_WAVE_MAX_LEADS=4,
        REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0,
        REALTIME_BACKLOG_RETRY_BACKOFF_SECONDS=10.0,
        REALTIME_BACKLOG_MAX_BACKOFF_SECONDS=50.0,
    )
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    # Trigger 6 consecutive failures
    for i in range(1, 7):
        outcome = scheduler.poll_once()
        assert outcome.cycle == CYCLE_06
        assert scheduler._backlog_failures[CYCLE_06.label] == i
        # Advance clock to after the backoff
        world.clock_time = scheduler._backlog_blocked_until[CYCLE_06.label] + 1.0

    # Backoff is clamped at max_backoff (50s), not exponentially unbounded
    backoff_duration = scheduler._backlog_blocked_until[CYCLE_06.label] - (world.clock_time - 1.0)
    assert backoff_duration <= 50.0
    # Candidate remains in candidate pool (never permanently dropped)
    assert CYCLE_06 in world.candidates


def test_active_failure_fairness_allows_backlog_progress() -> None:
    """Active cycle wave dispatch fails and enters failure backoff.

    While active is in backoff, backlog candidate progresses instead of being starved.
    When active backoff elapses, active regains Priority 1.
    """
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # Active 12Z has wave ready (4 leads for max_leads=4)
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0, 3, 6, 9), CYCLE_12),
        gefs_snapshot((0, 3, 6, 9), CYCLE_12),
    )

    # Backlog 06Z has wave ready (lead 3)
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((0, 3), CYCLE_06),
        gefs_snapshot((0, 3), CYCLE_06),
    )
    world.candidates = [CYCLE_06]

    settings = _settings(
        REALTIME_WAVE_MAX_LEADS=4,
        REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0,
        REALTIME_ACTIVE_FAILURE_BACKOFF_SECONDS=60.0,
        REALTIME_ACTIVE_MAX_BACKOFF_SECONDS=300.0,
    )
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    # Iteration 1: Active 12Z dispatches and fails
    world.dispatch_failures.add("gfs")
    outcome1 = scheduler.poll_once()
    assert outcome1.cycle == CYCLE_12
    assert not outcome1.dispatches[0].ok
    assert scheduler._active_failures == 1
    assert scheduler._active_blocked_until == world.clock_time + 60.0

    # Iteration 2: 10s later (still in active backoff)
    world.clock_time += 10.0
    world.dispatch_failures.clear()  # Backlog can succeed
    outcome2 = scheduler.poll_once()

    # Active yields; Backlog 06Z dispatches!
    assert outcome2.cycle == CYCLE_06
    assert outcome2.dispatches[0].ok
    assert outcome2.dispatches[0].targets == (3,)

    # Iteration 3: 60s later (active backoff expired)
    world.clock_time += 60.0
    outcome3 = scheduler.poll_once()

    # Active 12Z regains Priority 1!
    assert outcome3.cycle == CYCLE_12
    assert outcome3.dispatches[0].ok
    assert outcome3.dispatches[0].targets == (0, 3, 6, 9)


def test_restart_reconstruction_from_catalog() -> None:
    """Fresh scheduler startup reconstructs active + historical candidate state from catalog."""
    world = FakeWorld()
    # 21:30 UTC -> 18Z is newest eligible cycle
    world.clock_time = datetime(2026, 7, 21, 21, 30, tzinfo=timezone.utc).timestamp()

    # 18Z active has only L0 (no wave due with max_leads=2)
    world.snapshots[CYCLE_18.label] = (
        gfs_snapshot((0,), CYCLE_18),
        gefs_snapshot((0,), CYCLE_18),
    )

    # Catalog state before restart:
    # 12Z committed to L21
    leads_12 = frozenset(range(0, 24, 3))
    world.committed_states[CYCLE_12.label] = (
        ModelCommittedState(leads=leads_12),
        ModelCommittedState(leads=leads_12, pairs=frozenset((m, lead) for lead in leads_12 for m in MEMBERS)),
    )
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot(tuple(range(0, 51, 3)), CYCLE_12),
        gefs_snapshot(tuple(range(0, 51, 3)), CYCLE_12),
    )

    # 06Z committed to L48
    leads_06 = frozenset(range(0, 51, 3))
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=leads_06),
        ModelCommittedState(leads=leads_06, pairs=frozenset((m, lead) for lead in leads_06 for m in MEMBERS)),
    )
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot(tuple(range(0, 80, 3)), CYCLE_06),
        gefs_snapshot(tuple(range(0, 80, 3)), CYCLE_06),
    )

    world.candidates = [CYCLE_12, CYCLE_06]

    # Fresh scheduler instance with no prior memory
    settings = _settings(REALTIME_WAVE_MAX_LEADS=2, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    outcome = scheduler.poll_once()

    # Reconstructs active 18Z, selects 12Z (newest backlog), dispatches L24..L27
    assert outcome.cycle == CYCLE_12
    assert outcome.dispatches[0].targets == (24, 27)


def test_predecessor_safe_backlog_recovery() -> None:
    """Predecessor check passes for 6h-reset lead L24 because L21 is durably committed."""
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()
    world.snapshots[CYCLE_12.label] = (gfs_snapshot((0,), CYCLE_12), gefs_snapshot((0,), CYCLE_12))

    # 06Z has L0..L21 committed
    leads_21 = frozenset(range(0, 24, 3))
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=leads_21),
        ModelCommittedState(leads=leads_21, pairs=frozenset((m, lead) for lead in leads_21 for m in MEMBERS)),
    )
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((0, 3, 6, 9, 12, 15, 18, 21, 24), CYCLE_06),
        gefs_snapshot((0, 3, 6, 9, 12, 15, 18, 21, 24), CYCLE_06),
    )
    world.candidates = [CYCLE_06]

    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    outcome = scheduler.poll_once()
    assert outcome.cycle == CYCLE_06
    # Lead 24 (6h reset lead requiring predecessor L21) is safely planned and dispatched!
    assert outcome.dispatches[0].targets == (24,)


def test_active_cycle_transition_resets_active_failure_backoff() -> None:
    """Transitioning to a new active cycle clears active failure backoff immediately."""
    world = FakeWorld()
    # Start at 09:30 UTC -> 06Z active
    world.clock_time = datetime(2026, 7, 21, 9, 30, tzinfo=timezone.utc).timestamp()
    world.snapshots[CYCLE_06.label] = (gfs_snapshot((0,), CYCLE_06), gefs_snapshot((0,), CYCLE_06))
    world.dispatch_failures.add("gfs")

    scheduler = _scheduler(world, cycle_override=None)
    scheduler.poll_once()
    assert scheduler._active_failures == 1
    assert scheduler._active_blocked_until > world.clock_time

    # Advance time to 15:30 UTC -> 12Z becomes active!
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()
    world.snapshots[CYCLE_12.label] = (gfs_snapshot((0,), CYCLE_12), gefs_snapshot((0,), CYCLE_12))
    world.dispatch_failures.clear()

    outcome = scheduler.poll_once()
    # 12Z is adopted, active failures reset to 0, wave dispatches immediately
    assert outcome.cycle == CYCLE_12
    assert scheduler._active_failures == 0
    assert scheduler._active_blocked_until == 0.0
    assert outcome.dispatches[0].ok


def test_no_unnecessary_duplicate_ingestion_of_committed_leads() -> None:
    """Leads already committed are deducted from wave targets."""
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()
    world.snapshots[CYCLE_12.label] = (gfs_snapshot((0,), CYCLE_12), gefs_snapshot((0,), CYCLE_12))

    # 06Z has leads 0, 3 committed
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=frozenset({0, 3})),
        ModelCommittedState(leads=frozenset({0, 3}), pairs=frozenset((m, lead) for lead in (0, 3) for m in MEMBERS)),
    )
    # Upstream has 0, 3, 6, 9
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((0, 3, 6, 9), CYCLE_06),
        gefs_snapshot((0, 3, 6, 9), CYCLE_06),
    )
    world.candidates = [CYCLE_06]

    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    outcome = scheduler.poll_once()
    assert outcome.cycle == CYCLE_06
    # Only missing leads 6, 9 are targeted; 0, 3 are not re-ingested
    assert outcome.dispatches[0].targets == (6, 9)
    assert outcome.dispatches[1].targets == (6, 9)


def test_backlog_snapshot_reprobe_after_blocked_backoff_expires() -> None:
    """When a backlog candidate was blocked upstream on L24, backoff expiration triggers

    a fresh discovery/snapshot rather than indefinitely reusing the stale cached snapshot.
    """
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()
    world.snapshots[CYCLE_12.label] = (gfs_snapshot((0,), CYCLE_12), gefs_snapshot((0,), CYCLE_12))

    # 06Z committed through L21
    leads_21 = frozenset(range(0, 24, 3))
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=leads_21),
        ModelCommittedState(leads=leads_21, pairs=frozenset((m, lead) for lead in leads_21 for m in MEMBERS)),
    )
    # Initial upstream snapshot does NOT contain complete L24 (only up to L21)
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot(tuple(range(0, 24, 3)), CYCLE_06),
        gefs_snapshot(tuple(range(0, 24, 3)), CYCLE_06),
    )
    world.candidates = [CYCLE_06]

    settings = _settings(
        REALTIME_WAVE_MAX_LEADS=4,
        REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0,
        REALTIME_BACKLOG_RETRY_BACKOFF_SECONDS=300.0,
    )
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    # Poll 1: 06Z is blocked upstream on L24 -> enters backoff (300s)
    outcome1 = scheduler.poll_once()
    assert outcome1.cycle == CYCLE_12  # Active cycle outcome (no wave dispatched)
    assert outcome1.dispatches == []
    assert CYCLE_06.label in scheduler._backlog_blocked_until

    # Later upstream publishes complete L24
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot(tuple(range(0, 27, 3)), CYCLE_06),
        gefs_snapshot(tuple(range(0, 27, 3)), CYCLE_06),
    )

    # Advance clock past backoff
    world.clock_time += 301.0

    # Poll 2: Fresh discovery probes upstream, observes L24, predecessor L21 holds -> dispatches L24!
    outcome2 = scheduler.poll_once()
    assert outcome2.cycle == CYCLE_06
    assert [d.model for d in outcome2.dispatches] == ["gfs", "gefs"]
    assert outcome2.dispatches[0].targets == (24,)
    assert outcome2.dispatches[1].targets == (24,)


def test_gefs_success_gfs_failure_retries_only_gfs() -> None:
    """When GEFS succeeds and GFS fails, the retry dispatches ONLY the missing GFS work."""
    world = FakeWorld()
    world.snapshots[CYCLE.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    world.dispatch_failures.add("gfs")
    scheduler = _scheduler(world)
    outcome = scheduler.poll_once()

    assert [d.model for d in outcome.dispatches] == ["gfs", "gefs"]
    assert not outcome.dispatches[0].ok  # GFS failed
    assert outcome.dispatches[1].ok      # GEFS succeeded

    # GEFS committed (simulated durable catalog result); GFS not committed
    world.committed_state = (
        ModelCommittedState(),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    world.clock_time += 60.0
    world.dispatch_failures.clear()

    outcome2 = scheduler.poll_once()
    # Next reconciliation retries ONLY missing GFS work
    assert [d.model for d in outcome2.dispatches] == ["gfs"]
    assert outcome2.dispatches[0].targets == (0,)


def test_process_restart_with_instance_discard_reconstructs_correctly() -> None:
    """Simulate discarding scheduler instance A and launching fresh scheduler instance B."""
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 21, 30, tzinfo=timezone.utc).timestamp()

    world.snapshots[CYCLE_18.label] = (gfs_snapshot((0,), CYCLE_18), gefs_snapshot((0,), CYCLE_18))

    leads_12 = frozenset(range(0, 24, 3))
    world.committed_states[CYCLE_12.label] = (
        ModelCommittedState(leads=leads_12),
        ModelCommittedState(leads=leads_12, pairs=frozenset((m, lead) for lead in leads_12 for m in MEMBERS)),
    )
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot(tuple(range(0, 51, 3)), CYCLE_12),
        gefs_snapshot(tuple(range(0, 51, 3)), CYCLE_12),
    )

    leads_06 = frozenset(range(0, 51, 3))
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=leads_06),
        ModelCommittedState(leads=leads_06, pairs=frozenset((m, lead) for lead in leads_06 for m in MEMBERS)),
    )
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot(tuple(range(0, 80, 3)), CYCLE_06),
        gefs_snapshot(tuple(range(0, 80, 3)), CYCLE_06),
    )

    world.candidates = [CYCLE_12, CYCLE_06]

    settings = _settings(REALTIME_WAVE_MAX_LEADS=2, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)

    # Instance A runs
    scheduler_a = _scheduler(world, settings=settings, cycle_override=None)
    outcome_a = scheduler_a.poll_once()
    assert outcome_a.cycle == CYCLE_12

    # Discard Instance A completely
    del scheduler_a

    # Instance B starts with completely fresh in-memory state
    scheduler_b = _scheduler(world, settings=settings, cycle_override=None)
    assert scheduler_b._active_cycle is None
    assert scheduler_b._selected_backlog_cycle is None
    assert scheduler_b._cached_candidates is None

    outcome_b = scheduler_b.poll_once()
    # Reconstructs active 18Z, selects 12Z backlog from catalog, targets L24..L27
    assert outcome_b.cycle == CYCLE_12
    assert outcome_b.dispatches[0].targets == (24, 27)


def test_healthy_path_performance_isolation() -> None:
    """During healthy active dispatch, zero backlog discovery, planning, or DB scans occur."""
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # Active 12Z has 4 leads ready (full wave due under max_leads=4)
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0, 3, 6, 9), CYCLE_12),
        gefs_snapshot((0, 3, 6, 9), CYCLE_12),
    )

    # Backlog candidates exist in catalog
    world.candidates = [CYCLE_06, CYCLE_00]
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot((0, 3), CYCLE_06),
        gefs_snapshot((0, 3), CYCLE_06),
    )

    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    outcome = scheduler.poll_once()

    # 1. Active wave is dispatched
    assert outcome.cycle == CYCLE_12
    assert len(outcome.dispatches) == 2
    # 2. Backlog discovery was NEVER called: only active 12Z was probed!
    assert world.discover_calls == [CYCLE_12.label]
    # 3. Only 1 wave (active wave) dispatched in total
    assert len(world.dispatch_calls) == 2
    assert {d[2] for d in world.dispatch_calls} == {CYCLE_12.label}


def test_no_loss_of_existing_historical_horizon_state() -> None:
    """Cycle N has durable committed leads through L180.

    When N+1 begins publishing and becomes active, N's committed state is untouched,
    no L0..L180 leads are re-ingested, and catch-up resumes strictly after L180.
    """
    world = FakeWorld()
    world.clock_time = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).timestamp()

    # Active 12Z waiting (only L0)
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0,), CYCLE_12),
        gefs_snapshot((0,), CYCLE_12),
    )

    # 06Z committed all leads through L180!
    committed_180_leads = frozenset(range(0, 183, 3))
    world.committed_states[CYCLE_06.label] = (
        ModelCommittedState(leads=committed_180_leads),
        ModelCommittedState(
            leads=committed_180_leads,
            pairs=frozenset((m, lead) for lead in committed_180_leads for m in MEMBERS),
        ),
    )
    # Upstream has all leads through L192
    upstream_leads = tuple(range(0, 195, 3))
    world.snapshots[CYCLE_06.label] = (
        gfs_snapshot(upstream_leads, CYCLE_06),
        gefs_snapshot(upstream_leads, CYCLE_06),
    )
    world.candidates = [CYCLE_06]

    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)
    scheduler = _scheduler(world, settings=settings, cycle_override=None)

    outcome = scheduler.poll_once()

    assert outcome.cycle == CYCLE_06
    # Targets resume strictly at L183 (committed frontier L180 + 3), no duplicate L0..L180!
    assert outcome.dispatches[0].targets == (183, 186, 189, 192)
    assert outcome.dispatches[1].targets == (183, 186, 189, 192)


def test_backlog_horizon_expiry_uses_configured_version(_restore_horizons) -> None:
    """Verify that RealtimeScheduler backlog cycle horizon expiry uses the configured version.

    GFS v1.0 / GEFS v1.0: 240h
    GFS v2.0 / GEFS v2.0: 300h

    For candidate at serving_start - 270h:
    Under v1.0 -> candidate.cycle_time + 240h < serving_start -> expired, skipped.
    Under v2.0 -> candidate.cycle_time + 300h >= serving_start -> not expired, dispatched.
    """
    from domain.horizon import register_canonical_lead_horizon
    from domain.temporal import serving_start_valid_time

    register_canonical_lead_horizon("gfs", tuple(range(0, 301, 3)), version_string="v2.0")
    register_canonical_lead_horizon("gefs", tuple(range(0, 301, 3)), version_string="v2.0")

    world = FakeWorld()
    # 15:30 UTC -> 12Z is newest eligible cycle
    now_dt = datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc)
    world.clock_time = now_dt.timestamp()

    # Active cycle: 12Z (only lead 0 published -> no wave due under max_leads=4)
    world.snapshots[CYCLE_12.label] = (
        gfs_snapshot((0,), CYCLE_12),
        gefs_snapshot((0,), CYCLE_12),
    )

    # Backlog candidate 270h old (e.g. 270h before serving_start)
    c_270h_dt = serving_start_valid_time(now_dt) - timedelta(hours=270)
    c_backlog = CycleIdentity(cycle_date=c_270h_dt.date(), cycle_hour=c_270h_dt.hour)

    world.snapshots[c_backlog.label] = (
        gfs_snapshot((0, 3, 6, 9), c_backlog),
        gefs_snapshot((0, 3, 6, 9), c_backlog),
    )
    world.candidates = [c_backlog]

    settings = _settings(REALTIME_WAVE_MAX_LEADS=4, REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0)

    # 1. Under v1.0 (240h horizon): 270h candidate is expired -> NOT dispatched (outcome is active cycle)
    scheduler_v1 = _scheduler(
        world, settings=settings, cycle_override=None, version_string="v1.0"
    )
    outcome_v1 = scheduler_v1.poll_once()
    assert outcome_v1.cycle == CYCLE_12
    assert outcome_v1.dispatches == []

    # 2. Under v2.0 (300h horizon): 270h candidate is NOT expired -> backlog wave dispatched!
    scheduler_v2 = _scheduler(
        world, settings=settings, cycle_override=None, version_string="v2.0"
    )
    outcome_v2 = scheduler_v2.poll_once()
    assert outcome_v2.cycle == c_backlog
    assert len(outcome_v2.dispatches) == 2




