"""Deterministic offline tests for the realtime scheduler (Phase 5C).

All external effects are faked (discovery, committed-state reads, wave
dispatch, leadership, clock, sleep) — no network, no real sleeps, no NOAA
dependency. The committed-state reader itself is exercised against SQLite in
``test_realtime_committed.py``.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timezone


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
MEMBERS = tuple(range(1, 31))


def _obs(key: str) -> ArtifactObservation:
    return ArtifactObservation(key=key, size=1, etag=None, last_modified=None)


def gfs_snapshot(complete: tuple[int, ...]) -> CycleSnapshot:
    regions = {
        (None, lead): RegionArtifacts(data=_obs(f"g{lead}"), idx=_obs(f"g{lead}i"))
        for lead in complete
    }
    return CycleSnapshot(
        model="gfs", cycle_date=CYCLE.cycle_date, cycle_hour=CYCLE.cycle_hour,
        prefix="p", regions=regions,
    )


def gefs_snapshot(complete: tuple[int, ...]) -> CycleSnapshot:
    regions = {
        (member, lead): RegionArtifacts(data=_obs(f"m{member}l{lead}"), idx=_obs(f"m{member}l{lead}i"))
        for lead in complete
        for member in MEMBERS
    }
    return CycleSnapshot(
        model="gefs", cycle_date=CYCLE.cycle_date, cycle_hour=CYCLE.cycle_hour,
        prefix="p", regions=regions,
    )


def _settings(**overrides) -> IngestionSettings:
    return IngestionSettings(
        REALTIME_ACTIVE_POLL_SECONDS=600.0,
        REALTIME_PUBLICATION_POLL_SECONDS=120.0,
        REALTIME_IDLE_BACKOFF_INITIAL_SECONDS=1800.0,
        REALTIME_IDLE_BACKOFF_MAX_SECONDS=3600.0,
        REALTIME_POLL_JITTER_FRACTION=0.10,
        REALTIME_WAVE_MAX_LEADS=1,
        REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0,
        **overrides,
    )


class FakeWorld:
    """Configurable fakes for discovery, committed state, and dispatch."""

    def __init__(self) -> None:
        self.snapshots: dict[str, tuple[CycleSnapshot, CycleSnapshot]] = {}
        self.discover_responses: list = []  # explicit queue if set
        self.committed_state: tuple[ModelCommittedState, ModelCommittedState] = (
            ModelCommittedState(),
            ModelCommittedState(),
        )
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
        return self.committed_state

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
        leadership=leadership,
        clock=lambda: 1000.0,
        sleep=_sleep,
        stop_event=stop_event,
        cycle_override=cycle_override,
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
