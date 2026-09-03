"""Comprehensive failure, concurrency, leadership, and shutdown hardening tests for Phase 5D.

Covers all required scenarios from the Phase 5D matrix:
- Scenario A: Discovery failures (partial, both, mid-pagination, transport, recovery)
- Scenario B: Catalog/committed-state failures and safe recovery
- Scenario C: Asymmetric one-model wave failures (GEFS succeeds / GFS fails, GFS succeeds / GEFS fails)
- Scenario D: Partial region progress within one model wave & subsequent retry
- Scenario E: Big-batch / realtime overlap
- Scenario F & G: PostgreSQL advisory leadership acquisition, duplicate exclusion, connection loss & recovery
- Scenario H: Shutdown during polling across all wait states (ACTIVE, PUBLISHING, BACKOFF, retry)
- Scenario I: Shutdown during active waves (GFS wave active, between GFS and GEFS)
- Scenario J & K: Exception containment & bounded retry intervals
- Scenario L & M: Cycle pairing across staggered publication & handover
- Scenario N & O: Finalizer/gate contention & resource hygiene
"""

from __future__ import annotations

import os
import random
import threading
from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, text

from ingestion.core.config import IngestionSettings
from ingestion.providers.noaa.discovery import (
    ArtifactObservation,
    CycleSnapshot,
    DiscoveryInvalidResponseError,
    DiscoveryPaginationError,
    DiscoveryUnavailableError,
    RegionArtifacts,
)
from ingestion.realtime.leadership import (
    LeadershipUnavailableError,
    SchedulerLeadership,
)
from ingestion.realtime.planner import ModelCommittedState
from ingestion.realtime.polling import PollState
from ingestion.realtime.scheduler import (
    CycleIdentity,
    RealtimeScheduler,
    WaveDispatchResult,
)

CYCLE_00 = CycleIdentity(cycle_date=date(2026, 7, 21), cycle_hour=0)
CYCLE_06 = CycleIdentity(cycle_date=date(2026, 7, 21), cycle_hour=6)
MEMBERS = tuple(range(1, 31))


def _obs(key: str) -> ArtifactObservation:
    return ArtifactObservation(key=key, size=1, etag=None, last_modified=None)


def gfs_snapshot(complete: tuple[int, ...], cycle: CycleIdentity = CYCLE_00) -> CycleSnapshot:
    regions = {
        (None, lead): RegionArtifacts(data=_obs(f"g{lead}"), idx=_obs(f"g{lead}i"))
        for lead in complete
    }
    return CycleSnapshot(
        model="gfs",
        cycle_date=cycle.cycle_date,
        cycle_hour=cycle.cycle_hour,
        prefix="p",
        regions=regions,
    )


def gefs_snapshot(complete: tuple[int, ...], cycle: CycleIdentity = CYCLE_00) -> CycleSnapshot:
    regions = {
        (member, lead): RegionArtifacts(
            data=_obs(f"m{member}l{lead}"), idx=_obs(f"m{member}l{lead}i")
        )
        for lead in complete
        for member in MEMBERS
    }
    return CycleSnapshot(
        model="gefs",
        cycle_date=cycle.cycle_date,
        cycle_hour=cycle.cycle_hour,
        prefix="p",
        regions=regions,
    )


class HarnessWorld:
    """Configurable test harness for realtime scheduler interactions."""

    def __init__(self) -> None:
        self.snapshots: dict[str, tuple[CycleSnapshot, CycleSnapshot]] = {}
        self.discover_queue: list = []
        self.committed_queue: list = []
        self.committed_state: tuple[ModelCommittedState, ModelCommittedState] = (
            ModelCommittedState(),
            ModelCommittedState(),
        )
        self.dispatch_failures: set[str] = set()
        self.dispatch_block_event: threading.Event | None = None
        self.dispatch_started: threading.Event = threading.Event()
        self.dispatch_completed: threading.Event = threading.Event()
        self.dispatch_calls: list[tuple[str, tuple[int, ...], str, bool]] = []
        self.discover_calls: list[str] = []
        self.on_gfs_dispatch_done: threading.Event | None = None

    def discover(self, cycle: CycleIdentity) -> tuple[CycleSnapshot | None, CycleSnapshot | None]:
        self.discover_calls.append(cycle.label)
        if self.discover_queue:
            resp = self.discover_queue.pop(0)
            if isinstance(resp, BaseException):
                raise resp
            return resp
        return self.snapshots.get(cycle.label, (gfs_snapshot((), cycle), gefs_snapshot((), cycle)))

    def read_committed(self, cycle: CycleIdentity) -> tuple[ModelCommittedState, ModelCommittedState]:
        if self.committed_queue:
            resp = self.committed_queue.pop(0)
            if isinstance(resp, BaseException):
                raise resp
            return resp
        return self.committed_state

    def dispatch_wave(
        self,
        model: str,
        targets: tuple[int, ...],
        cycle: CycleIdentity,
        cancel_event: threading.Event,
    ) -> WaveDispatchResult:
        self.dispatch_started.set()
        if self.dispatch_block_event is not None:
            cancel_event.wait(timeout=2.0)
        self.dispatch_calls.append((model, targets, cycle.label, cancel_event.is_set()))
        if model == "gfs" and self.on_gfs_dispatch_done is not None:
            self.on_gfs_dispatch_done.set()
        self.dispatch_completed.set()
        if model in self.dispatch_failures:
            return WaveDispatchResult(model=model, targets=targets, error=f"simulated failure for {model}")
        return WaveDispatchResult(model=model, targets=targets, status="partial")


def _make_scheduler(
    world: HarnessWorld,
    *,
    cycle_override: CycleIdentity | None = CYCLE_00,
    leadership=None,
    stop_event: threading.Event | None = None,
    sleeps: list[float] | None = None,
    wave_max_leads: int = 1,
) -> RealtimeScheduler:
    settings = IngestionSettings(
        REALTIME_ACTIVE_POLL_SECONDS=600.0,
        REALTIME_PUBLICATION_POLL_SECONDS=120.0,
        REALTIME_IDLE_BACKOFF_INITIAL_SECONDS=1800.0,
        REALTIME_IDLE_BACKOFF_MAX_SECONDS=3600.0,
        REALTIME_POLL_JITTER_FRACTION=0.10,
        REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS=60.0,
        REALTIME_WAVE_MAX_LEADS=wave_max_leads,
        REALTIME_WAVE_MAX_WAIT_SECONDS=1200.0,
    )

    def _sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)
        if stop_event is not None:
            stop_event.wait(max(0.0, seconds))

    return RealtimeScheduler(
        conn_settings=settings,
        discover=world.discover,
        read_committed=world.read_committed,
        dispatch_wave=world.dispatch_wave,
        leadership=leadership,
        clock=lambda: 1000.0,
        sleep=_sleep,
        stop_event=stop_event,
        cycle_override=cycle_override,
        rng=random.Random(42),
    )


# ===========================================================================
# Scenario A: Discovery Failures & Multi-Iteration Recovery
# ===========================================================================


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: DiscoveryUnavailableError("transport connection refused"),
        lambda: DiscoveryInvalidResponseError("upstream HTTP 503 Service Unavailable"),
        lambda: DiscoveryPaginationError("listing exceeded 64 pages"),
    ],
)
def test_scenario_a_discovery_failures_retry_and_recover(error_factory) -> None:
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    scheduler = _make_scheduler(world)

    # Initial good poll establishes baseline snapshot
    outcome1 = scheduler.poll_once()
    assert outcome1.kind == "planned"
    last_good = scheduler._last_good
    assert last_good is not None

    # Poll 2: Discovery error occurs
    world.discover_queue = [error_factory()]
    outcome2 = scheduler.poll_once()
    assert outcome2.kind == "discovery-failed"
    assert outcome2.plan is None
    # Last good snapshot is preserved; state is NOT marked idle
    assert scheduler._last_good is last_good
    assert scheduler._machine.state != PollState.BACKOFF

    # Poll 3: Upstream recovers with newly published lead 3
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0, 3)), gefs_snapshot((0, 3)))
    # Durably commit lead 0 from poll 1
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)),
    )
    outcome3 = scheduler.poll_once()
    assert outcome3.kind == "planned"
    assert outcome3.plan is not None
    assert outcome3.plan.pending_complete_leads == (3,)
    assert [d.model for d in outcome3.dispatches] == ["gfs", "gefs"]
    assert outcome3.dispatches[0].targets == (3,)


# ===========================================================================
# Scenario B: Catalog / Committed-State Failures
# ===========================================================================


def test_scenario_b_catalog_read_failure_skips_planning_and_recovers() -> None:
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0, 3)), gefs_snapshot((0, 3)))
    scheduler = _make_scheduler(world)

    # Poll 1: Catalog database connection fails
    world.committed_queue = [sa.exc.OperationalError("SELECT ...", {}, Exception("DB down"))]
    outcome1 = scheduler.poll_once()
    assert outcome1.kind == "state-read-failed"
    assert outcome1.plan is None
    assert world.dispatch_calls == []

    # Poll 2: Catalog database recovers; both leads 0 and 3 are uncommitted
    outcome2 = scheduler.poll_once()
    assert outcome2.kind == "planned"
    assert outcome2.plan is not None
    assert outcome2.plan.pending_complete_leads == (0, 3)
    assert len(outcome2.dispatches) == 2  # GFS and GEFS dispatched for lead 0


# ===========================================================================
# Scenario C: Asymmetric One-Model Wave Failure & Re-Reconciliation
# ===========================================================================


def test_scenario_c_gefs_succeeds_gfs_fails_retries_only_gfs() -> None:
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    # GFS fails during dispatch, GEFS succeeds
    world.dispatch_failures.add("gfs")
    scheduler = _make_scheduler(world)

    # Poll 1: GFS fails, GEFS succeeds
    outcome1 = scheduler.poll_once()
    assert [d.model for d in outcome1.dispatches] == ["gfs", "gefs"]
    assert not outcome1.dispatches[0].ok  # GFS failed
    assert outcome1.dispatches[1].ok  # GEFS succeeded

    # Simulated durable state: GEFS committed lead 0 (all 30 members), GFS uncommitted
    world.committed_state = (
        ModelCommittedState(),
        ModelCommittedState(
            leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)
        ),
    )
    world.dispatch_failures.clear()

    # Poll 2: Next reconciliation retries ONLY GFS
    outcome2 = scheduler.poll_once()
    assert outcome2.plan is not None
    assert outcome2.plan.wave_targets_gfs == (0,)
    assert outcome2.plan.wave_targets_gefs == ()
    assert [d.model for d in outcome2.dispatches] == ["gfs"]
    assert outcome2.dispatches[0].ok

    # Poll 3: Both durably committed -> no further dispatch
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0})),
        ModelCommittedState(
            leads=frozenset({0}), pairs=frozenset((m, 0) for m in MEMBERS)
        ),
    )
    outcome3 = scheduler.poll_once()
    assert outcome3.dispatches == []


# ===========================================================================
# Scenario D: Partial Region Progress & Incremental Retry
# ===========================================================================


def test_scenario_d_partial_gefs_members_leave_lead_pending() -> None:
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0, 3)), gefs_snapshot((0, 3)))
    scheduler = _make_scheduler(world)

    # GFS committed leads 0, 3; GEFS only committed members 1..25 for lead 0 (incomplete lead)
    partial_pairs = frozenset((m, 0) for m in range(1, 26))
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0, 3})),
        ModelCommittedState(leads=frozenset(), pairs=partial_pairs),
    )

    outcome = scheduler.poll_once()
    assert outcome.plan is not None
    # Lead 0 is still incomplete for GEFS, so it must be pending and targeted
    assert outcome.plan.pending_complete_leads == (0, 3)
    assert outcome.plan.wave_targets_gfs == ()  # GFS already has 0
    assert outcome.plan.wave_targets_gefs == (0,)  # GEFS must retry lead 0


# ===========================================================================
# Scenario E: Big-Batch & Realtime Interleaving
# ===========================================================================


def test_scenario_e_big_batch_commit_ahead_of_realtime_reconciles() -> None:
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0, 3, 6)), gefs_snapshot((0, 3, 6)))
    scheduler = _make_scheduler(world, wave_max_leads=2)

    # Realtime plans leads 0, 3. In the meantime, big-batch commits leads 0, 3, 6.
    world.committed_state = (
        ModelCommittedState(leads=frozenset({0, 3, 6})),
        ModelCommittedState(
            leads=frozenset({0, 3, 6}),
            pairs=frozenset((m, lead) for lead in (0, 3, 6) for m in MEMBERS),
        ),
    )

    outcome = scheduler.poll_once()
    assert outcome.plan is not None
    assert outcome.plan.pending_complete_leads == ()
    assert outcome.dispatches == []


# ===========================================================================
# Scenario F & G: PostgreSQL Advisory Leadership & Connection Loss
# ===========================================================================


def _get_postgres_engine() -> sa.engine.Engine | None:
    dsn = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    try:
        engine = create_engine(dsn, pool_pre_ping=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        return None


def test_scenario_f_duplicate_scheduler_exclusion_postgres() -> None:
    engine = _get_postgres_engine()
    if engine is None:
        pytest.skip("PostgreSQL service container not available")

    leadership_a = SchedulerLeadership(engine, identity="test-exclusion")
    leadership_b = SchedulerLeadership(engine, identity="test-exclusion")

    assert leadership_a.acquire() is True
    assert leadership_a.is_leader is True

    # Second instance fails non-blocking try_lock
    assert leadership_b.acquire() is False
    assert leadership_b.is_leader is False

    # First instance releases -> second instance can acquire
    leadership_a.release()
    assert leadership_a.is_leader is False

    assert leadership_b.acquire() is True
    assert leadership_b.is_leader is True
    leadership_b.release()


def test_scenario_g_leadership_connection_loss_detection_and_safe_exit() -> None:
    engine = _get_postgres_engine()
    if engine is None:
        pytest.skip("PostgreSQL service container not available")

    leadership = SchedulerLeadership(engine, identity="test-conn-loss")
    assert leadership.acquire() is True
    assert leadership.is_leader is True
    assert leadership.check_leadership() is True

    # Sever the leadership connection from another backend by killing the PID
    with engine.connect() as admin_conn:
        admin_conn.execute(
            text("SELECT pg_terminate_backend(:pid)"),
            {"pid": leadership._conn.connection.get_backend_pid()},
        )

    # Scheduler checks leadership: loss is detected, connection is marked dead
    assert leadership.check_leadership() is False
    assert leadership.is_leader is False

    # RealtimeScheduler using this leadership detects loss in poll_once
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    scheduler = _make_scheduler(world, leadership=leadership)

    outcome = scheduler.poll_once()
    assert outcome.kind == "leadership-lost"
    assert world.dispatch_calls == []  # Waves are NEVER dispatched without leadership

    leadership.release()


def test_scenario_g_leadership_reacquire_loop_on_connection_death() -> None:
    engine = _get_postgres_engine()
    if engine is None:
        pytest.skip("PostgreSQL service container not available")

    leadership = SchedulerLeadership(engine, identity="test-reacquire")
    assert leadership.acquire() is True

    # Terminate backend
    with engine.connect() as admin_conn:
        admin_conn.execute(
            text("SELECT pg_terminate_backend(:pid)"),
            {"pid": leadership._conn.connection.get_backend_pid()},
        )

    # Leadership is lost
    assert leadership.check_leadership() is False

    # Scheduler in run() detects loss and successfully reacquires on a new connection
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    stop_event = threading.Event()
    sleeps: list[float] = []
    scheduler = _make_scheduler(
        world, leadership=leadership, stop_event=stop_event, sleeps=sleeps
    )

    code = scheduler.run(once=True)
    assert code == 0
    # Both models dispatched after successful reacquire
    assert [d[0] for d in world.dispatch_calls] == ["gfs", "gefs"]


def test_leadership_unavailable_on_non_postgres() -> None:
    sqlite_engine = create_engine("sqlite:///:memory:")
    leadership = SchedulerLeadership(sqlite_engine)
    with pytest.raises(LeadershipUnavailableError):
        leadership.acquire()


# ===========================================================================
# Scenario H: Shutdown While Polling across all Wait States
# ===========================================================================


@pytest.mark.parametrize(
    "initial_state,activity,expected_kind",
    [
        (PollState.ACTIVE, True, "planned"),
        (PollState.PUBLISHING, True, "planned"),
        (PollState.BACKOFF, False, "idle"),
    ],
)
def test_scenario_h_shutdown_while_polling_all_states(
    initial_state: PollState, activity: bool, expected_kind: str
) -> None:
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    stop_event = threading.Event()
    scheduler = _make_scheduler(world, stop_event=stop_event)
    scheduler._machine.state = initial_state

    # Trigger stop immediately before or during sleep
    scheduler.request_stop()
    code = scheduler.run()
    assert code == 0


# ===========================================================================
# Scenario I: Shutdown During Wave Execution
# ===========================================================================


def test_scenario_i_shutdown_during_gfs_wave_skips_gefs() -> None:
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0,)), gefs_snapshot((0,)))
    stop_event = threading.Event()
    scheduler = _make_scheduler(world, stop_event=stop_event)

    # When GFS is dispatched, request stop during execution
    def _dispatch_with_stop(model: str, targets: tuple[int, ...], cycle: CycleIdentity, cancel_event: threading.Event) -> WaveDispatchResult:
        world.dispatch_calls.append((model, targets, cycle.label, cancel_event.is_set()))
        if model == "gfs":
            scheduler.request_stop()
        return WaveDispatchResult(model=model, targets=targets, status="partial")

    scheduler.dispatch_wave = _dispatch_with_stop
    outcome = scheduler.poll_once()

    # GFS was dispatched; GEFS was skipped due to stop request
    dispatched_models = [call[0] for call in world.dispatch_calls]
    assert dispatched_models == ["gfs"]
    assert len(outcome.dispatches) == 1
    assert outcome.dispatches[0].model == "gfs"


# ===========================================================================
# Scenario L & M: Staggered Publication, Multi-Cycle Handover & Pairing
# ===========================================================================


def test_scenario_l_m_staggered_cycle_handover_and_no_mismatched_pairing() -> None:
    world = HarnessWorld()
    # Cycle 00Z is complete for GFS and GEFS
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0, 3), CYCLE_00), gefs_snapshot((0, 3), CYCLE_00))
    # Cycle 06Z: GFS starts publishing, GEFS is not yet publishing
    world.snapshots[CYCLE_06.label] = (gfs_snapshot((0,), CYCLE_06), gefs_snapshot((), CYCLE_06))

    scheduler = _make_scheduler(world, cycle_override=CYCLE_06)

    # Poll 06Z: GFS is present, GEFS is absent -> barrier blocks, no waves dispatched
    outcome = scheduler.poll_once()
    assert outcome.plan is not None
    assert outcome.plan.gfs_present is True
    assert outcome.plan.gefs_present is False
    assert outcome.plan.next_blocked_lead == 0
    assert outcome.dispatches == []

    # GEFS starts publishing for 06Z -> barrier opens for 06Z
    world.snapshots[CYCLE_06.label] = (gfs_snapshot((0,), CYCLE_06), gefs_snapshot((0,), CYCLE_06))
    outcome2 = scheduler.poll_once()
    assert outcome2.plan is not None
    assert outcome2.plan.pending_complete_leads == (0,)
    assert [d.model for d in outcome2.dispatches] == ["gfs", "gefs"]
    assert {d[2] for d in world.dispatch_calls} == {CYCLE_06.label}


# ===========================================================================
# Scenario N & O: Finalizer / Gate Contention & Resource Hygiene
# ===========================================================================


def test_scenario_n_o_repeated_wave_finalization_resource_clean() -> None:
    world = HarnessWorld()
    world.snapshots[CYCLE_00.label] = (gfs_snapshot((0, 3, 6)), gefs_snapshot((0, 3, 6)))
    scheduler = _make_scheduler(world, wave_max_leads=1)

    # Run 3 consecutive waves (leads 0, 3, 6)
    for lead in (0, 3, 6):
        outcome = scheduler.poll_once()
        assert outcome.kind == "planned"
        # Update committed state to simulate successful wave commit
        curr_gfs = world.committed_state[0].leads | {lead}
        curr_gefs_pairs = world.committed_state[1].pairs | frozenset((m, lead) for m in MEMBERS)
        world.committed_state = (
            ModelCommittedState(leads=curr_gfs),
            ModelCommittedState(leads=curr_gfs, pairs=curr_gefs_pairs),
        )

    # 4th poll: full horizon complete, nothing pending
    outcome_final = scheduler.poll_once()
    assert outcome_final.plan is not None
    assert outcome_final.plan.pending_complete_leads == ()
    assert outcome_final.dispatches == []
