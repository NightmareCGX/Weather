"""Unit tests for the realtime scheduler watchdog (2026-10-28 remediation).

The watchdog must detect a wedged scheduler loop from an independent thread,
alert through the platform's dispatch path, and — past the exit threshold —
terminate the process so the container restart policy can recover it. Data
staleness, by contrast, must stay alert-only (upstream publication delays are
legitimate). All checks run synchronously via ``check_once`` with injected
clock, exit function, and alert engine, so no thread timing is involved.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ingestion.monitoring.alerts import AlertSeverity
from ingestion.monitoring.ingestion_collector import INGESTION_COLLECTOR
from ingestion.realtime.watchdog import RealtimeWatchdog


class _FakeScheduler:
    """Minimal scheduler double: only the watchdog's surface."""

    def __init__(self, last_poll_success_at: float | None = None) -> None:
        self.last_poll_success_at = last_poll_success_at


class _RecordingEngine:
    """Records every dispatch_alerts payload (recovery = empty list)."""

    def __init__(self) -> None:
        self.dispatched: list[list[Any]] = []

    def dispatch_alerts(self, current_alerts: list[Any]) -> list[Any]:
        self.dispatched.append(list(current_alerts))
        return []


class _RecordingExit:
    """Records the exit code instead of terminating the process."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, code: int) -> None:
        self.calls.append(code)


@pytest.fixture
def collector_states_snapshot(monkeypatch):
    """Snapshot/restore the global collector's per-model state dict."""
    original = dict(INGESTION_COLLECTOR._states)
    yield INGESTION_COLLECTOR._states
    monkeypatch.setattr(INGESTION_COLLECTOR, "_states", original, raising=False)


def _make_watchdog(
    scheduler,
    monkeypatch,
    *,
    stall_warn: float = 100.0,
    stall_exit: float = 0.0,
    stale_warn: float = 1000.0,
) -> tuple[RealtimeWatchdog, _RecordingEngine, _RecordingExit, list[float]]:
    """Build a watchdog with injected settings, engine, exit fn, and clock."""
    from ingestion.core.config import IngestionSettings

    custom = IngestionSettings(
        REALTIME_POLL_STALL_WARN_SECONDS=stall_warn,
        REALTIME_POLL_STALL_EXIT_SECONDS=stall_exit,
        REALTIME_DATA_STALE_WARN_SECONDS=stale_warn,
        REALTIME_WATCHDOG_INTERVAL_SECONDS=60.0,
    )
    monkeypatch.setattr("ingestion.core.config.settings", custom)

    engine = _RecordingEngine()
    exit_recorder = _RecordingExit()
    clock_holder = {"now": 1_000_000.0}
    watchdog = RealtimeWatchdog(
        scheduler,
        alert_engine=engine,
        exit_fn=exit_recorder,
        clock=lambda: clock_holder["now"],
    )
    watchdog._started_at = clock_holder["now"]
    return watchdog, engine, exit_recorder, clock_holder


def _set_model_success(model: str, age_seconds: float) -> None:
    """Record one model's last-success ``age_seconds`` in the past (real time).

    The collector computes ages against the real wall clock (its accessor has
    no clock injection), so the helper converts the intended age into a real
    epoch timestamp. Replaces the state object wholesale so the
    ``collector_states_snapshot`` fixture's dict restore fully reverts it.
    """
    from ingestion.monitoring.ingestion_collector import ModelIngestionState

    INGESTION_COLLECTOR._states[model] = ModelIngestionState(
        model=model, last_success_ts=time.time() - age_seconds
    )


def _clear_model_state(model: str) -> None:
    """Drop one model's collector state (never-succeeded in this process).

    The full suite pollutes the global collector (other tests record real
    successes); the snapshot fixture restores the dict afterwards, so a
    targeted drop here keeps the never-succeeded scenarios hermetic.
    """
    INGESTION_COLLECTOR._states.pop(model, None)


# ---------------------------------------------------------------------------
# Poll-stall signal: alert, recovery, and bounded auto-recovery
# ---------------------------------------------------------------------------


def test_healthy_loop_dispatches_nothing(monkeypatch) -> None:
    scheduler = _FakeScheduler(last_poll_success_at=1_000_000.0 - 10.0)
    watchdog, engine, exit_recorder, _ = _make_watchdog(scheduler, monkeypatch)

    firing = watchdog.check_once()

    assert firing == []
    # check_once always dispatches the current snapshot (an empty list is the
    # engine's recovery signal); healthy means NO non-empty payload ever went
    # out and no process exit was requested.
    assert not any(engine.dispatched)
    assert exit_recorder.calls == []


def test_poll_stall_beyond_warn_dispatches_critical_alert(monkeypatch) -> None:
    scheduler = _FakeScheduler(last_poll_success_at=1_000_000.0 - 2_000.0)
    watchdog, engine, exit_recorder, _ = _make_watchdog(
        scheduler, monkeypatch, stall_warn=100.0, stall_exit=0.0
    )

    firing = watchdog.check_once()

    assert firing == ["realtime_scheduler_stalled"]
    assert len(engine.dispatched) == 1
    (alerts,) = engine.dispatched
    assert alerts[0].name == "realtime_scheduler_stalled"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].value == pytest.approx(2_000.0)
    assert alerts[0].threshold == pytest.approx(100.0)
    assert exit_recorder.calls == []  # exit disabled


def test_poll_stall_alert_clears_on_recovery(monkeypatch) -> None:
    scheduler = _FakeScheduler(last_poll_success_at=1_000_000.0 - 2_000.0)
    watchdog, engine, _, clock = _make_watchdog(scheduler, monkeypatch)

    watchdog.check_once()
    assert engine.dispatched[-1]  # firing

    # Heartbeat resumes: the next tick must pass an empty list so the engine's
    # deduplicator emits the recovery event.
    scheduler.last_poll_success_at = clock["now"]
    firing = watchdog.check_once()
    assert firing == []
    assert engine.dispatched[-1] == []


def test_poll_stall_beyond_exit_exits_process(monkeypatch) -> None:
    scheduler = _FakeScheduler(last_poll_success_at=1_000_000.0 - 30_000.0)
    watchdog, engine, exit_recorder, _ = _make_watchdog(
        scheduler, monkeypatch, stall_warn=100.0, stall_exit=21_600.0
    )

    firing = watchdog.check_once()

    assert firing == ["realtime_scheduler_stalled"]
    assert engine.dispatched, "alert must be dispatched BEFORE the exit"
    assert exit_recorder.calls == [1]


def test_poll_stall_exit_zero_never_exits(monkeypatch) -> None:
    scheduler = _FakeScheduler(last_poll_success_at=1_000_000.0 - 1_000_000.0)
    watchdog, engine, exit_recorder, _ = _make_watchdog(
        scheduler, monkeypatch, stall_warn=100.0, stall_exit=0.0
    )

    firing = watchdog.check_once()

    assert firing == ["realtime_scheduler_stalled"]
    assert exit_recorder.calls == []


def test_startup_wedge_is_bounded_without_heartbeat(monkeypatch) -> None:
    """No completed poll yet + no heartbeat progress => stall clock runs.

    The stall is measured from watchdog start, so a loop wedged before its
    first poll still reaches the exit threshold instead of never arming.
    """
    _clear_model_state("gfs")
    _clear_model_state("gefs")
    scheduler = _FakeScheduler(last_poll_success_at=None)
    watchdog, engine, exit_recorder, clock = _make_watchdog(
        scheduler, monkeypatch, stall_warn=100.0, stall_exit=21_600.0
    )
    clock["now"] += 30_000.0  # watchdog started 30_000s ago, still no poll

    firing = watchdog.check_once()

    # The wedged startup also means zero successes for the whole window, so
    # the data-staleness alert fires alongside the stall alert — both are
    # factually correct. The exit escalation is stall-driven only.
    assert set(firing) == {"realtime_scheduler_stalled", "realtime_ingestion_stale"}
    assert exit_recorder.calls == [1]
    assert engine.dispatched


# ---------------------------------------------------------------------------
# Data-staleness signal: alert-only, all models must be stale
# ---------------------------------------------------------------------------


def test_single_fresh_model_success_masks_staleness(monkeypatch, collector_states_snapshot) -> None:
    """One model finishing proves the pipeline is alive: no stale alert."""
    now = 1_000_000.0
    _set_model_success("gfs", 50_000.0)  # stale
    _set_model_success("gefs", 60.0)  # fresh
    scheduler = _FakeScheduler(last_poll_success_at=now - 10.0)
    watchdog, engine, exit_recorder, clock = _make_watchdog(
        scheduler, monkeypatch, stall_warn=100.0, stall_exit=0.0, stale_warn=1_000.0
    )
    clock["now"] = now

    firing = watchdog.check_once()

    assert firing == []
    assert not any(engine.dispatched)
    assert exit_recorder.calls == []


def test_all_models_stale_dispatches_warning_only(monkeypatch, collector_states_snapshot) -> None:
    now = 1_000_000.0
    _set_model_success("gfs", 50_000.0)
    _set_model_success("gefs", 49_000.0)
    scheduler = _FakeScheduler(last_poll_success_at=now - 10.0)
    watchdog, engine, exit_recorder, clock = _make_watchdog(
        scheduler, monkeypatch, stall_warn=100.0, stall_exit=0.0, stale_warn=1_000.0
    )
    clock["now"] = now

    firing = watchdog.check_once()

    assert firing == ["realtime_ingestion_stale"]
    (alerts,) = engine.dispatched
    assert alerts[0].name == "realtime_ingestion_stale"
    assert alerts[0].severity == AlertSeverity.WARNING
    # Alert-only: a data stall never terminates the process.
    assert exit_recorder.calls == []


def test_never_succeeded_models_measured_from_watchdog_start(
    monkeypatch, collector_states_snapshot
) -> None:
    """A fresh process with no successes must not fire the stale alert."""
    _clear_model_state("gfs")
    _clear_model_state("gefs")
    scheduler = _FakeScheduler(last_poll_success_at=None)
    watchdog, engine, _, clock = _make_watchdog(
        scheduler, monkeypatch, stall_warn=100.0, stall_exit=0.0, stale_warn=1_000.0
    )
    clock["now"] += 30.0  # 30s since start — far below the 1000s threshold

    firing = watchdog.check_once()

    assert firing == []


def test_data_staleness_alone_never_exits_process(
    monkeypatch, collector_states_snapshot
) -> None:
    """Even a huge data stall with a healthy loop must not exit the process."""
    now = 1_000_000.0
    _set_model_success("gfs", 500_000.0)
    _set_model_success("gefs", 500_000.0)
    scheduler = _FakeScheduler(last_poll_success_at=now - 10.0)
    watchdog, engine, exit_recorder, clock = _make_watchdog(
        scheduler,
        monkeypatch,
        stall_warn=100.0,
        stall_exit=21_600.0,
        stale_warn=1_000.0,
    )
    clock["now"] = now

    firing = watchdog.check_once()

    assert firing == ["realtime_ingestion_stale"]
    assert exit_recorder.calls == []
