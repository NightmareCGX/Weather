"""Deterministic tests for the realtime poll state machine (no real sleeps)."""

from __future__ import annotations

import random

import pytest

from ingestion.realtime.polling import (
    PollConfig,
    PollState,
    PollStateMachine,
    jitter_interval,
)


def _machine(**overrides) -> PollStateMachine:
    config = PollConfig(
        active_interval=600.0,
        publication_interval=120.0,
        backoff_initial=1800.0,
        backoff_max=3600.0,
        jitter_fraction=0.10,
        **overrides,
    )
    return PollStateMachine(config)


def test_initial_state_is_active_with_active_interval() -> None:
    machine = _machine()
    assert machine.state is PollState.ACTIVE
    assert machine.base_interval() == 600.0


def test_activity_moves_active_to_publishing() -> None:
    machine = _machine()
    machine.on_poll_success(activity=True)
    assert machine.state is PollState.PUBLISHING
    assert machine.base_interval() == 120.0


def test_idle_success_moves_to_backoff_at_initial_level() -> None:
    machine = _machine()
    machine.on_poll_success(activity=True)  # PUBLISHING
    machine.on_poll_success(activity=False)  # first idle poll
    assert machine.state is PollState.BACKOFF
    assert machine.base_interval() == 1800.0


def test_backoff_grows_to_max() -> None:
    machine = _machine()
    machine.on_poll_success(activity=False)  # ACTIVE → BACKOFF 1800
    machine.on_poll_success(activity=False)  # 3600
    assert machine.base_interval() == 3600.0
    machine.on_poll_success(activity=False)  # capped at max
    assert machine.base_interval() == 3600.0
    machine.on_poll_success(activity=False)  # stays capped
    assert machine.base_interval() == 3600.0


def test_activity_resets_backoff_level_and_returns_to_publishing() -> None:
    machine = _machine()
    machine.on_poll_success(activity=False)
    machine.on_poll_success(activity=False)
    assert machine.backoff_level == 3600.0
    machine.on_poll_success(activity=True)
    assert machine.state is PollState.PUBLISHING
    assert machine.backoff_level == 1800.0  # reset
    assert machine.base_interval() == 120.0


def test_jitter_stays_within_bounds_for_seeded_rng() -> None:
    rng = random.Random(1234)
    machine = _machine()
    for _ in range(200):
        value = machine.next_interval(rng)
        assert 540.0 <= value <= 660.0  # 600 ± 10%


def test_jitter_extremes_with_fake_rng() -> None:
    class _FakeRng:
        def __init__(self, value: float) -> None:
            self.value = value

        def uniform(self, low: float, high: float) -> float:
            return min(high, max(low, self.value))

    assert jitter_interval(600.0, 0.10, _FakeRng(-0.10)) == 540.0
    assert jitter_interval(600.0, 0.10, _FakeRng(0.10)) == 660.0
    assert jitter_interval(600.0, 0.0, _FakeRng(0.5)) == 600.0


def test_zero_fraction_disables_jitter() -> None:
    rng = random.Random(0)
    for _ in range(50):
        assert jitter_interval(120.0, 0.0, rng) == 120.0


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="backoff_max"):
        PollStateMachine(
            PollConfig(backoff_initial=3600.0, backoff_max=1800.0)
        )
    with pytest.raises(ValueError, match="jitter_fraction"):
        PollStateMachine(PollConfig(jitter_fraction=1.0))
