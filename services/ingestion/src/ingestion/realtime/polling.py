"""Realtime poll state machine with injectable time and randomness.

Three states drive the poll cadence (defaults from ``IngestionSettings``):

* ``ACTIVE``     — normal cadence (~10 min) while tracking a cycle.
* ``PUBLISHING`` — fast cadence (~2 min) while upstream publication activity
  is observed (new data object, new ``.idx``, GEFS member-count growth such as
  8/30 → 22/30, new observed lead, or complete-frontier growth).
* ``BACKOFF``    — no publication activity observed: back off progressively
  (~30 min initial, doubling to a configurable maximum, ~1 h).

A successful poll with an **unchanged** snapshot is an idle signal and moves
the machine toward BACKOFF. A **discovery failure is not an idle signal**: the
state machine never sees it (the scheduler retries on a dedicated failure
interval while preserving the last good snapshot), so network problems can
never masquerade as "upstream idle".

Jitter (configurable ± fraction) applies only to poll intervals; the random
source is injectable so tests are deterministic and never sleep.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass


class PollState(enum.Enum):
    """States of the realtime poll cadence machine."""

    ACTIVE = "active"
    PUBLISHING = "publishing"
    BACKOFF = "backoff"


@dataclass(frozen=True)
class PollConfig:
    """Cadence configuration (mirrors the ``REALTIME_*`` settings)."""

    active_interval: float = 600.0
    publication_interval: float = 120.0
    backoff_initial: float = 1800.0
    backoff_max: float = 3600.0
    jitter_fraction: float = 0.10
    discovery_failure_retry: float = 60.0


class PollStateMachine:
    """Explicit, testable poll-cadence state machine.

    The machine only consumes *successful* poll outcomes; discovery failures
    are handled by the scheduler with ``discovery_failure_retry`` and leave the
    machine (and its view of upstream activity) untouched.
    """

    def __init__(self, config: PollConfig) -> None:
        if config.backoff_max < config.backoff_initial:
            raise ValueError("backoff_max must be >= backoff_initial")
        if not 0.0 <= config.jitter_fraction < 1.0:
            raise ValueError("jitter_fraction must be in [0.0, 1.0)")
        self.config = config
        self.state = PollState.ACTIVE
        self._backoff_level = config.backoff_initial

    def base_interval(self) -> float:
        """The un-jittered interval for the current state."""
        if self.state is PollState.ACTIVE:
            return self.config.active_interval
        if self.state is PollState.PUBLISHING:
            return self.config.publication_interval
        return self._backoff_level

    def next_interval(self, rng: random.Random) -> float:
        """The jittered seconds to sleep before the next poll (>= 0)."""
        return jitter_interval(
            self.base_interval(), self.config.jitter_fraction, rng
        )

    def on_poll_success(self, *, activity: bool) -> None:
        """Record a successful poll outcome.

        Args:
            activity: Whether the upstream snapshot changed in any way
                (publication activity). An unchanged snapshot is NOT activity.
        """
        if activity:
            self.state = PollState.PUBLISHING
            self._backoff_level = self.config.backoff_initial
            return
        # Idle: back off progressively from any state.
        if self.state is not PollState.BACKOFF:
            self.state = PollState.BACKOFF
            self._backoff_level = self.config.backoff_initial
            return
        self._backoff_level = min(
            self._backoff_level * 2.0, self.config.backoff_max
        )

    @property
    def backoff_level(self) -> float:
        """The current backoff level (diagnostics)."""
        return self._backoff_level


def jitter_interval(
    interval: float, jitter_fraction: float, rng: random.Random
) -> float:
    """Apply symmetric relative jitter to a poll interval.

    Args:
        interval: The base interval in seconds.
        jitter_fraction: Relative jitter in [0.0, 1.0); the result lies in
            ``interval * (1 ± jitter_fraction)``.
        rng: Injectable random source (deterministic in tests).

    Returns:
        The jittered interval, clamped at zero from below.
    """
    factor = 1.0 + rng.uniform(-jitter_fraction, jitter_fraction)
    return max(0.0, interval * factor)
