"""Realtime scheduler watchdog: liveness alerts + bounded auto-recovery.

The 2026-10-28 incident: one wedged decode froze the realtime scheduler's
event loop for 9 days while every OS-level signal looked healthy — the
container stayed ``Up``, no error was logged, and the only observable was
*absence* of progress. This watchdog is the process-level answer. It runs as
a daemon thread that is deliberately independent of the scheduler's event
loop (in that incident the metrics HTTP thread kept serving while the loop
was frozen, which is why a thread-level liveness check is reliable exactly
when it matters).

Two independent signals, two distinct responses:

* **Poll stall** (the scheduler loop has not completed a poll iteration):
  this can only mean the loop itself is wedged — healthy polls complete in
  seconds and even the longest bounded wave is minutes. It raises a CRITICAL
  ``realtime_scheduler_stalled`` alert at
  :attr:`IngestionSettings.REALTIME_POLL_STALL_WARN_SECONDS` and, past
  :attr:`IngestionSettings.REALTIME_POLL_STALL_EXIT_SECONDS`, exits the
  process so the container's restart policy recovers it. The restart is safe
  by design: scheduler state is optimization-only and every poll reconstructs
  pending work from the upstream snapshots and the durable catalog, so a
  restarted process simply re-derives the frontier and catches up.
* **Data staleness** (no successful wave completion for any model): this can
  be entirely legitimate — upstream (NOAA) publication delays stall data
  without any local defect, and the healthy worst-case gap between
  consecutive successes is cycle interval + first-publication delay (~9.5h).
  It therefore only ever raises a WARNING ``realtime_ingestion_stale`` alert
  at :attr:`IngestionSettings.REALTIME_DATA_STALE_WARN_SECONDS`; it must
  never auto-restart a healthy process.

Alerts are dispatched through :meth:`AlertEngine.dispatch_alerts`, so
deduplication, cooldown, escalation, and recovery events behave identically
to the platform's rule-driven alerting (the ``[ALERT:...]`` log lines, the
in-memory ring, and the webhook sink all come for free).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, NoReturn, Protocol

from ingestion.monitoring.alerts import ALERT_ENGINE, Alert, AlertEngine, AlertSeverity
from ingestion.monitoring.ingestion_collector import INGESTION_COLLECTOR

logger = logging.getLogger(__name__)

#: Models whose last-success ages feed the data-staleness signal. GFS and GEFS
#: are the two realtime models; both must go stale before the alert fires (a
#: single model finishing proves the pipeline is alive).
_WATCHED_MODELS: tuple[str, ...] = ("gfs", "gefs")

#: Process exit code used for the watchdog-initiated self-termination. Any
#: non-zero code triggers the container's ``restart: unless-stopped`` policy.
_WATCHDOG_EXIT_CODE = 1


class _SchedulerLike(Protocol):
    """The scheduler surface the watchdog depends on (decoupled for tests)."""

    @property
    def last_poll_success_at(self) -> float | None:
        """Epoch seconds of the last completed poll iteration."""
        ...


class RealtimeWatchdog:
    """Thread-level liveness monitor for the realtime scheduler loop.

    Args:
        scheduler: The running :class:`~ingestion.realtime.scheduler.RealtimeScheduler`;
            polled for its ``last_poll_success_at`` heartbeat.
        alert_engine: Injected alert engine (defaults to the global singleton).
        exit_fn: Process-exit function (defaults to ``os._exit``; tests inject
            a recorder). Called only when the poll-stall exit threshold fires.
        clock: Time source returning epoch seconds (injectable for tests).
    """

    def __init__(
        self,
        scheduler: _SchedulerLike,
        *,
        alert_engine: AlertEngine | None = None,
        exit_fn: Callable[[int], NoReturn] = os._exit,
        clock: Callable[[], float] = time.time,
    ) -> None:
        from ingestion.core.config import settings

        self._scheduler = scheduler
        self._engine = alert_engine or ALERT_ENGINE
        self._exit_fn = exit_fn
        self._clock = clock
        self._interval = float(settings.REALTIME_WATCHDOG_INTERVAL_SECONDS)
        self._poll_stall_warn = float(settings.REALTIME_POLL_STALL_WARN_SECONDS)
        self._poll_stall_exit = float(settings.REALTIME_POLL_STALL_EXIT_SECONDS)
        self._data_stale_warn = float(settings.REALTIME_DATA_STALE_WARN_SECONDS)
        self._started_at: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the watchdog daemon thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._started_at = self._clock()
        self._thread = threading.Thread(
            target=self._run, name="realtime-watchdog", daemon=True
        )
        self._thread.start()
        logger.info(
            "realtime watchdog started (interval=%.0fs, poll stall warn=%.0fs, "
            "poll stall exit=%.0fs, data stale warn=%.0fs)",
            self._interval,
            self._poll_stall_warn,
            self._poll_stall_exit,
            self._data_stale_warn,
        )

    def stop(self) -> None:
        """Request the watchdog thread to stop at its next tick."""
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                self.check_once()
            except Exception:  # noqa: BLE001 - the watchdog must never die
                logger.exception("realtime watchdog check failed")

    # ------------------------------------------------------------------
    # One check
    # ------------------------------------------------------------------

    def check_once(self) -> list[str]:
        """Run one liveness check and dispatch any firing alerts.

        Returns:
            The names of the alerts currently firing after this check
            (empty when everything is healthy).
        """
        now = self._clock()
        firing: list[Alert] = []

        poll_stall = self._poll_stall_seconds(now)
        if poll_stall is not None and poll_stall > self._poll_stall_warn:
            firing.append(
                Alert(
                    name="realtime_scheduler_stalled",
                    severity=AlertSeverity.CRITICAL,
                    summary=(
                        "realtime scheduler loop has not completed a poll for "
                        f"{poll_stall / 60.0:.0f} minutes"
                    ),
                    description=(
                        "The scheduler's event loop is wedged (a poll iteration "
                        "has not returned). Data ingestion is frozen. "
                        + (
                            "The watchdog will exit the process for container "
                            f"restart recovery after {self._poll_stall_exit / 3600.0:.1f}h."
                            if self._poll_stall_exit > 0.0
                            else "Auto-recovery is disabled "
                            "(REALTIME_POLL_STALL_EXIT_SECONDS=0); restart "
                            "weather_ingestion_realtime manually."
                        )
                    ),
                    scope="realtime",
                    value=poll_stall,
                    threshold=self._poll_stall_warn,
                    runbook_anchor="#realtime-scheduler-stalled",
                )
            )

        data_stale = self._data_stale_seconds(now)
        if data_stale is not None and data_stale > self._data_stale_warn:
            firing.append(
                Alert(
                    name="realtime_ingestion_stale",
                    severity=AlertSeverity.WARNING,
                    summary=(
                        "no successful ingestion run for "
                        f"{data_stale / 3600.0:.1f}h (all realtime models)"
                    ),
                    description=(
                        "No model has completed a successful wave in this "
                        "window. May be legitimate upstream (NOAA) publication "
                        "delay; check upstream availability before treating it "
                        "as a local defect."
                    ),
                    scope="realtime",
                    value=data_stale,
                    threshold=self._data_stale_warn,
                    runbook_anchor="#realtime-ingestion-stale",
                )
            )

        self._engine.dispatch_alerts(firing)

        if poll_stall is not None and self._poll_stall_exit > 0.0:
            if poll_stall > self._poll_stall_exit:
                logger.critical(
                    "realtime watchdog: scheduler loop stalled for %.0fs "
                    "(> %.0fs exit threshold); exiting the process so the "
                    "container restart policy can recover it",
                    poll_stall,
                    self._poll_stall_exit,
                )
                self._exit_fn(_WATCHDOG_EXIT_CODE)
        return [alert.name for alert in firing]

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _poll_stall_seconds(self, now: float) -> float | None:
        """Seconds since the last completed poll (``None`` pre-first-poll).

        Before the first poll completes there is no heartbeat yet; the stall
        clock is measured from watchdog start so a wedged *startup* is still
        bounded without penalizing normal initialization.
        """
        last_poll = self._scheduler.last_poll_success_at
        if last_poll is None:
            reference = self._started_at
            if reference is None:
                return None
            return max(0.0, now - reference)
        return max(0.0, now - float(last_poll))

    def _data_stale_seconds(self, now: float) -> float | None:
        """Seconds since the MOST RECENT success across the watched models.

        Alert semantics: the pipeline is proven alive by a single fresh
        success, so the alert fires only when EVERY model is stale — i.e.
        when even the freshest model's last success is older than the
        threshold (``min`` of the per-model ages). A model with no success in
        this process (fresh start) is measured from watchdog start rather
        than firing immediately, which doubles as a natural post-restart
        grace period.
        """
        ages: list[float] = []
        for model in _WATCHED_MODELS:
            age = INGESTION_COLLECTOR.last_success_age_seconds(model)
            if age is None:
                age = (
                    max(0.0, now - self._started_at)
                    if self._started_at is not None
                    else 0.0
                )
            ages.append(age)
        if not ages:
            return None
        return min(ages)
