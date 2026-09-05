"""Realtime lead-wave scheduler: discovery → plan → bounded wave dispatch.

This module is the Phase 5C orchestration layer. It owns cycle selection,
discovery polling, snapshot comparison, planner invocation, wave timing,
poll state, leadership, graceful shutdown, and structured logging. It owns NO
download/decode/write/finalize logic — waves are dispatched through the
existing :func:`ingestion.core.wave_runner._run_wave`, exactly like big-batch.

Correctness invariants:

* Scheduler state (poll phase, wait timers, tracked cycle, leadership) is
  optimization-only. Every poll reconstructs pending work from the upstream
  :class:`~ingestion.providers.noaa.discovery.CycleSnapshot`s and the durable
  catalog committed state, so restarts, crashes, and concurrent big-batch
  commits converge without scheduler-owned truth.
* The shared GFS+GEFS barrier lives in the planner (policy), not here.
* A shared wave is dispatched as independent per-model store operations
  (GFS wave, then GEFS wave — never concurrently against themselves). If one
  model succeeds and the other fails, the successful model stays committed and
  the next reconciliation retries only the missing model's work.
* Discovery failures and committed-state read failures are never treated as
  "upstream idle": the last good snapshot is preserved, the poll state machine
  is untouched, and the failure is retried on a dedicated shorter interval.
* Shutdown while polling/waiting is prompt (the sleep waits on the stop
  event); shutdown during an active wave triggers the wave runner's existing
  non-abandoning cancellation/drain via its external cancel event and waits
  for the finalizer to finish — commit invariants are never violated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from ingestion.core.config import IngestionSettings, settings
from ingestion.core.wave_runner import RunSpec, _build_spec, _run_wave
from ingestion.providers.noaa.discovery import (
    CycleSnapshot,
    publication_changed,
    snapshot_gefs_cycle,
    snapshot_gfs_cycle,
)
from ingestion.realtime.committed import read_cycle_committed_state
from ingestion.realtime.planner import (
    FrontierPlan,
    ModelCommittedState,
    WavePolicy,
    plan_wave,
)
from ingestion.realtime.leadership import NoopLeadership, SchedulerLeadership
from ingestion.realtime.polling import PollConfig, PollStateMachine

logger = logging.getLogger(__name__)

#: Nominal cycle hours shared by GFS and GEFS.
_CYCLE_HOURS: tuple[int, ...] = (0, 6, 12, 18)


@dataclass(frozen=True)
class CycleIdentity:
    """One (model-agnostic) forecast cycle shared by both models."""

    cycle_date: date
    cycle_hour: int

    @property
    def cycle_time(self) -> datetime:
        """The UTC cycle time."""
        return datetime(
            self.cycle_date.year,
            self.cycle_date.month,
            self.cycle_date.day,
            self.cycle_hour,
            tzinfo=timezone.utc,
        )

    @property
    def label(self) -> str:
        """Diagnostics label, e.g. ``2026-07-21T00Z``."""
        return f"{self.cycle_date:%Y-%m-%d}T{self.cycle_hour:02d}Z"


def newest_eligible_cycle(
    now_utc: datetime, *, first_publication_delay_seconds: float
) -> CycleIdentity:
    """The most recent cycle whose publication window has plausibly opened.

    Publication begins roughly 3–3.5 h after cycle time (Phase 5A probe), so a
    cycle becomes eligible for probing after
    ``cycle_time + first_publication_delay_seconds``.

    Args:
        now_utc: The injected current UTC time.
        first_publication_delay_seconds: Delay after the nominal cycle time.

    Returns:
        The newest eligible cycle (yesterday's 18Z at the latest, so this
        always resolves).
    """
    threshold = now_utc - timedelta(seconds=first_publication_delay_seconds)
    for day_offset in (0, -1, -2):
        day = (now_utc + timedelta(days=day_offset)).date()
        for hour in reversed(_CYCLE_HOURS):
            candidate = CycleIdentity(cycle_date=day, cycle_hour=hour)
            if candidate.cycle_time <= threshold:
                return candidate
    # Unreachable: day_offset -2 always yields a cycle before the threshold.
    raise RuntimeError("no eligible cycle found")  # pragma: no cover


@dataclass(frozen=True)
class WaveDispatchResult:
    """Per-model result of one dispatched wave."""

    model: str
    targets: tuple[int, ...]
    status: str | None = None
    failures: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the wave completed without an exception or file failures."""
        return self.error is None and not self.failures


@dataclass
class PollOutcome:
    """The structured result of one scheduler poll iteration."""

    kind: str  # "planned" | "idle" | "discovery-failed" | "state-read-failed"
    cycle: CycleIdentity | None = None
    plan: FrontierPlan | None = None
    dispatches: list[WaveDispatchResult] = field(default_factory=list)
    error: str | None = None
    activity: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)


DispatchFn = Callable[[str, tuple[int, ...], CycleIdentity, threading.Event], WaveDispatchResult]
DiscoverFn = Callable[[CycleIdentity], tuple[CycleSnapshot | None, CycleSnapshot | None]]
ReadCommittedFn = Callable[[CycleIdentity], tuple[ModelCommittedState, ModelCommittedState]]
SleepFn = Callable[[float], None]


class RealtimeScheduler:
    """Poll-plan-dispatch loop for realtime lead-wave ingestion."""

    def __init__(
        self,
        *,
        conn_settings: IngestionSettings | None = None,
        discover: DiscoverFn | None = None,
        read_committed: ReadCommittedFn | None = None,
        dispatch_wave: DispatchFn | None = None,
        leadership: SchedulerLeadership | NoopLeadership | None = None,
        clock: Callable[[], float] = time.time,
        sleep: SleepFn | None = None,
        rng: random.Random | None = None,
        stop_event: threading.Event | None = None,
        cycle_override: CycleIdentity | None = None,
        download_dir: str = "downloads",
        concurrency: int = 4,
    ) -> None:
        """Create a scheduler.

        All external effects are injectable: ``discover`` wraps the Phase 5B
        snapshot functions, ``read_committed`` the catalog reader,
        ``dispatch_wave`` the wave runner, ``leadership`` the advisory-lock
        guard, ``clock``/``sleep``/``rng`` time and randomness (tests never
        sleep for real). Defaults wire the production implementations.
        """
        self.settings = conn_settings or settings
        self._clock = clock
        self._stop_event = stop_event or threading.Event()
        self._sleep_fn = sleep or self._default_sleep
        self._rng = rng or random.Random()
        self._cycle_override = cycle_override
        self._download_dir = download_dir
        self._concurrency = concurrency

        self.discover = discover or self._discover_production
        self.read_committed = read_committed or self._read_committed_production
        self.dispatch_wave = dispatch_wave or self._dispatch_production
        self.leadership = leadership

        self._machine = PollStateMachine(
            PollConfig(
                active_interval=float(self.settings.REALTIME_ACTIVE_POLL_SECONDS),
                publication_interval=float(
                    self.settings.REALTIME_PUBLICATION_POLL_SECONDS
                ),
                backoff_initial=float(
                    self.settings.REALTIME_IDLE_BACKOFF_INITIAL_SECONDS
                ),
                backoff_max=float(self.settings.REALTIME_IDLE_BACKOFF_MAX_SECONDS),
                jitter_fraction=float(self.settings.REALTIME_POLL_JITTER_FRACTION),
                discovery_failure_retry=float(
                    self.settings.REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS
                ),
            )
        )
        self._tracked_cycle: CycleIdentity | None = None
        self._last_good: tuple[CycleSnapshot | None, CycleSnapshot | None] | None = None
        self._first_seen_complete_at: dict[int, float] = {}
        self._last_poll_success_at: float | None = None
        self._last_activity_at: float | None = None
        self._stop_listeners: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Production wiring
    # ------------------------------------------------------------------

    def _default_sleep(self, seconds: float) -> None:
        """Sleep that returns promptly when shutdown is requested."""
        self._stop_event.wait(max(0.0, seconds))

    def _discover_production(
        self, cycle: CycleIdentity
    ) -> tuple[CycleSnapshot | None, CycleSnapshot | None]:
        gfs = asyncio.run(
            snapshot_gfs_cycle(
                cycle.cycle_date, cycle.cycle_hour, conn_settings=self.settings
            )
        )
        gefs = asyncio.run(
            snapshot_gefs_cycle(
                cycle.cycle_date, cycle.cycle_hour, conn_settings=self.settings
            )
        )
        return gfs, gefs

    def _read_committed_production(
        self, cycle: CycleIdentity
    ) -> tuple[ModelCommittedState, ModelCommittedState]:
        from ingestion.core.db import engine

        return read_cycle_committed_state(engine, cycle_time=cycle.cycle_time)

    def _dispatch_production(
        self, model: str, targets: tuple[int, ...], cycle: CycleIdentity, cancel_event: threading.Event
    ) -> WaveDispatchResult:
        args = self._wave_args()
        spec = RunSpec(
            model=model,
            cycle_date=cycle.cycle_date,
            cycle_hour=cycle.cycle_hour,
            target_lead_time_hours=targets,
            members=tuple(range(1, 31)) if model == "gefs" else (),
        )
        from ingestion.cli import derive_store_path

        store_path = derive_store_path(model, cycle.cycle_date, cycle.cycle_hour)
        catalog_spec = _build_spec(spec, args, store_path)
        failures: list[str] = []
        try:
            status = asyncio.run(
                _run_wave(
                    spec=spec,
                    args=args,
                    catalog_spec=catalog_spec,
                    store_path=store_path,
                    concurrency=self._concurrency,
                    failures=failures,
                    cancel_event=cancel_event,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one model failing must not stop the other
            logger.exception("realtime wave failed for model=%s", model)
            return WaveDispatchResult(
                model=model, targets=targets, error=str(exc)
            )
        return WaveDispatchResult(
            model=model, targets=targets, status=status, failures=tuple(failures)
        )

    def _wave_args(self) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(
            download_dir=self._download_dir,
            keep_downloads=False,
            no_progress=True,
            lock_timeout=float(self.settings.ADVISORY_LOCK_TIMEOUT_SECONDS),
            center_id="noaa",
            version_string="v1.0",
            grid_id="global_025deg",
            variable=None,
        )

    # ------------------------------------------------------------------
    # Cycle resolution
    # ------------------------------------------------------------------

    def _resolve_cycle(
        self,
    ) -> tuple[CycleIdentity | None, tuple[CycleSnapshot | None, CycleSnapshot | None] | None, bool]:
        """Resolve the cycle to track this poll and snapshot it.

        Returns ``(cycle, snapshots, switched)``. ``(None, None, False)`` when
        automatic selection finds no published cycle yet.

        Cycle identity always matches between models: both snapshots are taken
        for the SAME ``CycleIdentity`` by construction. If one model appears
        before the other, the cycle is adopted and the shared barrier simply
        stays blocked until the second model publishes.
        """
        now_utc = datetime.fromtimestamp(self._clock(), tz=timezone.utc)
        if self._cycle_override is not None:
            return (
                self._cycle_override,
                self.discover(self._cycle_override),
                self._tracked_cycle != self._cycle_override,
            )
        candidate = newest_eligible_cycle(
            now_utc,
            first_publication_delay_seconds=float(
                self.settings.REALTIME_FIRST_PUBLICATION_DELAY_SECONDS
            ),
        )
        snaps = self.discover(candidate)
        if self._snapshots_empty(snaps):
            # Newest eligible cycle has not started publishing: fall back to
            # the tracked cycle (it may still be publishing / ingesting) or,
            # at startup, to the previous cycle.
            if self._tracked_cycle is not None and self._tracked_cycle != candidate:
                previous = self._tracked_cycle
            else:
                previous = CycleIdentity(
                    cycle_date=candidate.cycle_date
                    - timedelta(days=1 if candidate.cycle_hour == 0 else 0),
                    cycle_hour=(candidate.cycle_hour - 6) % 24,
                )
            snaps = self.discover(previous)
            if self._snapshots_empty(snaps):
                return None, None, False
            return previous, snaps, self._tracked_cycle != previous
        return candidate, snaps, self._tracked_cycle != candidate

    @staticmethod
    def _changed(
        prev: CycleSnapshot | None, curr: CycleSnapshot | None
    ) -> bool:
        """Whether publication advanced between two good snapshots."""
        return (
            prev is not None
            and curr is not None
            and publication_changed(prev, curr)
        )

    @staticmethod
    def _snapshots_empty(
        snaps: tuple[CycleSnapshot | None, CycleSnapshot | None],
    ) -> bool:
        gfs, gefs = snaps
        gfs_empty = gfs is None or len(gfs.regions) == 0
        gefs_empty = gefs is None or len(gefs.regions) == 0
        return gfs_empty and gefs_empty

    def _is_leader_active(self) -> bool:
        """Whether leadership is currently held and active."""
        if self.leadership is None:
            return True
        check_fn = getattr(self.leadership, "check_leadership", None)
        if callable(check_fn):
            return bool(check_fn())
        return bool(getattr(self.leadership, "is_leader", False))

    # ------------------------------------------------------------------
    # One poll iteration
    # ------------------------------------------------------------------

    def poll_once(self, *, dry_run: bool = False) -> PollOutcome:
        """Run one full poll iteration: discover → plan → (maybe) dispatch.

        Args:
            dry_run: Plan and log diagnostics WITHOUT dispatching any wave.

        Returns:
            The structured :class:`PollOutcome`.
        """
        if self.leadership is not None and not self._is_leader_active():
            logger.warning(
                "realtime leadership is no longer held (connection lost or "
                "released); skipping planning and wave dispatch"
            )
            return self._log_outcome(
                PollOutcome(
                    kind="leadership-lost",
                    error="scheduler does not hold leadership",
                ),
                next_interval=None,
            )
        try:
            cycle, snaps, switched = self._resolve_cycle()
        except Exception as exc:  # noqa: BLE001 - discovery failure != idle
            logger.warning(
                "realtime discovery failed; preserving last good snapshot and "
                "poll state: %s",
                exc,
            )
            return self._log_outcome(
                PollOutcome(kind="discovery-failed", error=str(exc)),
                next_interval=None,
            )
        if cycle is None or snaps is None:
            return self._log_outcome(PollOutcome(kind="idle"))

        gfs_snap, gefs_snap = snaps
        activity = switched and not self._snapshots_empty(snaps)
        if self._last_good is not None and not switched:
            prev_gfs, prev_gefs = self._last_good
            activity = self._changed(prev_gfs, gfs_snap) or self._changed(
                prev_gefs, gefs_snap
            )

        try:
            committed_gfs, committed_gefs = self.read_committed(cycle)
        except Exception as exc:  # noqa: BLE001 - never plan without committed truth
            logger.warning(
                "realtime committed-state read failed; skipping planning to "
                "avoid duplicate work: %s",
                exc,
            )
            return self._log_outcome(
                PollOutcome(
                    kind="state-read-failed",
                    cycle=cycle,
                    error=str(exc),
                    activity=activity,
                ),
                next_interval=None,
            )

        if switched:
            # New cycle adopted: reset wait timers (new-cycle pending timing
            # starts fresh) and adopt the new diff baseline.
            self._first_seen_complete_at = {}
            self._last_good = snaps
            self._tracked_cycle = cycle

        now = self._clock()
        wave_policy = WavePolicy(
            max_leads=int(self.settings.REALTIME_WAVE_MAX_LEADS),
            max_wait_seconds=float(self.settings.REALTIME_WAVE_MAX_WAIT_SECONDS),
        )
        plan = plan_wave(
            cycle_label=cycle.label,
            gfs_snapshot=gfs_snap,
            gefs_snapshot=gefs_snap,
            committed_gfs=committed_gfs,
            committed_gefs=committed_gefs,
            policy=wave_policy,
            first_seen_complete_at=self._first_seen_complete_at,
            now=now,
        )
        # Timing-only scheduler memory: record first-seen for newly pending
        # leads and drop entries that are no longer pending.
        self._first_seen_complete_at = {
            lead: self._first_seen_complete_at.get(lead, now)
            for lead in plan.pending_complete_leads
        }

        dispatches: list[WaveDispatchResult] = []
        if plan.wave_due and plan.wave_candidate and not dry_run:
            cancel_event = threading.Event()

            def _request_cancel() -> None:
                cancel_event.set()

            self._stop_listeners.append(_request_cancel)
            try:
                # Sequential per-model dispatch of the SAME shared candidate:
                # never two waves against the same cycle store concurrently.
                for model, targets in (
                    ("gfs", plan.wave_targets_gfs),
                    ("gefs", plan.wave_targets_gefs),
                ):
                    if not targets:
                        continue
                    if self._stop_event.is_set() or cancel_event.is_set():
                        logger.info(
                            "realtime shutdown requested; skipping wave dispatch for model=%s",
                            model,
                        )
                        break
                    dispatches.append(
                        self.dispatch_wave(model, targets, cycle, cancel_event)
                    )
            finally:
                self._stop_listeners.remove(_request_cancel)
        elif plan.wave_due and dry_run:
            logger.info(
                "realtime dry-run: wave due for %s (targets gfs=%s gefs=%s); "
                "no dispatch",
                cycle.label,
                list(plan.wave_targets_gfs),
                list(plan.wave_targets_gefs),
            )

        self._machine.on_poll_success(activity=activity)
        self._last_good = snaps
        self._last_poll_success_at = now
        if activity:
            self._last_activity_at = now

        kind = "planned" if (plan.pending_complete_leads or dispatches) else "idle"
        return self._log_outcome(
            PollOutcome(
                kind=kind,
                cycle=cycle,
                plan=plan,
                dispatches=dispatches,
                activity=activity,
            )
        )

    # ------------------------------------------------------------------
    # Loop / lifecycle
    # ------------------------------------------------------------------

    def run(self, *, once: bool = False, dry_run: bool = False) -> int:
        """Run the scheduler loop.

        Args:
            once: Execute exactly ONE poll iteration (including wave dispatch
                unless ``dry_run``) and return — cron-driven operation and
                deterministic testing.
            dry_run: With ``once``, plan and log without dispatching.

        Returns:
            Process exit code: 0 normally; 0 with a clear log line when another
            instance holds leadership (benign duplicate start).
        """
        if self.leadership is not None:
            try:
                if not self.leadership.acquire():
                    logger.warning(
                        "another realtime scheduler instance holds leadership "
                        "(advisory lock); exiting without doing realtime work"
                    )
                    return 0
            except Exception:
                logger.exception("failed to acquire realtime leadership")
                return 1

        if not dry_run:
            from ingestion.core.s3 import verify_object_store_preflight

            verify_object_store_preflight(self.settings)

        try:
            while not self._stop_event.is_set():
                if self.leadership is not None and not self._is_leader_active():
                    logger.warning(
                        "realtime leadership lost; attempting to reacquire"
                    )
                    try:
                        if not self.leadership.acquire():
                            logger.warning(
                                "could not reacquire realtime leadership (held "
                                "by another instance); exiting"
                            )
                            return 0
                    except Exception:
                        logger.exception("exception while attempting to reacquire leadership")
                        return 1
                outcome = self.poll_once(dry_run=dry_run)
                if once:
                    return 0
                if outcome.kind in ("discovery-failed", "state-read-failed", "leadership-lost"):
                    interval = self._machine.config.discovery_failure_retry
                else:
                    interval = self._machine.next_interval(self._rng)
                logger.info(
                    "realtime sleeping %.1fs (state=%s)", interval, self._machine.state.value
                )
                self._sleep_fn(interval)
            logger.info("realtime scheduler stop requested; exiting")
            return 0
        finally:
            if self.leadership is not None:
                self.leadership.release()

    def request_stop(self) -> None:
        """Request graceful shutdown (signal-safe; idempotent).

        Prompts the sleep to return and, if a wave is in flight, triggers the
        wave runner's non-abandoning cancellation/drain.
        """
        self._stop_event.set()
        for listener in list(self._stop_listeners):
            listener()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _log_outcome(
        self, outcome: PollOutcome, *, next_interval: float | None = None
    ) -> PollOutcome:
        """Attach the structured diagnostics snapshot and log one line."""
        plan = outcome.plan
        diagnostics: dict[str, object] = {
            "kind": outcome.kind,
            "cycle": outcome.cycle.label if outcome.cycle else None,
            "poll_state": self._machine.state.value,
            "poll_interval": next_interval,
            "last_poll_success": self._last_poll_success_at,
            "last_publication_activity": self._last_activity_at,
            "leader": getattr(self.leadership, "is_leader", None),
            "discovery_error": outcome.error,
            "activity": outcome.activity,
            "wave_results": [
                {
                    "model": d.model,
                    "targets": list(d.targets),
                    "status": d.status,
                    "failures": len(d.failures),
                    "error": d.error,
                }
                for d in outcome.dispatches
            ],
        }
        if plan is not None:
            diagnostics.update(
                {
                    "observed_frontier": plan.observed_frontier,
                    "complete_frontier": plan.complete_frontier,
                    "committed_frontier": plan.committed_frontier,
                    "pending_leads": list(plan.pending_complete_leads),
                    "oldest_pending_age": plan.oldest_pending_age_seconds,
                    "blocked_lead": plan.next_blocked_lead,
                    "blocked_reason": plan.blocked_reason,
                    "missing_gfs_artifacts": list(plan.missing_gfs_artifacts),
                    "missing_gefs_members": list(plan.missing_gefs_members),
                    "wave_due": plan.wave_due,
                    "wave_candidate": list(plan.wave_candidate),
                    "wave_targets_gfs": list(plan.wave_targets_gfs),
                    "wave_targets_gefs": list(plan.wave_targets_gefs),
                }
            )
        outcome.diagnostics = diagnostics
        logger.info("realtime_poll %s", json.dumps(diagnostics, default=str))
        return outcome
