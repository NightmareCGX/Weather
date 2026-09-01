"""Runtime observability, startup timeline tracking, and live progress UI for ingestion.

Provides decoupled, bounded-memory progress monitoring and diagnostic timing
for the pipelined ingestion execution without modifying scheduling behavior,
adding database connections, or blocking worker threads.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS or S.sss."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes:02d}:{secs:04.1f}"
    return f"{secs:.2f}s"


def _format_time_mm_ss(seconds: float) -> str:
    """Format duration to MM:SS."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


@runtime_checkable
class StartupObserver(Protocol):
    """Protocol for observing milestone and cold-start progress events.

    Decouples coordinator and worker components from the concrete
    progress tracker or UI renderer.
    """

    def record_milestone(self, name: str, timestamp: float | None = None) -> None:
        """Record an ordered lifecycle milestone with its monotonic timestamp."""
        ...

    def set_init_phase(self, phase: str) -> None:
        """Update the active initialization phase name."""
        ...

    def set_marker_progress(self, *, done: int, total: int, active: int = 0) -> None:
        """Update progress of UPDATING marker PUTs during pre-update."""
        ...


@dataclass
class StageCounters:
    """Bounded, fixed-size stage counters for in-flight ingestion workloads."""

    total_regions: int = 0
    overall_done: int = 0

    download_queued: int = 0
    download_active: int = 0
    download_done: int = 0
    download_failed: int = 0

    decode_queued: int = 0
    decode_active: int = 0
    decode_done: int = 0
    decode_failed: int = 0

    write_waiting: int = 0
    write_active: int = 0
    write_done: int = 0
    write_failed: int = 0

    init_phase: str = "starting"
    init_done: bool = False
    init_failed: bool = False

    marker_done: int = 0
    marker_total: int = 0
    marker_active: int = 0

    finalize_state: str = "waiting"  # "waiting", "active", "done", "failed"
    finalize_start_time: float | None = None


class StartupTimeline:
    """Tracks ordered startup milestones and calculates stage latencies."""

    MILESTONE_KEYS: tuple[str, ...] = (
        "run_start",
        "seed_download_start",
        "seed_download_complete",
        "seed_decode_start",
        "seed_decode_complete",
        "catalog_init_start",
        "catalog_init_complete",
        "store_gate_wait_start",
        "store_gate_acquired",
        "prepare_run_store_start",
        "prepare_run_store_complete",
        "pre_update_start",
        "pre_update_complete",
        "store_ready",
        "wave_tasks_created",
        "seed_write_start",
        "seed_write_complete",
        "first_non_seed_download_start",
        "downloads_drained",
        "decodes_drained",
        "writes_drained",
        "finalize_start",
        "marker_listing_start",
        "marker_listing_complete",
        "marker_read_validation_start",
        "marker_read_validation_complete",
        "manifest_write_start",
        "manifest_write_complete",
        "catalog_reconcile_start",
        "catalog_reconcile_complete",
        "finalize_complete",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._milestones: dict[str, float] = {}
        self._wall_start: datetime = datetime.now(timezone.utc)
        self._mono_start: float = time.monotonic()

    def record(self, name: str, timestamp: float | None = None) -> None:
        """Record a milestone timestamp if not already recorded."""
        ts = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            if name not in self._milestones:
                self._milestones[name] = ts

    def get(self, name: str) -> float | None:
        """Get timestamp for a milestone."""
        with self._lock:
            return self._milestones.get(name)

    def get_all(self) -> dict[str, float]:
        """Return a copy of all recorded milestone timestamps."""
        with self._lock:
            return dict(self._milestones)

    def duration(self, start_key: str, end_key: str) -> float | None:
        """Calculate duration between two milestones."""
        with self._lock:
            t0 = self._milestones.get(start_key)
            t1 = self._milestones.get(end_key)
        if t0 is not None and t1 is not None:
            return max(0.0, t1 - t0)
        return None

    def offset_from_start(self, key: str) -> float | None:
        """Calculate offset from run_start in seconds."""
        with self._lock:
            t_start = self._milestones.get("run_start", self._mono_start)
            t = self._milestones.get(key)
        if t is not None:
            return max(0.0, t - t_start)
        return None

    def format_report(self, model: str, cycle_str: str, total_items: int) -> str:
        """Format a comprehensive startup timeline report."""
        with self._lock:
            m = dict(self._milestones)
            t0 = m.get("run_start", self._mono_start)

        def _fmt_ts(key: str) -> str:
            val = m.get(key)
            if val is None:
                return "   --:--.---"
            off = max(0.0, val - t0)
            mins = int(off // 60)
            secs = off % 60
            return f"+{mins:02d}:{secs:06.3f}"

        def _fmt_dur(start_k: str, end_k: str) -> str:
            d = self.duration(start_k, end_k)
            return f"{d:.3f}s" if d is not None else "--"

        seed_dl_dur = _fmt_dur("seed_download_start", "seed_download_complete")
        seed_dec_dur = _fmt_dur("seed_decode_start", "seed_decode_complete")
        cat_init_dur = _fmt_dur("catalog_init_start", "catalog_init_complete")
        gate_wait_dur = _fmt_dur("store_gate_wait_start", "store_gate_acquired")
        prep_store_dur = _fmt_dur("prepare_run_store_start", "prepare_run_store_complete")
        pre_up_dur = _fmt_dur("pre_update_start", "pre_update_complete")
        seed_wr_dur = _fmt_dur("seed_write_start", "seed_write_complete")

        total_startup_delay = self.duration("seed_download_start", "first_non_seed_download_start")
        startup_delay_str = (
            f"{total_startup_delay:.3f}s" if total_startup_delay is not None else "N/A"
        )
        tail_write_dur = self.duration("decodes_drained", "writes_drained")
        tail_write_str = (
            f"{tail_write_dur:.3f}s" if tail_write_dur is not None else "--"
        )
        teardown_dur = self.duration("writes_drained", "finalize_start")
        teardown_str = (
            f"{teardown_dur:.3f}s" if teardown_dur is not None else "--"
        )

        marker_list_dur = _fmt_dur("marker_listing_start", "marker_listing_complete")
        marker_val_dur = _fmt_dur("marker_read_validation_start", "marker_read_validation_complete")
        manifest_wr_dur = _fmt_dur("manifest_write_start", "manifest_write_complete")
        cat_rec_dur = _fmt_dur("catalog_reconcile_start", "catalog_reconcile_complete")
        finalize_dur = _fmt_dur("finalize_start", "finalize_complete")

        lines = [
            "=" * 80,
            "                       INGESTION STARTUP TIMELINE REPORT",
            "=" * 80,
            f"Model: {model.upper()} | Cycle: {cycle_str} | Total Target Regions: {total_items}",
            "-" * 80,
            f"{'Phase / Event':<42} {'Timestamp (offset)':<20} {'Duration':<15}",
            "-" * 80,
            f"1. Run Start                              {_fmt_ts('run_start'):<20} -",
            f"2. Seed Download                          {_fmt_ts('seed_download_start'):<20} {seed_dl_dur}",
            f"   ├─ seed_download_start                 {_fmt_ts('seed_download_start')}",
            f"   └─ seed_download_complete              {_fmt_ts('seed_download_complete')}",
            f"3. Seed Decode & Normalize                {_fmt_ts('seed_decode_start'):<20} {seed_dec_dur}",
            f"   ├─ seed_decode_start                   {_fmt_ts('seed_decode_start')}",
            f"   └─ seed_decode_complete                {_fmt_ts('seed_decode_complete')}",
            f"4. Catalog / Run Lookup                   {_fmt_ts('catalog_init_start'):<20} {cat_init_dur}",
            f"   ├─ catalog_init_start                  {_fmt_ts('catalog_init_start')}",
            f"   └─ catalog_init_complete               {_fmt_ts('catalog_init_complete')}",
            f"5. Store Gate Wait (Exclusive Lock)       {_fmt_ts('store_gate_wait_start'):<20} {gate_wait_dur}",
            f"   ├─ store_gate_wait_start               {_fmt_ts('store_gate_wait_start')}",
            f"   └─ store_gate_acquired                 {_fmt_ts('store_gate_acquired')}",
            f"6. Prepare Run Store (Zarr Init)          {_fmt_ts('prepare_run_store_start'):<20} {prep_store_dur}",
            f"   ├─ prepare_run_store_start             {_fmt_ts('prepare_run_store_start')}",
            f"   └─ prepare_run_store_complete          {_fmt_ts('prepare_run_store_complete')}",
            f"7. Pre-Update Marker PUTs                 {_fmt_ts('pre_update_start'):<20} {pre_up_dur}",
            f"   ├─ pre_update_start                    {_fmt_ts('pre_update_start')}",
            f"   └─ pre_update_complete                 {_fmt_ts('pre_update_complete')}",
            f"8. Store Ready & Wave Tasks Created       {_fmt_ts('store_ready'):<20} -",
            f"   ├─ store_ready                         {_fmt_ts('store_ready')}",
            f"   └─ wave_tasks_created                  {_fmt_ts('wave_tasks_created')}",
            f"9. First Non-Seed Download Start          {_fmt_ts('first_non_seed_download_start'):<20} -",
            "-" * 80,
            f"* Total Cold-Start Delay (Seed DL Start -> 1st Non-Seed DL Start): {startup_delay_str}",
            "-" * 80,
            "Seed Write (Async Task):",
            f"   ├─ seed_write_start                    {_fmt_ts('seed_write_start')}",
            f"   └─ seed_write_complete                 {_fmt_ts('seed_write_complete'):<20} {seed_wr_dur}",
            "-" * 80,
            "Pipeline Drain Milestones:",
            f"   ├─ downloads_drained                   {_fmt_ts('downloads_drained')}",
            f"   ├─ decodes_drained                     {_fmt_ts('decodes_drained')}",
            f"   └─ writes_drained                      {_fmt_ts('writes_drained')}",
            "-" * 80,
            f"* Tail Physical Write Drain (decodes_drained -> writes_drained): {tail_write_str}",
            f"* Task Teardown / Gate Transition (writes_drained -> finalize_start): {teardown_str}",
            "-" * 80,
            "Finalization Breakdown:",
            f"   ├─ finalize_start                      {_fmt_ts('finalize_start')}",
            f"   ├─ Marker Listing                      {_fmt_ts('marker_listing_start'):<20} {marker_list_dur}",
            f"   │  ├─ marker_listing_start             {_fmt_ts('marker_listing_start')}",
            f"   │  └─ marker_listing_complete          {_fmt_ts('marker_listing_complete')}",
            f"   ├─ Marker Read & Validation            {_fmt_ts('marker_read_validation_start'):<20} {marker_val_dur}",
            f"   │  ├─ marker_read_validation_start     {_fmt_ts('marker_read_validation_start')}",
            f"   │  └─ marker_read_validation_complete  {_fmt_ts('marker_read_validation_complete')}",
            f"   ├─ Manifest Write                      {_fmt_ts('manifest_write_start'):<20} {manifest_wr_dur}",
            f"   │  ├─ manifest_write_start             {_fmt_ts('manifest_write_start')}",
            f"   │  └─ manifest_write_complete          {_fmt_ts('manifest_write_complete')}",
            f"   ├─ Catalog Reconciliation              {_fmt_ts('catalog_reconcile_start'):<20} {cat_rec_dur}",
            f"   │  ├─ catalog_reconcile_start          {_fmt_ts('catalog_reconcile_start')}",
            f"   │  └─ catalog_reconcile_complete       {_fmt_ts('catalog_reconcile_complete')}",
            f"   └─ finalize_complete                   {_fmt_ts('finalize_complete'):<20} {finalize_dur}",
            "=" * 80,
        ]
        return "\n".join(lines)


class PipelineProgressTracker:
    """Thread-safe state manager for pipeline progress tracking.

    Maintains bounded $O(1)$ memory counters for workloads up to 2430+ regions,
    tracks timeline milestones, and emits structured DEBUG logs without overhead
    when DEBUG logging is disabled.
    """

    def __init__(
        self,
        *,
        model: str,
        cycle_str: str,
        total_items: int,
    ) -> None:
        self.model = model
        self.cycle_str = cycle_str
        self.total_items = max(1, total_items)
        self.counters = StageCounters(
            total_regions=self.total_items,
            download_queued=max(0, self.total_items - 1),
        )
        self.timeline = StartupTimeline()
        self._lock = threading.Lock()
        self._start_mono: float = time.monotonic()
        self._first_non_seed_dl_recorded: bool = False
        self._init_start_mono: float = time.monotonic()

    @property
    def start_time(self) -> float:
        return self._start_mono

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self._start_mono)

    def record_milestone(self, name: str, timestamp: float | None = None) -> None:
        """Record an ordered lifecycle milestone."""
        self.timeline.record(name, timestamp)
        if name == "run_start" and timestamp is not None:
            self._start_mono = timestamp

    def set_init_phase(self, phase: str) -> None:
        """Update the active initialization phase name."""
        with self._lock:
            self.counters.init_phase = phase
            if phase in ("store_ready", "done"):
                self.counters.init_done = True
            elif phase == "failed":
                self.counters.init_failed = True

    def set_marker_progress(self, *, done: int, total: int, active: int = 0) -> None:
        """Update progress of UPDATING marker PUTs during pre-update."""
        with self._lock:
            self.counters.marker_done = done
            self.counters.marker_total = total
            self.counters.marker_active = active

    def log_stage_transition(
        self,
        *,
        member: int | None,
        lead: int | None,
        stage: str,
        event: str,
        duration_ms: float | None = None,
    ) -> None:
        """Emit structured DEBUG-level log with zero formatting cost when disabled."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        duration_str = f" duration_ms={duration_ms:.2f}" if duration_ms is not None else ""
        logger.debug(
            "stage_transition: model=%s cycle=%s member=%s lead=%s stage=%s event=%s%s",
            self.model,
            self.cycle_str,
            member,
            lead,
            stage,
            event,
            duration_str,
        )

    def _check_drain_milestones_locked(self) -> None:
        """Evaluate pipeline drain milestones under self._lock."""
        # 1. downloads_drained
        if (
            self.counters.download_active == 0
            and (self.counters.download_done + self.counters.download_failed >= self.total_items)
        ):
            self.timeline.record("downloads_drained")

        # 2. decodes_drained:
        # All items have either completed/failed decode or failed before reaching decode (download_failed),
        # and no item is active in decode or queued for decode.
        if (
            self.counters.decode_active == 0
            and self.counters.decode_queued == 0
            and (self.counters.decode_done + self.counters.decode_failed + self.counters.download_failed >= self.total_items)
        ):
            self.timeline.record("decodes_drained")

        # 3. writes_drained:
        # All items have settled (write_done, write_failed, decode_failed, or download_failed),
        # and no item is active in write or waiting to enter write.
        terminal_items = (
            self.counters.write_done
            + self.counters.write_failed
            + self.counters.decode_failed
            + self.counters.download_failed
        )
        if (
            self.counters.write_active == 0
            and self.counters.write_waiting == 0
            and terminal_items >= self.total_items
        ):
            self.timeline.record("writes_drained")

    # -------------------------------------------------------------------------
    # Stage lifecycle hooks
    # -------------------------------------------------------------------------

    def on_download_start(self, member: int | None, lead: int, *, is_seed: bool = False) -> None:
        with self._lock:
            if not is_seed:
                if not self._first_non_seed_dl_recorded:
                    self._first_non_seed_dl_recorded = True
                    self.timeline.record("first_non_seed_download_start")
                self.counters.download_queued = max(0, self.counters.download_queued - 1)
            self.counters.download_active += 1
        self.log_stage_transition(
            member=member, lead=lead, stage="download", event="start"
        )

    def on_download_complete(
        self, member: int | None, lead: int, *, duration_ms: float
    ) -> None:
        with self._lock:
            self.counters.download_active = max(0, self.counters.download_active - 1)
            self.counters.download_done += 1
            self.counters.decode_queued += 1
            self._check_drain_milestones_locked()
        self.log_stage_transition(
            member=member,
            lead=lead,
            stage="download",
            event="complete",
            duration_ms=duration_ms,
        )

    def on_download_failed(
        self, member: int | None, lead: int, *, duration_ms: float
    ) -> None:
        with self._lock:
            self.counters.download_active = max(0, self.counters.download_active - 1)
            self.counters.download_failed += 1
            self._check_drain_milestones_locked()
        self.log_stage_transition(
            member=member,
            lead=lead,
            stage="download",
            event="failed",
            duration_ms=duration_ms,
        )

    def on_decode_start(self, member: int | None, lead: int) -> None:
        with self._lock:
            self.counters.decode_queued = max(0, self.counters.decode_queued - 1)
            self.counters.decode_active += 1
        self.log_stage_transition(
            member=member, lead=lead, stage="decode", event="start"
        )

    def on_decode_complete(
        self, member: int | None, lead: int, *, duration_ms: float
    ) -> None:
        with self._lock:
            self.counters.decode_active = max(0, self.counters.decode_active - 1)
            self.counters.decode_done += 1
            self.counters.write_waiting += 1
            self._check_drain_milestones_locked()
        self.log_stage_transition(
            member=member,
            lead=lead,
            stage="decode",
            event="complete",
            duration_ms=duration_ms,
        )

    def on_decode_failed(
        self, member: int | None, lead: int, *, duration_ms: float
    ) -> None:
        with self._lock:
            self.counters.decode_active = max(0, self.counters.decode_active - 1)
            self.counters.decode_failed += 1
            self._check_drain_milestones_locked()
        self.log_stage_transition(
            member=member,
            lead=lead,
            stage="decode",
            event="failed",
            duration_ms=duration_ms,
        )

    def on_write_start(self, member: int | None, lead: int, *, is_seed: bool = False) -> None:
        with self._lock:
            self.counters.write_waiting = max(0, self.counters.write_waiting - 1)
            self.counters.write_active += 1
        self.log_stage_transition(
            member=member, lead=lead, stage="write", event="start"
        )

    def on_write_complete(
        self, member: int | None, lead: int, *, duration_ms: float
    ) -> None:
        with self._lock:
            self.counters.write_active = max(0, self.counters.write_active - 1)
            self.counters.write_done += 1
            self.counters.overall_done += 1
            self._check_drain_milestones_locked()
        self.log_stage_transition(
            member=member,
            lead=lead,
            stage="write",
            event="complete",
            duration_ms=duration_ms,
        )

    def on_write_failed(
        self, member: int | None, lead: int, *, duration_ms: float
    ) -> None:
        with self._lock:
            self.counters.write_active = max(0, self.counters.write_active - 1)
            self.counters.write_failed += 1
            self._check_drain_milestones_locked()
        self.log_stage_transition(
            member=member,
            lead=lead,
            stage="write",
            event="failed",
            duration_ms=duration_ms,
        )

    def on_finalize_start(self) -> None:
        with self._lock:
            self.counters.finalize_state = "active"
            self.counters.finalize_start_time = time.monotonic()
            self.timeline.record("finalize_start")
        self.log_stage_transition(
            member=None, lead=None, stage="finalize", event="start"
        )

    def on_finalize_complete(self, *, duration_ms: float) -> None:
        with self._lock:
            self.counters.finalize_state = "done"
            self.timeline.record("finalize_complete")
        self.log_stage_transition(
            member=None,
            lead=None,
            stage="finalize",
            event="complete",
            duration_ms=duration_ms,
        )

    def on_finalize_failed(self, *, duration_ms: float) -> None:
        with self._lock:
            self.counters.finalize_state = "failed"
        self.log_stage_transition(
            member=None,
            lead=None,
            stage="finalize",
            event="failed",
            duration_ms=duration_ms,
        )

    # -------------------------------------------------------------------------
    # Progress rendering helpers
    # -------------------------------------------------------------------------

    def get_snapshot(self) -> StageCounters:
        """Return a copy of the current stage counters."""
        with self._lock:
            import copy
            return copy.copy(self.counters)

    def format_progress_lines(self) -> list[str]:
        """Format progress lines matching the approved target UX."""
        with self._lock:
            c = self.counters
            done = c.overall_done
            total = self.total_items
            elapsed_sec = max(0.0, time.monotonic() - self._start_mono)
            elapsed_str = _format_time_mm_ss(elapsed_sec)

            # Overall progress bar: 20 chars width
            pct = (done / total) * 100.0 if total > 0 else 0.0
            filled_len = int(20 * done // total) if total > 0 else 0
            bar = "█" * filled_len + "-" * (20 - filled_len)

            # Initialize line logic
            if c.init_done:
                init_str = f"done  elapsed={_format_duration(self.timeline.duration('run_start', 'store_ready') or 0.0)}"
            elif c.init_failed:
                init_str = "failed"
            elif c.init_phase == "pre_update_markers" and c.marker_total > 0:
                init_str = (
                    f"phase=pre_update_markers ({c.marker_done}/{c.marker_total} "
                    f"active={c.marker_active})  elapsed={_format_duration(elapsed_sec)}"
                )
            else:
                init_str = f"phase={c.init_phase}  elapsed={_format_duration(elapsed_sec)}"

            # Finalize line logic
            if c.finalize_state == "active" and c.finalize_start_time is not None:
                fin_elapsed = max(0.0, time.monotonic() - c.finalize_start_time)
                fin_str = f"active  elapsed={_format_duration(fin_elapsed)}"
            else:
                fin_str = c.finalize_state

            lines = [
                f"{'Overall':<12} [{bar}] {done}/{total}  {pct:4.1f}%  elapsed={elapsed_str}",
                f"{'Download':<12} active={c.download_active:<3} done={c.download_done:<4} queued={c.download_queued:<4}"
                + (f" failed={c.download_failed}" if c.download_failed > 0 else ""),
                f"{'Decode':<12} active={c.decode_active:<3} done={c.decode_done:<4} queued={c.decode_queued:<4}"
                + (f" failed={c.decode_failed}" if c.decode_failed > 0 else ""),
                f"{'Initialize':<12} {init_str}",
                f"{'Write':<12} active={c.write_active:<3} done={c.write_done:<4} waiting={c.write_waiting:<4}"
                + (f" failed={c.write_failed}" if c.write_failed > 0 else ""),
                f"{'Finalize':<12} {fin_str}",
            ]
            return lines

    def format_plain_summary(self) -> str:
        """Format a single-line summary for CI / non-TTY environments."""
        with self._lock:
            c = self.counters
            done = c.overall_done
            total = self.total_items
            elapsed_sec = max(0.0, time.monotonic() - self._start_mono)
            elapsed_str = _format_time_mm_ss(elapsed_sec)
            pct = (done / total) * 100.0 if total > 0 else 0.0

            init_part = f"Init:{c.init_phase}" if not c.init_done else "Init:done"
            if c.init_phase == "pre_update_markers" and c.marker_total > 0:
                init_part += f"({c.marker_done}/{c.marker_total})"

            return (
                f"[PROGRESS] {elapsed_str} | Overall: {done}/{total} ({pct:.1f}%) | "
                f"DL: act={c.download_active} done={c.download_done} | "
                f"Dec: act={c.decode_active} done={c.decode_done} | "
                f"Wr: act={c.write_active} done={c.write_done} | "
                f"{init_part}"
            )


# -----------------------------------------------------------------------------
# Renderer Implementations
# -----------------------------------------------------------------------------


class ProgressRenderer(ABC):
    """Abstract base class for terminal progress UI rendering."""

    @abstractmethod
    def start(self) -> None:
        """Start renderer."""
        ...

    @abstractmethod
    def update(self) -> None:
        """Render an update."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop renderer and cleanup terminal."""
        ...


class RichLiveRenderer(ProgressRenderer):
    """Interactive multi-stage live terminal renderer using Rich."""

    def __init__(self, tracker: PipelineProgressTracker) -> None:
        self.tracker = tracker
        self._live: Any | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        try:
            from rich.console import Console, Group
            from rich.live import Live
            from rich.text import Text

            console = Console(file=sys.stdout)
            self._console = console

            def _get_renderable() -> Group:
                lines = self.tracker.format_progress_lines()
                return Group(*(Text(line) for line in lines))

            self._live = Live(
                _get_renderable(),
                console=console,
                refresh_per_second=4,
                transient=False,
                redirect_stdout=True,
                redirect_stderr=True,
            )
            self._live.start()
            self._started = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to initialize Rich Live renderer: %s; falling back", exc)
            self._started = False

    def update(self) -> None:
        if not self._started or self._live is None:
            return
        try:
            from rich.text import Text
            from rich.console import Group
            lines = self.tracker.format_progress_lines()
            self._live.update(Group(*(Text(line) for line in lines)))
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        if not self._started or self._live is None:
            return
        try:
            self.update()
            self._live.stop()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._started = False
            self._live = None


class PlainTextSummaryRenderer(ProgressRenderer):
    """Periodic plain-text summary renderer for CI / non-TTY environments."""

    def __init__(
        self,
        tracker: PipelineProgressTracker,
        *,
        interval_seconds: float = 5.0,
        stream: Any = sys.stdout,
    ) -> None:
        self.tracker = tracker
        self.interval = interval_seconds
        self.stream = stream
        self._last_print = 0.0
        self._last_pct = -1.0

    def start(self) -> None:
        summary = self.tracker.format_plain_summary()
        self.stream.write(summary + "\n")
        self.stream.flush()
        self._last_print = time.monotonic()

    def update(self) -> None:
        now = time.monotonic()
        snapshot = self.tracker.get_snapshot()
        pct = (snapshot.overall_done / self.tracker.total_items) * 100.0
        # Print if interval elapsed or milestone % reached (25%, 50%, 75%, 100%)
        milestone = int(pct // 25) * 25
        if (now - self._last_print >= self.interval) or (
            milestone > self._last_pct and milestone > 0
        ):
            summary = self.tracker.format_plain_summary()
            self.stream.write(summary + "\n")
            self.stream.flush()
            self._last_print = now
            self._last_pct = milestone

    def stop(self) -> None:
        summary = self.tracker.format_plain_summary()
        self.stream.write(summary + "\n")
        self.stream.flush()


class NullProgressRenderer(ProgressRenderer):
    """No-op renderer when progress is disabled (--no-progress)."""

    def start(self) -> None:
        pass

    def update(self) -> None:
        pass

    def stop(self) -> None:
        pass


def create_progress_renderer(
    tracker: PipelineProgressTracker,
    *,
    no_progress: bool = False,
    is_tty: bool | None = None,
) -> ProgressRenderer:
    """Factory to create appropriate progress renderer based on environment."""
    if no_progress:
        return NullProgressRenderer()
    if is_tty is None:
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    if is_tty:
        try:
            import rich  # noqa: F401
            return RichLiveRenderer(tracker)
        except ImportError:
            return PlainTextSummaryRenderer(tracker)
    return PlainTextSummaryRenderer(tracker)
