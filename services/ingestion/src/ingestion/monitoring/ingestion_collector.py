"""Per-model ingestion health, stuck detection, lag tracking, and logical completeness.

Tracks download, decode, write, and finalize throughput and durations with bounded
Prometheus metric cardinality (model="gfs"|"gefs"). Distinguishes normal long-running
progress (e.g. GEFS 30 members) from stuck pipelines with zero forward progress,
and distinguishes upstream publication delay from local ingestion lag.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from domain.cadence import canonical_cycle_cadence_hours
from domain.coverage import get_expected_members, is_lead_servable
from domain.horizon import canonical_lead_time_hours, model_max_lead_hours
from domain.temporal import serving_start_valid_time
from ingestion.core.observability import PipelineProgressTracker
from ingestion.monitoring.metrics import REGISTRY

logger = logging.getLogger(__name__)


def latest_synoptic_cycle(now_utc: datetime, cadence_hours: int = 6) -> datetime:
    """Return the latest synoptic cycle initialization time (00Z, 06Z, 12Z, 18Z) <= now_utc."""
    hour = (now_utc.hour // cadence_hours) * cadence_hours
    return now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)


def _as_utc(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime (naive values are UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

# Prometheus metrics with bounded label cardinality (model="gfs"|"gefs")
INGESTION_LAST_START_TIMESTAMP = REGISTRY.gauge(
    "weather_ingestion_last_start_timestamp_seconds",
    "Timestamp (epoch seconds) when ingestion last started for model",
    labelnames=("model",),
)
INGESTION_LAST_SUCCESS_TIMESTAMP = REGISTRY.gauge(
    "weather_ingestion_last_success_timestamp_seconds",
    "Timestamp (epoch seconds) when ingestion last succeeded for model",
    labelnames=("model",),
)
INGESTION_LAST_FAILURE_TIMESTAMP = REGISTRY.gauge(
    "weather_ingestion_last_failure_timestamp_seconds",
    "Timestamp (epoch seconds) when ingestion last failed for model",
    labelnames=("model",),
)
INGESTION_CYCLE_DURATION_SECONDS = REGISTRY.gauge(
    "weather_ingestion_cycle_duration_seconds",
    "Duration in seconds of the most recent completed ingestion cycle",
    labelnames=("model",),
)
INGESTION_COLD_START_DURATION_SECONDS = REGISTRY.gauge(
    "weather_ingestion_cold_start_duration_seconds",
    "Duration in seconds from seed download start to first non-seed download",
    labelnames=("model",),
)
INGESTION_PREPARE_STORE_DURATION_SECONDS = REGISTRY.gauge(
    "weather_ingestion_prepare_store_duration_seconds",
    "Duration in seconds to initialize Zarr store and metadata",
    labelnames=("model",),
)
INGESTION_PRE_UPDATE_DURATION_SECONDS = REGISTRY.gauge(
    "weather_ingestion_pre_update_duration_seconds",
    "Duration in seconds to write pre-update UPDATING markers",
    labelnames=("model",),
)
INGESTION_FINALIZATION_DURATION_SECONDS = REGISTRY.gauge(
    "weather_ingestion_finalization_duration_seconds",
    "Duration in seconds to validate markers, write manifest, and reconcile catalog",
    labelnames=("model",),
)

INGESTION_ITEMS_COMPLETED_TOTAL = REGISTRY.counter(
    "weather_ingestion_items_completed_total",
    "Total target regions completed by phase",
    labelnames=("model", "phase"),
)
INGESTION_ITEMS_FAILED_TOTAL = REGISTRY.counter(
    "weather_ingestion_items_failed_total",
    "Total target regions failed by phase",
    labelnames=("model", "phase"),
)

INGESTION_STUCK_WARNING = REGISTRY.gauge(
    "weather_ingestion_stuck_warning",
    "Ingestion pipeline stuck indicator (1 if stuck with zero progress, 0 otherwise)",
    labelnames=("model",),
)
INGESTION_LAG_CYCLES = REGISTRY.gauge(
    "weather_ingestion_lag_cycles",
    "Ingestion lag in 6-hour cycles relative to available upstream data",
    labelnames=("model",),
)
INGESTION_LAG_HOURS = REGISTRY.gauge(
    "weather_ingestion_lag_hours",
    "Ingestion lag in hours relative to the newest cycle that should be complete",
    labelnames=("model",),
)
INGESTION_LAG_KNOWN = REGISTRY.gauge(
    "weather_ingestion_lag_known",
    "Whether ingestion lag is a real measurement (1) or could not be computed (0)",
    labelnames=("model",),
)
INGESTION_DATA_MISSING_CYCLES = REGISTRY.gauge(
    "weather_ingestion_data_missing_cycles",
    "Due cycles with no servable data at all (distinct from a status that has not been promoted)",
    labelnames=("model",),
)
INGESTION_CYCLES_COMPLETE_NOT_READY = REGISTRY.gauge(
    "weather_ingestion_cycles_complete_not_ready",
    "Cycles whose catalog contents are complete while their run status is not ready",
    labelnames=("model",),
)

GEFS_AVAILABLE_MEMBERS = REGISTRY.gauge(
    "weather_gefs_available_members_count",
    "Number of committed GEFS perturbation members for the latest active cycle",
)
GEFS_SERVABLE_STATUS = REGISTRY.gauge(
    "weather_gefs_servable_status",
    "GEFS runtime servability status (1 if >= 26/30 members, 0 otherwise)",
)
GEFS_READY_STATUS = REGISTRY.gauge(
    "weather_gefs_ready_status",
    "GEFS formal readiness status (1 if 30/30 members committed, 0 otherwise)",
)
MODEL_COMMITTED_LEADS = REGISTRY.gauge(
    "weather_model_committed_leads_count",
    "Number of committed distinct leads for the latest active cycle",
    labelnames=("model",),
)
MODEL_EXPECTED_LEADS = REGISTRY.gauge(
    "weather_model_expected_leads_count",
    "Authoritative expected distinct leads count for model and version",
    labelnames=("model",),
)
MODEL_MAX_LEAD_HOURS = REGISTRY.gauge(
    "weather_model_max_lead_hours",
    "Authoritative maximum lead time in hours for model and version",
    labelnames=("model",),
)
MODEL_READY_STATUS = REGISTRY.gauge(
    "weather_model_ready_status",
    "Model formal readiness status (1 if fully committed and ready, 0 otherwise)",
    labelnames=("model",),
)


@dataclass
class ModelIngestionState:
    """In-memory operational tracking for an NWP model."""

    model: str
    last_start_ts: float | None = None
    last_success_ts: float | None = None
    last_failure_ts: float | None = None
    current_cycle_str: str | None = None
    current_status: str = "idle"  # "idle", "running", "finalizing", "failed"
    current_phase: str = "none"

    # Progress tracking timestamps
    last_progress_ts: float = field(default_factory=time.monotonic)
    last_download_ts: float = field(default_factory=time.monotonic)
    last_decode_ts: float = field(default_factory=time.monotonic)
    last_write_ts: float = field(default_factory=time.monotonic)
    finalize_start_ts: float | None = None

    # Active tracker reference
    active_tracker: PipelineProgressTracker | None = None

    # Durations
    last_duration_s: float = 0.0
    cold_start_s: float = 0.0
    prep_store_s: float = 0.0
    pre_update_s: float = 0.0
    finalize_s: float = 0.0


@dataclass(frozen=True)
class CycleFact:
    """Catalog evidence for one ``(model, cycle)`` pair at one scrape.

    The row is deliberately *fact-shaped* rather than status-shaped: lag is
    measured against what the platform can actually serve, so a cycle that is
    still filling (``partial``) but already has servable leads must not be
    counted as lag. ``status`` is carried alongside for readiness reporting and
    for the "complete but not promoted" probe.

    Attributes:
        cycle_time: The UTC cycle time.
        run_created_at: ``model_runs.created_at`` — the platform's f000 ingest
            anchor for the cycle (see
            :meth:`IngestionHealthCollector.evaluate_lag`).
        status: The run's ``model_runs.status`` (``processing``/``partial``/
            ``ready``).
        committed_leads: Number of distinct leads with catalog product rows.
        servable: Whether at least one lead satisfies the serving coverage
            contract (``domain.coverage.is_lead_servable`` for ensembles; any
            committed lead for deterministic models).
        complete: Whether the catalog holds every canonical lead and, for
            ensembles, every expected member at every expected lead —
            regardless of the run's status.
    """

    cycle_time: datetime
    run_created_at: datetime
    status: str
    committed_leads: int
    servable: bool
    complete: bool


@dataclass(frozen=True)
class IngestionLagReport:
    """Lag analysis over the fill anchor, the servable baseline, and readiness.

    Three distinct questions that used to collapse into one number are reported
    separately:

    * **lag** — how far the newest *servable* cycle trails the newest cycle
      that *should* already be complete. The "should" comes from our own f000
      ingest anchor plus a fill budget, never from the wall clock.
    * **readiness** — the newest cycle whose run status is ``ready``.
    * **data absence** — cycles whose deadline passed with no servable data at
      all, which is a different failure from "data present but status not
      promoted".

    Attributes:
        model: Platform model identifier.
        latest_expected_cycle: Clock-derived newest synoptic cycle. Diagnostic
            only: it is deliberately NOT the lag target, because a cycle is
            only due once its fill window has closed.
        latest_upstream_cycle: Explicit upstream cycle when the caller probed
            one; ``None`` in production (no upstream discovery yet).
        latest_servable_cycle: Lag baseline — newest cycle with a servable lead.
        latest_ready_cycle: Newest cycle whose run status is ``ready``.
        lag_target_cycle: Newest cycle that should already be complete.
        latest_missing_run_cycle: Newest deadline-passed cycle with no
            ``model_runs`` row at all (publication window and fill budget both
            closed, so there is no f000 anchor to measure from).
        upstream_available: Whether an explicit upstream cycle was supplied.
        lag_known: Whether :attr:`lag_cycles`/:attr:`lag_hours` are real
            measurements rather than placeholders.
        lag_cycles: Whole cadence cycles between target and baseline.
        lag_hours: Hours between target and baseline.
        data_missing_cycles: Deadline-passed cycles with no servable data.
        is_behind: Whether a *known* lag measurement exceeds zero.
        lag_target_cycle: Newest cycle that should already be complete.
        latest_missing_run_cycle: Newest deadline-passed cycle with no
            ``model_runs`` row at all (publication window and fill budget both
            closed, so there is no f000 anchor to measure from).
        lag_target_ready: Whether the target cycle's run status is ``ready``,
            or ``None`` when the target has no run row. Deliberately not the
            newest run's readiness: the newest cycle is *expected* to be
            unready while it fills, so only the due cycle's verdict can drive a
            stalled-promotion alert.
        lag_target_overdue_seconds: Seconds the target cycle is past its fill
            deadline (0 when it is not due yet). A full grace period here is
            the "sustained" evidence an alert needs.
        complete_not_ready_cycles: Cycles whose catalog contents are complete
            while their status is not ``ready`` — the direct probe for a
            promotion defect.
        oldest_complete_not_ready_overdue_seconds: Overdue time of the oldest
            such cycle, so a freshly completed cycle mid-promotion is not
            alerted on.
    """

    model: str
    latest_expected_cycle: datetime
    latest_upstream_cycle: datetime | None
    latest_servable_cycle: datetime | None
    latest_ready_cycle: datetime | None
    upstream_available: bool
    lag_known: bool
    lag_cycles: int
    lag_hours: float
    data_missing_cycles: int
    is_behind: bool
    lag_target_cycle: datetime | None = None
    latest_missing_run_cycle: datetime | None = None
    lag_target_ready: bool | None = None
    lag_target_overdue_seconds: float | None = None
    complete_not_ready_cycles: int = 0
    oldest_complete_not_ready_overdue_seconds: float | None = None


@dataclass(frozen=True)
class IngestionStuckReport:
    """Report on whether an in-flight ingestion run has stalled."""

    model: str
    is_stuck: bool
    reason: str | None
    stuck_duration_seconds: float


class IngestionHealthCollector:
    """Monitors ingestion lifecycle, throughput, stuck conditions, and lag."""

    STUCK_DOWNLOAD_TIMEOUT_S = 600.0  # 10 minutes zero progress
    STUCK_DECODE_TIMEOUT_S = 600.0  # 10 minutes zero progress
    STUCK_WRITE_TIMEOUT_S = 600.0  # 10 minutes zero progress
    STUCK_FINALIZE_TIMEOUT_S = 600.0  # 10 minutes finalize

    def __init__(
        self,
        engine: Engine | Connection | None = None,
        *,
        fill_grace_seconds: float | None = None,
        publication_delay_seconds: float | None = None,
        version_string: str = "v1.0",
    ) -> None:
        """Create the collector.

        Args:
            engine: Catalog engine used for lag/readiness queries. ``None``
                degrades every catalog-derived value to "unknown" (never to a
                fabricated one).
            fill_grace_seconds: Override for the fill budget a cycle may spend
                filling before it counts as lagging. Defaults to
                ``IngestionSettings.INGESTION_FILL_IN_GRACE_SECONDS``, resolved
                lazily so this module stays importable without service
                settings.
            publication_delay_seconds: Override for the upstream publication
                delay used to date cycles that never got a run row. Defaults to
                ``IngestionSettings.REALTIME_FIRST_PUBLICATION_DELAY_SECONDS``.
            version_string: Model version whose runs are considered.
        """
        self.engine = engine
        self.version_string = version_string
        self._fill_grace_seconds = fill_grace_seconds
        self._publication_delay_seconds = publication_delay_seconds
        self._lock = threading.Lock()
        self._states: dict[str, ModelIngestionState] = {
            "gfs": ModelIngestionState(model="gfs"),
            "gefs": ModelIngestionState(model="gefs"),
        }

    @property
    def fill_grace_seconds(self) -> float:
        """Seconds a cycle may spend filling before it counts as lagging."""
        if self._fill_grace_seconds is not None:
            return float(self._fill_grace_seconds)
        from ingestion.core.config import settings

        return float(settings.INGESTION_FILL_IN_GRACE_SECONDS)

    @property
    def publication_delay_seconds(self) -> float:
        """Seconds after cycle time before upstream publication is expected."""
        if self._publication_delay_seconds is not None:
            return float(self._publication_delay_seconds)
        from ingestion.core.config import settings

        return float(settings.REALTIME_FIRST_PUBLICATION_DELAY_SECONDS)

    def register_run_start(
        self,
        model: str,
        cycle_str: str,
        tracker: PipelineProgressTracker | None = None,
    ) -> None:
        """Record the start of an ingestion run."""
        m = model.lower()
        now_ts = time.time()
        now_mono = time.monotonic()
        with self._lock:
            state = self._states.setdefault(m, ModelIngestionState(model=m))
            state.last_start_ts = now_ts
            state.current_cycle_str = cycle_str
            state.current_status = "running"
            state.current_phase = "starting"
            state.last_progress_ts = now_mono
            state.last_download_ts = now_mono
            state.last_decode_ts = now_mono
            state.last_write_ts = now_mono
            state.finalize_start_ts = None
            state.active_tracker = tracker

        INGESTION_LAST_START_TIMESTAMP.labels(model=m).set(now_ts)
        INGESTION_STUCK_WARNING.labels(model=m).set(0.0)

    def record_progress(
        self,
        model: str,
        stage: str,
        success: bool = True,
    ) -> None:
        """Record a forward progress event from a worker."""
        m = model.lower()
        now_mono = time.monotonic()
        with self._lock:
            state = self._states.get(m)
            if state is not None:
                state.last_progress_ts = now_mono
                if stage == "download":
                    state.last_download_ts = now_mono
                elif stage == "decode":
                    state.last_decode_ts = now_mono
                elif stage == "write":
                    state.last_write_ts = now_mono
                elif stage == "finalize":
                    if state.finalize_start_ts is None:
                        state.finalize_start_ts = now_mono

        if success:
            INGESTION_ITEMS_COMPLETED_TOTAL.labels(model=m, phase=stage).inc(1.0)
        else:
            INGESTION_ITEMS_FAILED_TOTAL.labels(model=m, phase=stage).inc(1.0)

    def register_run_finish(
        self,
        model: str,
        success: bool,
        duration_s: float = 0.0,
        timeline_breakdown: dict[str, float] | None = None,
    ) -> None:
        """Record the completion of an ingestion run."""
        m = model.lower()
        now_ts = time.time()
        with self._lock:
            state = self._states.setdefault(m, ModelIngestionState(model=m))
            state.current_status = "idle" if success else "failed"
            state.current_phase = "done" if success else "failed"
            state.last_duration_s = duration_s
            state.active_tracker = None
            if success:
                state.last_success_ts = now_ts
                INGESTION_LAST_SUCCESS_TIMESTAMP.labels(model=m).set(now_ts)
            else:
                state.last_failure_ts = now_ts
                INGESTION_LAST_FAILURE_TIMESTAMP.labels(model=m).set(now_ts)

            if timeline_breakdown:
                state.cold_start_s = timeline_breakdown.get("cold_start", 0.0)
                state.prep_store_s = timeline_breakdown.get("prepare_run_store", 0.0)
                state.pre_update_s = timeline_breakdown.get("pre_update", 0.0)
                state.finalize_s = timeline_breakdown.get("finalize", 0.0)

                INGESTION_COLD_START_DURATION_SECONDS.labels(model=m).set(state.cold_start_s)
                INGESTION_PREPARE_STORE_DURATION_SECONDS.labels(model=m).set(state.prep_store_s)
                INGESTION_PRE_UPDATE_DURATION_SECONDS.labels(model=m).set(state.pre_update_s)
                INGESTION_FINALIZATION_DURATION_SECONDS.labels(model=m).set(state.finalize_s)

        INGESTION_CYCLE_DURATION_SECONDS.labels(model=m).set(duration_s)
        INGESTION_STUCK_WARNING.labels(model=m).set(0.0)

    def check_stuck(self, model: str) -> IngestionStuckReport:
        """Evaluate if the pipeline is currently stuck without forward progress."""
        m = model.lower()
        now_mono = time.monotonic()
        with self._lock:
            state = self._states.get(m)
            if state is None or state.current_status not in ("running", "finalizing"):
                return IngestionStuckReport(
                    model=m,
                    is_stuck=False,
                    reason=None,
                    stuck_duration_seconds=0.0,
                )

            tracker = state.active_tracker
            if tracker is None:
                # No active tracker, check overall progress timeout
                stuck_dur = now_mono - state.last_progress_ts
                is_stuck = stuck_dur > self.STUCK_DOWNLOAD_TIMEOUT_S
                INGESTION_STUCK_WARNING.labels(model=m).set(1.0 if is_stuck else 0.0)
                return IngestionStuckReport(
                    model=m,
                    is_stuck=is_stuck,
                    reason="No progress recorded for active run" if is_stuck else None,
                    stuck_duration_seconds=stuck_dur,
                )

            snap = tracker.get_snapshot()

            # 1. Download stuck: queued/active remain, but no download progress
            if snap.download_active > 0 or snap.download_queued > 0:
                dl_idle = now_mono - state.last_download_ts
                if dl_idle > self.STUCK_DOWNLOAD_TIMEOUT_S:
                    INGESTION_STUCK_WARNING.labels(model=m).set(1.0)
                    return IngestionStuckReport(
                        model=m,
                        is_stuck=True,
                        reason=f"Download stage stalled for {dl_idle:.1f}s ({snap.download_active} active, {snap.download_queued} queued)",
                        stuck_duration_seconds=dl_idle,
                    )

            # 2. Decode stuck: queued/active remain, but no decode progress
            if snap.decode_active > 0 or snap.decode_queued > 0:
                dec_idle = now_mono - state.last_decode_ts
                if dec_idle > self.STUCK_DECODE_TIMEOUT_S:
                    INGESTION_STUCK_WARNING.labels(model=m).set(1.0)
                    return IngestionStuckReport(
                        model=m,
                        is_stuck=True,
                        reason=f"Decode stage stalled for {dec_idle:.1f}s ({snap.decode_active} active, {snap.decode_queued} queued)",
                        stuck_duration_seconds=dec_idle,
                    )

            # 3. Write stuck: waiting/active remain, but no write progress
            if snap.write_active > 0 or snap.write_waiting > 0:
                wr_idle = now_mono - state.last_write_ts
                if wr_idle > self.STUCK_WRITE_TIMEOUT_S:
                    INGESTION_STUCK_WARNING.labels(model=m).set(1.0)
                    return IngestionStuckReport(
                        model=m,
                        is_stuck=True,
                        reason=f"Write stage stalled for {wr_idle:.1f}s ({snap.write_active} active, {snap.write_waiting} waiting)",
                        stuck_duration_seconds=wr_idle,
                    )

            # 4. Finalize stuck
            if snap.finalize_state == "active" and snap.finalize_start_time is not None:
                fin_dur = now_mono - snap.finalize_start_time
                if fin_dur > self.STUCK_FINALIZE_TIMEOUT_S:
                    INGESTION_STUCK_WARNING.labels(model=m).set(1.0)
                    return IngestionStuckReport(
                        model=m,
                        is_stuck=True,
                        reason=f"Finalization stalled for {fin_dur:.1f}s",
                        stuck_duration_seconds=fin_dur,
                    )

        INGESTION_STUCK_WARNING.labels(model=m).set(0.0)
        return IngestionStuckReport(
            model=m,
            is_stuck=False,
            reason=None,
            stuck_duration_seconds=0.0,
        )

    def _serving_horizon_start(self, model: str, now_utc: datetime) -> datetime:
        """Return the oldest cycle time still inside the model's serving horizon."""
        max_lead = model_max_lead_hours(
            model, version_string=self.version_string, default_if_unknown=0
        )
        return serving_start_valid_time(now_utc) - timedelta(hours=max_lead)

    def _synoptic_ladder(self, model: str, now_utc: datetime, cadence: int) -> list[datetime]:
        """Return cycle times from the newest synoptic cycle back to the horizon start."""
        oldest = self._serving_horizon_start(model, now_utc)
        ladder: list[datetime] = []
        current = latest_synoptic_cycle(now_utc, cadence_hours=cadence)
        while current >= oldest:
            ladder.append(current)
            current = current - timedelta(hours=cadence)
        return ladder

    def read_cycle_facts(self, model: str, *, now: datetime) -> list[CycleFact]:
        """Read per-cycle catalog facts for one model inside the serving horizon.

        This is the single seam between lag/readiness evaluation and the
        catalog, so callers that need deterministic inputs (tests, future
        multi-source probes) can substitute their own implementation instead of
        mocking SQL. Returns an empty list when no engine is wired or the query
        fails — "unknown", never a fabricated value.

        Args:
            model: Platform model identifier (``gfs``/``gefs``).
            now: The reference UTC time bounding the serving horizon.

        Returns:
            One :class:`CycleFact` per cycle with a ``model_runs`` row, newest
            first.
        """
        m = model.lower().strip()
        if self.engine is None:
            return []
        expected_leads = canonical_lead_time_hours(
            m, version_string=self.version_string, default_if_unknown=()
        )
        expected_members = get_expected_members(m, default_if_unknown=1)
        min_cycle = self._serving_horizon_start(m, now)
        try:
            conn_ctx = (
                self.engine.connect()
                if hasattr(self.engine, "connect")
                else nullcontext(self.engine)
            )
            with conn_ctx as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT r.cycle_time AS cycle_time,
                               r.created_at AS created_at,
                               r.status AS status,
                               p.lead_time_hours AS lead_time_hours,
                               COUNT(DISTINCT emp.member_index) AS member_count
                        FROM model_runs r
                        JOIN model_versions v ON v.id = r.model_version_id
                        LEFT JOIN forecast_products p ON p.run_id = r.id
                        LEFT JOIN ensemble_member_products emp
                               ON emp.run_id = r.id
                              AND emp.lead_time_hours = p.lead_time_hours
                        WHERE v.model_id = :mid
                          AND v.version_string = :ver
                          AND r.cycle_time >= :min_cycle
                        GROUP BY r.cycle_time, r.created_at, r.status, p.lead_time_hours
                        """
                    ),
                    {"mid": m, "ver": self.version_string, "min_cycle": min_cycle},
                ).fetchall()
        except Exception as exc:
            logger.debug("Failed reading cycle facts for %s: %s", m, exc)
            return []

        per_cycle: dict[datetime, dict[str, Any]] = {}
        for row in rows:
            cycle_utc = _as_utc(row.cycle_time)
            entry = per_cycle.setdefault(
                cycle_utc,
                {
                    "created_at": _as_utc(row.created_at),
                    "status": str(row.status),
                    "leads": {},
                },
            )
            if row.lead_time_hours is not None:
                entry["leads"][int(row.lead_time_hours)] = int(row.member_count or 0)

        facts: list[CycleFact] = []
        for cycle_utc, entry in per_cycle.items():
            leads: dict[int, int] = entry["leads"]
            if expected_members <= 1:
                servable = bool(leads)
            else:
                servable = any(
                    is_lead_servable(count, expected_members)
                    for count in leads.values()
                )
            complete = bool(expected_leads) and set(expected_leads).issubset(leads)
            if complete and expected_members > 1:
                complete = all(
                    leads.get(lead, 0) >= expected_members for lead in expected_leads
                )
            facts.append(
                CycleFact(
                    cycle_time=cycle_utc,
                    run_created_at=entry["created_at"],
                    status=entry["status"],
                    committed_leads=len(leads),
                    servable=servable,
                    complete=complete,
                )
            )
        return sorted(facts, key=lambda fact: fact.cycle_time, reverse=True)

    def evaluate_lag(
        self,
        model: str,
        upstream_latest_cycle: datetime | None = None,
        now: datetime | None = None,
    ) -> IngestionLagReport:
        """Evaluate ingestion lag against the newest cycle that should be complete.

        **Target selection (the f000 fill-budget rule).** A cycle counts as
        "should already be complete" only once
        ``now > model_runs.created_at + INGESTION_FILL_IN_GRACE_SECONDS``. The
        anchor is ``model_runs.created_at``: the wave runner creates the row
        lazily when the cycle's first lead settles and is published, so it
        marks the moment the platform finished ingesting f000. There is no
        per-lead commit timestamp in the schema (``forecast_products`` carries
        no time column and the commit manifest only holds fingerprints), which
        makes this the only durable anchor the platform has. Before the budget
        expires the cycle is simply still filling and is not lag — which is
        why the target cannot come from the wall clock, since a clock-derived
        target reports a freshly published cycle as lagging immediately.

        A cycle with no ``model_runs`` row has no anchor: it is reported
        through :attr:`latest_missing_run_cycle` and
        :attr:`data_missing_cycles`, never through a clock-extrapolated lag.

        **Baseline.** The baseline is the newest *servable* cycle — one with at
        least one lead meeting the coverage contract — not the newest
        ``ready`` cycle. A cycle that is fully ingested but whose status was
        never promoted still serves traffic; using ``ready`` as the baseline
        translates a bookkeeping defect into an apparent ingestion shortfall.

        ``ingestion_lag_cycles``/``_hours`` are written only when the lag is a
        real measurement; ``weather_ingestion_lag_known`` carries the 0/1
        signal explicitly, because an untouched Prometheus gauge keeps its last
        value.

        Args:
            model: Platform model identifier (``gfs``/``gefs``).
            upstream_latest_cycle: Optional explicit upstream cycle. Retained
                for a future upstream-discovery path and for callers that
                already know what the center published; when omitted the target
                comes from the local f000 anchor.
            now: Optional injected current UTC time.

        Returns:
            The :class:`IngestionLagReport`.
        """
        m = model.lower()
        now_utc = now or datetime.now(timezone.utc)
        cadence = canonical_cycle_cadence_hours(m, default_if_unknown=6)
        expected_cycle = latest_synoptic_cycle(now_utc, cadence_hours=cadence)

        facts = self.read_cycle_facts(m, now=now_utc)
        by_cycle = {fact.cycle_time: fact for fact in facts}

        servable_cycles = [f.cycle_time for f in facts if f.servable]
        latest_servable = max(servable_cycles) if servable_cycles else None
        ready_cycles = [f.cycle_time for f in facts if f.status == "ready"]
        latest_ready = max(ready_cycles) if ready_cycles else None

        grace = timedelta(seconds=self.fill_grace_seconds)
        publication = timedelta(seconds=self.publication_delay_seconds)

        def _due_at(cycle_utc: datetime) -> datetime:
            """The moment a cycle is expected to be complete."""
            fact = by_cycle.get(cycle_utc)
            anchor = fact.run_created_at if fact is not None else cycle_utc + publication
            return anchor + grace

        due_cycles = [f.cycle_time for f in facts if now_utc >= _due_at(f.cycle_time)]
        anchor_target = max(due_cycles) if due_cycles else None
        target_cycle = (
            _as_utc(upstream_latest_cycle)
            if upstream_latest_cycle is not None
            else anchor_target
        )

        lag_known = target_cycle is not None and latest_servable is not None
        if lag_known:
            assert target_cycle is not None and latest_servable is not None
            lag_hours = max(
                0.0, (target_cycle - latest_servable).total_seconds() / 3600.0
            )
            lag_cycles = int(lag_hours // cadence) if cadence > 0 else 0
        else:
            lag_hours = 0.0
            lag_cycles = 0
        lag_hours = round(lag_hours, 2)

        ladder = self._synoptic_ladder(m, now_utc, cadence)
        missing_run = [
            cycle
            for cycle in ladder
            if cycle not in by_cycle and now_utc >= _due_at(cycle)
        ]
        # Data absence is measured from the newest cycle the platform can
        # actually serve (or, when nothing is servable, the newest cycle it has
        # any row for). Cycles older than that reference are accounted for;
        # everything newer that passed its deadline with no servable data is
        # the current gap. Counting the whole horizon instead would report
        # thousands of long-expired cycles and drown the signal.
        reference_cycle = latest_servable or (max(by_cycle) if by_cycle else None)
        data_missing_cycles = sum(
            1
            for cycle in ladder
            if now_utc >= _due_at(cycle)
            and (reference_cycle is None or cycle > reference_cycle)
            and not (by_cycle[cycle].servable if cycle in by_cycle else False)
        )

        # Readiness view: the shape of a promotion defect. A cycle can be
        # complete in the catalog while its status was never promoted, which
        # neither the lag gauge nor a status-based query can see. A complete
        # cycle is only alerted on once it is a full fill window past its
        # deadline, so the brief window where a wave fence has downgraded an
        # otherwise complete run is not reported as a fault.
        complete_not_ready = [
            fact for fact in facts if fact.complete and fact.status != "ready"
        ]
        complete_not_ready_cycles = len(complete_not_ready)
        oldest_probe_overdue = (
            max(
                max(0.0, (now_utc - _due_at(fact.cycle_time)).total_seconds())
                for fact in complete_not_ready
            )
            if complete_not_ready
            else None
        )
        target_fact = by_cycle.get(target_cycle) if target_cycle is not None else None
        target_overdue = (
            max(0.0, (now_utc - _due_at(target_cycle)).total_seconds())
            if target_cycle is not None
            else None
        )

        INGESTION_LAG_KNOWN.labels(model=m).set(1.0 if lag_known else 0.0)
        if lag_known:
            INGESTION_LAG_CYCLES.labels(model=m).set(float(lag_cycles))
            INGESTION_LAG_HOURS.labels(model=m).set(lag_hours)
        INGESTION_DATA_MISSING_CYCLES.labels(model=m).set(float(data_missing_cycles))
        INGESTION_CYCLES_COMPLETE_NOT_READY.labels(model=m).set(
            float(complete_not_ready_cycles)
        )

        return IngestionLagReport(
            model=m,
            latest_expected_cycle=expected_cycle,
            latest_upstream_cycle=(
                _as_utc(upstream_latest_cycle)
                if upstream_latest_cycle is not None
                else None
            ),
            latest_servable_cycle=latest_servable,
            latest_ready_cycle=latest_ready,
            upstream_available=upstream_latest_cycle is not None,
            lag_known=lag_known,
            lag_cycles=lag_cycles,
            lag_hours=lag_hours,
            data_missing_cycles=data_missing_cycles,
            is_behind=lag_known and lag_cycles > 0,
            lag_target_cycle=target_cycle,
            latest_missing_run_cycle=max(missing_run) if missing_run else None,
            lag_target_ready=(
                target_fact.status == "ready" if target_fact is not None else None
            ),
            lag_target_overdue_seconds=target_overdue,
            complete_not_ready_cycles=complete_not_ready_cycles,
            oldest_complete_not_ready_overdue_seconds=oldest_probe_overdue,
        )

    def evaluate_model_completeness(
        self,
        model: str,
        version_string: str = "v1.0",
        cycle_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Check logical completeness using authoritative per-model/version horizons and member contracts."""
        m = model.lower().strip()
        expected_leads_seq = canonical_lead_time_hours(m, version_string=version_string, default_if_unknown=())
        expected_leads_count = len(expected_leads_seq)
        max_lead_hours = model_max_lead_hours(m, version_string=version_string, default_if_unknown=0)
        expected_members = get_expected_members(m, default_if_unknown=1)

        MODEL_EXPECTED_LEADS.labels(model=m).set(float(expected_leads_count))
        MODEL_MAX_LEAD_HOURS.labels(model=m).set(float(max_lead_hours))

        if self.engine is None:
            return {
                "model": m,
                "version": version_string,
                "cycle_time": None,
                "committed_leads": 0,
                "expected_leads": expected_leads_count,
                "max_lead_hours": max_lead_hours,
                "members": 0,
                "expected_members": expected_members,
                "servable": False,
                "ready": False,
            }

        try:
            conn_ctx = self.engine.connect() if hasattr(self.engine, "connect") else nullcontext(self.engine)
            with conn_ctx as conn:
                if expected_members > 1:
                    # Ensemble model (e.g. GEFS)
                    q = text(
                        """
                        SELECT r.cycle_time,
                               COUNT(DISTINCT m.member_index) AS member_count,
                               COUNT(DISTINCT p.lead_time_hours) AS committed_leads,
                               r.status
                        FROM model_runs r
                        JOIN model_versions v ON v.id = r.model_version_id
                        LEFT JOIN ensemble_members m ON m.run_id = r.id AND m.member_index BETWEEN 1 AND :max_member
                        LEFT JOIN forecast_products p ON p.run_id = r.id
                        WHERE v.model_id = :mid
                          AND (:ctime IS NULL OR r.cycle_time = :ctime)
                        GROUP BY r.id, r.cycle_time, r.status
                        ORDER BY r.cycle_time DESC
                        LIMIT 1
                        """
                    )
                    row = conn.execute(q, {"mid": m, "max_member": expected_members, "ctime": cycle_time}).fetchone()
                    if row is None:
                        return {
                            "model": m,
                            "version": version_string,
                            "cycle_time": None,
                            "committed_leads": 0,
                            "expected_leads": expected_leads_count,
                            "max_lead_hours": max_lead_hours,
                            "members": 0,
                            "expected_members": expected_members,
                            "servable": False,
                            "ready": False,
                        }

                    member_cnt = int(row.member_count or 0)
                    committed_leads = int(row.committed_leads or 0)
                    is_servable = is_lead_servable(member_cnt, expected_members)
                    is_ready = (
                        member_cnt >= expected_members
                        and (committed_leads >= expected_leads_count if expected_leads_count > 0 else True)
                        and row.status == "ready"
                    )

                    if m == "gefs":
                        GEFS_AVAILABLE_MEMBERS.set(float(member_cnt))
                        GEFS_SERVABLE_STATUS.set(1.0 if is_servable else 0.0)
                        GEFS_READY_STATUS.set(1.0 if is_ready else 0.0)

                    MODEL_COMMITTED_LEADS.labels(model=m).set(float(committed_leads))
                    MODEL_READY_STATUS.labels(model=m).set(1.0 if is_ready else 0.0)

                    return {
                        "model": m,
                        "version": version_string,
                        "cycle_time": row.cycle_time,
                        "members": member_cnt,
                        "expected_members": expected_members,
                        "committed_leads": committed_leads,
                        "expected_leads": expected_leads_count,
                        "max_lead_hours": max_lead_hours,
                        "servable": is_servable,
                        "ready": is_ready,
                        "status": row.status,
                    }
                else:
                    # Deterministic model (e.g. GFS)
                    q = text(
                        """
                        SELECT r.cycle_time,
                               COUNT(DISTINCT p.lead_time_hours) AS committed_leads,
                               r.status
                        FROM model_runs r
                        JOIN model_versions v ON v.id = r.model_version_id
                        LEFT JOIN forecast_products p ON p.run_id = r.id
                        WHERE v.model_id = :mid
                          AND (:ctime IS NULL OR r.cycle_time = :ctime)
                        GROUP BY r.id, r.cycle_time, r.status
                        ORDER BY r.cycle_time DESC
                        LIMIT 1
                        """
                    )
                    row = conn.execute(q, {"mid": m, "ctime": cycle_time}).fetchone()
                    if row is None:
                        return {
                            "model": m,
                            "version": version_string,
                            "cycle_time": None,
                            "committed_leads": 0,
                            "expected_leads": expected_leads_count,
                            "max_lead_hours": max_lead_hours,
                            "members": 1,
                            "expected_members": 1,
                            "servable": False,
                            "ready": False,
                        }

                    committed_leads = int(row.committed_leads or 0)
                    is_servable = committed_leads > 0
                    is_ready = (
                        (committed_leads >= expected_leads_count if expected_leads_count > 0 else True)
                        and row.status == "ready"
                    )

                    MODEL_COMMITTED_LEADS.labels(model=m).set(float(committed_leads))
                    MODEL_READY_STATUS.labels(model=m).set(1.0 if is_ready else 0.0)

                    return {
                        "model": m,
                        "version": version_string,
                        "cycle_time": row.cycle_time,
                        "members": 1,
                        "expected_members": 1,
                        "committed_leads": committed_leads,
                        "expected_leads": expected_leads_count,
                        "max_lead_hours": max_lead_hours,
                        "servable": is_servable,
                        "ready": is_ready,
                        "status": row.status,
                    }
        except Exception as exc:
            logger.debug("Failed model completeness check for %s: %s", m, exc)
            return {
                "model": m,
                "version": version_string,
                "error": str(exc),
                "servable": False,
                "ready": False,
            }

    def evaluate_gefs_completeness(self, cycle_time: datetime | None = None) -> dict[str, Any]:
        """Check GEFS perturbation member availability (1..30), servability (>=26), ready (30)."""
        return self.evaluate_model_completeness("gefs", cycle_time=cycle_time)

    def evaluate_gfs_completeness(self, cycle_time: datetime | None = None) -> dict[str, Any]:
        """Check GFS lead availability against authoritative canonical horizon."""
        return self.evaluate_model_completeness("gfs", cycle_time=cycle_time)

    def get_model_state(self, model: str) -> ModelIngestionState | None:
        with self._lock:
            return self._states.get(model.lower())


#: Global ingestion health collector singleton
INGESTION_COLLECTOR = IngestionHealthCollector()
