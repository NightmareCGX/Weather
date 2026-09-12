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
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from domain.cadence import canonical_cycle_cadence_hours
from domain.coverage import get_expected_members, is_lead_servable
from domain.horizon import canonical_lead_time_hours, model_max_lead_hours
from ingestion.core.observability import PipelineProgressTracker
from ingestion.monitoring.metrics import REGISTRY

logger = logging.getLogger(__name__)


def latest_synoptic_cycle(now_utc: datetime, cadence_hours: int = 6) -> datetime:
    """Return the latest synoptic cycle initialization time (00Z, 06Z, 12Z, 18Z) <= now_utc."""
    hour = (now_utc.hour // cadence_hours) * cadence_hours
    return now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)

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
    "Ingestion lag in hours relative to available upstream data",
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
class IngestionLagReport:
    """Lag analysis comparing expected, upstream, and local catalog states."""

    model: str
    latest_expected_cycle: datetime
    latest_upstream_cycle: datetime | None
    latest_ready_cycle: datetime | None
    upstream_available: bool
    lag_cycles: int
    lag_hours: float
    is_behind: bool


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

    def __init__(self, engine: Engine | Connection | None = None) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._states: dict[str, ModelIngestionState] = {
            "gfs": ModelIngestionState(model="gfs"),
            "gefs": ModelIngestionState(model="gefs"),
        }

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

    def evaluate_lag(
        self,
        model: str,
        upstream_latest_cycle: datetime | None = None,
        now: datetime | None = None,
    ) -> IngestionLagReport:
        """Evaluate ingestion lag relative to available upstream publication."""
        m = model.lower()
        now_utc = now or datetime.now(timezone.utc)

        # Expected latest 6h synoptic cycle (00Z, 06Z, 12Z, 18Z)
        cadence = canonical_cycle_cadence_hours(m, default_if_unknown=6)
        expected_cycle = latest_synoptic_cycle(now_utc, cadence_hours=cadence)

        # Ready cycle from PostgreSQL catalog
        ready_cycle: datetime | None = None
        if self.engine is not None:
            try:
                conn_ctx = self.engine.connect() if hasattr(self.engine, "connect") else nullcontext(self.engine)
                with conn_ctx as conn:
                    q = text(
                        """
                        SELECT MAX(r.cycle_time) AS max_ready
                        FROM model_runs r
                        JOIN model_versions v ON v.id = r.model_version_id
                        WHERE v.model_id = :mid AND r.status = 'ready'
                        """
                    )
                    res = conn.execute(q, {"mid": m}).scalar()
                    if res is not None:
                        ready_cycle = (
                            res if res.tzinfo is not None else res.replace(tzinfo=timezone.utc)
                        )
            except Exception as exc:
                logger.debug("Failed query for latest ready cycle: %s", exc)

        upstream_avail = upstream_latest_cycle is not None
        target_cycle = upstream_latest_cycle or expected_cycle

        if ready_cycle is not None:
            diff_hours = max(0.0, (target_cycle - ready_cycle).total_seconds() / 3600.0)
            lag_cycles = int(diff_hours // cadence) if cadence > 0 else 0
        else:
            # No ready runs in catalog
            diff_hours = max(0.0, (now_utc - target_cycle).total_seconds() / 3600.0)
            lag_cycles = 1

        # Only consider "behind" if upstream has published a cycle that we do not have ready
        is_behind = upstream_avail and (lag_cycles > 0)

        INGESTION_LAG_CYCLES.labels(model=m).set(float(lag_cycles))
        INGESTION_LAG_HOURS.labels(model=m).set(round(diff_hours, 2))

        return IngestionLagReport(
            model=m,
            latest_expected_cycle=expected_cycle,
            latest_upstream_cycle=upstream_latest_cycle,
            latest_ready_cycle=ready_cycle,
            upstream_available=upstream_avail,
            lag_cycles=lag_cycles,
            lag_hours=round(diff_hours, 2),
            is_behind=is_behind,
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
