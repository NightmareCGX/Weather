"""Alerting engine with severity model, deduplication, cooldown, and recovery notifications.

Implements INFO / WARNING / CRITICAL severities, state-transition notifications,
cooldown suppression to avoid alert spam, and pluggable delivery sinks (structured
logging, optional webhook, in-memory ring buffer). All alert operations are fail-open
relative to production execution.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ingestion.monitoring.metrics import REGISTRY

logger = logging.getLogger(__name__)

#: Cumulative count of webhook deliveries that failed after all retries.
#: Process-local (like every registry metric): exposed on the exporter and
#: daemon metrics surfaces (MONITORING.md section 2.9) so "alerts are being
#: lost" is itself observable, and surfaced as the ``alert_delivery_failed``
#: self-alert through the surviving sinks.
ALERT_DELIVERY_FAILURES_TOTAL = REGISTRY.counter(
    "weather_alert_delivery_failures_total",
    "Alert webhook deliveries that failed after all retries (fail-open)",
)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Alert:
    """An alert triggered by an evaluation rule."""

    name: str
    severity: AlertSeverity
    summary: str
    description: str
    scope: str = "platform"
    value: Any = None
    threshold: Any = None
    timestamp: float = field(default_factory=time.time)
    runbook_anchor: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.name, self.scope)


@dataclass(frozen=True)
class AlertEvent:
    """A notification event representing an alert trigger, escalation, or recovery."""

    event_type: str  # "triggered", "escalated", "recovered"
    alert: Alert
    previous_severity: AlertSeverity | None = None
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class AlertSink(Protocol):
    """Protocol for pluggable alert dispatch sinks."""

    def emit(self, event: AlertEvent) -> None:
        ...


class LogAlertSink:
    """Dispatches alerts to Python logging with structured severity prefixes."""

    def emit(self, event: AlertEvent) -> None:
        alert = event.alert
        if event.event_type == "recovered":
            logger.info(
                "[ALERT:RECOVERY] %s (%s): %s recovered",
                alert.name,
                alert.scope,
                alert.summary,
            )
        elif alert.severity == AlertSeverity.CRITICAL:
            logger.critical(
                "[ALERT:CRITICAL] %s (%s): %s (value=%s, threshold=%s)",
                alert.name,
                alert.scope,
                alert.description,
                alert.value,
                alert.threshold,
            )
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(
                "[ALERT:WARNING] %s (%s): %s (value=%s, threshold=%s)",
                alert.name,
                alert.scope,
                alert.description,
                alert.value,
                alert.threshold,
            )
        else:
            logger.info(
                "[ALERT:INFO] %s (%s): %s",
                alert.name,
                alert.scope,
                alert.summary,
            )


class MemoryAlertSink:
    """Maintains a thread-safe in-memory ring buffer of recent alerts for status/diagnostics."""

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self._events: deque[AlertEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, event: AlertEvent) -> None:
        with self._lock:
            self._events.append(event)

    def get_recent(self, limit: int = 20) -> list[AlertEvent]:
        with self._lock:
            return list(self._events)[-limit:]


class WebhookAlertSink:
    """Optional HTTP webhook alert delivery (e.g. Slack, Discord, Alertmanager).

    Delivery is retried with exponential backoff (``retry_attempts`` total
    attempts, ``retry_backoff_seconds`` base delay, per-attempt
    ``timeout_seconds``). The sink still fails open: after the final failed
    attempt it records the failure on ``last_delivery_ok``, increments
    ``weather_alert_delivery_failures_total``, and returns without raising.
    The owning engine turns the failure into the ``alert_delivery_failed``
    self-alert through the surviving sinks (MONITORING.md section 5.2).
    """

    def __init__(
        self,
        webhook_url: str,
        timeout_seconds: float = 2.0,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if retry_attempts < 1:
            raise ValueError("retry_attempts must be >= 1")
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        #: Result of the most recent delivery: True = delivered (2xx),
        #: False = failed after all retries, None = never attempted.
        self.last_delivery_ok: bool | None = None
        self.last_delivery_error: str = ""

    def emit(self, event: AlertEvent) -> None:
        if not self.webhook_url:
            return
        payload = {
            "event_type": event.event_type,
            "name": event.alert.name,
            "severity": event.alert.severity.value,
            "summary": event.alert.summary,
            "description": event.alert.description,
            "scope": event.alert.scope,
            "value": event.alert.value,
            "threshold": event.alert.threshold,
            "timestamp": event.timestamp,
            "runbook": event.alert.runbook_anchor,
        }
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                import httpx

                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(self.webhook_url, json=payload)
                if response.is_success:
                    self.last_delivery_ok = True
                    self.last_delivery_error = ""
                    return
                # Non-2xx is a failed delivery (rate limits, gateway 5xx) and
                # is retried like a transport error.
                last_error = RuntimeError(f"webhook responded HTTP {response.status_code}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        self.last_delivery_ok = False
        self.last_delivery_error = str(last_error) if last_error else "unknown error"
        try:
            ALERT_DELIVERY_FAILURES_TOTAL.inc()
        except Exception:  # noqa: BLE001 - metrics must never break alerting
            pass
        logger.warning(
            "Failed to dispatch alert webhook to %s after %d attempt(s): %s",
            self.webhook_url,
            self.retry_attempts,
            self.last_delivery_error,
        )


class AlertDeduplicator:
    """Manages active alert states, cooldown suppression, escalation, and recovery events."""

    def __init__(self, default_cooldown_seconds: float = 3600.0) -> None:
        self.default_cooldown_seconds = default_cooldown_seconds
        self._active_alerts: dict[tuple[str, str], Alert] = {}
        self._last_notified_ts: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def process_alerts(
        self,
        current_alerts: list[Alert],
        now: float | None = None,
    ) -> list[AlertEvent]:
        """Process current alerts, deduping repeated warnings and creating recovery events."""
        current_ts = time.time() if now is None else now
        events: list[AlertEvent] = []
        current_keys = {a.key: a for a in current_alerts}

        with self._lock:
            # 1. Check for resolved / recovered alerts
            resolved_keys = [k for k in self._active_alerts if k not in current_keys]
            for key in resolved_keys:
                old_alert = self._active_alerts.pop(key)
                self._last_notified_ts.pop(key, None)
                recovery_alert = Alert(
                    name=old_alert.name,
                    severity=AlertSeverity.INFO,
                    summary=f"{old_alert.name} has recovered",
                    description=f"{old_alert.summary} is no longer firing.",
                    scope=old_alert.scope,
                    timestamp=current_ts,
                    runbook_anchor=old_alert.runbook_anchor,
                )
                events.append(
                    AlertEvent(
                        event_type="recovered",
                        alert=recovery_alert,
                        previous_severity=old_alert.severity,
                        timestamp=current_ts,
                    )
                )

            # 2. Process currently firing alerts
            for key, alert in current_keys.items():
                existing_alert = self._active_alerts.get(key)
                last_ts = self._last_notified_ts.get(key, 0.0)

                if existing_alert is None:
                    # New alert trigger
                    self._active_alerts[key] = alert
                    self._last_notified_ts[key] = current_ts
                    events.append(
                        AlertEvent(
                            event_type="triggered",
                            alert=alert,
                            previous_severity=None,
                            timestamp=current_ts,
                        )
                    )
                elif alert.severity == AlertSeverity.CRITICAL and existing_alert.severity == AlertSeverity.WARNING:
                    # Escalation from WARNING to CRITICAL: notify immediately!
                    self._active_alerts[key] = alert
                    self._last_notified_ts[key] = current_ts
                    events.append(
                        AlertEvent(
                            event_type="escalated",
                            alert=alert,
                            previous_severity=AlertSeverity.WARNING,
                            timestamp=current_ts,
                        )
                    )
                else:
                    # Same severity: check cooldown window
                    self._active_alerts[key] = alert
                    if current_ts - last_ts >= self.default_cooldown_seconds:
                        self._last_notified_ts[key] = current_ts
                        events.append(
                            AlertEvent(
                                event_type="triggered",
                                alert=alert,
                                previous_severity=existing_alert.severity,
                                timestamp=current_ts,
                            )
                        )
                    else:
                        # Suppress repeated notification within cooldown
                        pass

        return events

    def get_active_alerts(self) -> list[Alert]:
        with self._lock:
            return list(self._active_alerts.values())


def _resolve_default_delivery_settings() -> tuple[str, float, int]:
    """Resolve the alert delivery defaults from the ingestion settings.

    Returns ``(webhook_url, timeout_seconds, retry_attempts)``. Deliberately
    fail-open: an unavailable settings module degrades to the hardcoded
    defaults (empty webhook URL = delivery disabled).
    """
    try:
        from ingestion.core.config import settings

        return (
            str(getattr(settings, "ALERT_WEBHOOK_URL", "") or ""),
            float(getattr(settings, "ALERT_WEBHOOK_TIMEOUT_SECONDS", 2.0)),
            int(getattr(settings, "ALERT_WEBHOOK_RETRY_ATTEMPTS", 3)),
        )
    except Exception:  # noqa: BLE001 - defaults must never break construction
        return ("", 2.0, 3)


class AlertEngine:
    """Coordinates rules evaluation, deduplication, and multi-sink dispatching."""

    def __init__(
        self,
        cooldown_seconds: float = 3600.0,
        webhook_url: str | None = None,
        webhook_timeout_seconds: float | None = None,
        webhook_retry_attempts: int | None = None,
    ) -> None:
        default_url, default_timeout, default_retries = _resolve_default_delivery_settings()
        self.deduplicator = AlertDeduplicator(default_cooldown_seconds=cooldown_seconds)
        self.log_sink = LogAlertSink()
        self.memory_sink = MemoryAlertSink()
        self.webhook_sink = WebhookAlertSink(
            # An explicitly passed empty URL disables delivery; None defers to
            # the ALERT_WEBHOOK_URL setting.
            webhook_url=webhook_url if webhook_url is not None else default_url,
            timeout_seconds=(
                webhook_timeout_seconds
                if webhook_timeout_seconds is not None
                else default_timeout
            ),
            retry_attempts=(
                webhook_retry_attempts
                if webhook_retry_attempts is not None
                else default_retries
            ),
        )
        self._sinks: list[AlertSink] = [self.log_sink, self.memory_sink, self.webhook_sink]
        self._delivery_failure_lock = threading.Lock()
        self._last_delivery_failure_ts = 0.0

    def add_sink(self, sink: AlertSink) -> None:
        self._sinks.append(sink)

    def evaluate_rules(
        self,
        resource_data: dict[str, Any] | None = None,
        postgres_data: Any | None = None,
        lifecycle_data: Any | None = None,
        ingestion_data: dict[str, Any] | None = None,
        storage_data: Any | None = None,
        leak_data: Any | None = None,
        gc_data: dict[str, Any] | None = None,
    ) -> list[Alert]:
        """Evaluate all system, resource, database, lifecycle, and gc rules against snapshots."""
        alerts: list[Alert] = []

        # 1. Disk & Memory rules
        if resource_data is not None:
            disks = resource_data.get("disks", [])
            for d in disks:
                if d.is_critical:
                    alerts.append(
                        Alert(
                            name="disk_space_critical",
                            severity=AlertSeverity.CRITICAL,
                            summary=f"Disk space critical on {d.path} ({d.used_percent}%)",
                            description=f"Disk utilization has reached {d.used_percent}% on {d.path} ({d.free_bytes / (1024**3):.1f} GB free).",
                            scope=d.path,
                            value=d.used_percent,
                            threshold=90.0,
                            runbook_anchor="#disk-space-critical",
                        )
                    )
                elif d.is_warning:
                    alerts.append(
                        Alert(
                            name="disk_space_warning",
                            severity=AlertSeverity.WARNING,
                            summary=f"Disk space warning on {d.path} ({d.used_percent}%)",
                            description=f"Disk utilization is {d.used_percent}% on {d.path}.",
                            scope=d.path,
                            value=d.used_percent,
                            threshold=80.0,
                            runbook_anchor="#disk-space-warning",
                        )
                    )

        if leak_data is not None and getattr(leak_data, "is_leak", False):
            alerts.append(
                Alert(
                    name="memory_leak_warning",
                    severity=AlertSeverity.WARNING,
                    summary="Sustained memory leak detected across cycles",
                    description=leak_data.message,
                    scope="process",
                    value=leak_data.retained_growth_bytes,
                    threshold=leak_data.retained_growth_pct,
                    runbook_anchor="#memory-leak",
                )
            )

        # 2. PostgreSQL rules
        if postgres_data is not None:
            if not postgres_data.connected:
                alerts.append(
                    Alert(
                        name="postgres_unreachable",
                        severity=AlertSeverity.CRITICAL,
                        summary="PostgreSQL is unreachable",
                        description=f"Database connectivity failed: {postgres_data.error}",
                        scope="postgres",
                        runbook_anchor="#database-unreachable",
                    )
                )
            else:
                if postgres_data.is_connection_critical:
                    alerts.append(
                        Alert(
                            name="postgres_connections_critical",
                            severity=AlertSeverity.CRITICAL,
                            summary=f"PostgreSQL connections near exhaustion ({postgres_data.connection_utilization_pct}%)",
                            description=f"Active connections ({postgres_data.active_connections}/{postgres_data.max_connections}) reached {postgres_data.connection_utilization_pct}%.",
                            scope="postgres",
                            value=postgres_data.connection_utilization_pct,
                            threshold=95.0,
                            runbook_anchor="#database-connections",
                        )
                    )
                elif postgres_data.is_connection_warning:
                    alerts.append(
                        Alert(
                            name="postgres_connections_warning",
                            severity=AlertSeverity.WARNING,
                            summary=f"PostgreSQL connections high ({postgres_data.connection_utilization_pct}%)",
                            description=f"Active connections ({postgres_data.active_connections}/{postgres_data.max_connections}) reached {postgres_data.connection_utilization_pct}%.",
                            scope="postgres",
                            value=postgres_data.connection_utilization_pct,
                            threshold=80.0,
                            runbook_anchor="#database-connections",
                        )
                    )

                if postgres_data.lock_waits > 0:
                    alerts.append(
                        Alert(
                            name="postgres_lock_waits",
                            severity=AlertSeverity.WARNING,
                            summary=f"PostgreSQL lock waits detected ({postgres_data.lock_waits} sessions)",
                            description=f"{postgres_data.lock_waits} sessions are blocked waiting for locks.",
                            scope="postgres",
                            value=postgres_data.lock_waits,
                            threshold=0,
                            runbook_anchor="#database-lock-waits",
                        )
                    )

                if getattr(postgres_data, "deadlocks_delta", 0) > 0:
                    alerts.append(
                        Alert(
                            name="postgres_deadlocks_detected",
                            severity=AlertSeverity.WARNING,
                            summary=f"{postgres_data.deadlocks_delta} new PostgreSQL deadlocks detected",
                            description=f"{postgres_data.deadlocks_delta} new deadlock(s) occurred during the observation interval (cumulative total: {postgres_data.deadlocks}).",
                            scope="postgres",
                            value=postgres_data.deadlocks_delta,
                            threshold=0,
                            runbook_anchor="#database-deadlocks",
                        )
                    )

        # 3. Object Storage rules
        if storage_data is not None and not storage_data.connected:
            alerts.append(
                Alert(
                    name="minio_unreachable",
                    severity=AlertSeverity.CRITICAL,
                    summary="Object storage (MinIO/S3) is unreachable",
                    description=f"MinIO health probe failed on endpoint {storage_data.endpoint}: {storage_data.error}",
                    scope="storage",
                    runbook_anchor="#minio-unreachable",
                )
            )

        # 4. Lifecycle & Reclamation rules
        if lifecycle_data is not None:
            if lifecycle_data.stuck_claims_critical > 0:
                alerts.append(
                    Alert(
                        name="finalizer_claim_stuck_critical",
                        severity=AlertSeverity.CRITICAL,
                        summary=f"{lifecycle_data.stuck_claims_critical} cycle deletion claims stuck > 4 hours",
                        description=f"Physical cycle finalizer has stuck claims older than 4 hours (oldest: {lifecycle_data.oldest_claim_age_s / 3600.0:.1f}h).",
                        scope="lifecycle",
                        value=lifecycle_data.oldest_claim_age_s,
                        threshold=14400.0,
                        runbook_anchor="#stuck-deletion-claim",
                    )
                )
            elif lifecycle_data.stuck_claims_warning > 0:
                alerts.append(
                    Alert(
                        name="finalizer_claim_stuck_warning",
                        severity=AlertSeverity.WARNING,
                        summary=f"{lifecycle_data.stuck_claims_warning} cycle deletion claims older than 1 hour",
                        description=f"Deletion claim age ({lifecycle_data.oldest_claim_age_s / 3600.0:.1f}h) exceeds 1 hour.",
                        scope="lifecycle",
                        value=lifecycle_data.oldest_claim_age_s,
                        threshold=3600.0,
                        runbook_anchor="#stuck-deletion-claim",
                    )
                )

            # Metadata sweeper backlog overdue
            if lifecycle_data.sweeper_unpurged_count > 0 and lifecycle_data.sweeper_oldest_overdue_s > 86400.0:
                alerts.append(
                    Alert(
                        name="metadata_sweeper_backlog_overdue",
                        severity=AlertSeverity.WARNING,
                        summary=f"Metadata sweeper backlog overdue ({lifecycle_data.sweeper_unpurged_count} cycles)",
                        description=(
                            f"{lifecycle_data.sweeper_unpurged_count} tombstones retain detailed metadata past retention deadline "
                            f"(oldest overdue by {lifecycle_data.sweeper_oldest_overdue_s / 86400.0:.1f} days)."
                        ),
                        scope="sweeper",
                        value=lifecycle_data.sweeper_oldest_overdue_s,
                        threshold=86400.0,
                        runbook_anchor="#sweeper-backlog",
                    )
                )

            # Reclamation failures
            if lifecycle_data.reclamation.failed_count > 0:
                alerts.append(
                    Alert(
                        name="reclamation_failed_shards",
                        severity=AlertSeverity.WARNING,
                        summary=f"{lifecycle_data.reclamation.failed_count} reclamation targets in failed quarantine",
                        description=f"Granular shard reclamation has {lifecycle_data.reclamation.failed_count} failed records in reclamation_queue.",
                        scope="reclamation",
                        value=lifecycle_data.reclamation.failed_count,
                        threshold=0,
                        runbook_anchor="#reclamation-failures",
                    )
                )

            if lifecycle_data.reclamation.oldest_deleting_age_s > 600.0:
                alerts.append(
                    Alert(
                        name="reclamation_deleting_stuck",
                        severity=AlertSeverity.WARNING,
                        summary="Reclamation worker lease expired / stuck deleting",
                        description=f"Reclamation deleting target has been leased for {lifecycle_data.reclamation.oldest_deleting_age_s:.1f}s (>600s).",
                        scope="reclamation",
                        value=lifecycle_data.reclamation.oldest_deleting_age_s,
                        threshold=600.0,
                        runbook_anchor="#reclamation-stuck",
                    )
                )

            # Invariant / Anti-resurrection violations
            for v in lifecycle_data.violations:
                sev = AlertSeverity.CRITICAL
                alerts.append(
                    Alert(
                        name=f"invariant_{v.violation_type}",
                        severity=sev,
                        summary=f"Data Lifecycle contract violation: {v.violation_type}",
                        description=f"{v.description} on model={v.model_id} cycle={v.cycle_time}",
                        scope=v.model_id,
                        runbook_anchor="#anti-resurrection-violation",
                    )
                )

        # 5. Ingestion stuck & lag rules
        if ingestion_data is not None:
            for model_id, mdata in ingestion_data.items():
                stuck = mdata.get("stuck")
                if stuck is not None and stuck.is_stuck:
                    alerts.append(
                        Alert(
                            name="ingestion_pipeline_stuck",
                            severity=AlertSeverity.CRITICAL,
                            summary=f"Ingestion pipeline stuck for {model_id.upper()}",
                            description=f"Pipeline forward progress has stalled: {stuck.reason}",
                            scope=model_id,
                            value=stuck.stuck_duration_seconds,
                            threshold=600.0,
                            runbook_anchor="#ingestion-stuck",
                        )
                    )

                lag = mdata.get("lag")
                if lag is not None and lag.is_behind:
                    if lag.lag_cycles >= 2:
                        alerts.append(
                            Alert(
                                name="ingestion_lag_critical",
                                severity=AlertSeverity.CRITICAL,
                                summary=f"Ingestion {model_id.upper()} is {lag.lag_cycles} cycles behind upstream",
                                description=f"Local catalog is {lag.lag_cycles} cycles behind NOAA upstream publication ({lag.lag_hours:.1f} hours).",
                                scope=model_id,
                                value=lag.lag_cycles,
                                threshold=2,
                                runbook_anchor="#ingestion-lag",
                            )
                        )
                    elif lag.lag_cycles == 1:
                        alerts.append(
                            Alert(
                                name="ingestion_lag_warning",
                                severity=AlertSeverity.WARNING,
                                summary=f"Ingestion {model_id.upper()} is 1 cycle behind upstream",
                                description=f"Local catalog is 1 cycle behind available NOAA upstream ({lag.lag_hours:.1f} hours).",
                                scope=model_id,
                                value=lag.lag_cycles,
                                threshold=1,
                                runbook_anchor="#ingestion-lag",
                            )
                        )

                # Stalled readiness: the newest cycle that should already be
                # complete is still not `ready` a full fill window past its
                # deadline. The lag gate above cannot see this class of defect —
                # a cycle can be fully ingested and servable (lag 0) while its
                # status was never promoted, which is exactly how a promotion
                # bug stays invisible in every other view.
                #
                # The verdict is deliberately about the due cycle and not about
                # `weather_model_ready_status`, whose newest run is *expected*
                # to be unready while it fills.
                fill_grace = float(mdata.get("fill_grace_seconds") or 0.0)
                overdue = lag.lag_target_overdue_seconds if lag is not None else None
                if (
                    lag is not None
                    and lag.lag_target_ready is False
                    and overdue is not None
                    and fill_grace > 0.0
                    and overdue >= fill_grace
                ):
                    alerts.append(
                        Alert(
                            name="model_ready_not_promoted",
                            severity=AlertSeverity.WARNING,
                            summary=(
                                f"{model_id.upper()} cycle "
                                f"{overdue / 3600.0:.1f}h past its fill deadline "
                                "without being promoted to ready"
                            ),
                            description=(
                                f"The newest cycle that should already be complete "
                                f"({lag.lag_target_cycle}) is still not 'ready' "
                                f"{overdue / 3600.0:.1f}h after its f000 ingest anchor plus the "
                                f"{fill_grace / 3600.0:.1f}h fill budget. Data may be complete; "
                                "readiness promotion is stalled."
                            ),
                            scope=model_id,
                            value=overdue,
                            threshold=fill_grace,
                            runbook_anchor="#model-ready-not-promoted",
                        )
                    )

                # Direct probe for the same defect class: cycles whose catalog
                # contents are complete but whose status is not `ready`. A
                # single stuck cycle is invisible to the lag gauge (it serves
                # traffic), so this is what surfaces it within minutes rather
                # than through an operator noticing a stale badge.
                probe = int(mdata.get("cycles_complete_not_ready") or 0)
                probe_overdue = mdata.get("oldest_complete_not_ready_overdue_seconds")
                if (
                    probe > 0
                    and probe_overdue is not None
                    and fill_grace > 0.0
                    and float(probe_overdue) >= fill_grace
                ):
                    alerts.append(
                        Alert(
                            name="cycles_complete_not_ready",
                            severity=AlertSeverity.WARNING,
                            summary=(
                                f"{probe} {model_id.upper()} cycle(s) complete in the "
                                "catalog but not promoted to ready"
                            ),
                            description=(
                                f"{probe} cycle(s) hold every expected lead (and member) in the "
                                f"catalog while their run status is not 'ready'; the oldest has "
                                f"been past its fill deadline for {float(probe_overdue) / 3600.0:.1f}h. "
                                "This is a readiness-promotion defect, not an ingestion shortfall."
                            ),
                            scope=model_id,
                            value=probe,
                            threshold=0,
                            runbook_anchor="#cycles-complete-not-ready",
                        )
                    )

        # 6. GC pipeline pass rules (in-process stage snapshot from the GC
        # daemon; see ingestion.monitoring.gc_metrics.snapshot_alert_state).
        # Complements the queue-state rules above: reclamation_failed_shards
        # watches the persisted quarantine backlog, these watch what actually
        # happened during the latest pass.
        if gc_data is not None:
            worker_failed = int(gc_data.get("worker_failed_total", 0))
            if worker_failed > 0:
                alerts.append(
                    Alert(
                        name="gc_worker_failed_shards",
                        severity=AlertSeverity.WARNING,
                        summary=f"Reclamation worker failed {worker_failed} shard targets in latest pass",
                        description=(
                            f"The GC reclamation worker moved {worker_failed} shard target(s) into failed "
                            "quarantine during its latest pass. Inspect reclamation_queue and consider "
                            "`weather-ingest reclamation requeue`."
                        ),
                        scope="reclamation",
                        value=worker_failed,
                        threshold=0,
                        runbook_anchor="#gc-worker-failures",
                    )
                )

            sweeper_failed = int(gc_data.get("sweeper_failed_total", 0))
            if sweeper_failed > 0:
                alerts.append(
                    Alert(
                        name="gc_sweeper_failed_cycles",
                        severity=AlertSeverity.WARNING,
                        summary=f"Metadata sweeper failed {sweeper_failed} cycles in latest pass",
                        description=(
                            f"{sweeper_failed} cycle(s) failed the metadata retention sweeper pass. "
                            "The cycles stay tombstoned with detailed metadata retained; the next pass retries."
                        ),
                        scope="sweeper",
                        value=sweeper_failed,
                        threshold=0,
                        runbook_anchor="#gc-sweeper-failures",
                    )
                )

            orphans_within_frontier = int(gc_data.get("inventory_orphans_within_frontier", 0))
            if orphans_within_frontier > 0:
                alerts.append(
                    Alert(
                        name="gc_orphan_stores_detected",
                        severity=AlertSeverity.WARNING,
                        summary=f"{orphans_within_frontier} orphan cycle store(s) within recoverability frontier",
                        description=(
                            f"Store<->catalog reconciliation found {orphans_within_frontier} physical cycle store(s) "
                            "with no catalog identity whose serving horizon has not expired. Catalog recovery from "
                            "COMPLETE marker evidence is possible but deliberately not automated — run "
                            "`weather-ingest gc --inventory` for the detailed listing."
                        ),
                        scope="inventory",
                        value=orphans_within_frontier,
                        threshold=0,
                        runbook_anchor="#orphan-inventory",
                    )
                )

        return alerts

    def _dispatch_events(self, events: list[AlertEvent]) -> None:
        """Dispatch events to every sink and watch for webhook delivery loss."""
        for event in events:
            for sink in self._sinks:
                try:
                    sink.emit(event)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Alert sink %s failed: %s", sink, exc)
            if (
                self.webhook_sink.webhook_url
                and self.webhook_sink.last_delivery_ok is False
            ):
                self._notify_delivery_failure()

    def _notify_delivery_failure(self) -> None:
        """Raise the ``alert_delivery_failed`` self-alert (alert-on-alert).

        Dispatched through the surviving sinks only (structured log + memory
        ledger) — re-posting to a failing webhook would be pointless
        recursion. Cooldown-tracked so a dead webhook produces one warning
        per cooldown window instead of one per alert event.
        """
        now = time.time()
        with self._delivery_failure_lock:
            if (
                now - self._last_delivery_failure_ts
                < self.deduplicator.default_cooldown_seconds
            ):
                return
            self._last_delivery_failure_ts = now
        alert = Alert(
            name="alert_delivery_failed",
            severity=AlertSeverity.WARNING,
            summary="Alert webhook delivery is failing",
            description=(
                f"Webhook delivery to {self.webhook_sink.webhook_url} failed after "
                f"{self.webhook_sink.retry_attempts} attempt(s): "
                f"{self.webhook_sink.last_delivery_error}. Alerts are still recorded "
                "in structured logs and the diagnostics ledger."
            ),
            scope="alerting",
            value=self.webhook_sink.last_delivery_error,
            threshold=0,
            runbook_anchor="#alert-delivery-failure",
        )
        event = AlertEvent(event_type="triggered", alert=alert)
        for sink in (self.log_sink, self.memory_sink):
            try:
                sink.emit(event)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Alert sink %s failed: %s", sink, exc)

    def dispatch_alerts(self, current_alerts: list[Alert]) -> list[AlertEvent]:
        """Deduplicate and dispatch an externally built alert list.

        Used by callers that derive alerts outside :meth:`evaluate_rules`
        (e.g. the scrape-time exporter's sustained collector-failure monitor).
        Dedup, cooldown, escalation, and recovery semantics are identical to
        the regular evaluation path.

        Returns:
            The alert events dispatched during this call (may be empty when
            everything is suppressed by cooldown).
        """
        events = self.deduplicator.process_alerts(current_alerts)
        self._dispatch_events(events)
        return events

    def evaluate_and_dispatch(
        self,
        resource_data: dict[str, Any] | None = None,
        postgres_data: Any | None = None,
        lifecycle_data: Any | None = None,
        ingestion_data: dict[str, Any] | None = None,
        storage_data: Any | None = None,
        leak_data: Any | None = None,
        gc_data: dict[str, Any] | None = None,
    ) -> list[AlertEvent]:
        """Evaluate rules and dispatch new, escalated, and recovered alert events."""
        current_alerts = self.evaluate_rules(
            resource_data=resource_data,
            postgres_data=postgres_data,
            lifecycle_data=lifecycle_data,
            ingestion_data=ingestion_data,
            storage_data=storage_data,
            leak_data=leak_data,
            gc_data=gc_data,
        )
        events = self.deduplicator.process_alerts(current_alerts)
        self._dispatch_events(events)
        return events


#: Global alert engine singleton
ALERT_ENGINE = AlertEngine()
