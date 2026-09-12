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

logger = logging.getLogger(__name__)


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
    """Optional HTTP webhook alert delivery (e.g. Slack, Discord, Alertmanager). Fail-open."""

    def __init__(self, webhook_url: str, timeout_seconds: float = 2.0) -> None:
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds

    def emit(self, event: AlertEvent) -> None:
        if not self.webhook_url:
            return
        try:
            import httpx

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
            # Asynchronous or short-timeout post
            with httpx.Client(timeout=self.timeout_seconds) as client:
                client.post(self.webhook_url, json=payload)
        except Exception as exc:  # noqa: BLE001
            # Never raise or break production execution
            logger.debug("Failed to dispatch alert webhook to %s: %s", self.webhook_url, exc)


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


class AlertEngine:
    """Coordinates rules evaluation, deduplication, and multi-sink dispatching."""

    def __init__(
        self,
        cooldown_seconds: float = 3600.0,
        webhook_url: str = "",
    ) -> None:
        self.deduplicator = AlertDeduplicator(default_cooldown_seconds=cooldown_seconds)
        self.log_sink = LogAlertSink()
        self.memory_sink = MemoryAlertSink()
        self.webhook_sink = WebhookAlertSink(webhook_url=webhook_url)
        self._sinks: list[AlertSink] = [self.log_sink, self.memory_sink, self.webhook_sink]

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
    ) -> list[Alert]:
        """Evaluate all system, resource, database, and lifecycle rules against snapshots."""
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
                        summary=f"14-day metadata sweeper backlog overdue ({lifecycle_data.sweeper_unpurged_count} cycles)",
                        description=(
                            f"{lifecycle_data.sweeper_unpurged_count} tombstones retain detailed metadata past 14 days "
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

        return alerts

    def evaluate_and_dispatch(
        self,
        resource_data: dict[str, Any] | None = None,
        postgres_data: Any | None = None,
        lifecycle_data: Any | None = None,
        ingestion_data: dict[str, Any] | None = None,
        storage_data: Any | None = None,
        leak_data: Any | None = None,
    ) -> list[AlertEvent]:
        """Evaluate rules and dispatch new, escalated, and recovered alert events."""
        current_alerts = self.evaluate_rules(
            resource_data=resource_data,
            postgres_data=postgres_data,
            lifecycle_data=lifecycle_data,
            ingestion_data=ingestion_data,
            storage_data=storage_data,
            leak_data=leak_data,
        )
        events = self.deduplicator.process_alerts(current_alerts)
        for event in events:
            for sink in self._sinks:
                try:
                    sink.emit(event)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Alert sink %s failed: %s", sink, exc)
        return events


#: Global alert engine singleton
ALERT_ENGINE = AlertEngine()
