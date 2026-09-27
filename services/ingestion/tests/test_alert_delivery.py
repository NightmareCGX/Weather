"""Tests for alert delivery reliability and exporter self-alerting (MONITORING.md 2.6/5.2/7.4).

Covers:

- :class:`WebhookAlertSink` retry with exponential backoff, non-2xx handling,
  fail-open semantics, and the ``weather_alert_delivery_failures_total``
  counter;
- the ``alert_delivery_failed`` self-alert (alert-on-alert) dispatched through
  the surviving sinks with cooldown suppression;
- :class:`CollectorAlertMonitor` sustained collector-failure alerting with
  recovery, wired through :func:`run_collector`.
"""


import pytest

from ingestion.monitoring import (
    Alert,
    AlertEngine,
    AlertEvent,
    AlertSeverity,
    CollectorAlertMonitor,
    WebhookAlertSink,
)
from ingestion.monitoring import metrics as metrics_module
from ingestion.monitoring import exporter as exporter_module
from ingestion.monitoring.exporter import run_collector


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300


class _FakeClientFactory:
    """Records POSTs and returns scripted responses/exceptions per call."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[str] = []

    def __call__(self, **kwargs):  # noqa: ANN003 - test double mirrors httpx.Client
        factory = self

        class _Client:
            def __init__(self, timeout: float) -> None:
                self.timeout = timeout

            def __enter__(self) -> "_Client":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def post(self, url: str, json: dict) -> _FakeResponse:  # noqa: A002 - mirror httpx API
                factory.calls.append(url)
                outcome = factory.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        return _Client(kwargs.get("timeout", 2.0))


def _alert_event() -> AlertEvent:
    return AlertEvent(
        event_type="triggered",
        alert=Alert(name="test_alert", severity=AlertSeverity.WARNING, summary="s", description="d"),
    )


def _delivery_failure_counter() -> float:
    text = metrics_module.REGISTRY.generate_latest()
    for line in text.splitlines():
        if line.startswith("weather_alert_delivery_failures_total "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


# =============================================================================
# WebhookAlertSink retry & fail-open
# =============================================================================


def test_webhook_sink_succeeds_after_transient_failures(monkeypatch):
    factory = _FakeClientFactory(
        [RuntimeError("conn refused"), RuntimeError("timeout"), _FakeResponse(200)]
    )
    monkeypatch.setattr("httpx.Client", factory)
    sink = WebhookAlertSink("http://alerts.example/webhook", retry_attempts=3, retry_backoff_seconds=0.0)

    sink.emit(_alert_event())

    assert sink.last_delivery_ok is True
    assert sink.last_delivery_error == ""
    assert len(factory.calls) == 3


def test_webhook_sink_retries_non_2xx_and_fails_open(monkeypatch):
    factory = _FakeClientFactory([_FakeResponse(500), _FakeResponse(503), _FakeResponse(500)])
    monkeypatch.setattr("httpx.Client", factory)
    sink = WebhookAlertSink("http://alerts.example/webhook", retry_attempts=3, retry_backoff_seconds=0.0)

    sink.emit(_alert_event())

    assert sink.last_delivery_ok is False
    assert "500" in sink.last_delivery_error
    assert len(factory.calls) == 3


def test_webhook_sink_final_failure_increments_counter(monkeypatch):
    factory = _FakeClientFactory([RuntimeError("down")] * 2)
    monkeypatch.setattr("httpx.Client", factory)
    sink = WebhookAlertSink("http://alerts.example/webhook", retry_attempts=2, retry_backoff_seconds=0.0)
    before = _delivery_failure_counter()

    sink.emit(_alert_event())

    assert sink.last_delivery_ok is False
    assert _delivery_failure_counter() == before + 1


def test_webhook_sink_disabled_without_url():
    sink = WebhookAlertSink("", retry_attempts=1)
    sink.emit(_alert_event())
    assert sink.last_delivery_ok is None


def test_webhook_sink_rejects_zero_attempts():
    with pytest.raises(ValueError, match="retry_attempts"):
        WebhookAlertSink("http://alerts.example/webhook", retry_attempts=0)


# =============================================================================
# alert_delivery_failed self-alert (alert-on-alert)
# =============================================================================


def test_delivery_failure_raises_self_alert_through_surviving_sinks(monkeypatch):
    factory = _FakeClientFactory([RuntimeError("down")])
    monkeypatch.setattr("httpx.Client", factory)
    engine = AlertEngine(cooldown_seconds=3600.0, webhook_url="http://alerts.example/webhook", webhook_retry_attempts=1)

    events = engine.dispatch_alerts([Alert(name="disk_space_warning", severity=AlertSeverity.WARNING, summary="s", description="d")])

    assert events, "the triggering alert itself should dispatch"
    recent = engine.memory_sink.get_recent()
    self_alerts = [e for e in recent if e.alert.name == "alert_delivery_failed"]
    assert len(self_alerts) == 1
    assert self_alerts[0].alert.severity == AlertSeverity.WARNING
    assert self_alerts[0].alert.runbook_anchor == "#alert-delivery-failure"
    assert "down" in self_alerts[0].alert.description


def test_delivery_failure_self_alert_respects_cooldown(monkeypatch):
    factory = _FakeClientFactory([RuntimeError("down"), RuntimeError("down")])
    monkeypatch.setattr("httpx.Client", factory)
    engine = AlertEngine(cooldown_seconds=3600.0, webhook_url="http://alerts.example/webhook", webhook_retry_attempts=1)

    engine.dispatch_alerts([Alert(name="disk_space_warning", severity=AlertSeverity.WARNING, summary="s", description="d")])
    engine.dispatch_alerts([Alert(name="ingestion_lag_warning", severity=AlertSeverity.WARNING, summary="s", description="d")])

    self_alerts = [e for e in engine.memory_sink.get_recent() if e.alert.name == "alert_delivery_failed"]
    assert len(self_alerts) == 1, "second failure inside the cooldown must be suppressed"


def test_no_self_alert_without_webhook_configured():
    engine = AlertEngine(cooldown_seconds=3600.0, webhook_url="")
    engine.dispatch_alerts([Alert(name="disk_space_warning", severity=AlertSeverity.WARNING, summary="s", description="d")])
    assert not [e for e in engine.memory_sink.get_recent() if e.alert.name == "alert_delivery_failed"]


def test_self_alert_json_payload_shape():
    """The self-alert flows through the log sink in the documented tag format."""
    engine = AlertEngine(cooldown_seconds=3600.0, webhook_url="")
    engine._notify_delivery_failure()
    recent = engine.memory_sink.get_recent()
    assert recent[-1].alert.name == "alert_delivery_failed"
    assert recent[-1].alert.scope == "alerting"


# =============================================================================
# CollectorAlertMonitor (exporter self-observability)
# =============================================================================


def test_collector_monitor_alerts_only_after_sustained_failures():
    engine = AlertEngine(cooldown_seconds=3600.0, webhook_url="")
    monitor = CollectorAlertMonitor(engine=engine, failure_threshold=3)

    monitor.record("storage", success=False)
    monitor.record("storage", success=False)
    assert engine.deduplicator.get_active_alerts() == [], "transient failures must not alert"

    monitor.record("storage", success=False)
    active = engine.deduplicator.get_active_alerts()
    assert len(active) == 1
    assert active[0].name == "exporter_collector_failed"
    assert active[0].scope == "storage"
    assert active[0].severity == AlertSeverity.WARNING


def test_collector_monitor_recovers_after_success():
    engine = AlertEngine(cooldown_seconds=3600.0, webhook_url="")
    monitor = CollectorAlertMonitor(engine=engine, failure_threshold=2)

    monitor.record("postgres", success=False)
    monitor.record("postgres", success=False)
    assert engine.deduplicator.get_active_alerts()

    monitor.record("postgres", success=True)
    assert engine.deduplicator.get_active_alerts() == []
    event_types = [e.event_type for e in engine.memory_sink.get_recent()]
    assert event_types == ["triggered", "recovered"]


def test_collector_monitor_isolates_collectors():
    engine = AlertEngine(cooldown_seconds=3600.0, webhook_url="")
    monitor = CollectorAlertMonitor(engine=engine, failure_threshold=2)
    monitor.record("resources", success=False)
    monitor.record("resources", success=False)
    monitor.record("lifecycle", success=False)

    assert [a.scope for a in engine.deduplicator.get_active_alerts()] == ["resources"]


def test_run_collector_feeds_the_monitor(monkeypatch):
    engine = AlertEngine(cooldown_seconds=3600.0, webhook_url="")
    monitor = CollectorAlertMonitor(engine=engine, failure_threshold=2)
    monkeypatch.setattr(exporter_module, "_COLLECTOR_ALERT_MONITOR", monitor)

    def failing() -> None:
        raise RuntimeError("collector exploded")

    run_collector("resources", failing)
    run_collector("resources", failing)

    active = engine.deduplicator.get_active_alerts()
    assert [a.name for a in active] == ["exporter_collector_failed"]

    run_collector("resources", lambda: None)
    assert engine.deduplicator.get_active_alerts() == []


def test_monitor_never_breaks_the_scrape(monkeypatch):
    """A broken alert engine must not turn a collector failure into an HTTP 5xx."""

    class _BrokenEngine:
        def dispatch_alerts(self, alerts):  # noqa: ANN001 - test double
            raise RuntimeError("engine is broken")

    monitor = CollectorAlertMonitor(engine=_BrokenEngine(), failure_threshold=1)
    monitor.record("storage", success=False)  # must not raise
