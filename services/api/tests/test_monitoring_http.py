"""Tests for the API serving-layer observability features (MONITORING.md 2.7/2.8/8).

Covers:

- HTTP request metrics (RED): bounded route/method/status_class labels,
  latency histogram rendering, in-flight gauge, and the ``/v1/metrics``
  self-scrape exclusion;
- the client telemetry endpoint (``POST /v1/telemetry/client``) validation
  limits, acceptance receipt, and counter cardinality;
- the structured JSON logging formatter and request-ID correlation.
"""

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import app
from api.monitoring.client_telemetry_metrics import CLIENT_TELEMETRY_EVENTS_TOTAL
from api.monitoring.http_metrics import (
    DURATION_BUCKETS,
    HTTP_METRICS,
    UNMATCHED_ROUTE_LABEL,
    HttpMetrics,
    HTTPMetricsMiddleware,
)
from api.monitoring.logging import (
    JsonLogFormatter,
    configure_logging,
    get_request_id,
    set_request_id,
)
from api.routers import admin as admin_router


def _parse_exposition(text: str) -> dict[str, float]:
    """Parse a Prometheus text exposition into a sample-name -> value map."""
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        samples[name] = float(value)
    return samples


# =============================================================================
# HttpMetrics unit tests (isolated registry instances)
# =============================================================================


class TestHttpMetrics:
    def test_record_and_render_round_trip(self):
        metrics = HttpMetrics()
        metrics.record("/v1/points", "GET", "2xx", 0.25)
        metrics.record("/v1/points", "GET", "2xx", 0.03)
        metrics.record("/v1/points", "GET", "5xx", 1.0)

        samples = _parse_exposition("\n".join(metrics.render_lines()))
        counter = 'weather_api_http_requests_total{method="GET",route="/v1/points",status_class="2xx"}'
        assert samples[counter] == 2
        error_counter = 'weather_api_http_requests_total{method="GET",route="/v1/points",status_class="5xx"}'
        assert samples[error_counter] == 1
        count = 'weather_api_http_request_duration_seconds_count{method="GET",route="/v1/points"}'
        assert samples[count] == 3
        cumulative = 'weather_api_http_request_duration_seconds_bucket{le="0.5",method="GET",route="/v1/points"}'
        # 0.25 and 0.03 fall into <=0.5; 1.0 does not.
        assert samples[cumulative] == 2

    def test_render_is_deterministic_and_sorted(self):
        metrics = HttpMetrics()
        metrics.record("/v1/b", "POST", "2xx", 0.1)
        metrics.record("/v1/a", "GET", "4xx", 0.1)
        lines = metrics.render_lines()
        again = HttpMetrics()
        again.record("/v1/a", "GET", "4xx", 0.1)
        again.record("/v1/b", "POST", "2xx", 0.1)
        assert lines == again.render_lines()

    def test_in_flight_gauge_tracks_enter_exit(self):
        metrics = HttpMetrics()
        metrics.enter()
        metrics.enter()
        assert metrics.in_flight() == 2
        metrics.exit()
        assert metrics.in_flight() == 1
        metrics.exit()
        assert metrics.in_flight() == 0

    def test_buckets_are_sorted_and_terminate_above_maximum(self):
        assert DURATION_BUCKETS == tuple(sorted(DURATION_BUCKETS))
        metrics = HttpMetrics()
        metrics.record("/v1/maps", "GET", "2xx", 999.0)
        samples = _parse_exposition("\n".join(metrics.render_lines()))
        inf_bucket = 'weather_api_http_request_duration_seconds_bucket{le="+Inf",method="GET",route="/v1/maps"}'
        assert samples[inf_bucket] == 1


# =============================================================================
# Middleware integration through the real application
# =============================================================================


@pytest.fixture(name="metrics_client")
def metrics_client_fixture():
    return TestClient(app)


class TestHTTPMetricsMiddleware:
    def test_known_route_uses_template_and_status_class(self, metrics_client, monkeypatch):
        # Pin the health probes so the endpoint returns 200 regardless of the
        # environment's object-storage availability (the CI api-pipeline job
        # provisions PostgreSQL + Redis but no MinIO, which would otherwise
        # degrade the response to a 503 / 5xx status class).
        monkeypatch.setattr(admin_router, "_database_connected", lambda: True)
        monkeypatch.setattr(admin_router, "_redis_connected", lambda: True)
        monkeypatch.setattr(admin_router, "_object_storage_connected", lambda: True)

        before = _parse_exposition(metrics_client.get("/v1/metrics").text)
        key = 'weather_api_http_requests_total{method="GET",route="/v1/health",status_class="2xx"}'
        before_count = before.get(key, 0.0)

        metrics_client.get("/v1/health")

        after = _parse_exposition(metrics_client.get("/v1/metrics").text)
        assert after[key] == before_count + 1

    def test_unmatched_route_collapses_to_bounded_label(self, metrics_client):
        metrics_client.get("/v1/definitely-not-a-route-xyz")

        after = _parse_exposition(metrics_client.get("/v1/metrics").text)
        key = f'weather_api_http_requests_total{{method="GET",route="{UNMATCHED_ROUTE_LABEL}",status_class="4xx"}}'
        assert after[key] >= 1

    def test_metrics_scrapes_are_excluded_from_request_metrics(self, metrics_client):
        text = metrics_client.get("/v1/metrics").text
        assert 'weather_api_http_requests_total{method="GET",route="/v1/metrics"' not in text
        # The scrape itself must not leave the in-flight gauge elevated.
        assert _parse_exposition(text)["weather_api_http_requests_in_flight"] == 0

    def test_hostile_method_collapses_to_other_label(self, metrics_client):
        resp = metrics_client.request("BREW", "/v1/health")
        assert resp.status_code == 405

        after = _parse_exposition(metrics_client.get("/v1/metrics").text)
        key = 'weather_api_http_requests_total{method="OTHER",route="/v1/health",status_class="4xx"}'
        assert after.get(key, 0.0) >= 1

    def test_unhandled_exception_records_5xx(self):
        # A minimal app without catch-all handlers so the exception escapes
        # to ServerErrorMiddleware through our middleware's failure path.
        probe_app = FastAPI()
        probe_app.add_middleware(HTTPMetricsMiddleware)

        @probe_app.get("/boom")
        def boom() -> dict:
            raise RuntimeError("synthetic failure")

        client = TestClient(probe_app, raise_server_exceptions=False)
        client.get("/boom")

        text = "\n".join(HTTP_METRICS.render_lines())
        samples = _parse_exposition(text)
        key = 'weather_api_http_requests_total{method="GET",route="/boom",status_class="5xx"}'
        assert samples[key] == 1
        assert samples["weather_api_http_requests_in_flight"] == 0

    def test_histogram_series_rendered_for_new_route(self, metrics_client):
        metrics_client.get("/v1/health")
        text = metrics_client.get("/v1/metrics").text
        assert 'weather_api_http_request_duration_seconds_count{method="GET",route="/v1/health"}' in text


# =============================================================================
# Client telemetry endpoint
# =============================================================================


def _telemetry_counter_value(event_type: str) -> float:
    for line in CLIENT_TELEMETRY_EVENTS_TOTAL.render_lines():
        if line.startswith(f'weather_client_telemetry_events_total{{event_type="{event_type}"}}'):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


class TestClientTelemetryEndpoint:
    def test_valid_batch_accepted_with_receipt(self, metrics_client):
        payload = {
            "events": [
                {
                    "type": "error",
                    "name": "TypeError",
                    "timestamp": 1730000000.0,
                    "message": "x is undefined",
                    "stack": "at f()",
                    "page_url": "https://example.test/",
                    "session_id": "sess_abc",
                },
                {"type": "web_vital", "name": "LCP", "timestamp": 1730000001.0, "value": 900.5, "rating": "good"},
            ]
        }
        before_error = _telemetry_counter_value("error")
        before_vital = _telemetry_counter_value("web_vital")

        resp = metrics_client.post("/v1/telemetry/client", json=payload)

        assert resp.status_code == 202
        body = resp.json()
        assert body == {
            "object": "client_telemetry_receipt",
            "data": {"accepted": 2},
            "has_more": False,
            "next_cursor": None,
        }
        assert resp.headers["Cache-Control"] == "no-store"
        assert _telemetry_counter_value("error") == before_error + 1
        assert _telemetry_counter_value("web_vital") == before_vital + 1

    def test_unknown_event_type_rejected(self, metrics_client):
        resp = metrics_client.post(
            "/v1/telemetry/client",
            json={"events": [{"type": "bogus", "name": "x", "timestamp": 1.0}]},
        )
        assert resp.status_code == 422

    def test_oversized_field_rejected(self, metrics_client):
        resp = metrics_client.post(
            "/v1/telemetry/client",
            json={
                "events": [
                    {
                        "type": "error",
                        "name": "E",
                        "timestamp": 1.0,
                        "message": "m" * 600,
                    }
                ]
            },
        )
        assert resp.status_code == 422

    def test_batch_over_twenty_events_rejected(self, metrics_client):
        events = [{"type": "error", "name": "E", "timestamp": 1.0} for _ in range(21)]
        resp = metrics_client.post("/v1/telemetry/client", json={"events": events})
        assert resp.status_code == 422

    def test_empty_batch_rejected(self, metrics_client):
        resp = metrics_client.post("/v1/telemetry/client", json={"events": []})
        assert resp.status_code == 422


# =============================================================================
# Structured JSON logging
# =============================================================================


class TestJsonLogFormatter:
    def _record(self, **extra) -> logging.LogRecord:
        record = logging.LogRecord(
            name="api.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        # Mirror Logger.makeRecord: `extra` keys land in the record __dict__.
        record.__dict__.update(extra)
        return record

    def test_basic_fields_and_message_interpolation(self):
        formatted = json.loads(JsonLogFormatter().format(self._record()))
        assert formatted["level"] == "INFO"
        assert formatted["logger"] == "api.test"
        assert formatted["message"] == "hello world"
        assert "timestamp" in formatted

    def test_request_id_from_context(self):
        token = set_request_id("req_abc123")
        try:
            formatted = json.loads(JsonLogFormatter().format(self._record()))
            assert formatted["request_id"] == "req_abc123"
        finally:
            # Context restoration keeps unrelated log lines clean.
            from api.monitoring.logging import _request_id_contextvar

            _request_id_contextvar.reset(token)
        assert get_request_id() == ""

    def test_telemetry_extra_fields_forwarded(self):
        record = self._record(telemetry_type="error", telemetry_name="TypeError", unrelated="dropped")
        formatted = json.loads(JsonLogFormatter().format(record))
        assert formatted["telemetry_type"] == "error"
        assert formatted["telemetry_name"] == "TypeError"
        assert "unrelated" not in formatted

    def test_exception_info_rendered_as_string(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys

            record = logging.LogRecord(
                name="api.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=None,
                exc_info=sys.exc_info(),
            )
        formatted = json.loads(JsonLogFormatter().format(record))
        assert "RuntimeError: boom" in formatted["exc_info"]


class TestConfigureLogging:
    def test_json_mode_installs_handlers(self):
        configure_logging("json")
        root = logging.getLogger()
        assert any(isinstance(handler.formatter, JsonLogFormatter) for handler in root.handlers)

    def test_text_mode_installs_plain_formatter(self):
        configure_logging("text")
        root = logging.getLogger()
        assert not any(isinstance(handler.formatter, JsonLogFormatter) for handler in root.handlers)
        # Restore JSON for the remainder of the suite (create_app default).
        configure_logging("json")

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="Unsupported API_LOG_FORMAT"):
            configure_logging("xml")
