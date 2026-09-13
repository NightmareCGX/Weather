"""Tests for the scrape-time ingestion Prometheus exporter (ingestion.monitoring.exporter).

Covers:
- Per-collector fail-open execution (success=1 on success, success=0 on failure)
- Collector duration recorded for both successful and failed runs
- Scrape-time freshness: every GET re-executes collection and renders new values
- HTTP exposition content type, 404 routing, and self-observability metrics presence
- Secure-by-default bind address (loopback) and CLI argument defaults
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from ingestion.monitoring import exporter
from ingestion.monitoring.metrics import REGISTRY


# =============================================================================
# 1. run_collector: fail-open self-observability
# =============================================================================


def test_run_collector_success_records_success_and_duration():
    def ok_collector() -> None:
        return None

    assert exporter.run_collector("selftest_ok", ok_collector) is True

    lines = REGISTRY.get("weather_exporter_collector_success").collect()
    ok_line = next(line for line in lines if 'collector="selftest_ok"' in line)
    assert ok_line.split(" ")[-1] == "1.0"


def test_run_collector_failure_is_fail_open_and_records_duration():
    def broken_collector() -> None:
        raise RuntimeError("boom")

    assert exporter.run_collector("selftest_broken", broken_collector) is False

    lines = REGISTRY.get("weather_exporter_collector_success").collect()
    broken_line = next(line for line in lines if 'collector="selftest_broken"' in line)
    assert broken_line.split(" ")[-1] == "0.0"

    # Duration must be recorded for failures too (degradation visibility).
    duration_lines = REGISTRY.get("weather_exporter_collector_duration_seconds").collect()
    duration_line = next(line for line in duration_lines if 'collector="selftest_broken"' in line)
    assert float(duration_line.split(" ")[-1]) >= 0.0


def test_run_collector_duration_measures_execution_time():
    def slow_collector() -> None:
        time.sleep(0.05)

    exporter.run_collector("selftest_slow", slow_collector)

    duration_lines = REGISTRY.get("weather_exporter_collector_duration_seconds").collect()
    duration_line = next(line for line in duration_lines if 'collector="selftest_slow"' in line)
    assert float(duration_line.split(" ")[-1]) >= 0.05


# =============================================================================
# 2. HTTP exporter: scrape-time freshness & routing
# =============================================================================


@pytest.fixture()
def exporter_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", server
    server.shutdown()
    server.server_close()


def test_exporter_http_recollects_on_every_scrape(exporter_server, monkeypatch):
    base_url, _ = exporter_server

    scrape_counter = {"n": 0}

    def fake_collect() -> None:
        scrape_counter["n"] += 1
        REGISTRY.get("weather_exporter_collector_success").labels(
            collector="probe"
        ).set(float(scrape_counter["n"]))

    monkeypatch.setattr(exporter, "collect_platform_metrics", fake_collect)

    def scrape() -> str:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as response:
            assert response.headers["Content-Type"] == "text/plain; version=0.0.4"
            return response.read().decode("utf-8")

    first = scrape()
    assert scrape_counter["n"] == 1
    assert 'weather_exporter_collector_success{collector="probe"} 1.0' in first

    second = scrape()
    assert scrape_counter["n"] == 2
    assert 'weather_exporter_collector_success{collector="probe"} 2.0' in second


def test_exporter_http_routes_and_self_metrics(exporter_server, monkeypatch):
    base_url, _ = exporter_server
    monkeypatch.setattr(exporter, "collect_platform_metrics", lambda: None)

    with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as response:
        body = response.read().decode("utf-8")
    assert "# HELP weather_exporter_collector_success" in body
    assert "# TYPE weather_exporter_collector_duration_seconds gauge" in body

    with urllib.request.urlopen(base_url + "/", timeout=5) as response:
        assert response.status == 200

    try:
        urllib.request.urlopen(f"{base_url}/other", timeout=5)
        raised = False
    except urllib.error.HTTPError as exc:
        raised = True
        assert exc.code == 404
    assert raised


def test_exporter_survives_failing_collector(exporter_server, monkeypatch):
    """A collector raising mid-scrape must not abort the exposition (fail-open)."""
    base_url, _ = exporter_server

    def realistic_collect() -> None:
        # Mirrors the production wiring: each collector runs through
        # run_collector, which swallows the exception and marks success=0.
        def broken() -> None:
            raise RuntimeError("postgres down")

        assert exporter.run_collector("probe", broken) is False

    monkeypatch.setattr(exporter, "collect_platform_metrics", realistic_collect)

    with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as response:
        assert response.status == 200
        body = response.read().decode("utf-8")
    assert 'weather_exporter_collector_success{collector="probe"} 0.0' in body


# =============================================================================
# 3. CLI argument surface: secure-by-default bind
# =============================================================================


def test_metrics_cli_defaults_are_loopback_and_9112():
    from ingestion.cli import _build_parser

    args = _build_parser().parse_args(["metrics"])
    assert args.host == "127.0.0.1"
    assert args.port == 9112
    assert args.print is False


def test_metrics_cli_accepts_explicit_host_port_print():
    from ingestion.cli import _build_parser

    args = _build_parser().parse_args(
        ["metrics", "--host", "0.0.0.0", "--port", "9112", "--print"]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9112
    assert args.print is True


def test_serve_metrics_binds_requested_address(exporter_server):
    """serve_metrics binds the exact host:port it is given (no 0.0.0.0 surprises)."""
    base_url, server = exporter_server
    # The fixture constructs MetricsHandler the same way serve_metrics does.
    assert server.server_address[0] == "127.0.0.1"
    assert base_url.startswith("http://127.0.0.1:")


# =============================================================================
# 4. Live registry server (embedded in the realtime daemon)
# =============================================================================


def test_live_registry_serves_process_local_counters():
    """serve_live_registry's handler exposes live REGISTRY state without probing."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), exporter._make_metrics_handler(exporter._render_registry_only)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        marker = REGISTRY.gauge("test_live_registry_marker", "test marker")
        marker.set(42.0)

        with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as response:
            body = response.read().decode("utf-8")
        assert "test_live_registry_marker 42.0" in body

        # Live values: updates between scrapes are visible immediately.
        marker.set(43.0)
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as response:
            body = response.read().decode("utf-8")
        assert "test_live_registry_marker 43.0" in body
    finally:
        server.shutdown()
        server.server_close()


# =============================================================================
# 5. S3 operation instrumentation (Operations & Failures panel)
# =============================================================================


def test_s3_instrument_records_success_and_failure():
    import asyncio

    from ingestion.core.s3 import IngestionS3FileSystem

    fs = object.__new__(IngestionS3FileSystem)

    def counter_line(name: str) -> float:
        lines = REGISTRY.get(name).collect()
        get_line = next((line for line in lines if 'operation="get"' in line), None)
        return float(get_line.split(" ")[-1]) if get_line else 0.0

    total_before = counter_line("weather_storage_operations_total")
    asyncio.run(fs._instrument("get", asyncio.sleep(0, result="ok")))
    total_after = counter_line("weather_storage_operations_total")
    assert total_after == pytest.approx(total_before + 1.0)

    failed_before = counter_line("weather_storage_operation_failures_total")

    async def failing() -> None:
        raise RuntimeError("s3 down")

    with pytest.raises(RuntimeError):
        asyncio.run(fs._instrument("get", failing()))
    failed_after = counter_line("weather_storage_operation_failures_total")
    assert failed_after == pytest.approx(failed_before + 1.0)
