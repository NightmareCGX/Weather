"""Scrape-time Prometheus exporter for the ingestion metrics surface.

The exporter is a probe-style daemon: on every Prometheus scrape it re-runs
each platform health collector and renders a fresh exposition. It holds no
durable business state and is safe to stop/start at any time.

Per-collector execution is fail-open: a failing collector logs a warning,
publishes ``weather_exporter_collector_success{collector=...} = 0``, and never
aborts the scrape, its sibling collectors, or the exporter process. Collector
duration is recorded for both successful and failed runs so degradation (e.g.
a storage probe degrading from 20 ms to timeout) remains visible.

Security: the HTTP server binds ``127.0.0.1`` by default. Metrics endpoints
must not be exposed to public networks; pass ``--host 0.0.0.0`` explicitly
only when a local Docker-based Prometheus must reach the host exporter.
"""

from __future__ import annotations

import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from ingestion.monitoring.metrics import REGISTRY, Gauge

logger = logging.getLogger(__name__)

#: 1 = the named collector completed successfully during the latest scrape,
#: 0 = it raised. Distinguishes "the system really is at 0" from "the exporter
#: never managed to observe the system".
COLLECTOR_SUCCESS: Gauge = REGISTRY.gauge(
    "weather_exporter_collector_success",
    "1 if the named exporter collector completed successfully during the "
    "latest scrape, 0 if it failed (fail-open)",
    labelnames=("collector",),
)

#: Wall-clock duration of the latest collector execution, recorded for both
#: successful and failed runs.
COLLECTOR_DURATION: Gauge = REGISTRY.gauge(
    "weather_exporter_collector_duration_seconds",
    "Duration of the latest execution of the named exporter collector "
    "(recorded for both success and failure)",
    labelnames=("collector",),
)


def run_collector(name: str, collect: Callable[[], object]) -> bool:
    """Execute one collector with per-collector fail-open self-observability.

    Args:
        name: Bounded label value identifying the collector in the
            ``weather_exporter_collector_*`` metrics.
        collect: Zero-argument callable performing the probe/collection.

    Returns:
        True if the collector completed successfully, False otherwise.
        Exceptions are logged and swallowed (fail-open).
    """
    started = time.perf_counter()
    try:
        collect()
    except Exception:  # noqa: BLE001 - a failing collector must never abort the scrape
        logger.warning("exporter collector %r failed", name, exc_info=True)
        COLLECTOR_SUCCESS.labels(collector=name).set(0.0)
        COLLECTOR_DURATION.labels(collector=name).set(time.perf_counter() - started)
        return False
    COLLECTOR_SUCCESS.labels(collector=name).set(1.0)
    COLLECTOR_DURATION.labels(collector=name).set(time.perf_counter() - started)
    return True


def collect_platform_metrics() -> None:
    """Run every ingestion-surface collector exactly once (one scrape worth).

    Each collector is independent: a failure in one never prevents the others
    from being observed. Heavy imports are deferred to call time so that CLI
    subcommands which never touch metrics do not pay for them.
    """
    from ingestion.core.config import settings
    from ingestion.core.db import engine
    from ingestion.monitoring import (
        INGESTION_COLLECTOR,
        LifecycleHealthCollector,
        PostgresHealthCollector,
        RESOURCE_COLLECTOR,
        StorageHealthCollector,
    )

    # The INGESTION_COLLECTOR module singleton is constructed without an
    # engine; without wiring one here, evaluate_lag/evaluate_*_completeness
    # silently skip the PostgreSQL catalog and publish degraded, clock-derived
    # values (and no weather_gefs_*/weather_model_* gauges at all).
    INGESTION_COLLECTOR.engine = engine

    run_collector("resources", RESOURCE_COLLECTOR.collect_and_export)
    run_collector("postgres", PostgresHealthCollector(engine).collect)
    run_collector(
        "storage",
        StorageHealthCollector(
            endpoint_url=getattr(settings, "MINIO_ENDPOINT", "http://localhost:9000"),
            bucket=getattr(settings, "MINIO_BUCKET_NAME", "weather-data"),
            access_key=getattr(settings, "MINIO_ACCESS_KEY", "minio_admin"),
            secret_key=getattr(settings, "MINIO_SECRET_KEY", "minio_password"),
        ).probe,
    )
    run_collector("lifecycle", LifecycleHealthCollector(engine).collect)

    def _evaluate_ingestion_state() -> None:
        INGESTION_COLLECTOR.evaluate_lag("gfs")
        INGESTION_COLLECTOR.evaluate_lag("gefs")
        # Feed the member-completeness/servability gauges
        # (weather_gefs_* and weather_model_* families) that the dashboard's
        # GEFS Member Completeness panel renders; evaluate_lag alone does not
        # populate them.
        INGESTION_COLLECTOR.evaluate_gefs_completeness()
        INGESTION_COLLECTOR.evaluate_gfs_completeness()

    run_collector("ingestion", _evaluate_ingestion_state)


def _collect_and_render() -> bytes:
    """Collect one fresh scrape worth of metrics and render the exposition."""
    collect_platform_metrics()
    return REGISTRY.generate_latest().encode("utf-8")


def _make_live_renderer(component: str = "realtime") -> Callable[[], bytes]:
    """Build a renderer that updates component process resource metrics on scrape."""

    def _render() -> bytes:
        try:
            from ingestion.monitoring.resources import RESOURCE_COLLECTOR

            RESOURCE_COLLECTOR.sample_component_resources(component)
        except Exception:  # noqa: BLE001 - fail open
            pass
        return REGISTRY.generate_latest().encode("utf-8")

    return _render


def _render_registry_only() -> bytes:
    """Render the current in-process registry without running any probes.

    Used by long-running ingestion processes (e.g. the realtime daemon) whose
    registries already hold live, continuously-updated pipeline counters.
    """
    return _make_live_renderer("realtime")()


def _make_metrics_handler(render: Callable[[], bytes]) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP handler serving the given exposition renderer per GET."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/metrics", "/"):
                content = render()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return None

    return Handler


#: Scrape-time exporter handler: re-collects on every GET.
MetricsHandler = _make_metrics_handler(_collect_and_render)


def _serve_http(host: str, port: int, render: Callable[[], bytes]) -> int:
    server = ThreadingHTTPServer((host, port), _make_metrics_handler(render))
    print(f"Prometheus exporter running on http://{host}:{port}/metrics (Press Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
    return 0


def serve_metrics(host: str = "127.0.0.1", port: int = 9112) -> int:
    """Run the scrape-time Prometheus exporter HTTP daemon until interrupted.

    Args:
        host: Bind address. Defaults to loopback (production-safe); use an
            explicit non-loopback bind only when a local Docker-based
            Prometheus must scrape this exporter from inside a container.
        port: TCP port to listen on.

    Returns:
        Process exit code (0 after a clean Ctrl+C shutdown).
    """
    return _serve_http(host, port, _collect_and_render)


def serve_live_registry(
    host: str = "127.0.0.1", port: int = 9113, component: str = "realtime"
) -> int:
    """Serve the current process's live registry until interrupted.

    For embedding in long-running ingestion processes (realtime daemon, gc):
    the registry already contains live process-local pipeline counters, so
    serving it needs no probing and adds only one background HTTP thread.
    On each scrape, samples the component process's CPU and RSS.

    Args:
        host: Bind address (loopback default, production-safe).
        port: TCP port to listen on.
        component: Component name label ("realtime", "gc", etc.).

    Returns:
        Process exit code (0 after a clean Ctrl+C shutdown).
    """
    return _serve_http(host, port, _make_live_renderer(component))
