"""Object storage (MinIO / S3) health, latency, and operation instrumentation.

Provides low-overhead connectivity probes, latency measurement, operation counters
(GET, PUT, HEAD, LIST, DELETE), failure tracking, and phase-specific statistics
for sensitive operations (prepare_run_store, marker PUT, manifest write, prefix deletion).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ingestion.monitoring.metrics import REGISTRY

logger = logging.getLogger(__name__)

# Register Prometheus metrics
STORAGE_CONNECTED = REGISTRY.gauge(
    "weather_storage_connected",
    "Object storage connectivity status (1 if connected, 0 otherwise)",
)
STORAGE_PROBE_LATENCY_MS = REGISTRY.gauge(
    "weather_storage_probe_latency_milliseconds",
    "Object storage health probe latency in milliseconds",
)

STORAGE_OPERATIONS_TOTAL = REGISTRY.counter(
    "weather_storage_operations_total",
    "Total object storage operations by operation class",
    labelnames=("operation",),  # "get", "put", "head", "list", "delete"
)
STORAGE_OPERATION_FAILURES_TOTAL = REGISTRY.counter(
    "weather_storage_operation_failures_total",
    "Total failed object storage operations by operation class",
    labelnames=("operation",),
)
STORAGE_OPERATION_RETRIES_TOTAL = REGISTRY.counter(
    "weather_storage_operation_retries_total",
    "Total retried object storage operations by operation class",
    labelnames=("operation",),
)

STORAGE_OPERATION_DURATION_SECONDS = REGISTRY.histogram(
    "weather_storage_operation_duration_seconds",
    "Object storage operation latencies in seconds",
    labelnames=("operation",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

STORAGE_PHASE_DURATION_SECONDS = REGISTRY.gauge(
    "weather_storage_phase_duration_seconds",
    "Duration in seconds of the most recent sensitive storage phase",
    labelnames=("phase",),
    # phases: "prepare_run_store", "marker_put", "marker_validation", "manifest_write", "prefix_deletion"
)


@dataclass(frozen=True)
class StorageHealthReport:
    """Object storage health and latency report."""

    connected: bool
    latency_ms: float
    endpoint: str
    bucket: str
    error: str | None = None


class StorageHealthCollector:
    """Collects reachability, latency, and operational telemetry for object storage."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:9000",
        bucket: str = "weather-data",
        access_key: str = "minio_admin",
        secret_key: str = "minio_password",
    ) -> None:
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key

    def probe(self, timeout_seconds: float = 3.0) -> StorageHealthReport:
        """Perform a live, non-blocking connectivity and latency probe against MinIO/S3."""
        t0 = time.monotonic()
        try:
            import boto3  # type: ignore[import-untyped]
            from botocore.config import Config  # type: ignore[import-untyped]

            cfg = Config(connect_timeout=timeout_seconds, read_timeout=timeout_seconds, retries={"max_attempts": 1})
            s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url if "://" in self.endpoint_url else f"http://{self.endpoint_url}",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=cfg,
            )
            s3.head_bucket(Bucket=self.bucket)
            latency_ms = max(0.1, (time.monotonic() - t0) * 1000.0)
            STORAGE_CONNECTED.set(1.0)
            STORAGE_PROBE_LATENCY_MS.set(round(latency_ms, 2))
            return StorageHealthReport(
                connected=True,
                latency_ms=round(latency_ms, 2),
                endpoint=self.endpoint_url,
                bucket=self.bucket,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            logger.debug("Storage health probe failed: %s", exc)
            STORAGE_CONNECTED.set(0.0)
            STORAGE_PROBE_LATENCY_MS.set(round(latency_ms, 2))
            return StorageHealthReport(
                connected=False,
                latency_ms=round(latency_ms, 2),
                endpoint=self.endpoint_url,
                bucket=self.bucket,
                error=str(exc),
            )

    def record_operation(
        self,
        operation: str,
        duration_s: float,
        success: bool = True,
        retried: bool = False,
    ) -> None:
        """Record an S3 operation metric."""
        op = operation.lower()
        STORAGE_OPERATIONS_TOTAL.labels(operation=op).inc(1.0)
        STORAGE_OPERATION_DURATION_SECONDS.labels(operation=op).observe(duration_s)
        if not success:
            STORAGE_OPERATION_FAILURES_TOTAL.labels(operation=op).inc(1.0)
        if retried:
            STORAGE_OPERATION_RETRIES_TOTAL.labels(operation=op).inc(1.0)

    def record_phase_duration(self, phase: str, duration_s: float) -> None:
        """Record the latency of a known sensitive storage phase."""
        STORAGE_PHASE_DURATION_SECONDS.labels(phase=phase).set(round(duration_s, 4))


#: Global storage collector singleton
STORAGE_COLLECTOR = StorageHealthCollector()
