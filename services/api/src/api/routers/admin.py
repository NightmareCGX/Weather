"""Administration endpoint: system health (API.md section 8.1).

Returns the connectivity status of the API's backend dependencies (PostgreSQL,
Redis, and object storage). The router is thin (ENGINEERING_CONTRACT section
2): it runs lightweight connectivity probes and serializes the documented
``health_check`` envelope. When every dependency is connected the response is
``200`` with ``status: healthy``; when any dependency is unavailable it is
``503`` with ``status: degraded`` and the failing dependency reported as
``disconnected``.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Any

import redis as redis_lib
import s3fs  # type: ignore[import-untyped]
from fastapi import APIRouter, Response
from sqlalchemy import create_engine, text

from api.core.config import settings
from api.schemas import HealthCheckData, HealthCheckEnvelope

router = APIRouter()

logger = logging.getLogger(__name__)

#: Cache policy for health checks (API.md 8.1: no-store).
CACHE_CONTROL_HEALTH = "no-store"
#: Per-dependency status reported when a dependency is reachable.
CONNECTED = "connected"
#: Per-dependency status reported when a dependency is unreachable.
DISCONNECTED = "disconnected"
#: Timeout (seconds) applied to the Redis connectivity probe.
REDIS_PROBE_TIMEOUT_SECONDS = 2.0


@router.get(
    "/health",
    response_model=HealthCheckEnvelope,
    summary="Get system health",
)
def get_system_health(response: Response) -> HealthCheckEnvelope:
    """Return connectivity status for database, Redis, and object storage.

    The probe result is deterministic per request: each dependency is probed
    live and reported as ``connected`` or ``disconnected``. The overall
    ``status`` is ``healthy`` when every dependency is connected and
    ``degraded`` otherwise (with an HTTP ``503`` status).
    """
    database = _database_connected()
    redis_connected = _redis_connected()
    object_storage = _object_storage_connected()
    healthy = database and redis_connected and object_storage

    data = HealthCheckData(
        status="healthy" if healthy else "degraded",
        version=_app_version(),
        database=_status(database),
        redis=_status(redis_connected),
        object_storage=_status(object_storage),
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_HEALTH
    if not healthy:
        response.status_code = 503
    return HealthCheckEnvelope(data=data)


def _status(connected: bool) -> str:
    """Map a probe result to the documented per-dependency status string."""
    return CONNECTED if connected else DISCONNECTED


@router.post(
    "/admin/prewarm-vector",
    summary="Run one wind vector-field cache prewarm pass",
)
async def trigger_vector_prewarm(response: Response) -> dict[str, Any]:
    """Run one bounded prewarm pass synchronously and report its counters.

    Ops/debug switch for the background prewarm loop: computes at most
    ``API_VECTOR_PREWARM_MAX_COMPUTES_PER_PASS`` cache-missing vector fields
    (throttled) and returns the pass counters. Cache-warm passes return
    immediately.
    """
    from api.services.vector_prewarm import run_prewarm_pass

    response.headers["Cache-Control"] = CACHE_CONTROL_HEALTH
    result = await asyncio.to_thread(run_prewarm_pass)
    return {"data": asdict(result)}


@router.post(
    "/admin/prewarm-tiles",
    summary="Run one low-zoom map tile cache prewarm pass",
)
async def trigger_tile_prewarm(response: Response) -> dict[str, Any]:
    """Run one low-zoom map tile prewarm pass synchronously and report counters.

    Ops/debug and ingestion-completion trigger: prewarms Zoom 0..2 raster tiles
    for configured default models and variables (21 tiles) and returns pass
    counters.
    """
    from api.services.tile_prewarm import run_tile_prewarm_pass

    response.headers["Cache-Control"] = CACHE_CONTROL_HEALTH
    result = await asyncio.to_thread(run_tile_prewarm_pass)
    return {"data": asdict(result)}


def _app_version() -> str:
    """Return the API contract version.

    Imported lazily to avoid a circular import: ``api.main`` imports this
    router module before its ``APP_VERSION`` module constant is defined, so a
    module-level import here would fail during application construction.
    """
    from api.main import APP_VERSION

    return APP_VERSION


def _database_connected() -> bool:
    """Probe PostgreSQL connectivity with ``SELECT 1``.

    Returns:
        True when the database answers the probe, False otherwise.
    """
    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - connectivity probe
        logger.warning("Database health probe failed: %s", exc)
        return False
    finally:
        engine.dispose()


def _redis_connected() -> bool:
    """Probe Redis connectivity with ``PING``.

    Returns:
        True when Redis answers the probe, False otherwise.
    """
    try:
        client = redis_lib.from_url(  # type: ignore[no-untyped-call]
            settings.REDIS_URL,
            socket_connect_timeout=REDIS_PROBE_TIMEOUT_SECONDS,
            socket_timeout=REDIS_PROBE_TIMEOUT_SECONDS,
        )
        client.ping()
        return True
    except redis_lib.RedisError as exc:
        logger.warning("Redis health probe failed: %s", exc)
        return False


def _redis_memory_stats() -> tuple[float, float, float]:
    """Probe Redis memory usage, configured cap, and cumulative evictions.

    All values are best-effort: ``(0, 0, 0)`` is returned when Redis is
    unreachable or answers with an unexpected payload, so the metrics
    endpoint itself never fails. ``max_bytes`` of ``0`` means "no cap
    configured" (Redis ``maxmemory 0``).

    Returns:
        ``(used_bytes, max_bytes, evicted_keys_total)``.
    """
    try:
        client = redis_lib.from_url(  # type: ignore[no-untyped-call]
            settings.REDIS_URL,
            socket_connect_timeout=REDIS_PROBE_TIMEOUT_SECONDS,
            socket_timeout=REDIS_PROBE_TIMEOUT_SECONDS,
        )
        memory = client.info("memory")
        stats = client.info("stats")
        return (
            float(memory.get("used_memory", 0)),
            float(memory.get("maxmemory", 0)),
            float(stats.get("evicted_keys", 0)),
        )
    except redis_lib.RedisError as exc:
        logger.warning("Redis memory stats probe failed: %s", exc)
        return (0.0, 0.0, 0.0)
    except Exception:  # noqa: BLE001 - metrics must never fail the endpoint
        return (0.0, 0.0, 0.0)


def _object_storage_connected() -> bool:
    """Probe object storage (MinIO/S3) connectivity by listing the root.

    The probe lists the root of the configured endpoint, which requires no
    bucket name (the API settings do not define one).

    Returns:
        True when object storage answers the probe, False otherwise.
    """
    scheme = "https" if settings.MINIO_SECURE else "http"
    try:
        fs = s3fs.S3FileSystem(
            key=settings.MINIO_ACCESS_KEY,
            secret=settings.MINIO_SECRET_KEY,
            client_kwargs={"endpoint_url": f"{scheme}://{settings.MINIO_ENDPOINT}"},
            use_listings_cache=False,
        )
        fs.ls("")
        return True
    except Exception as exc:  # noqa: BLE001 - connectivity probe
        logger.warning("Object storage health probe failed: %s", exc)
        return False


def _process_memory() -> tuple[int, int]:
    """Return current process (RSS, VMS) in bytes."""
    import sys

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            windll = getattr(ctypes, "windll", None)
            if windll is not None:
                psapi = getattr(windll, "psapi", None)
                kernel32 = getattr(windll, "kernel32", None)
                if psapi is not None and kernel32 is not None:
                    psapi.GetProcessMemoryInfo.argtypes = [
                        wintypes.HANDLE,
                        ctypes.POINTER(PMC),
                        wintypes.DWORD,
                    ]
                    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
                    c = PMC()
                    c.cb = ctypes.sizeof(PMC)
                    h = kernel32.GetCurrentProcess()
                    if psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                        return int(c.WorkingSetSize), int(c.PagefileUsage)
        except Exception:
            pass
    else:
        try:
            from pathlib import Path

            status = Path("/proc/self/status")
            if status.is_file():
                rss, vms = 0, 0
                for line in status.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("VmSize:"):
                        vms = int(line.split()[1]) * 1024
                return rss, vms
        except Exception:
            pass
    return 0, 0


_api_cpu_lock = threading.Lock()
_api_last_cpu_time: float | None = None
_api_last_monotonic: float | None = None


def _process_cpu_percent() -> float:
    """Return current API process CPU % since previous invocation (non-blocking)."""
    global _api_last_cpu_time, _api_last_monotonic
    now_mono = time.monotonic()
    try:
        cpu_times = time.process_time()
    except Exception:
        return 0.0

    with _api_cpu_lock:
        if _api_last_cpu_time is None or _api_last_monotonic is None:
            _api_last_cpu_time = cpu_times
            _api_last_monotonic = now_mono
            return 0.0

        dt_mono = now_mono - _api_last_monotonic
        dt_cpu = cpu_times - _api_last_cpu_time
        _api_last_cpu_time = cpu_times
        _api_last_monotonic = now_mono

        if dt_mono <= 0:
            return 0.0
        cores = os.cpu_count() or 1
        pct = ((dt_cpu / dt_mono) / cores) * 100.0
        return max(0.0, min(100.0, round(pct, 2)))


@router.get(
    "/metrics",
    summary="Get Prometheus metrics for the API serving layer",
)
def get_api_metrics() -> Response:
    """Return API serving layer and platform metrics in Prometheus text exposition format."""
    import threading

    database = _database_connected()
    redis_connected = _redis_connected()
    object_storage = _object_storage_connected()
    rss, vms = _process_memory()
    cpu_pct = _process_cpu_percent()
    threads = threading.active_count()
    redis_used_bytes, redis_max_bytes, redis_evicted = _redis_memory_stats()

    reader_checked_out = 0
    reader_pool_size = 0
    try:
        from api.main import reader_pool

        if reader_pool and hasattr(reader_pool, "_engine") and reader_pool._engine:
            pool = reader_pool._engine.pool
            reader_checked_out = int(getattr(pool, "checkedout", lambda: 0)())
            reader_pool_size = int(getattr(pool, "size", lambda: 0)())
    except Exception:
        pass

    lines = [
        "# HELP weather_api_database_connected Database connectivity (1=connected, 0=disconnected)",
        "# TYPE weather_api_database_connected gauge",
        f"weather_api_database_connected {1.0 if database else 0.0}",
        "# HELP weather_api_redis_connected Redis connectivity (1=connected, 0=disconnected)",
        "# TYPE weather_api_redis_connected gauge",
        f"weather_api_redis_connected {1.0 if redis_connected else 0.0}",
        "# HELP weather_api_storage_connected Object storage connectivity (1=connected, 0=disconnected)",
        "# TYPE weather_api_storage_connected gauge",
        f"weather_api_storage_connected {1.0 if object_storage else 0.0}",
        "# HELP weather_api_process_memory_rss_bytes Process RSS in bytes",
        "# TYPE weather_api_process_memory_rss_bytes gauge",
        f"weather_api_process_memory_rss_bytes {float(rss)}",
        "# HELP weather_api_process_memory_vms_bytes Process virtual memory in bytes",
        "# TYPE weather_api_process_memory_vms_bytes gauge",
        f"weather_api_process_memory_vms_bytes {float(vms)}",
        "# HELP weather_api_process_active_threads Active threads",
        "# TYPE weather_api_process_active_threads gauge",
        f"weather_api_process_active_threads {float(threads)}",
        "# HELP weather_api_reader_pool_checked_out Reader lock pool checked out connections",
        "# TYPE weather_api_reader_pool_checked_out gauge",
        f"weather_api_reader_pool_checked_out {float(reader_checked_out)}",
        "# HELP weather_api_reader_pool_size Reader lock pool configured size",
        "# TYPE weather_api_reader_pool_size gauge",
        f"weather_api_reader_pool_size {float(reader_pool_size)}",
        "# HELP weather_api_redis_memory_used_bytes Redis memory used by all caches (bytes)",
        "# TYPE weather_api_redis_memory_used_bytes gauge",
        f"weather_api_redis_memory_used_bytes {redis_used_bytes}",
        "# HELP weather_api_redis_memory_max_bytes Redis maxmemory cap (bytes; 0 = uncapped)",
        "# TYPE weather_api_redis_memory_max_bytes gauge",
        f"weather_api_redis_memory_max_bytes {redis_max_bytes}",
        "# HELP weather_api_redis_evicted_keys_total Redis keys evicted under memory pressure since start",
        "# TYPE weather_api_redis_evicted_keys_total counter",
        f"weather_api_redis_evicted_keys_total {redis_evicted}",
        "# HELP weather_api_process_cpu_percent Process CPU utilization percentage",
        "# TYPE weather_api_process_cpu_percent gauge",
        f"weather_api_process_cpu_percent {float(cpu_pct)}",
        "# HELP weather_component_cpu_percent Component process CPU utilization percentage (0-100+)",
        "# TYPE weather_component_cpu_percent gauge",
        f'weather_component_cpu_percent{{component="api"}} {float(cpu_pct)}',
        "# HELP weather_component_memory_rss_bytes Component process resident memory size (RSS) in bytes",
        "# TYPE weather_component_memory_rss_bytes gauge",
        f'weather_component_memory_rss_bytes{{component="api"}} {float(rss)}',
    ]
    content = "\n".join(lines) + "\n"
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4",
        headers={"Cache-Control": CACHE_CONTROL_HEALTH},
    )


@router.get(
    "/health/detailed",
    summary="Get detailed system health diagnostics",
)
def get_detailed_system_health(response: Response) -> dict[str, Any]:
    """Return detailed system diagnostics including memory, threads, and pool state."""
    import threading

    database = _database_connected()
    redis_connected = _redis_connected()
    object_storage = _object_storage_connected()
    healthy = database and redis_connected and object_storage
    rss, vms = _process_memory()

    reader_checked_out = 0
    reader_pool_size = 0
    try:
        from api.main import reader_pool

        if reader_pool and hasattr(reader_pool, "_engine") and reader_pool._engine:
            pool = reader_pool._engine.pool
            reader_checked_out = int(getattr(pool, "checkedout", lambda: 0)())
            reader_pool_size = int(getattr(pool, "size", lambda: 0)())
    except Exception:
        pass

    response.headers["Cache-Control"] = CACHE_CONTROL_HEALTH
    if not healthy:
        response.status_code = 503

    return {
        "status": "healthy" if healthy else "degraded",
        "version": _app_version(),
        "dependencies": {
            "database": _status(database),
            "redis": _status(redis_connected),
            "object_storage": _status(object_storage),
        },
        "resources": {
            "rss_bytes": rss,
            "vms_bytes": vms,
            "threads": threading.active_count(),
        },
        "reader_pool": {
            "checked_out": reader_checked_out,
            "size": reader_pool_size,
        },
        "model_registration": _model_registration_report(database),
    }


def _model_registration_report(database_connected: bool) -> dict[str, Any]:
    """Report catalog models missing from the domain model registries.

    Surfaced here (as well as logged at startup) so the gap is alertable: a
    missing canonical horizon raises on the ingestion and reclamation paths,
    while a missing cadence or member count degrades silently to a default that
    distorts lag and coverage. Neither is visible to the serving tier otherwise,
    because the API needs no registry entry to keep serving.

    Best-effort and fail-open: this is a diagnostics endpoint, so an unavailable
    database is reported as "unknown" rather than failing the whole response.
    """
    if not database_connected:
        return {"status": "unknown", "detail": "database unreachable"}
    try:
        from api.core.database import SessionLocal
        from api.services.model_registration import audit_catalog_model_registration

        with SessionLocal() as db:
            audit = audit_catalog_model_registration(db)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail the probe
        return {"status": "unknown", "detail": type(exc).__name__}

    if audit.is_clean:
        return {"status": "ok", "unregistered": []}
    return {
        "status": "fatal" if audit.has_fatal else "degraded",
        "unregistered": list(audit.audited_models),
        "missing_horizon": list(audit.missing_horizon),
        "missing_cadence": list(audit.missing_cadence),
        "missing_expected_members": list(audit.missing_expected_members),
        "detail": audit.describe(),
    }

