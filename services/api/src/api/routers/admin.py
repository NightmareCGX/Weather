"""Administration endpoint: system health (API.md section 8.1).

Returns the connectivity status of the API's backend dependencies (PostgreSQL,
Redis, and object storage). The router is thin (ENGINEERING_CONTRACT section
2): it runs lightweight connectivity probes and serializes the documented
``health_check`` envelope. When every dependency is connected the response is
``200`` with ``status: healthy``; when any dependency is unavailable it is
``503`` with ``status: degraded`` and the failing dependency reported as
``disconnected``.
"""

import logging

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
    # A stalled connection (e.g. a network partition) must not block a
    # request thread for tens of seconds: apply a 2s connect timeout so the
    # probe fails fast, matching the Redis probe's timeout behavior.
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 2},
    )
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
        )
        fs.ls("")
        return True
    except Exception as exc:  # noqa: BLE001 - connectivity probe
        logger.warning("Object storage health probe failed: %s", exc)
        return False
