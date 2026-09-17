"""FastAPI application entrypoint for the weather platform API.

Run with ``uvicorn api.main:app`` from ``services/api``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from api.core.config import settings
from api.core.reader_gate import (
    ReaderGateShutdownTimeout,
    ReaderLockPool,
    ReaderGateLifecycle,
)
from api.core.zarr import shutdown_chunk_executor, shutdown_member_executor
from api.errors import install_exception_handlers
from api.middleware import RequestIDMiddleware
from api.routers.admin import router as admin_router
from api.routers.availability import router as availability_router
from api.routers.catalog import router as catalog_router
from api.routers.elevation import router as elevation_router
from api.routers.ensembles import router as ensembles_router
from api.routers.locate import router as locate_router
from api.routers.maps import router as maps_router
from api.routers.points import router as points_router
from api.routers.probabilities import router as probabilities_router
from api.routers.search import router as search_router
from api.routers.verifications import router as verifications_router
from api.services.tiles import shutdown_wind_executor

#: Application title reported in the OpenAPI schema.
APP_TITLE = "Global Probabilistic Weather Forecasting Platform API"
#: Application version (matches the API contract version in docs/API.md).
APP_VERSION = "1.1.0"

logger = logging.getLogger(__name__)

#: Process-wide reader-gate infrastructure (created in lifespan).
reader_pool: ReaderLockPool
reader_lifecycle: ReaderGateLifecycle


def _log_model_registration_audit() -> None:
    """Report catalog models that are missing from the domain registries.

    A model can be ingested and served while absent from the registries that
    describe its horizon, cadence, and member count. The horizon gap raises on
    the ingestion and reclamation paths, but the cadence and member-count gaps
    degrade silently to defaults that distort lag and coverage numbers — and
    nothing in the serving path would otherwise mention it. This is the one
    place that sees every catalog model at once, so it is where the gap becomes
    visible.

    Deliberately never raises. An unregistered model is a configuration gap, and
    the API does not need those registries to serve (every serving call site
    passes an explicit default), so refusing to start would convert a
    misconfiguration into a platform outage. The report is logged at CRITICAL so
    it cannot be mistaken for routine startup noise, and is also exposed on
    ``/v1/health/detailed`` so monitoring can alert on it.
    """
    try:
        from api.core.database import SessionLocal
        from api.services.model_registration import audit_catalog_model_registration

        with SessionLocal() as db:
            audit = audit_catalog_model_registration(db)
    except Exception:  # noqa: BLE001 - startup must not depend on this probe
        logger.warning(
            "model registration audit could not run at startup; "
            "/v1/health/detailed reports it on demand",
            exc_info=True,
        )
        return

    if audit.is_clean:
        return
    logger.critical(
        "model registration audit: %s. Serving continues (these gaps do not raise "
        "on the read path), but ingestion and reclamation do for a missing "
        "horizon, and cadence/member gaps silently distort lag and coverage.",
        audit.describe(),
    )


def create_app() -> FastAPI:
    """Build the FastAPI application with middleware, error handling, and routes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> "AsyncIterator[None]":
        global reader_pool, reader_lifecycle
        reader_lifecycle = ReaderGateLifecycle()
        reader_pool = ReaderLockPool(
            settings.DATABASE_URL,
            pool_size=int(settings.API_READER_LOCK_POOL_SIZE),
            max_overflow=int(settings.API_READER_LOCK_MAX_OVERFLOW),
            pool_timeout=float(settings.API_READER_LOCK_POOL_TIMEOUT_SECONDS),
        )
        _log_model_registration_audit()
        prewarm_task: asyncio.Task[None] | None = None
        if settings.API_VECTOR_PREWARM_ENABLED:
            from api.services.vector_prewarm import vector_prewarm_loop

            prewarm_task = asyncio.create_task(
                vector_prewarm_loop(), name="vector-field-prewarm"
            )
        tile_prewarm_task: asyncio.Task[None] | None = None
        if settings.API_TILE_PREWARM_ENABLED:
            from api.services.tile_prewarm import tile_prewarm_loop

            tile_prewarm_task = asyncio.create_task(
                tile_prewarm_loop(), name="tile-cache-prewarm"
            )
        try:
            yield
        finally:
            if tile_prewarm_task is not None:
                tile_prewarm_task.cancel()
                with suppress(asyncio.CancelledError):
                    await tile_prewarm_task
            if prewarm_task is not None:
                prewarm_task.cancel()
                with suppress(asyncio.CancelledError):
                    await prewarm_task
            # Shutdown drain: reject new gated ops, wait for active handlers,
            # then dispose the reader-lock engine.
            reader_lifecycle.begin_shutdown()
            try:
                reader_lifecycle.wait_drained(
                    float(settings.API_SHUTDOWN_DRAIN_TIMEOUT_SECONDS)
                )
            except ReaderGateShutdownTimeout as exc:
                logger.critical(
                    "reader-gate shutdown timed out with %d active handler(s); "
                    "not disposing the reader-lock Engine safely",
                    exc.args[0] if exc.args else 0,
                )
                raise
            reader_pool.dispose()
            shutdown_member_executor()
            shutdown_chunk_executor()
            shutdown_wind_executor()

    app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(catalog_router, prefix="/v1")
    app.include_router(availability_router, prefix="/v1")
    app.include_router(search_router, prefix="/v1")
    app.include_router(points_router, prefix="/v1")
    app.include_router(elevation_router, prefix="/v1")
    app.include_router(locate_router, prefix="/v1")
    app.include_router(probabilities_router, prefix="/v1")
    app.include_router(maps_router, prefix="/v1")
    app.include_router(ensembles_router, prefix="/v1")
    app.include_router(verifications_router, prefix="/v1")
    app.include_router(admin_router, prefix="/v1")
    return app


app = create_app()
