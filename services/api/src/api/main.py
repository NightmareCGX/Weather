"""FastAPI application entrypoint for the weather platform API.

Run with ``uvicorn api.main:app`` from ``services/api``.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.core.config import settings
from api.core.reader_gate import (
    ReaderGateShutdownTimeout,
    ReaderLockPool,
    ReaderGateLifecycle,
)
from api.errors import install_exception_handlers
from api.middleware import RequestIDMiddleware
from api.routers.admin import router as admin_router
from api.routers.availability import router as availability_router
from api.routers.catalog import router as catalog_router
from api.routers.ensembles import router as ensembles_router
from api.routers.maps import router as maps_router
from api.routers.points import router as points_router
from api.routers.probabilities import router as probabilities_router
from api.routers.search import router as search_router
from api.routers.verifications import router as verifications_router

#: Application title reported in the OpenAPI schema.
APP_TITLE = "Global Probabilistic Weather Forecasting Platform API"
#: Application version (matches the API contract version in docs/API.md).
APP_VERSION = "1.1.0"

logger = logging.getLogger(__name__)

#: Process-wide reader-gate infrastructure (created in lifespan).
reader_pool: ReaderLockPool
reader_lifecycle: ReaderGateLifecycle


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
        try:
            yield
        finally:
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

    app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(catalog_router, prefix="/v1")
    app.include_router(availability_router, prefix="/v1")
    app.include_router(search_router, prefix="/v1")
    app.include_router(points_router, prefix="/v1")
    app.include_router(probabilities_router, prefix="/v1")
    app.include_router(maps_router, prefix="/v1")
    app.include_router(ensembles_router, prefix="/v1")
    app.include_router(verifications_router, prefix="/v1")
    app.include_router(admin_router, prefix="/v1")
    return app


app = create_app()
