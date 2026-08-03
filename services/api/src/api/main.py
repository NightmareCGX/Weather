"""FastAPI application entrypoint for the weather platform API.

Run with ``uvicorn api.main:app`` from ``services/api``.
"""

from fastapi import FastAPI

from api.errors import install_exception_handlers
from api.middleware import RequestIDMiddleware
from api.routers.catalog import router as catalog_router

#: Application title reported in the OpenAPI schema.
APP_TITLE = "Global Probabilistic Weather Forecasting Platform API"
#: Application version (matches the API contract version in docs/API.md).
APP_VERSION = "1.1.0"


def create_app() -> FastAPI:
    """Build the FastAPI application with middleware, error handling, and routes."""
    app = FastAPI(title=APP_TITLE, version=APP_VERSION)
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(catalog_router, prefix="/v1")
    return app


app = create_app()
