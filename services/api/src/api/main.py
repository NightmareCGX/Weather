"""FastAPI application entrypoint for the weather platform API.

Run with ``uvicorn api.main:app`` from ``services/api``.
"""

from fastapi import FastAPI

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


def create_app() -> FastAPI:
    """Build the FastAPI application with middleware, error handling, and routes."""
    app = FastAPI(title=APP_TITLE, version=APP_VERSION)
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
