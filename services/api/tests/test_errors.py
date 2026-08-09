"""Regression tests for RFC 7807 error handling and the X-Request-Id header.

``docs/API.md`` section 2.7 requires ``X-Request-Id`` on **every** response.
The generic unhandled-exception handler runs on Starlette's outermost
``ServerErrorMiddleware``, outside ``RequestIDMiddleware``, so the middleware's
post-``call_next`` header assignment never executes on unhandled 500s. These
tests pin that behavior and the non-regression of the handled 200/404/422
paths.

The module is self-contained: it builds the real application with
``create_app`` and mounts a single route that raises, so it exercises the
production error-handling path with no database dependency.
"""

import sys
from pathlib import Path

# Ensure services/api/src is on sys.path (mirrors conftest.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

from api.main import create_app

REQUEST_ID_HEADER = "X-Request-Id"


@pytest.fixture(scope="module")
def error_client():
    """A TestClient for the real app with an unhandled-exception route mounted.

    ``raise_server_exceptions=False`` keeps an unhandled ``RuntimeError`` from
    bubbling out of the client so the RFC 7807 500 response (and its headers)
    can be asserted.
    """
    app = create_app()

    @app.get("/__ok__")
    def _ok():
        return {"ok": True}

    @app.get("/__boom__")
    def _boom():
        raise RuntimeError("test boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_ok_response_carries_request_id(error_client):
    """A normal 200 response carries X-Request-Id (regression guard)."""
    resp = error_client.get("/__ok__")
    assert resp.status_code == 200
    assert REQUEST_ID_HEADER in resp.headers
    assert resp.headers[REQUEST_ID_HEADER].startswith("req_")


def test_not_found_error_carries_request_id(error_client):
    """A 404 response carries X-Request-Id and the header matches the body."""
    resp = error_client.get("/__does_not_exist__")
    assert resp.status_code == 404
    assert REQUEST_ID_HEADER in resp.headers
    assert resp.headers[REQUEST_ID_HEADER].startswith("req_")
    assert resp.json()["error"]["request_id"] == resp.headers[REQUEST_ID_HEADER]


def test_validation_error_carries_request_id(error_client):
    """A 422 validation-error response carries X-Request-Id and matches."""
    resp = error_client.get("/v1/search")  # missing required ``q``
    assert resp.status_code == 422
    assert REQUEST_ID_HEADER in resp.headers
    assert resp.headers[REQUEST_ID_HEADER].startswith("req_")
    assert resp.json()["error"]["request_id"] == resp.headers[REQUEST_ID_HEADER]


def test_unhandled_500_carries_request_id(error_client):
    """An unhandled 500 response carries X-Request-Id (API.md section 2.7).

    This is the path that previously dropped the header: the generic
    ``Exception`` handler runs on Starlette's ServerErrorMiddleware, outside
    ``RequestIDMiddleware``. The error body carries ``request_id``; the header
    must now match it.
    """
    resp = error_client.get("/__boom__")
    assert resp.status_code == 500
    assert REQUEST_ID_HEADER in resp.headers
    assert resp.headers[REQUEST_ID_HEADER].startswith("req_")
    assert resp.json()["error"]["request_id"] == resp.headers[REQUEST_ID_HEADER]
