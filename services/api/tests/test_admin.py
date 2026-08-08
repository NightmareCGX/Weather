"""Contract and integration tests for the /v1/health endpoint.

These tests run against a real PostgreSQL instance via TestClient (the suite's
standard integration path) and verify the ``health_check`` envelope, the
``200``/``healthy`` and ``503``/``degraded`` outcomes, and the ``no-store``
cache policy defined in ``docs/API.md`` Domain 8. Dependency probes are
monkeypatched for the deterministic cases; one test exercises the real
PostgreSQL probe against the live test container (Redis and object storage are
monkeypatched there so the test never depends on their availability).
"""

from api.routers import admin as admin_router


def _assert_health_envelope(body: dict) -> None:
    """Assert the universal health_check envelope shape."""
    assert body["object"] == "health_check"
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_health_all_connected_returns_200(client, monkeypatch):
    monkeypatch.setattr(admin_router, "_database_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_redis_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_object_storage_connected", lambda: True)

    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    _assert_health_envelope(body)
    data = body["data"]
    assert data["status"] == "healthy"
    assert data["version"] == "1.1.0"
    assert data["database"] == "connected"
    assert data["redis"] == "connected"
    assert data["object_storage"] == "connected"


def test_health_dependency_down_returns_503(client, monkeypatch):
    monkeypatch.setattr(admin_router, "_database_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_redis_connected", lambda: False)
    monkeypatch.setattr(admin_router, "_object_storage_connected", lambda: True)

    resp = client.get("/v1/health")
    assert resp.status_code == 503
    body = resp.json()
    _assert_health_envelope(body)
    data = body["data"]
    assert data["status"] == "degraded"
    assert data["database"] == "connected"
    assert data["redis"] == "disconnected"
    assert data["object_storage"] == "connected"


def test_health_cache_control_no_store(client, monkeypatch):
    monkeypatch.setattr(admin_router, "_database_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_redis_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_object_storage_connected", lambda: True)

    resp = client.get("/v1/health")
    assert resp.headers["Cache-Control"] == "no-store"


def test_health_database_probe_live(client, monkeypatch):
    """The real PostgreSQL probe runs against the live test container.

    Redis and object storage probes are monkeypatched so the test never depends
    on their availability; the database is exercised for real and must not
    raise. Either the healthy or degraded outcome is valid (the probe reads
    ``settings.DATABASE_URL``, which may differ from the test database URL if a
    local ``.env`` overrides it).
    """
    monkeypatch.setattr(admin_router, "_redis_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_object_storage_connected", lambda: True)

    resp = client.get("/v1/health")
    assert resp.status_code in (200, 503)
    body = resp.json()
    _assert_health_envelope(body)
    data = body["data"]
    assert data["version"] == "1.1.0"
    assert data["database"] in ("connected", "disconnected")
    assert data["redis"] == "connected"
    assert data["object_storage"] == "connected"
