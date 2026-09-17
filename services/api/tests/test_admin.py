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


def test_api_metrics_endpoint(client, monkeypatch):
    monkeypatch.setattr(admin_router, "_database_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_redis_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_object_storage_connected", lambda: True)

    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["Content-Type"]
    text = resp.text
    assert "weather_api_database_connected 1.0" in text
    assert "weather_api_redis_connected 1.0" in text
    assert "weather_api_storage_connected 1.0" in text
    assert "weather_api_process_memory_rss_bytes" in text
    assert "weather_api_process_active_threads" in text


def test_api_health_detailed_endpoint(client, monkeypatch):
    monkeypatch.setattr(admin_router, "_database_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_redis_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_object_storage_connected", lambda: True)

    resp = client.get("/v1/health/detailed")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["version"] == "1.1.0"
    assert body["dependencies"]["database"] == "connected"
    assert "resources" in body
    assert "rss_bytes" in body["resources"]
    assert "threads" in body["resources"]


def test_health_detailed_reports_clean_model_registration_for_seeded_catalog(client, monkeypatch):
    """The seeded catalog (gfs, gefs) is fully registered, so the audit is clean."""
    monkeypatch.setattr(admin_router, "_database_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_redis_connected", lambda: True)
    monkeypatch.setattr(admin_router, "_object_storage_connected", lambda: True)

    body = client.get("/v1/health/detailed").json()

    assert body["model_registration"]["status"] == "ok"
    assert body["model_registration"]["unregistered"] == []


def test_health_detailed_surfaces_unregistered_model(monkeypatch):
    """A catalog model missing from the domain registries is reported, not hidden.

    This is the only place that sees every catalog model at once, so it is where
    a model ingested without its horizon/cadence/member registration becomes
    visible — and it is alertable, unlike a line in the startup log.
    """
    monkeypatch.setattr(admin_router, "_database_connected", lambda: True)

    class _FakeScalars:
        def all(self) -> list[str]:
            return ["gfs", "aigfs"]

    class _FakeSession:
        def __enter__(self) -> "_FakeSession":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def execute(self, _stmt: object) -> "_FakeSession":
            return self

        # SQLAlchemy exposes `scalars` as a method on Result, not a property.
        def scalars(self) -> _FakeScalars:
            return _FakeScalars()

    import api.core.database as database_module

    monkeypatch.setattr(database_module, "SessionLocal", lambda: _FakeSession())

    report = admin_router._model_registration_report(database_connected=True)

    assert report["status"] == "fatal"
    assert report["unregistered"] == ["aigfs"]
    assert report["missing_horizon"] == ["aigfs"]
    assert "aigfs" in report["detail"]


def test_health_detailed_model_registration_unknown_when_database_down():
    """A diagnostics endpoint must not fail its whole probe on a DB outage."""
    report = admin_router._model_registration_report(database_connected=False)
    assert report == {"status": "unknown", "detail": "database unreachable"}


def test_health_detailed_model_registration_unknown_on_query_error(monkeypatch):
    """A failing audit query degrades to "unknown" rather than raising."""
    import api.core.database as database_module

    def _boom() -> None:
        raise RuntimeError("catalog unreachable")

    monkeypatch.setattr(database_module, "SessionLocal", _boom)

    report = admin_router._model_registration_report(database_connected=True)

    assert report["status"] == "unknown"
    assert report["detail"] == "RuntimeError"

