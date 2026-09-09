"""Integration tests for the /v1/elevation endpoint and forecast regression protection.

Verifies:
* GET /v1/elevation endpoint contract, coordinate validation, Cache-Control header.
* Provider failure returning HTTP 200 with elevation_m = null (unavailable).
* Forecast regression protection: /v1/points NEVER calls or blocks on the elevation provider.
* Cities with persisted elevation return their DB value; cities with null elevation return null.
"""

from __future__ import annotations

import threading
import pytest

from api.core import config as config_mod
from api.services import elevation as elevation_mod
from api.services.elevation import (
    ElevationProvider,
    OpenMeteoElevationProvider,
    _reset_elevation_provider_cache,
)


class _FailingElevationProvider(ElevationProvider):
    """Provider that raises an exception if invoked (used to verify non-blocking)."""

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        raise AssertionError("Elevation provider must NOT be called on this path!")


def test_elevation_endpoint_success(client, monkeypatch) -> None:
    """GET /v1/elevation returns elevation_m on provider success."""
    _reset_elevation_provider_cache()
    monkeypatch.setattr(config_mod.settings, "ELEVATION_PROVIDER", "open_meteo")

    # Mock provider returning 2404.0
    mock_provider = OpenMeteoElevationProvider(
        transport=lambda url: (200, {"elevation": [2404.0]})
    )
    monkeypatch.setattr(elevation_mod, "_PROVIDER", mock_provider)

    resp = client.get("/v1/elevation?lat=39.1911&lon=-106.8175")
    assert resp.status_code == 200
    data = resp.json()
    assert data["latitude"] == pytest.approx(39.1911)
    assert data["longitude"] == pytest.approx(-106.8175)
    assert data["elevation_m"] == pytest.approx(2404.0)
    assert resp.headers["Cache-Control"] == "public, max-age=86400"
    _reset_elevation_provider_cache()


def test_elevation_endpoint_provider_failure_returns_null(client, monkeypatch) -> None:
    """GET /v1/elevation returns 200 with elevation_m=null and no-store when provider fails."""
    _reset_elevation_provider_cache()
    monkeypatch.setattr(config_mod.settings, "ELEVATION_PROVIDER", "open_meteo")

    # Mock provider failing (e.g. 500 or timeout)
    mock_provider = OpenMeteoElevationProvider(
        transport=lambda url: (500, {})
    )
    monkeypatch.setattr(elevation_mod, "_PROVIDER", mock_provider)

    resp = client.get("/v1/elevation?lat=39.1911&lon=-106.8175")
    assert resp.status_code == 200
    data = resp.json()
    assert data["latitude"] == pytest.approx(39.1911)
    assert data["longitude"] == pytest.approx(-106.8175)
    assert data["elevation_m"] is None
    # Verify no-store prevents negative caching of transient failures
    assert resp.headers["Cache-Control"] == "no-store"
    _reset_elevation_provider_cache()


def test_elevation_endpoint_disabled_provider_returns_null(client, monkeypatch) -> None:
    """GET /v1/elevation returns 200 with elevation_m=null and no-store when provider is 'none'."""
    _reset_elevation_provider_cache()
    monkeypatch.setattr(config_mod.settings, "ELEVATION_PROVIDER", "none")

    resp = client.get("/v1/elevation?lat=39.1911&lon=-106.8175")
    assert resp.status_code == 200
    data = resp.json()
    assert data["elevation_m"] is None
    assert resp.headers["Cache-Control"] == "no-store"
    _reset_elevation_provider_cache()


def test_elevation_endpoint_invalid_latitude(client) -> None:
    """Invalid latitude (>90 or <-90) returns 422 validation error."""
    resp = client.get("/v1/elevation?lat=95.0&lon=-106.8")
    assert resp.status_code == 422


def test_elevation_endpoint_invalid_longitude(client) -> None:
    """Invalid longitude (>180 or <-180) returns 422 validation error."""
    resp = client.get("/v1/elevation?lat=39.0&lon=195.0")
    assert resp.status_code == 422


def test_forecast_regression_protection_no_elevation_call(client, monkeypatch) -> None:
    """Critical guarantee: /v1/points does NOT call any elevation provider.

    Injects a failing provider that raises AssertionError if called.
    Verifies that coordinate forecasts, city forecasts, and resort forecasts
    all succeed without invoking the elevation provider.
    """
    _reset_elevation_provider_cache()
    monkeypatch.setattr(elevation_mod, "_PROVIDER", _FailingElevationProvider())

    # 1. Coordinate forecast: succeeds normally, elevation_m is None
    resp_coords = client.get("/v1/points?lat=38.19&lon=-106.82&models=gefs")
    assert resp_coords.status_code == 200
    coords_loc = resp_coords.json()["data"]["location"]
    assert coords_loc["resolved_via"] == "coordinates"
    assert coords_loc["elevation_m"] is None

    # 2. City forecast: succeeds normally, elevation_m is read from DB row
    resp_city = client.get("/v1/points?city_id=city_aspen&models=gefs")
    assert resp_city.status_code == 200
    city_loc = resp_city.json()["data"]["location"]
    assert city_loc["resolved_via"] == "city"
    assert city_loc["elevation_m"] == pytest.approx(2405.0)

    # 3. City with null elevation: succeeds normally, elevation_m is None
    resp_boulder = client.get("/v1/points?city_id=city_boulder&models=gefs")
    assert resp_boulder.status_code == 200
    boulder_loc = resp_boulder.json()["data"]["location"]
    assert boulder_loc["resolved_via"] == "city"
    assert boulder_loc["elevation_m"] is None

    # 4. Resort forecast: succeeds normally, elevation_m is summit_elevation_m
    resp_resort = client.get("/v1/points?resort_id=resort_aspen_mountain&models=gefs")
    assert resp_resort.status_code == 200
    resort_loc = resp_resort.json()["data"]["location"]
    assert resort_loc["resolved_via"] == "resort"
    assert resort_loc["elevation_m"] == pytest.approx(3417.0)

    _reset_elevation_provider_cache()


def test_search_exposes_city_elevation(client) -> None:
    """Location search exposes persisted elevation_m for cities."""
    resp = client.get("/v1/search?q=Aspen&type=city")
    assert resp.status_code == 200
    data = resp.json()["data"]
    aspen = next(item for item in data if item["name"] == "Aspen")
    assert aspen["elevation_m"] == pytest.approx(2405.0)

    resp_denver = client.get("/v1/search?q=Denver&type=city")
    assert resp_denver.status_code == 200
    denver = next(item for item in resp_denver.json()["data"] if item["name"] == "Denver")
    assert denver["elevation_m"] == pytest.approx(1609.0)

    resp_boulder = client.get("/v1/search?q=Boulder&type=city")
    assert resp_boulder.status_code == 200
    boulder = next(item for item in resp_boulder.json()["data"] if item["name"] == "Boulder")
    assert boulder["elevation_m"] is None


def test_forecast_concurrency_isolation_during_slow_elevation(client, monkeypatch) -> None:
    """A slow /v1/elevation call does not serialize or block concurrent /v1/points serving.

    Uses synchronization primitives (threading.Event) rather than fragile timing:
    1. The elevation provider pauses on `elevation_gate` when invoked.
    2. A background thread fires `GET /v1/elevation`.
    3. `elevation_started` confirms the elevation request is in-flight and paused.
    4. While the elevation request is still held inside the provider, the main
       thread executes `GET /v1/points`.
    5. The point forecast request MUST complete with 200 OK before `elevation_gate`
       is released.
    6. Finally, `elevation_gate` is released, and the elevation request completes.
    """
    _reset_elevation_provider_cache()
    monkeypatch.setattr(config_mod.settings, "ELEVATION_PROVIDER", "open_meteo")

    elevation_started = threading.Event()
    elevation_gate = threading.Event()
    elevation_result: dict[str, int | float | None] = {}

    def slow_transport(url: str):
        elevation_started.set()
        # Hold the elevation request until explicitly released by the test
        released = elevation_gate.wait(timeout=5.0)
        if not released:
            raise TimeoutError("Test elevation gate timed out")
        return 200, {"elevation": [2404.0]}

    mock_provider = OpenMeteoElevationProvider(transport=slow_transport)
    monkeypatch.setattr(elevation_mod, "_PROVIDER", mock_provider)

    def run_elevation():
        resp = client.get("/v1/elevation?lat=39.1911&lon=-106.8175")
        elevation_result["status"] = resp.status_code
        elevation_result["elevation_m"] = resp.json().get("elevation_m")

    bg_thread = threading.Thread(target=run_elevation, daemon=True)
    bg_thread.start()

    # Wait until the elevation request has entered the provider and is paused
    assert elevation_started.wait(timeout=2.0), "Elevation request did not start in time"

    try:
        # At this exact moment, the elevation request is blocked in the provider.
        # Now execute a point forecast request: it must return 200 immediately!
        resp_points = client.get("/v1/points?lat=38.19&lon=-106.82&models=gefs")
        assert resp_points.status_code == 200
        assert resp_points.json()["object"] == "point_forecast"

        # Verify the background elevation call is STILL blocked and has not completed yet
        assert "status" not in elevation_result, "Elevation completed prematurely"
    finally:
        # Release the elevation gate to allow the background thread to finish cleanly
        elevation_gate.set()
        bg_thread.join(timeout=3.0)

    # Now verify the elevation request completed successfully with 200
    assert elevation_result.get("status") == 200
    assert elevation_result.get("elevation_m") == pytest.approx(2404.0)

    _reset_elevation_provider_cache()
