"""Unit and integration tests for the /v1/locate endpoint."""

import pytest
from fastapi.testclient import TestClient

from api.core.config import settings
from api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_locate_untrusted_mode_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """When TRUST_CLOUDFLARE_LOCATION_HEADERS is False, headers are ignored and 404 is returned."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", False)
    resp = client.get(
        "/v1/locate",
        headers={
            "cf-iplatitude": "39.7392",
            "cf-iplongitude": "-104.9903",
            "cf-ipcity": "Denver",
            "cf-region": "Colorado",
            "cf-ipcountry": "US",
        },
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["type"] == "not_found_error"
    assert "Location unavailable" in body["error"]["message"]


def test_locate_trusted_valid_headers(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """When trusted mode is enabled and valid headers are provided, returns normalized location."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", True)
    resp = client.get(
        "/v1/locate",
        headers={
            "cf-iplatitude": "39.7392",
            "cf-iplongitude": "-104.9903",
            "cf-ipcity": "Denver",
            "cf-region": "Colorado",
            "cf-ipcountry": "US",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "private, no-cache"
    data = resp.json()
    assert data["latitude"] == pytest.approx(39.7392)
    assert data["longitude"] == pytest.approx(-104.9903)
    assert data["city"] == "Denver"
    assert data["region"] == "Colorado"
    assert data["country"] == "US"
    assert data["approximate"] is True


def test_locate_missing_headers(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing latitude or longitude header returns 404."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", True)
    # Neither header
    resp = client.get("/v1/locate")
    assert resp.status_code == 404

    # Only latitude
    resp = client.get("/v1/locate", headers={"cf-iplatitude": "39.74"})
    assert resp.status_code == 404

    # Only longitude
    resp = client.get("/v1/locate", headers={"cf-iplongitude": "-104.99"})
    assert resp.status_code == 404


def test_locate_malformed_coordinates(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-numeric coordinate strings return 404 safely without crashing."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", True)
    resp = client.get(
        "/v1/locate",
        headers={"cf-iplatitude": "invalid", "cf-iplongitude": "-104.99"},
    )
    assert resp.status_code == 404


def test_locate_nan_and_infinity_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """NaN and Infinity values are rejected with 404."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", True)
    resp1 = client.get(
        "/v1/locate",
        headers={"cf-iplatitude": "nan", "cf-iplongitude": "-104.99"},
    )
    assert resp1.status_code == 404

    resp2 = client.get(
        "/v1/locate",
        headers={"cf-iplatitude": "39.74", "cf-iplongitude": "inf"},
    )
    assert resp2.status_code == 404


def test_locate_out_of_range_coordinates_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Geodetically out-of-range coordinates return 404."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", True)
    resp1 = client.get(
        "/v1/locate",
        headers={"cf-iplatitude": "999", "cf-iplongitude": "-104.99"},
    )
    assert resp1.status_code == 404

    resp2 = client.get(
        "/v1/locate",
        headers={"cf-iplatitude": "-95.0", "cf-iplongitude": "-104.99"},
    )
    assert resp2.status_code == 404

    resp3 = client.get(
        "/v1/locate",
        headers={"cf-iplatitude": "39.74", "cf-iplongitude": "200.0"},
    )
    assert resp3.status_code == 404


def test_locate_canonicalizes_negative_180_longitude(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Longitude -180.0 canonicalizes to +180.0."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", True)
    resp = client.get(
        "/v1/locate",
        headers={"cf-iplatitude": "15.0", "cf-iplongitude": "-180.0"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["longitude"] == 180.0
