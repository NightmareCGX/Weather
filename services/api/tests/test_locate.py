"""Unit and integration tests for the /v1/locate endpoint."""

import ipaddress
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.core import geoip
from api.core.config import settings
from api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


#: Minimal GeoLite2-City record for a Denver visitor.
DENVER_RECORD: dict[str, Any] = {
    "city": {"names": {"en": "Denver"}},
    "subdivisions": [{"iso_code": "CO", "names": {"en": "Colorado"}}],
    "country": {"iso_code": "US"},
    "location": {"latitude": 39.7392, "longitude": -104.9903},
}


class _StubReader:
    """Stand-in for ``maxminddb.Reader`` mirroring its ``get`` contract."""

    def __init__(self, records: dict[str, Any]) -> None:
        self._records = records
        self.closed = False

    def get(self, ip_address: str) -> Any:
        # maxminddb validates the address and raises ValueError on garbage.
        ipaddress.ip_address(ip_address)
        return self._records.get(ip_address)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_geoip_state() -> Iterator[None]:
    """Keep the module-level reader cache from leaking between tests."""
    geoip._reset_for_testing()
    yield
    geoip._reset_for_testing()


@pytest.fixture
def geolite_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _StubReader:
    """Enable the local-database source against a stub database file and reader."""
    db_path = tmp_path / "GeoLite2-City.mmdb"
    db_path.write_bytes(b"stub")
    reader = _StubReader({"8.8.8.8": DENVER_RECORD})
    monkeypatch.setattr(settings, "LOCATE_PROVIDER", "maxmind")
    monkeypatch.setattr(settings, "LOCATE_GEOIP_DB_PATH", str(db_path))
    monkeypatch.setattr(geoip, "_open_reader", lambda path: reader)
    return reader


#: The gateway always sits in front of this tier, so a request carries the
#: visitor address in X-Real-IP. The TestClient peer ("testclient") is not a
#: routable address, which is what puts ``LOCATE_PROXY_MODE=auto`` into its
#: trusted-proxy branch — exactly as a real gateway request would.
VISITOR_HEADERS = {"x-real-ip": "8.8.8.8"}



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


def test_locate_database_source_disabled_by_default(client: TestClient) -> None:
    """With LOCATE_PROVIDER at its default, a routable visitor still gets 404."""
    resp = client.get("/v1/locate", headers=VISITOR_HEADERS)
    assert resp.status_code == 404
    assert "Location unavailable" in resp.json()["error"]["message"]


def test_locate_database_resolves_public_address(
    client: TestClient, geolite_database: _StubReader
) -> None:
    """A routable visitor is resolved from the local database."""
    resp = client.get("/v1/locate", headers=VISITOR_HEADERS)

    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "private, no-cache"
    data = resp.json()
    assert data["latitude"] == pytest.approx(39.7392)
    assert data["longitude"] == pytest.approx(-104.9903)
    assert data["city"] == "Denver"
    assert data["region"] == "Colorado"
    assert data["country"] == "US"
    assert data["approximate"] is True


def test_locate_database_rejects_non_public_address(
    client: TestClient, geolite_database: _StubReader
) -> None:
    """Loopback, private and malformed addresses never reach the database."""
    for address in ("127.0.0.1", "192.168.1.20", "10.0.0.5", "not-an-address"):
        resp = client.get("/v1/locate", headers={"x-real-ip": address})
        assert resp.status_code == 404, address


def test_locate_proxy_mode_never_ignores_header(
    client: TestClient, geolite_database: _StubReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LOCATE_PROXY_MODE", "never")
    resp = client.get("/v1/locate", headers=VISITOR_HEADERS)
    assert resp.status_code == 404


def test_locate_database_unknown_address_returns_404(
    client: TestClient, geolite_database: _StubReader
) -> None:
    """A public address absent from the database degrades to 404."""
    resp = client.get("/v1/locate", headers={"x-real-ip": "1.1.1.1"})
    assert resp.status_code == 404


def test_locate_database_missing_file_returns_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unprovisioned database degrades to 404 instead of failing the request."""
    monkeypatch.setattr(settings, "LOCATE_PROVIDER", "maxmind")
    monkeypatch.setattr(settings, "LOCATE_GEOIP_DB_PATH", str(tmp_path / "absent.mmdb"))
    resp = client.get("/v1/locate", headers=VISITOR_HEADERS)
    assert resp.status_code == 404
    assert "Location unavailable" in resp.json()["error"]["message"]


def test_locate_cloudflare_headers_win_over_database(
    client: TestClient, geolite_database: _StubReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both sources are enabled, the header source is consulted first."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", True)
    resp = client.get(
        "/v1/locate",
        headers={
            **VISITOR_HEADERS,
            "cf-iplatitude": "47.6062",
            "cf-iplongitude": "-122.3321",
            "cf-ipcity": "Seattle",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["latitude"] == pytest.approx(47.6062)
    assert data["city"] == "Seattle"


def test_locate_unusable_cloudflare_headers_fall_through_to_database(
    client: TestClient, geolite_database: _StubReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed or absent header pair must not mask a working database source."""
    monkeypatch.setattr(settings, "TRUST_CLOUDFLARE_LOCATION_HEADERS", True)

    no_headers = client.get("/v1/locate", headers=VISITOR_HEADERS)
    assert no_headers.status_code == 200
    assert no_headers.json()["city"] == "Denver"

    malformed = client.get(
        "/v1/locate",
        headers={**VISITOR_HEADERS, "cf-iplatitude": "invalid", "cf-iplongitude": "-104.99"},
    )
    assert malformed.status_code == 200
    assert malformed.json()["city"] == "Denver"


def test_locate_database_partial_record_keeps_coordinates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A record without city/subdivision still yields a usable viewport."""
    db_path = tmp_path / "GeoLite2-City.mmdb"
    db_path.write_bytes(b"stub")
    reader = _StubReader({"8.8.8.8": {"location": {"latitude": 34.05, "longitude": -118.24}}})
    monkeypatch.setattr(settings, "LOCATE_PROVIDER", "maxmind")
    monkeypatch.setattr(settings, "LOCATE_GEOIP_DB_PATH", str(db_path))
    monkeypatch.setattr(geoip, "_open_reader", lambda path: reader)

    resp = client.get("/v1/locate", headers=VISITOR_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["latitude"] == pytest.approx(34.05)
    assert data["city"] is None
    assert data["region"] is None
