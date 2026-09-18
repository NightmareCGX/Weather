"""Unit tests for the local GeoLite2 locator (api.core.geoip)."""

import ipaddress
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from maxminddb.errors import InvalidDatabaseError
from starlette.requests import Request

from api.core import geoip
from api.core.config import settings

#: A realistic GeoLite2-City record for Denver.
DENVER_RECORD: dict[str, Any] = {
    "city": {"names": {"en": "Denver", "de": "Denver"}},
    "subdivisions": [{"iso_code": "CO", "names": {"en": "Colorado"}}],
    "country": {"iso_code": "US"},
    "location": {"latitude": 39.7392, "longitude": -104.9903, "time_zone": "America/Denver"},
}


class _StubReader:
    """Stand-in for ``maxminddb.Reader`` mirroring its ``get`` contract."""

    def __init__(self, records: dict[str, Any] | None = None) -> None:
        self._records = records or {}
        self.closed = False
        self.queries: list[str] = []

    def get(self, ip_address: str) -> Any:
        # maxminddb validates the address and raises ValueError on garbage.
        ipaddress.ip_address(ip_address)
        self.queries.append(ip_address)
        return self._records.get(ip_address)

    def close(self) -> None:
        self.closed = True


class _FailingReader:
    """Reader that reports a corrupt database on every lookup."""

    def __init__(self) -> None:
        self.closed = False

    def get(self, ip_address: str) -> Any:
        msg = "corrupt search tree"
        raise InvalidDatabaseError(msg)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    """Keep the module-level reader cache from leaking between tests."""
    geoip._reset_for_testing()
    yield
    geoip._reset_for_testing()


@pytest.fixture
def database_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the locator at a stub database file."""
    db_path = tmp_path / "GeoLite2-City.mmdb"
    db_path.write_bytes(b"stub")
    monkeypatch.setattr(settings, "LOCATE_GEOIP_DB_PATH", str(db_path))
    return db_path


def _request(
    *,
    headers: dict[str, str] | None = None,
    client: tuple[str, int] | None = ("8.8.8.8", 51234),
) -> Request:
    raw_headers = [(name.lower().encode(), value.encode()) for name, value in (headers or {}).items()]
    return Request({"type": "http", "headers": raw_headers, "client": client})


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("8.8.8.8", True),
        ("2001:4860:4860::8888", True),
        ("::ffff:8.8.8.8", True),
        ("127.0.0.1", False),
        ("10.11.12.13", False),
        ("192.168.1.20", False),
        ("169.254.1.1", False),
        ("100.64.0.1", False),
        ("::1", False),
        ("not-an-address", False),
        ("", False),
    ],
)
def test_is_public_ip(address: str, expected: bool) -> None:
    assert geoip.is_public_ip(address) is expected


def test_client_ip_auto_trusts_header_from_a_non_routable_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that can only have come from the gateway carries the visitor."""
    monkeypatch.setattr(settings, "LOCATE_PROXY_MODE", "auto")
    request = _request(headers={"x-real-ip": " 1.2.3.4 "}, client=("172.17.0.1", 51234))
    assert geoip.client_ip(request) == "1.2.3.4"


def test_client_ip_auto_ignores_header_from_a_routable_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that reached the published API port cannot choose its location."""
    monkeypatch.setattr(settings, "LOCATE_PROXY_MODE", "auto")
    request = _request(headers={"x-real-ip": "1.2.3.4"}, client=("93.184.216.34", 51234))
    assert geoip.client_ip(request) == "93.184.216.34"


def test_client_ip_always_trusts_header_regardless_of_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LOCATE_PROXY_MODE", "always")
    request = _request(headers={"x-real-ip": "1.2.3.4"}, client=("93.184.216.34", 51234))
    assert geoip.client_ip(request) == "1.2.3.4"


def test_client_ip_never_trusts_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LOCATE_PROXY_MODE", "never")
    request = _request(headers={"x-real-ip": "1.2.3.4"}, client=("172.17.0.1", 51234))
    assert geoip.client_ip(request) == "172.17.0.1"


@pytest.mark.parametrize("header", [None, "", "   "])
def test_client_ip_falls_back_to_peer_without_usable_header(
    monkeypatch: pytest.MonkeyPatch, header: str | None
) -> None:
    monkeypatch.setattr(settings, "LOCATE_PROXY_MODE", "auto")
    headers = {} if header is None else {"x-real-ip": header}
    assert geoip.client_ip(_request(headers=headers, client=("172.17.0.1", 51234))) == "172.17.0.1"


def test_client_ip_without_peer_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LOCATE_PROXY_MODE", "auto")
    assert geoip.client_ip(_request(client=None)) is None


def test_get_reader_warns_once_when_database_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "LOCATE_GEOIP_DB_PATH", str(tmp_path / "absent.mmdb"))

    with caplog.at_level(logging.WARNING, logger="api.core.geoip"):
        assert geoip._get_reader() is None
        assert geoip._get_reader() is None

    warnings = [record for record in caplog.records if record.name == "api.core.geoip"]
    assert len(warnings) == 1


def test_get_reader_opens_once_and_caches(
    database_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = _StubReader()
    opened: list[str] = []

    def _open(path: str) -> _StubReader:
        opened.append(path)
        return reader

    monkeypatch.setattr(geoip, "_open_reader", _open)

    assert geoip._get_reader() is reader
    assert geoip._get_reader() is reader
    assert opened == [str(database_file)]


def test_get_reader_reopens_and_closes_previous_after_refresh(
    database_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readers = [_StubReader(), _StubReader()]
    monkeypatch.setattr(geoip, "_open_reader", lambda path: readers.pop(0))

    first = geoip._get_reader()
    assert first is not None

    # An atomic replace with a different file size is how geoipupdate lands.
    database_file.write_bytes(b"refreshed-database")
    second = geoip._get_reader()

    assert second is not None
    assert second is not first
    assert first.closed is True


def test_get_reader_reopens_on_mtime_change_alone(
    database_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readers = [_StubReader(), _StubReader()]
    monkeypatch.setattr(geoip, "_open_reader", lambda path: readers.pop(0))

    first = geoip._get_reader()
    assert first is not None
    assert geoip._get_reader() is first

    os.utime(database_file, (1_700_000_000, 1_700_000_000))
    assert geoip._get_reader() is not first


def test_get_reader_returns_none_when_open_fails(
    database_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _open(path: str) -> _StubReader:
        msg = "not a MaxMind DB file"
        raise InvalidDatabaseError(msg)

    monkeypatch.setattr(geoip, "_open_reader", _open)

    with caplog.at_level(logging.WARNING, logger="api.core.geoip"):
        assert geoip._get_reader() is None
    assert any("Failed to open GeoLite2 database" in record.message for record in caplog.records)


def _reader_for(records: dict[str, Any]) -> _StubReader:
    return _StubReader(records)


def test_lookup_location_parses_full_record(
    database_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(geoip, "_open_reader", lambda path: _reader_for({"8.8.8.8": DENVER_RECORD}))

    location = geoip.lookup_location("8.8.8.8")

    assert location is not None
    assert location.latitude == pytest.approx(39.7392)
    assert location.longitude == pytest.approx(-104.9903)
    assert location.city == "Denver"
    assert location.region == "Colorado"
    assert location.country == "US"


def test_lookup_location_falls_back_to_subdivision_iso_code(
    database_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {**DENVER_RECORD, "subdivisions": [{"iso_code": "CO"}]}
    monkeypatch.setattr(geoip, "_open_reader", lambda path: _reader_for({"8.8.8.8": record}))

    location = geoip.lookup_location("8.8.8.8")

    assert location is not None
    assert location.region == "CO"


def test_lookup_location_without_subdivisions_keeps_coordinates(
    database_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {**DENVER_RECORD, "subdivisions": []}
    monkeypatch.setattr(geoip, "_open_reader", lambda path: _reader_for({"8.8.8.8": record}))

    location = geoip.lookup_location("8.8.8.8")

    assert location is not None
    assert location.region is None
    assert location.city == "Denver"


@pytest.mark.parametrize(
    "record",
    [
        pytest.param({"city": {"names": {"en": "Nowhere"}}}, id="no-location"),
        pytest.param({"location": {"latitude": 12.0}}, id="latitude-only"),
        pytest.param({"location": {"latitude": 0.0, "longitude": 0.0}}, id="null-island"),
        pytest.param({"location": "not-a-dict"}, id="malformed-location"),
    ],
)
def test_lookup_location_returns_none_for_unusable_records(
    database_file: Path, monkeypatch: pytest.MonkeyPatch, record: dict[str, Any]
) -> None:
    monkeypatch.setattr(geoip, "_open_reader", lambda path: _reader_for({"8.8.8.8": record}))
    assert geoip.lookup_location("8.8.8.8") is None


def test_lookup_location_returns_none_for_unknown_address(
    database_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(geoip, "_open_reader", lambda path: _reader_for({"8.8.8.8": DENVER_RECORD}))
    assert geoip.lookup_location("1.1.1.1") is None


def test_lookup_location_returns_none_for_malformed_address(
    database_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(geoip, "_open_reader", lambda path: _reader_for({"8.8.8.8": DENVER_RECORD}))
    assert geoip.lookup_location("not-an-address") is None


def test_lookup_location_returns_none_without_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LOCATE_GEOIP_DB_PATH", str(tmp_path / "absent.mmdb"))
    assert geoip.lookup_location("8.8.8.8") is None


def test_lookup_location_discards_corrupt_reader(
    database_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt database must not poison every later request in the process."""
    corrupt = _FailingReader()
    healthy = _StubReader({"8.8.8.8": DENVER_RECORD})
    monkeypatch.setattr(geoip, "_open_reader", lambda path: corrupt if not corrupt.closed else healthy)

    with caplog.at_level(logging.WARNING, logger="api.core.geoip"):
        assert geoip.lookup_location("8.8.8.8") is None

    assert corrupt.closed is True
    location = geoip.lookup_location("8.8.8.8")
    assert location is not None
    assert location.city == "Denver"
