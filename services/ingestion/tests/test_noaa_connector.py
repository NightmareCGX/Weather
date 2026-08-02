"""Unit tests for the NOAA NOMADS connector with mocked HTTP transport."""

from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from ingestion.core.base import (
    DownloadFailedError,
    InvalidRunError,
    UpstreamUnavailableError,
)
from ingestion.core.config import IngestionSettings
from ingestion.providers.noaa.connector import NOAAConnector

CYCLE = date(2026, 7, 21)

BASE = "https://nomads.ncep.noaa.gov"

GFS_000_URL = (
    f"{BASE}/pub/data/nccf/com/gfs/prod/gfs.20260721/12/atmos/gfs.t12z.pgrb2.0p25.f000"
)
GFS_006_URL = (
    f"{BASE}/pub/data/nccf/com/gfs/prod/gfs.20260721/00/atmos/gfs.t00z.pgrb2.0p25.f006"
)
GFS_384_URL = (
    f"{BASE}/pub/data/nccf/com/gfs/prod/gfs.20260721/18/atmos/gfs.t18z.pgrb2.0p25.f384"
)
GEFS_006_URL = (
    f"{BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/00/atmos/pgrb2ap25/"
    "gefs.t00z.pgrb2a.0p25.f006"
)
GEFS_240_URL = (
    f"{BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/06/atmos/pgrb2ap25/"
    "gefs.t06z.pgrb2a.0p25.f240"
)
GEFS_252_URL = (
    f"{BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/06/atmos/pgrb2bp25/"
    "gefs.t06z.pgrb2b.0p25.f252"
)
GEFS_384_URL = (
    f"{BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/12/atmos/pgrb2bp25/"
    "gefs.t12z.pgrb2b.0p25.f384"
)


def _settings(**overrides: object) -> IngestionSettings:
    """Build test settings with fast, non-blocking retries."""
    values: dict[str, object] = {
        "NOMADS_BASE_URL": BASE,
        "REQUEST_TIMEOUT_SECONDS": 5.0,
        "DOWNLOAD_RETRIES": 2,
        "RETRY_BACKOFF_SECONDS": 0.0,
    }
    values.update(overrides)
    return IngestionSettings(**values)


class _BrokenBody(httpx.AsyncByteStream):
    """Streams a partial body, then fails mid-transfer like a dropped connection."""

    entered = 0

    async def __aiter__(self):
        type(self).entered += 1
        yield b"partial-grib-data"
        raise httpx.ReadError("connection lost mid-stream")

    async def aclose(self) -> None:
        pass


@pytest.mark.parametrize(
    ("model", "cycle_hour", "lead_time_hours", "expected_url"),
    [
        ("gfs", 0, 6, GFS_006_URL),
        ("gfs", 12, 0, GFS_000_URL),
        ("gfs", 18, 384, GFS_384_URL),
        ("gefs", 0, 6, GEFS_006_URL),
        ("gefs", 6, 240, GEFS_240_URL),
        ("gefs", 6, 252, GEFS_252_URL),
        ("gefs", 12, 384, GEFS_384_URL),
    ],
)
def test_build_url_is_deterministic(
    model: str, cycle_hour: int, lead_time_hours: int, expected_url: str
) -> None:
    connector = NOAAConnector(conn_settings=_settings())
    assert (
        connector.build_url(model, CYCLE, cycle_hour, lead_time_hours) == expected_url
    )


@pytest.mark.parametrize(
    ("model", "cycle_hour", "lead_time_hours"),
    [
        ("ecmwf", 0, 6),
        ("gfs", 3, 6),
        ("gfs", 0, -1),
        ("gfs", 0, 385),
    ],
)
def test_build_url_rejects_invalid_runs(
    model: str, cycle_hour: int, lead_time_hours: int
) -> None:
    connector = NOAAConnector(conn_settings=_settings())
    with pytest.raises(InvalidRunError):
        connector.build_url(model, CYCLE, cycle_hour, lead_time_hours)


@respx.mock
async def test_download_streams_to_destination(tmp_path: Path) -> None:
    content = b"GRIB\x01\x02data"
    respx.get(GFS_006_URL).mock(return_value=httpx.Response(200, content=content))
    async with NOAAConnector(conn_settings=_settings()) as connector:
        destination = await connector.download(
            "gfs", CYCLE, 0, 6, tmp_path / "nested" / "gfs.t00z.pgrb2.0p25.f006"
        )
    assert destination == tmp_path / "nested" / "gfs.t00z.pgrb2.0p25.f006"
    assert destination.read_bytes() == content


@respx.mock
async def test_download_gefs_beyond_240_hours(tmp_path: Path) -> None:
    content = b"gefs-grib"
    respx.get(GEFS_252_URL).mock(return_value=httpx.Response(200, content=content))
    async with NOAAConnector(conn_settings=_settings()) as connector:
        destination = await connector.download(
            "gefs", CYCLE, 6, 252, tmp_path / "gefs.f252.grib2"
        )
    assert destination.read_bytes() == content


@respx.mock
async def test_download_missing_file_fails_without_retry(tmp_path: Path) -> None:
    route = respx.get(GFS_006_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(DownloadFailedError, match="not found"):
        async with NOAAConnector(conn_settings=_settings()) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "missing.grib2")
    assert route.call_count == 1
    assert not (tmp_path / "missing.grib2").exists()


@respx.mock
async def test_download_4xx_forbidden_fails_without_retry(tmp_path: Path) -> None:
    route = respx.get(GFS_006_URL).mock(return_value=httpx.Response(403))
    with pytest.raises(DownloadFailedError, match="HTTP 403"):
        async with NOAAConnector(conn_settings=_settings()) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "forbidden.grib2")
    assert route.call_count == 1
    assert not (tmp_path / "forbidden.grib2").exists()


@respx.mock
async def test_download_3xx_redirect_is_rejected(tmp_path: Path) -> None:
    route = respx.get(GFS_006_URL).mock(
        return_value=httpx.Response(
            302, headers={"Location": "https://example.test/redirected.grib2"}
        )
    )
    with pytest.raises(DownloadFailedError, match="HTTP 302"):
        async with NOAAConnector(conn_settings=_settings()) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "redirected.grib2")
    assert route.call_count == 1
    assert not (tmp_path / "redirected.grib2").exists()


@respx.mock
async def test_download_retries_transient_5xx_then_succeeds(tmp_path: Path) -> None:
    content = b"retried-grib"
    route = respx.get(GFS_006_URL).mock(
        side_effect=[httpx.Response(500), httpx.Response(200, content=content)]
    )
    async with NOAAConnector(conn_settings=_settings()) as connector:
        destination = await connector.download(
            "gfs", CYCLE, 0, 6, tmp_path / "retried.grib2"
        )
    assert route.call_count == 2
    assert destination.read_bytes() == content


@respx.mock
async def test_download_5xx_exhausts_retries(tmp_path: Path) -> None:
    route = respx.get(GFS_006_URL).mock(return_value=httpx.Response(500))
    with pytest.raises(DownloadFailedError, match="HTTP 500"):
        async with NOAAConnector(
            conn_settings=_settings(DOWNLOAD_RETRIES=2)
        ) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "fail.grib2")
    assert route.call_count == 3  # initial attempt + 2 retries
    assert not (tmp_path / "fail.grib2").exists()


@respx.mock
async def test_download_upstream_unavailable_exhausts_retries(
    tmp_path: Path,
) -> None:
    route = respx.get(GFS_006_URL).mock(
        side_effect=[httpx.ConnectError("connection refused")] * 3
    )
    with pytest.raises(UpstreamUnavailableError, match="unreachable"):
        async with NOAAConnector(
            conn_settings=_settings(DOWNLOAD_RETRIES=2)
        ) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "net.grib2")
    assert route.call_count == 3
    assert not (tmp_path / "net.grib2").exists()


@respx.mock
async def test_download_stream_failure_deletes_partial_file(tmp_path: Path) -> None:
    _BrokenBody.entered = 0
    route = respx.get(GFS_006_URL).mock(
        return_value=httpx.Response(200, stream=_BrokenBody())
    )
    with pytest.raises(UpstreamUnavailableError, match="unreachable"):
        async with NOAAConnector(
            conn_settings=_settings(DOWNLOAD_RETRIES=2)
        ) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "partial.grib2")
    assert route.call_count == 3
    # The write loop was entered on every attempt, so a partial file was
    # created each time and must have been cleaned up by the failure handler.
    assert _BrokenBody.entered == 3
    assert not (tmp_path / "partial.grib2").exists()
