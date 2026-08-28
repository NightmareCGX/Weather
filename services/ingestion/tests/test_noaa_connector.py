"""Unit and regression tests for the NOAA connector (AWS S3 primary with NOMADS fallback)."""

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

AWS_GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
AWS_GEFS_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
NOMADS_BASE = "https://nomads.ncep.noaa.gov"

# AWS S3 URLs (Native 0.25° grid across all leads up to 384h)
AWS_GFS_000_URL = f"{AWS_GFS_BASE}/gfs.20260721/12/atmos/gfs.t12z.pgrb2.0p25.f000"
AWS_GFS_006_URL = f"{AWS_GFS_BASE}/gfs.20260721/00/atmos/gfs.t00z.pgrb2.0p25.f006"
AWS_GFS_384_URL = f"{AWS_GFS_BASE}/gfs.20260721/18/atmos/gfs.t18z.pgrb2.0p25.f384"

AWS_GEFS_006_URL = (
    f"{AWS_GEFS_BASE}/gefs.20260721/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f006"
)
AWS_GEFS_240_URL = (
    f"{AWS_GEFS_BASE}/gefs.20260721/06/atmos/pgrb2sp25/gep17.t06z.pgrb2s.0p25.f240"
)
AWS_GEFS_252_URL = (
    f"{AWS_GEFS_BASE}/gefs.20260721/06/atmos/pgrb2sp25/gep17.t06z.pgrb2s.0p25.f252"
)
AWS_GEFS_384_URL = (
    f"{AWS_GEFS_BASE}/gefs.20260721/12/atmos/pgrb2sp25/gep30.t12z.pgrb2s.0p25.f384"
)

# NOMADS URLs (Native 0.25° grid across all leads up to 384h)
NOMADS_GFS_006_URL = (
    f"{NOMADS_BASE}/pub/data/nccf/com/gfs/prod/gfs.20260721/00/atmos/gfs.t00z.pgrb2.0p25.f006"
)
NOMADS_GEFS_006_URL = (
    f"{NOMADS_BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/00/atmos/pgrb2sp25/"
    "gep01.t00z.pgrb2s.0p25.f006"
)
NOMADS_GEFS_240_URL = (
    f"{NOMADS_BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/06/atmos/pgrb2sp25/"
    "gep17.t06z.pgrb2s.0p25.f240"
)
NOMADS_GEFS_252_URL = (
    f"{NOMADS_BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/06/atmos/pgrb2sp25/"
    "gep17.t06z.pgrb2s.0p25.f252"
)
NOMADS_GEFS_384_URL = (
    f"{NOMADS_BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/12/atmos/pgrb2sp25/"
    "gep30.t12z.pgrb2s.0p25.f384"
)


def _settings(**overrides: object) -> IngestionSettings:
    """Build test settings with fast, non-blocking retries and default AWS primary."""
    values: dict[str, object] = {
        "NOAA_DOWNLOAD_SOURCE": "aws_s3",
        "AWS_GFS_BASE_URL": AWS_GFS_BASE,
        "AWS_GEFS_BASE_URL": AWS_GEFS_BASE,
        "NOMADS_BASE_URL": NOMADS_BASE,
        "ENABLE_NOMADS_FALLBACK": True,
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


def _make_mock_grib2(data: bytes = b"payload") -> bytes:
    """Construct a mock GRIB2 message with a valid Section 0 declared length."""
    total = 16 + len(data) + 4
    return b"GRIB\x00\x00\x00\x02" + total.to_bytes(8, "big") + data + b"7777"


# ==============================================================================
# URL & Key Construction Tests
# ==============================================================================


@pytest.mark.parametrize(
    ("model", "cycle_hour", "lead_time_hours", "member", "expected_url"),
    [
        ("gfs", 0, 6, None, AWS_GFS_006_URL),
        ("gfs", 12, 0, None, AWS_GFS_000_URL),
        ("gfs", 18, 384, None, AWS_GFS_384_URL),
        ("gefs", 0, 6, 1, AWS_GEFS_006_URL),
        ("gefs", 6, 240, 17, AWS_GEFS_240_URL),
        # Required: GEFS f252 and f384 build native 0.25° pgrb2sp25 keys, NOT 0.50°
        ("gefs", 6, 252, 17, AWS_GEFS_252_URL),
        ("gefs", 12, 384, 30, AWS_GEFS_384_URL),
    ],
)
def test_build_s3_url_is_deterministic(
    model: str,
    cycle_hour: int,
    lead_time_hours: int,
    member: int | None,
    expected_url: str,
) -> None:
    connector = NOAAConnector(conn_settings=_settings())
    assert (
        connector.build_url(model, CYCLE, cycle_hour, lead_time_hours, member)
        == expected_url
    )
    assert (
        connector.build_s3_url(model, CYCLE, cycle_hour, lead_time_hours, member)
        == expected_url
    )


@pytest.mark.parametrize(
    ("model", "cycle_hour", "lead_time_hours", "member", "expected_url"),
    [
        ("gfs", 0, 6, None, NOMADS_GFS_006_URL),
        ("gefs", 0, 6, 1, NOMADS_GEFS_006_URL),
        ("gefs", 6, 240, 17, NOMADS_GEFS_240_URL),
        # Required: NOMADS mode also builds native 0.25° pgrb2sp25 for f252 and f384
        ("gefs", 6, 252, 17, NOMADS_GEFS_252_URL),
        ("gefs", 12, 384, 30, NOMADS_GEFS_384_URL),
    ],
)
def test_build_nomads_url_is_deterministic(
    model: str,
    cycle_hour: int,
    lead_time_hours: int,
    member: int | None,
    expected_url: str,
) -> None:
    connector = NOAAConnector(
        conn_settings=_settings(NOAA_DOWNLOAD_SOURCE="nomads")
    )
    assert (
        connector.build_url(model, CYCLE, cycle_hour, lead_time_hours, member)
        == expected_url
    )
    assert (
        connector.build_nomads_url(model, CYCLE, cycle_hour, lead_time_hours, member)
        == expected_url
    )


@pytest.mark.parametrize(
    ("model", "cycle_hour", "lead_time_hours", "member"),
    [
        ("ecmwf", 0, 6, None),
        ("gfs", 3, 6, None),
        ("gfs", 0, -1, None),
        ("gfs", 0, 385, None),
        ("gefs", 0, 6, None),
        ("gefs", 0, 6, 0),
        ("gefs", 0, 6, 31),
        ("gefs", 0, 385, 1),
    ],
)
def test_build_url_rejects_invalid_runs(
    model: str, cycle_hour: int, lead_time_hours: int, member: int | None
) -> None:
    connector = NOAAConnector(conn_settings=_settings())
    with pytest.raises(InvalidRunError):
        connector.build_url(model, CYCLE, cycle_hour, lead_time_hours, member)


# ==============================================================================
# AWS S3 Selective Range Download Tests
# ==============================================================================


@respx.mock
async def test_download_selective_gfs_aws_success(tmp_path: Path) -> None:
    """GFS selective download against AWS S3 fetches .idx, resolves ranges, and writes combined artifact."""
    chunk1 = _make_mock_grib2(b"temperature-data")
    chunk2 = _make_mock_grib2(b"precipitation-data")
    len1 = len(chunk1)
    len2 = len(chunk2)

    idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len1}:d=2026072100:PRATE:surface:6 hour fcst:\n"
        f"3:{len1 + len2}:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )

    total_bytes = len1 + len2 + 5000
    respx.get(AWS_GFS_006_URL, headers={"Range": f"bytes=0-{len1 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk1,
            headers={"Content-Range": f"bytes 0-{len1 - 1}/{total_bytes}"},
        )
    )
    respx.get(AWS_GFS_006_URL, headers={"Range": f"bytes={len1}-{len1 + len2 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk2,
            headers={"Content-Range": f"bytes {len1}-{len1 + len2 - 1}/{total_bytes}"},
        )
    )

    dest = tmp_path / "selective_gfs.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == chunk1 + chunk2


@respx.mock
async def test_download_selective_gefs_member_aws_success(tmp_path: Path) -> None:
    """GEFS selective download against AWS S3 fetches TMP for the requested member only."""
    chunk = _make_mock_grib2(b"gefs-member-17-tmp")
    len1 = len(chunk)
    idx_text = (
        f"1:0:d=2026072106:VIS:surface:240 hour fcst:ENS=+17\n"
        f"2:500:d=2026072106:TMP:2 m above ground:240 hour fcst:ENS=+17\n"
        f"3:{500 + len1}:d=2026072106:APCP:surface:0-240 hour acc fcst:ENS=+17\n"
    )
    respx.get(f"{AWS_GEFS_240_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(AWS_GEFS_240_URL, headers={"Range": f"bytes=500-{500 + len1 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk,
            headers={"Content-Range": f"bytes 500-{500 + len1 - 1}/50000"},
        )
    )

    dest = tmp_path / "selective_gefs.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gefs", CYCLE, 6, 240, dest, member=17)

    assert out == dest
    assert dest.read_bytes() == chunk


@respx.mock
async def test_download_gefs_long_lead_025deg_exists_success(tmp_path: Path) -> None:
    """Test future hypothetical 0.25° GEFS availability for extended leads (>240h, e.g. f252):

    Verifies that if NOAA extends 0.25° (pgrb2sp25) publishing upstream in the future,
    the connector accepts and ingests it normally without an artificial hardcoded cutoff.
    """
    chunk = _make_mock_grib2(b"future-gefs-f252-0p25-tmp")
    len1 = len(chunk)
    idx_text = (
        f"1:0:d=2026072106:VIS:surface:252 hour fcst:ENS=+17\n"
        f"2:500:d=2026072106:TMP:2 m above ground:252 hour fcst:ENS=+17\n"
        f"3:{500 + len1}:d=2026072106:APCP:surface:246-252 hour acc fcst:ENS=+17\n"
    )
    # AWS has the hypothetical 0.25° object (pgrb2sp25, not 0.50° pgrb2ap5)
    respx.get(f"{AWS_GEFS_252_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(AWS_GEFS_252_URL, headers={"Range": f"bytes=500-{500 + len1 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk,
            headers={"Content-Range": f"bytes 500-{500 + len1 - 1}/50000"},
        )
    )

    dest = tmp_path / "future_gefs_252.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gefs", CYCLE, 6, 252, dest, member=17)

    assert out == dest
    assert dest.read_bytes() == chunk


@respx.mock
async def test_download_gefs_long_lead_025deg_aws_404_falls_back_to_nomads_025deg(
    tmp_path: Path,
) -> None:
    """Test fallback for hypothetical 0.25° GEFS extended leads:

    When 0.25° GEFS object for f252 is 404 on AWS, fallback queries the same 0.25°
    product on NOMADS (pgrb2sp25), never silently switching to 0.50°.
    """
    # AWS 0.25° returns 404
    respx.get(f"{AWS_GEFS_252_URL}.idx").mock(return_value=httpx.Response(404))
    respx.get(AWS_GEFS_252_URL).mock(return_value=httpx.Response(404))

    # NOMADS 0.25° (pgrb2sp25) has the file
    chunk = _make_mock_grib2(b"nomads-gefs-f252-0p25-tmp")
    len1 = len(chunk)
    nomads_idx_text = (
        f"1:0:d=2026072106:VIS:surface:252 hour fcst:ENS=+17\n"
        f"2:500:d=2026072106:TMP:2 m above ground:252 hour fcst:ENS=+17\n"
        f"3:{500 + len1}:d=2026072106:APCP:surface:246-252 hour acc fcst:ENS=+17\n"
    )
    respx.get(f"{NOMADS_GEFS_252_URL}.idx").mock(
        return_value=httpx.Response(200, text=nomads_idx_text)
    )
    respx.get(NOMADS_GEFS_252_URL, headers={"Range": f"bytes=500-{500 + len1 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk,
            headers={"Content-Range": f"bytes 500-{500 + len1 - 1}/50000"},
        )
    )

    dest = tmp_path / "nomads_gefs_252.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gefs", CYCLE, 6, 252, dest, member=17)

    assert out == dest
    assert dest.read_bytes() == chunk


@respx.mock
async def test_download_gefs_long_lead_both_providers_404_fails_cleanly(
    tmp_path: Path,
) -> None:
    """Test current real NOAA availability for extended leads (>240h):

    Since NOAA currently only publishes 0.50° (pgrb2ap5/pgrb2bp5) upstream for >240h
    and does not publish 0.25° pgrb2sp25, requesting f252 returns 404 on both AWS
    and NOMADS and fails cleanly with DownloadFailedError without corrupting the store
    or silently changing resolution.
    """
    # AWS 0.25° returns 404
    respx.get(f"{AWS_GEFS_252_URL}.idx").mock(return_value=httpx.Response(404))
    respx.get(AWS_GEFS_252_URL).mock(return_value=httpx.Response(404))

    # NOMADS 0.25° also returns 404
    respx.get(f"{NOMADS_GEFS_252_URL}.idx").mock(return_value=httpx.Response(404))
    respx.get(NOMADS_GEFS_252_URL).mock(return_value=httpx.Response(404))

    dest = tmp_path / "gefs_252_unavailable.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        with pytest.raises(DownloadFailedError, match="not found"):
            await connector.download("gefs", CYCLE, 6, 252, dest, member=17)

    assert not dest.exists()


# ==============================================================================
# Fallback Hierarchy: AWS Selective -> AWS Full Fallback
# ==============================================================================


@respx.mock
async def test_download_selective_aws_idx_404_falls_back_to_aws_full(tmp_path: Path) -> None:
    """Missing .idx on AWS triggers fallback to full download FROM AWS first."""
    full_content = _make_mock_grib2(b"full-aws-grib-content")
    idx_route = respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    aws_full_route = respx.get(AWS_GFS_006_URL).mock(
        return_value=httpx.Response(200, content=full_content)
    )
    nomads_route = respx.get(NOMADS_GFS_006_URL).mock(return_value=httpx.Response(200))

    dest = tmp_path / "fallback_aws_full.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content
    assert idx_route.call_count == 1
    assert aws_full_route.call_count == 1
    assert nomads_route.call_count == 0


@respx.mock
async def test_download_selective_aws_framing_error_falls_back_to_aws_full(
    tmp_path: Path,
) -> None:
    """Corrupted selective range on AWS falls back to full download FROM AWS first."""
    bad_chunk = b"GRIB\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1aBAD_PAYLOAD_NO_7777"
    full_content = _make_mock_grib2(b"full-aws-file-after-framing-error")

    idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len(bad_chunk)}:d=2026072100:PRATE:surface:6 hour fcst:\n"
    )
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(AWS_GFS_006_URL, headers={"Range": f"bytes=0-{len(bad_chunk) - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=bad_chunk,
            headers={"Content-Range": f"bytes 0-{len(bad_chunk) - 1}/50000"},
        )
    )
    aws_full_route = respx.get(AWS_GFS_006_URL).mock(
        return_value=httpx.Response(200, content=full_content)
    )
    nomads_route = respx.get(NOMADS_GFS_006_URL).mock(return_value=httpx.Response(200))

    dest = tmp_path / "fallback_framing.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content
    assert aws_full_route.call_count == 1
    assert nomads_route.call_count == 0


@respx.mock
async def test_download_selective_aws_range_ignored_200_streams_directly(
    tmp_path: Path,
) -> None:
    """When AWS Range request returns 200 OK (Range ignored), streams full file directly from AWS."""
    full_content = _make_mock_grib2(b"full-content-from-range-ignored-200")
    idx_text = "1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "stream_200.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content


# ==============================================================================
# Fallback Hierarchy: AWS Provider Failure -> NOMADS Fallback
# ==============================================================================


@respx.mock
async def test_download_aws_404_falls_back_to_nomads(tmp_path: Path) -> None:
    """When AWS object returns 404 (e.g. publication lag), falls back to NOMADS selective download."""
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    aws_full_route = respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(404))

    chunk1 = _make_mock_grib2(b"nomads-temperature-data")
    chunk2 = _make_mock_grib2(b"nomads-precipitation-data")
    len1 = len(chunk1)
    len2 = len(chunk2)

    nomads_idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len1}:d=2026072100:PRATE:surface:6 hour fcst:\n"
        f"3:{len1 + len2}:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{NOMADS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=nomads_idx_text)
    )
    respx.get(NOMADS_GFS_006_URL, headers={"Range": f"bytes=0-{len1 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk1,
            headers={"Content-Range": f"bytes 0-{len1 - 1}/50000"},
        )
    )
    respx.get(NOMADS_GFS_006_URL, headers={"Range": f"bytes={len1}-{len1 + len2 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk2,
            headers={"Content-Range": f"bytes {len1}-{len1 + len2 - 1}/50000"},
        )
    )

    dest = tmp_path / "nomads_fallback.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == chunk1 + chunk2
    assert aws_full_route.call_count == 1


@respx.mock
async def test_download_aws_404_without_nomads_fallback_fails_fast(tmp_path: Path) -> None:
    """When ENABLE_NOMADS_FALLBACK is False and AWS returns 404, fails immediately without touching NOMADS."""
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(404))
    nomads_route = respx.get(NOMADS_GFS_006_URL).mock(return_value=httpx.Response(200))

    dest = tmp_path / "no_fallback.grib2"
    async with NOAAConnector(
        conn_settings=_settings(ENABLE_NOMADS_FALLBACK=False)
    ) as connector:
        with pytest.raises(DownloadFailedError, match="not found"):
            await connector.download("gfs", CYCLE, 0, 6, dest)

    assert nomads_route.call_count == 0
    assert not dest.exists()


@respx.mock
async def test_download_aws_5xx_exhausts_retries_before_nomads_fallback(
    tmp_path: Path,
) -> None:
    """AWS 5xx errors are retried up to DOWNLOAD_RETRIES before falling back to NOMADS."""
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    aws_route = respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(500))

    nomads_content = _make_mock_grib2(b"nomads-content-after-aws-5xx")
    respx.get(f"{NOMADS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    nomads_route = respx.get(NOMADS_GFS_006_URL).mock(
        return_value=httpx.Response(200, content=nomads_content)
    )

    dest = tmp_path / "nomads_after_aws_5xx.grib2"
    async with NOAAConnector(
        conn_settings=_settings(DOWNLOAD_RETRIES=2)
    ) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == nomads_content
    assert aws_route.call_count == 3
    assert nomads_route.call_count == 1


@respx.mock
async def test_download_both_aws_and_nomads_unavailable_fails_cleanly(
    tmp_path: Path,
) -> None:
    """When both AWS and NOMADS fail, cleanly raises DownloadFailedError."""
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(404))

    respx.get(f"{NOMADS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    respx.get(NOMADS_GFS_006_URL).mock(return_value=httpx.Response(404))

    dest = tmp_path / "both_fail.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        with pytest.raises(DownloadFailedError, match="not found"):
            await connector.download("gfs", CYCLE, 0, 6, dest)

    assert not dest.exists()


# ==============================================================================
# Direct NOMADS Mode Tests
# ==============================================================================


@respx.mock
async def test_download_explicit_nomads_mode(tmp_path: Path) -> None:
    """When NOAA_DOWNLOAD_SOURCE='nomads', downloads directly from NOMADS without hitting AWS."""
    content = _make_mock_grib2(b"direct-nomads-grib")
    respx.get(f"{NOMADS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    nomads_route = respx.get(NOMADS_GFS_006_URL).mock(
        return_value=httpx.Response(200, content=content)
    )
    aws_route = respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(200))

    dest = tmp_path / "direct_nomads.grib2"
    async with NOAAConnector(
        conn_settings=_settings(NOAA_DOWNLOAD_SOURCE="nomads")
    ) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == content
    assert nomads_route.call_count == 1
    assert aws_route.call_count == 0


@respx.mock
async def test_download_idx_fetches_index(tmp_path: Path) -> None:
    """The .idx index file is fetched from the S3 URL + .idx suffix."""
    content = b"1:0:d=2026072100:2mTMP:surface:anl:"
    respx.get(f"{AWS_GEFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, content=content)
    )
    async with NOAAConnector(conn_settings=_settings()) as connector:
        destination = await connector.download_idx(
            "gefs", CYCLE, 0, 6, tmp_path / "gep01.f006.idx", member=1
        )
    assert destination.read_bytes() == content


@respx.mock
async def test_download_selective_later_range_returns_200_fallback(
    tmp_path: Path,
) -> None:
    """When a subsequent Range request returns 200 OK after record 1 was written,
    the partial selective artifact is discarded and the full file is streamed safely
    without producing a mixed/corrupted artifact.
    """
    chunk1 = _make_mock_grib2(b"chunk1-temperature-data")
    full_content = _make_mock_grib2(b"complete-upstream-grib2-file-contents")
    len1 = len(chunk1)

    idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len1}:d=2026072100:PRATE:surface:6 hour fcst:\n"
        f"3:{len1 + 500}:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )

    respx.get(AWS_GFS_006_URL, headers={"Range": f"bytes=0-{len1 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk1,
            headers={"Content-Range": f"bytes 0-{len1 - 1}/50000"},
        )
    )
    respx.get(AWS_GFS_006_URL, headers={"Range": f"bytes={len1}-{len1 + 499}"}).mock(
        return_value=httpx.Response(200, content=full_content)
    )

    dest = tmp_path / "gfs_later_200.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    result_bytes = dest.read_bytes()
    assert result_bytes == full_content
    assert result_bytes != chunk1 + full_content

    temp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(temp_files) == 0


@respx.mock
async def test_download_disabled_selective_uses_full_download(tmp_path: Path) -> None:
    """When ENABLE_SELECTIVE_DOWNLOAD is False, downloads full file directly without .idx."""
    full_content = _make_mock_grib2(b"full-direct-download")
    idx_route = respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    data_route = respx.get(AWS_GFS_006_URL).mock(
        return_value=httpx.Response(200, content=full_content)
    )

    dest = tmp_path / "full_direct.grib2"
    async with NOAAConnector(
        conn_settings=_settings(ENABLE_SELECTIVE_DOWNLOAD=False)
    ) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content
    assert idx_route.call_count == 0
    assert data_route.call_count == 1


@respx.mock
async def test_download_selective_malformed_idx_falls_back_to_full(
    tmp_path: Path,
) -> None:
    """Malformed .idx triggers fallback to full download."""
    full_content = _make_mock_grib2(b"full-download-fallback-malformed")
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text="not:a:valid:idx:file")
    )
    respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_malformed_idx.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content


@respx.mock
async def test_download_selective_missing_required_variable_falls_back(
    tmp_path: Path,
) -> None:
    """When required variable is missing in .idx for GFS, falls back to full download."""
    full_content = _make_mock_grib2(b"full-download-fallback-missing-var")
    idx_text = (
        "1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        "2:500:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_missing_var.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content


@respx.mock
async def test_download_selective_section0_length_mismatch_falls_back(
    tmp_path: Path,
) -> None:
    """GRIB Section 0 declared length mismatch triggers fallback to full download."""
    full_content = _make_mock_grib2(b"full-download-fallback-length-mismatch")
    fake_len = 9999
    bad_chunk = b"GRIB\x00\x00\x00\x02" + fake_len.to_bytes(8, "big") + b"data7777"
    idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len(bad_chunk)}:d=2026072100:PRATE:surface:6 hour fcst:\n"
        f"3:{len(bad_chunk) + 100}:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(AWS_GFS_006_URL, headers={"Range": f"bytes=0-{len(bad_chunk) - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=bad_chunk,
            headers={"Content-Range": f"bytes 0-{len(bad_chunk) - 1}/5000"},
        )
    )
    respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_length_mismatch.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content


@respx.mock
async def test_download_selective_content_range_mismatch_falls_back(
    tmp_path: Path,
) -> None:
    """Mismatched Content-Range start offset triggers fallback to full download."""
    full_content = _make_mock_grib2(b"full-download-fallback-cr-mismatch")
    chunk = _make_mock_grib2(b"valid-chunk")
    idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len(chunk)}:d=2026072100:PRATE:surface:6 hour fcst:\n"
        f"3:{len(chunk) + 100}:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(AWS_GFS_006_URL, headers={"Range": f"bytes=0-{len(chunk) - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk,
            headers={"Content-Range": f"bytes 100-{100 + len(chunk) - 1}/5000"},
        )
    )
    respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_cr_mismatch.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content


@respx.mock
async def test_download_4xx_forbidden_fails_without_retry(tmp_path: Path) -> None:
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(403))
    route = respx.get(AWS_GFS_006_URL).mock(return_value=httpx.Response(403))
    with pytest.raises(DownloadFailedError, match="HTTP 403"):
        async with NOAAConnector(conn_settings=_settings(ENABLE_NOMADS_FALLBACK=False)) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "forbidden.grib2")
    assert route.call_count == 1
    assert not (tmp_path / "forbidden.grib2").exists()


@respx.mock
async def test_download_3xx_redirect_is_rejected(tmp_path: Path) -> None:
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    route = respx.get(AWS_GFS_006_URL).mock(
        return_value=httpx.Response(
            302, headers={"Location": "https://example.test/redirected.grib2"}
        )
    )
    with pytest.raises(DownloadFailedError, match="HTTP 302"):
        async with NOAAConnector(conn_settings=_settings(ENABLE_NOMADS_FALLBACK=False)) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "redirected.grib2")
    assert route.call_count == 1
    assert not (tmp_path / "redirected.grib2").exists()


@respx.mock
async def test_download_retries_transient_5xx_then_succeeds(tmp_path: Path) -> None:
    content = _make_mock_grib2(b"retried-grib")
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    route = respx.get(AWS_GFS_006_URL).mock(
        side_effect=[httpx.Response(500), httpx.Response(200, content=content)]
    )
    async with NOAAConnector(conn_settings=_settings()) as connector:
        destination = await connector.download(
            "gfs", CYCLE, 0, 6, tmp_path / "retried.grib2"
        )
    assert route.call_count == 2
    assert destination.read_bytes() == content


@respx.mock
async def test_download_stream_failure_deletes_partial_file(tmp_path: Path) -> None:
    _BrokenBody.entered = 0
    respx.get(f"{AWS_GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    route = respx.get(AWS_GFS_006_URL).mock(
        return_value=httpx.Response(200, stream=_BrokenBody())
    )
    with pytest.raises(UpstreamUnavailableError, match="unreachable"):
        async with NOAAConnector(
            conn_settings=_settings(DOWNLOAD_RETRIES=2, ENABLE_NOMADS_FALLBACK=False)
        ) as connector:
            await connector.download("gfs", CYCLE, 0, 6, tmp_path / "partial.grib2")
    assert route.call_count == 3
    assert _BrokenBody.entered == 3
    assert not (tmp_path / "partial.grib2").exists()
