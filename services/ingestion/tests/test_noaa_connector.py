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
# GEFS per-member perturbation files (gep01..gep30), the actual NOMADS layout.
# 0.25 deg short/medium range: pgrb2sp25/. Beyond 240 h: pgrb2bp5/ 0.5 deg.
GEFS_006_URL = (
    f"{BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/00/atmos/pgrb2sp25/"
    "gep01.t00z.pgrb2s.0p25.f006"
)
GEFS_240_URL = (
    f"{BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/06/atmos/pgrb2sp25/"
    "gep17.t06z.pgrb2s.0p25.f240"
)
GEFS_252_URL = (
    f"{BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/06/atmos/pgrb2bp5/"
    "gep17.t06z.pgrb2s.0p50.f252"
)
GEFS_384_URL = (
    f"{BASE}/pub/data/nccf/com/gens/prod/gefs.20260721/12/atmos/pgrb2bp5/"
    "gep30.t12z.pgrb2s.0p50.f384"
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
    ("model", "cycle_hour", "lead_time_hours", "member", "expected_url"),
    [
        ("gfs", 0, 6, None, GFS_006_URL),
        ("gfs", 12, 0, None, GFS_000_URL),
        ("gfs", 18, 384, None, GFS_384_URL),
        ("gefs", 0, 6, 1, GEFS_006_URL),
        ("gefs", 6, 240, 17, GEFS_240_URL),
        ("gefs", 6, 252, 17, GEFS_252_URL),
        ("gefs", 12, 384, 30, GEFS_384_URL),
    ],
)
def test_build_url_is_deterministic(
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


@pytest.mark.parametrize(
    ("model", "cycle_hour", "lead_time_hours", "member"),
    [
        ("ecmwf", 0, 6, None),
        ("gfs", 3, 6, None),
        ("gfs", 0, -1, None),
        ("gfs", 0, 385, None),
        # GEFS requires a member identity and rejects out-of-range members.
        ("gefs", 0, 6, None),
        ("gefs", 0, 6, 0),
        ("gefs", 0, 6, 31),
    ],
)
def test_build_url_rejects_invalid_runs(
    model: str, cycle_hour: int, lead_time_hours: int, member: int | None
) -> None:
    connector = NOAAConnector(conn_settings=_settings())
    with pytest.raises(InvalidRunError):
        connector.build_url(model, CYCLE, cycle_hour, lead_time_hours, member)


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
            "gefs", CYCLE, 6, 252, tmp_path / "gefs.f252.grib2", member=17
        )
    assert destination.read_bytes() == content


@respx.mock
async def test_download_gefs_member_file(tmp_path: Path) -> None:
    """A per-member GEFS file downloads with its real gepNN identity."""
    content = b"gep17-grib"
    respx.get(GEFS_240_URL).mock(return_value=httpx.Response(200, content=content))
    async with NOAAConnector(conn_settings=_settings()) as connector:
        destination = await connector.download(
            "gefs", CYCLE, 6, 240, tmp_path / "gep17.f240.grib2", member=17
        )
    assert destination.read_bytes() == content


@respx.mock
async def test_download_gefs_without_member_rejected(tmp_path: Path) -> None:
    """A GEFS download without a member identity is refused (no combined file)."""
    async with NOAAConnector(conn_settings=_settings()) as connector:
        with pytest.raises(InvalidRunError, match="member"):
            await connector.download(
                "gefs", CYCLE, 0, 6, tmp_path / "gefs.f006.grib2"
            )


@respx.mock
async def test_download_idx_fetches_index(tmp_path: Path) -> None:
    """The .idx index file is fetched from the .grib2 URL + .idx suffix."""
    content = b"1:0:d=2026072100:2mTMP:surface:anl:"
    respx.get(f"{GEFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, content=content)
    )
    async with NOAAConnector(conn_settings=_settings()) as connector:
        destination = await connector.download_idx(
            "gefs", CYCLE, 0, 6, tmp_path / "gep01.f006.idx", member=1
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


def _make_mock_grib2(data: bytes = b"payload") -> bytes:
    """Construct a mock GRIB2 message with a valid Section 0 declared length."""
    total = 16 + len(data) + 4
    return b"GRIB\x00\x00\x00\x02" + total.to_bytes(8, "big") + data + b"7777"


@respx.mock
async def test_download_selective_gfs_success(tmp_path: Path) -> None:
    """GFS selective download fetches .idx, resolves ranges, and writes combined artifact."""
    chunk1 = _make_mock_grib2(b"temperature-data")
    chunk2 = _make_mock_grib2(b"precipitation-data")
    len1 = len(chunk1)
    len2 = len(chunk2)

    # Offset 0..len1-1 for record 1 (TMP)
    # Offset len1..len1+len2-1 for record 2 (PRATE)
    # Offset len1+len2 for record 3 (SPFH)
    idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len1}:d=2026072100:PRATE:surface:6 hour fcst:\n"
        f"3:{len1 + len2}:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )

    total_bytes = len1 + len2 + 5000
    respx.get(GFS_006_URL, headers={"Range": f"bytes=0-{len1 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk1,
            headers={"Content-Range": f"bytes 0-{len1 - 1}/{total_bytes}"},
        )
    )
    respx.get(GFS_006_URL, headers={"Range": f"bytes={len1}-{len1 + len2 - 1}"}).mock(
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
async def test_download_selective_gefs_member_success(tmp_path: Path) -> None:
    """GEFS selective download fetches TMP for the requested member only."""
    chunk = _make_mock_grib2(b"gefs-member-17-tmp")
    len1 = len(chunk)
    idx_text = (
        f"1:0:d=2026072106:VIS:surface:240 hour fcst:ENS=+17\n"
        f"2:500:d=2026072106:TMP:2 m above ground:240 hour fcst:ENS=+17\n"
        f"3:{500 + len1}:d=2026072106:APCP:surface:0-240 hour acc fcst:ENS=+17\n"
    )
    respx.get(f"{GEFS_240_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(GEFS_240_URL, headers={"Range": f"bytes=500-{500 + len1 - 1}"}).mock(
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
async def test_download_selective_server_returns_200_fallback(tmp_path: Path) -> None:
    """When Range request returns 200 OK (Range ignored), streams full file directly."""
    full_content = _make_mock_grib2(b"full-grib-content")
    idx_text = (
        "1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        "2:500:d=2026072100:PRATE:surface:6 hour fcst:\n"
        "3:1000:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    # Server returns 200 OK instead of 206
    respx.get(GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_200.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content


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
    respx.get(f"{GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )

    # Range #1 for TMP succeeds with 206
    respx.get(GFS_006_URL, headers={"Range": f"bytes=0-{len1 - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk1,
            headers={"Content-Range": f"bytes 0-{len1 - 1}/50000"},
        )
    )

    # Range #2 for PRATE returns 200 OK with full file content
    respx.get(GFS_006_URL, headers={"Range": f"bytes={len1}-{len1 + 499}"}).mock(
        return_value=httpx.Response(200, content=full_content)
    )

    dest = tmp_path / "gfs_later_200.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    # Assert final artifact is EXACTLY the full content, NOT chunk1 + full_content
    result_bytes = dest.read_bytes()
    assert result_bytes == full_content
    assert result_bytes != chunk1 + full_content
    assert not result_bytes.startswith(chunk1 + full_content)

    # Verify no leftover temporary files in directory
    temp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(temp_files) == 0


@respx.mock
async def test_download_disabled_selective_uses_full_download(tmp_path: Path) -> None:
    """When ENABLE_SELECTIVE_DOWNLOAD is False, downloads full file directly without .idx."""
    full_content = _make_mock_grib2(b"full-direct-download")
    idx_route = respx.get(f"{GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    data_route = respx.get(GFS_006_URL).mock(
        return_value=httpx.Response(200, content=full_content)
    )

    dest = tmp_path / "full_direct.grib2"
    async with NOAAConnector(
        conn_settings=_settings(ENABLE_SELECTIVE_DOWNLOAD=False)
    ) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content
    # .idx route was never even requested
    assert idx_route.call_count == 0
    assert data_route.call_count == 1


@respx.mock
async def test_download_selective_idx_404_falls_back_to_full(tmp_path: Path) -> None:
    """Missing .idx file triggers fallback to full download."""
    full_content = _make_mock_grib2(b"full-download-fallback")
    respx.get(f"{GFS_006_URL}.idx").mock(return_value=httpx.Response(404))
    respx.get(GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_idx_404.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content


@respx.mock
async def test_download_selective_malformed_idx_falls_back_to_full(
    tmp_path: Path,
) -> None:
    """Malformed .idx triggers fallback to full download."""
    full_content = _make_mock_grib2(b"full-download-fallback-malformed")
    respx.get(f"{GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text="not:a:valid:idx:file")
    )
    respx.get(GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

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
    # GFS index has TMP but lacks PRATE
    idx_text = (
        "1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        "2:500:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_missing_var.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content


@respx.mock
async def test_download_selective_binary_framing_corrupted_falls_back(
    tmp_path: Path,
) -> None:
    """Corrupted binary framing (missing 7777) triggers fallback to full download."""
    full_content = _make_mock_grib2(b"full-download-fallback-corrupt-framing")
    # Bad chunk lacks '7777'
    bad_chunk = b"GRIB\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1aBAD_PAYLOAD_NO_7777"
    idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len(bad_chunk)}:d=2026072100:PRATE:surface:6 hour fcst:\n"
        f"3:{len(bad_chunk) + 100}:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(GFS_006_URL, headers={"Range": f"bytes=0-{len(bad_chunk) - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=bad_chunk,
            headers={"Content-Range": f"bytes 0-{len(bad_chunk) - 1}/5000"},
        )
    )
    respx.get(GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_corrupted_framing.grib2"
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
    # Declares 9999 bytes length but only has 25 bytes
    fake_len = 9999
    bad_chunk = b"GRIB\x00\x00\x00\x02" + fake_len.to_bytes(8, "big") + b"data7777"
    idx_text = (
        f"1:0:d=2026072100:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{len(bad_chunk)}:d=2026072100:PRATE:surface:6 hour fcst:\n"
        f"3:{len(bad_chunk) + 100}:d=2026072100:SPFH:2 m above ground:6 hour fcst:\n"
    )
    respx.get(f"{GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    respx.get(GFS_006_URL, headers={"Range": f"bytes=0-{len(bad_chunk) - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=bad_chunk,
            headers={"Content-Range": f"bytes 0-{len(bad_chunk) - 1}/5000"},
        )
    )
    respx.get(GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

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
    respx.get(f"{GFS_006_URL}.idx").mock(
        return_value=httpx.Response(200, text=idx_text)
    )
    # Returns Content-Range for byte 100 instead of 0
    respx.get(GFS_006_URL, headers={"Range": f"bytes=0-{len(chunk) - 1}"}).mock(
        return_value=httpx.Response(
            206,
            content=chunk,
            headers={"Content-Range": f"bytes 100-{100 + len(chunk) - 1}/5000"},
        )
    )
    respx.get(GFS_006_URL).mock(return_value=httpx.Response(200, content=full_content))

    dest = tmp_path / "fallback_cr_mismatch.grib2"
    async with NOAAConnector(conn_settings=_settings()) as connector:
        out = await connector.download("gfs", CYCLE, 0, 6, dest)

    assert out == dest
    assert dest.read_bytes() == full_content

