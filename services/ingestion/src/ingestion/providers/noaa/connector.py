"""NOAA NOMADS connector for operational GFS and GEFS products."""

import asyncio
from datetime import date
import logging
import os
from pathlib import Path
import re
import time
from types import TracebackType
from typing import Self
import uuid

import httpx

from ingestion.core.base import (
    BaseConnector,
    DownloadFailedError,
    IngestionError,
    InvalidRunError,
    UpstreamUnavailableError,
)
from ingestion.core.config import IngestionSettings, settings
from ingestion.providers.noaa.idx_parser import (
    DEFAULT_SELECTION_VARIABLES,
    IdxParseError,
    merge_adjacent_records,
    parse_idx,
    select_records,
)

logger = logging.getLogger(__name__)

_SUPPORTED_MODELS = frozenset({"gfs", "gefs"})
_CYCLE_HOURS = frozenset({0, 6, 12, 18})
_MAX_LEAD_TIME_HOURS = 384
#: GEFS perturbation member range (gep01..gep30). Member identity is the real
#: upstream perturbation number, never a positional completion index.
_GEFS_MEMBER_MIN = 1
_GEFS_MEMBER_MAX = 30

_CONTENT_RANGE_PATTERN = re.compile(
    r"^bytes\s+(\d+)-(\d+)/(?:\d+|\*)$", re.IGNORECASE
)


class SelectiveFallbackError(IngestionError):
    """Raised when a selective byte-range download fails and requires full-file fallback.

    Attributes:
        reason: Stable telemetry reason code explaining the fallback trigger.
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or f"Selective download fallback triggered: {reason}")
        self.reason = reason


class NOAAConnector(BaseConnector):
    """Connector for NOAA operational GFS and GEFS forecast products.

    Supports the deterministic GFS model and the GEFS ensemble perturbation
    members (``gep01``..``gep30``) for the four daily cycles (00Z, 06Z, 12Z,
    18Z) on the canonical 0.25° grid across lead times up to 384 hours.
    Download URLs and S3 keys are built deterministically from the official
    directory layout:

    * GFS 0.25°: ``gfs.YYYYMMDD/CC/atmos/gfs.tCCz.pgrb2.0p25.fXXX``
    * GEFS 0.25°: ``gefs.YYYYMMDD/CC/atmos/pgrb2sp25/gepNN.tCCz.pgrb2s.0p25.fXXX``

    Only the 30 perturbation members are ingested; the control (``gec00``),
    ensemble-mean (``geavg``), and spread (``gespr``) files are out of scope.

    Downloads use sequential single-range HTTP requests based on ``.idx``
    offsets to selectively fetch only the required GRIB messages, saving >97%
    bandwidth. If the selective path encounters index anomalies, missing required
    records, or range rejection, it cleanly falls back to the full-file download path.
    """

    def __init__(self, conn_settings: IngestionSettings | None = None) -> None:
        """Create a connector backed by an isolated HTTP client.

        Args:
            conn_settings: Ingestion settings; defaults to the module-level
                settings instance.
        """
        self._settings = conn_settings or settings
        max_connections = int(getattr(self._settings, "HTTP_MAX_CONNECTIONS", 100))
        max_keepalive = int(getattr(self._settings, "HTTP_MAX_KEEPALIVE_CONNECTIONS", 50))
        keepalive_expiry = float(getattr(self._settings, "HTTP_KEEPALIVE_EXPIRY_SECONDS", 30.0))
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=keepalive_expiry,
        )
        self._client = httpx.AsyncClient(
            timeout=self._settings.REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": self._settings.NOAA_USER_AGENT},
            limits=limits,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def build_s3_key(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        member: int | None = None,
    ) -> str:
        """Return the deterministic AWS S3 object key for a GRIB2 file.

        Key layouts on AWS Open Data (us-east-1):
        * GFS 0.25°: ``gfs.YYYYMMDD/CC/atmos/gfs.tCCz.pgrb2.0p25.fXXX``
        * GEFS 0.25°: ``gefs.YYYYMMDD/CC/atmos/pgrb2sp25/gepNN.tCCz.pgrb2s.0p25.fXXX``

        Args:
            model: Model identifier, either ``gfs`` or ``gefs``.
            cycle_date: UTC date of the model run.
            cycle_hour: UTC cycle hour; one of 0, 6, 12, 18.
            lead_time_hours: Forecast lead time in hours (0-384).
            member: GEFS perturbation member (1..30). Required for GEFS;
                ignored for GFS.

        Returns:
            Relative S3 object key.

        Raises:
            InvalidRunError: If the model, cycle hour, lead time, or member is
                unsupported.
        """
        self._validate_run(model, cycle_hour, lead_time_hours)
        if model == "gefs":
            if member is None:
                raise InvalidRunError(
                    "A GEFS member identity (1..30) is required to build a "
                    "per-member download URL; the combined "
                    "'gefs.tCCz.pgrb2a.0p25' product no longer exists."
                )
            if not _GEFS_MEMBER_MIN <= member <= _GEFS_MEMBER_MAX:
                raise InvalidRunError(
                    f"Invalid GEFS member: {member}; expected "
                    f"{_GEFS_MEMBER_MIN}-{_GEFS_MEMBER_MAX} (gepNN)."
                )
        date_str = cycle_date.strftime("%Y%m%d")
        hour_str = f"{cycle_hour:02d}"
        lead_str = f"{lead_time_hours:03d}"
        if model == "gfs":
            return f"gfs.{date_str}/{hour_str}/atmos/gfs.t{hour_str}z.pgrb2.0p25.f{lead_str}"
        return (
            f"gefs.{date_str}/{hour_str}/atmos/pgrb2sp25/"
            f"gep{member:02d}.t{hour_str}z.pgrb2s.0p25.f{lead_str}"
        )

    def build_s3_url(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        member: int | None = None,
    ) -> str:
        """Return the deterministic public HTTPS URL for an AWS S3 GRIB2 object."""
        key = self.build_s3_key(model, cycle_date, cycle_hour, lead_time_hours, member)
        base = (
            self._settings.AWS_GFS_BASE_URL
            if model == "gfs"
            else self._settings.AWS_GEFS_BASE_URL
        )
        return f"{base}/{key}"

    def build_nomads_url(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        member: int | None = None,
    ) -> str:
        """Return the deterministic NOMADS download URL for a GRIB2 file."""
        self._validate_run(model, cycle_hour, lead_time_hours)
        if model == "gefs":
            if member is None:
                raise InvalidRunError(
                    "A GEFS member identity (1..30) is required to build a "
                    "per-member download URL; the combined "
                    "'gefs.tCCz.pgrb2a.0p25' product no longer exists on NOMADS."
                )
            if not _GEFS_MEMBER_MIN <= member <= _GEFS_MEMBER_MAX:
                raise InvalidRunError(
                    f"Invalid GEFS member: {member}; expected "
                    f"{_GEFS_MEMBER_MIN}-{_GEFS_MEMBER_MAX} (gepNN)."
                )
        date_str = cycle_date.strftime("%Y%m%d")
        hour_str = f"{cycle_hour:02d}"
        lead_str = f"{lead_time_hours:03d}"
        if model == "gfs":
            return (
                f"{self._settings.NOMADS_BASE_URL}/pub/data/nccf/com/gfs/prod/"
                f"gfs.{date_str}/{hour_str}/atmos/"
                f"gfs.t{hour_str}z.pgrb2.0p25.f{lead_str}"
            )
        return (
            f"{self._settings.NOMADS_BASE_URL}/pub/data/nccf/com/gens/prod/"
            f"gefs.{date_str}/{hour_str}/atmos/pgrb2sp25/"
            f"gep{member:02d}.t{hour_str}z.pgrb2s.0p25.f{lead_str}"
        )

    def build_url(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        member: int | None = None,
        source: str | None = None,
    ) -> str:
        """Return the deterministic download URL for a GRIB2 file.

        Args:
            model: Model identifier, either ``gfs`` or ``gefs``.
            cycle_date: UTC date of the model run.
            cycle_hour: UTC cycle hour; one of 0, 6, 12, 18.
            lead_time_hours: Forecast lead time in hours (0-384).
            member: GEFS perturbation member (1..30). Required for GEFS;
                ignored for GFS.
            source: Upstream provider identifier (``"aws_s3"`` or ``"nomads"``).
                Defaults to the configured ``NOAA_DOWNLOAD_SOURCE``.

        Returns:
            Absolute URL of the requested GRIB2 file.

        Raises:
            InvalidRunError: If the model, cycle hour, lead time, or member is
                unsupported.
        """
        src = source or getattr(self._settings, "NOAA_DOWNLOAD_SOURCE", "aws_s3")
        if src == "nomads":
            return self.build_nomads_url(
                model, cycle_date, cycle_hour, lead_time_hours, member
            )
        return self.build_s3_url(
            model, cycle_date, cycle_hour, lead_time_hours, member
        )

    async def download(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        destination: Path,
        member: int | None = None,
        variables: tuple[str, ...] = DEFAULT_SELECTION_VARIABLES,
    ) -> Path:
        """Download a GRIB2 file to ``destination`` using selective byte ranges.

        Default primary source is AWS Open Data on S3. If an upstream availability
        or publication failure occurs (e.g. 404 Not Found due to publication lag,
        or 5xx after retries) and NOMADS fallback is enabled, automatically falls
        back to NOMADS.

        Fallback Hierarchy:
        1. AWS S3 selective Range download.
        2. If selective-specific failure occurs (index/framing/Range 200/length),
           fall back to AWS S3 full download first.
        3. Fall back to NOMADS only for provider availability/publication failures
           (404 Not Found after retry, or 5xx/transport exhaustion).
        4. If NOMADS fallback is disabled or NOMADS also fails, raise the error.

        Args:
            model: Model identifier, either ``gfs`` or ``gefs``.
            cycle_date: UTC date of the model run.
            cycle_hour: UTC cycle hour; one of 0, 6, 12, 18.
            lead_time_hours: Forecast lead time in hours (0-384).
            destination: Local path the GRIB2 file is written to; the
                parent directory is created if missing.
            member: GEFS perturbation member (1..30). Required for GEFS;
                ignored for GFS.
            variables: Canonical platform variables to select.

        Returns:
            The path the file was written to.

        Raises:
            InvalidRunError: If the model, cycle hour, lead time, or member is
                unsupported.
            UpstreamUnavailableError: If the upstream provider cannot be
                reached after all retry attempts.
            DownloadFailedError: If the upstream provider returns an error
                status, or keeps returning 5xx after all retry attempts.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        primary_source = getattr(self._settings, "NOAA_DOWNLOAD_SOURCE", "aws_s3")
        primary_url = self.build_url(
            model, cycle_date, cycle_hour, lead_time_hours, member, source=primary_source
        )

        try:
            return await self._download_single_provider(
                url=primary_url,
                model=model,
                cycle_date=cycle_date,
                cycle_hour=cycle_hour,
                lead_time_hours=lead_time_hours,
                destination=destination,
                member=member,
                variables=variables,
            )
        except (DownloadFailedError, UpstreamUnavailableError) as primary_exc:
            if (
                primary_source == "aws_s3"
                and getattr(self._settings, "ENABLE_NOMADS_FALLBACK", True)
            ):
                logger.warning(
                    "download_provider_fallback: model=%s cycle=%sT%02dZ lead=%d member=%s "
                    "primary_url=%s error=%s; attempting NOMADS fallback.",
                    model,
                    cycle_date,
                    cycle_hour,
                    lead_time_hours,
                    member,
                    primary_url,
                    primary_exc,
                )
                nomads_url = self.build_nomads_url(
                    model, cycle_date, cycle_hour, lead_time_hours, member
                )
                return await self._download_single_provider(
                    url=nomads_url,
                    model=model,
                    cycle_date=cycle_date,
                    cycle_hour=cycle_hour,
                    lead_time_hours=lead_time_hours,
                    destination=destination,
                    member=member,
                    variables=variables,
                )
            raise

    async def _download_single_provider(
        self,
        url: str,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        destination: Path,
        member: int | None,
        variables: tuple[str, ...],
    ) -> Path:
        """Download from a single provider URL using selective ranges with full-download fallback."""
        if getattr(self._settings, "ENABLE_SELECTIVE_DOWNLOAD", True):
            try:
                return await self._download_selective_with_retry(
                    url=url,
                    model=model,
                    cycle_date=cycle_date,
                    cycle_hour=cycle_hour,
                    lead_time_hours=lead_time_hours,
                    destination=destination,
                    member=member,
                    variables=variables,
                )
            except SelectiveFallbackError as fallback_exc:
                logger.warning(
                    "download_selective_fallback: url=%s model=%s cycle=%sT%02dZ lead=%d member=%s "
                    "reason=%s; falling back to full download on same provider.",
                    url,
                    model,
                    cycle_date,
                    cycle_hour,
                    lead_time_hours,
                    member,
                    fallback_exc.reason,
                )
            except Exception as exc:
                # Catch any unexpected selective-path error to ensure reliable full-file fallback
                logger.warning(
                    "download_selective_fallback: url=%s model=%s cycle=%sT%02dZ lead=%d member=%s "
                    "reason=unexpected_selective_error error=%s; falling back to full download on same provider.",
                    url,
                    model,
                    cycle_date,
                    cycle_hour,
                    lead_time_hours,
                    member,
                    exc,
                )

        return await self._download_full_with_retry(url, destination)

    async def _download_selective_with_retry(
        self,
        url: str,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        destination: Path,
        member: int | None,
        variables: tuple[str, ...],
    ) -> Path:
        """Attempt selective range download with artifact-transaction retry semantics."""
        attempts = self._settings.DOWNLOAD_RETRIES + 1
        last_error: BaseException | None = None

        for attempt in range(1, attempts + 1):
            temp_dest = destination.with_suffix(f".tmp.{uuid.uuid4().hex}.grib2")
            start_time = time.monotonic()
            try:
                # 1. Fetch fresh .idx (never reuse cached .idx across retries)
                idx_url = f"{url}.idx"
                try:
                    idx_resp = await self._client.get(idx_url)
                except httpx.TransportError as exc:
                    raise exc

                if idx_resp.status_code == httpx.codes.NOT_FOUND:
                    raise SelectiveFallbackError("idx_not_found")
                if idx_resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Upstream returned HTTP {idx_resp.status_code} for .idx: {idx_url}",
                        request=idx_resp.request,
                        response=idx_resp,
                    )
                if not 200 <= idx_resp.status_code < 300:
                    raise SelectiveFallbackError(f"idx_http_{idx_resp.status_code}")

                # 2. Strict parse into IdxRecords
                try:
                    records = parse_idx(idx_resp.text)
                except IdxParseError as exc:
                    raise SelectiveFallbackError(
                        "idx_parse_error", str(exc)
                    ) from exc

                # 3. Product-aware selection
                selection = select_records(
                    model=model,
                    records=records,
                    lead_time_hours=lead_time_hours,
                    variables=variables,
                    member=member,
                )
                if not selection.is_valid:
                    if selection.missing_required:
                        raise SelectiveFallbackError(
                            "required_record_missing_in_idx",
                            f"Missing required records for {selection.missing_required}",
                        )
                    if selection.ambiguous:
                        raise SelectiveFallbackError(
                            "ambiguous_record_selection",
                            f"Ambiguous matching records for {selection.ambiguous}",
                        )

                if not selection.selected_records:
                    raise SelectiveFallbackError("no_records_selected")

                # 4. Adjacent-merged range requests in original source order (Gap=0, 0% extra bytes)
                record_groups = merge_adjacent_records(selection.selected_records, max_gap=0)
                total_downloaded = 0
                fallback_full_bytes: bytes | None = None
                with temp_dest.open("wb") as handle:
                    for group in record_groups:
                        start_offset = group[0].start_offset
                        end_offset = group[-1].end_offset
                        range_header = (
                            f"bytes={start_offset}-{end_offset}"
                            if end_offset is not None
                            else f"bytes={start_offset}-"
                        )
                        async with self._client.stream(
                            "GET", url, headers={"Range": range_header}
                        ) as resp:
                            if resp.status_code == httpx.codes.NOT_FOUND:
                                raise DownloadFailedError(
                                    f"Requested file not found: {url}"
                                )
                            if resp.status_code == 200:
                                # Upstream ignored Range header. Capture full response directly
                                # to destination as an immediate in-stream fallback.
                                logger.warning(
                                    "Server returned 200 OK instead of 206 for Range request; "
                                    "streaming full file directly."
                                )
                                fallback_full_bytes = await resp.aread()
                                break
                            if resp.status_code == 416:
                                raise SelectiveFallbackError(
                                    "range_not_satisfiable_416"
                                )
                            if resp.status_code >= 500:
                                raise httpx.HTTPStatusError(
                                    f"Upstream returned HTTP {resp.status_code}: {url}",
                                    request=resp.request,
                                    response=resp,
                                )
                            if resp.status_code != 206:
                                raise SelectiveFallbackError(
                                    f"range_unexpected_status_{resp.status_code}"
                                )

                            # Validate Content-Range header
                            content_range = resp.headers.get("content-range", "")
                            match = _CONTENT_RANGE_PATTERN.match(content_range.strip())
                            if not match:
                                raise SelectiveFallbackError(
                                    "range_content_mismatch",
                                    f"Malformed Content-Range: {content_range!r}",
                                )
                            cr_start = int(match.group(1))
                            cr_end = int(match.group(2))
                            if cr_start != start_offset:
                                raise SelectiveFallbackError(
                                    "range_content_mismatch",
                                    f"Start offset mismatch: expected {start_offset}, got {cr_start}",
                                )
                            if end_offset is not None and cr_end != end_offset:
                                raise SelectiveFallbackError(
                                    "range_content_mismatch",
                                    f"End offset mismatch: expected {end_offset}, got {cr_end}",
                                )

                            chunk = await resp.aread()

                            # Body length validation
                            expected_len = cr_end - cr_start + 1
                            if len(chunk) != expected_len:
                                raise SelectiveFallbackError(
                                    "range_truncated",
                                    f"Truncated body: expected {expected_len} bytes, got {len(chunk)}",
                                )

                            # Binary GRIB message validation across all messages in group
                            offset = 0
                            for rec in group:
                                if offset + 4 > len(chunk) or chunk[offset : offset + 4] != b"GRIB":
                                    raise SelectiveFallbackError(
                                        "grib_magic_invalid",
                                        f"Chunk missing 'GRIB' magic header at offset {offset}",
                                    )
                                if offset + 16 > len(chunk):
                                    raise SelectiveFallbackError(
                                        "grib_length_invalid",
                                        f"Chunk shorter than Section 0 header at offset {offset}",
                                    )
                                declared_len = int.from_bytes(
                                    chunk[offset + 8 : offset + 16], byteorder="big", signed=False
                                )
                                if declared_len <= 0 or offset + declared_len > len(chunk):
                                    raise SelectiveFallbackError(
                                        "grib_length_invalid",
                                        f"GRIB declared length {declared_len} invalid at offset {offset}",
                                    )
                                if rec.byte_length is not None and declared_len != rec.byte_length:
                                    raise SelectiveFallbackError(
                                        "grib_length_invalid",
                                        f"GRIB Section 0 length {declared_len} != expected record length {rec.byte_length}",
                                    )
                                if chunk[offset + declared_len - 4 : offset + declared_len] != b"7777":
                                    raise SelectiveFallbackError(
                                        "grib_terminator_invalid",
                                        f"Chunk missing '7777' terminator for record #{rec.record_number}",
                                    )
                                offset += declared_len

                            if offset != len(chunk):
                                raise SelectiveFallbackError(
                                    "grib_length_invalid",
                                    f"GRIB decoded cumulative length {offset} != chunk length {len(chunk)}",
                                )

                            handle.write(chunk)
                            total_downloaded += len(chunk)

                    if fallback_full_bytes is None:
                        handle.flush()
                        os.fsync(handle.fileno())

                if fallback_full_bytes is not None:
                    temp_dest.unlink(missing_ok=True)
                    temp_dest_full = destination.with_suffix(f".tmp.{uuid.uuid4().hex}.grib2")
                    with temp_dest_full.open("wb") as h_full:
                        h_full.write(fallback_full_bytes)
                        h_full.flush()
                        os.fsync(h_full.fileno())
                    temp_dest_full.replace(destination)
                    return destination

                # Atomic publish
                temp_dest.replace(destination)
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                logger.info(
                    "download_complete: model=%s cycle=%sT%02dZ lead=%d member=%s "
                    "mode=selective records_selected=%d range_requests=%d downloaded_bytes=%d duration_ms=%d",
                    model,
                    cycle_date,
                    cycle_hour,
                    lead_time_hours,
                    member,
                    len(selection.selected_records),
                    len(record_groups),
                    total_downloaded,
                    elapsed_ms,
                )
                return destination

            except (SelectiveFallbackError, DownloadFailedError):
                temp_dest.unlink(missing_ok=True)
                raise
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                temp_dest.unlink(missing_ok=True)
                last_error = exc
                if attempt < attempts:
                    await asyncio.sleep(self._settings.RETRY_BACKOFF_SECONDS * attempt)
            except Exception as exc:
                temp_dest.unlink(missing_ok=True)
                raise SelectiveFallbackError("unexpected_selective_exception", str(exc)) from exc

        assert last_error is not None
        raise SelectiveFallbackError("selective_retry_exhausted", str(last_error)) from last_error

    async def _stream_response_to_file(
        self, response: httpx.Response, destination: Path
    ) -> Path:
        """Stream an already-open HTTP response safely to destination with atomic publication."""
        temp_dest = destination.with_suffix(f".tmp.{uuid.uuid4().hex}.grib2")
        try:
            with temp_dest.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            temp_dest.replace(destination)
            return destination
        except Exception:
            temp_dest.unlink(missing_ok=True)
            raise

    async def _download_full_with_retry(self, url: str, destination: Path) -> Path:
        """Download full GRIB2 file with retries (operational fallback path)."""
        attempts = self._settings.DOWNLOAD_RETRIES + 1
        last_error: BaseException | None = None
        start_time = time.monotonic()

        for attempt in range(1, attempts + 1):
            try:
                path = await self._download_full_once(url, destination)
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                logger.info(
                    "download_complete: url=%s mode=full duration_ms=%d",
                    url,
                    elapsed_ms,
                )
                return path
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
            if attempt < attempts:
                await asyncio.sleep(self._settings.RETRY_BACKOFF_SECONDS * attempt)

        assert last_error is not None
        if isinstance(last_error, httpx.HTTPStatusError):
            raise DownloadFailedError(
                f"Upstream returned HTTP {last_error.response.status_code} "
                f"after {attempts} attempts: {url}"
            ) from last_error
        raise UpstreamUnavailableError(
            f"Upstream unreachable after {attempts} attempts: {url}"
        ) from last_error

    async def _download_full_once(self, url: str, destination: Path) -> Path:
        """Perform a single full-file download attempt with atomic staging."""
        temp_dest = destination.with_suffix(f".tmp.{uuid.uuid4().hex}.grib2")
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code == httpx.codes.NOT_FOUND:
                    raise DownloadFailedError(f"Requested file not found: {url}")
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Upstream returned HTTP {response.status_code}: {url}",
                        request=response.request,
                        response=response,
                    )
                if not 200 <= response.status_code < 300:
                    raise DownloadFailedError(
                        f"Upstream rejected the request with HTTP "
                        f"{response.status_code}: {url}"
                    )
                with temp_dest.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            temp_dest.replace(destination)
            return destination
        except Exception:
            temp_dest.unlink(missing_ok=True)
            raise

    async def download_idx(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        destination: Path,
        member: int | None = None,
        source: str | None = None,
    ) -> Path:
        """Download the GRIB2 index (``.idx``) file for a forecast product.

        Fetches the companion ``.idx`` file for a GRIB2 product from the active
        or requested upstream source. The index is a source artifact that is
        cleaned up together with the GRIB2 file after successful ingestion.
        Transient failures are retried like the data download.

        Args:
            model: Model identifier, either ``gfs`` or ``gefs``.
            cycle_date: UTC date of the model run.
            cycle_hour: UTC cycle hour; one of 0, 6, 12, 18.
            lead_time_hours: Forecast lead time in hours (0-384).
            destination: Local path the ``.idx`` file is written to.
            member: GEFS perturbation member (1..30). Required for GEFS.
            source: Optional upstream source override (``"aws_s3"`` or ``"nomads"``).

        Returns:
            The path the file was written to.

        Raises:
            InvalidRunError: If the run parameters are unsupported.
            UpstreamUnavailableError: If the upstream provider cannot be
                reached after all retry attempts.
            DownloadFailedError: If the upstream provider returns an error
                status.
        """
        url = self.build_url(
            model, cycle_date, cycle_hour, lead_time_hours, member, source=source
        )
        url = f"{url}.idx"
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return await self._download_full_with_retry(url, destination)

    def _validate_run(self, model: str, cycle_hour: int, lead_time_hours: int) -> None:
        """Validate model, cycle hour, and lead time against NOMADS limits."""
        if model not in _SUPPORTED_MODELS:
            raise InvalidRunError(
                f"Unsupported model: {model!r}; expected one of "
                f"{sorted(_SUPPORTED_MODELS)}"
            )
        if cycle_hour not in _CYCLE_HOURS:
            raise InvalidRunError(
                f"Invalid cycle hour: {cycle_hour}; expected one of "
                f"{sorted(_CYCLE_HOURS)}"
            )
        if not 0 <= lead_time_hours <= _MAX_LEAD_TIME_HOURS:
            raise InvalidRunError(
                f"Invalid lead time: {lead_time_hours}; "
                f"expected 0-{_MAX_LEAD_TIME_HOURS} hours"
            )
