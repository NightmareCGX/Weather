"""NOAA NOMADS connector for operational GFS and GEFS products."""

import asyncio
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx

from ingestion.core.base import (
    BaseConnector,
    DownloadFailedError,
    InvalidRunError,
    UpstreamUnavailableError,
)
from ingestion.core.config import IngestionSettings, settings

_SUPPORTED_MODELS = frozenset({"gfs", "gefs"})
_CYCLE_HOURS = frozenset({0, 6, 12, 18})
_MAX_LEAD_TIME_HOURS = 384
_GEFS_LEAD_TIME_PRODUCT_SPLIT = 240


class NOAAConnector(BaseConnector):
    """Connector for NOAA NOMADS GRIB2 forecast products.

    Supports the deterministic GFS model and the GEFS ensemble control
    product for the four daily cycles (00Z, 06Z, 12Z, 18Z). Download URLs
    are built deterministically from the official NOMADS directory layout:
    GEFS products split between the ``pgrb2ap25`` (<= 240 h) and
    ``pgrb2bp25`` (> 240 h) file sets.

    Downloads are streamed to disk, validated against HTTP status codes,
    and retried with progressive backoff on transient failures (network
    errors and 5xx responses). All other non-2xx responses (including
    redirects) fail immediately.
    """

    def __init__(self, conn_settings: IngestionSettings | None = None) -> None:
        """Create a connector backed by an isolated HTTP client.

        Args:
            conn_settings: Ingestion settings; defaults to the module-level
                settings instance.
        """
        self._settings = conn_settings or settings
        self._client = httpx.AsyncClient(
            timeout=self._settings.REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": self._settings.NOAA_USER_AGENT},
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

    def build_url(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
    ) -> str:
        """Return the deterministic NOMADS download URL for a GRIB2 file.

        Args:
            model: Model identifier, either ``gfs`` or ``gefs``.
            cycle_date: UTC date of the model run.
            cycle_hour: UTC cycle hour; one of 0, 6, 12, 18.
            lead_time_hours: Forecast lead time in hours (0-384).

        Returns:
            Absolute NOMADS URL of the requested GRIB2 file.

        Raises:
            InvalidRunError: If the model, cycle hour, or lead time is
                unsupported.
        """
        self._validate_run(model, cycle_hour, lead_time_hours)
        date_str = cycle_date.strftime("%Y%m%d")
        hour_str = f"{cycle_hour:02d}"
        lead_str = f"{lead_time_hours:03d}"
        if model == "gfs":
            return (
                f"{self._settings.NOMADS_BASE_URL}/pub/data/nccf/com/gfs/prod/"
                f"gfs.{date_str}/{hour_str}/atmos/"
                f"gfs.t{hour_str}z.pgrb2.0p25.f{lead_str}"
            )
        if lead_time_hours <= _GEFS_LEAD_TIME_PRODUCT_SPLIT:
            return (
                f"{self._settings.NOMADS_BASE_URL}/pub/data/nccf/com/gens/prod/"
                f"gefs.{date_str}/{hour_str}/atmos/pgrb2ap25/"
                f"gefs.t{hour_str}z.pgrb2a.0p25.f{lead_str}"
            )
        return (
            f"{self._settings.NOMADS_BASE_URL}/pub/data/nccf/com/gens/prod/"
            f"gefs.{date_str}/{hour_str}/atmos/pgrb2bp25/"
            f"gefs.t{hour_str}z.pgrb2b.0p25.f{lead_str}"
        )

    async def download(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        destination: Path,
    ) -> Path:
        """Download a GRIB2 file to ``destination``.

        Transient failures (network errors and 5xx responses) are retried
        up to ``DOWNLOAD_RETRIES`` extra attempts with progressive backoff;
        all other non-2xx responses (including redirects) fail immediately.

        Args:
            model: Model identifier, either ``gfs`` or ``gefs``.
            cycle_date: UTC date of the model run.
            cycle_hour: UTC cycle hour; one of 0, 6, 12, 18.
            lead_time_hours: Forecast lead time in hours (0-384).
            destination: Local path the GRIB2 file is written to; the
                parent directory is created if missing.

        Returns:
            The path the file was written to.

        Raises:
            InvalidRunError: If the model, cycle hour, or lead time is
                unsupported.
            UpstreamUnavailableError: If the upstream provider cannot be
                reached after all retry attempts.
            DownloadFailedError: If the upstream provider returns an error
                status, or keeps returning 5xx after all retry attempts.
        """
        url = self.build_url(model, cycle_date, cycle_hour, lead_time_hours)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        attempts = self._settings.DOWNLOAD_RETRIES + 1
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._download_once(url, destination)
            except httpx.TransportError as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
            if attempt < attempts:
                await asyncio.sleep(self._settings.RETRY_BACKOFF_SECONDS * attempt)
        assert last_error is not None, "download loop must record a failure"
        if isinstance(last_error, httpx.HTTPStatusError):
            raise DownloadFailedError(
                f"Upstream returned HTTP {last_error.response.status_code} "
                f"after {attempts} attempts: {url}"
            ) from last_error
        raise UpstreamUnavailableError(
            f"Upstream unreachable after {attempts} attempts: {url}"
        ) from last_error

    async def _download_once(self, url: str, destination: Path) -> Path:
        """Perform a single download attempt with HTTP validation."""
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
            try:
                with destination.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        return destination

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
