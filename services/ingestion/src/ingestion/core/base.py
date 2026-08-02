"""Provider-agnostic connector interfaces and domain exceptions."""

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path


class IngestionError(Exception):
    """Base error for all ingestion pipeline failures."""


class InvalidRunError(IngestionError):
    """Raised when a requested model run is invalid or unsupported."""


class UpstreamUnavailableError(IngestionError):
    """Raised when the upstream provider cannot be reached."""


class DownloadFailedError(IngestionError):
    """Raised when the upstream provider fails to serve a requested file."""


class BaseConnector(ABC):
    """Base interface implemented by all upstream model connectors.

    A connector is responsible for deterministically locating and
    downloading the raw forecast files (e.g. GRIB2) of a single weather
    center.
    """

    @abstractmethod
    def build_url(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
    ) -> str:
        """Return the deterministic download URL for a forecast file.

        Args:
            model: Provider-specific model identifier.
            cycle_date: UTC date of the model run.
            cycle_hour: UTC hour of the model run.
            lead_time_hours: Forecast lead time offset from cycle time.

        Returns:
            Absolute URL of the requested forecast file.
        """

    @abstractmethod
    async def download(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        destination: Path,
    ) -> Path:
        """Download a forecast file to ``destination``.

        Args:
            model: Provider-specific model identifier.
            cycle_date: UTC date of the model run.
            cycle_hour: UTC hour of the model run.
            lead_time_hours: Forecast lead time offset from cycle time.
            destination: Local path the downloaded file is written to.

        Returns:
            The path the file was written to.

        Raises:
            InvalidRunError: If the requested run parameters are invalid.
            UpstreamUnavailableError: If the upstream provider is
                unreachable.
            DownloadFailedError: If the upstream provider returns an error
                status.
        """
