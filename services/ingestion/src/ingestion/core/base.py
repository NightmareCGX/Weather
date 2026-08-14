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


class LeadTimeMismatchError(IngestionError):
    """Raised when a downloaded GRIB2 file's lead time disagrees with the run.

    The requested ``--lead-time-hours`` is used to build the download URL, but
    the file's actual lead is authoritative (derived from the GRIB ``step``
    coordinate by the parser). A mismatch indicates the file is not the
    forecast the caller asked for (e.g. a stale or mislabeled upstream file),
    so ingestion must abort rather than silently re-ingest or mislabel a lead.
    """


class CycleStoreMismatchError(IngestionError):
    """Raised when a forecast dataset belongs to a different cycle than a store.

    A Zarr store represents exactly one forecast cycle
    (``UNIQUE(model_version_id, cycle_time)`` per DATABASE.md). Merging a
    dataset from another cycle into that store would silently corrupt the
    run's data, so ingestion must fail fast instead. This is a domain-level
    condition: the caller requested a cycle that does not match the existing
    store's identity, and the merge is refused.
    """


class StoreSchemaMismatchError(IngestionError):
    """Raised when an incoming lead is structurally incompatible with a store.

    Same-cycle leads merged into a cycle store must share the same spatial
    grid (latitude/longitude axes), variable set, and (for ensembles) member
    axis. A schema mismatch indicates the upstream product layout changed
    mid-cycle; merging would corrupt the store, so ingestion fails fast.
    """


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
