"""Unit tests for the ingestion connector base interface."""

from datetime import date
from pathlib import Path

import pytest

from ingestion.core.base import (
    BaseConnector,
    DownloadFailedError,
    IngestionError,
    InvalidRunError,
    UpstreamUnavailableError,
)


class _StubConnector(BaseConnector):
    """Concrete connector used to verify the base interface contract."""

    def build_url(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
    ) -> str:
        return "https://example.test/file"

    async def download(
        self,
        model: str,
        cycle_date: date,
        cycle_hour: int,
        lead_time_hours: int,
        destination: Path,
    ) -> Path:
        return Path(destination)


def test_base_connector_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseConnector()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated() -> None:
    class _IncompleteConnector(BaseConnector):
        pass

    with pytest.raises(TypeError):
        _IncompleteConnector()


def test_concrete_connector_satisfies_interface() -> None:
    connector = _StubConnector()
    assert isinstance(connector, BaseConnector)
    assert (
        connector.build_url("gfs", date(2026, 7, 21), 0, 6)
        == "https://example.test/file"
    )


def test_exception_hierarchy() -> None:
    assert issubclass(InvalidRunError, IngestionError)
    assert issubclass(UpstreamUnavailableError, IngestionError)
    assert issubclass(DownloadFailedError, IngestionError)
