"""Unit tests for the ``weather-ingest`` CLI production entrypoint.

These tests exercise the real CLI entrypoint (``ingestion.cli:main``) end to
end, mocking only the network download (the connector's HTTP call) so no live
NOMADS access is needed. The parse -> Zarr write -> catalog write pipeline is
run for real against the committed GRIB fixture and a local Zarr store; the
catalog write is routed to an in-memory SQLite database.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.catalog import (
    CatalogBase,
    CenterRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    record_run,
)

#: Path to the committed GRIB2 fixture, resolved from this file so the tests
#: run correctly regardless of the current working directory (root-level CI).
FIXTURE = str(Path(__file__).parent / "fixtures" / "gfs.t00z.pgrb2.0p25.f006.grib2")


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def _run_cli(argv: list[str], session: Session, monkeypatch) -> None:
    """Run the CLI, mocking the download and routing the catalog write to SQLite."""

    async def _fake_download(
        self, model, cycle_date, cycle_hour, lead_time_hours, destination
    ):
        import shutil
        from pathlib import Path

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURE, destination)
        return destination

    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download",
        _fake_download,
    )

    recorded: list[ModelRunRecord] = []

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        run = record_run(session, spec, dataset)
        recorded.append(run)
        return run

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    from ingestion.cli import main

    code = main(argv)
    assert code == 0
    assert len(recorded) == 1
    assert recorded[0].status == "ready"


def test_cli_ingest_end_to_end(session: Session, tmp_path, monkeypatch) -> None:
    """The CLI downloads, parses, writes Zarr, and records a ready run."""
    store = str(tmp_path / "gfs.zarr")
    _run_cli(
        [
            "ingest",
            "--model",
            "gfs",
            "--cycle-date",
            "2026-07-21",
            "--cycle-hour",
            "0",
            "--lead-time-hours",
            "6",
            "--store",
            store,
            "--download-dir",
            str(tmp_path / "dl"),
        ],
        session,
        monkeypatch,
    )

    # Catalog rows created via the SQLite session.
    assert session.query(ModelRunRecord).count() == 1
    assert session.query(ModelVersionRecord).count() == 1
    assert session.query(ModelRecord).count() == 1
    assert session.query(CenterRecord).count() == 1

    # The run id encodes the model + UTC cycle, so it proves the cycle was
    # normalized to UTC and the run is ready.
    run = session.query(ModelRunRecord).one()
    assert run.id == "run_202607210000_gfs"
    assert run.status == "ready"
    assert run.cycle_time.year == 2026
    assert run.cycle_time.month == 7
    assert run.cycle_time.day == 21
    assert run.cycle_time.hour == 0

    # Zarr store written.
    import os

    assert os.path.isdir(store)


def test_cli_ingest_custom_variables(session: Session, tmp_path, monkeypatch) -> None:
    """A custom --variable spec flows through to the catalog writer."""
    store = str(tmp_path / "gfs.zarr")
    _run_cli(
        [
            "ingest",
            "--model",
            "gfs",
            "--cycle-date",
            "2026-07-21",
            "--cycle-hour",
            "0",
            "--lead-time-hours",
            "6",
            "--store",
            store,
            "--download-dir",
            str(tmp_path / "dl"),
            "--variable",
            "temperature_2m:2-Meter Temperature:°C:t",
        ],
        session,
        monkeypatch,
    )
    run = session.query(ModelRunRecord).one()
    assert run.status == "ready"


def test_cli_rejects_bad_variable_spec(session: Session, tmp_path, monkeypatch) -> None:
    """An invalid --variable spec is rejected by the CLI parser."""
    from ingestion.cli import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "ingest",
                "--model",
                "gfs",
                "--cycle-date",
                "2026-07-21",
                "--cycle-hour",
                "0",
                "--lead-time-hours",
                "6",
                "--store",
                str(tmp_path / "x.zarr"),
                "--variable",
                "temperature_2m",  # too few parts
            ]
        )
    assert exc.value.code == 2
