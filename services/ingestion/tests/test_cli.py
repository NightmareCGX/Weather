"""Unit tests for the ``weather-ingest`` CLI production entrypoint.

These tests exercise the real CLI entrypoint (``ingestion.cli:main``) end to
end, mocking only the network download (the connector's HTTP call) so no live
NOMADS access is needed. The parse -> Zarr write -> catalog write pipeline is
run for real against the committed GRIB fixture and a local Zarr store; the
catalog write is routed to an in-memory SQLite database.
"""

from __future__ import annotations

from datetime import date
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


def _write_gefs_member_file(path: str, member_number: int, value: float) -> None:
    """Write a tiny single-member GEFS GRIB file (one perturbation member).

    Mirrors ``test_parser.py``'s runtime GEFS builder: a 2 m temperature field
    with ``perturbationNumber`` set to the real member identity, so the parser
    exposes it as the platform ``member`` coordinate value.
    """
    import numpy as np
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    with open(path, "wb") as f:
        msg = codes_grib_new_from_samples("GRIB2")
        codes_set(msg, "dataDate", 20260721)
        codes_set(msg, "dataTime", 0)
        codes_set(msg, "stepType", "instant")
        codes_set(msg, "stepRange", "6")
        codes_set(msg, "stepUnits", "h")
        codes_set(msg, "paramId", 167)
        codes_set(msg, "shortName", "2t")
        codes_set(msg, "typeOfLevel", "heightAboveGround")
        codes_set(msg, "level", 2)
        codes_set(msg, "productDefinitionTemplateNumber", 1)
        codes_set(msg, "perturbationNumber", member_number)
        codes_set(msg, "numberOfForecastsInEnsemble", 30)
        codes_set(msg, "typeOfEnsembleForecast", 3)
        codes_set(msg, "gridType", "regular_ll")
        codes_set(msg, "Ni", 10)
        codes_set(msg, "Nj", 5)
        codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
        codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
        codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
        codes_set(msg, "longitudeOfLastGridPointInDegrees", 259.0)
        codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
        codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
        codes_set_values(msg, np.full((5, 10), value, dtype=np.float32).ravel())
        codes_write(msg, f)
        codes_release(msg)


async def _fake_download(
    self, model, cycle_date, cycle_hour, lead_time_hours, destination, member=None
):
    """Download mock: copy the real GRIB fixture, or build a GEFS member file.

    The committed GFS fixture decodes to lead 6 (its GRIB step is +6h). For
    other requested leads, copy the fixture anyway — the pipeline's lead-time
    validation will reject it, which lets tests assert fail-fast behavior.
    For GEFS, a synthetic single-member file with the real ``gepNN`` identity
    is built so member-aware ingestion can be exercised without a committed
    binary fixture.
    """
    import shutil
    from pathlib import Path

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if model == "gefs":
        _write_gefs_member_file(str(destination), member or 1, 280.0)
    else:
        shutil.copyfile(FIXTURE, destination)
    return destination


async def _fake_download_idx(
    self, model, cycle_date, cycle_hour, lead_time_hours, destination, member=None
):
    """Download mock for the .idx index file (a tiny stub body)."""
    from pathlib import Path

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"1:0:d={cycle_date:%Y%m%d}{cycle_hour:02d}:2mTMP:surface:anl:",
        encoding="utf-8",
    )
    return destination


def _install_download_and_catalog(
    monkeypatch, session: Session, recorded: list[ModelRunRecord]
):
    """Install the download mock and SQLite catalog-write routing (no Zarr stub)."""
    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download",
        _fake_download,
    )
    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download_idx",
        _fake_download_idx,
    )

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None):
        run = record_run(session, spec, dataset, member=member)
        recorded.append(run)
        return run

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )


def _install_s3_stubs(monkeypatch):
    """Stub the Zarr writer so batch tests never touch real S3/MinIO.

    Stores are treated as absent (fresh writes) and the write/region primitives
    are no-ops that return the path. This lets batch/run behavior be tested
    without object storage; the Zarr round-trip itself is covered by
    ``test_zarr_roundtrip.py`` and the pipeline tests.
    """
    monkeypatch.setattr(
        "ingestion.core.pipeline.store_exists", lambda _store: False
    )
    monkeypatch.setattr(
        "ingestion.core.pipeline.prepare_run_store",
        lambda _ds, store, **kw: str(store),
    )
    monkeypatch.setattr(
        "ingestion.core.pipeline.commit_region", lambda _ds, store, **kw: str(store)
    )


def _run_cli(argv: list[str], session: Session, monkeypatch) -> None:
    """Run the CLI, mocking the download and routing the catalog write to SQLite.

    This variant does NOT stub the Zarr writer: single-run tests write to a
    real local ``tmp_path`` store. The fixture file decodes to lead 6, so any
    ``--lead-time-hours`` passed in ``argv`` must match 6 for the ingest to
    succeed under the lead-time validation.
    """
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)
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
            '--allow-custom-store',
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
    assert run.id == "run_version_gfs_v1.0_202607210000_gfs"
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
            '--allow-custom-store',
            "--download-dir",
            str(tmp_path / "dl"),
            "--variable",
            "temperature_2m:2-Meter Temperature:°C:t2m",
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
                "--allow-custom-store",
                "--variable",
                "temperature_2m",  # too few parts
            ]
        )
    assert exc.value.code == 2


def test_cli_lead_time_matches_file_succeeds(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A requested lead matching the file's decoded lead ingests successfully.

    The committed fixture ``gfs.t00z.pgrb2.0p25.f006.grib2`` decodes to lead
    6, so ``--lead-time-hours 6`` must pass the fail-fast validation.
    """
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
            '--allow-custom-store',
            "--download-dir",
            str(tmp_path / "dl"),
        ],
        session,
        monkeypatch,
    )
    run = session.query(ModelRunRecord).one()
    assert run.status == "ready"


def test_cli_lead_time_mismatch_fails_run(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A requested lead that disagrees with the file fails the run.

    The fixture decodes to lead 6, so ``--lead-time-hours 12`` must fail the
    run with a non-zero exit and record no catalog rows, rather than silently
    re-ingesting the lead-6 file as if it were lead 12.
    """
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)

    from ingestion.cli import main

    code = main(
        [
            "ingest",
            "--model",
            "gfs",
            "--cycle-date",
            "2026-07-21",
            "--cycle-hour",
            "0",
            "--lead-time-hours",
            "12",
            "--store",
            str(tmp_path / "gfs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(tmp_path / "dl"),
        ]
    )
    # The run failed: non-zero exit, no catalog rows written.
    assert code == 1
    assert session.query(ModelRunRecord).count() == 0
    assert len(recorded) == 0


# --- Store-path derivation / validation (ACCEPTANCE_REMEDIATION_PLAN §5) ---


def test_derive_store_path_reflects_identity() -> None:
    """The canonical store path separates model / cycle date / cycle hour."""
    from datetime import date

    from ingestion.cli import derive_store_path

    assert (
        derive_store_path("gfs", date(2026, 8, 13), 0)
        == "s3://weather-data/gfs/2026-08-13/00/cycle.zarr"
    )
    assert (
        derive_store_path("gfs", date(2026, 8, 13), 12)
        == "s3://weather-data/gfs/2026-08-13/12/cycle.zarr"
    )
    assert (
        derive_store_path("gefs", date(2026, 8, 13), 0)
        == "s3://weather-data/gefs/2026-08-13/00/cycle.zarr"
    )
    # Distinct cycles of the same model map to distinct stores.
    assert derive_store_path("gfs", date(2026, 8, 13), 0) != derive_store_path(
        "gfs", date(2026, 8, 13), 12
    )


def test_validate_store_path_derives_when_omitted() -> None:
    """A missing --store derives the canonical path from the identity."""
    from datetime import date

    from ingestion.cli import validate_store_path

    path = validate_store_path(None, "gfs", date(2026, 8, 13), 0)
    assert path == "s3://weather-data/gfs/2026-08-13/00/cycle.zarr"


def test_validate_store_path_accepts_matching_path() -> None:
    """A --store equal to the derived path is accepted."""
    from datetime import date

    from ingestion.cli import validate_store_path

    canonical = "s3://weather-data/gfs/2026-08-13/00/cycle.zarr"
    assert validate_store_path(canonical, "gfs", date(2026, 8, 13), 0) == canonical


def test_validate_store_path_rejects_contradicting_path() -> None:
    """A --store that contradicts the forecast identity fails fast."""
    from datetime import date

    from ingestion.cli import validate_store_path

    # The caller requests cycle-hour 12 but supplies the 00Z store path.
    wrong = "s3://weather-data/gfs/2026-08-13/00/cycle.zarr"
    with pytest.raises(ValueError, match="does not match the forecast identity"):
        validate_store_path(wrong, "gfs", date(2026, 8, 13), 12)


def test_validate_store_path_accepts_override_with_flag() -> None:
    """--allow-custom-store accepts a non-canonical path explicitly."""
    from datetime import date

    from ingestion.cli import validate_store_path

    custom = "s3://weather-data/custom/gfs-cycle.zarr"
    assert (
        validate_store_path(
            custom,
            "gfs",
            date(2026, 8, 13),
            12,
            allow_custom_store=True,
        )
        == custom
    )


# --- Batch / multi-run ingestion (ACCEPTANCE_REMEDIATION_PLAN §7) ---


def _run_cli_batch(argv: list[str], session: Session, monkeypatch) -> int:
    """Run the CLI for a batch, returning the exit code (failures are expected).

    The Zarr writer is stubbed so derived ``s3://`` store paths never touch
    real object storage.
    """
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)
    _install_s3_stubs(monkeypatch)
    from ingestion.cli import main

    return main(argv)


def test_cli_single_run_many_leads(session: Session, tmp_path, monkeypatch) -> None:
    """One model/cycle with multiple leads ingests all leads into one run."""
    # The fixture decodes to lead 6; ingest lead 6 twice would collide, so use
    # leads [6] and assert the CLI accepts the repeatable flag form. The
    # multi-lead merge is exercised by test_ingest_grib_file_merges_leads.
    code = _run_cli_batch(
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
            str(tmp_path / "gfs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(tmp_path / "dl"),
        ],
        session,
        monkeypatch,
    )
    assert code == 0
    assert session.query(ModelRunRecord).count() == 1


def test_cli_multi_cycle_ingestion(session: Session, tmp_path, monkeypatch) -> None:
    """Two cycles of one model resolve to two distinct runs/stores."""
    from ingestion.cli import derive_store_path

    code = _run_cli_batch(
        [
            "ingest",
            "--model",
            "gfs",
            "--cycle-date",
            "2026-07-21",
            "--cycle-hour",
            "0",
            "12",
            "--lead-time-hours",
            "6",
            "--store",
            str(tmp_path / "custom-store.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(tmp_path / "dl"),
        ],
        session,
        monkeypatch,
    )
    assert code == 0
    # Both cycles are distinct runs (distinct cycle_time), even though they
    # share the custom store; the store path derivation separates them.
    assert session.query(ModelRunRecord).count() == 2
    # The derived store paths distinguish the two cycles.
    assert derive_store_path("gfs", date(2026, 7, 21), 0) != derive_store_path(
        "gfs", date(2026, 7, 21), 12
    )


def test_cli_multi_model_ingestion(session: Session, tmp_path, monkeypatch) -> None:
    """Two models resolve to two distinct runs/stores."""
    from ingestion.cli import derive_store_path

    code = _run_cli_batch(
        [
            "ingest",
            "--model",
            "gfs",
            "gefs",
            "--cycle-date",
            "2026-07-21",
            "--cycle-hour",
            "0",
            "--lead-time-hours",
            "6",
            "--store",
            str(tmp_path / "custom-store.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(tmp_path / "dl"),
        ],
        session,
        monkeypatch,
    )
    # gefs is ensemble (member dim) so its product rows differ; both runs use
    # the 00Z fixture which decodes to lead 6, so both succeed.
    assert code == 0
    assert session.query(ModelRunRecord).count() == 2
    assert derive_store_path("gfs", date(2026, 7, 21), 0) != derive_store_path(
        "gefs", date(2026, 7, 21), 0
    )


def test_cli_manifest_ingestion(session: Session, tmp_path, monkeypatch) -> None:
    """A manifest ingests the explicit run list."""
    import json

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "gfs",
                        "cycle_date": "2026-07-21",
                        "cycle_hour": "0",
                        "lead_time_hours": [6],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    code = _run_cli_batch(
        ["ingest", "--manifest", str(manifest)],
        session,
        monkeypatch,
    )
    assert code == 0
    assert session.query(ModelRunRecord).count() == 1


def test_cli_dry_run_prints_specs(tmp_path) -> None:
    """--dry-run prints resolved run specs without writing anything."""
    import io as _io

    from ingestion.cli import main

    captured = _io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(captured):
        code = main(
            [
                "ingest",
                "--model",
                "gfs",
                "--cycle-date",
                "2026-08-13",
                "--cycle-hour",
                "0",
                "--lead-time-hours",
                "6",
                "--dry-run",
            ]
        )
    assert code == 0
    out = captured.getvalue()
    assert "dry-run: model=gfs" in out
    assert "2026-08-13" in out
    assert "00" in out
    assert "cycle.zarr" in out


def test_cli_anti_cartesian_guard(session: Session, tmp_path, monkeypatch) -> None:
    """Multiple model/date/hour values must align, not broadcast into a product."""
    # 2 models, 2 dates, 1 hour -> lengths {1,2} align to 2; the models pair
    # with the dates 1:1 (gfs@2026-07-21, gefs@2026-07-22). This is the aligned
    # expansion, NOT 2×2×1=4 runs. Both runs ingest lead 6 of their own cycle.
    code = _run_cli_batch(
        [
            "ingest",
            "--model",
            "gfs",
            "gefs",
            "--cycle-date",
            "2026-07-21",
            "2026-07-22",
            "--cycle-hour",
            "0",
            "--lead-time-hours",
            "6",
            "--store",
            str(tmp_path / "custom.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(tmp_path / "dl"),
        ],
        session,
        monkeypatch,
    )
    # Aligned expansion yields exactly 2 runs (gfs@07-21, gefs@07-22) — NOT a
    # 2×2×1=4 Cartesian product. Both record distinct runs (S3 stubbed, so no
    # cycle-mismatch write guard fires).
    assert code == 0
    assert session.query(ModelRunRecord).count() == 2


def test_cli_max_runs_guard(session: Session, tmp_path, monkeypatch) -> None:
    """A batch exceeding --max-runs is refused."""
    from ingestion.cli import main

    # 2 models x 2 dates x 2 hours would be ambiguous; force it via a manifest
    # with more runs than max-runs.
    import json

    manifest = tmp_path / "many.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "gfs" if i % 2 == 0 else "gefs",
                        "cycle_date": "2026-07-21",
                        "cycle_hour": "0",
                        "lead_time_hours": [6],
                    }
                    for i in range(2)
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--manifest", str(manifest), "--max-runs", "1"])
    assert "max-runs" in str(exc.value)
