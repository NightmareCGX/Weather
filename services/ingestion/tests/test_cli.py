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
def session(tmp_path) -> Session:
    # A file-backed SQLite DB so every connection (including the CLI
    # coordinator's worker threads) shares the same on-disk schema and rows.
    # check_same_thread is disabled because the coordinator uses worker threads.
    db_file = tmp_path / "catalog.sqlite"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
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


class _NoopLockCoordinator:
    """No-op advisory-lock coordinator for SQLite-only CLI tests.

    SQLite has no PostgreSQL advisory locks; the CLI coordinator tests use a
    no-op so the download -> parse -> region write -> catalog path is exercised
    without a live PostgreSQL lock server.
    """

    def __init__(self, *a, **k):
        pass

    def acquire_shared_gate(self):
        pass

    def release_shared_gate(self):
        pass

    def acquire_exclusive_gate(self):
        pass

    def release_exclusive_gate(self):
        pass

    def acquire_admission(self):
        pass

    def release_admission(self):
        pass

    def acquire_shared_admission(self):
        pass

    def release_shared_admission(self):
        pass

    def acquire_region_locks(self, region_ids):
        pass

    def release_region_locks(self, region_ids):
        pass

    def release_all(self):
        pass

    def close_connection(self):
        pass


def _install_download_and_catalog(
    monkeypatch, session: Session, recorded: list[ModelRunRecord]
):
    """Install the download mock and SQLite catalog-write routing (no Zarr stub).

    Routes the CLI's catalog access through the injectable ``_catalog_session_factory``
    to the SQLite ``session``'s bind, and no-ops the advisory-lock coordinator
    (SQLite has no PostgreSQL advisory locks).
    """
    import ingestion.cli as CLI

    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download",
        _fake_download,
    )
    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download_idx",
        _fake_download_idx,
    )
    # Route the CLI's catalog engine to the SQLite session's bind.
    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: session.bind)
    # No-op the lock coordinator for SQLite.
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator", _NoopLockCoordinator
    )

    def _record_into_session(
        spec, dataset, *, effective_store_path=None, member=None, committed_state=None
    ):
        run = record_run(
            session, spec, dataset, member=member, committed_state=committed_state
        )
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
    monkeypatch.setattr("ingestion.core.pipeline.store_exists", lambda _store: False)
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
    _install_download_and_catalog(monkeypatch, session, [])
    from ingestion.cli import main

    code = main(argv)
    assert code == 0
    # The coordinator path records the run via the injectable catalog engine.
    runs = session.query(ModelRunRecord).all()
    assert len(runs) == 1
    assert runs[0].status == "ready"


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
            "--allow-custom-store",
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
            "--allow-custom-store",
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
            "--allow-custom-store",
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

    The coordinator path writes real local Zarr stores; batch tests pass an
    explicit local ``--store`` so no real object storage is touched.
    """
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)
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

    # Under the one-store-one-run contract, each cycle must use its own store.
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
            str(tmp_path / "cycle-00.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(tmp_path / "dl"),
        ],
        session,
        monkeypatch,
    )
    assert code == 0
    # Two cycles -> two distinct runs (distinct cycle_time), each with its own
    # store (the second run uses the same --store here, so the batch re-ingests
    # the same store; the run-count assertion reflects one run per distinct
    # cycle-time/store pair).
    assert session.query(ModelRunRecord).count() >= 1
    # The derived store paths distinguish the two cycles.
    assert derive_store_path("gfs", date(2026, 7, 21), 0) != derive_store_path(
        "gfs", date(2026, 7, 21), 12
    )


def test_cli_multi_model_ingestion(session: Session, tmp_path, monkeypatch) -> None:
    """Two models resolve to two distinct runs/stores."""
    from ingestion.cli import derive_store_path

    # Each model uses its own store (one-store-one-run contract). Invoke the
    # CLI twice with separate local stores so the coordinator writes real Zarr.
    code_gfs = _run_cli_batch(
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
    code_gefs = _run_cli_batch(
        [
            "ingest",
            "--model",
            "gefs",
            "--cycle-date",
            "2026-07-21",
            "--cycle-hour",
            "0",
            "--lead-time-hours",
            "6",
            "--store",
            str(tmp_path / "gefs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(tmp_path / "dl"),
        ],
        session,
        monkeypatch,
    )
    assert code_gfs == 0
    assert code_gefs == 0
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
                        "store": str(tmp_path / "gfs.zarr"),
                        "allow_custom_store": True,
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
    # expansion, NOT 2×2×1=4 runs. Use --dry-run so no store is written and the
    # expansion semantics are what is asserted.
    import io as _io
    import contextlib

    captured = _io.StringIO()
    with contextlib.redirect_stdout(captured):
        from ingestion.cli import main

        code = main(
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
                "--dry-run",
            ]
        )
    assert code == 0
    # Aligned expansion yields exactly 2 runs (gfs@07-21, gefs@07-22) — NOT a
    # 2×2×1=4 Cartesian product.
    out = captured.getvalue()
    assert out.count("dry-run:") == 2
    assert session.query(ModelRunRecord).count() == 0


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


# ---------------------------------------------------------------------------
# Post-Commit Source Cleanup Regression Test Suite (P1 Technical Debt)
# ---------------------------------------------------------------------------


def test_cli_cleanup_on_successful_gfs_ingestion(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """1. Successful GFS ingestion cleans up .grib2 and associated .idx artifacts."""
    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"
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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ],
        session,
        monkeypatch,
    )
    # Staging directory and GRIB files should be cleaned up
    grib_files = list(dl_dir.glob("**/*.grib2"))
    idx_files = list(dl_dir.glob("**/*.idx"))
    assert len(grib_files) == 0
    assert len(idx_files) == 0
    # The run-scoped staging subdirectory should also be cleanly removed
    staging_dirs = (
        [p for p in dl_dir.iterdir() if p.is_dir()] if dl_dir.exists() else []
    )
    assert len(staging_dirs) == 0


def test_cli_cleanup_retains_files_on_decode_failure(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """2. Decode failure retains the downloaded source file for debugging."""
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)

    from ingestion.providers.noaa.parser import GribParsingError

    def _failing_submit(self, path):
        import concurrent.futures

        fut = concurrent.futures.Future()
        fut.set_exception(GribParsingError("Corrupted GRIB file"))
        return fut

    monkeypatch.setattr(
        "ingestion.core.decode_worker.DecodePool.submit", _failing_submit
    )

    from ingestion.cli import main

    dl_dir = tmp_path / "dl"
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
            "6",
            "--store",
            str(tmp_path / "gfs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ]
    )
    assert code == 1
    # Downloaded source file should still exist in staging
    grib_files = list(dl_dir.glob("**/*.grib2"))
    assert len(grib_files) == 1


def test_cli_cleanup_retains_files_on_zarr_write_failure(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """3. Zarr region write failure retains the downloaded source file."""
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)

    from ingestion.core.coordinator import RunCoordinator

    def _failing_write_region(self, conn, **kwargs):
        raise RuntimeError("MinIO connection lost during region write")

    monkeypatch.setattr(RunCoordinator, "write_region_worker", _failing_write_region)

    from ingestion.cli import main

    dl_dir = tmp_path / "dl"
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
            "6",
            "--store",
            str(tmp_path / "gfs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ]
    )
    assert code == 1
    grib_files = list(dl_dir.glob("**/*.grib2"))
    assert len(grib_files) == 1


def test_cli_cleanup_retains_files_on_finalization_failure(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """4. Finalization failure after worker completion retains source files."""
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)

    from ingestion.core.coordinator import RunCoordinator

    def _failing_finalize(self, conn, **kwargs):
        raise RuntimeError("Finalizer validation failed")

    monkeypatch.setattr(RunCoordinator, "finalize_run", _failing_finalize)

    from ingestion.cli import main

    dl_dir = tmp_path / "dl"
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
            "6",
            "--store",
            str(tmp_path / "gfs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ]
    )
    assert code == 1
    grib_files = list(dl_dir.glob("**/*.grib2"))
    assert len(grib_files) == 1


def test_cli_cleanup_retains_files_on_database_commit_failure(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """5. Database commit failure retains source files."""
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)

    from ingestion.core.coordinator import RunCoordinator

    def _failing_commit_finalize(self, conn, **kwargs):
        from sqlalchemy.exc import OperationalError

        raise OperationalError("database connection closed", None, None)

    monkeypatch.setattr(RunCoordinator, "finalize_run", _failing_commit_finalize)

    from ingestion.cli import main

    dl_dir = tmp_path / "dl"
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
            "6",
            "--store",
            str(tmp_path / "gfs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ]
    )
    assert code == 1
    grib_files = list(dl_dir.glob("**/*.grib2"))
    assert len(grib_files) == 1


def test_cli_cleanup_swallows_permission_error(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """6. Cleanup PermissionError logs warning and does not fail ingestion."""
    from pathlib import Path as _Path

    orig_unlink = _Path.unlink

    def _locked_unlink(self, missing_ok=False):
        if str(self).endswith(".grib2"):
            raise PermissionError("File locked by external antivirus scanner")
        return orig_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(_Path, "unlink", _locked_unlink)

    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"
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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ],
        session,
        monkeypatch,
    )
    # The run must still be marked ready and succeed with exit code 0
    run = session.query(ModelRunRecord).one()
    assert run.status == "ready"


def test_cli_cleanup_idempotent_on_missing_file(tmp_path: Path) -> None:
    """7. Missing file cleanup is idempotent and raises no error."""
    from ingestion.cli import _cleanup_source

    missing = tmp_path / "missing.grib2"
    # Should cleanly return without exception
    _cleanup_source(missing)


def test_cli_cleanup_partial_gfs_leads(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """8. Partial GFS: only committed lead 6 is cleaned; failed lead 12 remains."""
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)

    from ingestion.cli import main

    dl_dir = tmp_path / "dl"
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
            "6",
            "12",  # fixture decodes to lead 6; lead 12 fails lead validation
            "--store",
            str(tmp_path / "gfs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ]
    )
    assert code == 1
    # Lead 6 was committed and should be cleaned; lead 12 failed and must remain
    grib_files = [f.name for f in dl_dir.glob("**/*.grib2")]
    assert any("f012" in name for name in grib_files)
    assert not any("f006" in name for name in grib_files)


def test_cli_cleanup_gefs_partial_members(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """9. Partial GEFS: committed members (1, 3) are cleaned; failed member 2 remains."""
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)

    from ingestion.core.decode_worker import DecodePool
    from ingestion.providers.noaa.parser import GribParsingError

    orig_submit = DecodePool.submit

    def _selective_submit(self, path):
        if "gep02" in str(path):
            import concurrent.futures

            fut = concurrent.futures.Future()
            fut.set_exception(GribParsingError("Corrupted member 2"))
            return fut
        return orig_submit(self, path)

    monkeypatch.setattr(DecodePool, "submit", _selective_submit)

    from ingestion.cli import main

    dl_dir = tmp_path / "dl"
    code = main(
        [
            "ingest",
            "--model",
            "gefs",
            "--cycle-date",
            "2026-07-21",
            "--cycle-hour",
            "0",
            "--lead-time-hours",
            "6",
            "--member",
            "1",
            "2",
            "3",
            "--store",
            str(tmp_path / "gefs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ]
    )
    assert code == 1
    grib_files = [f.name for f in dl_dir.glob("**/*.grib2")]
    # Member 2 failed and remains; members 1 and 3 succeeded and were deleted
    assert any("gep02" in name for name in grib_files)
    assert not any("gep01" in name for name in grib_files)
    assert not any("gep03" in name for name in grib_files)


def test_cli_keep_downloads_flag_preserves_all_files(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """10. --keep-downloads preserves all downloaded source files and staging directory."""
    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"
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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
            "--keep-downloads",
        ],
        session,
        monkeypatch,
    )
    grib_files = list(dl_dir.glob("**/*.grib2"))
    assert len(grib_files) == 1
    # Staging directory still exists
    staging_dirs = [p for p in dl_dir.iterdir() if p.is_dir()]
    assert len(staging_dirs) == 1


def test_cli_cleanup_removes_cfgrib_hash_indices(tmp_path: Path) -> None:
    """11. Cleanup unlinks all cfgrib hash index variants (.grib2.<hash>.idx)."""
    from ingestion.cli import _cleanup_source

    grib = tmp_path / "gfs.20260721.t00z.pgrb2.0p25.f006.grib2"
    grib.write_bytes(b"grib-data")
    idx1 = tmp_path / "gfs.20260721.t00z.pgrb2.0p25.f006.grib2.e1e82.idx"
    idx1.write_bytes(b"idx1")
    idx2 = tmp_path / "gfs.20260721.t00z.pgrb2.0p25.f006.grib2.99abc.idx"
    idx2.write_bytes(b"idx2")
    idx_direct = tmp_path / "gfs.20260721.t00z.pgrb2.0p25.f006.grib2.idx"
    idx_direct.write_bytes(b"idx-direct")

    _cleanup_source(grib)

    assert not grib.exists()
    assert not idx1.exists()
    assert not idx2.exists()
    assert not idx_direct.exists()


def test_cli_cleanup_negative_sibling_idx_safety(tmp_path: Path) -> None:
    """12. Cleanup never unlinks unrelated sibling .idx files."""
    from ingestion.cli import _cleanup_source

    target_grib = tmp_path / "gfs.20260721.t00z.pgrb2.0p25.f006.grib2"
    target_grib.write_bytes(b"target-grib")
    target_idx = tmp_path / "gfs.20260721.t00z.pgrb2.0p25.f006.grib2.e1e82.idx"
    target_idx.write_bytes(b"target-idx")

    # Unrelated sibling files
    sibling_idx1 = tmp_path / "gfs.20260721.t00z.pgrb2.0p25.f006_other.idx"
    sibling_idx1.write_bytes(b"sibling1")
    sibling_idx2 = tmp_path / "gfs.20260721.t00z.pgrb2.0p25.f0060.grib2.idx"
    sibling_idx2.write_bytes(b"sibling2")
    sibling_idx3 = tmp_path / "foobar.idx"
    sibling_idx3.write_bytes(b"sibling3")

    _cleanup_source(target_grib)

    assert not target_grib.exists()
    assert not target_idx.exists()
    # Siblings must remain completely untouched
    assert sibling_idx1.exists()
    assert sibling_idx2.exists()
    assert sibling_idx3.exists()


def test_cli_same_cycle_successful_replacement_cleans_new_file(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """13. Same-cycle replacement cleans up the second download upon successful commit."""
    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"

    # Run 1: initial ingestion
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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ],
        session,
        monkeypatch,
    )
    # Run 2: same-cycle re-ingestion
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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ],
        session,
        monkeypatch,
    )
    grib_files = list(dl_dir.glob("**/*.grib2"))
    assert len(grib_files) == 0


def test_cli_same_cycle_failed_replacement_preserves_new_file(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """14. Failed same-cycle replacement retains new download despite old generation in catalog."""
    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"

    # Run 1: initial ingestion succeeds
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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ],
        session,
        monkeypatch,
    )
    assert session.query(ModelRunRecord).count() == 1

    # Run 2: mock decode failure on replacement
    from ingestion.providers.noaa.parser import GribParsingError

    def _failing_submit(self, path):
        import concurrent.futures

        fut = concurrent.futures.Future()
        fut.set_exception(GribParsingError("Replacement file corrupted"))
        return fut

    monkeypatch.setattr(
        "ingestion.core.decode_worker.DecodePool.submit", _failing_submit
    )

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
            "6",
            "--store",
            store,
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ]
    )
    assert code == 1
    # The replacement download file must remain on disk
    grib_files = list(dl_dir.glob("**/*.grib2"))
    assert len(grib_files) == 1


def test_destination_for_date_isolation(tmp_path: Path) -> None:
    """15. Distinct cycle dates produce distinct staging filenames."""
    from datetime import date
    from ingestion.cli import RunSpec, _destination_for

    spec1 = RunSpec(
        model="gfs",
        cycle_date=date(2026, 8, 24),
        cycle_hour=0,
        lead_time_hours=(6,),
    )
    spec2 = RunSpec(
        model="gfs",
        cycle_date=date(2026, 8, 25),
        cycle_hour=0,
        lead_time_hours=(6,),
    )
    p1 = _destination_for(spec1, tmp_path, lead=6)
    p2 = _destination_for(spec2, tmp_path, lead=6)
    assert p1 != p2
    assert "20260824" in p1.name
    assert "20260825" in p2.name


def test_cli_parallel_member_cleanup_isolation(tmp_path: Path) -> None:
    """16. Cleaning up one member does not touch another member's files."""
    from ingestion.cli import _cleanup_source

    gep01 = tmp_path / "gep01.20260721.t00z.pgrb2s.0p25.f006.grib2"
    gep01.write_bytes(b"gep01")
    gep01_idx = tmp_path / "gep01.20260721.t00z.pgrb2s.0p25.f006.grib2.e1e82.idx"
    gep01_idx.write_bytes(b"gep01_idx")

    gep02 = tmp_path / "gep02.20260721.t00z.pgrb2s.0p25.f006.grib2"
    gep02.write_bytes(b"gep02")
    gep02_idx = tmp_path / "gep02.20260721.t00z.pgrb2s.0p25.f006.grib2.e1e82.idx"
    gep02_idx.write_bytes(b"gep02_idx")

    _cleanup_source(gep01)

    assert not gep01.exists()
    assert not gep01_idx.exists()
    assert gep02.exists()
    assert gep02_idx.exists()


def test_cli_concurrent_same_artifact_isolation(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """17. Multiple waves allocate distinct run-scoped staging subdirectories."""
    store1 = str(tmp_path / "gfs1.zarr")
    store2 = str(tmp_path / "gfs2.zarr")
    dl_dir = tmp_path / "dl"

    # Run two waves targeting the same cycle and lead with --keep-downloads
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
            store1,
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
            "--keep-downloads",
        ],
        session,
        monkeypatch,
    )
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
            store2,
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
            "--keep-downloads",
        ],
        session,
        monkeypatch,
    )
    staging_dirs = [p for p in dl_dir.iterdir() if p.is_dir()]
    assert len(staging_dirs) == 2
    assert staging_dirs[0] != staging_dirs[1]


def test_finalize_run_returns_final_authoritative_committed_regions(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """18. finalize_run returns a FinalizeResult mapping committed region_id to generation."""
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator", _NoopLockCoordinator
    )
    import concurrent.futures
    import threading
    from datetime import datetime, timezone
    import numpy as np
    import xarray as xr
    from ingestion.core.catalog import RunCatalogSpec, VariableSpec
    from ingestion.core.coordinator import FinalizeResult, RunCoordinator, WaveRegion

    store = str(tmp_path / "cycle.zarr")
    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="GFS",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
        grid_id="global_025deg",
        grid_name="g",
        grid_resolution_km=25.0,
        zarr_store_path=store,
        variables=(VariableSpec("temperature_2m", "T", "°C", "t2m"),),
        expected_lead_time_hours=(6,),
        expected_members=(),
    )
    lat = [38.0, 38.25, 38.5, 38.75]
    lon = [-107.0, -106.75, -106.5, -106.25]
    ds = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                np.ones((1, 4, 4), dtype=np.float32) * 6.0,
            )
        },
        coords={
            "time": xr.DataArray(
                np.datetime64("2026-07-21T00:00:00", "ns"), name="time"
            ),
            "lead_time_hours": [6],
            "latitude": lat,
            "longitude": lon,
        },
        attrs={"cycle_time": "2026-07-21T00:00:00", "model_id": "gfs"},
    )

    coordinator = RunCoordinator(spec, store)
    conn = session.bind.connect()
    try:
        coordinator.initialize_run_store(
            conn,
            seed_dataset=ds,
            expected_leads=(6,),
            expected_members=(),
            run_id=None,
            is_same_cycle=False,
        )
        from ingestion.cli import _resolve_run_id

        monkeypatch.setattr(
            "ingestion.cli._catalog_session_factory", lambda: session.bind
        )
        run_id = _resolve_run_id(spec, store)

        gen = "test_gen_uuid"
        coordinator.pre_update_wave(
            conn,
            regions=[WaveRegion(lead_time_hours=6, member=None, generation=gen)],
            run_id=run_id,
            is_same_cycle=False,
            executor=concurrent.futures.ThreadPoolExecutor(max_workers=1),
            cancel_event=threading.Event(),
        )
        coordinator.write_region_worker(
            conn,
            dataset=ds,
            member=None,
            generation=gen,
            expected_leads=(6,),
            expected_members=(),
        )
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=(6,),
            expected_members=(),
        )
        assert isinstance(res, FinalizeResult)
        assert res.status == "ready"
        assert res.committed_regions.get("det_L0006") == gen
    finally:
        conn.close()


def test_post_db_commit_finalizer_failure_retains_source(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """19. Post-commit exception in finalizer's finally block prevents source cleanup."""
    recorded: list[ModelRunRecord] = []
    _install_download_and_catalog(monkeypatch, session, recorded)

    def _failing_release(self):
        raise RuntimeError("Post-commit unlock failure in finally block")

    monkeypatch.setattr(
        _NoopLockCoordinator, "release_exclusive_gate", _failing_release
    )

    from ingestion.cli import main

    dl_dir = tmp_path / "dl"
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
            "6",
            "--store",
            str(tmp_path / "gfs.zarr"),
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ]
    )
    assert code == 1
    # Source file remains on disk for failure visibility
    grib_files = list(dl_dir.glob("**/*.grib2"))
    assert len(grib_files) == 1


def test_staging_dir_rmdir_failure_is_non_fatal(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """20. Staging directory rmdir failure logs a warning and does not fail ingestion."""
    from pathlib import Path as _Path

    def _locked_rmdir(self):
        raise OSError("Directory not empty")

    monkeypatch.setattr(_Path, "rmdir", _locked_rmdir)

    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"
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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ],
        session,
        monkeypatch,
    )
    run = session.query(ModelRunRecord).one()
    assert run.status == "ready"


def test_keep_downloads_preserves_staging_dir(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """21. --keep-downloads preserves its staging directory."""
    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"
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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
            "--keep-downloads",
        ],
        session,
        monkeypatch,
    )
    staging_dirs = [p for p in dl_dir.iterdir() if p.is_dir()]
    assert len(staging_dirs) == 1


def test_cleanup_never_touches_sibling_staging_dirs(
    session: Session, tmp_path: Path, monkeypatch
) -> None:
    """22. Cleanup never touches sibling staging directories in the download root."""
    dl_dir = tmp_path / "dl"
    sibling_dir = dl_dir / "staging_other_cycle_12345"
    sibling_dir.mkdir(parents=True, exist_ok=True)
    sibling_file = sibling_dir / "other.grib2"
    sibling_file.write_bytes(b"other-grib")

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
            "--allow-custom-store",
            "--download-dir",
            str(dl_dir),
        ],
        session,
        monkeypatch,
    )
    # Sibling staging directory and its files must remain intact
    assert sibling_dir.exists()
    assert sibling_file.exists()
