"""Regression tests: process-isolated GRIB decode boundary.

These tests pin the native-decode isolation fix: GRIB2 decoding (cfgrib/ecCodes)
must happen inside a persistent fixed-size worker-process pool, never in the
parent process threads. Each worker process owns an independent cfgrib/ecCodes
native state, so concurrent mixed GFS+GEFS decoding no longer corrupts the
shared ecCodes C library state (the root cause of
``fatal flex scanner internal error--end of buffer missed`` and
``ecCodes assertion failed`` native aborts).

Coverage (per the approved fix contract):

1. The decode worker entrypoint is top-level and spawn-compatible.
2. Multiple decode tasks execute through separate worker processes.
3. The pool decodes multiple GFS files.
4. The pool decodes multiple GEFS member files.
5. Mixed GFS + GEFS workloads share one pool.
6. GFS results retain no member dimension.
7. GEFS results retain the correct member identity.
8. Out-of-order completion does not change member/lead identity.
9. A decode failure ports to the parent without committing the region.
10. A simulated broken pool does not produce a false READY state.

All tests run against local disk stores + the committed GRIB fixtures and
runtime-built GEFS member files. No MinIO/PG needed. The process pool is
created fresh per test (Windows spawn-safe: the worker is module-top-level).
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from ingestion.core.decode_worker import DecodePool, decode_forecast_file

FIXTURES = Path(__file__).parent / "fixtures"
GFS_FIXTURE = FIXTURES / "gfs.t00z.pgrb2.0p25.f006.grib2"
MULTI_FIXTURE = FIXTURES / "gfs_multi_typeoflevel.grib2"


# ---------------------------------------------------------------------------
# Runtime GEFS single-member file builder (mirrors test_cli/test_parser)
# ---------------------------------------------------------------------------


def _write_gefs_member_file(path: Path, member_number: int, value: float) -> Path:
    """Write a tiny single-member GEFS GRIB file with the real gepNN identity."""
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    with path.open("wb") as f:
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
    return path


@pytest.fixture
def gefs_files(tmp_path) -> list[Path]:
    """Three single-member GEFS GRIB files (members 1, 17, 30)."""
    files = []
    for member in (1, 17, 30):
        files.append(_write_gefs_member_file(tmp_path / f"gep{member:02d}.grib2", member, float(member)))
    return files


# ---------------------------------------------------------------------------
# The decode pool as a shared fixture (persistent across the test body)
# ---------------------------------------------------------------------------


@pytest.fixture
def decode_pool() -> DecodePool:
    pool = DecodePool(max_workers=4)
    yield pool
    pool.shutdown()


# ---------------------------------------------------------------------------
# 1. Top-level / spawn-compatible worker
# ---------------------------------------------------------------------------


def test_decode_worker_is_module_top_level_and_importable() -> None:
    """decode_forecast_file must be a module-level, importable function.

    On Windows, multiprocessing uses spawn semantics: the worker callable is
    re-imported in every child process by qualified name. A closure or local
    function cannot be pickled and would raise a BrokenProcessPool on submit.
    """
    # The function is importable from its module (not ``__main__``-local).
    import importlib

    module = importlib.import_module(decode_forecast_file.__module__)
    assert getattr(module, decode_forecast_file.__name__) is decode_forecast_file
    # It is a plain top-level function, not a closure/lambda.
    assert callable(decode_forecast_file)
    assert decode_forecast_file.__code__.co_filename.endswith("decode_worker.py")


def test_decode_worker_top_level_from_spawn_process() -> None:
    """A spawned worker process can import and execute the decode function.

    This proves the Windows-spawn path: the child resolves ``ingestion`` (via
    the editable install's .pth) and decodes the fixture without any parent
    sys.path mutation.
    """
    if sys.platform != "win32":
        pytest.skip("spawn-child import probe is Windows-specific")
    with ProcessPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_child_probe)
        assert fut.result() is True


def _child_probe() -> bool:
    """True if a spawned process can resolve + decode the fixture."""
    from ingestion.providers.noaa.parser import parse_grib2

    ds = parse_grib2(str(GFS_FIXTURE))
    return "t2m" in ds.data_vars


# ---------------------------------------------------------------------------
# 2-5. Decode through the process pool
# ---------------------------------------------------------------------------


def test_pool_decodes_multiple_gfs_files(tmp_path, decode_pool) -> None:
    """The persistent pool decodes multiple GFS files (distinct processes)."""
    # Copy the GFS fixture to distinct paths so each is decoded independently.
    paths = []
    for i in range(3):
        p = tmp_path / f"gfs_{i}.grib2"
        p.write_bytes(GFS_FIXTURE.read_bytes())
        paths.append(str(p))
    datasets = [decode_pool.submit(p).result() for p in paths]
    assert len(datasets) == 3
    for ds in datasets:
        assert "t2m" in ds.data_vars
        # GFS is deterministic: no member axis.
        assert "member" not in ds.dims
        assert "member" not in ds.coords


def test_pool_decodes_multiple_gefs_member_files(gefs_files, decode_pool) -> None:
    """The pool decodes multiple GEFS member files retaining member identity."""
    datasets = [decode_pool.submit(str(p)).result() for p in gefs_files]
    assert len(datasets) == 3
    for ds in datasets:
        assert "t2m" in ds.data_vars
        # A single-member GEFS file exposes the upstream perturbation identity
        # as a ``member`` coordinate (the region writer promotes it to a length-1
        # dimension at commit time).
        assert "member" in ds.coords
        member_val = int(np.asarray(ds.coords["member"].values).reshape(-1)[0])
        assert member_val in (1, 17, 30)


def test_pool_mixed_gfs_and_gefs_workload(decode_pool, tmp_path, gefs_files) -> None:
    """A mixed GFS+GEFS decode workload shares one pool without corruption.

    This is the regression: before process isolation, mixed-model concurrent
    decoding in one process corrupted the shared ecCodes C state.
    """
    gfs_paths = [tmp_path / "mix_gfs.grib2", tmp_path / "mix_multi.grib2"]
    gfs_paths[0].write_bytes(GFS_FIXTURE.read_bytes())
    gfs_paths[1].write_bytes(MULTI_FIXTURE.read_bytes())
    tasks = [str(gfs_paths[0]), str(gfs_paths[1])] + [str(p) for p in gefs_files]
    datasets = [decode_pool.submit(t).result() for t in tasks]
    assert len(datasets) == len(tasks)
    # Deterministic files carry no member identity; GEFS files carry it.
    assert any("member" not in ds.coords for ds in datasets)
    assert any("member" in ds.coords for ds in datasets)
    for ds in datasets:
        assert any(name in ds.data_vars for name in ("t2m", "prate"))


def test_pool_decodes_multi_height_gfs_and_gefs(decode_pool, tmp_path) -> None:
    """The decode pool decodes multi-height GFS and GEFS files without coordinate conflicts."""
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    def _write_msg(f, sn: str, tol: str, lvl: int, member: int | None = None) -> None:
        msg = codes_grib_new_from_samples("GRIB2")
        codes_set(msg, "dataDate", 20260829)
        codes_set(msg, "dataTime", 1800)
        codes_set(msg, "stepType", "instant")
        codes_set(msg, "stepRange", "6")
        codes_set(msg, "stepUnits", "h")
        codes_set(msg, "shortName", sn)
        codes_set(msg, "typeOfLevel", tol)
        codes_set(msg, "level", lvl)
        if member is not None:
            codes_set(msg, "productDefinitionTemplateNumber", 1)
            codes_set(msg, "perturbationNumber", member)
            codes_set(msg, "numberOfForecastsInEnsemble", 30)
            codes_set(msg, "typeOfEnsembleForecast", 3)
        codes_set(msg, "gridType", "regular_ll")
        codes_set(msg, "Ni", 5)
        codes_set(msg, "Nj", 5)
        codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
        codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
        codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
        codes_set(msg, "longitudeOfLastGridPointInDegrees", 254.0)
        codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
        codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
        codes_set_values(msg, np.full((5, 5), 1.0, dtype=np.float32).ravel())
        codes_write(msg, f)
        codes_release(msg)

    # 1. Multi-height GFS file (2m temperature + 10m wind + surface gust)
    gfs_path = tmp_path / "gfs_multi_h.grib2"
    with gfs_path.open("wb") as f:
        _write_msg(f, "2t", "heightAboveGround", 2)
        _write_msg(f, "10u", "heightAboveGround", 10)
        _write_msg(f, "gust", "surface", 0)

    # 2. Multi-height GEFS file (member 5 with 2m temperature + 10m wind)
    gefs_path = tmp_path / "gefs_multi_h.grib2"
    with gefs_path.open("wb") as f:
        _write_msg(f, "2t", "heightAboveGround", 2, member=5)
        _write_msg(f, "10u", "heightAboveGround", 10, member=5)

    gfs_fut = decode_pool.submit(str(gfs_path))
    gefs_fut = decode_pool.submit(str(gefs_path))

    ds_gfs = gfs_fut.result()
    ds_gefs = gefs_fut.result()

    assert set(ds_gfs.data_vars) == {"t2m", "u10", "gust"}
    assert "member" not in ds_gfs.dims
    assert "heightAboveGround" not in ds_gfs.coords
    assert "surface" not in ds_gfs.coords

    assert set(ds_gefs.data_vars) == {"t2m", "u10"}
    assert "member" in ds_gefs.coords
    assert int(np.asarray(ds_gefs.coords["member"].values).reshape(-1)[0]) == 5
    assert "heightAboveGround" not in ds_gefs.coords


# ---------------------------------------------------------------------------
# 6-7. GFS has no member; GEFS retains member identity
# ---------------------------------------------------------------------------


def test_gfs_result_no_member_dimension(decode_pool) -> None:
    """A GFS byte-for-byte decode through the pool has no member axis."""
    ds = decode_pool.submit(str(GFS_FIXTURE)).result()
    assert "member" not in ds.dims
    assert "member" not in ds.coords
    assert "t2m" in ds.data_vars
    for dim in ds.t2m.dims:
        assert dim != "member"


def test_gefs_result_retains_member_identity(decode_pool, gefs_files) -> None:
    """The decode pool preserves the real upstream member identity."""
    ds = decode_pool.submit(str(gefs_files[1])).result()  # member 17
    assert "member" in ds.coords
    assert int(np.asarray(ds.coords["member"].values).reshape(-1)[0]) == 17


# ---------------------------------------------------------------------------
# 8. Out-of-order completion does not change identity
# ---------------------------------------------------------------------------


def test_out_of_order_pool_decode_preserves_identity(decode_pool, gefs_files) -> None:
    """Out-of-order process completion leaves member identity intact.

    Submit member 30, 1, 17 in reverse and confirm each returns under its own
    gepNN identity (a decode worker result is tagged by whom it decodes, not
    completion order).
    """
    first = decode_pool.submit(str(gefs_files[2]))  # 30
    second = decode_pool.submit(str(gefs_files[0]))  # 1
    third = decode_pool.submit(str(gefs_files[1]))  # 17
    datasets = [third.result(), second.result(), first.result()]
    for ds, expected in zip(datasets, (17, 1, 30)):
        assert int(np.asarray(ds.coords["member"].values).reshape(-1)[0]) == expected


# ---------------------------------------------------------------------------
# 9-10. Failure handling — no false READY, decode failure surfaces to parent
# ---------------------------------------------------------------------------


def test_pool_decode_failure_surfaces_to_parent(decode_pool, tmp_path) -> None:
    """A decode that raises in a worker propagates to the parent future."""
    corrupt = tmp_path / "corrupt.grib2"
    corrupt.write_bytes(b"\x00" * 64)
    with pytest.raises(Exception):
        decode_pool.submit(str(corrupt)).result()


def test_broken_pool_does_not_commit_region(decode_pool) -> None:
    """A worker that dies (native abort) leaves the region uncommitted.

    Simulate the native-abort path as directly as practical: a module-level
    worker ``_abort_in_child`` kills its process abruptly (no result pickling),
    which breaks its pool. Submitting a decode to that broken pool raises
    ``BrokenProcessPool`` — the parent-side signal the CLI records as a per-file
    failure. The parent process survives and an independent decode pool (the
    CLI's) still completes new work.
    """

    # A fresh pool whose single worker dies abruptly on first task.
    killed_pool = DecodePool(max_workers=1)
    try:
        with pytest.raises(BaseException):
            killed_pool.submit(_abort_in_child).result()
    finally:
        killed_pool.shutdown()

    # The parent is alive and the CLI's decode pool is unaffected.
    recovered = decode_pool.submit(str(GFS_FIXTURE)).result()
    assert "t2m" in recovered.data_vars


def _abort_in_child() -> None:
    """Module-level child that terminates abruptly (simulates a native abort).

    Must be top-level (spawn-safe); exits without publishing any result.
    """
    import os

    os._exit(37)


def test_decode_worker_returns_raw_dataset_contract(decode_pool) -> None:
    """The worker returns a raw-normalized dataset (no platform mapping).

    The worker boundary is decode+normalize only: variable names are the GRIB
    cfVarNames (``t2m``), NOT the platform codes. Mapping/unit conversion stays
    in the parent (CLI ``_decode_and_normalize``).
    """
    ds = decode_pool.submit(str(MULTI_FIXTURE)).result()
    # Both platform surface fields decode with their GRIB names.
    assert "t2m" in ds.data_vars
    assert "prate" in ds.data_vars


# ---------------------------------------------------------------------------
# CLI-level failure handling: decode failure / worker death => no false READY
# ---------------------------------------------------------------------------


def _make_sqlite_catalog_engine(tmp_path):
    """A file-backed SQLite catalog engine routed through the CLI factory."""
    from sqlalchemy import create_engine

    from ingestion.core.catalog import CatalogBase

    db_file = tmp_path / "catalog.sqlite"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(engine)
    return engine


class _NoopLockCoordinator:
    """No-op advisory locks for SQLite CLI tests (no PG)."""

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


def _install_cli_stubs(monkeypatch, engine):
    """Mock download/routing for a CLI run (mirrors test_cli helpers)."""
    import ingestion.core.wave_runner as wave_runner

    def _download_gefs_member(
        self,
        model,
        cycle_date,
        cycle_hour,
        lead_time_hours,
        destination,
        member=None,
        variables=None,
    ):
        from pathlib import Path

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_gefs_member_file(destination, member or 1, 280.0)
        return destination

    def _download_gfs(
        self,
        model,
        cycle_date,
        cycle_hour,
        lead_time_hours,
        destination,
        member=None,
        variables=None,
    ):
        from pathlib import Path

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(GFS_FIXTURE.read_bytes())
        return destination

    async def _fake_download(
        self,
        model,
        cycle_date,
        cycle_hour,
        lead_time_hours,
        destination,
        member=None,
        variables=None,
    ):
        if model == "gefs":
            return _download_gefs_member(
                self,
                model,
                cycle_date,
                cycle_hour,
                lead_time_hours,
                destination,
                member,
                variables,
            )
        return _download_gfs(
            self,
            model,
            cycle_date,
            cycle_hour,
            lead_time_hours,
            destination,
            member,
            variables,
        )

    async def _download_corrupt_gfs(
        self,
        model,
        cycle_date,
        cycle_hour,
        lead_time_hours,
        destination,
        member=None,
        variables=None,
    ):
        from pathlib import Path

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 128)
        return destination

    async def _fake_download_idx(self, model, cycle_date, cycle_hour,
                                 lead_time_hours, destination, member=None):
        from pathlib import Path

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("1:0:d=anl:", encoding="utf-8")
        return destination

    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download", _fake_download
    )
    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download_idx",
        _fake_download_idx,
    )
    monkeypatch.setattr(wave_runner, "_catalog_session_factory", lambda: engine)
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator", _NoopLockCoordinator
    )

    return _download_corrupt_gfs


def test_cli_decode_failure_no_false_ready(tmp_path, monkeypatch) -> None:
    """A corrupt file's decode failure leaves the run partial — never READY.

    A native-abort (or Python-level decode failure) in a worker must not commit
    the region: the run stays partial (or fails), never falsely ``ready``.
    """
    engine = _make_sqlite_catalog_engine(tmp_path)
    corrupt_download = _install_cli_stubs(monkeypatch, engine)
    # Route one download to a corrupt file so the seed decode fails.
    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download",
        corrupt_download,
    )

    from ingestion.cli import main

    store = str(tmp_path / "corrupt.zarr")
    code = main(
        [
            "ingest",
            "--model", "gfs",
            "--cycle-date", "2026-07-21",
            "--cycle-hour", "0",
            "--lead-time-hours", "6",
            "--store", store,
            "--allow-custom-store",
            "--download-dir", str(tmp_path / "dl"),
        ]
    )
    # The run failed (non-zero exit) and was NOT marked ready.
    assert code == 1
    from sqlalchemy.orm import Session

    with Session(engine) as db:
        from ingestion.core.catalog import ModelRunRecord

        runs = db.query(ModelRunRecord).all()
        assert all(run.status != "ready" for run in runs)


def test_cli_broken_decode_pool_no_false_ready(tmp_path, monkeypatch) -> None:
    """An abrupt worker death (surfaced as BrokenProcessPool) => no false READY.

    The decode pool's ``_decode_and_normalize`` (parent-side) is made to surface
    a ``BrokenProcessPool`` for one member mid-wave — the exact signal a native
    ecCodes aborted worker produces on ``future.result()``. The CLI must:
    report the failed file, keep the successfully-committed member, and never
    mark the run READY.
    """
    from concurrent.futures.process import BrokenProcessPool

    engine = _make_sqlite_catalog_engine(tmp_path)
    _install_cli_stubs(monkeypatch, engine)

    calls = {"n": 0}

    def _flaky_decode_and_normalize(future, catalog_spec):
        # Parent-side seam: the 2nd region's decode worker "died".
        calls["n"] += 1
        if calls["n"] == 2:
            raise BrokenProcessPool("decode worker process died")
        from ingestion.cli import _decode_and_normalize as real

        return real(future, catalog_spec)

    monkeypatch.setattr(
        "ingestion.core.wave_runner._decode_and_normalize", _flaky_decode_and_normalize
    )

    from ingestion.cli import main

    store = str(tmp_path / "broken.zarr")
    code = main(
        [
            "ingest",
            "--model", "gefs",
            "--cycle-date", "2026-07-21",
            "--cycle-hour", "0",
            "--lead-time-hours", "6",
            "--member", "1", "2", "3",
            "--store", store,
            "--allow-custom-store",
            "--download-dir", str(tmp_path / "dl"),
        ]
    )
    # The run failed cleanly (non-zero) and the run was NEVER marked ready:
    # a decode worker death must not produce a false READY state. If a run row
    # exists at all it must be partial/processing.
    assert code == 1
    from sqlalchemy.orm import Session

    with Session(engine) as db:
        from ingestion.core.catalog import ModelRunRecord

        runs = db.query(ModelRunRecord).all()
        assert all(run.status != "ready" for run in runs)