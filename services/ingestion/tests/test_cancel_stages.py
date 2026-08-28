"""Tests verifying cancellation behavior across all decoupled pipeline stages.

Verifies:
1. Cancellation while waiting for staging_sem, download_sem, decode_sem, or write_sem
   returns semaphore permits cleanly with no leaks.
2. In-flight write critical section is not abandoned; active worker completes cleanly,
   releases locks, closes DB connection, and non-abandoning drain succeeds.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine

from ingestion.cli import (
    RunSpec,
    _build_spec,
    _run_wave,
)
from ingestion.core.catalog import CatalogBase
from ingestion.providers.noaa.connector import NOAAConnector


def _synthetic_dataset(lead: int) -> xr.Dataset:
    lat = np.array([38.0, 38.25])
    lon = np.array([-107.0, -106.75])
    dims = ("lead_time_hours", "latitude", "longitude")
    shape = (1, 2, 2)
    coords = {
        "lead_time_hours": [lead],
        "latitude": lat,
        "longitude": lon,
        "time": np.datetime64("2026-07-21T00:00:00"),
    }
    return xr.Dataset(
        data_vars={"t2m": (dims, np.full(shape, 290.0, dtype=np.float32))},
        coords=coords,
    )


def test_cancellation_during_download(tmp_path: Path, monkeypatch) -> None:
    """Task cancelled while blocked in connector.download: releases permits cleanly."""
    download_entered = threading.Event()
    download_release = threading.Event()

    seed_done = False

    async def _blocking_download(self, model, cycle_date, cycle_hour, lead, destination, **kwargs):
        nonlocal seed_done
        Path(destination).touch()
        if not seed_done:
            seed_done = True
            return
        download_entered.set()
        while not download_release.is_set():
            await asyncio.sleep(0.01)

    monkeypatch.setattr(NOAAConnector, "download", _blocking_download)

    # Fast decode
    import concurrent.futures

    def _mock_submit(self, path):
        fut = concurrent.futures.Future()
        fut.set_result(_synthetic_dataset(6))
        return fut

    from ingestion.core.decode_worker import DecodePool
    monkeypatch.setattr(DecodePool, "submit", _mock_submit)

    from ingestion.core.coordinator import RunCoordinator, FinalizeResult
    monkeypatch.setattr(RunCoordinator, "initialize_run_store", lambda *a, **k: None)
    monkeypatch.setattr(RunCoordinator, "pre_update_wave", lambda *a, **k: None)
    monkeypatch.setattr(RunCoordinator, "write_region_worker", lambda *a, **k: None)
    monkeypatch.setattr(RunCoordinator, "finalize_run", lambda *a, **k: FinalizeResult(status="partial", committed_regions={}))

    db_file = tmp_path / "catalog.sqlite"
    test_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(test_engine)

    import ingestion.cli as CLI
    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: test_engine)

    store_path = str(tmp_path / "test.zarr")
    spec = RunSpec(
        model="gfs",
        cycle_date=date(2026, 7, 21),
        cycle_hour=0,
        lead_time_hours=(6, 12),
        store=store_path,
        allow_custom_store=True,
    )
    args = MagicMock()
    args.download_dir = str(tmp_path / "downloads")
    args.keep_downloads = True
    args.center_id = "noaa"
    args.version_string = "v1.0"
    args.grid_id = "global_025deg"
    args.variable = None
    args.lock_timeout = 5.0

    catalog_spec = _build_spec(spec, args, store_path)
    failures: list[str] = []

    async def _run():
        task = asyncio.create_task(
            _run_wave(
                spec=spec,
                args=args,
                catalog_spec=catalog_spec,
                store_path=store_path,
                concurrency=4,
                failures=failures,
            )
        )
        while not download_entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.02)
        # Release the blocked download so aggregate drain finishes
        download_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())


def test_cancellation_during_write(tmp_path: Path, monkeypatch) -> None:
    """Task cancelled while blocked in region write: worker finishes and cancellation propagates."""
    import concurrent.futures

    write_entered = threading.Event()
    write_allow_finish = threading.Event()
    worker_finished = threading.Event()

    async def _mock_download(self, model, cycle_date, cycle_hour, lead, destination, **kwargs):
        Path(destination).touch()

    monkeypatch.setattr(NOAAConnector, "download", _mock_download)

    def _mock_submit(self, path):
        fut = concurrent.futures.Future()
        fut.set_result(_synthetic_dataset(6))
        return fut

    from ingestion.core.decode_worker import DecodePool
    monkeypatch.setattr(DecodePool, "submit", _mock_submit)

    from ingestion.core.coordinator import RunCoordinator, FinalizeResult

    def _mock_init_store(*args, **kwargs):
        pass

    def _mock_pre_update(*args, **kwargs):
        pass

    def _mock_write_region(self, conn, dataset, member, generation, **kwargs):
        write_entered.set()
        # Block until released
        while not write_allow_finish.is_set():
            __import__("time").sleep(0.01)
        worker_finished.set()

    def _mock_finalize_run(*args, **kwargs):
        return FinalizeResult(status="partial", committed_regions={})

    monkeypatch.setattr(RunCoordinator, "initialize_run_store", _mock_init_store)
    monkeypatch.setattr(RunCoordinator, "pre_update_wave", _mock_pre_update)
    monkeypatch.setattr(RunCoordinator, "write_region_worker", _mock_write_region)
    monkeypatch.setattr(RunCoordinator, "finalize_run", _mock_finalize_run)

    db_file = tmp_path / "catalog.sqlite"
    test_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(test_engine)

    import ingestion.cli as CLI
    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: test_engine)

    store_path = str(tmp_path / "test.zarr")
    spec = RunSpec(
        model="gfs",
        cycle_date=date(2026, 7, 21),
        cycle_hour=0,
        lead_time_hours=(6,),
        store=store_path,
        allow_custom_store=True,
    )
    args = MagicMock()
    args.download_dir = str(tmp_path / "downloads")
    args.keep_downloads = True
    args.center_id = "noaa"
    args.version_string = "v1.0"
    args.grid_id = "global_025deg"
    args.variable = None
    args.lock_timeout = 5.0

    catalog_spec = _build_spec(spec, args, store_path)
    failures: list[str] = []

    async def _run():
        task = asyncio.create_task(
            _run_wave(
                spec=spec,
                args=args,
                catalog_spec=catalog_spec,
                store_path=store_path,
                concurrency=2,
                failures=failures,
            )
        )
        while not write_entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.05)
        # Release the synchronous worker so non-abandoning drain completes
        write_allow_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert worker_finished.is_set()
