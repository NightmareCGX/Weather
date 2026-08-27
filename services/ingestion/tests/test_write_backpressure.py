"""Tests verifying dedicated application-level write backpressure.

Proves that even when logical concurrency and download concurrency are high,
the number of concurrently active region write critical sections is strictly
bounded by ``write_concurrency`` (``write_sem``).
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import xarray as xr
from sqlalchemy import create_engine

from ingestion.cli import (
    RunSpec,
    _build_spec,
    _run_wave,
)
from ingestion.core.catalog import CatalogBase
from ingestion.core.config import IngestionSettings
from ingestion.providers.noaa.connector import NOAAConnector


def _synthetic_dataset(lead: int, member: int | None = None) -> xr.Dataset:
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


def test_write_semaphore_bounds_active_writers(tmp_path: Path, monkeypatch) -> None:
    """Test that with 8 items and write_concurrency=2, active writes never exceed 2."""
    custom_settings = IngestionSettings(
        MAX_DOWNLOAD_CONCURRENCY=16,
        MAX_DECODE_CONCURRENCY=8,
        MAX_WRITE_CONCURRENCY=2,  # Strict cap on active write workers
        DB_POOL_SIZE=10,
        DB_MAX_OVERFLOW=5,
    )
    monkeypatch.setattr("ingestion.core.config.settings", custom_settings)

    # Mock fast download
    async def _mock_download(self, model, cycle_date, cycle_hour, lead, destination, **kwargs):
        Path(destination).touch()

    monkeypatch.setattr(NOAAConnector, "download", _mock_download)

    # Mock fast decode
    import concurrent.futures

    def _mock_submit(self, path):
        p = Path(path)
        # Parse lead from filename fXXX
        lead_str = p.name.split(".f")[-1].split(".")[0] if ".f" in p.name else "6"
        fut = concurrent.futures.Future()
        fut.set_result(_synthetic_dataset(int(lead_str)))
        return fut

    from ingestion.core.decode_worker import DecodePool
    monkeypatch.setattr(DecodePool, "submit", _mock_submit)

    # Track active writers
    active_writers = 0
    peak_active_writers = 0
    writers_lock = threading.Lock()

    from ingestion.core.coordinator import RunCoordinator, FinalizeResult

    def _mock_init_store(*args, **kwargs):
        pass

    def _mock_pre_update(*args, **kwargs):
        pass

    def _mock_write_region(self, conn, dataset, member, generation, **kwargs):
        nonlocal active_writers, peak_active_writers
        with writers_lock:
            active_writers += 1
            if active_writers > peak_active_writers:
                peak_active_writers = active_writers

        # Deliberate sleep to provoke concurrency overlap
        time.sleep(0.05)

        with writers_lock:
            active_writers -= 1

    def _mock_finalize_run(*args, **kwargs):
        committed = {f"det_L{lead:04d}": f"g_{lead}" for lead in [0, 3, 6, 9, 12, 15, 18, 21]}
        return FinalizeResult(status="ready", committed_regions=committed)

    monkeypatch.setattr(RunCoordinator, "initialize_run_store", _mock_init_store)
    monkeypatch.setattr(RunCoordinator, "pre_update_wave", _mock_pre_update)
    monkeypatch.setattr(RunCoordinator, "write_region_worker", _mock_write_region)
    monkeypatch.setattr(RunCoordinator, "finalize_run", _mock_finalize_run)

    # Test engine
    db_file = tmp_path / "catalog.sqlite"
    test_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(test_engine)

    import ingestion.cli as CLI
    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: test_engine)

    store_path = str(tmp_path / "test.zarr")
    leads = (0, 3, 6, 9, 12, 15, 18, 21)
    spec = RunSpec(
        model="gfs",
        cycle_date=date(2026, 7, 21),
        cycle_hour=0,
        lead_time_hours=leads,
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

    status = asyncio.run(
        _run_wave(
            spec=spec,
            args=args,
            catalog_spec=catalog_spec,
            store_path=store_path,
            concurrency=8,
            failures=failures,
        )
    )

    assert status == "ready"
    assert len(failures) == 0

    # The peak active write workers must never exceed the configured MAX_WRITE_CONCURRENCY (2)
    assert peak_active_writers <= 2
    assert peak_active_writers > 0
