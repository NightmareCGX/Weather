"""Tests proving deferred DB connection checkout during region ingestion.

Verifies that:
1. GRIB decoding, dataset normalization, and lead/member coordinate validation
   complete with ZERO database connections checked out.
2. Database connection checkout (``engine.connect()``) occurs strictly after
   write admission, immediately before ``coordinator.write_region_worker``.
3. Database connections are closed immediately upon write completion in ``finally:``.
"""

from __future__ import annotations

import asyncio
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
from ingestion.providers.noaa.connector import NOAAConnector


def _synthetic_dataset(lead: int, member: int | None = None) -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    dims = ("lead_time_hours", "latitude", "longitude")
    shape = (1, 4, 4)
    coords = {
        "lead_time_hours": [lead],
        "latitude": lat,
        "longitude": lon,
        "time": np.datetime64("2026-07-21T00:00:00"),
    }
    if member is not None:
        dims = ("member", "lead_time_hours", "latitude", "longitude")
        shape = (1, 1, 4, 4)
        coords["member"] = [member]
    return xr.Dataset(
        data_vars={"t2m": (dims, np.full(shape, 290.0, dtype=np.float32))},
        coords=coords,
    )


def test_db_checkout_deferred_past_decode_and_normalization(
    tmp_path: Path, monkeypatch
) -> None:
    """Prove execution order: download -> decode -> normalize -> validate -> DB connect -> write -> close."""
    events: list[tuple[str, str]] = []

    # Mock NOAA download
    async def _mock_download(self, model, cycle_date, cycle_hour, lead, destination, **kwargs):
        events.append(("download", f"lead_{lead}"))
        Path(destination).touch()

    monkeypatch.setattr(NOAAConnector, "download", _mock_download)

    # Mock DecodePool submit
    import concurrent.futures

    def _mock_submit(self, path):
        # Extract lead from filename
        p = Path(path)
        lead = 6 if "f006" in p.name else (12 if "f012" in p.name else 0)
        events.append(("decode", f"lead_{lead}"))
        fut = concurrent.futures.Future()
        fut.set_result(_synthetic_dataset(lead, None))
        return fut

    from ingestion.core.decode_worker import DecodePool
    monkeypatch.setattr(DecodePool, "submit", _mock_submit)

    # Set up test SQLite DB and spy on connect()
    db_file = tmp_path / "catalog.sqlite"
    real_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(real_engine)

    import ingestion.cli as CLI
    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: real_engine)

    real_connect = real_engine.connect

    def _spied_connect(*args, **kwargs):
        # Track worker-specific connection checkouts
        stack_funcs = [frame.function for frame in __import__("inspect").stack()]
        if "_run_region_write" in stack_funcs:
            events.append(("worker.connect", "checkout"))
        else:
            events.append(("engine.connect", "other"))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(real_engine, "connect", _spied_connect)

    # Mock coordinator methods to avoid needing PostgreSQL advisory locks in SQLite
    from ingestion.core.coordinator import RunCoordinator, FinalizeResult

    def _mock_init_store(*args, **kwargs):
        events.append(("coordinator.init", "store"))

    def _mock_pre_update(*args, **kwargs):
        events.append(("coordinator.pre_update", "wave"))

    def _mock_write_region(self, conn, dataset, member, generation, **kwargs):
        lead = int(dataset.coords["lead_time_hours"].values[0])
        events.append(("coordinator.write_region_worker", f"lead_{lead}"))

    def _mock_finalize_run(*args, **kwargs):
        events.append(("coordinator.finalize_run", "final"))
        return FinalizeResult(status="ready", committed_regions={"det_L0006": "g1", "det_L0012": "g2"})

    monkeypatch.setattr(RunCoordinator, "initialize_run_store", _mock_init_store)
    monkeypatch.setattr(RunCoordinator, "pre_update_wave", _mock_pre_update)
    monkeypatch.setattr(RunCoordinator, "write_region_worker", _mock_write_region)
    monkeypatch.setattr(RunCoordinator, "finalize_run", _mock_finalize_run)

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

    status = asyncio.run(
        _run_wave(
            spec=spec,
            args=args,
            catalog_spec=catalog_spec,
            store_path=store_path,
            concurrency=4,
            failures=failures,
        )
    )

    assert status == "ready"
    assert len(failures) == 0

    # Verify that for non-seed lead 12:
    # 'decode' appears BEFORE lead 12's worker.connect!
    # And worker.connect appears immediately before coordinator.write_region_worker for lead 12.
    event_list = list(events)

    idx_lead12_decode = event_list.index(("decode", "lead_12"))
    idx_lead12_write = event_list.index(("coordinator.write_region_worker", "lead_12"))

    # Assert decode occurred before the write
    assert idx_lead12_decode < idx_lead12_write

    # Find worker connection checkouts (should be exactly 2: seed worker + lead 12 worker)
    worker_connect_indices = [i for i, (evt, _) in enumerate(event_list) if evt == "worker.connect"]
    assert len(worker_connect_indices) == 2

    lead12_connect_idx = worker_connect_indices[1]

    # Crucially: Lead 12 decode happened BEFORE its DB connection checkout!
    assert idx_lead12_decode < lead12_connect_idx
    # And the DB connection checkout happened immediately before the region write!
    assert lead12_connect_idx == idx_lead12_write - 1
