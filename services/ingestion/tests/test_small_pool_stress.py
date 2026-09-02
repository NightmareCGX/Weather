"""Tests proving scheduler-driven DB backpressure prevents QueuePool exhaustion.

Exercises real PostgreSQL with a deliberately small QueuePool:
    DB_POOL_SIZE = 2
    DB_MAX_OVERFLOW = 1
    DB_POOL_TIMEOUT_SECONDS = 5.0

Proves that when write concurrency is bounded by application backpressure (write_sem <= 2),
high logical concurrency (e.g. 8) across 10 regions causes ZERO QueuePool timeouts,
all regions commit, and finalization succeeds.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine, text

from ingestion.cli import (
    RunSpec,
    _build_spec,
    _run_wave,
)
from ingestion.core.catalog import CatalogBase
from ingestion.core.config import IngestionSettings
from ingestion.providers.noaa.connector import NOAAConnector

def _pg_reachable() -> bool:
    try:
        from ingestion.core.config import settings

        eng = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_reachable(), reason="PostgreSQL test instance not reachable"
)


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
    return xr.Dataset(
        data_vars={"t2m": (dims, np.full(shape, 290.0, dtype=np.float32))},
        coords=coords,
    )


def test_small_queue_pool_stress_with_high_logical_concurrency(
    tmp_path: Path, monkeypatch
) -> None:
    """Run 10 regions through a real PostgreSQL QueuePool with size=2, overflow=1, timeout=5.0s."""
    from ingestion.core.config import settings

    # Build a small-pool PG engine
    small_pool_engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=1,
        pool_timeout=5.0,
    )
    CatalogBase.metadata.create_all(small_pool_engine)

    # Ingestion settings with matching write_concurrency=2
    custom_settings = IngestionSettings(
        MAX_DOWNLOAD_CONCURRENCY=8,
        MAX_DECODE_CONCURRENCY=4,
        MAX_WRITE_CONCURRENCY=2,  # write_concurrency <= pool_size
        DB_POOL_SIZE=2,
        DB_MAX_OVERFLOW=1,
        DB_POOL_TIMEOUT_SECONDS=5.0,
    )
    monkeypatch.setattr("ingestion.core.config.settings", custom_settings)

    import ingestion.core.wave_runner as wave_runner
    monkeypatch.setattr(wave_runner, "_catalog_session_factory", lambda: small_pool_engine)

    # Fast download mock
    async def _mock_download(self, model, cycle_date, cycle_hour, lead, destination, **kwargs):
        Path(destination).touch()

    monkeypatch.setattr(NOAAConnector, "download", _mock_download)

    # Fast decode mock
    import concurrent.futures

    def _mock_submit(self, path):
        p = Path(path)
        lead_str = p.name.split(".f")[-1].split(".")[0] if ".f" in p.name else "6"
        fut = concurrent.futures.Future()
        fut.set_result(_synthetic_dataset(int(lead_str)))
        return fut

    from ingestion.core.decode_worker import DecodePool
    monkeypatch.setattr(DecodePool, "submit", _mock_submit)

    leads = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
    store_path = str(tmp_path / "small_pool_test.zarr")
    spec = RunSpec(
        model="gfs",
        cycle_date=date(2026, 7, 21),
        cycle_hour=0,
        target_lead_time_hours=leads,
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
    args.lock_timeout = 10.0

    catalog_spec = _build_spec(spec, args, store_path)
    failures: list[str] = []

    # Run with requested concurrency=8 (well above pool size of 2)
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

    # Clean up small pool
    small_pool_engine.dispose()

    # Assertions: 0 QueuePool timeout failures, 0 ingestion failures, and the
    # run stays partial (Phase 5B: the 10 targets are part of the canonical
    # 81-lead horizon, so the horizon is incomplete).
    assert len(failures) == 0, f"Failures occurred: {failures}"
    assert status == "partial"

    # Every requested target lead was committed; representative non-target
    # horizon leads remain uncommitted. The synthetic decode fixture carries
    # no GRIB units attribute, so canonical-unit normalization is a no-op and
    # the stored value is the raw 290.0 K (documented _normalize_canonical_units
    # behavior for unit-less datasets).
    from ingestion.core.zarr_writer import read_dataset

    restored = read_dataset(store_path)
    t2m = restored["temperature_2m"]
    for lead_val in leads:
        values = t2m.sel(lead_time_hours=lead_val).values
        assert not np.all(np.isnan(values))
        assert np.allclose(values, 290.0, atol=1e-3)
    for lead_val in (30, 240):
        assert np.all(np.isnan(t2m.sel(lead_time_hours=lead_val).values))
