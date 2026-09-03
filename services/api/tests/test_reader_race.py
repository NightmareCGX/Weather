"""Reader-race tests: a reader that selected READY before a downgrade must
revalidate under the SHARED gate and refuse to read the updating store.

Uses real PostgreSQL (for the advisory-lock gate + fresh Core revalidation).
Skipped when PostgreSQL is unreachable.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from api.core.config import settings
from api.core.reader_gate import _ReaderGateSession, ReaderLockPool
from api.models.entities import (
    ForecastCenter,
    ForecastGrid,
    Model,
    ModelRun,
    ModelVersion,
)
from tests._zarr_writer import write_dataset

DB_URL = settings.DATABASE_URL


def _pg_reachable() -> bool:
    try:
        eng = create_engine(DB_URL, pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_reachable(), reason="PostgreSQL test instance not reachable"
)


def _ensure_reader_race_version(session: Session) -> str:
    """Ensure base center/model/version exist and return version_id."""
    v_id = "version_gfs_v1"
    if not session.get(ModelVersion, v_id):
        if not session.get(ForecastCenter, "center_noaa"):
            session.add(ForecastCenter(id="center_noaa", center_id="noaa", name="NOAA", country="USA"))
            session.flush()
        if not session.get(Model, "model_gfs"):
            session.add(Model(id="model_gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0))
            session.flush()
        session.add(ModelVersion(id=v_id, model_id="gfs", version_string="v1.0"))
        session.flush()
    return v_id


def test_reader_revalidation_observes_downgrade(migrated_db, tmp_path) -> None:
    """A reader that selected READY before a downgrade revalidates and sees
    ``partial`` -> it refuses to read the store (FileNotFoundError)."""
    store_path = str(tmp_path / "cycle.zarr")
    cycle_time = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)

    with Session(migrated_db) as db:
        v_id = _ensure_reader_race_version(db)
        run = ModelRun(
            id="run_race_downgrade",
            model_version_id=v_id,
            cycle_time=cycle_time,
            status="ready",
            zarr_store_path=store_path,
        )
        db.add(run)
        db.commit()

    # Simulate downgrade to failed
    with Session(migrated_db) as db:
        r = db.get(ModelRun, "run_race_downgrade")
        assert r is not None
        setattr(r, "status", "failed")
        db.commit()

    pool = ReaderLockPool(DB_URL, pool_size=2, max_overflow=2, pool_timeout=2.0)
    session = _ReaderGateSession(pool, store_path)
    try:
        session.acquire(timeout_seconds=5.0)
        ok, path = session.revalidate(DB_URL)
        assert ok is False  # run is failed -> revalidation fails
        assert path == store_path
    finally:
        session.release()
        pool.dispose()


def test_reader_revalidation_ready_succeeds(migrated_db, tmp_path) -> None:
    """A READY run whose store is stable passes fresh Core revalidation."""
    store_path = str(tmp_path / "ready.zarr")
    cycle_time = datetime(2026, 7, 22, 6, 0, tzinfo=timezone.utc)

    with Session(migrated_db) as db:
        v_id = _ensure_reader_race_version(db)
        run = ModelRun(
            id="run_race_ready",
            model_version_id=v_id,
            cycle_time=cycle_time,
            status="ready",
            zarr_store_path=store_path,
        )
        db.add(run)
        db.commit()

    pool = ReaderLockPool(DB_URL, pool_size=2, max_overflow=2, pool_timeout=2.0)
    session = _ReaderGateSession(pool, store_path)
    try:
        session.acquire(timeout_seconds=5.0)
        ok, path = session.revalidate(DB_URL)
        assert ok is True
        assert path == store_path
    finally:
        session.release()
        pool.dispose()


def test_gated_read_materializes_bounded_selection_under_lock(migrated_db, tmp_path) -> None:
    """A gated read's selector runs while the SHARED advisory lock is held."""
    store_path = str(tmp_path / "gated.zarr")
    cycle_time = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)

    ds = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                np.ones((1, 4, 4), dtype=np.float32),
            )
        },
        coords={
            "lead_time_hours": [0],
            "latitude": [38.0, 38.25, 38.5, 38.75],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )
    write_dataset(ds, store_path)

    with Session(migrated_db) as db:
        v_id = _ensure_reader_race_version(db)
        run = ModelRun(
            id="run_race_gated",
            model_version_id=v_id,
            cycle_time=cycle_time,
            status="ready",
            zarr_store_path=store_path,
        )
        db.add(run)
        db.commit()

    from domain.locks import store_gate_key
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.main import reader_lifecycle
    reader_lifecycle._closing = False

    gate_key = store_gate_key(store_path)

    def select_with_lock_probe(dataset: xr.Dataset) -> float:
        # Probe from a second connection: attempt EXCLUSIVE lock on the SAME gate key
        eng = create_engine(DB_URL, pool_pre_ping=True)
        with eng.connect() as conn:
            acquired = conn.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": gate_key}
            ).scalar()
            if acquired:
                conn.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": gate_key}
                )
                exclusive_acquired = True
            else:
                exclusive_acquired = False
        eng.dispose()

        assert exclusive_acquired is False, (
            "EXCLUSIVE store gate lock was acquired while the selector was running! "
            "The SHARED reader gate was not held during materialization."
        )
        return float(dataset["temperature_2m"].values.sum())

    val = gated_read_dataset_with_selector(store_path, select_with_lock_probe)
    assert val == 16.0
