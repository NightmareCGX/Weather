"""Reader-race tests: a reader that selected READY before a downgrade must
revalidate under the SHARED gate and refuse to read the updating store.

Uses real PostgreSQL (for the advisory-lock gate + fresh Core revalidation).
Skipped when PostgreSQL is unreachable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from api.core.config import settings
from api.core.reader_gate import (
    ReaderGateLifecycle,
    ReaderLockPool,
    _ReaderGateSession,
    gated_read,
)
from api.models.entities import (
    ForecastCenter,
    Model,
    ModelRun,
    ModelVersion,
)
from domain.locks import store_gate_key
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
    """A gated read's selector runs while the SHARED advisory lock is held.

    The materialize callback opens a second DB connection and attempts a
    non-blocking EXCLUSIVE advisory lock on the store-gate key. Because the
    gate's SHARED lock is still held at that moment, the EXCLUSIVE attempt must
    fail. This proves the selector/read does its Zarr S3/local I/O while the
    gate is held (Phase 1 contract).
    """
    store_path = str(tmp_path / "gated.zarr")
    cycle_time = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)

    lat = np.arange(40.0, 40.0 + 0.25 * 8, 0.25)
    lon = np.arange(-107.0, -107.0 + 0.25 * 8, 0.25)
    lead = np.array([0, 6, 12, 18])
    lead_g, lat_g, lon_g = np.meshgrid(lead, lat, lon, indexing="ij")
    temperature = 10.0 + 0.1 * lat_g + 0.2 * lon_g + 0.5 * lead_g
    ds = xr.Dataset(
        {"temperature_2m": (("lead_time_hours", "latitude", "longitude"), temperature)},
        coords={"lead_time_hours": lead, "latitude": lat, "longitude": lon},
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

    pool = ReaderLockPool(DB_URL, pool_size=2, max_overflow=2, pool_timeout=2.0)
    lifecycle = ReaderGateLifecycle()
    lock_held_during_selector: list[bool] = []
    key = store_gate_key(store_path)

    def materialize():
        eng2 = create_engine(DB_URL, pool_pre_ping=True)
        with eng2.connect() as c2:
            acquired = c2.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": key},
            ).scalar()
            if acquired:
                c2.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": key},
                )
        eng2.dispose()
        lock_held_during_selector.append(not acquired)

        ds_open = xr.open_zarr(store_path, consolidated=False)
        sel = ds_open["temperature_2m"].sel(lead_time_hours=6, latitude=slice(40, 41), longitude=slice(-107, -106))
        return sel.values

    try:
        out = gated_read(
            pool,
            lifecycle,
            store_path=store_path,
            revalidate_db_url=DB_URL,
            materialize=materialize,
            timeout_seconds=10.0,
        )
    finally:
        pool.dispose()

    assert lock_held_during_selector == [True], (
        "SHARED advisory lock was not held during gated materialization: "
        f"{lock_held_during_selector}"
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 5), f"bounded read shape wrong: {out.shape}"


def test_reader_gate_timeout_when_exclusive_held(migrated_db, tmp_path) -> None:
    """ReaderGateTimeout is raised near the configured bound when an EXCLUSIVE lock is held."""
    from api.core.reader_gate import ReaderGateTimeout

    store_path = str(tmp_path / "contested.zarr")
    key = store_gate_key(store_path)

    # Acquire EXCLUSIVE lock on an independent connection
    eng = create_engine(DB_URL, pool_pre_ping=True)
    c_ex = eng.connect()
    c_ex.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})

    pool = ReaderLockPool(DB_URL, pool_size=2, max_overflow=2, pool_timeout=2.0)
    session = _ReaderGateSession(pool, store_path)
    try:
        with pytest.raises(ReaderGateTimeout):
            session.acquire(timeout_seconds=1.0)
    finally:
        session.release()
        c_ex.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
        c_ex.close()
        eng.dispose()
        pool.dispose()


def test_reader_disconnect_during_acquire_invalidates(migrated_db, tmp_path) -> None:
    """A DBAPI/OperationalError during reader acquire invalidates the connection and raises."""
    from unittest.mock import MagicMock
    from sqlalchemy.exc import OperationalError

    store_path = str(tmp_path / "broken_reader.zarr")
    pool = ReaderLockPool(DB_URL, pool_size=2, max_overflow=2, pool_timeout=2.0)
    session = _ReaderGateSession(pool, store_path)

    # Patch pool.connect to return a connection with a broken execute
    mock_conn = pool.connect()
    mock_conn.execute = MagicMock(
        side_effect=OperationalError("server closed the connection unexpectedly", params=None, orig=Exception("socket closed"))
    )
    pool.connect = MagicMock(return_value=mock_conn)

    try:
        with pytest.raises(OperationalError) as exc_info:
            session.acquire(timeout_seconds=2.0)

        assert "server closed the connection unexpectedly" in str(exc_info.value)
        assert mock_conn.invalidated or mock_conn.closed
    finally:
        session.release()
        pool.dispose()

