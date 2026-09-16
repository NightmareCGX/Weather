"""Tests for P2 First Cold-Load Latency Optimizations.

Covers:
1. In-Process Shared Gate Leasing:
   - Concurrent reads on same store share a single reader_pool connection (concurrency collapse).
   - Tiny pool (pool_size=2, max_overflow=0) easily handles 30 concurrent requests on the same store.
   - Immediate release when ref_count reaches 0 (no lingering connections).
   - Failure propagation to all waiters if lease initialization fails.
   - Expiration / anti-starvation: requests arriving after max lifetime trigger a fresh lease.
2. Post-Read Fencing Single-Flight:
   - 20 concurrent tile requests check fencing simultaneously; exactly 1 DB query is executed.
   - Immediate cache hit for subsequent requests.
   - Correct 404 propagation when shard is actually fenced.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.core.reader_gate import (
    ReaderGateLifecycle,
    ReaderLockPool,
    _clear_reader_gate_leases,
    gated_read,
)
from api.models.entities import (
    ForecastCenter,
    Model,
    ModelRun,
    ModelVersion,
    ReclamationQueue,
)
from api.services.tiles import (
    _clear_fencing_cache,
    _fencing_recently_verified,
    _verify_fencing_with_single_flight,
)
from tests._integration_db import integration_db_url_or_skip_module
from tests._zarr_writer import write_dataset

DB_URL = integration_db_url_or_skip_module()


@pytest.fixture(autouse=True)
def _cleanup_leases_and_fencing():
    _clear_reader_gate_leases()
    _clear_fencing_cache()
    yield
    _clear_reader_gate_leases()
    _clear_fencing_cache()


def _ensure_p2_version(session: Session) -> str:
    v_id = "version_p2_test"
    if not session.get(ModelVersion, v_id):
        if not session.get(ForecastCenter, "center_noaa"):
            session.add(ForecastCenter(id="center_noaa", center_id="noaa", name="NOAA", country="USA"))
            session.flush()
        if not session.get(Model, "model_p2"):
            session.add(Model(id="model_p2", model_id="p2", name="P2Model", center_id="noaa", is_ensemble=False, resolution_km=25.0))
            session.flush()
        session.add(ModelVersion(id=v_id, model_id="p2", version_string="v1.0"))
        session.flush()
    return v_id


def test_shared_gate_leasing_concurrency_collapse(migrated_db, tmp_path):
    """30 concurrent reads on the same store must share 1 DB connection from reader_pool."""
    store_path = str(tmp_path / "p2_lease_test.zarr")
    cycle_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)

    lat = np.array([40.0, 41.0])
    lon = np.array([-105.0, -104.0])
    lead = np.array([0, 6])
    ds = xr.Dataset(
        {"temp": (("lead_time_hours", "latitude", "longitude"), np.ones((2, 2, 2)))},
        coords={"lead_time_hours": lead, "latitude": lat, "longitude": lon},
    )
    write_dataset(ds, store_path)

    with Session(migrated_db) as db:
        v_id = _ensure_p2_version(db)
        run = ModelRun(
            id="run_p2_leasing",
            model_version_id=v_id,
            cycle_time=cycle_time,
            status="ready",
            zarr_store_path=store_path,
        )
        db.add(run)
        db.commit()

    # Intentionally tiny pool: size=2, overflow=0.
    # Without leasing, 30 concurrent requests would exhaust this immediately and timeout.
    pool = ReaderLockPool(DB_URL, pool_size=2, max_overflow=0, pool_timeout=1.0)
    lifecycle = ReaderGateLifecycle()

    max_checked_out = 0
    checked_out_lock = threading.Lock()

    def slow_materialize():
        # Artificial slow S3/disk read to widen concurrency window
        with checked_out_lock:
            nonlocal max_checked_out
            current = pool._engine.pool.checkedout()
            if current > max_checked_out:
                max_checked_out = current
        time.sleep(0.1)
        return 42

    def run_one():
        return gated_read(
            pool,
            lifecycle,
            store_path=store_path,
            revalidate_db_url=DB_URL,
            materialize=slow_materialize,
            timeout_seconds=5.0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(run_one) for _ in range(30)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 30
    assert all(r == 42 for r in results)
    # The leasing collapsed all 30 concurrent readers into exactly 1 checked-out DB connection!
    assert max_checked_out == 1, f"Expected exactly 1 checked out connection, got {max_checked_out}"
    # When all readers finished, connection was immediately returned to the pool (ref_count == 0)
    assert pool._engine.pool.checkedout() == 0
    pool.dispose()


def test_shared_gate_leasing_failure_propagation(migrated_db, tmp_path):
    """If creator fails status revalidation (e.g. status='failed'), all joiners fail cleanly."""
    store_path = str(tmp_path / "p2_failed_run.zarr")
    cycle_time = datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc)

    with Session(migrated_db) as db:
        v_id = _ensure_p2_version(db)
        run = ModelRun(
            id="run_p2_failed",
            model_version_id=v_id,
            cycle_time=cycle_time,
            status="failed",
            zarr_store_path=store_path,
        )
        db.add(run)
        db.commit()

    pool = ReaderLockPool(DB_URL, pool_size=2, max_overflow=0, pool_timeout=2.0)
    lifecycle = ReaderGateLifecycle()

    errors = []
    error_lock = threading.Lock()

    def run_one():
        try:
            gated_read(
                pool,
                lifecycle,
                store_path=store_path,
                revalidate_db_url=DB_URL,
                materialize=lambda: 1,
                timeout_seconds=5.0,
            )
        except Exception as exc:
            with error_lock:
                errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_one) for _ in range(10)]
        concurrent.futures.wait(futures)

    assert len(errors) == 10
    assert all(isinstance(e, FileNotFoundError) for e in errors)
    assert pool._engine.pool.checkedout() == 0
    pool.dispose()


def test_fencing_single_flight_collapses_db_queries(migrated_db, monkeypatch):
    """20 concurrent checks on the same shard must execute exactly 1 DB query."""
    store_path = "s3://weather-data/p2_fence_test.zarr"
    rel_key = "temperature_2m/det/0"

    from api.core import database

    real_session_local = database.SessionLocal
    db_query_count = 0
    query_lock = threading.Lock()

    class CountingSession:
        def __init__(self):
            self._real = real_session_local()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._real.close()

        def execute(self, *args, **kwargs):
            nonlocal db_query_count
            with query_lock:
                db_query_count += 1
            time.sleep(0.05)  # Widen concurrency window
            return self._real.execute(*args, **kwargs)

    monkeypatch.setattr(database, "SessionLocal", CountingSession)

    def run_check():
        _verify_fencing_with_single_flight(store_path, rel_key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(run_check) for _ in range(20)]
        concurrent.futures.wait(futures)

    # All 20 threads completed, but Single-Flight ensured only 1 DB query hit reclamation_queue!
    assert db_query_count == 1, f"Expected 1 DB query, got {db_query_count}"
    assert _fencing_recently_verified(store_path, rel_key) is True


def test_fencing_single_flight_detects_fenced_shard(migrated_db):
    """If a shard is in reclamation_queue with status='deleting', 404 is raised."""
    store_path = "s3://weather-data/p2_fence_deleting.zarr"
    rel_key = "temperature_2m/det/6"
    now = datetime.now(timezone.utc)

    with Session(migrated_db) as db:
        v_id = _ensure_p2_version(db)
        run = ModelRun(
            id="run_p2_fenced",
            model_version_id=v_id,
            cycle_time=now,
            status="ready",
            zarr_store_path=store_path,
        )
        db.add(run)
        db.flush()

        item = ReclamationQueue(
            id="rq_p2_test_deleting",
            run_id="run_p2_fenced",
            model_id="p2",
            cycle_time=now,
            lead_time_hours=6,
            variable_code="temperature_2m",
            target_kind="det",
            member_index=0,
            valid_time=now,
            store_path=store_path,
            physical_key=rel_key,
            status="deleting",
            created_at=now,
        )
        db.add(item)
        db.commit()

    with pytest.raises(HTTPException) as exc_info:
        _verify_fencing_with_single_flight(store_path, rel_key)

    assert exc_info.value.status_code == 404
    # Fenced shards must NEVER be marked verified
    assert _fencing_recently_verified(store_path, rel_key) is False


def test_shared_gate_leasing_anti_starvation_expiry(migrated_db, tmp_path, monkeypatch):
    """Requests arriving after MAX_READER_GATE_LEASE_LIFETIME_SECONDS do not join an expired lease."""
    store_path = str(tmp_path / "p2_expiry_test.zarr")
    cycle_time = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

    lat = np.array([40.0, 41.0])
    lon = np.array([-105.0, -104.0])
    lead = np.array([0])
    ds = xr.Dataset(
        {"temp": (("lead_time_hours", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={"lead_time_hours": lead, "latitude": lat, "longitude": lon},
    )
    write_dataset(ds, store_path)

    with Session(migrated_db) as db:
        v_id = _ensure_p2_version(db)
        run = ModelRun(
            id="run_p2_expiry",
            model_version_id=v_id,
            cycle_time=cycle_time,
            status="ready",
            zarr_store_path=store_path,
        )
        db.add(run)
        db.commit()

    import api.core.reader_gate as rg
    monkeypatch.setattr(rg, "MAX_READER_GATE_LEASE_LIFETIME_SECONDS", 0.05)

    pool = ReaderLockPool(DB_URL, pool_size=5, max_overflow=0, pool_timeout=2.0)
    lifecycle = ReaderGateLifecycle()

    # Request 1 runs and finishes
    res1 = gated_read(
        pool,
        lifecycle,
        store_path=store_path,
        revalidate_db_url=DB_URL,
        materialize=lambda: "pass1",
    )
    assert res1 == "pass1"
    # Lease is closed immediately when refcount reached 0
    assert len(rg._leases) == 0

    # Wait for lifetime to elapse
    time.sleep(0.06)

    # Request 2 runs after expiry, creates fresh lease and completes cleanly
    res2 = gated_read(
        pool,
        lifecycle,
        store_path=store_path,
        revalidate_db_url=DB_URL,
        materialize=lambda: "pass2",
    )
    assert res2 == "pass2"
    assert len(rg._leases) == 0
    assert pool._engine.pool.checkedout() == 0
    pool.dispose()

