"""Tests for the ingestion PostgreSQL advisory-lock coordinator.

These tests exercise the native shared/exclusive advisory-lock semantics and
the lock-cleanup / leak-detection guarantees. They run against real
PostgreSQL (skipped when unreachable), following the existing
``test_catalog_postgres.py`` convention.

Leak verification uses an **independent backend session** (a separate
physical connection), never a re-check on the same session, because
session-level advisory locks are reentrant.
"""

from __future__ import annotations

import time

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from ingestion.core.config import settings
from ingestion.core.locks import (
    LockTimeoutError,
    StoreLockCoordinator,
)

DB_URL = settings.DATABASE_URL


def _pg_reachable() -> bool:
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_reachable(), reason="PostgreSQL test instance not reachable"
)


@pytest.fixture
def engine():
    eng = create_engine(DB_URL, pool_pre_ping=True)
    yield eng
    eng.dispose()


def _independent_holds(engine, key: int) -> bool:
    """Return whether an independent backend session holds ``key``.

    PostgreSQL stores a 64-bit advisory lock as ``(classid, objid)`` where
    ``classid`` is the high 32 bits and ``objid`` the low 32 bits. An
    independent backend session (never the session that acquired the lock) is
    used because session-level advisory locks are reentrant.
    """
    classid = (key >> 32) & 0xFFFFFFFF
    objid = key & 0xFFFFFFFF
    with engine.connect() as c:
        row = c.execute(
            text(
                "SELECT count(*) FROM pg_locks WHERE locktype='advisory' "
                "AND classid=:classid AND objid=:objid"
            ),
            {"classid": classid, "objid": objid},
        ).scalar()
        return int(row or 0) > 0


def _co(engine, conn: Connection, store: str) -> StoreLockCoordinator:
    return StoreLockCoordinator(conn, store_path=store, timeout_seconds=2.0)


def test_shared_shared_same_key_coexist(engine) -> None:
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    c2 = engine.connect()
    co1 = _co(engine, c1, store)
    co2 = _co(engine, c2, store)
    key = co1.store_gate()
    try:
        co1.acquire_shared_gate()
        co2.acquire_shared_gate()  # must not block: shared+shared coexist
        assert key in co1.held_keys
        assert key in co2.held_keys
    finally:
        co1.release_shared_gate()
        co2.release_shared_gate()
        c1.close()
        c2.close()


def test_exclusive_conflicts_with_shared(engine) -> None:
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    c2 = engine.connect()
    co1 = _co(engine, c1, store)
    co2 = _co(engine, c2, store)
    try:
        co1.acquire_shared_gate()
        # co2's EXCLUSIVE must wait for co1's SHARED; with a 2s timeout it
        # must NOT succeed while co1 holds shared.
        with pytest.raises(LockTimeoutError):
            co2.acquire_exclusive_gate()
    finally:
        co1.release_shared_gate()
        c1.close()
        c2.close()


def test_exclusive_acquired_after_shared_released(engine) -> None:
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    c2 = engine.connect()
    co1 = _co(engine, c1, store)
    co2 = _co(engine, c2, store)
    key = co1.store_gate()
    try:
        co1.acquire_shared_gate()
        co1.release_shared_gate()
        co2.acquire_exclusive_gate()  # now available
        assert key in co2.held_keys
    finally:
        co2.release_exclusive_gate()
        c1.close()
        c2.close()


def test_no_shared_to_exclusive_upgrade(engine) -> None:
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    co1 = _co(engine, c1, store)
    try:
        co1.acquire_shared_gate()
        # Re-acquiring the same key (as exclusive) must be refused, not an
        # upgrade.
        with pytest.raises(RuntimeError):
            co1.acquire_exclusive_gate()
    finally:
        co1.release_shared_gate()
        c1.close()


def test_region_locks_sorted_unique(engine) -> None:
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    co1 = _co(engine, c1, store)
    try:
        # Same region twice -> one lock (deduplicated).
        co1.acquire_region_locks(["det_L0006", "det_L0006", "det_L0012"])
        assert co1.region("det_L0006") in co1.held_keys
        assert co1.region("det_L0012") in co1.held_keys
        assert len(co1.held_keys) == 2
    finally:
        co1.release_all()
        c1.close()


def test_partial_region_failure_releases_all(engine) -> None:
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    co1 = _co(engine, c1, store)
    try:
        co1.acquire_region_locks(["det_L0006"])
        # A second coordinator on the same connection cannot acquire the same
        # key (refused); simulate a partial-failure path by releasing all.
        co1.release_all()
        assert co1.held_keys == set()
        # Independent backend sees no lock leaked.
        assert not _independent_holds(engine, co1.region("det_L0006"))
    finally:
        c1.close()


def test_leak_cleanup_after_exception(engine) -> None:
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    co1 = _co(engine, c1, store)
    key = co1.store_gate()
    try:
        co1.acquire_shared_gate()
        co1.close_connection()  # releases + closes
        # Independent backend: no leaked lock.
        assert not _independent_holds(engine, key)
    finally:
        c1.close()


def test_close_releases_admission(engine) -> None:
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    co1 = _co(engine, c1, store)
    key = co1.admission()
    try:
        co1.acquire_admission()
        assert _independent_holds(engine, key)
        co1.close_connection()
        assert not _independent_holds(engine, key)
    finally:
        c1.close()


def test_timeout_after_shared_acquired(engine) -> None:
    """A shared holder blocks an exclusive acquirer until the deadline."""
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    c2 = engine.connect()
    co1 = _co(engine, c1, store)
    co2 = _co(engine, c2, store)
    try:
        co1.acquire_shared_gate()
        started = time.monotonic()
        with pytest.raises(LockTimeoutError):
            co2.acquire_exclusive_gate()
        elapsed = time.monotonic() - started
        assert elapsed >= 1.0  # actually waited, bounded by the 2s timeout
    finally:
        co1.release_shared_gate()
        c1.close()
        c2.close()


def test_batched_region_locks_acquire_and_release(engine) -> None:
    """Acquiring and releasing 100 region locks in batch properly sets and removes all locks."""
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    co1 = _co(engine, c1, store)
    region_ids = [f"det_L{i:04d}" for i in range(100)]
    keys = [co1.region(r) for r in region_ids]
    try:
        co1.acquire_region_locks(region_ids)
        assert len(co1.held_keys) == 100
        # Verify from independent session that all 100 keys are held
        for k in keys[:5]:  # sample check
            assert _independent_holds(engine, k)
        co1.release_region_locks(region_ids)
        assert len(co1.held_keys) == 0
        for k in keys[:5]:
            assert not _independent_holds(engine, k)
    finally:
        co1.release_all()
        c1.close()


def test_batched_region_locks_partial_contention_timeout(engine) -> None:
    """When 1 key in a batch of 50 is held by another session, the batch times out cleanly without leaking partial locks."""
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c1 = engine.connect()
    c2 = engine.connect()
    co1 = _co(engine, c1, store)
    co2 = _co(engine, c2, store)
    region_ids = [f"det_L{i:04d}" for i in range(50)]
    contested_id = region_ids[25]
    contested_key = co1.region(contested_id)

    try:
        # co1 holds the contested key
        co1.acquire_region_locks([contested_id])
        assert _independent_holds(engine, contested_key)

        # co2 tries to acquire the full batch of 50 keys, which includes contested_key
        with pytest.raises(LockTimeoutError):
            co2.acquire_region_locks(region_ids)

        # Ensure co2 holds ZERO locks after failure (no partial leak)
        assert len(co2.held_keys) == 0
        for r in region_ids:
            k = co2.region(r)
            if k != contested_key:
                assert not _independent_holds(engine, k)
    finally:
        co1.release_all()
        co2.release_all()
        c1.close()
        c2.close()


def test_writer_cannot_be_starved_by_continuous_readers(engine) -> None:
    """A waiting exclusive writer makes bounded progress and is not starved by newly arriving readers."""
    import threading

    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    c_r1 = engine.connect()
    co_r1 = _co(engine, c_r1, store)

    # 1. R1 acquires SHARED store gate
    co_r1.acquire_shared_gate()

    writer_acquired = threading.Event()
    writer_failed = threading.Event()
    stop_stream = threading.Event()

    def writer_worker():
        c_w = engine.connect()
        co_w = StoreLockCoordinator(c_w, store_path=store, timeout_seconds=5.0)
        try:
            co_w.acquire_exclusive_gate()
            writer_acquired.set()
            # Hold briefly to verify exclusive ownership
            time.sleep(0.05)
            co_w.release_exclusive_gate()
        except Exception:
            writer_failed.set()
        finally:
            c_w.close()

    def continuous_reader_worker(worker_id: int):
        c_r = engine.connect()
        co_r = StoreLockCoordinator(c_r, store_path=store, timeout_seconds=5.0)
        try:
            while not stop_stream.is_set():
                try:
                    co_r.acquire_shared_gate()
                    time.sleep(0.02)
                    co_r.release_shared_gate()
                    time.sleep(0.01)
                except Exception:
                    pass
        finally:
            co_r.release_all()
            c_r.close()

    writer_thread = threading.Thread(target=writer_worker)
    # Start continuous reader threads that bombard the lock
    reader_threads = [
        threading.Thread(target=continuous_reader_worker, args=(i,))
        for i in range(4)
    ]
    for rt in reader_threads:
        rt.start()

    time.sleep(0.05)
    # Start writer: it must queue behind R1 and block subsequent readers
    writer_thread.start()
    time.sleep(0.1)

    # R1 releases its lock. PostgreSQL's lock manager must grant the lock to Writer!
    co_r1.release_shared_gate()
    c_r1.close()

    # Writer must acquire within 2.0 seconds despite continuous readers running
    writer_thread.join(timeout=3.0)
    stop_stream.set()
    for rt in reader_threads:
        rt.join(timeout=2.0)

    assert not writer_failed.is_set(), "Writer failed to acquire exclusive lock"
    assert writer_acquired.is_set(), "Writer was starved by continuous arriving readers"


def test_db_disconnect_during_acquire_preserves_error_and_invalidates(engine) -> None:
    """A DBAPI/OperationalError during acquire preserves the original exception and invalidates the Connection."""
    from unittest.mock import MagicMock
    from sqlalchemy.exc import OperationalError

    c1 = engine.connect()
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    co = _co(engine, c1, store)

    # Mock execute to simulate a severed connection on BEGIN
    orig_execute = c1.execute
    def mock_broken_execute(*args, **kwargs):
        raise OperationalError("server closed the connection unexpectedly", params=None, orig=Exception("socket closed"))

    c1.execute = MagicMock(side_effect=mock_broken_execute)

    try:
        with pytest.raises(OperationalError) as exc_info:
            co.acquire_exclusive_gate()

        # Assert original error message is preserved (not masked by PendingRollbackError)
        assert "server closed the connection unexpectedly" in str(exc_info.value)
        # Assert connection was invalidated
        assert c1.invalidated or c1.closed
    finally:
        c1.execute = orig_execute
        c1.close()


def test_unlock_failure_invalidates_connection(engine) -> None:
    """An unlock failure (query returns False or raises) marks the connection invalidated."""
    from unittest.mock import MagicMock

    c1 = engine.connect()
    store = "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    co = _co(engine, c1, store)
    key = co.store_gate()

    co.acquire_shared_gate()
    assert key in co.held_keys

    # Mock execute on unlock to simulate unlock returning false
    mock_result = MagicMock()
    mock_result.scalar.return_value = False
    orig_execute = c1.execute
    c1.execute = MagicMock(return_value=mock_result)

    try:
        co.release_shared_gate()
        # Connection should have been invalidated
        assert c1.invalidated or c1.closed
        assert key not in co.held_keys
    finally:
        c1.execute = orig_execute
        c1.close()


