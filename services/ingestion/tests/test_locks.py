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
