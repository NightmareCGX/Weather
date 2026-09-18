"""PostgreSQL advisory-lock tests for Phase 6D GC leadership."""

from __future__ import annotations

import os
import pytest
from sqlalchemy import create_engine, text
from tests._integration_db import integration_db_url

from ingestion.gc.leadership import (
    GcLeadership,
    GcLeadershipUnavailableError,
    NoopGcLeadership,
)


def _pg_reachable() -> bool:
    db_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        return False
    try:
        eng = create_engine(db_url, pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


def test_noop_gc_leadership():
    noop = NoopGcLeadership()
    assert noop.is_leader is True
    assert noop.acquire() is True
    assert noop.check_leadership() is True
    noop.release()
    assert noop.is_leader is False


def test_sqlite_raises_leadership_unavailable():
    engine = create_engine("sqlite:///:memory:")
    leader = GcLeadership(engine)
    with pytest.raises(GcLeadershipUnavailableError, match="requires a PostgreSQL catalog"):
        leader.acquire()
    engine.dispose()


@pytest.mark.skipif(not _pg_reachable(), reason="PostgreSQL test database not reachable")
def test_postgres_gc_leadership_lifecycle():
    db_url = integration_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)

    leader1 = GcLeadership(engine, identity="test-gc-deployment")
    leader2 = GcLeadership(engine, identity="test-gc-deployment")

    try:
        # 1. First process acquires leadership
        assert leader1.acquire() is True
        assert leader1.is_leader is True
        assert leader1.check_leadership() is True

        # 2. Second process fails non-blocking acquisition
        assert leader2.acquire() is False
        assert leader2.is_leader is False

        # 3. First process releases leadership
        leader1.release()
        assert leader1.is_leader is False

        # 4. Second process can now acquire
        assert leader2.acquire() is True
        assert leader2.is_leader is True
    finally:
        leader1.release()
        leader2.release()
        engine.dispose()


class _FakeResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar(self) -> object:
        return self._value


class _FakeConnection:
    """Minimal Connection stand-in that records transaction boundaries.

    SQLAlchemy autobegins a transaction on ``execute``; a leadership connection
    that never ends that transaction is ``idle in transaction`` for its whole
    (process-long) life, which pins the xmin horizon and blocks VACUUM.
    """

    def __init__(self, *, lock_acquired: bool = True) -> None:
        self.closed = False
        self.invalidated = False
        self.commits = 0
        self._in_transaction = False
        self._lock_acquired = lock_acquired

    def execute(self, statement: object, params: object = None) -> _FakeResult:
        del statement, params
        self._in_transaction = True
        return _FakeResult(self._lock_acquired)

    def in_transaction(self) -> bool:
        return self._in_transaction

    def commit(self) -> None:
        self.commits += 1
        self._in_transaction = False

    def close(self) -> None:
        self.closed = True

    def invalidate(self) -> None:
        self.invalidated = True


class _FakeEngine:
    """Engine stand-in reporting a PostgreSQL dialect."""

    class _Dialect:
        name = "postgresql"

    dialect = _Dialect()

    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def connect(self) -> _FakeConnection:
        return self._conn


def test_gc_leadership_ends_its_transaction():
    """Acquiring and health-checking leadership must not leave a transaction open."""
    conn = _FakeConnection()
    leader = GcLeadership(_FakeEngine(conn))  # type: ignore[arg-type]

    assert leader.acquire() is True
    # The session-level lock survives the commit; the transaction does not.
    assert conn.commits == 1
    assert not conn.in_transaction()

    assert leader.check_leadership() is True
    assert conn.commits == 2
    assert not conn.in_transaction()


def test_gc_leadership_health_check_still_reports_lost_lock():
    """The transaction fix must not mask a lost lock."""
    conn = _FakeConnection(lock_acquired=False)
    leader = GcLeadership(_FakeEngine(conn))  # type: ignore[arg-type]
    leader._held = True
    leader._conn = conn  # type: ignore[assignment]

    assert leader.check_leadership() is False
    assert leader.is_leader is False
    assert not conn.in_transaction()
