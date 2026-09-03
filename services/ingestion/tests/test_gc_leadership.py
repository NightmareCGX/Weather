"""PostgreSQL advisory-lock tests for Phase 6D GC leadership."""

from __future__ import annotations

import os
import pytest
from sqlalchemy import create_engine, text

from ingestion.gc.leadership import (
    GcLeadership,
    GcLeadershipUnavailableError,
    NoopGcLeadership,
)


def _pg_reachable() -> bool:
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
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
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
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
