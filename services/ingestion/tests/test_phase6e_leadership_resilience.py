"""Phase 6E GC Leadership Resilience, Connection Loss, and Contention Acceptance Tests.

Validates under real PostgreSQL advisory locks:
1. Leadership loss detection when physical connection is terminated server-side.
2. Non-blocking contention between concurrent GC daemon instances.
3. Leadership release on normal exit and clean shutdown.
4. Reacquisition of leadership after connection restoration.
"""

from __future__ import annotations

import os
import pytest
from sqlalchemy import create_engine, text

from ingestion.gc.leadership import GcLeadership


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


@pytest.fixture(scope="function")
def postgres_leader_env():
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            "PostgreSQL test instance not running or reachable; skipping leadership tests."
        )

    yield engine
    engine.dispose()


def test_gc_leadership_connection_loss_and_reacquisition(postgres_leader_env):
    """Prove that server-side connection termination marks leadership as lost and allows clean reacquisition."""
    engine = postgres_leader_env
    identity = "resilience-test-gc-leader"

    leader = GcLeadership(engine, identity=identity)
    assert leader.acquire() is True
    assert leader.is_leader is True

    # Get the backend PID of the leader's dedicated connection
    assert leader._conn is not None
    backend_pid = leader._conn.execute(text("SELECT pg_backend_pid()")).scalar()
    assert backend_pid is not None

    # Terminate the backend connection from another admin connection
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as admin_conn:
        admin_conn.execute(
            text("SELECT pg_terminate_backend(:pid)"), {"pid": backend_pid}
        )

    # Leader's health check must detect the termination and report lost leadership
    assert leader.check_leadership() is False
    assert leader.is_leader is False

    # Leader reacquires leadership on a fresh connection
    assert leader.acquire() is True
    assert leader.is_leader is True
    leader.release()
    assert leader.is_leader is False


def test_concurrent_gc_workers_strict_mutual_exclusion(postgres_leader_env):
    """Prove that two concurrent GC workers strictly serialize on advisory leadership."""
    engine = postgres_leader_env
    identity = "mutual-exclusion-gc-leader"

    worker1 = GcLeadership(engine, identity=identity)
    worker2 = GcLeadership(engine, identity=identity)

    try:
        # Worker 1 takes leadership
        assert worker1.acquire() is True
        assert worker1.is_leader is True

        # Worker 2 must be refused
        assert worker2.acquire() is False
        assert worker2.is_leader is False

        # Worker 1 releases leadership
        worker1.release()
        assert worker1.is_leader is False

        # Worker 2 can now claim leadership
        assert worker2.acquire() is True
        assert worker2.is_leader is True
    finally:
        worker1.release()
        worker2.release()
