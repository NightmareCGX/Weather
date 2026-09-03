"""PostgreSQL-backed integration tests for Phase 6B lifecycle operations.

Verifies paired-ready queries, lifecycle persistence, retirement reconciliation,
and tombstone survival against real PostgreSQL and the migrated schema.
"""

import os
import sys
from datetime import datetime, timezone

# Ensure services/api/src and services/ingestion/src are on sys.path.
_here = os.path.dirname(__file__)
_api_src = os.path.abspath(os.path.join(_here, "../../api/src"))
_ing_src = os.path.abspath(os.path.join(_here, "../src"))
for _p in (_api_src, _ing_src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from ingestion.core.catalog import (
    CenterRecord,
    ForecastCycleLifecycleRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    list_paired_ready_cycle_times,
    mark_cycle_retired,
    reconcile_cycle_lifecycle,
)


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="function")
def postgres_catalog_session():
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            "PostgreSQL test instance not running or reachable; skipping lifecycle postgres test."
        )

    # Clean schema and apply Alembic migrations up to head
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public;"))

    api_dir = os.path.abspath(os.path.join(_here, "../../api"))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))
    command.upgrade(alembic_cfg, "head")

    with Session(engine) as session:
        # Seed standard centers and models
        center = CenterRecord(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="US",
            created_at=_dt(2026, 1, 1, 0),
        )
        session.add(center)
        gfs = ModelRecord(
            id="model_gfs",
            model_id="gfs",
            name="GFS",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs = ModelRecord(
            id="model_gefs",
            model_id="gefs",
            name="GEFS",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        session.add_all([gfs, gefs])
        gfs_v1 = ModelVersionRecord(
            id="version_gfs_v1.0",
            model_id="gfs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs_v1 = ModelVersionRecord(
            id="version_gefs_v1.0",
            model_id="gefs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        session.add_all([gfs_v1, gefs_v1])
        session.commit()
        yield session

    engine.dispose()


def _add_run(
    db: Session,
    model_id: str,
    cycle_time: datetime,
    status: str,
    version_string: str = "v1.0",
) -> ModelRunRecord:
    v_id = f"version_{model_id}_{version_string}"
    run = ModelRunRecord(
        id=f"run_{v_id}_{cycle_time.strftime('%Y%m%d%H%M')}_{model_id}",
        model_version_id=v_id,
        cycle_time=cycle_time,
        status=status,
        created_at=cycle_time,
    )
    db.add(run)
    db.commit()
    return run


def test_postgres_paired_ready_query(postgres_catalog_session: Session) -> None:
    c1 = _dt(2026, 9, 1, 0)
    c2 = _dt(2026, 9, 1, 6)
    c3 = _dt(2026, 9, 1, 12)

    # c1: both ready -> paired-ready
    _add_run(postgres_catalog_session, "gfs", c1, "ready")
    _add_run(postgres_catalog_session, "gefs", c1, "ready")

    # c2: gfs ready, gefs partial -> NOT paired-ready
    _add_run(postgres_catalog_session, "gfs", c2, "ready")
    _add_run(postgres_catalog_session, "gefs", c2, "partial")

    # c3: both ready -> paired-ready
    _add_run(postgres_catalog_session, "gfs", c3, "ready")
    _add_run(postgres_catalog_session, "gefs", c3, "ready")

    paired = list_paired_ready_cycle_times(postgres_catalog_session)
    assert paired == [c1, c3]


def test_postgres_reconcile_cycle_lifecycle_e2e(postgres_catalog_session: Session) -> None:
    c0 = _dt(2026, 9, 1, 6)   # Will retire
    c1 = _dt(2026, 9, 2, 6)   # R1 for c0
    c2 = _dt(2026, 9, 2, 12)  # R2 for c0

    _add_run(postgres_catalog_session, "gfs", c0, "ready")
    _add_run(postgres_catalog_session, "gefs", c0, "ready")

    _add_run(postgres_catalog_session, "gfs", c1, "ready")
    _add_run(postgres_catalog_session, "gefs", c1, "ready")

    _add_run(postgres_catalog_session, "gfs", c2, "ready")
    _add_run(postgres_catalog_session, "gefs", c2, "ready")

    now = _dt(2026, 9, 2, 12, 45)
    plan = reconcile_cycle_lifecycle(postgres_catalog_session, now=now)

    assert len(plan.retirements) == 1
    assert plan.retirements[0].cycle_time == c0
    assert plan.retirements[0].retired_by_cycle_time == c1

    # Verify persisted in PostgreSQL
    row = postgres_catalog_session.get(ForecastCycleLifecycleRecord, c0)
    assert row is not None
    assert row.retired_at == now
    assert row.retired_by_cycle_time == c1
    assert row.deleted_at is None
    assert row.created_at is not None
    assert row.updated_at is not None

    # Reconciling again is idempotent
    plan2 = reconcile_cycle_lifecycle(postgres_catalog_session, now=_dt(2026, 9, 2, 13, 0))
    assert len(plan2.retirements) == 0


def test_postgres_tombstone_survives_model_run_deletion(postgres_catalog_session: Session) -> None:
    c = _dt(2026, 9, 1, 0)
    gfs_run = _add_run(postgres_catalog_session, "gfs", c, "ready")
    gefs_run = _add_run(postgres_catalog_session, "gefs", c, "ready")

    mark_cycle_retired(postgres_catalog_session, c, _dt(2026, 9, 2, 0), _dt(2026, 9, 2, 0))
    postgres_catalog_session.commit()

    # Delete the model_runs rows directly via SQL in PostgreSQL
    postgres_catalog_session.delete(gfs_run)
    postgres_catalog_session.delete(gefs_run)
    postgres_catalog_session.commit()

    # Verify lifecycle record still exists and holds tombstone
    row = postgres_catalog_session.get(ForecastCycleLifecycleRecord, c)
    assert row is not None
    assert row.retired_by_cycle_time == _dt(2026, 9, 2, 0)
