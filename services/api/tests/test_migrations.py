"""PostgreSQL & PostGIS migration and schema smoke tests."""

import os
import pytest
from sqlalchemy import create_engine, inspect, text
from alembic import command
from alembic.config import Config


@pytest.fixture(scope="function")
def postgres_engine():
    db_url = os.getenv("DATABASE_URL", "postgresql://weather_user:weather_password@localhost:5432/weather_db")
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL test instance not running or reachable; skipping migration smoke test.")

    # Drop and recreate public schema to guarantee a truly clean PostGIS database state before each test
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public;"))

    yield engine
    engine.dispose()


def test_postgres_alembic_migration_smoke(postgres_engine):
    """Verify Alembic upgrade/downgrade, PostGIS availability, tables, geometry columns, and GIST indexes idempotently."""
    db_url = str(postgres_engine.url)
    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alembic_cfg_path = os.path.join(api_dir, "alembic.ini")

    alembic_cfg = Config(alembic_cfg_path)
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    # 1. First Alembic upgrade head
    command.upgrade(alembic_cfg, "head")

    inspector = inspect(postgres_engine)
    tables = inspector.get_table_names()

    expected_tables = [
        "forecast_centers",
        "models",
        "model_versions",
        "model_runs",
        "ensemble_members",
        "ensemble_member_products",
        "forecast_variables",
        "forecast_grids",
        "forecast_products",
        "stations",
        "cities",
        "ski_resorts",
        "verification_observations",
        "point_query_fallback_audit",
        "forecast_cycle_lifecycle",
        "reclamation_queue",
    ]

    for table in expected_tables:
        assert table in tables, f"Expected table {table} missing after alembic upgrade head."

    with postgres_engine.connect() as conn:
        # 2. PostGIS extension check
        res = conn.execute(text("SELECT PostGIS_Version();")).fetchone()
        assert res is not None, "PostGIS extension is not installed or available."

        # 3. Geometry columns check for spatial tables (stations, cities, ski_resorts)
        geom_check = conn.execute(
            text(
                "SELECT f_table_name FROM geometry_columns WHERE f_table_name IN ('stations', 'cities', 'ski_resorts');"
            )
        ).fetchall()
        spatial_tables = [row[0] for row in geom_check]
        assert "stations" in spatial_tables
        assert "cities" in spatial_tables
        assert "ski_resorts" in spatial_tables

        # 4. GIST index presence check
        indexes_check = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes WHERE indexname IN ('idx_stations_geom', 'idx_cities_geom', 'idx_ski_resorts_geom');"
            )
        ).fetchall()
        gist_indexes = [row[0] for row in indexes_check]
        assert "idx_stations_geom" in gist_indexes
        assert "idx_cities_geom" in gist_indexes
        assert "idx_ski_resorts_geom" in gist_indexes

    # 5. Alembic downgrade base
    command.downgrade(alembic_cfg, "base")

    # 6. Second Alembic upgrade head (idempotency test)
    command.upgrade(alembic_cfg, "head")


def test_migration_008_upgrade_downgrade_postgres(postgres_engine):
    """Test focused Migration 008 upgrade, downgrade, and re-upgrade on PostgreSQL."""
    db_url = str(postgres_engine.url)
    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    # 1. Start at revision 007
    command.upgrade(alembic_cfg, "007_reclamation_queue")

    # Verify 007 has retired_at and retired_by_cycle_time
    inspector_007 = inspect(postgres_engine)
    cols_007 = {c["name"] for c in inspector_007.get_columns("forecast_cycle_lifecycle")}
    assert "retired_at" in cols_007
    assert "retired_by_cycle_time" in cols_007
    indexes_007 = {idx["name"] for idx in inspector_007.get_indexes("forecast_cycle_lifecycle")}
    assert "idx_cycle_lifecycle_retired" in indexes_007

    # Seed prerequisite model and lifecycle row with non-null retired fields
    with postgres_engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO forecast_centers (id, center_id, name, country, created_at) "
                "VALUES ('center_noaa', 'noaa', 'NOAA', 'US', NOW()) ON CONFLICT DO NOTHING;"
            )
        )
        conn.execute(
            text(
                "INSERT INTO models (id, model_id, center_id, name, is_ensemble, resolution_km, created_at) "
                "VALUES ('model_gfs', 'gfs', 'noaa', 'GFS', false, 25.0, NOW()) ON CONFLICT DO NOTHING;"
            )
        )
        conn.execute(
            text(
                "INSERT INTO forecast_cycle_lifecycle "
                "(model_id, cycle_time, retired_at, retired_by_cycle_time, deletion_started_at, deleted_at, created_at, updated_at) "
                "VALUES ('gfs', '2026-09-01 00:00:00+00', '2026-09-02 06:00:00+00', '2026-09-02 12:00:00+00', "
                "'2026-09-02 13:00:00+00', '2026-09-02 14:00:00+00', NOW(), NOW());"
            )
        )
        conn.commit()

    # 2. Upgrade to 008
    command.upgrade(alembic_cfg, "008_drop_retired_fields")

    # Assert columns and index dropped
    inspector_008 = inspect(postgres_engine)
    cols_008 = {c["name"] for c in inspector_008.get_columns("forecast_cycle_lifecycle")}
    assert "retired_at" not in cols_008
    assert "retired_by_cycle_time" not in cols_008
    assert "deletion_started_at" in cols_008
    assert "deleted_at" in cols_008
    assert "created_at" in cols_008
    assert "updated_at" in cols_008
    assert "model_id" in cols_008
    assert "cycle_time" in cols_008

    indexes_008 = {idx["name"] for idx in inspector_008.get_indexes("forecast_cycle_lifecycle")}
    assert "idx_cycle_lifecycle_retired" not in indexes_008
    assert "idx_cycle_lifecycle_claimed" in indexes_008
    assert "idx_cycle_lifecycle_deleted" in indexes_008

    # Verify seeded row preserved V3 fields
    with postgres_engine.connect() as conn:
        row = conn.execute(
            text("SELECT model_id, cycle_time, deletion_started_at, deleted_at FROM forecast_cycle_lifecycle WHERE model_id = 'gfs'")
        ).fetchone()
        assert row is not None
        assert row[0] == "gfs"
        assert row[2] is not None
        assert row[3] is not None

    # 3. Downgrade to 007
    command.downgrade(alembic_cfg, "007_reclamation_queue")

    inspector_down = inspect(postgres_engine)
    cols_down = {c["name"]: c for c in inspector_down.get_columns("forecast_cycle_lifecycle")}
    assert "retired_at" in cols_down
    assert cols_down["retired_at"]["nullable"] is True
    assert "retired_by_cycle_time" in cols_down
    assert cols_down["retired_by_cycle_time"]["nullable"] is True

    indexes_down = {idx["name"]: idx for idx in inspector_down.get_indexes("forecast_cycle_lifecycle")}
    assert "idx_cycle_lifecycle_retired" in indexes_down
    assert indexes_down["idx_cycle_lifecycle_retired"]["column_names"] == ["model_id", "retired_at"]

    # Verify NULL-only restoration semantics (existing row has NULL retired fields, V3 preserved)
    with postgres_engine.connect() as conn:
        row_down = conn.execute(
            text("SELECT retired_at, retired_by_cycle_time, deletion_started_at, deleted_at FROM forecast_cycle_lifecycle WHERE model_id = 'gfs'")
        ).fetchone()
        assert row_down is not None
        assert row_down[0] is None  # retired_at NULL
        assert row_down[1] is None  # retired_by_cycle_time NULL
        assert row_down[2] is not None  # deletion_started_at preserved
        assert row_down[3] is not None  # deleted_at preserved

    # 4. Re-upgrade to 008 (idempotent roundtrip)
    command.upgrade(alembic_cfg, "008_drop_retired_fields")
    inspector_reup = inspect(postgres_engine)
    cols_reup = {c["name"] for c in inspector_reup.get_columns("forecast_cycle_lifecycle")}
    assert "retired_at" not in cols_reup
    assert "retired_by_cycle_time" not in cols_reup

    # 5. Verify API ORM works against contracted schema
    from sqlalchemy.orm import Session
    from api.models.entities import ForecastCycleLifecycle
    from datetime import datetime, timezone

    with Session(postgres_engine) as session:
        c_test = datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c_test,
                deletion_started_at=datetime.now(timezone.utc),
            )
        )
        session.commit()

        queried = session.get(ForecastCycleLifecycle, ("gfs", c_test))
        assert queried is not None
        assert queried.model_id == "gfs"
        assert queried.deletion_started_at is not None
        assert not hasattr(queried, "retired_at")
        assert not hasattr(queried, "retired_by_cycle_time")



