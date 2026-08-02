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
        "forecast_variables",
        "forecast_grids",
        "forecast_products",
        "stations",
        "cities",
        "ski_resorts",
        "verification_observations",
        "point_query_fallback_audit",
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

    # Final cleanup downgrade
    command.downgrade(alembic_cfg, "base")
