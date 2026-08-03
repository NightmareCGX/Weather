import os
import sys
from datetime import datetime, timezone

# Ensure services/api/src is on sys.path for test execution
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.main import app
from api.models.entities import (
    ForecastCenter,
    ForecastGrid,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)


@pytest.fixture(scope="session")
def db_engine():
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
            "PostgreSQL test instance not running or reachable; skipping integration tests."
        )
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def migrated_db(db_engine):
    """Apply the Alembic schema to a clean database, following the existing
    ``test_migrations.py`` reset-and-upgrade convention."""
    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", str(db_engine.url))
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    with db_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))

    command.upgrade(alembic_cfg, "head")
    yield db_engine
    command.downgrade(alembic_cfg, "base")


@pytest.fixture(scope="session")
def seed_data(migrated_db):
    """Seed deterministic catalog rows used by the contract tests."""
    with Session(migrated_db) as session:
        noaa = ForecastCenter(
            id="center_noaa",
            center_id="noaa",
            name="National Oceanic and Atmospheric Administration",
            country="USA",
        )
        gfs = Model(
            id="model_gfs",
            model_id="gfs",
            name="Global Forecast System",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
        )
        gefs = Model(
            id="model_gefs",
            model_id="gefs",
            name="Global Ensemble Forecast System",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
        )
        version_gfs = ModelVersion(id="version_gfs_v1", model_id="gfs", version_string="v1.0")
        version_gefs = ModelVersion(id="version_gefs_v1", model_id="gefs", version_string="v1.0")
        run_gfs_00 = ModelRun(
            id="run_2026072100_gfs",
            model_version_id="version_gfs_v1",
            cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
            status="ready",
            zarr_store_path="s3://weather-data/gfs/2026-07-21/00Z",
        )
        run_gfs_12 = ModelRun(
            id="run_2026072112_gfs",
            model_version_id="version_gfs_v1",
            cycle_time=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
            status="processing",
            zarr_store_path=None,
        )
        run_gefs_00 = ModelRun(
            id="run_2026072100_gefs",
            model_version_id="version_gefs_v1",
            cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
            status="ready",
            zarr_store_path="s3://weather-data/gefs/2026-07-21/00Z",
        )
        temperature = ForecastVariable(
            id="var_temperature_2m",
            variable_code="temperature_2m",
            name="2-Meter Temperature",
            unit="°C",
        )
        precipitation = ForecastVariable(
            id="var_precipitation_rate",
            variable_code="precipitation_rate",
            name="Precipitation Rate",
            unit="mm/h",
        )
        global_grid = ForecastGrid(
            id="grid_global_025deg",
            grid_code="global_025deg",
            name="Global 0.25 Degree Grid",
            resolution_km=25.0,
        )
        downscaled_grid = ForecastGrid(
            id="grid_downscaled_3km",
            grid_code="downscaled_3km",
            name="AI Downscaled Local Grid",
            resolution_km=3.0,
        )
        session.add_all(
            [
                noaa,
                gfs,
                gefs,
                version_gfs,
                version_gefs,
                run_gfs_00,
                run_gfs_12,
                run_gefs_00,
                temperature,
                precipitation,
                global_grid,
                downscaled_grid,
            ]
        )
        session.commit()
    yield


@pytest.fixture(scope="session")
def client(migrated_db, seed_data):
    """TestClient bound to the test database via a ``get_db`` override."""

    def override_get_db():
        db = Session(migrated_db)
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)
