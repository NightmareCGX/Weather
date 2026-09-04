"""Regression test: the API falls back to the next-newest readable run.

When the newest ``ready`` run's Zarr store cannot be read, point/ensemble/
probability requests must fall through to the next-newest readable run
instead of failing, so a single corrupted store cannot take down a model.

The module is self-contained: it seeds its own models/runs/stores and binds a
TestClient, leaving the shared ``conftest`` fixtures untouched.
"""

import os
import sys
from datetime import datetime, timezone

# Ensure services/api/src is on sys.path for test execution (mirrors conftest).
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import pytest
import xarray as xr
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.main import app
from api.models.entities import (
    ForecastCenter,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)

LAT = 38.125
LON = -106.875


@pytest.fixture(scope="module")
def fallback_client(tmp_path_factory):
    """A migrated schema with a valid older run and a broken newest run."""
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
            "PostgreSQL test instance not running or reachable; skipping fallback test."
        )

    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))

    command.upgrade(alembic_cfg, "head")

    # A valid full-cycle store for the OLDER run.
    valid_store = str(tmp_path_factory.mktemp("fallback") / "valid.zarr")
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    lead = np.array([0, 6, 12, 18])
    lead_grid, lat_grid, lon_grid = np.meshgrid(lead, lat, lon, indexing="ij")
    ds = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                10.0 + 10.0 * (lat_grid - 38.0) + 10.0 * (lon_grid + 107.0) + 0.5 * lead_grid,
            )
        },
        coords={
            "lead_time_hours": lead,
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds.to_zarr(valid_store, mode="w", zarr_format=2)

    with Session(engine) as session:
        session.add(
            ForecastCenter(id="center_noaa", center_id="noaa", name="NOAA", country="USA")
        )
        session.add(
            Model(
                id="model_gfs_fallback",
                model_id="gfs_fallback",
                name="Global Forecast System (fallback test)",
                center_id="noaa",
                is_ensemble=False,
                resolution_km=25.0,
            )
        )
        session.add(
            ModelVersion(
                id="version_gfs_fallback_v1",
                model_id="gfs_fallback",
                version_string="v1.0",
            )
        )
        session.add(
            ModelRun(
                id="run_old",
                model_version_id="version_gfs_fallback_v1",
                cycle_time=datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path=valid_store,
            )
        )
        session.add(
            ModelRun(
                id="run_new_broken",
                model_version_id="version_gfs_fallback_v1",
                cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path="C:/definitely/missing/store.zarr",
            )
        )
        session.add(
            ForecastVariable(
                id="var_temperature_2m",
                variable_code="temperature_2m",
                name="2-Meter Temperature",
                unit="°C",
            )
        )
        session.commit()

    def override_get_db():
        db = Session(engine)
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)
        command.downgrade(alembic_cfg, "base")
        engine.dispose()


def test_point_falls_back_to_older_readable_run(fallback_client):
    """A broken newest store must not 500 the request; the older run is used."""
    resp = fallback_client.get(
        f"/v1/points?lat={LAT}&lon={LON}&models=gfs_fallback"
    )
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]
    # The older run's cycle_time (2026-07-20) is the source of truth.
    assert data["generated_at"] == "2026-07-20T00:00:00Z"
    entry = next(e for e in data["forecasts"] if e["lead_time_hours"] == 6)
    assert "temperature_2m" in entry
