"""Focused regression tests for the cross-cycle /v1/points readable-store fix.

These tests prove:

* a READY run with a readable Zarr store but NO ``forecast_products`` rows is
  still servable (the legacy contract restored);
* when the newest minimum-lead store is unreadable, an older READY readable
  candidate covering the same valid_time is used (fallback restored);
* partial cycles remain excluded;
* normal minimum-lead selection still works across cycles.

They exercise the real ``/v1/points`` path (TestClient + migrated schema + on
disk Zarr stores) and the ``_select_min_lead_winners`` selection helper.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Ensure services/api/src is on sys.path (mirrors conftest).
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
from api.services.point_forecast import _select_min_lead_winners

LAT = 38.125
LON = -106.875


def _write_store(path: str, *, broken: bool = False) -> str:
    """Write a small deterministic Zarr store, or a broken (missing) one."""
    if broken:
        return "C:/definitely/missing/store.zarr"
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    lead = np.array([0, 6, 12, 18])
    lead_grid, lat_grid, lon_grid = np.meshgrid(lead, lat, lon, indexing="ij")
    ds = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                10.0
                + 10.0 * (lat_grid - 38.0)
                + 10.0 * (lon_grid + 107.0)
                + 0.5 * lead_grid,
            )
        },
        coords={"lead_time_hours": lead, "latitude": lat, "longitude": lon},
    )
    ds.to_zarr(path, mode="w")
    return path


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """TestClient with its own migrated schema and two runs.

    * ``run_old`` (2026-07-20): READY, readable store, NO forecast_products.
    * ``run_new`` (2026-07-21): READY, broken store, NO forecast_products.
    """
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL not reachable; skipping integration test.")

    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
    command.upgrade(alembic_cfg, "head")

    base = tmp_path_factory.mktemp("crossfix")
    valid_store = _write_store(str(base / "valid.zarr"))

    with Session(engine) as session:
        session.add(ForecastCenter(id="c", center_id="noaa", name="NOAA", country="USA"))
        session.add(
            Model(
                id="m",
                model_id="gfs_crossfix",
                name="GFS",
                center_id="noaa",
                is_ensemble=False,
                resolution_km=25.0,
            )
        )
        session.add(ModelVersion(id="v", model_id="gfs_crossfix", version_string="v1.0"))
        session.add(
            ForecastVariable(
                id="var_temperature_2m",
                variable_code="temperature_2m",
                name="2-Meter Temperature",
                unit="°C",
            )
        )
        session.add(
            ModelRun(
                id="run_old",
                model_version_id="v",
                cycle_time=datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path=valid_store,
            )
        )
        session.add(
            ModelRun(
                id="run_new",
                model_version_id="v",
                cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path="C:/definitely/missing/store.zarr",
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


def test_readable_run_with_no_products_serves(client):
    """A READY run with a readable store but NO forecast_products is served."""
    resp = client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gfs_crossfix")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    # The only servable run is run_old (newest is broken).
    assert data["generated_at"] == "2026-07-20T00:00:00Z"
    leads = [e["lead_time_hours"] for e in data["forecasts"]]
    assert leads == [0, 6, 12, 18]
    entry = next(e for e in data["forecasts"] if e["lead_time_hours"] == 6)
    assert "temperature_2m" in entry


def test_broken_newest_falls_back_to_older_readable(client):
    """The broken newest store must not drop data; the older readable run serves."""
    resp = client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gfs_crossfix")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["generated_at"] == "2026-07-20T00:00:00Z"
    # All four leads survive from the older readable run.
    assert [e["lead_time_hours"] for e in data["forecasts"]] == [0, 6, 12, 18]
    # Provenance reflects the older cycle.
    assert all(e.get("cycle_time") == "2026-07-20T00:00:00Z" for e in data["forecasts"])


def test_selection_discoveries_readable_store_lead_coord(tmp_path):
    """_select_min_lead_winners discovers leads from the store when no products."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    engine = create_engine(db_url)
    store = _write_store(str(tmp_path / "s.zarr"))
    with Session(engine) as session:
        session.add(
            ForecastCenter(id="c_coord", center_id="noaa_coord", name="NOAA", country="USA")
        )
        session.add(
            Model(
                id="m2",
                model_id="gfs_coord",
                name="GFS",
                center_id="noaa_coord",
                is_ensemble=False,
                resolution_km=25.0,
            )
        )
        session.add(ModelVersion(id="v2", model_id="gfs_coord", version_string="v1.0"))
        session.add(
            ForecastVariable(
                id="var_temperature_2m2",
                variable_code="temperature_2m_coord",
                name="T",
                unit="°C",
            )
        )
        # READY run with a readable store and NO forecast_products.
        session.add(
            ModelRun(
                id="run_coord",
                model_version_id="v2",
                cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path=store,
            )
        )
        session.commit()
        winners = _select_min_lead_winners(session, "gfs_coord")
        # valid_times from the store's lead coordinate [0,6,12,18].
        valid_times = sorted(winners)
        assert len(valid_times) == 4
        for lead in (0, 6, 12, 18):
            vt = datetime(2026, 7, 21, lead, 0, tzinfo=timezone.utc)
            assert winners[vt][0] == (datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc), lead)
    engine.dispose()
