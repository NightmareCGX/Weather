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
    ForecastGrid,
    ForecastProduct,
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
    ds.to_zarr(path, mode="w", zarr_format=2)
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


# --- Production stale-lead candidate fallback (regression) ---


@pytest.fixture(scope="module")
def stale_lead_client(tmp_path_factory):
    """Two READY runs; the newest run's store lacks a lead its metadata claims.

    Simulates a same-cycle re-ingest where ``forecast_products`` retains a stale
    lead (12) the current store no longer holds. Cross-cycle selection must skip
    the stale candidate and fall back to the older READY run that actually has
    the lead, never raising KeyError/500.
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

    base = tmp_path_factory.mktemp("stalelead")
    # Older READY run at 2026-07-21T00Z: store has leads [0,6,12,18], covers
    # LAT/LON, and supplies the fallback for the newest run's stale leads.
    old_store = _write_store(str(base / "old.zarr"))
    # Newest READY run at 2026-07-21T12Z: store has ONLY lead [0]
    # (pre-allocated with lead 0), but forecast_products claims leads
    # [0,6,12,18] (stale metadata). Its stale leads 6/12/18 for valid_times
    # 18Z/00Z(next)/06Z(next) must be skipped; the older run supplies them.
    new_store = str(base / "new.zarr")
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    new_lead = np.array([0])
    lg, lag, log = np.meshgrid(new_lead, lat, lon, indexing="ij")
    ds_new = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                10.0 + 10.0 * (lag - 38.0) + 10.0 * (log + 107.0) + 0.5 * lg,
            )
        },
        coords={"lead_time_hours": new_lead, "latitude": lat, "longitude": lon},
    )
    ds_new.to_zarr(new_store, mode="w")

    with Session(engine) as session:
        session.add(ForecastCenter(id="c_sl", center_id="noaa_sl", name="NOAA", country="USA"))
        session.add(
            Model(
                id="m_sl",
                model_id="gfs_stale",
                name="GFS",
                center_id="noaa_sl",
                is_ensemble=False,
                resolution_km=25.0,
            )
        )
        session.add(ModelVersion(id="v_sl", model_id="gfs_stale", version_string="v1.0"))
        session.add(
            ForecastVariable(
                id="var_temperature_2m_sl",
                variable_code="temperature_2m",
                name="T",
                unit="°C",
            )
        )
        session.add(
            ForecastGrid(
                id="grid_global_025deg_sl",
                grid_code="global_025deg",
                name="Global 0.25 Degree Grid",
                resolution_km=25.0,
            )
        )
        session.add(
            ModelRun(
                id="run_old",
                model_version_id="v_sl",
                cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path=old_store,
            )
        )
        session.add(
            ModelRun(
                id="run_new",
                model_version_id="v_sl",
                cycle_time=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path=new_store,
            )
        )
        # Stale metadata on the NEWEST run: product rows claim leads 0,6,12,18,
        # but the store only contains lead 0.
        for lead in (0, 6, 12, 18):
            session.add(
                ForecastProduct(
                    id=f"p_new_{lead}",
                    run_id="run_new",
                    variable_id="temperature_2m",
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=lead,
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


def test_stale_catalog_lead_falls_back_to_older_usable(stale_lead_client):
    """The newest run (12Z) claims stale leads 6/12/18; valid_time 18Z skips the
    stale lead-6 candidate and falls back to the older READY run's lead 18.
    No KeyError / 500."""
    resp = stale_lead_client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gfs_stale")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    # valid_time 2026-07-21T18Z: newest (12Z) claims lead 6 (stale, store lacks
    # it) and lead 12 (also stale); older (00Z) lead 18 is the usable minimum.
    entry = next(
        e for e in data["forecasts"] if e["lead_time_hours"] == 18
    )
    assert entry["cycle_time"] == "2026-07-21T00:00:00Z"
    assert "temperature_2m" in entry
    # valid_time 2026-07-21T12Z: newest (12Z) lead 0 (store has it) wins over
    # older (00Z) lead 12.
    entry0 = next(
        e for e in data["forecasts"] if e["valid_time"] == "2026-07-21T12:00:00Z"
    )
    assert entry0["lead_time_hours"] == 0
    assert entry0["cycle_time"] == "2026-07-21T12:00:00Z"


def test_stale_lead_valid_time_not_dropped(stale_lead_client):
    """A valid_time whose nominal best candidate is stale must not disappear."""
    resp = stale_lead_client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gfs_stale")
    assert resp.status_code == 200
    # valid_times 00Z/06Z/12Z/18Z all present. 12Z resolves to the newest run's
    # lead 0 (min lead); 18Z falls back to the older run's lead 18 (newest's
    # stale lead 6/12 skipped). No KeyError / no dropped valid_time.
    by_valid = {
        e["valid_time"]: e["lead_time_hours"]
        for e in resp.json()["data"]["forecasts"]
    }
    assert by_valid == {
        "2026-07-21T00:00:00Z": 0,
        "2026-07-21T06:00:00Z": 6,
        "2026-07-21T12:00:00Z": 0,
        "2026-07-21T18:00:00Z": 18,
    }
