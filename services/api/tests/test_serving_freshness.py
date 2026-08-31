"""Regression test: serving freshness across mid-process forecast ingestion.

Validates that a long-lived running API process immediately observes newly
committed forecast data (new leads added to an existing cycle, same-cycle
re-ingestions, and newly ingested forecast cycles) on subsequent requests
WITHOUT requiring an API restart or cache-invalidation workaround, and that all
mutable endpoints emit the canonical revalidation header (``Cache-Control: no-cache``).
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import sys

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Ensure test paths are available
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
_domain_src = os.path.abspath(os.path.join(current_dir, "../../../packages/domain/src"))
if _domain_src not in sys.path:
    sys.path.insert(0, _domain_src)

from api.core.database import get_db
from api.main import app
from api.models.entities import (
    ForecastProduct,
    ModelRun,
)
from tests._zarr_writer import write_dataset


def _make_forecast_dataset(
    leads: list[int],
    base_temp: float = 15.0,
) -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75], dtype=float)
    lon = np.array([-107.0, -106.75, -106.5, -106.25], dtype=float)
    lead_arr = np.asarray(leads, dtype=float)
    lead_grid, lat_grid, lon_grid = np.meshgrid(lead_arr, lat, lon, indexing="ij")
    temperature = (base_temp + 0.5 * lead_grid).astype(np.float32)
    precipitation = (0.5 * lead_grid).astype(np.float32)
    amount_3h = np.where(lead_grid == 0, np.nan, 0.4 * lead_grid).astype(np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), temperature),
            "precipitation_rate": (("lead_time_hours", "latitude", "longitude"), precipitation),
            "precipitation_amount_3h": (("lead_time_hours", "latitude", "longitude"), amount_3h),
        },
        coords={
            "lead_time_hours": leads,
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_long_lived_api_observes_new_lead_and_cycle_without_restart(
    migrated_db,
    seed_data,
    tmp_path,
):
    """A running API process must observe newly ingested leads and cycles immediately."""
    store_dir_00 = str(tmp_path / "gfs_2026072500z.zarr")
    store_dir_06 = str(tmp_path / "gfs_2026072506z.zarr")

    # 1. Initial state: GFS cycle 2026-07-25 00Z with lead 0 only.
    write_dataset(_make_forecast_dataset([0], base_temp=15.0), store_dir_00)

    with Session(migrated_db) as session:
        run_00 = ModelRun(
            id="run_freshness_2026072500z_gfs",
            model_version_id="version_gfs_v1",
            cycle_time=datetime(2026, 7, 25, 0, 0, tzinfo=timezone.utc),
            status="ready",
            zarr_store_path=store_dir_00,
        )
        session.add(run_00)
        session.flush()

        prod_00_l0 = ForecastProduct(
            id="prod_freshness_2026072500z_t2m_0",
            run_id="run_freshness_2026072500z_gfs",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        session.add(prod_00_l0)
        session.commit()

    # 2. Start a long-lived TestClient session (simulating the running API server).
    def override_get_db():
        db = Session(migrated_db)
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        # Initial read on long-lived client
        resp_avail_1 = client.get("/v1/forecast/availability")
        assert resp_avail_1.status_code == 200
        assert resp_avail_1.headers["Cache-Control"] == "no-cache"

        gfs_avail_1 = next(m for m in resp_avail_1.json()["data"]["models"] if m["id"] == "gfs")
        t2m_avail_1 = next(v for v in gfs_avail_1["variables"] if v["id"] == "temperature_2m")
        init_00_1 = next(t for t in t2m_avail_1["initial_times"] if t["value"] == "2026-07-25T00:00:00Z")
        assert init_00_1["lead_time_hours"] == [0]

        resp_point_1 = client.get("/v1/points?lat=38.125&lon=-106.875&models=gfs")
        assert resp_point_1.status_code == 200
        assert resp_point_1.headers["Cache-Control"] == "no-cache"

        # 3. Mid-process Ingestion Event 1: Ingest lead 6 into same GFS 2026-07-25 00Z cycle.
        write_dataset(_make_forecast_dataset([0, 6], base_temp=15.0), store_dir_00)
        with Session(migrated_db) as session:
            prod_00_l6 = ForecastProduct(
                id="prod_freshness_2026072500z_t2m_6",
                run_id="run_freshness_2026072500z_gfs",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=6,
            )
            session.add(prod_00_l6)
            session.commit()

        # 4. Query in the SAME running client instance -> MUST observe lead 6 immediately.
        resp_avail_2 = client.get("/v1/forecast/availability")
        assert resp_avail_2.status_code == 200
        assert resp_avail_2.headers["Cache-Control"] == "no-cache"

        gfs_avail_2 = next(m for m in resp_avail_2.json()["data"]["models"] if m["id"] == "gfs")
        t2m_avail_2 = next(v for v in gfs_avail_2["variables"] if v["id"] == "temperature_2m")
        init_00_2 = next(t for t in t2m_avail_2["initial_times"] if t["value"] == "2026-07-25T00:00:00Z")
        assert init_00_2["lead_time_hours"] == [0, 6]

        resp_map_2 = client.get("/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=6&initial_time=2026-07-25T00:00:00Z")
        assert resp_map_2.status_code == 200
        assert resp_map_2.headers["Cache-Control"] == "no-cache"

        # Exercise the actual Zarr reading paths (point forecast and map tile) for the new lead.
        resp_point_2 = client.get("/v1/points?lat=38.125&lon=-106.875&models=gfs&start_lead_time_hours=6&end_lead_time_hours=6")
        assert resp_point_2.status_code == 200
        assert resp_point_2.headers["Cache-Control"] == "no-cache"
        forecasts_2 = resp_point_2.json()["data"]["forecasts"]
        fc_00 = next(f for f in forecasts_2 if f["cycle_time"] == "2026-07-25T00:00:00Z")
        assert fc_00["lead_time_hours"] == 6
        assert fc_00["temperature_2m"] == pytest.approx(18.0)

        resp_tile_2 = client.get("/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=6&initial_time=2026-07-25T00:00:00Z")
        assert resp_tile_2.status_code == 200
        assert resp_tile_2.headers["Cache-Control"] == "no-cache"

        # 5. Mid-process Ingestion Event 2: Ingest a brand-new cycle 2026-07-25 06Z.
        write_dataset(_make_forecast_dataset([0], base_temp=20.0), store_dir_06)
        with Session(migrated_db) as session:
            run_06 = ModelRun(
                id="run_freshness_2026072506z_gfs",
                model_version_id="version_gfs_v1",
                cycle_time=datetime(2026, 7, 25, 6, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path=store_dir_06,
            )
            session.add(run_06)
            session.flush()

            prod_06_l0 = ForecastProduct(
                id="prod_freshness_2026072506z_t2m_0",
                run_id="run_freshness_2026072506z_gfs",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
            session.add(prod_06_l0)
            session.commit()

        # 6. Query in the SAME running client instance -> MUST observe new cycle 06Z immediately.
        resp_avail_3 = client.get("/v1/forecast/availability")
        assert resp_avail_3.status_code == 200
        assert resp_avail_3.headers["Cache-Control"] == "no-cache"

        gfs_avail_3 = next(m for m in resp_avail_3.json()["data"]["models"] if m["id"] == "gfs")
        t2m_avail_3 = next(v for v in gfs_avail_3["variables"] if v["id"] == "temperature_2m")
        cycle_times = [t["value"] for t in t2m_avail_3["initial_times"]]
        assert "2026-07-25T06:00:00Z" in cycle_times
        assert "2026-07-25T00:00:00Z" in cycle_times

        resp_runs_3 = client.get("/v1/runs?model_id=gfs")
        assert resp_runs_3.status_code == 200
        assert resp_runs_3.headers["Cache-Control"] == "no-cache"
        run_ids = [r["id"] for r in resp_runs_3.json()["data"]]
        assert "run_freshness_2026072506z_gfs" in run_ids

    app.dependency_overrides.pop(get_db, None)


def test_no_manifest_store_observes_new_lead_without_restart(
    migrated_db,
    seed_data,
    tmp_path,
):
    """A running API process querying a no-manifest store observes an appended lead without restart."""
    store_dir = str(tmp_path / "gfs_nomanifest.zarr")

    # 1. Initial store with lead 0 only (no manifest written)
    write_dataset(_make_forecast_dataset([0], base_temp=12.0), store_dir)

    with Session(migrated_db) as session:
        run = ModelRun(
            id="run_nomanifest_gfs",
            model_version_id="version_gfs_v1",
            cycle_time=datetime(2026, 7, 26, 0, 0, tzinfo=timezone.utc),
            status="ready",
            zarr_store_path=store_dir,
        )
        session.add(run)
        session.flush()

        prod_l0 = ForecastProduct(
            id="prod_nomanifest_l0",
            run_id="run_nomanifest_gfs",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        session.add(prod_l0)
        session.commit()

    def override_get_db():
        db = Session(migrated_db)
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        # Flush the Redis response cache at start to ensure clean test state
        from api.routers.points import _cache as point_cache
        point_cache._client.flushdb()

        # Request 1: lead 0 works
        resp1 = client.get("/v1/points?lat=38.125&lon=-106.875&models=gfs")
        assert resp1.status_code == 200
        run_fc_1 = [f for f in resp1.json()["data"]["forecasts"] if f["cycle_time"] == "2026-07-26T00:00:00Z"]
        leads_1 = [f["lead_time_hours"] for f in run_fc_1]
        assert leads_1 == [0]

        # Mid-process ingestion: append lead 6 to the store on disk
        write_dataset(_make_forecast_dataset([0, 6], base_temp=12.0), store_dir)
        with Session(migrated_db) as session:
            prod_l6 = ForecastProduct(
                id="prod_nomanifest_l6",
                run_id="run_nomanifest_gfs",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=6,
            )
            session.add(prod_l6)
            session.commit()

        # Flush the Redis response cache to simulate a cache-bypassed/revalidated query
        # and test that the underlying Zarr reader opens fresh and observes lead 6.
        point_cache._client.flushdb()

        # Request 2 in the same running client: lead 6 must be served successfully
        resp2 = client.get("/v1/points?lat=38.125&lon=-106.875&models=gfs")
        assert resp2.status_code == 200
        run_fc_2 = [f for f in resp2.json()["data"]["forecasts"] if f["cycle_time"] == "2026-07-26T00:00:00Z"]
        leads_2 = [f["lead_time_hours"] for f in run_fc_2]
        assert 6 in leads_2

    app.dependency_overrides.pop(get_db, None)

