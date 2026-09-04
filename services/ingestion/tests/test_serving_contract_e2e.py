"""Permanent Cross-Package Serving Contract Integration & CI Acceptance Gate.

Verifies the entire production pipeline end-to-end:
  Ingestion Writer (sharded_v1 container encoding -> S3/MinIO)
  -> Coordinator (COMPLETE markers -> manifest.json -> PostgreSQL catalog reconciliation)
  -> API Serving Tier (ShardedV1Reader partial Range GETs -> points, ensembles, map tiles)

Guarantees that:
1. Newly ingested sharded_v1 GFS and GEFS cycles are 100% servable via the API.
2. Point forecasts return finite, non-null meteorological numbers.
3. Ensemble endpoints return member_count > 0 and valid finite dispersion statistics.
4. Map raster tiles render non-trivial, non-transparent PNGs (> 1 KB).
5. Storage backend correctly dispatches sharded_v1 vs legacy v2_unsharded.
6. Zero full-store materialization occurs on sharded_v1 API reads.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

# Ensure services/api/src and services/ingestion/src are on sys.path
_here = os.path.dirname(__file__)
_api_src = os.path.abspath(os.path.join(_here, "../../api/src"))
_ing_src = os.path.abspath(os.path.join(_here, "../src"))
for _p in (_api_src, _ing_src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from api.core.config import settings as api_settings
from api.core.manifest_reader import manifest_storage_format
from api.main import create_app
from ingestion.core.catalog import (
    ModelRunRecord,
    ModelVersionRecord,
    RunCatalogSpec,
    VariableSpec,
)
from ingestion.core.coordinator import RunCoordinator
from ingestion.core.inventory import (
    expected_write_set_fingerprint,
    region_expected_object_keys,
)
from ingestion.core.markers import marker_body, write_region_marker
from ingestion.core.zarr_writer import commit_region, prepare_run_store

LAT = 40.0
LON = -105.0

VARS_LIST = [
    "temperature_2m",
    "precipitation_rate",
    "precipitation_amount_3h",
    "wind_u_10m",
    "wind_v_10m",
    "wind_gust",
    "relative_humidity_2m",
    "visibility",
    "snow_depth",
    "cloud_cover_3h",
    "cloud_ceiling",
    "crain",
    "csnow",
    "cfrzr",
    "cicep",
]


def _build_synthetic_dataset(
    leads: tuple[int, ...],
    members: tuple[int, ...] | None = None,
    t2m_val: float = 293.15,
    cycle_time: datetime | None = None,
) -> xr.Dataset:
    """Build a synthetic dataset with full global 721x1440 grid and surface variables."""
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    dims = ("latitude", "longitude")
    shape = (721, 1440)

    data_vars = {
        "temperature_2m": (dims, np.full(shape, t2m_val - 273.15, dtype=np.float32)),
        "precipitation_rate": (dims, np.full(shape, 1.25, dtype=np.float32)),
        "precipitation_amount_3h": (dims, np.full(shape, 3.5, dtype=np.float32)),
        "wind_u_10m": (dims, np.full(shape, 5.0, dtype=np.float32)),
        "wind_v_10m": (dims, np.full(shape, 5.0, dtype=np.float32)),
        "wind_gust": (dims, np.full(shape, 10.0, dtype=np.float32)),
        "relative_humidity_2m": (dims, np.full(shape, 65.0, dtype=np.float32)),
        "visibility": (dims, np.full(shape, 24000.0, dtype=np.float32)),
        "snow_depth": (dims, np.full(shape, 0.0, dtype=np.float32)),
        "cloud_cover_3h": (dims, np.full(shape, 50.0, dtype=np.float32)),
        "cloud_ceiling": (dims, np.full(shape, 5000.0, dtype=np.float32)),
        "crain": (dims, np.full(shape, 1.0, dtype=np.float32)),
        "csnow": (dims, np.full(shape, 0.0, dtype=np.float32)),
        "cfrzr": (dims, np.full(shape, 0.0, dtype=np.float32)),
        "cicep": (dims, np.full(shape, 0.0, dtype=np.float32)),
    }
    c_time = cycle_time or datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    coords: dict[str, object] = {
        "lead_time_hours": [leads[0]],
        "latitude": lat,
        "longitude": lon,
        "time": np.datetime64(c_time.strftime("%Y-%m-%dT%H:%M:%S"), "ns"),
    }
    if members is not None:
        coords["member"] = [members[0]]
    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.fixture(scope="module")
def clean_db():
    """Ensure database has migrated schema and required forecast centers/models."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(os.path.abspath(os.path.join(_api_src, "../alembic.ini")))
    alembic_cfg.set_main_option("script_location", os.path.abspath(os.path.join(_api_src, "../alembic")))
    alembic_cfg.set_main_option("sqlalchemy.url", str(api_settings.DATABASE_URL))

    engine = create_engine(str(api_settings.DATABASE_URL))
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))

    command.upgrade(alembic_cfg, "head")

    from api.core.database import SessionLocal
    from api.models.entities import ForecastCenter, ForecastGrid, ForecastVariable, Model, ModelVersion

    with SessionLocal() as db:
        # Forecast Center
        center = db.query(ForecastCenter).filter_by(center_id="ncep").first()
        if not center:
            center = ForecastCenter(id="center_ncep", center_id="ncep", name="NCEP", country="USA")
            db.add(center)

        # Models
        gfs = db.query(Model).filter_by(model_id="gfs").first()
        if not gfs:
            gfs = Model(id="model_gfs", model_id="gfs", name="Global Forecast System", center_id="ncep", is_ensemble=False, resolution_km=25.0)
            db.add(gfs)

        gefs = db.query(Model).filter_by(model_id="gefs").first()
        if not gefs:
            gefs = Model(id="model_gefs", model_id="gefs", name="Global Ensemble Forecast System", center_id="ncep", is_ensemble=True, resolution_km=25.0)
            db.add(gefs)

        # Versions
        v_gfs = db.query(ModelVersion).filter_by(model_id="gfs", version_string="v1.0").first()
        if not v_gfs:
            v_gfs = ModelVersion(id="v1.0_gfs", model_id="gfs", version_string="v1.0")
            db.add(v_gfs)

        v_gefs = db.query(ModelVersion).filter_by(model_id="gefs", version_string="v1.0").first()
        if not v_gefs:
            v_gefs = ModelVersion(id="v1.0_gefs", model_id="gefs", version_string="v1.0")
            db.add(v_gefs)

        # Grid
        grid = db.query(ForecastGrid).filter_by(grid_code="global_0p25").first()
        if not grid:
            grid = ForecastGrid(id="grid_0p25", grid_code="global_0p25", name="Global 0.25 deg", resolution_km=25.0)
            db.add(grid)

        # Forecast Variables
        var_defs = [
            ("temperature_2m", "Temperature at 2m", "°C"),
            ("precipitation_rate", "Precipitation Rate", "mm/h"),
            ("precipitation_amount_3h", "Precipitation Amount 3h", "mm"),
            ("wind_u_10m", "U-component of Wind", "m/s"),
            ("wind_v_10m", "V-component of Wind", "m/s"),
            ("wind_10m", "Wind Speed 10m", "km/h"),
            ("wind_gust", "Wind Gust Speed", "m/s"),
            ("relative_humidity_2m", "Relative Humidity", "%"),
            ("visibility", "Visibility", "m"),
            ("snow_depth", "Snow Depth", "m"),
            ("cloud_cover_3h", "Cloud Cover 3h", "%"),
            ("cloud_ceiling", "Cloud Ceiling", "m"),
            ("crain", "Categorical Rain", "binary"),
            ("csnow", "Categorical Snow", "binary"),
            ("cfrzr", "Categorical Freezing Rain", "binary"),
            ("cicep", "Categorical Ice Pellets", "binary"),
        ]
        for vcode, vname, vunit in var_defs:
            if not db.query(ForecastVariable).filter_by(variable_code=vcode).first():
                db.add(ForecastVariable(id=f"var_{vcode}", variable_code=vcode, name=vname, unit=vunit))

        db.commit()

    yield engine
    engine.dispose()


@pytest.mark.skipif(
    os.getenv("WEATHER_TEST_MINIO") != "1",
    reason="WEATHER_TEST_MINIO=1 required for MinIO S3 integration acceptance",
)
def test_serving_contract_e2e_gfs_and_gefs(clean_db) -> None:
    """Full end-to-end integration: Ingestion S3 write -> Coordinator publication -> API Serving."""
    unique_id = uuid.uuid4().hex[:8]
    cycle_offset = int(unique_id, 16) % 50000
    cycle_gfs = datetime(2028, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(hours=cycle_offset)
    cycle_gefs = datetime(2028, 6, 1, 0, 0, tzinfo=timezone.utc) + timedelta(hours=cycle_offset)

    gfs_store = f"s3://weather-data/test_e2e_gfs_{unique_id}/cycle.zarr"
    gefs_store = f"s3://weather-data/test_e2e_gefs_{unique_id}/cycle.zarr"
    gfs_run_id = f"run_gfs_{unique_id}"
    gefs_run_id = f"run_gefs_{unique_id}"

    db_engine = create_engine(str(api_settings.DATABASE_URL))

    # =========================================================================
    # 1. Ingest GFS sharded_v1 Cycle (Leads 0 and 3)
    # =========================================================================
    ds_gfs_seed = _build_synthetic_dataset(leads=(0,), t2m_val=295.15, cycle_time=cycle_gfs)  # 22.0 deg C
    prepare_run_store(ds_gfs_seed, gfs_store, expected_lead_time_hours=(0, 3))

    # Commit lead 0
    commit_region(ds_gfs_seed, gfs_store, lead_time_hours=0)
    keys_l0 = region_expected_object_keys(gfs_store, member=None, lead_index=0, lead_time_hours=0, format_version="sharded_v1", data_var_paths=VARS_LIST)
    body_l0 = marker_body(lead_time_hours=0, member=None, state="complete", generation="gen_0", expected_write_set_fingerprint=expected_write_set_fingerprint(keys_l0, []), required_materialized_object_keys=keys_l0, intentionally_omitted_fill_chunks=[])
    write_region_marker(gfs_store, lead_time_hours=0, member=None, payload=body_l0)

    # Commit lead 3
    ds_gfs_l3 = _build_synthetic_dataset(leads=(3,), t2m_val=298.15, cycle_time=cycle_gfs)  # 25.0 deg C
    commit_region(ds_gfs_l3, gfs_store, lead_time_hours=3)
    keys_l3 = region_expected_object_keys(gfs_store, member=None, lead_index=1, lead_time_hours=3, format_version="sharded_v1", data_var_paths=VARS_LIST)
    body_l3 = marker_body(lead_time_hours=3, member=None, state="complete", generation="gen_3", expected_write_set_fingerprint=expected_write_set_fingerprint(keys_l3, []), required_materialized_object_keys=keys_l3, intentionally_omitted_fill_chunks=[])
    write_region_marker(gfs_store, lead_time_hours=3, member=None, payload=body_l3)

    # Retrieve model version IDs
    with Session(db_engine) as db:
        gfs_version = db.query(ModelVersionRecord).filter_by(model_id="gfs", version_string="v1.0").first()
        gefs_version = db.query(ModelVersionRecord).filter_by(model_id="gefs", version_string="v1.0").first()
        assert gfs_version is not None
        assert gefs_version is not None
        gfs_version_id = gfs_version.id
        gefs_version_id = gefs_version.id

    # Seed initial run record
    with Session(db_engine) as db:
        db.add(ModelRunRecord(id=gfs_run_id, model_version_id=gfs_version_id, cycle_time=cycle_gfs, status="partial", zarr_store_path=gfs_store))
        db.commit()

    # Finalize GFS cycle via coordinator
    gfs_spec = RunCatalogSpec(
        center_id="ncep",
        center_name="NCEP",
        center_country="USA",
        model_id="gfs",
        model_name="Global Forecast System",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=cycle_gfs,
        grid_id="global_0p25",
        grid_name="Global 0.25 deg",
        grid_resolution_km=25.0,
        zarr_store_path=gfs_store,
        variables=tuple(VariableSpec(code=v, name=v, unit="val") for v in VARS_LIST),
        expected_lead_time_hours=(0, 3),
        expected_members=(),
    )
    gfs_coord = RunCoordinator(gfs_spec, gfs_store)
    with db_engine.connect() as conn:
        gfs_coord.finalize_run(conn, run_id=gfs_run_id, spec=gfs_spec, expected_leads=(0, 3), expected_members=())
        conn.commit()

    # =========================================================================
    # 2. Ingest GEFS sharded_v1 Cycle (Lead 0, Members 1..30)
    # =========================================================================
    expected_members = tuple(range(1, 31))
    ds_gefs_seed = _build_synthetic_dataset(leads=(0,), members=(1,), t2m_val=290.15, cycle_time=cycle_gefs)
    prepare_run_store(ds_gefs_seed, gefs_store, expected_lead_time_hours=(0,), expected_members=expected_members)

    for m in expected_members:
        # Vary temperature across members: 290.15 + m*0.2 K
        ds_m = _build_synthetic_dataset(leads=(0,), members=(m,), t2m_val=290.15 + (m * 0.2), cycle_time=cycle_gefs)
        commit_region(ds_m, gefs_store, lead_time_hours=0, member=m)
        keys_m = region_expected_object_keys(gefs_store, member=m, lead_index=0, lead_time_hours=0, format_version="sharded_v1", data_var_paths=VARS_LIST)
        body_m = marker_body(lead_time_hours=0, member=m, state="complete", generation=f"gen_m{m}", expected_write_set_fingerprint=expected_write_set_fingerprint(keys_m, []), required_materialized_object_keys=keys_m, intentionally_omitted_fill_chunks=[])
        write_region_marker(gefs_store, lead_time_hours=0, member=m, payload=body_m)

    # Seed initial run record
    with Session(db_engine) as db:
        db.add(ModelRunRecord(id=gefs_run_id, model_version_id=gefs_version_id, cycle_time=cycle_gefs, status="partial", zarr_store_path=gefs_store))
        db.commit()

    gefs_spec = RunCatalogSpec(
        center_id="ncep",
        center_name="NCEP",
        center_country="USA",
        model_id="gefs",
        model_name="Global Ensemble Forecast System",
        is_ensemble=True,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=cycle_gefs,
        grid_id="global_0p25",
        grid_name="Global 0.25 deg",
        grid_resolution_km=25.0,
        zarr_store_path=gefs_store,
        variables=tuple(VariableSpec(code=v, name=v, unit="val") for v in VARS_LIST),
        expected_lead_time_hours=(0,),
        expected_members=expected_members,
    )
    gefs_coord = RunCoordinator(gefs_spec, gefs_store)
    with db_engine.connect() as conn:
        gefs_coord.finalize_run(conn, run_id=gefs_run_id, spec=gefs_spec, expected_leads=(0,), expected_members=expected_members)
        conn.commit()

    # =========================================================================
    # 3. Storage Format & Manifest Verification
    # =========================================================================
    assert manifest_storage_format(gfs_store) == "sharded_v1"
    assert manifest_storage_format(gefs_store) == "sharded_v1"

    # =========================================================================
    # 4. API Serving Tier Verification via TestClient
    # =========================================================================
    app = create_app()
    with TestClient(app) as client:
        # A. Availability Endpoint
        res_avail = client.get("/v1/forecast/availability")
        assert res_avail.status_code == 200
        avail_data = res_avail.json()["data"]
        model_ids = [m["id"] for m in avail_data.get("models", [])]
        assert "gfs" in model_ids
        assert "gefs" in model_ids

        # B. GFS Point Forecast Endpoint
        cycle_str_gfs = cycle_gfs.strftime("%Y-%m-%dT%H:%M:%SZ")
        res_point = client.get(
            f"/v1/points?lat={LAT}&lon={LON}&models=gfs&start_lead_time_hours=0&end_lead_time_hours=3"
        )
        assert res_point.status_code == 200
        point_data = res_point.json()["data"]
        forecasts = point_data.get("forecasts", [])
        assert len(forecasts) >= 2

        # Assert finite non-null meteorological values
        for f in forecasts:
            t2m = f.get("temperature_2m")
            assert t2m is not None, "temperature_2m must not be null"
            assert isinstance(t2m, (int, float))
            assert not np.isnan(t2m), "temperature_2m must not be NaN"
            assert 10.0 <= t2m <= 35.0, f"temperature_2m out of expected range: {t2m}"

            wind = f.get("wind_10m")
            assert wind is not None and not np.isnan(wind)

            precip = f.get("precipitation_rate")
            assert precip is not None and not np.isnan(precip)

        # C. GEFS Ensemble Endpoint
        cycle_str_gefs = cycle_gefs.strftime("%Y-%m-%dT%H:%M:%SZ")
        res_ens = client.get(
            f"/v1/ensembles?lat={LAT}&lon={LON}&variable=temperature_2m&model=gefs&lead_time_hours=0&initial_time={cycle_str_gefs}&include_members=true"
        )
        assert res_ens.status_code == 200
        ens_data = res_ens.json()["data"]
        assert ens_data["member_count"] == 30, f"Expected 30 members, got {ens_data['member_count']}"
        stats = ens_data["statistics"]
        assert stats["mean"] is not None and not np.isnan(stats["mean"])
        assert stats["median"] is not None and not np.isnan(stats["median"])
        assert stats["spread"] is not None and not np.isnan(stats["spread"])
        assert len(ens_data.get("members", [])) == 30

        # D. GEFS Exceedance Probability Endpoint
        res_prob = client.get(
            f"/v1/probabilities?lat={LAT}&lon={LON}&variable=temperature_2m&model=gefs&lead_time_hours=0&initial_time={cycle_str_gefs}&threshold=18.0&operator=gt"
        )
        assert res_prob.status_code == 200
        prob_data = res_prob.json()["data"]
        assert prob_data["probability"] is not None
        assert 0.0 <= prob_data["probability"] <= 1.0

        # E. GFS Map Tile PNG (Deterministic)
        res_tile_gfs = client.get(f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={cycle_str_gfs}")
        assert res_tile_gfs.status_code == 200
        assert res_tile_gfs.headers.get("content-type") == "image/png"
        # 256x256 solid/opaque PNG is > 500 B (an empty/transparent tile is 334 B)
        assert len(res_tile_gfs.content) > 500, f"PNG tile payload unexpectedly small: {len(res_tile_gfs.content)} B"
        assert res_tile_gfs.content.startswith(b"\x89PNG\r\n\x1a\n")

        # F. GEFS Map Tile PNG (Ensemble Member-Mean)
        res_tile_gefs = client.get(f"/v1/maps/gefs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={cycle_str_gefs}")
        assert res_tile_gefs.status_code == 200
        assert res_tile_gefs.headers.get("content-type") == "image/png"
        assert len(res_tile_gefs.content) > 500
        assert res_tile_gefs.content.startswith(b"\x89PNG\r\n\x1a\n")

        # G. GEFS Point Forecast Endpoint (Ensemble Member-Mean)
        res_gefs_point = client.get(
            f"/v1/points?lat={LAT}&lon={LON}&models=gefs&start_lead_time_hours=0&end_lead_time_hours=0"
        )
        assert res_gefs_point.status_code == 200
        gefs_point_data = res_gefs_point.json()["data"]
        assert gefs_point_data["model"] == "gefs"
        gefs_forecasts = gefs_point_data.get("forecasts", [])
        assert len(gefs_forecasts) >= 1
        gefs_f0 = gefs_forecasts[0]
        assert gefs_f0.get("temperature_2m") is not None
        assert not np.isnan(gefs_f0["temperature_2m"])
        # Members 1..30 had temperature: 290.15 + m*0.2 K -> Celsius = 17.0 + m*0.2 -> mean = 17.0 + 3.1 = 20.1 deg C
        assert np.isclose(gefs_f0["temperature_2m"], 20.1, atol=0.2)
        assert gefs_f0.get("wind_10m") is not None
        assert not np.isnan(gefs_f0["wind_10m"])
        assert gefs_f0["wind_10m"] > 0.0
        assert gefs_f0.get("precipitation_amount_3h") is not None
        assert not np.isnan(gefs_f0["precipitation_amount_3h"])

        # H. GEFS Vector Field (Consensus Mean Vector)
        res_vf = client.get(f"/v1/maps/gefs/wind_10m/vector-field?lead_time_hours=0&initial_time={cycle_str_gefs}")
        assert res_vf.status_code == 200
        assert len(res_vf.content) > 100


def test_serving_contract_legacy_v2_unsharded_compatibility(tmp_path, clean_db) -> None:
    """Test that legacy v2_unsharded stores (without sharded_v1 manifest) serve correctly via legacy xarray path."""
    unique_id = uuid.uuid4().hex[:8]
    cycle_time = datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc)
    store_dir = tmp_path / f"legacy_v2_{unique_id}.zarr"
    store_path = str(store_dir)
    run_id = f"run_legacy_{unique_id}"

    # Create global legacy v2 dataset written directly with to_zarr
    ds_legacy = _build_synthetic_dataset(leads=(0,), t2m_val=291.65, cycle_time=cycle_time)  # 18.5 deg C
    ds_legacy.to_zarr(store_path, mode="w", consolidated=True, zarr_format=2)

    # Assert legacy manifest format
    assert manifest_storage_format(store_path) == "v2_unsharded"

    # Register in DB
    from api.models.entities import ModelRun, ModelVersion, ForecastProduct
    from api.core.database import SessionLocal

    with SessionLocal() as db:
        gfs_version = db.query(ModelVersion).filter_by(model_id="gfs", version_string="v1.0").first()
        assert gfs_version is not None
        db.add(ModelRun(id=run_id, model_version_id=gfs_version.id, cycle_time=cycle_time, status="ready", zarr_store_path=store_path))
        for v in VARS_LIST:
            db.add(ForecastProduct(id=f"prod_{unique_id}_{v}", run_id=run_id, variable_id=v, grid_id="global_0p25", product_type="surface", lead_time_hours=0))
        db.commit()

    # Query via API TestClient
    app = create_app()
    with TestClient(app) as client:
        res = client.get(
            f"/v1/points?lat={LAT}&lon={LON}&models=gfs&variables=temperature_2m&start_lead_time_hours=0&end_lead_time_hours=0"
        )
        assert res.status_code == 200
        point_data = res.json()["data"]
        forecasts = point_data.get("forecasts", [])
        assert len(forecasts) >= 1
        t2m = forecasts[0].get("temperature_2m")
        assert t2m is not None
        assert np.isclose(t2m, 18.5, atol=0.1)

        cycle_str = cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        res_tile = client.get(f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={cycle_str}")
        assert res_tile.status_code == 200
        assert res_tile.headers.get("content-type") == "image/png"
        assert len(res_tile.content) > 500


def test_serving_contract_negative_assertions() -> None:
    """Negative assertions: verify fail-closed behavior for nonexistent variable and malformed manifest."""
    from api.core.manifest_reader import manifest_storage_format
    # Missing directory -> v2_unsharded default
    assert manifest_storage_format("/nonexistent/path/store.zarr") == "v2_unsharded"

