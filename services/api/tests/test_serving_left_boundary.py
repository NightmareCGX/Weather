"""Comprehensive test suite for Data Lifecycle V3 Phase 1: Serving Left-Boundary Enforcement.

Validates:
1. Exact Cadence Boundary Advancement:
   - 05:59:59Z -> 03:00:00Z
   - 06:00:00Z -> 06:00:00Z (exact boundary advances anchor immediately)
   - 06:00:01Z -> 06:00:00Z
   - 07:00:00Z -> 06:00:00Z
   - 08:59:59Z -> 06:00:00Z
   - 09:00:00Z -> 09:00:00Z (exact boundary advances anchor immediately)
   - 09:00:01Z -> 09:00:00Z
2. Availability Endpoint (/v1/forecast/availability):
   - Exposes serving_start_valid_time matching the floored boundary.
   - valid_times array contains strictly NO valid times before serving_start_valid_time.
   - Earliest valid time in availability is exactly serving_start_valid_time.
   - Preserves rolling right boundary out to max available valid time across all cycles.
3. Point Forecast Endpoint (/v1/points):
   - Returns series with strictly NO points before serving_start_valid_time.
   - Earliest point in series is exactly serving_start_valid_time.
   - Preserves rolling far-future horizon from older cycles.
   - Overlapping valid times select newest committed cycle.
   - Lead-0 interval variable (precipitation_amount_3h) falls back to previous cycle positive lead at boundary.
   - Phase companions (crain, csnow) remain coupled to the fallback source.
4. Spatial Map Endpoint (/v1/maps):
   - Requests with valid_time < serving_start return HTTP 404.
   - Requests with valid_time == serving_start return HTTP 200.
   - Raster tile rendering with valid_time < serving_start returns HTTP 404.
   - Raster tile rendering with valid_time == serving_start returns HTTP 200.
5. Ensemble Statistics Endpoint (/v1/ensembles):
   - Requests with valid_time < serving_start return HTTP 404.
   - Requests with valid_time == serving_start return HTTP 200.
6. Multi-Day Soak Failure Regression Scenario:
   - Older cycle from Sep 8 covering through Sep 12.
   - Current time = Sep 11 07:00Z -> serving_start = Sep 11 06:00Z.
   - Valid times before Sep 11 06:00Z (Sep 8, 9, 10) are NOT served by availability, points, maps, or ensembles.
   - Right-side Hourly Forecast does not contain stale historical points.
   - Far horizon (Sep 12) remains fully intact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.core.database import Base, get_db
from api.core.time import get_current_time
from api.main import app
from api.models.entities import (
    EnsembleMember,
    EnsembleMemberProduct,
    ForecastCenter,
    ForecastCycleLifecycle,
    ForecastGrid,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from domain.temporal import serving_start_valid_time
from tests._zarr_writer import write_dataset


@pytest.fixture(autouse=True)
def _no_reader_pool(monkeypatch):
    """Force direct bounded read path for standalone SQLite tests."""
    def _direct_read(store_path, selector):
        from api.core.zarr import open_serving_dataset

        with open_serving_dataset(store_path) as ds:
            return selector(ds)

    monkeypatch.setattr(
        "api.core.reader_gate.gated_read_dataset_with_selector",
        _direct_read,
    )
    yield


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


LAT_START = 38.0
LON_START = -107.0
LATS = np.array([38.0, 38.25, 38.5, 38.75], dtype=float)
LONS = np.array([-107.0, -106.75, -106.5, -106.25], dtype=float)


def _build_dataset(
    leads: list[int],
    cycle: datetime,
    base_temp: float = 15.0,
    precip_rate: float = 1.0,
    has_interval: bool = True,
) -> xr.Dataset:
    lead_arr = np.asarray(leads, dtype=float)
    lg, lat_g, lon_g = np.meshgrid(lead_arr, LATS, LONS, indexing="ij")
    temp = (base_temp + 0.1 * lg).astype(np.float32)
    rate = np.full_like(lg, precip_rate, dtype=np.float32)

    data_vars = {
        "temperature_2m": (("lead_time_hours", "latitude", "longitude"), temp),
        "precipitation_rate": (("lead_time_hours", "latitude", "longitude"), rate),
        "wind_u_10m": (("lead_time_hours", "latitude", "longitude"), np.full_like(lg, 5.0, dtype=np.float32)),
        "wind_v_10m": (("lead_time_hours", "latitude", "longitude"), np.full_like(lg, 5.0, dtype=np.float32)),
    }

    if has_interval:
        # Lead 0 is NaN for interval variables
        precip_3h = np.where(lg == 0, np.nan, 2.5).astype(np.float32)
        cloud_3h = np.where(lg == 0, np.nan, 60.0).astype(np.float32)
        crain = np.where(lg == 0, 0.0, 1.0).astype(np.float32)  # Rain active at positive leads
        csnow = np.full_like(lg, 0.0, dtype=np.float32)
        data_vars["precipitation_amount_3h"] = (("lead_time_hours", "latitude", "longitude"), precip_3h)
        data_vars["cloud_cover_3h"] = (("lead_time_hours", "latitude", "longitude"), cloud_3h)
        data_vars["crain"] = (("lead_time_hours", "latitude", "longitude"), crain)
        data_vars["csnow"] = (("lead_time_hours", "latitude", "longitude"), csnow)

    return xr.Dataset(
        data_vars=data_vars,
        coords={"lead_time_hours": leads, "latitude": LATS, "longitude": LONS},
        attrs={"cycle_time": cycle.isoformat(), "model_id": "gfs"},
    )


def _build_ensemble_dataset(leads: list[int], cycle: datetime) -> xr.Dataset:
    members = list(range(1, 31))
    lead_arr = np.asarray(leads, dtype=float)
    mem_arr = np.asarray(members, dtype=float)
    mg, lg, lat_g, lon_g = np.meshgrid(mem_arr, lead_arr, LATS, LONS, indexing="ij")
    temp = (12.0 + 0.1 * mg + 0.05 * lg).astype(np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("member", "lead_time_hours", "latitude", "longitude"), temp),
        },
        coords={"member": members, "lead_time_hours": leads, "latitude": LATS, "longitude": LONS},
        attrs={"cycle_time": cycle.isoformat(), "model_id": "gefs"},
    )


@pytest.fixture
def left_boundary_env(tmp_path: Path):
    """Setup a multi-cycle environment replicating the soak test conditions.

    - Cycle 1 (Older): 2026-09-08 00Z, leads 0, 6, 12, ..., 96 (+96h = Sep 12 00Z).
    - Cycle 2 (Newer partial): 2026-09-11 06Z, leads 0, 3, 6 (+6h = Sep 11 12Z).
    """
    db_path = f"sqlite:///{tmp_path}/boundary_test.db"
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ModelRun.__table__,
        EnsembleMember.__table__,
        EnsembleMemberProduct.__table__,
        ForecastVariable.__table__,
        ForecastGrid.__table__,
        ForecastProduct.__table__,
        ForecastCycleLifecycle.__table__,
    ])

    c_old = _dt(2026, 9, 8, 0)
    c_new = _dt(2026, 9, 11, 6)

    leads_old = list(range(0, 97, 6))  # 0, 6, 12, ..., 96
    leads_new = [0, 3, 6]

    store_old_gfs = str(tmp_path / "gfs_old.zarr")
    store_new_gfs = str(tmp_path / "gfs_new.zarr")
    store_old_gefs = str(tmp_path / "gefs_old.zarr")
    store_new_gefs = str(tmp_path / "gefs_new.zarr")

    write_dataset(_build_dataset(leads_old, c_old, base_temp=10.0), store_old_gfs)
    write_dataset(_build_dataset(leads_new, c_new, base_temp=22.0), store_new_gfs)
    write_dataset(_build_ensemble_dataset(leads_old, c_old), store_old_gefs)
    write_dataset(_build_ensemble_dataset(leads_new, c_new), store_new_gefs)

    with Session(engine) as session:
        session.add(ForecastCenter(id="c_noaa", center_id="noaa", name="NOAA", country="US"))
        session.add(Model(id="m_gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0))
        session.add(Model(id="m_gefs", model_id="gefs", name="GEFS", center_id="noaa", is_ensemble=True, resolution_km=25.0))
        session.add(ModelVersion(id="v_gfs", model_id="gfs", version_string="v1.0"))
        session.add(ModelVersion(id="v_gefs", model_id="gefs", version_string="v1.0"))
        session.add(ForecastGrid(id="g_glob", grid_code="global_025deg", name="Global", resolution_km=25.0))

        for v_code, name, unit in [
            ("temperature_2m", "Temperature", "°C"),
            ("precipitation_rate", "Precipitation Rate", "mm/h"),
            ("precipitation_amount_3h", "3-Hour Precip", "mm"),
            ("cloud_cover_3h", "3-Hour Cloud Cover", "%"),
            ("crain", "Rain Flag", "flag"),
            ("csnow", "Snow Flag", "flag"),
            ("wind_u_10m", "10m U Wind", "m/s"),
            ("wind_v_10m", "10m V Wind", "m/s"),
        ]:
            session.add(ForecastVariable(id=f"v_{v_code}", variable_code=v_code, name=name, unit=unit))

        # Runs
        session.add(ModelRun(id="run_gfs_old", model_version_id="v_gfs", cycle_time=c_old, status="ready", zarr_store_path=store_old_gfs))
        session.add(ModelRun(id="run_gfs_new", model_version_id="v_gfs", cycle_time=c_new, status="ready", zarr_store_path=store_new_gfs))
        session.add(ModelRun(id="run_gefs_old", model_version_id="v_gefs", cycle_time=c_old, status="ready", zarr_store_path=store_old_gefs))
        session.add(ModelRun(id="run_gefs_new", model_version_id="v_gefs", cycle_time=c_new, status="ready", zarr_store_path=store_new_gefs))

        # Products GFS
        gfs_vars = ["temperature_2m", "precipitation_rate", "precipitation_amount_3h", "cloud_cover_3h", "crain", "csnow", "wind_u_10m", "wind_v_10m"]
        for lead in leads_old:
            for v_code in gfs_vars:
                session.add(ForecastProduct(id=f"p_gfs_old_{v_code}_{lead}", run_id="run_gfs_old", variable_id=v_code, grid_id="global_025deg", product_type="surface", lead_time_hours=lead))
        for lead in leads_new:
            for v_code in gfs_vars:
                session.add(ForecastProduct(id=f"p_gfs_new_{v_code}_{lead}", run_id="run_gfs_new", variable_id=v_code, grid_id="global_025deg", product_type="surface", lead_time_hours=lead))

        # Products GEFS
        for lead in leads_old:
            session.add(ForecastProduct(id=f"p_gefs_old_t2m_{lead}", run_id="run_gefs_old", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=lead))
            for m in range(1, 31):
                session.add(EnsembleMemberProduct(id=f"emp_old_{m}_{lead}", run_id="run_gefs_old", member_index=m, lead_time_hours=lead))
        for lead in leads_new:
            session.add(ForecastProduct(id=f"p_gefs_new_t2m_{lead}", run_id="run_gefs_new", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=lead))
            for m in range(1, 31):
                session.add(EnsembleMemberProduct(id=f"emp_new_{m}_{lead}", run_id="run_gefs_new", member_index=m, lead_time_hours=lead))

        session.commit()

    return engine


@pytest.fixture
def boundary_client(left_boundary_env, monkeypatch):
    engine = left_boundary_env
    sim_now = _dt(2026, 9, 11, 7, 0, 0)

    def override_get_db():
        session = Session(engine)
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("api.core.database.SessionLocal", lambda: Session(engine))
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_time] = lambda: sim_now
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_current_time, None)


def test_cadence_boundary_stepping_primitive():
    """Verify exact floor-to-3h cadence behavior across boundary steps."""
    assert serving_start_valid_time(_dt(2026, 9, 11, 5, 59, 59)) == _dt(2026, 9, 11, 3, 0, 0)
    assert serving_start_valid_time(_dt(2026, 9, 11, 6, 0, 0)) == _dt(2026, 9, 11, 6, 0, 0)
    assert serving_start_valid_time(_dt(2026, 9, 11, 6, 0, 1)) == _dt(2026, 9, 11, 6, 0, 0)
    assert serving_start_valid_time(_dt(2026, 9, 11, 7, 0, 0)) == _dt(2026, 9, 11, 6, 0, 0)
    assert serving_start_valid_time(_dt(2026, 9, 11, 8, 59, 59)) == _dt(2026, 9, 11, 6, 0, 0)
    assert serving_start_valid_time(_dt(2026, 9, 11, 9, 0, 0)) == _dt(2026, 9, 11, 9, 0, 0)
    assert serving_start_valid_time(_dt(2026, 9, 11, 9, 0, 1)) == _dt(2026, 9, 11, 9, 0, 0)


def test_soak_failure_regression_availability(boundary_client):
    """Simulate soak test failure at 2026-09-11 07:00Z -> serving_start = 06:00Z.

    - Availability exposes serving_start_valid_time = 06:00Z.
    - Stale valid times (Sep 8, 9, 10, and Sep 11 00Z) are EXCLUDED from valid_times.
    - Earliest valid time in availability is exactly 06:00Z.
    - Rolling right boundary out to Sep 12 00Z is preserved.
    """
    resp = boundary_client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    data = resp.json()["data"]

    # Authoritative boundary is exposed
    assert data["serving_start_valid_time"] == "2026-09-11T06:00:00Z"

    gfs = next(m for m in data["models"] if m["id"] == "gfs")
    t2m = next(v for v in gfs["variables"] if v["id"] == "temperature_2m")

    vt_times = [vt["valid_time"] for vt in t2m["valid_times"]]
    assert len(vt_times) > 0

    # 1. No valid times before 06:00Z
    for vt in vt_times:
        assert vt >= "2026-09-11T06:00:00Z"

    # 2. Earliest valid time is exactly 06:00Z
    assert vt_times[0] == "2026-09-11T06:00:00Z"

    # 3. Rolling right boundary preserved (Sep 12 00Z from older cycle survives)
    assert "2026-09-12T00:00:00Z" in vt_times


def test_soak_failure_regression_points(boundary_client):
    """Point forecast (/v1/points) at 2026-09-11 07:00Z:
    - No historical points before 06:00Z are returned.
    - Earliest point in series is 06:00Z.
    - 06:00Z has lead 0 from c_new: precipitation_amount_3h falls back to positive lead from c_old.
    - crain companion is coupled to the fallback source.
    - Far-future horizon (Sep 12 00Z) remains present.
    """
    resp = boundary_client.get(
        f"/v1/points?lat={LAT_START}&lon={LON_START}&models=gfs"
        "&variables=temperature_2m,precipitation_amount_3h,crain"
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    forecasts = data["forecasts"]

    # 1. No historical points before 06:00Z
    for entry in forecasts:
        assert entry["valid_time"] >= "2026-09-11T06:00:00Z"

    # 2. Earliest point is 06:00Z
    first_entry = forecasts[0]
    assert first_entry["valid_time"] == "2026-09-11T06:00:00Z"

    # 3. Overlapping resolution & Lead-0 interval fallback:
    # - temperature_2m is instantaneous: comes from c_new (22.0°C) with lead 0
    # - precipitation_amount_3h is interval: falls back to c_old (2.5mm) with lead 78
    # - crain is phase companion: couples to c_old with lead 78 (1.0)
    assert first_entry["temperature_2m"] == pytest.approx(22.0)
    assert first_entry["precipitation_amount_3h"] == pytest.approx(2.5)
    assert first_entry["crain"] == pytest.approx(1.0)

    # 4. Far-future horizon survives (last lead is Sep 12 00Z from c_old)
    last_entry = forecasts[-1]
    assert last_entry["valid_time"] == "2026-09-12T00:00:00Z"
    assert last_entry["cycle_time"] == "2026-09-08T00:00:00Z"


def test_maps_endpoint_boundary_rejection(boundary_client):
    """Map metadata and tile requests:
    - valid_time < serving_start returns 404.
    - valid_time == serving_start returns 200.
    """
    # Map metadata
    resp_expired = boundary_client.get("/v1/maps?model=gfs&variable=temperature_2m&level=surface&valid_time=2026-09-11T03:00:00Z")
    assert resp_expired.status_code == 404
    assert "before the active serving window" in resp_expired.json()["error"]["message"]

    resp_boundary = boundary_client.get("/v1/maps?model=gfs&variable=temperature_2m&level=surface&valid_time=2026-09-11T06:00:00Z")
    assert resp_boundary.status_code == 200

    # Raster tile PNG
    resp_tile_exp = boundary_client.get("/v1/maps/gfs/temperature_2m/surface/0/0/0.png?valid_time=2026-09-11T03:00:00Z")
    assert resp_tile_exp.status_code == 404

    resp_tile_ok = boundary_client.get("/v1/maps/gfs/temperature_2m/surface/0/0/0.png?valid_time=2026-09-11T06:00:00Z")
    assert resp_tile_ok.status_code == 200
    assert resp_tile_ok.headers["content-type"] == "image/png"


def test_ensembles_endpoint_boundary_rejection(boundary_client):
    """Ensemble statistics endpoint:
    - valid_time < serving_start returns 404.
    - valid_time == serving_start returns 200.
    """
    # Expired ensemble valid time
    resp_exp = boundary_client.get(
        f"/v1/ensembles?lat={LAT_START}&lon={LON_START}&model=gefs&variable=temperature_2m"
        "&valid_time=2026-09-11T03:00:00Z"
    )
    assert resp_exp.status_code == 404
    assert "before the active serving window" in resp_exp.json()["error"]["message"]

    # Boundary valid time (06:00Z)
    resp_ok = boundary_client.get(
        f"/v1/ensembles?lat={LAT_START}&lon={LON_START}&model=gefs&variable=temperature_2m"
        "&valid_time=2026-09-11T06:00:00Z"
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["data"]["valid_time"] == "2026-09-11T06:00:00Z"


def test_exact_cadence_boundary_advancement_immediate(left_boundary_env, monkeypatch):
    """Verify that at exactly 09:00:00Z, the boundary advances to 09:00:00Z immediately:
    - At 08:59:59Z: 06:00:00Z is servable.
    - At 09:00:00Z: 06:00:00Z is rejected with 404 (no 3h grace lag).
    """
    engine = left_boundary_env

    def override_get_db():
        session = Session(engine)
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("api.core.database.SessionLocal", lambda: Session(engine))
    app.dependency_overrides[get_db] = override_get_db

    try:
        # At 08:59:59Z: serving_start is 06:00Z
        app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 11, 8, 59, 59)
        with TestClient(app) as client:
            resp_0859 = client.get("/v1/maps?model=gfs&variable=temperature_2m&level=surface&valid_time=2026-09-11T06:00:00Z")
            assert resp_0859.status_code == 200

        # At exactly 09:00:00Z: serving_start advances to 09:00Z immediately
        app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 11, 9, 0, 0)
        with TestClient(app) as client:
            resp_0900 = client.get("/v1/maps?model=gfs&variable=temperature_2m&level=surface&valid_time=2026-09-11T06:00:00Z")
            assert resp_0900.status_code == 404
            assert "before the active serving window" in resp_0900.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_cache_rollover_safety_across_all_endpoints(left_boundary_env, monkeypatch):
    """Verify that cached data warmed at 08:59:59Z (06Z valid) cannot be served at 09:00:00Z.

    Rollover scenario:
      At 08:59:59Z (serving_start = 06:00Z):
        - /v1/points is queried and populates PointCache with 06Z as first point.
        - /v1/maps raster tile is queried and populates tile cache for 06Z.
        - /v1/maps vector field is queried and populates vector cache for 06Z.
        - /v1/ensembles is queried and populates ensemble cache for 06Z.
      At 09:00:00Z (serving_start = 09:00Z):
        - /v1/points request cannot return cached 06Z series; earliest point is 09Z.
        - /v1/maps tile request for 06Z returns 404 (not cached PNG).
        - /v1/maps vector field request for 06Z returns 404 (not cached vector).
        - /v1/ensembles request for 06Z returns 404 (not cached statistics).
        - /v1/forecast/availability returns serving_start = 09Z and earliest valid_time = 09Z.
    """
    engine = left_boundary_env

    def override_get_db():
        session = Session(engine)
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("api.core.database.SessionLocal", lambda: Session(engine))
    app.dependency_overrides[get_db] = override_get_db

    try:
        # 1. Warm caches at 08:59:59Z (serving_start = 06:00Z)
        app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 11, 8, 59, 59)
        with TestClient(app) as client:
            # Warm points cache
            resp_p1 = client.get(f"/v1/points?lat={LAT_START}&lon={LON_START}&models=gfs&variables=temperature_2m")
            assert resp_p1.status_code == 200
            assert resp_p1.json()["data"]["forecasts"][0]["valid_time"] == "2026-09-11T06:00:00Z"

            # Warm tile cache
            resp_t1 = client.get("/v1/maps/gfs/temperature_2m/surface/0/0/0.png?valid_time=2026-09-11T06:00:00Z")
            assert resp_t1.status_code == 200

            # Warm vector field cache
            resp_v1 = client.get("/v1/maps/gfs/wind_10m/vector-field?valid_time=2026-09-11T06:00:00Z")
            assert resp_v1.status_code == 200

            # Warm ensemble cache
            resp_e1 = client.get(f"/v1/ensembles?lat={LAT_START}&lon={LON_START}&model=gefs&variable=temperature_2m&valid_time=2026-09-11T06:00:00Z")
            assert resp_e1.status_code == 200

        # 2. Advance time to 09:00:00Z (serving_start advances to 09:00Z)
        app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 11, 9, 0, 0)
        with TestClient(app) as client:
            # Verify point forecast cache rollover
            resp_p2 = client.get(f"/v1/points?lat={LAT_START}&lon={LON_START}&models=gfs&variables=temperature_2m")
            assert resp_p2.status_code == 200
            forecasts = resp_p2.json()["data"]["forecasts"]
            assert forecasts[0]["valid_time"] == "2026-09-11T09:00:00Z"
            assert all(f["valid_time"] >= "2026-09-11T09:00:00Z" for f in forecasts)

            # Verify tile cache cannot serve expired 06Z
            resp_t2 = client.get("/v1/maps/gfs/temperature_2m/surface/0/0/0.png?valid_time=2026-09-11T06:00:00Z")
            assert resp_t2.status_code == 404

            # Verify vector cache cannot serve expired 06Z
            resp_v2 = client.get("/v1/maps/gfs/wind_10m/vector-field?valid_time=2026-09-11T06:00:00Z")
            assert resp_v2.status_code == 404

            # Verify ensemble cache cannot serve expired 06Z
            resp_e2 = client.get(f"/v1/ensembles?lat={LAT_START}&lon={LON_START}&model=gefs&variable=temperature_2m&valid_time=2026-09-11T06:00:00Z")
            assert resp_e2.status_code == 404

            # Verify availability returns new boundary
            resp_avail = client.get("/v1/forecast/availability")
            assert resp_avail.status_code == 200
            assert resp_avail.json()["data"]["serving_start_valid_time"] == "2026-09-11T09:00:00Z"
    finally:
        app.dependency_overrides.clear()
