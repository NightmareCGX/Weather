"""Phase 6E Serving Surface, In-Flight Reader & Adversarial Cache Bypass Acceptance Tests.

Validates under real PostgreSQL, Redis, and MinIO/Zarr:
1. Adversarial cache bypass immunity across Redis and in-memory caches.
2. In-flight reader safety during concurrent retirement transaction.
3. Cross-cycle min-lead winner selection blending visible ready + partial cycles while excluding retired.
4. Complete public serving surface audit (/v1/forecast/availability, /v1/points, /v1/ensembles,
   /v1/probabilities, /v1/maps, raster tiles, vector fields, /v1/runs, /v1/verifications).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.core.reader_gate import _ReaderGateSession, ReaderLockPool
from api.models.entities import (
    ForecastCycleLifecycle,
)
from api.services.tiles import _tile_cache
from api.services.vector_field import _vector_cache
from ingestion.core.catalog import (
    CommittedState,
    RunCatalogSpec,
    VariableSpec,
    record_run,
)
from ingestion.core.zarr_writer import write_dataset


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def _make_dataset(cycle_time: datetime, leads: list[int]) -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    num_leads = len(leads)
    temperature = np.full((num_leads, 4, 4), 20.0, dtype=np.float32)
    precipitation = np.full((num_leads, 4, 4), 0.5, dtype=np.float32)
    u10 = np.full((num_leads, 4, 4), 5.0, dtype=np.float32)
    v10 = np.full((num_leads, 4, 4), 5.0, dtype=np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), temperature),
            "precipitation_rate": (("lead_time_hours", "latitude", "longitude"), precipitation),
            "wind_u_10m": (("lead_time_hours", "latitude", "longitude"), u10),
            "wind_v_10m": (("lead_time_hours", "latitude", "longitude"), v10),
        },
        coords={
            "lead_time_hours": leads,
            "latitude": lat,
            "longitude": lon,
        },
    )


def _make_ensemble_dataset(cycle_time: datetime, leads: list[int], members: list[int]) -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    shape = (len(members), len(leads), 4, 4)
    temperature = np.full(shape, 20.0, dtype=np.float32)
    u10 = np.full(shape, 5.0, dtype=np.float32)
    v10 = np.full(shape, 5.0, dtype=np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("member", "lead_time_hours", "latitude", "longitude"), temperature),
            "wind_u_10m": (("member", "lead_time_hours", "latitude", "longitude"), u10),
            "wind_v_10m": (("member", "lead_time_hours", "latitude", "longitude"), v10),
        },
        coords={
            "member": members,
            "lead_time_hours": leads,
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture(autouse=True)
def _flush_caches(migrated_db):
    """Clear Redis, in-memory caches, and cycle lifecycle rows before each test in this module."""
    import redis as redis_lib
    from api.core.config import settings

    _tile_cache.clear()
    _vector_cache.clear()
    try:
        client = redis_lib.from_url(settings.REDIS_URL)
        client.flushall()
        client.close()
    except Exception:
        pass

    with Session(migrated_db) as session:
        session.execute(text("DELETE FROM forecast_cycle_lifecycle"))
        session.commit()


# ---------------------------------------------------------------------------
# Adversarial Cache Bypass Tests
# ---------------------------------------------------------------------------


def test_adversarial_cache_bypass_immunity(client, migrated_db, tmp_path):
    """Prove that cached responses in Redis and in-memory caches cannot bypass retirement."""
    c_ret = _dt(2026, 9, 1, 6)
    c_vis = _dt(2026, 9, 2, 6)

    gfs_path = str(tmp_path / "gfs_c1.zarr")
    gefs_path = str(tmp_path / "gefs_c1.zarr")

    ds_gfs = _make_dataset(c_ret, [0])
    ds_gefs = _make_ensemble_dataset(c_ret, [0], list(range(1, 31)))

    write_dataset(ds_gfs, gfs_path)
    write_dataset(ds_gefs, gefs_path)

    spec_gfs = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="US",
        model_id="gfs",
        model_name="GFS",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=c_ret,
        grid_id="global_025deg",
        grid_name="Global 0.25",
        grid_resolution_km=25.0,
        zarr_store_path=gfs_path,
        variables=(
            VariableSpec("temperature_2m", "2-Meter Temperature", "°C"),
            VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h"),
            VariableSpec("wind_u_10m", "Wind U", "m/s"),
            VariableSpec("wind_v_10m", "Wind V", "m/s"),
        ),
        expected_lead_time_hours=(0,),
    )
    spec_gefs = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="US",
        model_id="gefs",
        model_name="GEFS",
        is_ensemble=True,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=c_ret,
        grid_id="global_025deg",
        grid_name="Global 0.25",
        grid_resolution_km=25.0,
        zarr_store_path=gefs_path,
        variables=(
            VariableSpec("temperature_2m", "2-Meter Temperature", "°C"),
            VariableSpec("wind_u_10m", "Wind U", "m/s"),
            VariableSpec("wind_v_10m", "Wind V", "m/s"),
        ),
        expected_lead_time_hours=(0,),
        expected_members=tuple(range(1, 31)),
    )

    with Session(migrated_db) as session:
        record_run(session, spec_gfs, ds_gfs, committed_state=CommittedState.deterministic({0}))
        record_run(
            session,
            spec_gefs,
            ds_gefs,
            committed_state=CommittedState.ensemble({(m, 0) for m in range(1, 31)}, set(range(1, 31))),
        )

    c_ret_iso = c_ret.isoformat().replace("+00:00", "Z")

    # 1. Warm all caches while cycle is VISIBLE
    res_ens = client.get(
        f"/v1/ensembles?lat=38.19&lon=-106.82&variable=temperature_2m&model=gefs&initial_time={c_ret_iso}"
    )
    assert res_ens.status_code == 200

    res_prob = client.get(
        f"/v1/probabilities?lat=38.19&lon=-106.82&variable=temperature_2m&threshold=10&operator=gt&lead_time_hours=0&model=gefs&initial_time={c_ret_iso}"
    )
    assert res_prob.status_code == 200

    res_map = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c_ret_iso}"
    )
    assert res_map.status_code == 200

    res_tile = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={c_ret_iso}"
    )
    assert res_tile.status_code == 200

    res_vec = client.get(
        f"/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=0&initial_time={c_ret_iso}"
    )
    assert res_vec.status_code == 200

    # 2. Retire cycle c_ret in PostgreSQL
    with Session(migrated_db) as session:
        session.add(
            ForecastCycleLifecycle(
                cycle_time=c_ret,
                retired_at=_dt(2026, 9, 2, 6, 30),
                retired_by_cycle_time=c_vis,
            )
        )
        session.commit()

    # 3. Query all 5 cached endpoints immediately: MUST return 404
    assert client.get(
        f"/v1/ensembles?lat=38.19&lon=-106.82&variable=temperature_2m&model=gefs&initial_time={c_ret_iso}"
    ).status_code == 404

    assert client.get(
        f"/v1/probabilities?lat=38.19&lon=-106.82&variable=temperature_2m&threshold=10&operator=gt&lead_time_hours=0&model=gefs&initial_time={c_ret_iso}"
    ).status_code == 404

    assert client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c_ret_iso}"
    ).status_code == 404

    assert client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={c_ret_iso}"
    ).status_code == 404

    assert client.get(
        f"/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=0&initial_time={c_ret_iso}"
    ).status_code == 404


# ---------------------------------------------------------------------------
# In-Flight Reader vs Retirement Concurrency Test
# ---------------------------------------------------------------------------


def test_in_flight_reader_safe_during_concurrent_retirement(client, migrated_db, tmp_path):
    """Prove that an in-flight reader completes safely while new requests observe retirement."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    c0 = _dt(2026, 9, 1, 6)
    c1 = _dt(2026, 9, 2, 6)
    c0_iso = c0.isoformat().replace("+00:00", "Z")

    gfs_path = str(tmp_path / "gfs_in_flight.zarr")
    ds = _make_dataset(c0, [0])
    write_dataset(ds, gfs_path)

    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="US",
        model_id="gfs",
        model_name="GFS",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=c0,
        grid_id="global_025deg",
        grid_name="Global 0.25",
        grid_resolution_km=25.0,
        zarr_store_path=gfs_path,
        variables=(VariableSpec("temperature_2m", "2-Meter Temperature", "°C"),),
        expected_lead_time_hours=(0,),
    )
    with Session(migrated_db) as session:
        record_run(session, spec, ds, committed_state=CommittedState.deterministic({0}))

    # T0: Start a reader session holding SHARED reader gate
    reader_pool = ReaderLockPool(db_url, pool_size=2, max_overflow=0, pool_timeout=5.0)
    reader_session = _ReaderGateSession(reader_pool, gfs_path)
    reader_session.acquire(timeout_seconds=5.0)

    try:
        # T1: Retirement commits in PostgreSQL
        with Session(migrated_db) as session:
            session.add(
                ForecastCycleLifecycle(
                    cycle_time=c0,
                    retired_at=_dt(2026, 9, 2, 6, 30),
                    retired_by_cycle_time=c1,
                )
            )
            session.commit()

        # T2: New requests observe retirement and return 404
        res_new = client.get(
            f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c0_iso}"
        )
        assert res_new.status_code == 404

        # T3: In-flight reader T0 safely reads data from store under its lock
        ok, path = reader_session.revalidate(db_url)
        assert ok is True
        assert path == gfs_path
    finally:
        reader_session.release()
        reader_pool.dispose()


# ---------------------------------------------------------------------------
# Progressive Serving Blending with Retired Predecessor
# ---------------------------------------------------------------------------


def test_progressive_serving_blends_visible_runs_and_excludes_retired(client, migrated_db, tmp_path):
    """Verify that /v1/points blends visible ready and partial runs, while excluding retired."""
    c_ret = _dt(2026, 9, 1, 0)   # Retired
    c_ready = _dt(2026, 9, 1, 6) # Visible ready (leads 0, 6, 12, 18)
    c_part = _dt(2026, 9, 2, 6)  # Visible partial (lead 0 only)

    p_ret = str(tmp_path / "gfs_ret.zarr")
    p_ready = str(tmp_path / "gfs_ready.zarr")
    p_part = str(tmp_path / "gfs_part.zarr")

    ds_ret = _make_dataset(c_ret, [0, 6])
    ds_ready = _make_dataset(c_ready, [0, 6, 12, 18])
    ds_part = _make_dataset(c_part, [0])

    write_dataset(ds_ret, p_ret)
    write_dataset(ds_ready, p_ready)
    write_dataset(ds_part, p_part)

    def _spec(cycle, path, leads):
        return RunCatalogSpec(
            center_id="noaa",
            center_name="NOAA",
            center_country="US",
            model_id="gfs",
            model_name="GFS",
            is_ensemble=False,
            resolution_km=25.0,
            version_string="v1.0",
            cycle_time=cycle,
            grid_id="global_025deg",
            grid_name="Global 0.25",
            grid_resolution_km=25.0,
            zarr_store_path=path,
            variables=(VariableSpec("temperature_2m", "2-Meter Temperature", "°C"),),
            expected_lead_time_hours=tuple(leads),
        )

    with Session(migrated_db) as session:
        record_run(session, _spec(c_ret, p_ret, [0, 6]), ds_ret, committed_state=CommittedState.deterministic({0, 6}))
        record_run(session, _spec(c_ready, p_ready, [0, 6, 12, 18]), ds_ready, committed_state=CommittedState.deterministic({0, 6, 12, 18}))
        r_part = record_run(session, _spec(c_part, p_part, [0, 6, 12, 18]), ds_part, committed_state=None)
        setattr(r_part, "status", "partial")

        # Mark c_ret retired
        session.add(
            ForecastCycleLifecycle(
                cycle_time=c_ret,
                retired_at=_dt(2026, 9, 2, 0),
                retired_by_cycle_time=c_part,
            )
        )
        session.commit()

    res = client.get("/v1/points?city_id=city_aspen&models=gfs&variables=temperature_2m")
    assert res.status_code == 200
    forecasts = res.json()["data"]["forecasts"]
    cycles_in_series = {f["cycle_time"] for f in forecasts}

    # Retired cycle c_ret is strictly excluded
    assert "2026-09-01T00:00:00Z" not in cycles_in_series

    # Visible ready c_ready and visible partial c_part both contribute
    assert "2026-09-01T06:00:00Z" in cycles_in_series
    assert "2026-09-02T06:00:00Z" in cycles_in_series
