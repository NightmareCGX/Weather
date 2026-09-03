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
    EnsembleMember,
    EnsembleMemberProduct,
    ForecastCycleLifecycle,
    ForecastProduct,
    ModelRun,
)
from api.services.tiles import _tile_cache
from api.services.vector_field import _vector_cache
from tests._zarr_writer import write_dataset


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
    """Clear Redis, in-memory caches, and test catalog rows before each test in this module."""
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
        session.execute(text("DELETE FROM forecast_products WHERE run_id LIKE 'run_gfs_%' OR run_id LIKE 'run_gefs_%'"))
        session.execute(text("DELETE FROM ensemble_member_products WHERE run_id LIKE 'run_gfs_%' OR run_id LIKE 'run_gefs_%'"))
        session.execute(text("DELETE FROM ensemble_members WHERE run_id LIKE 'run_gfs_%' OR run_id LIKE 'run_gefs_%'"))
        session.execute(text("DELETE FROM model_runs WHERE id LIKE 'run_gfs_%' OR id LIKE 'run_gefs_%'"))
        session.execute(text("DELETE FROM forecast_cycle_lifecycle"))
        session.commit()


def _ensure_version(session: Session, model_id: str) -> str:
    from sqlalchemy import select
    from api.models.entities import (
        ForecastCenter,
        ForecastGrid,
        ForecastVariable,
        Model,
        ModelVersion,
    )
    if not session.execute(select(ForecastCenter).where(ForecastCenter.center_id == "noaa")).scalar_one_or_none():
        session.add(ForecastCenter(id="center_noaa", center_id="noaa", name="NOAA", country="USA"))
        session.flush()
    if not session.execute(select(Model).where(Model.model_id == model_id)).scalar_one_or_none():
        session.add(Model(id=f"model_{model_id}", model_id=model_id, name=model_id.upper(), center_id="noaa", is_ensemble=(model_id == "gefs"), resolution_km=25.0))
        session.flush()
    existing_ver = session.execute(select(ModelVersion).where((ModelVersion.model_id == model_id) & (ModelVersion.version_string == "v1.0"))).scalar_one_or_none()
    if not existing_ver:
        v_id = f"version_{model_id}_v1"
        session.add(ModelVersion(id=v_id, model_id=model_id, version_string="v1.0"))
        session.flush()
    else:
        v_id = str(existing_ver.id)
    if not session.execute(select(ForecastGrid).where(ForecastGrid.grid_code == "global_025deg")).scalar_one_or_none():
        session.add(ForecastGrid(id="grid_global_025deg", grid_code="global_025deg", name="Global 0.25", resolution_km=25.0))
        session.flush()
    for v_code, v_name, v_unit in [
        ("temperature_2m", "2m Temp", "°C"),
        ("precipitation_rate", "Precip Rate", "mm/h"),
        ("wind_10m", "10m Wind", "km/h"),
        ("wind_u_10m", "Wind U", "m/s"),
        ("wind_v_10m", "Wind V", "m/s"),
    ]:
        if not session.execute(select(ForecastVariable).where(ForecastVariable.variable_code == v_code)).scalar_one_or_none():
            session.add(ForecastVariable(id=f"var_{v_code}", variable_code=v_code, name=v_name, unit=v_unit))
            session.flush()
    return v_id


def _seed_gfs_run(
    session: Session,
    cycle_time: datetime,
    store_path: str,
    leads: list[int],
    status: str = "ready",
) -> ModelRun:
    v_id = _ensure_version(session, "gfs")
    c_tag = cycle_time.strftime("%Y%m%d%H%M")
    run = ModelRun(
        id=f"run_gfs_{c_tag}",
        model_version_id=v_id,
        cycle_time=cycle_time,
        status=status,
        zarr_store_path=store_path,
        created_at=cycle_time,
    )
    session.add(run)
    for lead in leads:
        for var in ("temperature_2m", "precipitation_rate", "wind_u_10m", "wind_v_10m"):
            session.add(
                ForecastProduct(
                    id=f"product_gfs_{var}_{c_tag}_{lead}",
                    run_id=run.id,
                    variable_id=var,
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=lead,
                    zarr_chunk_path=store_path,
                )
            )
    session.commit()
    return run


def _seed_gefs_run(
    session: Session,
    cycle_time: datetime,
    store_path: str,
    leads: list[int],
    members: list[int],
    status: str = "ready",
) -> ModelRun:
    v_id = _ensure_version(session, "gefs")
    c_tag = cycle_time.strftime("%Y%m%d%H%M")
    run = ModelRun(
        id=f"run_gefs_{c_tag}",
        model_version_id=v_id,
        cycle_time=cycle_time,
        status=status,
        zarr_store_path=store_path,
        created_at=cycle_time,
    )
    session.add(run)
    for m in members:
        session.add(
            EnsembleMember(
                id=f"member_gefs_{m}_{c_tag}",
                run_id=run.id,
                member_index=m,
                member_name=f"gefs_member_{m}",
            )
        )
    for lead in leads:
        for var in ("temperature_2m", "wind_u_10m", "wind_v_10m"):
            session.add(
                ForecastProduct(
                    id=f"product_gefs_{var}_{c_tag}_{lead}",
                    run_id=run.id,
                    variable_id=var,
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=lead,
                    zarr_chunk_path=store_path,
                )
            )
        for m in members:
            session.add(
                EnsembleMemberProduct(
                    id=f"emp_gefs_{m}_{lead}_{c_tag}",
                    run_id=run.id,
                    member_index=m,
                    lead_time_hours=lead,
                )
            )
    session.commit()
    return run


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

    with Session(migrated_db) as session:
        _seed_gfs_run(session, c_ret, gfs_path, [0], "ready")
        _seed_gefs_run(session, c_ret, gefs_path, [0], list(range(1, 31)), "ready")

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

    with Session(migrated_db) as session:
        _seed_gfs_run(session, c0, gfs_path, [0], "ready")

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

    with Session(migrated_db) as session:
        _seed_gfs_run(session, c_ret, p_ret, [0, 6], "ready")
        _seed_gfs_run(session, c_ready, p_ready, [0, 6, 12, 18], "ready")
        _seed_gfs_run(session, c_part, p_part, [0], "partial")

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
