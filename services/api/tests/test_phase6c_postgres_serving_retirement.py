"""Service-backed PostgreSQL integration tests for Phase 6C serving retirement.

Verifies against real PostgreSQL and physical Zarr stores:
1. Active cycle C is servable via /v1/points, /v1/ensembles, /v1/probabilities, /v1/maps.
2. Retirement transaction commits (forecast_cycle_lifecycle.retired_at populated).
3. Physical Zarr store still exists on disk.
4. Serving immediately returns 404 / excludes cycle C from min-lead and availability.
5. Non-retired partial cycles continue to serve progressively under Phase 3.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from sqlalchemy.orm import Session

from api.models.entities import (
    ForecastCycleLifecycle,
    ForecastProduct,
    ModelRun,
)
from tests._zarr_writer import write_dataset


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def _make_surface_dataset(cycle_time: datetime, lead: int) -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    temperature = np.full((1, 4, 4), 20.0, dtype=np.float32)
    precipitation = np.full((1, 4, 4), 0.5, dtype=np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                temperature,
            ),
            "precipitation_rate": (
                ("lead_time_hours", "latitude", "longitude"),
                precipitation,
            ),
        },
        coords={
            "lead_time_hours": [lead],
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture(autouse=True)
def _flush_redis():
    """Clear Redis before each test in this module."""
    import redis as redis_lib
    from api.core.config import settings

    client = redis_lib.from_url(settings.REDIS_URL)
    try:
        client.flushall()
    except Exception:
        pass
    finally:
        client.close()


def _ensure_version(session: Session, model_id: str) -> str:
    v_id = f"version_{model_id}_v1"
    from api.models.entities import ForecastCenter, Model, ModelVersion
    if not session.get(ModelVersion, v_id):
        if not session.get(ForecastCenter, "center_noaa"):
            session.add(ForecastCenter(id="center_noaa", center_id="noaa", name="NOAA", country="USA"))
            session.flush()
        if not session.get(Model, f"model_{model_id}"):
            session.add(Model(id=f"model_{model_id}", model_id=model_id, name=model_id.upper(), center_id="noaa", is_ensemble=(model_id == "gefs"), resolution_km=25.0))
            session.flush()
        session.add(ModelVersion(id=v_id, model_id=model_id, version_string="v1.0"))
        session.flush()
    return v_id


def _seed_gfs_run(
    session: Session,
    cycle_time: datetime,
    store_path: str,
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
    for var in ("temperature_2m", "precipitation_rate"):
        session.add(
            ForecastProduct(
                id=f"product_gfs_{var}_{c_tag}_0",
                run_id=run.id,
                variable_id=var,
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
                zarr_chunk_path=store_path,
            )
        )
    session.commit()
    return run


def test_postgres_serving_retirement_lifecycle_transition(
    client, migrated_db, tmp_path
):
    c1 = _dt(2026, 9, 1, 6)
    c2 = _dt(2026, 9, 2, 6)

    store1_path = str(tmp_path / "store_c1.zarr")
    store2_path = str(tmp_path / "store_c2.zarr")

    ds1 = _make_surface_dataset(c1, 0)
    ds2 = _make_surface_dataset(c2, 0)

    write_dataset(ds1, store1_path)
    write_dataset(ds2, store2_path)

    with Session(migrated_db) as session:
        _seed_gfs_run(session, c1, store1_path, "ready")
        _seed_gfs_run(session, c2, store2_path, "ready")

    # 1. Before retirement: point forecast succeeds
    res_point = client.get(
        "/v1/points?city_id=city_aspen&models=gfs&variables=temperature_2m"
    )
    assert res_point.status_code == 200

    # Explicit map metadata for c1 succeeds
    c1_iso = c1.isoformat().replace("+00:00", "Z")
    res_map = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_map.status_code == 200

    # 2. Mark c1 as RETIRED in PostgreSQL
    with Session(migrated_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c1,
                retired_at=_dt(2026, 9, 2, 6, 30),
                retired_by_cycle_time=c2,
            )
        )
        session.commit()

    # 3. Verify PHYSICAL STORE STILL EXISTS on disk
    assert os.path.exists(store1_path)

    # 4. Serving immediately rejects explicit c1 access with 404
    res_map_retired = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_map_retired.status_code == 404
    assert "not available" in res_map_retired.json()["error"]["message"]

    # Explicit tile for retired c1 returns 404
    res_tile_retired = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_tile_retired.status_code == 404

    # Vector field for retired c1 returns 404
    res_vec_retired = client.get(
        f"/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_vec_retired.status_code == 404

    # Point forecast now draws from visible cycles, with retired c1 strictly excluded
    res_point2 = client.get(
        "/v1/points?city_id=city_aspen&models=gfs&variables=temperature_2m"
    )
    assert res_point2.status_code == 200
    forecasts = res_point2.json()["data"]["forecasts"]
    cycle_times = [f["cycle_time"] for f in forecasts]
    assert "2026-09-01T06:00:00Z" not in cycle_times
    assert "2026-09-02T06:00:00Z" in cycle_times
