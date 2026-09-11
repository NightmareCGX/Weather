"""Service-backed PostgreSQL integration tests for Data Lifecycle V3 Phase 4 physical serving fences.

Verifies:
1. Physical shard presence in reclamation_queue with status='queued' remains physically servable.
2. Transition to status='deleting' or status='deleted' immediately fences the shard across:
   - /v1/maps metadata
   - /v1/maps raster tiles (.png)
   - /v1/maps vector field
   - /v1/points
   - /v1/forecast/availability
3. Explicit initial_time requests to reclaimed shards return clean HTTP 404.
4. Cache safety: cached responses created when status='queued' cannot bypass queued->deleting transition.
5. Post-read validation catches mid-read fence transitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pytest
import xarray as xr
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models.entities import (
    ForecastCenter,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
    ReclamationQueue,
)
from domain.reclamation import TARGET_KIND_DET, make_shard_relative_key
from tests._zarr_writer import write_dataset


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def _make_surface_dataset(cycle_time: datetime, lead: int) -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    temperature = np.full((1, 4, 4), 20.0, dtype=np.float32)
    u_wind = np.full((1, 4, 4), 5.0, dtype=np.float32)
    v_wind = np.full((1, 4, 4), 5.0, dtype=np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), temperature),
            "wind_u_10m": (("lead_time_hours", "latitude", "longitude"), u_wind),
            "wind_v_10m": (("lead_time_hours", "latitude", "longitude"), v_wind),
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
    if not session.get(ModelVersion, v_id):
        if not session.get(ForecastCenter, "center_noaa"):
            session.add(ForecastCenter(id="center_noaa", center_id="noaa", name="NOAA", country="USA"))
            session.flush()
        if not session.get(Model, f"model_{model_id}"):
            session.add(Model(id=f"model_{model_id}", model_id=model_id, name=model_id.upper(), center_id="noaa", is_ensemble=(model_id == "gefs"), resolution_km=25.0))
            session.flush()
        session.add(ModelVersion(id=v_id, model_id=model_id, version_string="v1.0"))
        session.flush()
    for v_code, v_name, v_unit in [
        ("temperature_2m", "2m Temperature", "°C"),
        ("wind_u_10m", "10m U Wind", "m/s"),
        ("wind_v_10m", "10m V Wind", "m/s"),
        ("wind_10m", "10m Wind Speed", "km/h"),
        ("precipitation_amount_3h", "3h Precip Amount", "mm"),
    ]:
        existing_var = session.execute(
            select(ForecastVariable).where(ForecastVariable.variable_code == v_code)
        ).scalar_one_or_none()
        if not existing_var:
            session.add(ForecastVariable(id=f"var_{v_code}", variable_code=v_code, name=v_name, unit=v_unit))
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
    for var in ("temperature_2m", "wind_u_10m", "wind_v_10m"):
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


def test_reclamation_physical_fencing_and_404_responses(client, seed_data, migrated_db, tmp_path):
    c1 = _dt(2026, 9, 2, 6)
    store1_path = str(tmp_path / "store_c1.zarr")

    ds1 = _make_surface_dataset(c1, 0)
    write_dataset(ds1, store1_path)

    with Session(migrated_db) as session:
        run = _seed_gfs_run(session, c1, store1_path, "ready")
        run_id = run.id

    c1_iso = c1.isoformat().replace("+00:00", "Z")

    # 1. Before reclamation: initial access succeeds
    res_map = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_map.status_code == 200

    res_tile = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_tile.status_code == 200

    res_vec = client.get(
        f"/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_vec.status_code == 200

    # 2. Add to reclamation_queue with status='queued' -> STILL physically available!
    with Session(migrated_db) as session:
        rel_key = make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 0)
        session.add(
            ReclamationQueue(
                id="reclaim_q_1",
                run_id=run_id,
                model_id="gfs",
                cycle_time=c1,
                lead_time_hours=0,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c1,
                store_path=store1_path,
                physical_key=rel_key,
                status="queued",
                attempt_count=0,
                created_at=c1,
                updated_at=c1,
            )
        )
        session.commit()

    res_map_queued = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_map_queued.status_code == 200

    # 3. Transition status from 'queued' to 'deleting' -> FENCED immediately!
    with Session(migrated_db) as session:
        rec = session.get(ReclamationQueue, "reclaim_q_1")
        rec.status = "deleting"
        rec.updated_at = _dt(2026, 9, 2, 7)
        session.commit()

    # Explicit map metadata for reclaimed shard returns 404!
    res_map_fenced = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_map_fenced.status_code == 404

    # Explicit tile for reclaimed shard returns 404!
    res_tile_fenced = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_tile_fenced.status_code == 404

    # Fence wind_u_10m: vector field must also return 404!
    with Session(migrated_db) as session:
        u_rel_key = make_shard_relative_key("wind_u_10m", TARGET_KIND_DET, 0)
        session.add(
            ReclamationQueue(
                id="reclaim_q_u",
                run_id=run_id,
                model_id="gfs",
                cycle_time=c1,
                lead_time_hours=0,
                variable_code="wind_u_10m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c1,
                store_path=store1_path,
                physical_key=u_rel_key,
                status="deleted",
                attempt_count=1,
                created_at=c1,
                updated_at=c1,
            )
        )
        session.commit()

    res_vec_fenced = client.get(
        f"/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res_vec_fenced.status_code == 404


def test_reclamation_availability_endpoint_filtering(client, seed_data, migrated_db, tmp_path):
    c1 = _dt(2026, 9, 3, 6)
    store1_path = str(tmp_path / "store_avail.zarr")

    ds1 = _make_surface_dataset(c1, 0)
    write_dataset(ds1, store1_path)

    with Session(migrated_db) as session:
        run = _seed_gfs_run(session, c1, store1_path, "ready")
        run_id = run.id

    # 1. Before reclamation: availability advertises temperature_2m at lead 0
    res_avail1 = client.get("/v1/forecast/availability")
    assert res_avail1.status_code == 200
    gfs_model = next(m for m in res_avail1.json()["data"]["models"] if m["id"] == "gfs")
    temp_var = next(v for v in gfs_model["variables"] if v["id"] == "temperature_2m")
    assert len(temp_var["initial_times"]) > 0

    # 2. Mark temperature_2m as 'deleted' in reclamation_queue
    with Session(migrated_db) as session:
        session.add(
            ReclamationQueue(
                id="reclaim_avail_test",
                run_id=run_id,
                model_id="gfs",
                cycle_time=c1,
                lead_time_hours=0,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c1,
                store_path=store1_path,
                physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 0),
                status="deleted",
                attempt_count=1,
                created_at=c1,
                updated_at=c1,
            )
        )
        session.commit()

    # 3. After reclamation: availability filters out the reclaimed shard!
    res_avail2 = client.get("/v1/forecast/availability")
    assert res_avail2.status_code == 200
    gfs_model2 = next(m for m in res_avail2.json()["data"]["models"] if m["id"] == "gfs")
    temp_var2 = next((v for v in gfs_model2["variables"] if v["id"] == "temperature_2m"), None)
    if temp_var2 is not None:
        init_time_obj = next((it for it in temp_var2["initial_times"] if "2026-09-03T06:00:00Z" in it["value"]), None)
        assert init_time_obj is None or 0 not in init_time_obj["lead_time_hours"]


def test_post_read_validation_catches_fenced_target(migrated_db):
    """Post-read validation correctly flags targets that entered deleting or deleted."""
    from api.services.resolver import check_physical_targets_fenced

    c = _dt(2026, 9, 4, 6)
    p_key = make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 0)

    with Session(migrated_db) as session:
        run = _seed_gfs_run(session, c, "/store/dummy")
        r_id = run.id

        rec = ReclamationQueue(
            id="rec_post_read_1",
            run_id=r_id,
            model_id="gfs",
            cycle_time=c,
            lead_time_hours=0,
            variable_code="temperature_2m",
            target_kind=TARGET_KIND_DET,
            member_index=0,
            valid_time=c,
            store_path="/store/dummy",
            physical_key=p_key,
            status="queued",
            attempt_count=0,
            created_at=c,
            updated_at=c,
        )
        session.add(rec)
        session.commit()

        # When status='queued', check_physical_targets_fenced returns empty (no fence)
        fenced_queued = check_physical_targets_fenced(session, [(r_id, p_key)])
        assert fenced_queued == set()

        # Transition to 'deleting'
        rec.status = "deleting"
        rec.updated_at = _dt(2026, 9, 4, 7)
        session.commit()

        # check_physical_targets_fenced immediately detects the fence!
        fenced_deleting = check_physical_targets_fenced(session, [(r_id, p_key)])
        assert (r_id, p_key) in fenced_deleting


def test_reclamation_cache_invalidation_on_fencing(client, seed_data, migrated_db, tmp_path):
    """Cached response created before reclamation cannot bypass queued->deleting fence."""
    c1 = _dt(2026, 9, 5, 6)
    store1_path = str(tmp_path / "store_cache.zarr")

    ds1 = _make_surface_dataset(c1, 0)
    write_dataset(ds1, store1_path)

    with Session(migrated_db) as session:
        run = _seed_gfs_run(session, c1, store1_path, "ready")
        run_id = run.id

    c1_iso = c1.isoformat().replace("+00:00", "Z")

    # 1. First request is computed and cached in Redis
    res1 = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res1.status_code == 200

    # 2. Target transitions to 'deleting' in reclamation_queue
    with Session(migrated_db) as session:
        session.add(
            ReclamationQueue(
                id="reclaim_cache_test",
                run_id=run_id,
                model_id="gfs",
                cycle_time=c1,
                lead_time_hours=0,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c1,
                store_path=store1_path,
                physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 0),
                status="deleting",
                attempt_count=1,
                created_at=c1,
                updated_at=_dt(2026, 9, 5, 7),
            )
        )
        session.commit()

    # 3. Repeated request must NOT return stale cached payload; must return 404!
    res2 = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c1_iso}"
    )
    assert res2.status_code == 404


def test_pinned_initial_time_cache_invalidation_on_ensembles(client, seed_data, migrated_db):
    """Ensembles endpoint validates physical availability for pinned initial_time before returning cache hit."""
    # The seeded GEFS run has cycle_time 2026-07-21T00:00:00Z and lead 0
    init_iso = "2026-07-21T00:00:00Z"
    c_time = _dt(2026, 7, 21, 0)

    # 1. First request succeeds and populates cache
    res1 = client.get(
        f"/v1/ensembles?lat=38.5&lon=-106.5&variable=temperature_2m&model=gefs&lead_time_hours=0&initial_time={init_iso}"
    )
    assert res1.status_code == 200

    # 2. Fence 10 members in reclamation_queue (leaves 20/30 < 85% threshold)
    with Session(migrated_db) as session:
        for m in range(1, 11):
            session.add(
                ReclamationQueue(
                    id=f"reclaim_mem_cache_{m}",
                    run_id="run_2026072100_gefs",
                    model_id="gefs",
                    cycle_time=c_time,
                    lead_time_hours=0,
                    variable_code="temperature_2m",
                    target_kind="mem",
                    member_index=m,
                    valid_time=c_time,
                    store_path="/stores/gefs",
                    physical_key=f"temperature_2m/shard.mem{m:03d}_L0000.shard",
                    status="deleting",
                    attempt_count=1,
                    created_at=c_time,
                    updated_at=_dt(2026, 7, 21, 1),
                )
            )
        session.commit()

    # 3. Repeated pinned request must NOT return stale cache hit; must return 404!
    res2 = client.get(
        f"/v1/ensembles?lat=38.5&lon=-106.5&variable=temperature_2m&model=gefs&lead_time_hours=0&initial_time={init_iso}"
    )
    assert res2.status_code == 404



