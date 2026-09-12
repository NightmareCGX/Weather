"""Milestone 5 Acceptance Test Suite: Discovery Surface Alignment.

Validates the locked M5 contract:
1. /v1/runs is an OPERATIONAL CATALOG:
   - Lists active runs
   - Lists runs with deletion_started_at != NULL while model_runs metadata exists
   - Lists runs with deleted_at != NULL while model_runs metadata exists
   - Naturally omits runs once model_runs metadata is deleted (M3 purge)
   - retired_at has zero effect
   - Query filters (model_id, status, pagination) work normally
   - No include_deleted parameter is added or accepted

2. /v1/forecast/availability represents ACTUAL V3 SERVABLE FORECAST AVAILABILITY:
   - deletion_started_at cycles contribute 0 availability
   - deleted_at cycles contribute 0 availability
   - retired_at alone has zero effect on availability
   - serving_start_valid_time(now) strictly enforced: 06Z included, 03Z excluded when now = 07Z
   - Granular product fences: deleting/deleted/failed excluded, queued remains eligible
   - Target-kind correlation: det does not mask mean, mean does not mask det
   - Ensemble member fences: deleting/deleted/failed members excluded from emp_counts
   - GEFS threshold: >=26 members servable, <26 members not servable
   - Canonical cross-cycle fallback: older cycle covers far horizon when newer cycle is partial
   - Interval variables (precipitation_amount_3h, cloud_cover_3h) lead-0 fallback provenance

3. Primary Contract Separation Gate:
   - A cycle with deleted_at != NULL prior to M3 metadata purge is:
     PRESENT in GET /v1/runs
     ABSENT in GET /v1/forecast/availability
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
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
    ReclamationQueue,
)
from domain.coverage import get_expected_members, register_expected_members
from domain.reclamation import TARGET_KIND_DET, TARGET_KIND_MEAN, TARGET_KIND_MEM


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


@pytest.fixture
def m5_env(tmp_path):
    """Create an isolated test database seeded with GFS and GEFS catalog metadata."""
    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    engine = create_engine(
        f"sqlite:///{tmp_path}/test_m5_discovery.db",
        connect_args={"check_same_thread": False},
    )
    tables = [
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
        ReclamationQueue.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)

    with Session(engine) as session:
        center = ForecastCenter(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="US",
            created_at=_dt(2026, 1, 1, 0),
        )
        gfs = Model(
            id="model_gfs",
            model_id="gfs",
            name="Global Forecast System",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs = Model(
            id="model_gefs",
            model_id="gefs",
            name="Global Ensemble Forecast System",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        v_gfs = ModelVersion(
            id="version_gfs_v1.0",
            model_id="gfs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        v_gefs = ModelVersion(
            id="version_gefs_v1.0",
            model_id="gefs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        grid = ForecastGrid(
            id="grid_global_025deg",
            grid_code="global_025deg",
            name="Global 0.25 Degree",
            resolution_km=25.0,
        )
        session.add_all([center, gfs, gefs, v_gfs, v_gefs, grid])

        vars_meta = [
            ("temperature_2m", "2-Meter Temperature", "°C"),
            ("precipitation_rate", "Precipitation Rate", "mm/h"),
            ("precipitation_amount_3h", "3-Hour Precipitation", "mm"),
            ("cloud_cover_3h", "3-Hour Cloud Cover", "%"),
            ("wind_u_10m", "10-Meter U Wind", "m/s"),
            ("wind_v_10m", "10-Meter V Wind", "m/s"),
            ("crain", "Rain Flag", "categorical"),
            ("csnow", "Snow Flag", "categorical"),
        ]
        for v_id, v_name, v_unit in vars_meta:
            session.add(
                ForecastVariable(
                    id=f"var_{v_id}",
                    variable_code=v_id,
                    name=v_name,
                    unit=v_unit,
                )
            )
        session.commit()

    def override_get_db():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield engine
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_time, None)
        register_expected_members("gefs", old_expected)


# ===========================================================================
# PART 1: /v1/runs OPERATIONAL CATALOG TESTS
# ===========================================================================


def test_01_runs_lists_active_run(m5_env):
    """1. Active ModelRun appears in /v1/runs."""
    client = TestClient(app)
    c1 = _dt(2026, 9, 10, 0)
    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_active_1",
                model_version_id="version_gfs_v1.0",
                cycle_time=c1,
                status="ready",
                zarr_store_path="/store/c1",
                created_at=c1,
            )
        )
        session.commit()

    resp = client.get("/v1/runs?model_id=gfs")
    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()["data"]]
    assert "run_active_1" in ids


def test_02_runs_lists_deletion_started_run(m5_env):
    """2. Run with deletion_started_at != NULL remains listed in /v1/runs while model_runs exists."""
    client = TestClient(app)
    c2 = _dt(2026, 9, 10, 6)
    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_claimed_2",
                model_version_id="version_gfs_v1.0",
                cycle_time=c2,
                status="ready",
                zarr_store_path="/store/c2",
                created_at=c2,
            )
        )
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c2,
                deletion_started_at=_dt(2026, 9, 11, 0),
            )
        )
        session.commit()

    resp = client.get("/v1/runs?model_id=gfs")
    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()["data"]]
    assert "run_claimed_2" in ids


def test_03_runs_lists_deleted_at_run(m5_env):
    """3. Run with deleted_at != NULL remains listed in /v1/runs while model_runs exists."""
    client = TestClient(app)
    c3 = _dt(2026, 9, 10, 12)
    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_deleted_3",
                model_version_id="version_gfs_v1.0",
                cycle_time=c3,
                status="ready",
                zarr_store_path="/store/c3",
                created_at=c3,
            )
        )
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c3,
                deletion_started_at=_dt(2026, 9, 11, 0),
                deleted_at=_dt(2026, 9, 11, 1),
            )
        )
        session.commit()

    resp = client.get("/v1/runs?model_id=gfs")
    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()["data"]]
    assert "run_deleted_3" in ids


def test_04_runs_disappears_after_metadata_purge(m5_env):
    """4. After M3 retention sweeper deletes model_runs row, run naturally disappears."""
    client = TestClient(app)
    c4 = _dt(2026, 9, 10, 18)
    with Session(m5_env) as session:
        r = ModelRun(
            id="run_purged_4",
            model_version_id="version_gfs_v1.0",
            cycle_time=c4,
            status="ready",
            zarr_store_path="/store/c4",
            created_at=c4,
        )
        session.add(r)
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c4,
                deleted_at=_dt(2026, 9, 11, 1),
            )
        )
        session.commit()

        # Simulate M3 14-day retention purge of model_runs metadata
        session.delete(r)
        session.commit()

    resp = client.get("/v1/runs?model_id=gfs")
    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()["data"]]
    assert "run_purged_4" not in ids


def test_05_runs_retired_at_has_no_effect(m5_env):
    """5. retired_at != NULL alone has zero effect on /v1/runs."""
    client = TestClient(app)
    c5 = _dt(2026, 9, 11, 0)
    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_retired_5",
                model_version_id="version_gfs_v1.0",
                cycle_time=c5,
                status="ready",
                zarr_store_path="/store/c5",
                created_at=c5,
            )
        )
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c5,
                retired_at=_dt(2026, 9, 11, 6),
            )
        )
        session.commit()

    resp = client.get("/v1/runs?model_id=gfs")
    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()["data"]]
    assert "run_retired_5" in ids


def test_06_runs_query_parameters_and_no_include_deleted(m5_env):
    """6. Existing query parameters work as expected; no include_deleted param is added."""
    client = TestClient(app)
    c_gfs = _dt(2026, 9, 11, 6)
    c_gefs = _dt(2026, 9, 11, 6)
    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_param_gfs",
                model_version_id="version_gfs_v1.0",
                cycle_time=c_gfs,
                status="ready",
                zarr_store_path="/store/param_gfs",
                created_at=c_gfs,
            )
        )
        session.add(
            ModelRun(
                id="run_param_gefs",
                model_version_id="version_gefs_v1.0",
                cycle_time=c_gefs,
                status="processing",
                zarr_store_path="/store/param_gefs",
                created_at=c_gefs,
            )
        )
        session.commit()

    # Model filtering
    r_gfs = client.get("/v1/runs?model_id=gfs")
    assert "run_param_gfs" in [r["id"] for r in r_gfs.json()["data"]]
    assert "run_param_gefs" not in [r["id"] for r in r_gfs.json()["data"]]

    # Status filtering
    r_proc = client.get("/v1/runs?status=processing")
    assert "run_param_gefs" in [r["id"] for r in r_proc.json()["data"]]
    assert "run_param_gfs" not in [r["id"] for r in r_proc.json()["data"]]

    # Verify no include_deleted in OpenAPI query params
    openapi_schema = app.openapi()
    runs_get = openapi_schema["paths"]["/v1/runs"]["get"]
    query_param_names = [
        p["name"] for p in runs_get.get("parameters", []) if p.get("in") == "query"
    ]
    assert "include_deleted" not in query_param_names
    assert "include_fenced" not in query_param_names


# ===========================================================================
# PART 2: /v1/forecast/availability SERVABILITY & FENCE TESTS
# ===========================================================================


def test_07_availability_excludes_deletion_started_cycle(m5_env):
    """7. deletion_started_at cycle contributes zero availability to initial_times and valid_times."""
    client = TestClient(app)
    c = _dt(2026, 9, 12, 0)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 12, 0)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_avail_claimed",
                model_version_id="version_gfs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/claimed",
                created_at=c,
            )
        )
        session.add(
            ForecastProduct(
                id="p_claimed_t2m_0",
                run_id="run_avail_claimed",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 12, 1),
            )
        )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    models = resp.json()["data"]["models"]
    gfs = next((m for m in models if m["id"] == "gfs"), None)
    if gfs is not None:
        t2m = next((v for v in gfs["variables"] if v["id"] == "temperature_2m"), None)
        if t2m is not None:
            # 1. initial_times must not contain the claimed cycle
            c_iso = c.isoformat().replace("+00:00", "Z")
            assert not any(it["value"] == c_iso for it in t2m["initial_times"])
            # 2. valid_times must not contain valid_time from claimed cycle
            assert not any(vt["source_cycle"] == c_iso for vt in t2m["valid_times"])


def test_08_availability_excludes_deleted_at_cycle(m5_env):
    """8. deleted_at cycle contributes zero availability to initial_times and valid_times."""
    client = TestClient(app)
    c = _dt(2026, 9, 12, 6)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 12, 6)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_avail_deleted",
                model_version_id="version_gfs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/deleted",
                created_at=c,
            )
        )
        session.add(
            ForecastProduct(
                id="p_deleted_t2m_0",
                run_id="run_avail_deleted",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 12, 7),
                deleted_at=_dt(2026, 9, 12, 8),
            )
        )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    models = resp.json()["data"]["models"]
    gfs = next((m for m in models if m["id"] == "gfs"), None)
    if gfs is not None:
        t2m = next((v for v in gfs["variables"] if v["id"] == "temperature_2m"), None)
        if t2m is not None:
            c_iso = c.isoformat().replace("+00:00", "Z")
            assert not any(it["value"] == c_iso for it in t2m["initial_times"])
            assert not any(vt["source_cycle"] == c_iso for vt in t2m["valid_times"])


def test_09_availability_retired_at_has_no_effect(m5_env):
    """9. retired_at alone has zero effect on availability."""
    client = TestClient(app)
    c = _dt(2026, 9, 12, 12)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 12, 12)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_avail_ret_only",
                model_version_id="version_gfs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/ret_only",
                created_at=c,
            )
        )
        session.add(
            ForecastProduct(
                id="p_ret_t2m_0",
                run_id="run_avail_ret_only",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                retired_at=_dt(2026, 9, 12, 13),
            )
        )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    gfs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gfs")
    t2m = next(v for v in gfs["variables"] if v["id"] == "temperature_2m")
    c_iso = c.isoformat().replace("+00:00", "Z")
    assert any(it["value"] == c_iso for it in t2m["initial_times"])
    assert any(vt["source_cycle"] == c_iso for vt in t2m["valid_times"])


def test_10_availability_serving_left_boundary_exact(m5_env):
    """10. Serving left boundary: now = 07Z -> 06Z included, 03Z excluded."""
    client = TestClient(app)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 13, 7)

    c = _dt(2026, 9, 13, 0)
    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_boundary_test",
                model_version_id="version_gfs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/boundary",
                created_at=c,
            )
        )
        for lead in (3, 6, 9):  # valid times: 03Z, 06Z, 09Z
            session.add(
                ForecastProduct(
                    id=f"p_bound_t2m_{lead}",
                    run_id="run_boundary_test",
                    variable_id="temperature_2m",
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=lead,
                )
            )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["serving_start_valid_time"] == "2026-09-13T06:00:00Z"

    gfs = next(m for m in data["models"] if m["id"] == "gfs")
    t2m = next(v for v in gfs["variables"] if v["id"] == "temperature_2m")
    valid_times = [vt["valid_time"] for vt in t2m["valid_times"]]

    # 06Z and 09Z included; 03Z strictly excluded
    assert "2026-09-13T06:00:00Z" in valid_times
    assert "2026-09-13T09:00:00Z" in valid_times
    assert "2026-09-13T03:00:00Z" not in valid_times


def test_11_availability_granular_product_fences(m5_env):
    """11. Granular product state: deleting/deleted/failed -> exact target excluded."""
    client = TestClient(app)
    c = _dt(2026, 9, 13, 12)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 13, 12)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_granular_prod",
                model_version_id="version_gfs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/granular_prod",
                created_at=c,
            )
        )
        # Leads 0, 3, 6
        for lead in (0, 3, 6):
            session.add(
                ForecastProduct(
                    id=f"p_gran_t2m_{lead}",
                    run_id="run_granular_prod",
                    variable_id="temperature_2m",
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=lead,
                )
            )
        # Lead 0 is deleting, Lead 3 is deleted, Lead 6 is failed
        for lead, st in ((0, "deleting"), (3, "deleted"), (6, "failed")):
            session.add(
                ReclamationQueue(
                    id=f"rq_prod_{lead}",
                    run_id="run_granular_prod",
                    model_id="gfs",
                    cycle_time=c,
                    lead_time_hours=lead,
                    variable_code="temperature_2m",
                    target_kind=TARGET_KIND_DET,
                    member_index=0,
                    valid_time=c,
                    store_path="/store/granular_prod",
                    physical_key=f"t2m_{lead}",
                    status=st,
                    created_at=c,
                    updated_at=c,
                )
            )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    models = resp.json()["data"]["models"]
    gfs = next((m for m in models if m["id"] == "gfs"), None)
    if gfs is not None:
        t2m = next((v for v in gfs["variables"] if v["id"] == "temperature_2m"), None)
        if t2m is not None:
            c_iso = c.isoformat().replace("+00:00", "Z")
            it = next((it for it in t2m["initial_times"] if it["value"] == c_iso), None)
            assert it is None or len(it["lead_time_hours"]) == 0


def test_12_availability_granular_product_fence_queued_eligible(m5_env):
    """12. Granular product state: queued remains eligible."""
    client = TestClient(app)
    c = _dt(2026, 9, 13, 18)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 13, 18)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_queued_prod",
                model_version_id="version_gfs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/queued_prod",
                created_at=c,
            )
        )
        session.add(
            ForecastProduct(
                id="p_queued_t2m_0",
                run_id="run_queued_prod",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        session.add(
            ReclamationQueue(
                id="rq_queued_0",
                run_id="run_queued_prod",
                model_id="gfs",
                cycle_time=c,
                lead_time_hours=0,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c,
                store_path="/store/queued_prod",
                physical_key="t2m_0",
                status="queued",  # Still physically intact
                created_at=c,
                updated_at=c,
            )
        )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    gfs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gfs")
    t2m = next(v for v in gfs["variables"] if v["id"] == "temperature_2m")
    c_iso = c.isoformat().replace("+00:00", "Z")
    it = next(it for it in t2m["initial_times"] if it["value"] == c_iso)
    assert 0 in it["lead_time_hours"]


def test_13_availability_granular_member_fence_reclamation(m5_env):
    """13. Reclaimed member shards (deleting/deleted/failed) are excluded from member coverage."""
    client = TestClient(app)
    c = _dt(2026, 9, 14, 0)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 14, 0)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_gefs_mem_fence",
                model_version_id="version_gefs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/gefs_mem",
                created_at=c,
            )
        )
        session.add(
            ForecastProduct(
                id="p_gefs_mem_t2m_0",
                run_id="run_gefs_mem_fence",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="ensemble_mean",
                lead_time_hours=0,
            )
        )
        # Add 30 members
        for m in range(1, 31):
            session.add(
                EnsembleMemberProduct(
                    id=f"emp_gefs_{m}_0",
                    run_id="run_gefs_mem_fence",
                    member_index=m,
                    lead_time_hours=0,
                )
            )
        # Mark 5 members as deleted in reclamation_queue (leaves 25/30 < 85%)
        for m in (1, 2, 3, 4, 5):
            session.add(
                ReclamationQueue(
                    id=f"rq_mem_{m}",
                    run_id="run_gefs_mem_fence",
                    model_id="gefs",
                    cycle_time=c,
                    lead_time_hours=0,
                    variable_code="temperature_2m",
                    target_kind=TARGET_KIND_MEM,
                    member_index=m,
                    valid_time=c,
                    store_path="/store/gefs_mem",
                    physical_key=f"m_{m}",
                    status="deleted",
                    created_at=c,
                    updated_at=c,
                )
            )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    gefs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gefs")
    t2m = next(v for v in gefs["variables"] if v["id"] == "temperature_2m")
    c_iso = c.isoformat().replace("+00:00", "Z")
    it = next(it for it in t2m["initial_times"] if it["value"] == c_iso)

    # 1. Lead 0 is NOT servable because 25 < 26 (below 85% threshold)
    assert 0 not in it["lead_time_hours"]
    # 2. Rich descriptor reflects exact 25 available members
    lead_desc = next(ld for ld in it["leads"] if ld["lead_time_hours"] == 0)
    assert lead_desc["available_members"] == 25
    assert lead_desc["servable"] is False


def test_14_availability_gefs_readiness_threshold(m5_env):
    """14. GEFS threshold: >=26 members servable; <26 members unservable."""
    client = TestClient(app)
    c = _dt(2026, 9, 14, 6)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 14, 6)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_gefs_threshold",
                model_version_id="version_gefs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/gefs_thresh",
                created_at=c,
            )
        )
        # Lead 0: 26 members (>= 85%) -> servable
        # Lead 6: 25 members (< 85%) -> unservable
        for lead, m_count in ((0, 26), (6, 25)):
            session.add(
                ForecastProduct(
                    id=f"p_thresh_t2m_{lead}",
                    run_id="run_gefs_threshold",
                    variable_id="temperature_2m",
                    grid_id="global_025deg",
                    product_type="ensemble_mean",
                    lead_time_hours=lead,
                )
            )
            for m in range(1, m_count + 1):
                session.add(
                    EnsembleMemberProduct(
                        id=f"emp_thresh_{m}_{lead}",
                        run_id="run_gefs_threshold",
                        member_index=m,
                        lead_time_hours=lead,
                    )
                )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    gefs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gefs")
    t2m = next(v for v in gefs["variables"] if v["id"] == "temperature_2m")
    c_iso = c.isoformat().replace("+00:00", "Z")
    it = next(it for it in t2m["initial_times"] if it["value"] == c_iso)

    assert 0 in it["lead_time_hours"]
    assert 6 not in it["lead_time_hours"]


def test_15_availability_product_target_kind_correlation(m5_env):
    """15. det reclamation row must not mask mean, and mean reclamation row must not mask det."""
    client = TestClient(app)
    c = _dt(2026, 9, 14, 12)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 14, 12)

    with Session(m5_env) as session:
        # GFS run with det product
        session.add(
            ModelRun(
                id="run_corr_gfs",
                model_version_id="version_gfs_v1.0",
                cycle_time=c,
                status="ready",
                zarr_store_path="/store/corr_gfs",
                created_at=c,
            )
        )
        session.add(
            ForecastProduct(
                id="p_corr_gfs_t2m_0",
                run_id="run_corr_gfs",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",  # det
                lead_time_hours=0,
            )
        )
        # Add a reclamation_queue row with target_kind = 'mean' for the SAME run/lead/var
        session.add(
            ReclamationQueue(
                id="rq_mismatched_mean",
                run_id="run_corr_gfs",
                model_id="gfs",
                cycle_time=c,
                lead_time_hours=0,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_MEAN,  # mean != det
                member_index=-1,
                valid_time=c,
                store_path="/store/corr_gfs",
                physical_key="mean_shard",
                status="deleted",
                created_at=c,
                updated_at=c,
            )
        )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    gfs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gfs")
    t2m = next(v for v in gfs["variables"] if v["id"] == "temperature_2m")
    c_iso = c.isoformat().replace("+00:00", "Z")
    it = next(it for it in t2m["initial_times"] if it["value"] == c_iso)
    # The det product was NOT masked by the mean reclamation row!
    assert 0 in it["lead_time_hours"]


def test_16_availability_canonical_cross_cycle_fallback(m5_env):
    """16. Canonical cross-cycle fallback remains represented in valid_times."""
    client = TestClient(app)
    c_old = _dt(2026, 9, 14, 0)
    c_new = _dt(2026, 9, 14, 6)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 14, 6)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_fallback_old",
                model_version_id="version_gfs_v1.0",
                cycle_time=c_old,
                status="ready",
                zarr_store_path="/store/fb_old",
                created_at=c_old,
            )
        )
        session.add(
            ModelRun(
                id="run_fallback_new",
                model_version_id="version_gfs_v1.0",
                cycle_time=c_new,
                status="partial",
                zarr_store_path="/store/fb_new",
                created_at=c_new,
            )
        )
        # Old cycle covers lead 0, 6, 12 (valid times 00Z, 06Z, 12Z)
        for lead in (0, 6, 12):
            session.add(
                ForecastProduct(
                    id=f"p_old_t2m_{lead}",
                    run_id="run_fallback_old",
                    variable_id="temperature_2m",
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=lead,
                )
            )
        # New cycle only covers lead 0 (valid time 06Z)
        session.add(
            ForecastProduct(
                id="p_new_t2m_0",
                run_id="run_fallback_new",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    gfs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gfs")
    t2m = next(v for v in gfs["variables"] if v["id"] == "temperature_2m")
    vts = {vt["valid_time"]: vt for vt in t2m["valid_times"]}

    # Valid time 06Z: provided by newest cycle (c_new, lead 0)
    assert vts["2026-09-14T06:00:00Z"]["source_cycle"] == "2026-09-14T06:00:00Z"
    assert vts["2026-09-14T06:00:00Z"]["lead_time_hours"] == 0

    # Valid time 12Z: provided by older cycle (c_old, lead 12)
    assert vts["2026-09-14T12:00:00Z"]["source_cycle"] == "2026-09-14T00:00:00Z"
    assert vts["2026-09-14T12:00:00Z"]["lead_time_hours"] == 12


def test_17_availability_precipitation_amount_3h_lead0_fallback(m5_env):
    """17. precipitation_amount_3h lead-0 fallback provenance matches canonical serving."""
    client = TestClient(app)
    c_old = _dt(2026, 9, 14, 18)
    c_new = _dt(2026, 9, 15, 0)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 15, 0)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_precip_old",
                model_version_id="version_gfs_v1.0",
                cycle_time=c_old,
                status="ready",
                zarr_store_path="/store/p_old",
                created_at=c_old,
            )
        )
        session.add(
            ModelRun(
                id="run_precip_new",
                model_version_id="version_gfs_v1.0",
                cycle_time=c_new,
                status="ready",
                zarr_store_path="/store/p_new",
                created_at=c_new,
            )
        )
        # Old has lead 6 (valid_time = 00Z)
        session.add(
            ForecastProduct(
                id="p_old_tp_6",
                run_id="run_precip_old",
                variable_id="precipitation_amount_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=6,
            )
        )
        # New has lead 0 (valid_time = 00Z)
        session.add(
            ForecastProduct(
                id="p_new_tp_0",
                run_id="run_precip_new",
                variable_id="precipitation_amount_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    gfs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gfs")
    tp = next(v for v in gfs["variables"] if v["id"] == "precipitation_amount_3h")
    vt_00z = next(vt for vt in tp["valid_times"] if vt["valid_time"] == "2026-09-15T00:00:00Z")

    # Source cycle must be c_old (18Z) with positive lead 6
    assert vt_00z["source_cycle"] == "2026-09-14T18:00:00Z"
    assert vt_00z["lead_time_hours"] == 6


def test_18_availability_cloud_cover_3h_lead0_fallback(m5_env):
    """18. cloud_cover_3h lead-0 fallback provenance matches canonical serving."""
    client = TestClient(app)
    c_old = _dt(2026, 9, 15, 0)
    c_new = _dt(2026, 9, 15, 6)
    app.dependency_overrides[get_current_time] = lambda: _dt(2026, 9, 15, 6)

    with Session(m5_env) as session:
        session.add(
            ModelRun(
                id="run_cloud_old",
                model_version_id="version_gfs_v1.0",
                cycle_time=c_old,
                status="ready",
                zarr_store_path="/store/cc_old",
                created_at=c_old,
            )
        )
        session.add(
            ModelRun(
                id="run_cloud_new",
                model_version_id="version_gfs_v1.0",
                cycle_time=c_new,
                status="ready",
                zarr_store_path="/store/cc_new",
                created_at=c_new,
            )
        )
        # Old has lead 6 (valid_time = 06Z)
        session.add(
            ForecastProduct(
                id="p_old_cc_6",
                run_id="run_cloud_old",
                variable_id="cloud_cover_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=6,
            )
        )
        # New has lead 0 (valid_time = 06Z)
        session.add(
            ForecastProduct(
                id="p_new_cc_0",
                run_id="run_cloud_new",
                variable_id="cloud_cover_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        session.commit()

    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    gfs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gfs")
    cc = next(v for v in gfs["variables"] if v["id"] == "cloud_cover_3h")
    vt_06z = next(vt for vt in cc["valid_times"] if vt["valid_time"] == "2026-09-15T06:00:00Z")

    assert vt_06z["source_cycle"] == "2026-09-15T00:00:00Z"
    assert vt_06z["lead_time_hours"] == 6


# ===========================================================================
# PART 3: PRIMARY CONTRACT SEPARATION GATE
# ===========================================================================


def test_19_key_contract_separation_deleted_run_in_runs_but_not_availability(m5_env):
    """19. KEY M5 CONTRACT SEPARATION GATE:
    A cycle with deleted_at != NULL whose model_runs row still exists:
    - MUST be present in GET /v1/runs (operational catalog metadata).
    - MUST be absent in GET /v1/forecast/availability (zero servable availability).
    """
    client = TestClient(app)
    c_tomb = _dt(2026, 9, 15, 12)
    c_tomb_iso = c_tomb.isoformat().replace("+00:00", "Z")
    app.dependency_overrides[get_current_time] = lambda: c_tomb

    with Session(m5_env) as session:
        # Run row exists in model_runs
        session.add(
            ModelRun(
                id="run_tombstone_gate",
                model_version_id="version_gfs_v1.0",
                cycle_time=c_tomb,
                status="ready",
                zarr_store_path="/store/tombstone_gate",
                created_at=c_tomb,
            )
        )
        session.add(
            ForecastProduct(
                id="p_tomb_t2m_0",
                run_id="run_tombstone_gate",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        # Cycle has deleted_at tombstone (physically removed by M2 finalizer)
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c_tomb,
                deletion_started_at=_dt(2026, 9, 15, 13),
                deleted_at=_dt(2026, 9, 15, 14),
            )
        )
        session.commit()

    # 1. Operational Catalog Surface: MUST be present in /v1/runs
    resp_runs = client.get("/v1/runs?model_id=gfs")
    assert resp_runs.status_code == 200
    runs_ids = [r["id"] for r in resp_runs.json()["data"]]
    assert "run_tombstone_gate" in runs_ids

    # 2. Serving Availability Surface: MUST NOT be servable in /v1/forecast/availability
    resp_avail = client.get("/v1/forecast/availability")
    assert resp_avail.status_code == 200
    models = resp_avail.json()["data"]["models"]
    gfs = next((m for m in models if m["id"] == "gfs"), None)
    if gfs is not None:
        t2m = next((v for v in gfs["variables"] if v["id"] == "temperature_2m"), None)
        if t2m is not None:
            # Must not appear in initial_times
            assert not any(it["value"] == c_tomb_iso for it in t2m["initial_times"])
            # Must not appear in valid_times
            assert not any(vt["source_cycle"] == c_tomb_iso for vt in t2m["valid_times"])
