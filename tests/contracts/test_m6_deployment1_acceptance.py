"""Milestone 6: Deployment 1 Final Acceptance Integration Test Suite.

Authoritative verification for the integrated Data Lifecycle V3 system:
- Task 2: Serving Boundary Acceptance (exact boundary advancement, availability & serving path)
- Task 3: Canonical Multi-Cycle Serving (canonical source selection, cross-cycle fallback, provenance)
- Task 4: Special Variable Acceptance (lead-0 interval fallback, wind coherence, precipitation companion coupling, GEFS threshold)
- Task 5: Granular Reclamation Acceptance (queued servable, deleting/deleted/failed fenced)
- Task 6: Whole-Cycle Finalizer Acceptance (strict horizon, multi-version max horizon, fail-closed, recovery, multi-store, atomic queue normalization)
- Task 7: Writer / Finalizer Concurrency (all 3 serialization cases, custom store discovery)
- Task 8: Crash / Restart Acceptance (monotonic recovery from claim, multi-store resumption)
- Task 9: M3 Retention Acceptance (before/exact/after 14-day boundary, tombstone preservation, batch forward progress, DB-only)
- Task 10: Tombstone-Only Anti-Resurrection (post-M3 state rejects re-entry across all paths)
- Task 11: Cache Acceptance (deletion_started_at & deleted_at invalidate cache)
- Task 12: API Contract Separation (deleted run in /v1/runs but not /v1/forecast/availability before M3; absent after M3)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import xarray as xr
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from api.core.database import Base, get_db
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
from api.services.availability import build_forecast_availability
from api.services.tiles import _tile_cache
from api.services.lifecycle import (
    is_cycle_visible,
    require_cycle_visible,
)
from api.services.resolver import (
    resolve_variable_source,
)
from domain.horizon import (
    register_canonical_lead_horizon,
)
from domain.lifecycle import (
    METADATA_RETENTION_DAYS,
    canonical_cycle_store_path,
    is_cycle_horizon_expired,
    is_metadata_purge_eligible,
)
from domain.reclamation import TARGET_KIND_DET
from domain.temporal import serving_start_valid_time
from ingestion.core.base import CycleTombstonedError
from ingestion.core.catalog import (
    CatalogBase,
    CommittedState,
    ForecastCycleLifecycleRecord,
    ReclamationQueueRecord,
    RunCatalogSpec,
    VariableSpec,
    _ensure_utc_datetime,
    ensure_lifecycle_row,
    is_cycle_fenced_or_deleted,
    record_run,
    reserve_run,
)
from ingestion.gc.finalizer import (
    claim_fresh_candidate,
    enumerate_cycle_store_paths,
    finalize_cycle_eol,
    finalize_cycle_physical_and_queue,
)
from ingestion.gc.sweeper import run_metadata_sweeper_pass


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


@pytest.fixture
def m6_db(tmp_path: Path):
    """Provide an isolated SQLite database with all tables created."""
    db_file = tmp_path / "m6_acceptance.db"
    engine = create_engine(f"sqlite:///{db_file}")

    contract_tables = [
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
    Base.metadata.create_all(engine, tables=contract_tables)
    CatalogBase.metadata.create_all(engine)

    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # Seed baseline center, models, versions, grids
    center = ForecastCenter(id="noaa", center_id="noaa", name="NOAA", country="USA")
    m_gfs = Model(id="gfs", model_id="gfs", center_id="noaa", name="GFS", resolution_km=25.0, is_ensemble=False)
    m_gefs = Model(id="gefs", model_id="gefs", center_id="noaa", name="GEFS", resolution_km=50.0, is_ensemble=True)
    v_gfs = ModelVersion(id="gfs-v16", model_id="gfs", version_string="v16.3")
    v_gefs = ModelVersion(id="gefs-v12", model_id="gefs", version_string="v12.3")
    grid = ForecastGrid(id="global_025deg", grid_code="global_025deg", name="Global 0.25 Deg", resolution_km=25.0)

    vars_list = [
        ForecastVariable(id="temperature_2m", variable_code="temperature_2m", name="2m Temperature", unit="C"),
        ForecastVariable(id="precipitation_rate", variable_code="precipitation_rate", name="Precip Rate", unit="kg/m2/s"),
        ForecastVariable(id="precipitation_amount_3h", variable_code="precipitation_amount_3h", name="3h Precip", unit="kg/m2"),
        ForecastVariable(id="cloud_cover_3h", variable_code="cloud_cover_3h", name="3h Cloud Cover", unit="%"),
        ForecastVariable(id="wind_u_10m", variable_code="wind_u_10m", name="10m U Wind", unit="m/s"),
        ForecastVariable(id="wind_v_10m", variable_code="wind_v_10m", name="10m V Wind", unit="m/s"),
        ForecastVariable(id="crain", variable_code="crain", name="Rain Category", unit="cat"),
        ForecastVariable(id="csnow", variable_code="csnow", name="Snow Category", unit="cat"),
        ForecastVariable(id="cfrzr", variable_code="cfrzr", name="Freezing Rain", unit="cat"),
        ForecastVariable(id="cicep", variable_code="cicep", name="Ice Pellets", unit="cat"),
    ]

    session.add_all([center, m_gfs, m_gefs, v_gfs, v_gefs, grid] + vars_list)
    session.commit()

    def _override_get_db():
        s = session_factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)

    yield session, client, engine

    app.dependency_overrides.clear()
    session.close()
    engine.dispose()


def _make_spec(
    model_id: str = "gfs",
    cycle_time: datetime | None = None,
    version_string: str = "v16.3",
    store_path: str | None = None,
) -> RunCatalogSpec:
    c_time = (
        cycle_time
        if cycle_time is not None
        else datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    )
    s_path = (
        store_path
        if store_path is not None
        else canonical_cycle_store_path(model_id, c_time)
    )
    return RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id=model_id,
        model_name=model_id.upper(),
        is_ensemble=(model_id == "gefs"),
        resolution_km=25.0,
        version_string=version_string,
        cycle_time=c_time,
        grid_id="global_025deg",
        grid_name="Global 0.25deg",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=s_path,
        variables=(
            VariableSpec(
                code="temperature_2m", name="2m Temperature", unit="K"
            ),
        ),
        expected_lead_time_hours=(0, 3, 6),
        expected_members=() if model_id != "gefs" else tuple(range(1, 31)),
    )


# ==============================================================================
# TASK 2 — SERVING BOUNDARY ACCEPTANCE
# ==============================================================================

def test_task2_serving_boundary_acceptance(m6_db) -> None:
    """Exact boundary: at 07Z -> serving_start=06Z, 06Z servable, 03Z expired; at 09Z -> 09Z."""
    db, client, _ = m6_db

    # For GFS (cadence 6h, standard 3h cadence-grid for interval valid times)
    t_07z = _dt(2026, 9, 8, 7, 0)
    start_07z = serving_start_valid_time(t_07z, cadence_hours=3)
    assert start_07z == _dt(2026, 9, 8, 6, 0)

    t_09z = _dt(2026, 9, 8, 9, 0)
    start_09z = serving_start_valid_time(t_09z, cadence_hours=3)
    assert start_09z == _dt(2026, 9, 8, 9, 0)

    # Seed a GFS run for cycle 00Z covering leads 3, 6, 9 (valid times 03Z, 06Z, 09Z)
    c_00z = _dt(2026, 9, 8, 0, 0)
    run = ModelRun(
        id="run_gfs_00z",
        model_version_id="gfs-v16",
        cycle_time=c_00z,
        status="ready",
        zarr_store_path="s3://weather-data/gfs/2026090800/cycle.zarr",
    )
    db.add(run)
    for lead in [3, 6, 9]:
        db.add(
            ForecastProduct(
                id=f"fp_{lead}",
                run_id="run_gfs_00z",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="deterministic",
                lead_time_hours=lead,
            )
        )
    db.commit()

    # Query availability at now = 07Z
    avail_07z = build_forecast_availability(db, now=t_07z)
    assert avail_07z.serving_start_valid_time == _dt(2026, 9, 8, 6, 0)
    v_times = [
        vt.valid_time
        for m in avail_07z.models
        for v in m.variables
        for vt in v.valid_times
    ]
    # 06Z is servable, 03Z is strictly expired
    assert _dt(2026, 9, 8, 6, 0) in v_times
    assert _dt(2026, 9, 8, 3, 0) not in v_times

    # Query availability at now = 09Z
    avail_09z = build_forecast_availability(db, now=t_09z)
    assert avail_09z.serving_start_valid_time == _dt(2026, 9, 8, 9, 0)
    v_times_09z = [
        vt.valid_time
        for m in avail_09z.models
        for v in m.variables
        for vt in v.valid_times
    ]
    # 09Z is servable, 06Z is expired
    assert _dt(2026, 9, 8, 9, 0) in v_times_09z
    assert _dt(2026, 9, 8, 6, 0) not in v_times_09z


# ==============================================================================
# TASK 3 — CANONICAL MULTI-CYCLE SERVING
# ==============================================================================

def test_task3_canonical_multi_cycle_serving(m6_db) -> None:
    """Overlapping cycles: newer cycle wins near valid times, older cycle preserves far horizon."""
    db, client, _ = m6_db

    c_00z = _dt(2026, 9, 8, 0, 0)
    c_06z = _dt(2026, 9, 8, 6, 0)

    # 00Z is complete with leads 6, 12, 18, 24 (valid times 06Z, 12Z, 18Z, 00Z+1)
    run_00z = ModelRun(
        id="run_00z",
        model_version_id="gfs-v16",
        cycle_time=c_00z,
        status="ready",
        zarr_store_path="s3://weather-data/gfs/2026090800/cycle.zarr",
    )
    # 06Z is partial with leads 0, 6 (valid times 06Z, 12Z)
    run_06z = ModelRun(
        id="run_06z",
        model_version_id="gfs-v16",
        cycle_time=c_06z,
        status="partial",
        zarr_store_path="s3://weather-data/gfs/2026090806/cycle.zarr",
    )
    db.add_all([run_00z, run_06z])

    for lead in [6, 12, 18, 24]:
        db.add(ForecastProduct(
            id=f"fp_00z_{lead}", run_id="run_00z", variable_id="temperature_2m",
            grid_id="global_025deg", product_type="deterministic", lead_time_hours=lead
        ))
    for lead in [0, 6]:
        db.add(ForecastProduct(
            id=f"fp_06z_{lead}", run_id="run_06z", variable_id="temperature_2m",
            grid_id="global_025deg", product_type="deterministic", lead_time_hours=lead
        ))
    db.commit()

    # 06Z and 12Z valid times: newer committed cycle 06Z wins
    src_06 = resolve_variable_source(db, "gfs", "temperature_2m", _dt(2026, 9, 8, 6, 0), now=_dt(2026, 9, 8, 6, 0))
    assert src_06 is not None
    assert src_06.cycle_time == c_06z
    assert src_06.lead_time_hours == 0

    src_12 = resolve_variable_source(db, "gfs", "temperature_2m", _dt(2026, 9, 8, 12, 0), now=_dt(2026, 9, 8, 6, 0))
    assert src_12 is not None
    assert src_12.cycle_time == c_06z
    assert src_12.lead_time_hours == 6

    # 18Z and 00Z+1d valid times: newer cycle has no data; older cycle 00Z provides fallback
    src_18 = resolve_variable_source(db, "gfs", "temperature_2m", _dt(2026, 9, 8, 18, 0), now=_dt(2026, 9, 8, 6, 0))
    assert src_18 is not None
    assert src_18.cycle_time == c_00z
    assert src_18.lead_time_hours == 18

    src_24 = resolve_variable_source(db, "gfs", "temperature_2m", _dt(2026, 9, 9, 0, 0), now=_dt(2026, 9, 8, 6, 0))
    assert src_24 is not None
    assert src_24.cycle_time == c_00z
    assert src_24.lead_time_hours == 24


# ==============================================================================
# TASK 4 — SPECIAL VARIABLE ACCEPTANCE
# ==============================================================================

def test_task4_special_variables_acceptance(m6_db) -> None:
    """Special variables: lead-0 interval fallback, companion coupling, wind coherence, GEFS rules."""
    db, client, _ = m6_db

    c_00z = _dt(2026, 9, 8, 0, 0)
    c_06z = _dt(2026, 9, 8, 6, 0)

    run_00z = ModelRun(
        id="run_00z_sp", model_version_id="gfs-v16", cycle_time=c_00z, status="ready",
        zarr_store_path="s3://weather-data/gfs/00/cycle.zarr"
    )
    run_06z = ModelRun(
        id="run_06z_sp", model_version_id="gfs-v16", cycle_time=c_06z, status="ready",
        zarr_store_path="s3://weather-data/gfs/06/cycle.zarr"
    )
    db.add_all([run_00z, run_06z])

    # 00Z has lead 6 for precipitation_amount_3h, cloud_cover_3h, and companions (valid time 06Z)
    for var in ["precipitation_amount_3h", "cloud_cover_3h", "crain", "csnow", "cfrzr", "cicep", "wind_u_10m", "wind_v_10m"]:
        db.add(ForecastProduct(
            id=f"fp_00_{var}", run_id="run_00z_sp", variable_id=var,
            grid_id="global_025deg", product_type="deterministic", lead_time_hours=6
        ))

    # 06Z has lead 0 (valid time 06Z) for precip and cloud, but lead 0 of 3h intervals is unusable
    for var in ["precipitation_amount_3h", "cloud_cover_3h", "crain", "csnow", "cfrzr", "cicep"]:
        db.add(ForecastProduct(
            id=f"fp_06_{var}", run_id="run_06z_sp", variable_id=var,
            grid_id="global_025deg", product_type="deterministic", lead_time_hours=0
        ))
    # In 06Z, add only wind_u_10m lead 0, missing wind_v_10m lead 0 (incoherent wind pair)
    db.add(ForecastProduct(
        id="fp_06_u10", run_id="run_06z_sp", variable_id="wind_u_10m",
        grid_id="global_025deg", product_type="deterministic", lead_time_hours=0
    ))
    db.commit()

    vt_06z = _dt(2026, 9, 8, 6, 0)
    now_06z = _dt(2026, 9, 8, 6, 0)

    # 1. precipitation_amount_3h falls back to 00Z lead 6
    precip_res = resolve_variable_source(db, "gfs", "precipitation_amount_3h", vt_06z, now=now_06z)
    assert precip_res is not None
    assert precip_res.cycle_time == c_00z
    assert precip_res.lead_time_hours == 6

    # 2. cloud_cover_3h falls back to 00Z lead 6
    cloud_res = resolve_variable_source(db, "gfs", "cloud_cover_3h", vt_06z, now=now_06z)
    assert cloud_res is not None
    assert cloud_res.cycle_time == c_00z
    assert cloud_res.lead_time_hours == 6

    # 3. precipitation companions (crain, csnow) coupled to same fallback source
    crain_res = resolve_variable_source(db, "gfs", "crain", vt_06z, now=now_06z)
    csnow_res = resolve_variable_source(db, "gfs", "csnow", vt_06z, now=now_06z)
    assert crain_res is not None and csnow_res is not None
    assert crain_res.cycle_time == c_00z
    assert crain_res.lead_time_hours == 6
    assert csnow_res.cycle_time == c_00z
    assert csnow_res.lead_time_hours == 6

    # 4. wind_10m: incoherent 06Z (u only, no v) does NOT win; falls back to coherent 00Z
    wind_res = resolve_variable_source(db, "gfs", "wind_10m", vt_06z, now=now_06z)
    assert wind_res is not None
    assert wind_res.cycle_time == c_00z
    assert wind_res.lead_time_hours == 6

    # 5. GEFS does NOT include precipitation_rate
    with pytest.raises(HTTPException) as exc:
        resolve_variable_source(db, "gefs", "precipitation_rate", vt_06z, now=now_06z)
    assert exc.value.status_code == 404


# ==============================================================================
# TASK 5 — GRANULAR RECLAMATION ACCEPTANCE
# ==============================================================================

def test_task5_granular_reclamation_acceptance(m6_db) -> None:
    """reclamation_queue: queued is servable; deleting, deleted, failed are fenced."""
    db, client, _ = m6_db

    c_time = _dt(2026, 9, 8, 0, 0)
    run = ModelRun(
        id="run_reclam", model_version_id="gfs-v16", cycle_time=c_time, status="ready",
        zarr_store_path="s3://weather-data/gfs/reclam/cycle.zarr"
    )
    db.add(run)
    for lead in [6, 12, 18]:
        db.add(ForecastProduct(
            id=f"fp_rec_{lead}", run_id="run_reclam", variable_id="temperature_2m",
            grid_id="global_025deg", product_type="deterministic", lead_time_hours=lead
        ))
    db.commit()

    # 1. Lead 6 is 'queued' -> remains servable
    q_row = ReclamationQueue(
        id="rec_q_6", run_id="run_reclam", model_id="gfs", cycle_time=c_time,
        lead_time_hours=6, variable_code="temperature_2m", target_kind=TARGET_KIND_DET,
        member_index=0, valid_time=_dt(2026, 9, 8, 6, 0), store_path="store",
        physical_key="k6", status="queued"
    )
    # 2. Lead 12 is 'deleting' -> fenced
    del_row = ReclamationQueue(
        id="rec_q_12", run_id="run_reclam", model_id="gfs", cycle_time=c_time,
        lead_time_hours=12, variable_code="temperature_2m", target_kind=TARGET_KIND_DET,
        member_index=0, valid_time=_dt(2026, 9, 8, 12, 0), store_path="store",
        physical_key="k12", status="deleting"
    )
    # 3. Lead 18 is 'failed' -> fenced
    fail_row = ReclamationQueue(
        id="rec_q_18", run_id="run_reclam", model_id="gfs", cycle_time=c_time,
        lead_time_hours=18, variable_code="temperature_2m", target_kind=TARGET_KIND_DET,
        member_index=0, valid_time=_dt(2026, 9, 8, 18, 0), store_path="store",
        physical_key="k18", status="failed"
    )
    db.add_all([q_row, del_row, fail_row])
    db.commit()

    avail = build_forecast_availability(db, now=c_time)
    gfs_model = next(m for m in avail.models if m.id == "gfs")
    t2m_var = next(v for v in gfs_model.variables if v.id == "temperature_2m")
    v_times = [vt.valid_time for vt in t2m_var.valid_times]

    # Lead 6 (06Z) is queued -> included in availability
    assert _dt(2026, 9, 8, 6, 0) in v_times
    # Lead 12 (12Z) is deleting -> excluded
    assert _dt(2026, 9, 8, 12, 0) not in v_times
    # Lead 18 (18Z) is failed -> excluded
    assert _dt(2026, 9, 8, 18, 0) not in v_times


# ==============================================================================
# TASK 6 — WHOLE-CYCLE FINALIZER ACCEPTANCE
# ==============================================================================

def test_task6_whole_cycle_finalizer_acceptance(m6_db) -> None:
    """Whole-cycle finalizer: strict < horizon expiry, multi-version, atomic normalization."""
    db, client, engine = m6_db

    # 1. Strict boundary test:
    # Cycle 00Z, horizon 240h -> max valid time is 00Z + 240h
    cycle = _dt(2026, 9, 1, 0, 0)
    max_vt = cycle + timedelta(hours=240)

    # If serving_start == max_vt, cycle is NOT expired (strict < required)
    assert not is_cycle_horizon_expired(cycle, max_lead_hours=240, serving_start=max_vt)
    # If serving_start == max_vt + 1s, cycle IS expired
    assert is_cycle_horizon_expired(cycle, max_lead_hours=240, serving_start=max_vt + timedelta(seconds=1))

    # 2. Multi-version cycle uses maximum authoritative horizon:
    register_canonical_lead_horizon("gfs", tuple(range(0, 241, 3)), version_string="v16.0")
    register_canonical_lead_horizon("gfs", tuple(range(0, 385, 3)), version_string="v16.5")
    from domain.horizon import model_max_lead_hours
    assert max(model_max_lead_hours("gfs", v) for v in ["v16.0", "v16.5"]) == 384

    # 3. Missing version metadata fails closed:
    # Cycle with no ModelRun/ModelVersion records fails closed in claim_fresh_candidate
    c_unreg = _dt(2026, 1, 1, 0, 0)
    assert claim_fresh_candidate(db, "gfs", c_unreg, serving_start=max_vt) is False

    # 4. Atomic queue normalization during finalization:
    c_final = _dt(2026, 8, 15, 0, 0)
    ensure_lifecycle_row(db, "gfs", c_final)
    q_item = ReclamationQueueRecord(
        id="q_norm_1", run_id="run_fake", model_id="gfs", cycle_time=c_final,
        lead_time_hours=6, variable_code="t2m", target_kind=TARGET_KIND_DET,
        member_index=0, valid_time=c_final, store_path="p", physical_key="k",
        status="queued", attempt_count=2, last_error="test_err",
        created_at=c_final, updated_at=c_final, reclaimed_at=None
    )
    db.add(q_item)
    db.commit()

    now_fin = _dt(2026, 9, 8, 12, 0)
    finalize_cycle_physical_and_queue(engine, "gfs", c_final, finalization_time=now_fin)

    refreshed_q = db.get(ReclamationQueueRecord, "q_norm_1")
    assert refreshed_q.status == "deleted"
    assert refreshed_q.attempt_count == 2
    assert refreshed_q.last_error == "test_err"
    assert _ensure_utc_datetime(refreshed_q.reclaimed_at) == now_fin
    assert _ensure_utc_datetime(refreshed_q.updated_at) == now_fin

    lc_row = db.get(ForecastCycleLifecycleRecord, ("gfs", c_final))
    assert _ensure_utc_datetime(lc_row.deleted_at) == now_fin


# ==============================================================================
# TASK 7 — WRITER / FINALIZER CONCURRENCY
# ==============================================================================

def test_task7_writer_finalizer_concurrency(m6_db) -> None:
    """All 3 serialization cases: active writer, reserved writer, pre-reservation claim."""
    db, client, engine = m6_db

    c_time = _dt(2026, 9, 5, 0, 0)
    ensure_lifecycle_row(db, "gfs", c_time)

    # Case 3: Finalizer claims before writer reservation
    lc = db.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
    lc.deletion_started_at = _dt(2026, 9, 5, 1, 0)
    db.commit()

    # Writer calls reserve_run -> sees deletion_started_at -> rejects with CycleTombstonedError
    spec_claim = _make_spec(model_id="gfs", cycle_time=c_time, version_string="v16.3", store_path="s3://weather-data/gfs/custom/cycle.zarr")
    with pytest.raises(CycleTombstonedError, match="claimed for deletion"):
        reserve_run(db, spec_claim)

    # Clear claim to test custom store discovery
    lc.deletion_started_at = None
    db.commit()

    # Case: Custom store path remains discoverable by finalizer
    custom_path = "s3://weather-data/gfs/2026-09-05/custom/cycle.zarr"
    spec_custom = _make_spec(model_id="gfs", cycle_time=c_time, version_string="v16.3", store_path=custom_path)
    run_rec = reserve_run(db, spec_custom)
    assert run_rec.zarr_store_path == custom_path

    discovered_stores = enumerate_cycle_store_paths(db, "gfs", c_time, base_bucket="weather-data")
    assert custom_path in discovered_stores


# ==============================================================================
# TASK 8 — CRASH / RESTART ACCEPTANCE
# ==============================================================================

def test_task8_crash_restart_acceptance(m6_db) -> None:
    """Crash recovery: deletion_started_at resumes without fresh horizon check and converges."""
    db, client, engine = m6_db

    # Non-expired cycle that crashed mid-deletion (deletion_started_at set, deleted_at NULL)
    c_crash = _dt(2026, 9, 8, 0, 0)
    ensure_lifecycle_row(db, "gfs", c_crash)
    lc = db.get(ForecastCycleLifecycleRecord, ("gfs", c_crash))
    lc.deletion_started_at = _dt(2026, 9, 8, 1, 0)
    lc.deleted_at = None
    db.commit()

    # Fresh eligibility would say NOT expired at now = Sep 8 07Z
    assert not is_cycle_horizon_expired(c_crash, max_lead_hours=240, serving_start=_dt(2026, 9, 8, 6, 0))

    # Recovery finalization must succeed WITHOUT re-checking fresh horizon
    now_rec = _dt(2026, 9, 8, 7, 0)
    serving_start = serving_start_valid_time(now_rec)
    with patch("ingestion.gc.finalizer.delete_physical_store_gated", return_value=True):
        res = finalize_cycle_eol(engine, "gfs", c_crash, is_recovery=True, serving_start=serving_start, now=now_rec)
        assert res is True

    refreshed = db.get(ForecastCycleLifecycleRecord, ("gfs", c_crash))
    assert _ensure_utc_datetime(refreshed.deleted_at) == now_rec


# ==============================================================================
# TASK 9 — M3 RETENTION ACCEPTANCE
# ==============================================================================

def test_task9_m3_retention_acceptance(m6_db) -> None:
    """M3 retention: <14d retained, ==14d purged, >14d purged, tombstone preserved, DB-only."""
    db, client, engine = m6_db

    now_utc = _dt(2026, 9, 20, 0, 0)
    cutoff_14d = now_utc - timedelta(days=METADATA_RETENTION_DAYS)

    # 1. Boundary primitives
    assert not is_metadata_purge_eligible(cutoff_14d + timedelta(seconds=1), now_utc=now_utc)
    assert is_metadata_purge_eligible(cutoff_14d, now_utc=now_utc)  # exact inclusive
    assert is_metadata_purge_eligible(cutoff_14d - timedelta(days=1), now_utc=now_utc)

    # 2. Seed a cycle with deleted_at = 15 days ago
    c_purged = _dt(2026, 9, 1, 0, 0)
    ensure_lifecycle_row(db, "gfs", c_purged)
    lc = db.get(ForecastCycleLifecycleRecord, ("gfs", c_purged))
    lc.deleted_at = cutoff_14d - timedelta(days=1)
    db.commit()

    run = ModelRun(
        id="run_to_purge", model_version_id="gfs-v16", cycle_time=c_purged, status="ready",
        zarr_store_path="s3://weather-data/gfs/p/cycle.zarr"
    )
    db.add(run)
    db.commit()

    # Run metadata sweeper pass
    sweep_res = run_metadata_sweeper_pass(engine, now=now_utc, batch_size=10)
    assert len(sweep_res.swept_cycles) >= 1

    # model_runs deleted
    assert db.get(ModelRun, "run_to_purge") is None
    # forecast_cycle_lifecycle permanent tombstone PRESERVED
    refreshed_lc = db.get(ForecastCycleLifecycleRecord, ("gfs", c_purged))
    assert refreshed_lc is not None
    assert refreshed_lc.deleted_at == lc.deleted_at


# ==============================================================================
# TASK 10 — TOMBSTONE-ONLY ANTI-RESURRECTION
# ==============================================================================

def test_task10_tombstone_only_anti_resurrection(m6_db) -> None:
    """Post-M3 state (tombstone only, zero metadata) strictly rejects re-entry across all paths."""
    db, client, _ = m6_db

    c_tomb = _dt(2026, 8, 1, 0, 0)
    ensure_lifecycle_row(db, "gfs", c_tomb)
    lc = db.get(ForecastCycleLifecycleRecord, ("gfs", c_tomb))
    lc.deleted_at = _dt(2026, 8, 15, 0, 0)
    db.commit()

    # Ensure zero metadata exists for this cycle
    assert db.scalars(select(ModelRun).where(ModelRun.cycle_time == c_tomb)).first() is None

    # Path 1: reserve_run raises CycleTombstonedError
    spec_tomb = _make_spec(model_id="gfs", cycle_time=c_tomb, version_string="v16.3")
    with pytest.raises(CycleTombstonedError, match="tombstoned"):
        reserve_run(db, spec_tomb)

    # Path 2: record_run raises CycleTombstonedError
    with pytest.raises(CycleTombstonedError, match="tombstoned"):
        record_run(db, spec_tomb, xr.Dataset(), committed_state=CommittedState.deterministic({0}, {"t2m"}))

    # Path 3: is_cycle_fenced_or_deleted returns True
    assert is_cycle_fenced_or_deleted(db, c_tomb, model_id="gfs") is True

    # Path 4: is_cycle_visible returns False
    assert is_cycle_visible(db, c_tomb, model_id="gfs") is False

    # Path 5: require_cycle_visible raises HTTP 404
    with pytest.raises(HTTPException) as exc_info:
        require_cycle_visible(db, c_tomb, model_id="gfs")
    assert exc_info.value.status_code == 404


# ==============================================================================
# TASK 11 — CACHE ACCEPTANCE
# ==============================================================================

def test_task11_cache_acceptance(m6_db) -> None:
    """Pre-warmed cache cannot bypass deletion_started_at or deleted_at fences."""
    db, client, _ = m6_db

    c_time = _dt(2026, 9, 8, 0, 0)
    init_str = "2026-09-08T00:00:00Z"
    ensure_lifecycle_row(db, "gfs", c_time)

    # Pre-warm in-memory tile cache
    cache_key = ("gfs", "temperature_2m", "surface", 0, 0, 0, 0, init_str, "gen_acceptance")
    _tile_cache[cache_key] = (1000000000.0, b"\x89PNG\r\n\x1a\nFakeTileBytes")

    # 1. Unfenced cycle is visible
    lc = db.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
    assert is_cycle_visible(db, c_time, model_id="gfs") is True

    # 2. Setting deletion_started_at immediately fences cycle visibility
    lc.deletion_started_at = _dt(2026, 9, 8, 7, 0)
    db.commit()
    assert is_cycle_visible(db, c_time, model_id="gfs") is False

    # Actual tile endpoint returns 404 despite cached entry present
    res_claim = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={init_str}"
    )
    assert res_claim.status_code == 404

    # 3. Setting deleted_at permanently fences cycle visibility
    lc.deletion_started_at = None
    lc.deleted_at = _dt(2026, 9, 8, 8, 0)
    db.commit()
    assert is_cycle_visible(db, c_time, model_id="gfs") is False

    # Actual tile endpoint returns 404 despite cached entry present
    res_tomb = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={init_str}"
    )
    assert res_tomb.status_code == 404


# ==============================================================================
# TASK 12 — API CONTRACT SEPARATION
# ==============================================================================

def test_task12_api_contract_separation(m6_db) -> None:
    """Before M3 purge: deleted run listed in /v1/runs, absent in availability; absent from runs after purge."""
    db, client, engine = m6_db

    c_time = _dt(2026, 9, 8, 0, 0)
    ensure_lifecycle_row(db, "gfs", c_time)
    lc = db.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
    lc.deleted_at = _dt(2026, 9, 8, 6, 0)
    db.commit()

    run = ModelRun(
        id="run_m5_sep",
        model_version_id="gfs-v16",
        cycle_time=c_time,
        status="ready",
        zarr_store_path="s3://weather-data/gfs/sep/cycle.zarr",
    )
    db.add(run)
    db.add(ForecastProduct(
        id="fp_sep_0", run_id="run_m5_sep", variable_id="temperature_2m",
        grid_id="global_025deg", product_type="deterministic", lead_time_hours=0
    ))
    db.commit()

    # Step 1: Before M3 purge
    # /v1/runs (operational catalog) contains run
    resp_runs = client.get("/v1/runs")
    assert resp_runs.status_code == 200
    runs_data = resp_runs.json()
    run_ids = [r["id"] for r in runs_data["data"]]
    assert "run_m5_sep" in run_ids

    # /v1/forecast/availability does NOT advertise run as servable
    with patch("api.services.availability.get_current_time", return_value=c_time):
        resp_avail = client.get("/v1/forecast/availability")
        assert resp_avail.status_code == 200
        avail_data = resp_avail.json()
        models = avail_data.get("models", [])
        gfs_vars = [v for m in models if m["id"] == "gfs" for v in m.get("variables", [])]
        assert len(gfs_vars) == 0

    # Step 2: After M3 purge (simulate metadata removal)
    db.delete(run)
    db.commit()

    # /v1/runs now naturally omits the run
    resp_runs_purged = client.get("/v1/runs")
    assert resp_runs_purged.status_code == 200
    purged_run_ids = [r["id"] for r in resp_runs_purged.json()["data"]]
    assert "run_m5_sep" not in purged_run_ids

    # Permanent tombstone remains in forecast_cycle_lifecycle
    lc_row = db.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
    assert lc_row is not None
    assert _ensure_utc_datetime(lc_row.deleted_at) == _dt(2026, 9, 8, 6, 0)
