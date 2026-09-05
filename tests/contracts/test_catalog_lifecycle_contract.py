"""Stage 7D-C — Ingestion Catalog Publication ↔ API Serving Interpretation Contract.

Asserts producer/consumer compatibility across the PostgreSQL catalog lifecycle:
1. Partial run: Ingestion publishes subset of settled leads -> API availability exposes only committed leads.
2. Ready run: Ingestion commits full canonical horizon -> API resolves run as ready.
3. Retired cycle: Cycle marked retired -> API default resolution filters it out.
4. Deletion-fenced & tombstoned cycle: deletion_started_at blocks ingestion, deleted_at filters serving.
"""

from datetime import datetime, timezone
import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from api.models.entities import (
    Base,
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
from api.services.availability import build_forecast_availability
from api.services.lifecycle import filter_visible_runs
from api.services.point_forecast import resolve_latest_run_cycle_time
from ingestion.core.catalog import (
    CommittedState,
    RunCatalogSpec,
    VariableSpec,
    is_cycle_fenced_or_deleted,
    record_run,
)


@pytest.fixture
def db_session() -> Session:
    """Create an isolated in-memory SQLite database for contract assertions."""
    engine = create_engine("sqlite:///:memory:")
    # Create only the catalog and lifecycle tables (omitting PostGIS geometry tables)
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
    ]
    Base.metadata.create_all(engine, tables=contract_tables)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _build_test_dataset(leads: list[int]) -> xr.Dataset:
    """Construct a minimal xarray dataset for catalog ingestion testing."""
    lat = np.array([38.0, 39.0], dtype=float)
    lon = np.array([-107.0, -106.0], dtype=float)
    t2m = np.zeros((len(leads), len(lat), len(lon)), dtype=np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), t2m)
        },
        coords={
            "lead_time_hours": leads,
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_partial_run_publication_to_api_availability(db_session: Session) -> None:
    """Ingestion publishes settled leads -> API availability exposes only committed leads."""
    cycle = datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)
    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="Global Forecast System",
        resolution_km=25.0,
        is_ensemble=False,
        version_string="v16",
        cycle_time=cycle,
        expected_lead_time_hours=(0, 6, 12, 18),
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree Grid",
        grid_resolution_km=25.0,
        zarr_store_path="s3://weather-data/gfs/2026-09-03/00/cycle.zarr",
        variables=(
            VariableSpec(
                code="temperature_2m", name="2-Meter Temperature", unit="°C", source_code="t2m"
            ),
        ),
    )

    # Ingestion publishes partial dataset (leads 0, 6 of expected 0, 6, 12, 18)
    ds_partial = _build_test_dataset([0, 6])
    committed = CommittedState.deterministic({0, 6}, {"temperature_2m"})
    run = record_run(db_session, spec, ds_partial, committed_state=committed)
    assert run.status == "partial"

    # API Availability Consumer Contract Check
    avail = build_forecast_availability(db_session)
    assert len(avail.models) == 1
    gfs_model = avail.models[0]
    assert gfs_model.id == "gfs"
    assert len(gfs_model.variables) == 1
    t2m_var = gfs_model.variables[0]
    assert len(t2m_var.initial_times) == 1
    init_time = t2m_var.initial_times[0]
    assert init_time.lead_time_hours == [0, 6]


def test_ready_run_completion_to_api_resolution(db_session: Session) -> None:
    """Ingestion commits full canonical horizon -> API resolves run as ready."""
    cycle = datetime(2026, 9, 3, 6, 0, tzinfo=timezone.utc)
    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="GFS",
        resolution_km=25.0,
        is_ensemble=False,
        version_string="v16",
        cycle_time=cycle,
        expected_lead_time_hours=(0, 6, 12, 18),
        grid_id="global_025deg",
        grid_name="Global 0.25",
        grid_resolution_km=25.0,
        zarr_store_path="s3://weather-data/gfs/2026-09-03/06/cycle.zarr",
        variables=(
            VariableSpec(code="temperature_2m", name="Temp", unit="°C", source_code="t2m"),
        ),
    )

    # Ingestion commits all expected leads
    ds_full = _build_test_dataset([0, 6, 12, 18])
    committed = CommittedState.deterministic({0, 6, 12, 18}, {"temperature_2m"})
    run = record_run(db_session, spec, ds_full, committed_state=committed)
    assert run.status == "ready"

    # API Run Resolution Consumer Contract Check
    resolved_cycle = resolve_latest_run_cycle_time(db_session, "gfs")
    assert resolved_cycle == "2026-09-03T06:00:00Z"


def test_retired_cycle_filtered_by_api_lifecycle(db_session: Session) -> None:
    """Cycle marked retired -> API default resolution filters it out."""
    c_old = datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)
    c_new = datetime(2026, 9, 3, 6, 0, tzinfo=timezone.utc)

    spec_old = RunCatalogSpec(
        center_id="noaa", center_name="NOAA", center_country="USA", model_id="gfs",
        model_name="GFS", resolution_km=25.0, is_ensemble=False,
        version_string="v16", cycle_time=c_old, expected_lead_time_hours=(0, 6),
        grid_id="global_025deg", grid_name="Global 0.25", grid_resolution_km=25.0,
        zarr_store_path="s3://weather-data/gfs/2026-09-03/00/cycle.zarr",
        variables=(VariableSpec(code="temperature_2m", name="Temp", unit="°C", source_code="t2m"),),
    )
    spec_new = RunCatalogSpec(
        center_id="noaa", center_name="NOAA", center_country="USA", model_id="gfs",
        model_name="GFS", resolution_km=25.0, is_ensemble=False,
        version_string="v16", cycle_time=c_new, expected_lead_time_hours=(0, 6),
        grid_id="global_025deg", grid_name="Global 0.25", grid_resolution_km=25.0,
        zarr_store_path="s3://weather-data/gfs/2026-09-03/06/cycle.zarr",
        variables=(VariableSpec(code="temperature_2m", name="Temp", unit="°C", source_code="t2m"),),
    )

    ds = _build_test_dataset([0, 6])
    committed = CommittedState.deterministic({0, 6}, {"temperature_2m"})
    record_run(db_session, spec_old, ds, committed_state=committed)
    record_run(db_session, spec_new, ds, committed_state=committed)

    # Mark c_old as retired in forecast_cycle_lifecycle
    lifecycle_old = ForecastCycleLifecycle(
        model_id="gfs",
        cycle_time=c_old,
        retired_at=datetime.now(timezone.utc),
        retired_by_cycle_time=c_new,
    )
    db_session.add(lifecycle_old)
    db_session.commit()

    # API resolution should pick the active non-retired cycle c_new
    resolved_cycle = resolve_latest_run_cycle_time(db_session, "gfs")
    assert resolved_cycle == "2026-09-03T06:00:00Z"


def test_deletion_fenced_cycle_rejected_and_safely_skipped(db_session: Session) -> None:
    """deletion_started_at fences writers; deleted_at tombstone filters serving."""
    cycle = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    spec = RunCatalogSpec(
        center_id="noaa", center_name="NOAA", center_country="USA", model_id="gfs",
        model_name="GFS", resolution_km=25.0, is_ensemble=False,
        version_string="v16", cycle_time=cycle, expected_lead_time_hours=(0, 6),
        grid_id="global_025deg", grid_name="Global 0.25", grid_resolution_km=25.0,
        zarr_store_path="s3://weather-data/gfs/2026-09-03/12/cycle.zarr",
        variables=(VariableSpec(code="temperature_2m", name="Temp", unit="°C", source_code="t2m"),),
    )
    ds = _build_test_dataset([0, 6])
    committed = CommittedState.deterministic({0, 6}, {"temperature_2m"})
    run = record_run(db_session, spec, ds, committed_state=committed)

    # Set deletion_started_at fence on cycle (GC deletion claim)
    fence = ForecastCycleLifecycle(
        model_id="gfs",
        cycle_time=cycle,
        deletion_started_at=datetime.now(timezone.utc),
    )
    db_session.add(fence)
    db_session.commit()

    # Ingestion writer fence check
    assert is_cycle_fenced_or_deleted(db_session, cycle) is True

    # Set deleted_at tombstone (GC deletion completed)
    fence.deleted_at = datetime.now(timezone.utc)
    db_session.commit()

    # API query check: filter_visible_runs filters out deleted_at cycles
    stmt = select(ModelRun).where(ModelRun.id == run.id)
    filtered = filter_visible_runs(stmt)
    assert db_session.execute(filtered).first() is None
