"""Service-backed concurrency, locking, crash recovery, and E2E lifecycle tests for Phase 6D."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from ingestion.core.base import CycleTombstonedError
from ingestion.core.catalog import (
    CommittedState,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ModelVersionRecord,
    RunCatalogSpec,
    VariableSpec,
    ensure_lifecycle_row,
    record_run,
)
from ingestion.core.zarr_writer import write_dataset
from ingestion.gc.finalizer import run_lifecycle_bookkeeping_pass
from ingestion.gc.planner import plan_reclamation_pass
from ingestion.gc.worker import run_reclamation_worker_pass
from tests._integration_db import integration_db_url


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def _make_dataset(cycle_time: datetime, lead: int = 0) -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    temperature = np.full((1, 4, 4), 20.0, dtype=np.float32)
    precipitation = np.full((1, 4, 4), 0.5, dtype=np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), temperature),
            "precipitation_rate": (("lead_time_hours", "latitude", "longitude"), precipitation),
        },
        coords={
            "lead_time_hours": [lead],
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture(scope="function")
def postgres_gc_env(tmp_path):
    db_url = integration_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            "PostgreSQL test instance not running or reachable; skipping GC integration tests."
        )

    # Clean schema and migrate to head
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public;"))

    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../api"))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))
    command.upgrade(alembic_cfg, "head")

    yield engine, tmp_path
    engine.dispose()


def _seed_cycle(
    engine,
    tmp_path,
    cycle_time: datetime,
    status: str = "ready",
) -> tuple[str, str]:
    """Helper to write GFS and GEFS datasets to disk and record catalog rows."""
    c_str = cycle_time.strftime("%Y%m%d%H%M")
    gfs_path = str(tmp_path / f"gfs_{c_str}.zarr")
    gefs_path = str(tmp_path / f"gefs_{c_str}.zarr")

    ds_gfs = _make_dataset(cycle_time, 0)
    ds_gefs = _make_dataset(cycle_time, 0)

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
        cycle_time=cycle_time,
        grid_id="global_025deg",
        grid_name="Global",
        grid_resolution_km=25.0,
        zarr_store_path=gfs_path,
        variables=(
            VariableSpec("temperature_2m", "2-Meter Temperature", "°C"),
            VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h"),
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
        cycle_time=cycle_time,
        grid_id="global_025deg",
        grid_name="Global",
        grid_resolution_km=25.0,
        zarr_store_path=gefs_path,
        variables=(
            VariableSpec("temperature_2m", "2-Meter Temperature", "°C"),
            VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h"),
        ),
        expected_lead_time_hours=(0,),
        expected_members=(1,),
    )

    with Session(engine) as session:
        record_run(
            session,
            spec_gfs,
            ds_gfs,
            committed_state=CommittedState.deterministic({0}) if status == "ready" else None,
        )
        record_run(
            session,
            spec_gefs,
            ds_gefs,
            member=1,
            committed_state=CommittedState.ensemble({(1, 0)}, {1}) if status == "ready" else None,
        )

    return gfs_path, gefs_path


# ---------------------------------------------------------------------------
# Reader & Writer Advisory Lock Race Tests
def test_claimed_cycle_blocks_stale_ingestion_and_reclaims_granularly(postgres_gc_env):
    """V3 main line: claim (serving fence) -> plan -> work -> bookkeeping tombstone.

    1. The claim (deletion_started_at) blocks stale ingestion writers with
       CycleTombstonedError before touching storage.
    2. The planner enqueues the claimed cycle's units (fence = unprotected).
    3. The worker deletes them at (variable, valid_time) granularity.
    4. The bookkeeping pass derives deleted_at — with zero cycle-prefix deletion.
    5. Catalog metadata is retained for the sweeper.
    """
    engine, tmp_path = postgres_gc_env

    c0 = _dt(2026, 9, 1, 6)
    c1 = _dt(2026, 9, 2, 6)
    c2 = _dt(2026, 9, 2, 12)

    gfs_path, gefs_path = _seed_cycle(engine, tmp_path, c0, "ready")
    _seed_cycle(engine, tmp_path, c1, "ready")
    _seed_cycle(engine, tmp_path, c2, "ready")

    # Step 1: Claim the retirement fence (no physical deletion anywhere)
    with Session(engine) as session:
        ensure_lifecycle_row(session, "gfs", c0)
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c0))
        lc.deletion_started_at = _dt(2026, 9, 2, 12, 30)
        session.commit()

    # Step 2: Stale ingestion for the claimed cycle is rejected before storage
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
        grid_name="Global",
        grid_resolution_km=25.0,
        zarr_store_path=gfs_path,
        variables=(
            VariableSpec("temperature_2m", "2-Meter Temperature", "°C"),
            VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h"),
        ),
        expected_lead_time_hours=(0,),
    )
    with Session(engine) as session:
        with pytest.raises(CycleTombstonedError, match="claimed for deletion or already tombstoned"):
            record_run(session, spec, _make_dataset(c0, 0))

    # Step 3: Planner enqueues the claimed cycle's remaining units
    with Session(engine) as session:
        plan = plan_reclamation_pass(session, models=("gfs",), dry_run=False, now=_dt(2026, 9, 2, 12, 30))
        assert plan.enqueued_count > 0

    # Step 4: Worker reclaims them granularly
    with Session(engine) as session:
        w_res = run_reclamation_worker_pass(session, delete_enabled=True, now=_dt(2026, 9, 2, 12, 35))
        assert w_res.deleted_count > 0

    # Step 5: Bookkeeping derives the tombstone (zero physical storage operations)
    res_pass = run_lifecycle_bookkeeping_pass(engine, dry_run=False, now=_dt(2026, 9, 2, 13, 0))
    assert ("gfs", c0) in res_pass.finalized_cycles

    with Session(engine) as session:
        lc_final = session.get(ForecastCycleLifecycleRecord, ("gfs", c0))
        assert lc_final is not None
        assert lc_final.deletion_started_at is not None
        assert lc_final.deleted_at is not None

        # Catalog metadata is retained (the metadata sweeper owns purging)
        runs_c0 = (
            session.execute(
                select(ModelRunRecord)
                .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
                .where(
                    ModelVersionRecord.model_id == "gfs",
                    ModelRunRecord.cycle_time == c0,
                )
            )
            .scalars()
            .all()
        )
        assert len(runs_c0) == 1
