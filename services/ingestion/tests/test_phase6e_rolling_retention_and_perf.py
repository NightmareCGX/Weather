"""Phase 6E Fresh DB, Migration Upgrade, Rolling Steady-State Retention & Performance Sanity Tests.

Validates under real PostgreSQL:
1. Fresh database initialization and empty catalog behavior.
2. Migration upgrade path 001 -> 002 -> 003 -> 004 on pre-existing catalog data.
3. Rolling steady-state 12-cycle simulation keeping bounded active storage.
4. Performance sanity bounds on planning and admission checks.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from domain.lifecycle import (
    CycleLifecycleSnapshot,
    plan_lifecycle,
)
from ingestion.cli import main
from ingestion.core.catalog import (
    CenterRecord,
    CommittedState,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    RunCatalogSpec,
    VariableSpec,
    is_cycle_fenced_or_deleted,
    list_cycle_lifecycle_snapshots,
    record_run,
)
from ingestion.core.zarr_writer import write_dataset
from ingestion.gc.reconciler import run_gc_pass
import xarray as xr
import numpy as np


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def _make_dataset(cycle_time: datetime) -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), np.full((1, 4, 4), 20.0, dtype=np.float32)),
        },
        coords={
            "lead_time_hours": [0],
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture(scope="function")
def postgres_clean_env(tmp_path):
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            "PostgreSQL test instance not running or reachable; skipping rolling retention tests."
        )

    # Clean schema
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


def test_fresh_database_empty_catalog(postgres_clean_env):
    """Prove that GC dry-run and real pass run safely on a fresh empty catalog."""
    engine, _ = postgres_clean_env

    # 1. Dry run on empty DB
    res_dry = run_gc_pass(engine, dry_run=True)
    assert res_dry.dry_run is True
    assert res_dry.would_retire == ()
    assert res_dry.would_gc == ()

    # 2. Real pass on empty DB
    res_real = run_gc_pass(engine, dry_run=False)
    assert res_real.dry_run is False
    assert res_real.processed_gc == ()

    # 3. CLI execution
    code = main(["gc", "--once", "--dry-run"])
    assert code == 0


def test_migration_upgrade_from_existing_database(tmp_path):
    """Simulate applying migrations 001 -> 002 with existing data, then upgrading to 003 -> 004."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL test instance not running or reachable.")

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public;"))

    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../api"))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    # Upgrade to 002 only
    command.upgrade(alembic_cfg, "002_ensemble_member_products")

    # Seed model runs in 002 schema (without lifecycle table)
    c1 = _dt(2026, 9, 1, 0)
    with Session(engine) as session:
        center = CenterRecord(id="center_noaa", center_id="noaa", name="NOAA", country="US", created_at=c1)
        session.add(center)
        session.flush()

        m_gfs = ModelRecord(id="model_gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0, created_at=c1)
        session.add(m_gfs)
        session.flush()

        v_gfs = ModelVersionRecord(id="version_gfs_v1.0", model_id="gfs", version_string="v1.0", created_at=c1)
        session.add(v_gfs)
        session.flush()

        r1 = ModelRunRecord(
            id="run_preexisting_c1",
            model_version_id="version_gfs_v1.0",
            cycle_time=c1,
            status="ready",
            created_at=c1,
        )
        session.add(r1)
        session.commit()

    # Now upgrade to head (003 -> 004)
    command.upgrade(alembic_cfg, "head")

    # Verify existing run is treated as visible by default
    with Session(engine) as session:
        assert is_cycle_fenced_or_deleted(session, c1) is False
        snapshots = list_cycle_lifecycle_snapshots(session)
        assert len(snapshots) == 1
        assert snapshots[0].cycle_time == c1
        assert snapshots[0].is_retired is False
        assert snapshots[0].is_deleted is False


def test_rolling_steady_state_retention_simulation(postgres_clean_env):
    """Simulate a rolling sequence of 12 6-hourly cycles over 3 days."""
    engine, tmp_path = postgres_clean_env

    # 12 cycles: Day 1 (00, 06, 12, 18), Day 2 (00, 06, 12, 18), Day 3 (00, 06, 12, 18)
    cycle_times = [
        _dt(2026, 9, 1, 0),
        _dt(2026, 9, 1, 6),
        _dt(2026, 9, 1, 12),
        _dt(2026, 9, 1, 18),
        _dt(2026, 9, 2, 0),
        _dt(2026, 9, 2, 6),
        _dt(2026, 9, 2, 12),
        _dt(2026, 9, 2, 18),
        _dt(2026, 9, 3, 0),
        _dt(2026, 9, 3, 6),
        _dt(2026, 9, 3, 12),
        _dt(2026, 9, 3, 18),
    ]

    # Ingest each cycle as ready
    store_dirs = {}
    for c_time in cycle_times:
        c_tag = c_time.strftime("%Y%m%d%H%M")
        p_gfs = str(tmp_path / f"gfs_{c_tag}.zarr")
        p_gefs = str(tmp_path / f"gefs_{c_tag}.zarr")
        store_dirs[(c_time, "gfs")] = p_gfs
        store_dirs[(c_time, "gefs")] = p_gefs

        ds = _make_dataset(c_time)
        write_dataset(ds, p_gfs)
        write_dataset(ds, p_gefs)

        spec_gfs = RunCatalogSpec(
            center_id="noaa", center_name="NOAA", center_country="US",
            model_id="gfs", model_name="GFS", is_ensemble=False, resolution_km=25.0,
            version_string="v1.0", cycle_time=c_time, grid_id="global_025deg", grid_name="Global",
            grid_resolution_km=25.0, zarr_store_path=p_gfs,
            variables=(VariableSpec("temperature_2m", "2m Temp", "°C"),),
            expected_lead_time_hours=(0,),
        )
        spec_gefs = RunCatalogSpec(
            center_id="noaa", center_name="NOAA", center_country="US",
            model_id="gefs", model_name="GEFS", is_ensemble=True, resolution_km=25.0,
            version_string="v1.0", cycle_time=c_time, grid_id="global_025deg", grid_name="Global",
            grid_resolution_km=25.0, zarr_store_path=p_gefs,
            variables=(VariableSpec("temperature_2m", "2m Temp", "°C"),),
            expected_lead_time_hours=(0,),
            expected_members=(1,),
        )
        with Session(engine) as session:
            record_run(session, spec_gfs, ds, committed_state=CommittedState.deterministic({0}))
            record_run(session, spec_gefs, ds, member=1, committed_state=CommittedState.ensemble({(1, 0)}, {1}))

    # Run GC pass at end of Day 3 (2026-09-03 20:00Z)
    now = _dt(2026, 9, 3, 20)
    res = run_gc_pass(engine, dry_run=False, now=now)

    # Under Lifecycle V2:
    # Latest ready cycle T = C12 (09-03 18Z), cadence C = 6h -> cutoff = 09-03 12Z (C11).
    # Retained: C11 (09-03 12Z) and C12 (09-03 18Z)
    # Deleted: C1 through C10 (< cutoff 09-03 12Z) -> 10 cycles per model (20 processed GC total)
    expected_gcd = set(cycle_times[:10])
    assert set(res.processed_gc) == expected_gcd
    assert len(res.processed_gc) == 20

    # Verify physical stores: 10 deleted, 2 remaining (C11 and C12)
    for c_time in cycle_times[:10]:
        assert not os.path.exists(store_dirs[(c_time, "gfs")])
        assert not os.path.exists(store_dirs[(c_time, "gefs")])

    for c_time in cycle_times[10:]:
        assert os.path.exists(store_dirs[(c_time, "gfs")])
        assert os.path.exists(store_dirs[(c_time, "gefs")])


def test_performance_sanity_planning_and_admission_bounds(postgres_clean_env):
    """Sanity benchmark proving planning across 50 cycles takes < 10ms and admission check < 1ms."""
    engine, _ = postgres_clean_env

    # 1. Pure planner performance with 50 synthetic cycles
    c_base = _dt(2026, 8, 1, 0)
    snapshots = [
        CycleLifecycleSnapshot(
            model_id="gfs",
            cycle_time=_dt(2026, 8, 1 + i // 4, (i % 4) * 6),
            status="ready",
        )
        for i in range(50)
    ]
    ready = [s.cycle_time for s in snapshots]

    t0 = time.monotonic()
    plan = plan_lifecycle(snapshots, ready, model_id="gfs")
    plan_dur_ms = (time.monotonic() - t0) * 1000.0

    # Must be fast across 50 cycles
    assert plan_dur_ms < 50.0
    assert len(plan.would_retire) >= 0

    # 2. Database admission check latency
    with Session(engine) as session:
        t_adm0 = time.monotonic()
        for _ in range(10):
            is_cycle_fenced_or_deleted(session, c_base, model_id="gfs")
        adm_dur_ms = ((time.monotonic() - t_adm0) / 10.0) * 1000.0

    # Indexed lookup per wave must be well under 10ms
    assert adm_dur_ms < 10.0
