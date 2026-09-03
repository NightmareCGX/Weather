"""Phase 6E Multi-Cycle Timeline, Staggered Cadence & Failure Fallback Acceptance Tests.

Validates the full lifecycle rules across a 10-cycle 3-day staggered matrix:
- C01: 2026-09-01 00Z (ready)   -> R1=09-02 00Z, R2=09-02 12Z (GC Eligible)
- C02: 2026-09-01 06Z (partial) -> R1=09-02 12Z (06Z failed), R2=09-03 00Z (GC Eligible)
- C03: 2026-09-01 12Z (ready)   -> R1=09-02 12Z, R2=09-03 00Z (GC Eligible)
- C04: 2026-09-01 18Z (failed)  -> R1=09-03 00Z, R2=09-03 06Z (GC Eligible)
- C05: 2026-09-02 00Z (ready)   -> R1=09-03 00Z, R2=09-03 06Z (GC Eligible)
- C06: 2026-09-02 06Z (failed)  -> R1=09-03 06Z (Retired), R2=None (Not GC Eligible)
- C07: 2026-09-02 12Z (ready)   -> Visible
- C08: 2026-09-02 18Z (partial) -> Visible (serves progressively)
- C09: 2026-09-03 00Z (ready)   -> Visible
- C10: 2026-09-03 06Z (ready)   -> Visible

Executes full lifecycle: Planning -> Retirement -> Dry Run -> Physical GC -> Tombstone verification.
"""

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

from ingestion.core.catalog import (
    CommittedState,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    RunCatalogSpec,
    VariableSpec,
    _ensure_utc_datetime,
    list_paired_ready_cycle_times,
    record_run,
)
from ingestion.core.zarr_writer import write_dataset
from ingestion.gc.reconciler import run_gc_pass


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
def postgres_acceptance_env(tmp_path):
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
            "PostgreSQL test instance not running or reachable; skipping multi-cycle acceptance test."
        )

    # Clean schema and apply Alembic migrations up to head
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
    status: str,
) -> tuple[str, str]:
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
        if status == "ready":
            record_run(session, spec_gfs, ds_gfs, committed_state=CommittedState.deterministic({0}))
            record_run(session, spec_gefs, ds_gefs, member=1, committed_state=CommittedState.ensemble({(1, 0)}, {1}))
        elif status == "partial":
            # Partial run: committed state subset
            r_gfs = record_run(session, spec_gfs, ds_gfs, committed_state=None)
            r_gefs = record_run(session, spec_gefs, ds_gefs, member=1, committed_state=None)
            setattr(r_gfs, "status", "partial")
            setattr(r_gefs, "status", "partial")
            session.commit()
        elif status == "failed":
            r_gfs = record_run(session, spec_gfs, ds_gfs, committed_state=None)
            r_gefs = record_run(session, spec_gefs, ds_gefs, member=1, committed_state=None)
            setattr(r_gfs, "status", "failed")
            setattr(r_gefs, "status", "failed")
            session.commit()

    return gfs_path, gefs_path


def test_phase6e_staggered_3day_multicycle_acceptance(postgres_acceptance_env):
    """Execute complete 10-cycle 3-day acceptance matrix on real PostgreSQL and filesystem."""
    engine, tmp_path = postgres_acceptance_env

    # Define 10 cycles
    c01 = _dt(2026, 9, 1, 0)   # ready
    c02 = _dt(2026, 9, 1, 6)   # partial
    c03 = _dt(2026, 9, 1, 12)  # ready
    c04 = _dt(2026, 9, 1, 18)  # failed
    c05 = _dt(2026, 9, 2, 0)   # ready
    c06 = _dt(2026, 9, 2, 6)   # failed
    c07 = _dt(2026, 9, 2, 12)  # ready
    c08 = _dt(2026, 9, 2, 18)  # partial
    c09 = _dt(2026, 9, 3, 0)   # ready
    c10 = _dt(2026, 9, 3, 6)   # ready

    cycles_data = [
        (c01, "ready"),
        (c02, "partial"),
        (c03, "ready"),
        (c04, "failed"),
        (c05, "ready"),
        (c06, "failed"),
        (c07, "ready"),
        (c08, "partial"),
        (c09, "ready"),
        (c10, "ready"),
    ]

    store_paths: dict[datetime, tuple[str, str]] = {}
    for c_time, status in cycles_data:
        store_paths[c_time] = _seed_cycle(engine, tmp_path, c_time, status)

    # 1. Verify Paired-Ready Discovery from PostgreSQL catalog
    with Session(engine) as session:
        paired_ready = list_paired_ready_cycle_times(session)
    assert paired_ready == [c01, c03, c05, c07, c09, c10]

    # 2. Run Dry Run at 2026-09-03 08:00Z
    now_sim = _dt(2026, 9, 3, 8)
    res_dry = run_gc_pass(engine, dry_run=True, now=now_sim)
    assert res_dry.dry_run is True

    # Assert expected retirements
    would_retire_times = [r.cycle_time for r in res_dry.would_retire]
    assert c01 in would_retire_times
    assert c02 in would_retire_times
    assert c03 in would_retire_times
    assert c04 in would_retire_times
    assert c05 in would_retire_times
    assert c06 in would_retire_times
    assert c07 not in would_retire_times
    assert c08 not in would_retire_times
    assert c09 not in would_retire_times
    assert c10 not in would_retire_times

    # Assert expected GC candidates (oldest first: C01, C02, C03, C04, C05)
    would_gc_times = [g.cycle_time for g in res_dry.would_gc]
    assert would_gc_times == [c01, c02, c03, c04, c05]
    assert c06 not in would_gc_times  # C06 is retired by C10, but needs R2 >= 09-03 12Z

    # Verify dry run did not delete any files
    for c_time, (gfs_p, gefs_p) in store_paths.items():
        assert os.path.exists(gfs_p)
        assert os.path.exists(gefs_p)

    # 3. Execute REAL GC pass
    res_real = run_gc_pass(engine, dry_run=False, now=now_sim)
    assert res_real.dry_run is False
    assert set(res_real.processed_gc) == {c01, c02, c03, c04, c05}

    # 4. Verify physical stores for deleted cycles are GONE
    for c_del in [c01, c02, c03, c04, c05]:
        gfs_p, gefs_p = store_paths[c_del]
        assert not os.path.exists(gfs_p)
        assert not os.path.exists(gefs_p)

    # 5. Verify physical stores for remaining cycles STILL EXIST
    for c_rem in [c06, c07, c08, c09, c10]:
        gfs_p, gefs_p = store_paths[c_rem]
        assert os.path.exists(gfs_p)
        assert os.path.exists(gefs_p)

    # 6. Verify catalog records:
    # - C01..C05: model_runs rows deleted, tombstones present with deleted_at
    # - C06: model_runs present, retired_at set, deleted_at NULL
    # - C07..C10: model_runs present, retired_at NULL
    with Session(engine) as session:
        for c_del in [c01, c02, c03, c04, c05]:
            assert session.execute(select(ModelRunRecord).where(ModelRunRecord.cycle_time == c_del)).scalars().all() == []
            lc = session.get(ForecastCycleLifecycleRecord, c_del)
            assert lc is not None
            assert lc.deleted_at is not None
            assert lc.deletion_started_at is not None
            assert lc.retired_at is not None
            assert lc.retired_by_cycle_time is not None

        # C06: retired but not deleted
        lc6 = session.get(ForecastCycleLifecycleRecord, c06)
        assert lc6 is not None
        assert lc6.retired_at is not None
        assert _ensure_utc_datetime(lc6.retired_by_cycle_time) == c10
        assert lc6.deleted_at is None

        # C07, C08, C09, C10: active visible
        for c_act in [c07, c08, c09, c10]:
            runs = session.execute(select(ModelRunRecord).where(ModelRunRecord.cycle_time == c_act)).scalars().all()
            assert len(runs) == 2
            lc_act = session.get(ForecastCycleLifecycleRecord, c_act)
            if lc_act is not None:
                assert lc_act.retired_at is None
                assert lc_act.deleted_at is None
