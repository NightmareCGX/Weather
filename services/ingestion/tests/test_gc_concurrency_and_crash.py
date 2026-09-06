"""Service-backed concurrency, locking, crash recovery, and E2E lifecycle tests for Phase 6D."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from api.core.reader_gate import _ReaderGateSession, ReaderLockPool
from ingestion.core.base import CycleTombstonedError
from ingestion.core.catalog import (
    CommittedState,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    RunCatalogSpec,
    VariableSpec,
    _ensure_utc_datetime,
    mark_cycle_retired,
    record_run,
)
from ingestion.core.locks import StoreLockCoordinator
from ingestion.core.zarr_writer import write_dataset
from ingestion.gc.reconciler import (
    GcCandidateInfo,
    claim_cycle_for_deletion,
    cleanup_cycle_catalog_and_tombstone,
    delete_physical_store_gated,
    process_gc_candidate,
    recheck_gc_eligibility,
    run_gc_pass,
)


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
# ---------------------------------------------------------------------------


def test_active_reader_gate_blocks_physical_gc(postgres_gc_env):
    """Prove that an active reader holding SHARED store gate blocks GC EXCLUSIVE gate."""
    engine, tmp_path = postgres_gc_env
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )

    c0 = _dt(2026, 9, 1, 6)   # To delete
    c1 = _dt(2026, 9, 2, 6)   # R1
    c2 = _dt(2026, 9, 2, 12)  # R2

    gfs_path, gefs_path = _seed_cycle(engine, tmp_path, c0, "ready")
    _seed_cycle(engine, tmp_path, c1, "ready")
    _seed_cycle(engine, tmp_path, c2, "ready")

    # Mark c0 retired by c1
    with Session(engine) as session:
        mark_cycle_retired(session, "gfs", c0, c1, c1)
        session.commit()

    # Start a reader session holding SHARED store gate on gfs_path
    reader_pool = ReaderLockPool(db_url, pool_size=2, max_overflow=0, pool_timeout=5.0)
    reader_session = _ReaderGateSession(reader_pool, gfs_path)
    reader_session.acquire(timeout_seconds=5.0)

    try:
        candidate = GcCandidateInfo(
            model_id="gfs",
            cycle_time=c0,
            retired_by_cycle_time=c1,
            cutoff=c1,
            store_path=gfs_path,
            gfs_store_path=gfs_path,
            gefs_store_path=gefs_path,
        )

        # GC pass attempts to delete with bounded 1.0s timeout -> must be BLOCKED
        success = process_gc_candidate(
            engine, candidate, timeout_seconds=1.0, now=_dt(2026, 9, 2, 13)
        )
        assert success is False

        # Store and catalog remain intact
        assert os.path.exists(gfs_path)
        with Session(engine) as session:
            assert session.get(ForecastCycleLifecycleRecord, ("gfs", c0)).deleted_at is None
    finally:
        reader_session.release()
        reader_pool.dispose()

    # Now that reader released SHARED gate, retry GC -> succeeds!
    success_after = process_gc_candidate(
        engine, candidate, timeout_seconds=5.0, now=_dt(2026, 9, 2, 13)
    )
    assert success_after is True
    assert not os.path.exists(gfs_path)
    with Session(engine) as session:
        assert session.get(ForecastCycleLifecycleRecord, ("gfs", c0)).deleted_at is not None


def test_active_writer_gate_blocks_physical_gc(postgres_gc_env):
    """Prove that an active writer holding SHARED store gate blocks GC EXCLUSIVE gate."""
    engine, tmp_path = postgres_gc_env

    c0 = _dt(2026, 9, 1, 6)
    c1 = _dt(2026, 9, 2, 6)
    c2 = _dt(2026, 9, 2, 12)

    gfs_path, gefs_path = _seed_cycle(engine, tmp_path, c0, "ready")
    _seed_cycle(engine, tmp_path, c1, "ready")
    _seed_cycle(engine, tmp_path, c2, "ready")

    with Session(engine) as session:
        mark_cycle_retired(session, "gfs", c0, c1, c1)
        session.commit()

    candidate = GcCandidateInfo(
        model_id="gfs",
        cycle_time=c0,
        retired_by_cycle_time=c1,
        cutoff=c1,
        store_path=gfs_path,
        gfs_store_path=gfs_path,
        gefs_store_path=gefs_path,
    )

    # Acquire SHARED gate simulating an in-progress writer wave
    with engine.connect() as writer_conn:
        writer_coord = StoreLockCoordinator(writer_conn, store_path=gfs_path, timeout_seconds=5.0)
        writer_coord.acquire_shared_gate()

        try:
            # GC attempt is blocked by writer
            success = process_gc_candidate(
                engine, candidate, timeout_seconds=1.0, now=_dt(2026, 9, 2, 13)
            )
            assert success is False
            assert os.path.exists(gfs_path)
        finally:
            writer_coord.release_shared_gate()

    # Retry after writer release -> succeeds
    success2 = process_gc_candidate(
        engine, candidate, timeout_seconds=5.0, now=_dt(2026, 9, 2, 13)
    )
    assert success2 is True
    assert not os.path.exists(gfs_path)


# ---------------------------------------------------------------------------
# Crash Recovery & Restart Tests
# ---------------------------------------------------------------------------


def test_crash_recovery_after_partial_store_deletion(postgres_gc_env):
    """Simulate crash where GFS was deleted but process died before GEFS was deleted."""
    engine, tmp_path = postgres_gc_env

    c0 = _dt(2026, 9, 1, 6)
    c1 = _dt(2026, 9, 2, 6)
    c2 = _dt(2026, 9, 2, 12)

    gfs_path, gefs_path = _seed_cycle(engine, tmp_path, c0, "ready")
    _seed_cycle(engine, tmp_path, c1, "ready")
    _seed_cycle(engine, tmp_path, c2, "ready")

    with Session(engine) as session:
        mark_cycle_retired(session, "gfs", c0, c1, c1)
        session.commit()

    # Manually delete GFS to simulate crash midway
    shutil.rmtree(gfs_path)
    assert not os.path.exists(gfs_path)
    assert os.path.exists(gefs_path)

    # Next GC pass runs from scratch
    candidate = GcCandidateInfo(
        model_id="gfs",
        cycle_time=c0,
        retired_by_cycle_time=c1,
        cutoff=c1,
        store_path=gfs_path,
        gfs_store_path=gfs_path,
        gefs_store_path=gefs_path,
    )
    success = process_gc_candidate(
        engine, candidate, timeout_seconds=5.0, now=_dt(2026, 9, 2, 13)
    )
    assert success is True
    with Session(engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c0))
        assert lc is not None
        assert lc.deleted_at is not None


# ---------------------------------------------------------------------------
# Full E2E Lifecycle Acceptance Test
# ---------------------------------------------------------------------------


def test_full_e2e_lifecycle_retirement_dryrun_gc_tombstone(postgres_gc_env):
    """Execute the full end-to-end Phase 6 lifecycle pipeline."""
    engine, tmp_path = postgres_gc_env

    c0 = _dt(2026, 9, 1, 6)
    c1 = _dt(2026, 9, 2, 6)   # R1
    c2 = _dt(2026, 9, 2, 12)  # R2

    gfs_path, gefs_path = _seed_cycle(engine, tmp_path, c0, "ready")
    _seed_cycle(engine, tmp_path, c1, "ready")
    _seed_cycle(engine, tmp_path, c2, "ready")

    # 1. Run GC pass in DRY RUN mode -> shows would_retire and would_gc
    res_dry = run_gc_pass(engine, dry_run=True, now=_dt(2026, 9, 2, 12, 30))
    assert res_dry.dry_run is True
    assert len(res_dry.would_retire) >= 1
    assert c0 in [r.cycle_time for r in res_dry.would_retire]
    assert len(res_dry.would_gc) >= 1
    assert c0 in [g.cycle_time for g in res_dry.would_gc]

    # Dry run did not delete anything
    assert os.path.exists(gfs_path)
    assert os.path.exists(gefs_path)
    with Session(engine) as session:
        assert session.get(ForecastCycleLifecycleRecord, ("gfs", c0)) is None

    # 2. Run REAL GC pass
    res_real = run_gc_pass(engine, dry_run=False, now=_dt(2026, 9, 2, 12, 30))
    assert res_real.dry_run is False
    assert c0 in res_real.processed_gc

    # 3. Stores are physically removed
    assert not os.path.exists(gfs_path)
    assert not os.path.exists(gefs_path)

    # 4. Catalog rows for c0 are deleted
    with Session(engine) as session:
        runs_c0 = session.execute(
            select(ModelRunRecord).where(ModelRunRecord.cycle_time == c0)
        ).scalars().all()
        assert runs_c0 == []

        # 5. Tombstone in forecast_cycle_lifecycle is retained
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c0))
        assert lc is not None
        assert _ensure_utc_datetime(lc.deleted_at) == _dt(2026, 9, 2, 12, 30)

    # 6. Successor cycles c1 and c2 remain completely intact!
    with Session(engine) as session:
        runs_c1 = session.execute(
            select(ModelRunRecord).where(ModelRunRecord.cycle_time == c1)
        ).scalars().all()
        assert len(runs_c1) == 2  # GFS + GEFS


# ---------------------------------------------------------------------------
# Deletion Fence Race Closure & Monotonicity Tests
# ---------------------------------------------------------------------------


def test_reingestion_race_closure_after_gfs_release(postgres_gc_env):
    """Prove that after GFS deletion gate release, new ingestion is blocked by deletion_started_at.

    Scenario:
    1. GC claims cycle c0 (deletion_started_at set).
    2. GFS is deleted and GFS store gate is released.
    3. GC pauses before deleting GEFS.
    4. Stale/manual ingestion for c0 attempts to start via wave runner / admission check.
    5. Stale ingestion is rejected with CycleTombstonedError before touching storage.
    6. GFS store is NOT recreated.
    7. GC resumes, deletes GEFS, cleans catalog, and sets deleted_at.
    """
    engine, tmp_path = postgres_gc_env

    c0 = _dt(2026, 9, 1, 6)
    c1 = _dt(2026, 9, 2, 6)
    c2 = _dt(2026, 9, 2, 12)

    gfs_path, gefs_path = _seed_cycle(engine, tmp_path, c0, "ready")
    _seed_cycle(engine, tmp_path, c1, "ready")
    _seed_cycle(engine, tmp_path, c2, "ready")

    with Session(engine) as session:
        mark_cycle_retired(session, "gfs", c0, c1, c1)
        session.commit()

    # Step 1: Claim deletion fence
    with Session(engine) as session:
        assert claim_cycle_for_deletion(session, "gfs", c0, now=_dt(2026, 9, 2, 12, 30)) is True

    # Step 2: Delete GFS and release GFS gate
    assert delete_physical_store_gated(engine, gfs_path, timeout_seconds=5.0) is True
    assert not os.path.exists(gfs_path)

    # Step 3: GC is paused before cleanup. deleted_at is still NULL!
    with Session(engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c0))
        assert lc.deletion_started_at is not None
        assert lc.deleted_at is None

    # Step 4: Stale ingestion attempts to write to GFS store
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
    dummy_ds = _make_dataset(c0, 0)

    # Step 5: Ingestion admission guard must reject before creating store
    with Session(engine) as session:
        with pytest.raises(CycleTombstonedError, match="claimed for deletion or already tombstoned"):
            record_run(session, spec, dummy_ds)

    # Step 6: Verify GFS store was NOT recreated
    assert not os.path.exists(gfs_path)

    # Step 7: GC cleans catalog, stamps deleted_at
    with Session(engine) as session:
        cleanup_cycle_catalog_and_tombstone(session, "gfs", c0, now=_dt(2026, 9, 2, 12, 35))

    assert not os.path.exists(gfs_path)
    with Session(engine) as session:
        lc_final = session.get(ForecastCycleLifecycleRecord, ("gfs", c0))
        assert lc_final is not None
        assert lc_final.deletion_started_at is not None
        assert lc_final.deleted_at is not None


def test_crash_fence_persists_across_process_restart(postgres_gc_env):
    """Simulate process crash after deletion_started_at is stamped; verify fence blocks writers while offline."""
    engine, tmp_path = postgres_gc_env

    c0 = _dt(2026, 9, 1, 6)
    c1 = _dt(2026, 9, 2, 6)
    c2 = _dt(2026, 9, 2, 12)

    gfs_path, gefs_path = _seed_cycle(engine, tmp_path, c0, "ready")
    _seed_cycle(engine, tmp_path, c1, "ready")
    _seed_cycle(engine, tmp_path, c2, "ready")

    with Session(engine) as session:
        mark_cycle_retired(session, "gfs", c0, c1, c1)
        # Process 1 claims cycle and deletes GFS, then crashes
        claim_cycle_for_deletion(session, "gfs", c0, now=_dt(2026, 9, 2, 12, 30))
        session.commit()

    delete_physical_store_gated(engine, gfs_path, timeout_seconds=5.0)
    assert not os.path.exists(gfs_path)

    # While GC is down, a manual ingestion tries to run
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
        with pytest.raises(CycleTombstonedError):
            record_run(session, spec, _make_dataset(c0, 0))

    # Process 2 (new GC leader) restarts and completes the pass
    res_pass = run_gc_pass(engine, dry_run=False, now=_dt(2026, 9, 2, 13, 0))
    assert c0 in res_pass.processed_gc

    assert not os.path.exists(gfs_path)
    with Session(engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c0))
        assert lc.deleted_at is not None


def test_claimed_deletion_monotonic_recovery_without_historical_r2(postgres_gc_env):
    """Prove that an already-claimed deletion resumes even if historical successor runs were removed."""
    engine, tmp_path = postgres_gc_env

    c0 = _dt(2026, 9, 1, 6)
    c1 = _dt(2026, 9, 2, 6)
    c2 = _dt(2026, 9, 2, 12)

    gfs_path, gefs_path = _seed_cycle(engine, tmp_path, c0, "ready")
    _seed_cycle(engine, tmp_path, c1, "ready")
    _seed_cycle(engine, tmp_path, c2, "ready")

    with Session(engine) as session:
        mark_cycle_retired(session, "gfs", c0, c1, c1)
        claim_cycle_for_deletion(session, "gfs", c0, now=_dt(2026, 9, 2, 12, 30))
        session.commit()

        # Delete successor runs for c1 and c2 simulating prior historical cleanup
        cleanup_cycle_catalog_and_tombstone(session, "gfs", c1, now=_dt(2026, 9, 2, 12, 35))
        cleanup_cycle_catalog_and_tombstone(session, "gfs", c2, now=_dt(2026, 9, 2, 12, 35))

        # Recheck eligibility must still return True (monotonic recovery)
        is_el, reason, _ = recheck_gc_eligibility(session, "gfs", c0)
        assert is_el is True
        assert reason == "gc_claimed_resumable"

