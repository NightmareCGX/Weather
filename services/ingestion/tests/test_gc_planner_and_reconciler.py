"""Unit tests for Phase 6D GC planner, dry-run, store deleter, and catalog cleanup."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ingestion.core.base import CycleTombstonedError
from ingestion.core.catalog import (
    CenterRecord,
    EnsembleMemberProductRecord,
    EnsembleMemberRecord,
    ForecastCycleLifecycleRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    _ensure_utc_datetime,
    ensure_lifecycle_row,
    is_cycle_tombstoned,
    mark_cycle_retired,
    record_run,
    RunCatalogSpec,
)
from ingestion.core.db import CatalogBase
from ingestion.gc.reconciler import (
    _delete_store_prefix,
    cleanup_cycle_catalog_and_tombstone,
    recheck_gc_eligibility,
    run_gc_pass,
)
import xarray as xr


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


@pytest.fixture
def catalog_engine():
    """Isolated in-memory SQLite catalog engine."""
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as session:
        center = CenterRecord(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="US",
            created_at=_dt(2026, 1, 1, 0),
        )
        gfs = ModelRecord(
            id="model_gfs",
            model_id="gfs",
            name="GFS",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs = ModelRecord(
            id="model_gefs",
            model_id="gefs",
            name="GEFS",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        gfs_v1 = ModelVersionRecord(
            id="version_gfs_v1.0",
            model_id="gfs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs_v1 = ModelVersionRecord(
            id="version_gefs_v1.0",
            model_id="gefs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        session.add_all([center, gfs, gefs, gfs_v1, gefs_v1])
        session.commit()
    yield engine
    engine.dispose()


def _seed_run(
    engine,
    model_id: str,
    cycle_time: datetime,
    status: str,
    version_string: str = "v1.0",
) -> ModelRunRecord:
    v_id = f"version_{model_id}_{version_string}"
    with Session(engine) as session:
        run = ModelRunRecord(
            id=f"run_{v_id}_{cycle_time.strftime('%Y%m%d%H%M')}_{model_id}",
            model_version_id=v_id,
            cycle_time=cycle_time,
            status=status,
            created_at=cycle_time,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


# ---------------------------------------------------------------------------
# Planner & Re-check Unit Tests
# ---------------------------------------------------------------------------


def test_gc_planner_identifies_eligible_candidates(catalog_engine):
    c0 = _dt(2026, 9, 1, 0)   # Retired by 09-02 00Z, R2 = 09-02 06Z (GC eligible)
    c1 = _dt(2026, 9, 1, 6)   # Retired by 09-02 06Z, R2 = 09-02 12Z (GC eligible)
    c2 = _dt(2026, 9, 1, 12)  # Retired by 09-02 12Z, R2 needs >= 09-02 18Z (not present)
    c3 = _dt(2026, 9, 1, 18)  # Active visible

    r1_0 = _dt(2026, 9, 2, 0)
    r1_1 = _dt(2026, 9, 2, 6)
    r1_2 = _dt(2026, 9, 2, 12)

    with Session(catalog_engine) as session:
        mark_cycle_retired(session, c0, r1_0, r1_0)
        mark_cycle_retired(session, c1, r1_1, r1_1)
        mark_cycle_retired(session, c2, r1_2, r1_2)
        session.commit()

        # Seed paired-ready runs for R1_0, R1_1, R1_2
        for t in [r1_0, r1_1, r1_2]:
            _seed_run(catalog_engine, "gfs", t, "ready")
            _seed_run(catalog_engine, "gefs", t, "ready")

    res = run_gc_pass(catalog_engine, dry_run=True, now=_dt(2026, 9, 2, 12, 30))
    assert res.dry_run is True
    gc_times = [c.cycle_time for c in res.would_gc]
    assert gc_times == [c0, c1]
    assert c2 not in gc_times
    assert c3 not in gc_times


def test_recheck_gc_eligibility_safely_rejects_unready_state(catalog_engine):
    c = _dt(2026, 9, 1, 6)
    r1 = _dt(2026, 9, 2, 6)

    with Session(catalog_engine) as session:
        # 1. Not in lifecycle table
        is_el, reason, _ = recheck_gc_eligibility(session, c)
        assert is_el is False
        assert reason == "no_lifecycle_record"

        # 2. Not retired
        ensure_lifecycle_row(session, c)
        session.commit()
        is_el2, reason2, _ = recheck_gc_eligibility(session, c)
        assert is_el2 is False
        assert reason2 == "cycle_not_retired"

        # 3. Retired but no R2 paired-ready
        mark_cycle_retired(session, c, r1, r1)
        session.commit()
        is_el3, reason3, _ = recheck_gc_eligibility(session, c)
        assert is_el3 is False
        assert reason3 == "no_qualifying_r2_paired_ready"

        # 4. R2 appears
        r2 = _dt(2026, 9, 2, 12)
        _seed_run(catalog_engine, "gfs", r2, "ready")
        _seed_run(catalog_engine, "gefs", r2, "ready")

        is_el4, reason4, r2_out = recheck_gc_eligibility(session, c)
        assert is_el4 is True
        assert r2_out == r2

        # 5. Already deleted (tombstone)
        lc = session.get(ForecastCycleLifecycleRecord, c)
        setattr(lc, "deleted_at", _dt(2026, 9, 2, 13))
        session.commit()
        is_el5, reason5, _ = recheck_gc_eligibility(session, c)
        assert is_el5 is False
        assert reason5 == "cycle_already_deleted"


# ---------------------------------------------------------------------------
# Physical Store Deletion & Idempotency Tests
# ---------------------------------------------------------------------------


def test_delete_store_prefix_local(tmp_path: Path):
    store_dir = tmp_path / "gfs_2026-09-01_06.zarr"
    store_dir.mkdir()
    (store_dir / "data.shard").write_text("shard-bytes")
    assert store_dir.exists()

    _delete_store_prefix(str(store_dir))
    assert not store_dir.exists()

    # Second call on already-deleted directory is idempotent no-op
    _delete_store_prefix(str(store_dir))


# ---------------------------------------------------------------------------
# Atomic Catalog Cleanup & Tombstone Tests
# ---------------------------------------------------------------------------


def test_cleanup_cycle_catalog_and_tombstone(catalog_engine):
    c = _dt(2026, 9, 1, 6)
    with Session(catalog_engine) as session:
        mark_cycle_retired(session, c, _dt(2026, 9, 2, 6), _dt(2026, 9, 2, 6))
        session.commit()

        gfs_run = _seed_run(catalog_engine, "gfs", c, "ready")
        gefs_run = _seed_run(catalog_engine, "gefs", c, "ready")

        # Seed products and members
        p1 = ProductRecord(
            id=f"p1_{gfs_run.id}",
            run_id=gfs_run.id,
            variable_id="var_t2m",
            grid_id="grid_global",
            product_type="surface",
            lead_time_hours=0,
        )
        m1 = EnsembleMemberRecord(
            id=f"m1_{gefs_run.id}",
            run_id=gefs_run.id,
            member_index=1,
            member_name="gefs_member_1",
        )
        emp1 = EnsembleMemberProductRecord(
            id=f"emp1_{gefs_run.id}",
            run_id=gefs_run.id,
            member_index=1,
            lead_time_hours=0,
        )
        session.add_all([p1, m1, emp1])
        session.commit()

    # Execute catalog cleanup
    del_time = _dt(2026, 9, 2, 13, 0)
    with Session(catalog_engine) as session:
        cleanup_cycle_catalog_and_tombstone(session, c, now=del_time)

    # Verify cycle rows are removed
    with Session(catalog_engine) as session:
        assert session.execute(select(ModelRunRecord).where(ModelRunRecord.cycle_time == c)).scalars().all() == []
        assert session.execute(select(ProductRecord).where(ProductRecord.run_id == gfs_run.id)).scalars().all() == []
        assert session.execute(select(EnsembleMemberRecord).where(EnsembleMemberRecord.run_id == gefs_run.id)).scalars().all() == []
        assert session.execute(select(EnsembleMemberProductRecord).where(EnsembleMemberProductRecord.run_id == gefs_run.id)).scalars().all() == []

        # Lifecycle row is retained as tombstone
        lc = session.get(ForecastCycleLifecycleRecord, c)
        assert lc is not None
        assert _ensure_utc_datetime(lc.deleted_at) == del_time
        assert _ensure_utc_datetime(lc.retired_by_cycle_time) == _dt(2026, 9, 2, 6)


# ---------------------------------------------------------------------------
# Stale Ingestion Resurrection Prevention Test
# ---------------------------------------------------------------------------


def test_stale_ingestion_resurrection_rejected(catalog_engine):
    c = _dt(2026, 9, 1, 6)
    with Session(catalog_engine) as session:
        mark_cycle_retired(session, c, _dt(2026, 9, 2, 6), _dt(2026, 9, 2, 6))
        cleanup_cycle_catalog_and_tombstone(session, c, now=_dt(2026, 9, 2, 13, 0))
        assert is_cycle_tombstoned(session, c) is True

    # Attempting to record a run for the tombstoned cycle must raise CycleTombstonedError
    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="US",
        model_id="gfs",
        model_name="GFS",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=c,
        grid_id="grid_global",
        grid_name="Global",
        grid_resolution_km=25.0,
    )
    dummy_ds = xr.Dataset({"temperature_2m": (("lead_time_hours",), [20.0])}, coords={"lead_time_hours": [0]})

    with Session(catalog_engine) as session:
        with pytest.raises(CycleTombstonedError, match="claimed for deletion or already tombstoned"):
            record_run(session, spec, dummy_ds)
