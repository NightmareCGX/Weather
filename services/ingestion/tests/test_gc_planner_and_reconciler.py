"""Unit tests for Phase 6D GC planner, dry-run, store deleter, and catalog cleanup (Lifecycle V2)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ingestion.core.base import CycleTombstonedError
from ingestion.core.catalog import (
    CenterRecord,
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
            zarr_store_path=f"s3://weather-data/{model_id}/{cycle_time.strftime('%Y-%m-%d/%H')}/cycle.zarr",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


# ---------------------------------------------------------------------------
# Planner & Re-check Unit Tests (Lifecycle V2)
# ---------------------------------------------------------------------------


def test_gc_planner_identifies_eligible_candidates(catalog_engine):
    """Under Lifecycle V2:
    Latest ready GFS cycle T = 09-02 06Z. Cadence C = 6h. Cutoff = 09-02 00Z.
    Cycles < 09-02 00Z (09-01 12Z, 09-01 18Z) are eligible for GC.
    Cycles >= 09-02 00Z (09-02 00Z, 09-02 06Z) are retained.
    """
    c0 = _dt(2026, 9, 1, 12)  # < cutoff -> GC eligible
    c1 = _dt(2026, 9, 1, 18)  # < cutoff -> GC eligible
    c2 = _dt(2026, 9, 2, 0)   # >= cutoff -> retained
    c3 = _dt(2026, 9, 2, 6)   # latest ready T -> retained

    # Seed GFS runs
    _seed_run(catalog_engine, "gfs", c0, "ready")
    _seed_run(catalog_engine, "gfs", c1, "ready")
    _seed_run(catalog_engine, "gfs", c2, "ready")
    _seed_run(catalog_engine, "gfs", c3, "ready")

    res = run_gc_pass(catalog_engine, dry_run=True, models=("gfs",), now=_dt(2026, 9, 2, 12, 30))
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
        is_el, reason, _ = recheck_gc_eligibility(session, "gfs", c)
        assert is_el is False
        assert reason == "no_lifecycle_record"

        # 2. Not retired
        ensure_lifecycle_row(session, "gfs", c)
        session.commit()
        is_el2, reason2, _ = recheck_gc_eligibility(session, "gfs", c)
        assert is_el2 is False
        assert reason2 == "cycle_not_retired"

        # 3. Retired but no qualifying ready cycle advances cutoff past c
        mark_cycle_retired(session, "gfs", c, r1, r1)
        session.commit()
        is_el3, reason3, _ = recheck_gc_eligibility(session, "gfs", c)
        assert is_el3 is False
        assert reason3 == "retained_at_or_above_cutoff"

        # 4. Ready cycle advances cutoff past c (e.g. 09-01 18Z is ready -> cutoff 09-01 12Z > c)
        t_ready = _dt(2026, 9, 1, 18)
        _seed_run(catalog_engine, "gfs", t_ready, "ready")

        is_el4, reason4, t_out = recheck_gc_eligibility(session, "gfs", c)
        assert is_el4 is True
        assert t_out == t_ready

        # 5. Already deleted (tombstone)
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c))
        setattr(lc, "deleted_at", _dt(2026, 9, 2, 13))
        session.commit()
        is_el5, reason5, _ = recheck_gc_eligibility(session, "gfs", c)
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
        mark_cycle_retired(session, "gfs", c, _dt(2026, 9, 2, 6), _dt(2026, 9, 2, 6))
        session.commit()

        gfs_run = _seed_run(catalog_engine, "gfs", c, "ready")

        # Seed products
        p1 = ProductRecord(
            id=f"p1_{gfs_run.id}",
            run_id=gfs_run.id,
            variable_id="var_t2m",
            grid_id="grid_global",
            product_type="surface",
            lead_time_hours=0,
        )
        session.add(p1)
        session.commit()

    # Execute catalog cleanup for gfs
    del_time = _dt(2026, 9, 2, 13, 0)
    with Session(catalog_engine) as session:
        cleanup_cycle_catalog_and_tombstone(session, "gfs", c, now=del_time)

    # Verify cycle rows are removed for gfs
    with Session(catalog_engine) as session:
        assert (
            session.execute(
                select(ModelRunRecord)
                .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
                .where(
                    (ModelVersionRecord.model_id == "gfs")
                    & (ModelRunRecord.cycle_time == c)
                )
            ).scalars().all()
            == []
        )
        assert session.execute(select(ProductRecord).where(ProductRecord.run_id == gfs_run.id)).scalars().all() == []

        # Lifecycle row is retained as tombstone
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c))
        assert lc is not None
        assert _ensure_utc_datetime(lc.deleted_at) == del_time
        assert _ensure_utc_datetime(lc.retired_by_cycle_time) == _dt(2026, 9, 2, 6)


# ---------------------------------------------------------------------------
# Stale Ingestion Resurrection Prevention Test
# ---------------------------------------------------------------------------


def test_stale_ingestion_resurrection_rejected(catalog_engine):
    c = _dt(2026, 9, 1, 6)
    with Session(catalog_engine) as session:
        mark_cycle_retired(session, "gfs", c, _dt(2026, 9, 2, 6), _dt(2026, 9, 2, 6))
        cleanup_cycle_catalog_and_tombstone(session, "gfs", c, now=_dt(2026, 9, 2, 13, 0))
        assert is_cycle_tombstoned(session, c, model_id="gfs") is True

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
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree",
        grid_resolution_km=25.0,
        product_type="surface",
        variables=(),
    )
    dataset = xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), [[[20.0]]]),
        },
        coords={
            "lead_time_hours": [0],
            "latitude": [0.0],
            "longitude": [0.0],
        },
    )
    with Session(catalog_engine) as session:
        with pytest.raises(CycleTombstonedError, match="Refusing to ingest"):
            record_run(session, spec, dataset)
