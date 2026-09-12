"""Data Lifecycle V3 Milestone 4: Admission Hardening & Legacy Writer Removal Tests.

Validates that:
1. Tombstone-only cycle (after M3 detailed metadata sweep: deleted_at != NULL,
   zero model_runs/products/queue rows) strictly rejects:
   - reserve_run (CycleTombstonedError)
   - record_run (CycleTombstonedError)
   - wave_runner._run_wave (CycleTombstonedError)
   - scheduler / backlog recovery (excluded)
2. Incomplete cycle with retired_at != NULL and deleted_at == NULL is admitted to
   backlog recovery (retired_at no longer blocks recovery).
3. Legacy GC eligibility (recheck_gc_eligibility) does not require retired_at.
4. run_gc_pass no longer calls reconcile_cycle_lifecycle.
5. Production finalizer and metadata sweeper passes perform zero writes to retired_at.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.base import CycleTombstonedError
from ingestion.core.catalog import (
    CatalogBase,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    RunCatalogSpec,
    VariableSpec,
    record_run,
    reserve_run,
)
from ingestion.core.wave_runner import RunSpec, _run_wave
from ingestion.gc.finalizer import run_finalizer_pass
from ingestion.gc.reconciler import recheck_gc_eligibility, run_gc_pass
from ingestion.gc.sweeper import run_metadata_sweeper_pass
from ingestion.realtime.committed import (
    discover_incomplete_historical_cycles,
    is_cycle_fenced_or_deleted,
)


def _dt(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def catalog_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    CatalogBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _make_spec(cycle_time: datetime, model: str = "gfs") -> RunCatalogSpec:
    return RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="US",
        model_id=model,
        model_name="Global Forecast System",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=cycle_time,
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=f"s3://weather-data/{model}/test/cycle.zarr",
        variables=(
            VariableSpec(
                code="temperature_2m",
                name="2m Temperature",
                unit="degC",
                source_code="t2m",
            ),
        ),
        expected_lead_time_hours=(0,),
        expected_members=(),
    )


def _make_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "lat", "lon"), np.zeros((1, 2, 2), dtype=np.float32)),
        },
        coords={"lead_time_hours": [0]},
    )


# ---------------------------------------------------------------------------
# 1. Tombstone-Only Cycle Anti-Resurrection
# ---------------------------------------------------------------------------


def test_tombstone_only_rejects_reserve_run(catalog_engine):
    """A tombstone-only lifecycle row (deleted_at != NULL, 0 runs) rejects reserve_run."""
    c = _dt(2026, 9, 1, 0)
    with Session(catalog_engine) as session:
        # Simulate an M3-swept cycle: only tombstone row remains
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 2, 0),
                deleted_at=_dt(2026, 9, 2, 1),
            )
        )
        session.commit()

        spec = _make_spec(c, "gfs")
        with pytest.raises(CycleTombstonedError) as exc_info:
            reserve_run(session, spec)
        assert "tombstoned" in str(exc_info.value)


def test_tombstone_only_rejects_record_run(catalog_engine):
    """A tombstone-only lifecycle row rejects record_run."""
    c = _dt(2026, 9, 1, 0)
    with Session(catalog_engine) as session:
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 2, 0),
                deleted_at=_dt(2026, 9, 2, 1),
            )
        )
        session.commit()

        spec = _make_spec(c, "gfs")
        ds = _make_dataset()
        with pytest.raises(CycleTombstonedError) as exc_info:
            record_run(session, spec, ds)
        assert "tombstoned" in str(exc_info.value)


def test_tombstone_only_rejects_wave_runner(catalog_engine, monkeypatch):
    """A tombstone-only lifecycle row rejects wave_runner._run_wave admission."""
    c = _dt(2026, 9, 1, 0)
    with Session(catalog_engine) as session:
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 2, 0),
                deleted_at=_dt(2026, 9, 2, 1),
            )
        )
        session.commit()

    monkeypatch.setattr("ingestion.core.wave_runner._catalog_session", lambda: Session(catalog_engine))

    spec = RunSpec(
        model="gfs",
        cycle_date=c.date(),
        cycle_hour=c.hour,
        target_lead_time_hours=(0,),
        members=(),
        allow_custom_store=True,
    )
    catalog_spec = _make_spec(c, "gfs")

    class _Args:
        download_dir = "/tmp"
        concurrency = 1

    failures: list[str] = []
    with pytest.raises(CycleTombstonedError) as exc_info:
        asyncio.run(
            _run_wave(
                spec=spec,
                args=_Args(),
                catalog_spec=catalog_spec,
                store_path=catalog_spec.zarr_store_path,
                concurrency=1,
                failures=failures,
            )
        )
    assert "tombstoned" in str(exc_info.value)


def test_tombstone_only_excluded_from_scheduler_recovery(catalog_engine):
    """A tombstone-only cycle is excluded from scheduler / backlog recovery."""
    now_utc = _dt(2026, 9, 2, 18)
    c_active = _dt(2026, 9, 2, 18)
    c_tombstone = _dt(2026, 9, 2, 6)

    with Session(catalog_engine) as session:
        # Swept cycle: lifecycle row with deleted_at set, NO model_runs
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c_tombstone,
                deletion_started_at=_dt(2026, 9, 2, 7),
                deleted_at=_dt(2026, 9, 2, 8),
            )
        )
        session.commit()

        # 1. Direct predicate confirms exclusion
        assert is_cycle_fenced_or_deleted(session, c_tombstone, model_id="gfs") is True

    # 2. Candidate discovery excludes it completely
    candidates = discover_incomplete_historical_cycles(
        catalog_engine,
        active_cycle_time=c_active,
        now_utc=now_utc,
    )
    candidate_times = [c.cycle_time for c in candidates]
    assert c_tombstone not in candidate_times


# ---------------------------------------------------------------------------
# 2. Incomplete Cycle with retired_at != NULL Admitted
# ---------------------------------------------------------------------------


def test_unfenced_cycle_with_retired_at_admitted_to_recovery(catalog_engine):
    """An incomplete cycle with retired_at != NULL and deleted_at == NULL is admitted."""
    now_utc = _dt(2026, 9, 2, 18)
    c_active = _dt(2026, 9, 2, 18)
    c_partial_retired = _dt(2026, 9, 2, 12)

    with Session(catalog_engine) as session:
        session.add(ModelVersionRecord(id="ver_gfs", model_id="gfs", version_string="v1.0"))
        session.add(
            ModelRunRecord(
                id="r_gfs_ret",
                model_version_id="ver_gfs",
                cycle_time=c_partial_retired,
                status="partial",
            )
        )
        session.add(
            ProductRecord(
                id="p_0",
                run_id="r_gfs_ret",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        # Marked retired, but NOT claimed or deleted
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c_partial_retired,
                retired_at=_dt(2026, 9, 2, 13),
                deletion_started_at=None,
                deleted_at=None,
            )
        )
        session.commit()

        assert is_cycle_fenced_or_deleted(session, c_partial_retired, model_id="gfs") is False

    candidates = discover_incomplete_historical_cycles(
        catalog_engine,
        active_cycle_time=c_active,
        now_utc=now_utc,
    )
    candidate_times = [c.cycle_time for c in candidates]
    assert c_partial_retired in candidate_times


# ---------------------------------------------------------------------------
# 3. Legacy GC Eligibility & Writer Removal
# ---------------------------------------------------------------------------


def test_recheck_gc_eligibility_ignores_retired_at(catalog_engine):
    """recheck_gc_eligibility does not require retired_at to be populated."""
    c_old = _dt(2026, 9, 1, 0)
    c_ready = _dt(2026, 9, 1, 12)

    with Session(catalog_engine) as session:
        session.add(ModelVersionRecord(id="ver_gfs", model_id="gfs", version_string="v1.0"))
        session.add(
            ModelRunRecord(
                id="r_ready",
                model_version_id="ver_gfs",
                cycle_time=c_ready,
                status="ready",
            )
        )
        # c_old has NO retired_at, but lifecycle row exists
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c_old,
                retired_at=None,
                deletion_started_at=None,
                deleted_at=None,
            )
        )
        session.commit()

        # Cadence is 6h -> cutoff is 12Z - 6h = 06Z. c_old (00Z) < 06Z is GC eligible!
        eligible, reason, t_ready = recheck_gc_eligibility(session, "gfs", c_old)
        assert eligible is True
        assert reason == "gc_eligible"
        assert t_ready == c_ready


def test_run_gc_pass_does_not_call_reconcile_cycle_lifecycle(catalog_engine):
    """run_gc_pass no longer calls reconcile_cycle_lifecycle (zero retirement writes)."""
    with patch("ingestion.core.catalog.reconcile_cycle_lifecycle") as mock_reconcile:
        # Run a real (non-dry-run) GC pass
        run_gc_pass(catalog_engine, dry_run=False, models=("gfs",))
        # Reconcile cycle lifecycle must NOT be invoked
        mock_reconcile.assert_not_called()


def test_finalizer_and_sweeper_zero_retired_at_writes(catalog_engine):
    """Production finalizer and sweeper passes perform zero writes to retired_at."""
    c = _dt(2026, 8, 1, 0)
    now = _dt(2026, 9, 1, 0)

    with Session(catalog_engine) as session:
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 8, 2, 0),
                deleted_at=_dt(2026, 8, 2, 1),
                retired_at=None,
                retired_by_cycle_time=None,
            )
        )
        session.commit()

    # 1. Run finalizer pass
    run_finalizer_pass(catalog_engine, dry_run=False, now=now)

    # 2. Run metadata sweeper pass
    run_metadata_sweeper_pass(catalog_engine, dry_run=False, now=now)

    # Verify retired_at and retired_by_cycle_time remain strictly None
    with Session(catalog_engine) as session:
        row = session.get(ForecastCycleLifecycleRecord, ("gfs", c))
        assert row is not None
        assert row.retired_at is None
        assert row.retired_by_cycle_time is None
