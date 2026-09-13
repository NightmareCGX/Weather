"""Unit tests for the V3 reclamation planner (canonical-necessity authority).

Covers:
* Window-exit fast-path: units whose valid_time exited the active serving
  window are reclaimable without canonical selection (derived cutoff, §7.5B).
* Active-window holds: canonical anchor units of protected valid times are held.
* Claimed cycles (deletion_started_at serving fence): ALL remaining units are
  directly reclaimable — the fence means the resolver no longer selects them.
* Tombstone anti-resurrection on ingestion (Guarantee A).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import xarray as xr

from ingestion.core.base import CycleTombstonedError
from ingestion.core.catalog import (
    CenterRecord,
    ForecastCycleLifecycleRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    is_cycle_tombstoned,
    record_run,
    RunCatalogSpec,
)
from ingestion.core.db import CatalogBase
from ingestion.gc.planner import plan_reclamation_pass


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
        gfs_v1 = ModelVersionRecord(
            id="version_gfs_v1.0",
            model_id="gfs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        session.add_all([center, gfs, gfs_v1])
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


def _seed_product(engine, run: ModelRunRecord, lead: int, variable: str = "temperature_2m") -> None:
    with Session(engine) as session:
        session.add(
            ProductRecord(
                id=f"prod_{run.id}_{lead}_{variable}",
                run_id=run.id,
                variable_id=variable,
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
                zarr_chunk_path="/fake",
            )
        )
        session.commit()


def _claim_cycle(engine, model_id: str, cycle_time: datetime) -> None:
    with Session(engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, (model_id, cycle_time))
        if lc is None:
            session.add(
                ForecastCycleLifecycleRecord(
                    model_id=model_id,
                    cycle_time=cycle_time,
                    deletion_started_at=_dt(2026, 9, 2, 12),
                    created_at=_dt(2026, 9, 2, 12),
                    updated_at=_dt(2026, 9, 2, 12),
                )
            )
        else:
            lc.deletion_started_at = _dt(2026, 9, 2, 12)
        session.commit()


def test_planner_window_exit_fast_path_and_active_anchor_holds(catalog_engine):
    """now = 09-02 12:30Z -> serving_start = 12:00Z.

    * Old cycle (08-01 00Z, horizon long past): every unit's valid_time has
      exited the window -> reclaimable (window-exit fast path, §7.5B).
    * Fresh cycle (09-02 06Z) with protected valid times (12Z, 15Z):
      canonical anchor units are HELD.
    """
    c_old = _dt(2026, 8, 1, 0)
    c_fresh = _dt(2026, 9, 2, 6)

    run_old = _seed_run(catalog_engine, "gfs", c_old, "ready")
    run_fresh = _seed_run(catalog_engine, "gfs", c_fresh, "ready")
    _seed_product(catalog_engine, run_old, 0)
    _seed_product(catalog_engine, run_fresh, 6)   # VT 12Z (== serving_start) -> anchor
    _seed_product(catalog_engine, run_fresh, 9)   # VT 15Z -> anchor

    with Session(catalog_engine) as session:
        res = plan_reclamation_pass(
            session, models=("gfs",), dry_run=True, now=_dt(2026, 9, 2, 12, 30)
        )

    enqueued_keys = {(t.run_id, t.lead_time_hours) for t in res.would_enqueue}
    # Window-exited unit of the old cycle is reclaimable
    assert (run_old.id, 0) in enqueued_keys
    # Protected anchor units of the fresh cycle are held
    assert (run_fresh.id, 6) not in enqueued_keys
    assert (run_fresh.id, 9) not in enqueued_keys
    assert res.active_held_shards == 2
    assert res.reclaimable_shards == 1


def test_planner_claimed_cycle_units_directly_reclaimable(catalog_engine):
    """Once claimed (serving fence), ALL remaining units are unprotected.

    The fresh cycle at 09-02 06Z would normally be the canonical anchor for its
    protected valid times — but its retirement claim means the resolver no
    longer selects it, so the planner must enqueue its units for the worker.
    """
    c_fresh = _dt(2026, 9, 2, 6)
    run_fresh = _seed_run(catalog_engine, "gfs", c_fresh, "ready")
    _seed_product(catalog_engine, run_fresh, 6)   # VT 12Z — protected, but cycle claimed
    _seed_product(catalog_engine, run_fresh, 9)   # VT 15Z
    _claim_cycle(catalog_engine, "gfs", c_fresh)

    with Session(catalog_engine) as session:
        res = plan_reclamation_pass(
            session, models=("gfs",), dry_run=True, now=_dt(2026, 9, 2, 12, 30)
        )

    assert {(t.run_id, t.lead_time_hours) for t in res.would_enqueue} == {
        (run_fresh.id, 6),
        (run_fresh.id, 9),
    }
    assert res.active_held_shards == 0
    assert res.reclaimable_shards == 2


def test_planner_tombstoned_cycle_ignored(catalog_engine):
    """Tombstoned cycles are terminal by construction — nothing planned."""
    c_old = _dt(2026, 8, 1, 0)
    run_old = _seed_run(catalog_engine, "gfs", c_old, "ready")
    _seed_product(catalog_engine, run_old, 0)
    with Session(catalog_engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c_old))
        if lc is None:
            session.add(
                ForecastCycleLifecycleRecord(
                    model_id="gfs",
                    cycle_time=c_old,
                    deletion_started_at=_dt(2026, 8, 20, 0),
                    deleted_at=_dt(2026, 8, 20, 0),
                    created_at=_dt(2026, 8, 20, 0),
                    updated_at=_dt(2026, 8, 20, 0),
                )
            )
        session.commit()

    with Session(catalog_engine) as session:
        res = plan_reclamation_pass(
            session, models=("gfs",), dry_run=True, now=_dt(2026, 9, 2, 12, 30)
        )
    assert res.would_enqueue == ()
    assert res.total_committed_shards == 0


# ---------------------------------------------------------------------------
# Stale Ingestion Resurrection Prevention Test
# ---------------------------------------------------------------------------


def test_stale_ingestion_resurrection_rejected(catalog_engine):
    c = _dt(2026, 9, 1, 6)
    with Session(catalog_engine) as session:
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 2, 13, 0),
                deleted_at=_dt(2026, 9, 2, 13, 0),
                created_at=_dt(2026, 9, 2, 13, 0),
                updated_at=_dt(2026, 9, 2, 13, 0),
            )
        )
        session.commit()
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
