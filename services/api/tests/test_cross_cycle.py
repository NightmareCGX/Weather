"""Cross-cycle deterministic time-series selection tests.

These tests exercise the minimum-lead-per-valid_time selection logic that the
``/v1/points`` endpoint now uses, against an in-memory SQLite catalog (no live
PostGIS or Zarr needed). The scenario from the product requirement:

    run A: 00Z + 0h -> 00Z, 00Z + 6h -> 06Z, ...
    run B: 06Z + 0h -> 06Z, 06Z + 6h -> 12Z, ...

For each valid_time, the record with the MINIMUM lead_time_hours is selected
(== the newest cycle that covers it). Only READY runs participate; a partial or
processing run is never a candidate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.models.entities import (
    Base,
    ForecastCenter,
    ForecastCycleLifecycle,
    ForecastGrid,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.point_forecast import _select_min_lead_winners

CYCLE_00 = datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)
CYCLE_06 = datetime(2026, 8, 14, 6, 0, tzinfo=timezone.utc)
CYCLE_12 = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    # Create only the catalog tables the selection query joins (the API entity
    # metadata also defines PostGIS Geometry tables that SQLite cannot build).
    _CATALOG_TABLES = [
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ModelRun.__table__,
        ForecastVariable.__table__,
        ForecastGrid.__table__,
        ForecastProduct.__table__,
        ForecastCycleLifecycle.__table__,
    ]
    Base.metadata.create_all(engine, tables=_CATALOG_TABLES)
    session = Session(engine)
    _seed(session)
    yield session
    session.close()
    engine.dispose()


def _seed(session: Session) -> None:
    """Seed a minimal catalog: model/version/runs/products for two cycles."""
    session.add(
        ForecastCenter(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="USA",
        )
    )
    session.add(
        Model(
            id="model_gfs",
            model_id="gfs",
            name="GFS",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
        )
    )
    session.add(
        ModelVersion(id="version_gfs_v1", model_id="gfs", version_string="v1.0")
    )
    session.add(
        ForecastVariable(
            id="var_temperature_2m",
            variable_code="temperature_2m",
            name="2-Meter Temperature",
            unit="°C",
        )
    )
    session.add(
        ForecastGrid(
            id="grid_global_025deg",
            grid_code="global_025deg",
            name="Global 0.25 Degree Grid",
            resolution_km=25.0,
        )
    )
    # Run A: 00Z READY with leads 0, 6, 12.
    session.add(
        ModelRun(
            id="run_00z",
            model_version_id="version_gfs_v1",
            cycle_time=CYCLE_00,
            status="ready",
            zarr_store_path="/tmp/gfs_00.zarr",
        )
    )
    # Run B: 06Z READY with leads 0, 6.
    session.add(
        ModelRun(
            id="run_06z",
            model_version_id="version_gfs_v1",
            cycle_time=CYCLE_06,
            status="ready",
            zarr_store_path="/tmp/gfs_06.zarr",
        )
    )
    # Run C: 12Z FAILED (must NOT participate).
    session.add(
        ModelRun(
            id="run_12z",
            model_version_id="version_gfs_v1",
            cycle_time=CYCLE_12,
            status="failed",
            zarr_store_path="/tmp/gfs_12.zarr",
        )
    )

    def _product(run_id: str, lead: int) -> None:
        session.add(
            ForecastProduct(
                id=f"p_{run_id}_{lead}",
                run_id=run_id,
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
                zarr_chunk_path="/tmp",
            )
        )

    for lead in (0, 6, 12):
        _product("run_00z", lead)
    for lead in (0, 6):
        _product("run_06z", lead)
    # The failed 12Z run has products but must be excluded.
    for lead in (0,):
        _product("run_12z", lead)
    session.commit()


def test_min_lead_wins_newest_cycle(db: Session) -> None:
    """For each valid_time the minimum-lead record is selected."""
    winners = _select_min_lead_winners(db, "gfs")
    # valid_time 00Z: 00Z+0h (only candidate) -> best (00Z, 0)
    assert winners[datetime(2026, 8, 14, 0, tzinfo=timezone.utc)][0] == (CYCLE_00, 0)
    # valid_time 06Z: 00Z+6h and 06Z+0h -> best min lead 0 (06Z).
    assert winners[datetime(2026, 8, 14, 6, tzinfo=timezone.utc)][0] == (CYCLE_06, 0)
    # valid_time 12Z: 00Z+12h and 06Z+6h -> best min lead 6 (06Z).
    assert winners[datetime(2026, 8, 14, 12, tzinfo=timezone.utc)][0] == (CYCLE_06, 6)
    # valid_time 18Z: 00Z+18h? not present; 06Z+12h not present -> no record.
    assert datetime(2026, 8, 14, 18, tzinfo=timezone.utc) not in winners


def test_newest_cycle_wins_because_lower_lead(db: Session) -> None:
    """The 06Z cycle wins the 12Z valid_time because its lead (6) < 00Z lead (12)."""
    winners = _select_min_lead_winners(db, "gfs")
    assert winners[datetime(2026, 8, 14, 12, tzinfo=timezone.utc)][0] == (CYCLE_06, 6)


def test_failed_run_excluded(db: Session) -> None:
    """A failed run's products never participate in selection."""
    winners = _select_min_lead_winners(db, "gfs")
    # The 12Z failed run would win 12Z valid_time with lead 0, but it must
    # NOT be a candidate. The 06Z run wins with lead 6.
    assert winners[datetime(2026, 8, 14, 12, tzinfo=timezone.utc)][0] == (CYCLE_06, 6)
    # No winner is attributed to the 12Z cycle.
    assert all(
        cycle != CYCLE_12
        for pairs in winners.values()
        for cycle, _ in pairs
    )


def test_processing_and_partial_runs_participate(db: Session) -> None:
    """Processing and partial runs with committed products participate in progressive selection."""
    db.query(ModelRun).filter(ModelRun.id == "run_12z").update({"status": "processing"})
    db.commit()
    winners = _select_min_lead_winners(db, "gfs")
    # With 12Z in processing state and lead 0 committed, valid_time 12Z selects (12Z, 0)
    assert winners[datetime(2026, 8, 14, 12, tzinfo=timezone.utc)][0] == (CYCLE_12, 0)

    db.query(ModelRun).filter(ModelRun.id == "run_12z").update({"status": "partial"})
    db.commit()
    winners = _select_min_lead_winners(db, "gfs")
    assert winners[datetime(2026, 8, 14, 12, tzinfo=timezone.utc)][0] == (CYCLE_12, 0)


def test_no_eligible_run_returns_empty(db: Session) -> None:
    """No eligible (ready/processing/partial) runs -> empty selection."""
    db.query(ModelRun).update({"status": "failed"})
    db.commit()
    winners = _select_min_lead_winners(db, "gfs")
    assert winners == {}


def test_lead_window_filter(db: Session) -> None:
    """start/end lead bounds restrict the candidates."""
    winners = _select_min_lead_winners(
        db, "gfs", start_lead_time_hours=6, end_lead_time_hours=6
    )
    # Only lead-6 records remain: 00Z+6h -> 06Z and 06Z+6h -> 12Z.
    assert set(winners) == {
        datetime(2026, 8, 14, 6, tzinfo=timezone.utc),
        datetime(2026, 8, 14, 12, tzinfo=timezone.utc),
    }


def test_older_cycle_covers_valid_time_when_newest_missing(db: Session) -> None:
    """When the 0h candidate is unavailable, the older cycle's min-lead wins."""
    # Remove the 06Z lead-0 product so the 06Z run only covers 12Z via lead 6.
    db.query(ForecastProduct).filter(
        ForecastProduct.run_id == "run_06z",
        ForecastProduct.lead_time_hours == 0,
    ).delete()
    db.commit()
    winners = _select_min_lead_winners(db, "gfs")
    # valid_time 06Z now only has 00Z+6h -> best (00Z, 6).
    assert winners[datetime(2026, 8, 14, 6, tzinfo=timezone.utc)][0] == (CYCLE_00, 6)
    # valid_time 12Z: 00Z+12h and 06Z+6h -> best min lead 6 (06Z).
    assert winners[datetime(2026, 8, 14, 12, tzinfo=timezone.utc)][0] == (CYCLE_06, 6)
