"""Unit and catalog integration tests for Phase 6B lifecycle metadata and operations (Lifecycle V2)."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.db import CatalogBase
from ingestion.core.catalog import (
    CenterRecord,
    ForecastCycleLifecycleRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    _ensure_utc_datetime,
    ensure_lifecycle_row,
    list_cycle_lifecycle_snapshots,
    list_model_ready_cycle_times,
    list_paired_ready_cycle_times,
)


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


@pytest.fixture
def db_session() -> Session:
    """Provide an isolated in-memory SQLite database session for catalog tests."""
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as session:
        # Seed standard centers and models
        center = CenterRecord(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="US",
            created_at=_dt(2026, 1, 1, 0),
        )
        session.add(center)
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
        session.add_all([gfs, gefs])
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
        session.add_all([gfs_v1, gefs_v1])
        session.commit()
        yield session


def _add_run(
    db: Session,
    model_id: str,
    cycle_time: datetime,
    status: str,
    version_string: str = "v1.0",
) -> ModelRunRecord:
    v_id = f"version_{model_id}_{version_string}"
    run = ModelRunRecord(
        id=f"run_{v_id}_{cycle_time.strftime('%Y%m%d%H%M')}_{model_id}",
        model_version_id=v_id,
        cycle_time=cycle_time,
        status=status,
        created_at=cycle_time,
    )
    db.add(run)
    db.commit()
    return run


# ---------------------------------------------------------------------------
# Ready Query Tests (Model-Scoped and Paired Legacy)
# ---------------------------------------------------------------------------


def test_list_model_ready_cycle_times_empty(db_session: Session) -> None:
    assert list_model_ready_cycle_times(db_session, "gfs") == []
    assert list_paired_ready_cycle_times(db_session) == []


def test_list_model_ready_cycle_times_unpaired_scenarios(db_session: Session) -> None:
    c1 = _dt(2026, 9, 1, 0)
    c2 = _dt(2026, 9, 1, 6)

    # c1: only GFS ready
    _add_run(db_session, "gfs", c1, "ready")

    # c2: only GEFS ready
    _add_run(db_session, "gefs", c2, "ready")

    # GFS list sees c1, GEFS list sees c2
    assert list_model_ready_cycle_times(db_session, "gfs") == [c1]
    assert list_model_ready_cycle_times(db_session, "gefs") == [c2]

    # Paired list sees neither
    assert list_paired_ready_cycle_times(db_session) == []


def test_list_model_ready_cycle_times_version_safety(db_session: Session) -> None:
    c1 = _dt(2026, 9, 1, 0)

    gefs_v2 = ModelVersionRecord(
        id="version_gefs_v2.0",
        model_id="gefs",
        version_string="v2.0",
        created_at=_dt(2026, 1, 1, 0),
    )
    db_session.add(gefs_v2)
    db_session.commit()

    _add_run(db_session, "gfs", c1, "ready", version_string="v1.0")
    _add_run(db_session, "gefs", c1, "ready", version_string="v2.0")

    assert list_model_ready_cycle_times(db_session, "gfs", version_string="v1.0") == [c1]
    assert list_model_ready_cycle_times(db_session, "gefs", version_string="v1.0") == []
    assert list_model_ready_cycle_times(db_session, "gefs", version_string="v2.0") == [c1]


# ---------------------------------------------------------------------------
# Lifecycle Snapshot Discovery Tests
# ---------------------------------------------------------------------------


def test_list_cycle_lifecycle_snapshots_discovery(db_session: Session) -> None:
    c1 = _dt(2026, 9, 1, 0)
    c2 = _dt(2026, 9, 1, 6)

    _add_run(db_session, "gfs", c1, "ready")
    _add_run(db_session, "gefs", c1, "partial")

    # c2 has a lifecycle row directly
    ensure_lifecycle_row(db_session, "gfs", c2)
    db_session.commit()

    snaps_gfs = list_cycle_lifecycle_snapshots(db_session, model_id="gfs")
    assert len(snaps_gfs) == 2
    assert snaps_gfs[0].model_id == "gfs"
    assert snaps_gfs[0].cycle_time == c1
    assert snaps_gfs[0].status == "ready"
    assert snaps_gfs[0].is_deletion_started is False
    assert snaps_gfs[0].is_deleted is False

    assert snaps_gfs[1].cycle_time == c2
    assert snaps_gfs[1].is_deletion_started is False
    assert snaps_gfs[1].is_deleted is False

    snaps_gefs = list_cycle_lifecycle_snapshots(db_session, model_id="gefs")
    assert len(snaps_gefs) == 1
    assert snaps_gefs[0].model_id == "gefs"
    assert snaps_gefs[0].cycle_time == c1
    assert snaps_gefs[0].status == "partial"


# ---------------------------------------------------------------------------
# Lifecycle Row Persistence & Idempotency Tests
# ---------------------------------------------------------------------------


def test_ensure_lifecycle_row(db_session: Session) -> None:
    c = _dt(2026, 9, 1, 6)
    row1 = ensure_lifecycle_row(db_session, "gfs", c)
    assert row1.model_id == "gfs"
    assert _ensure_utc_datetime(row1.cycle_time) == c
    assert row1.deletion_started_at is None
    assert row1.deleted_at is None
    db_session.commit()

    # Second call returns existing
    row2 = ensure_lifecycle_row(db_session, "gfs", c)
    assert (row1.model_id, _ensure_utc_datetime(row1.cycle_time)) == (
        row2.model_id,
        _ensure_utc_datetime(row2.cycle_time),
    )


def test_lifecycle_tombstone_survives_model_run_deletion(db_session: Session) -> None:
    """Prove that forecast_cycle_lifecycle has no FK dependency on model_runs."""
    c = _dt(2026, 9, 1, 0)
    run_gfs = _add_run(db_session, "gfs", c, "ready")

    row = ensure_lifecycle_row(db_session, "gfs", c)
    setattr(row, "deleted_at", _dt(2026, 9, 2, 0))
    db_session.commit()

    # Delete the model_runs rows (simulating physical GC cleanup)
    db_session.delete(run_gfs)
    db_session.commit()

    # The lifecycle record remains completely intact!
    row_after = db_session.get(ForecastCycleLifecycleRecord, ("gfs", c))
    assert row_after is not None
    assert _ensure_utc_datetime(row_after.deleted_at) == _dt(2026, 9, 2, 0)
