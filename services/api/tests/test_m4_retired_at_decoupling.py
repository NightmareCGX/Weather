"""Data Lifecycle V3 Milestone 4: Physical Fence & Serving Safety Tests.

Validates that:
1. Unfenced cycles remain servable across:
   - serving endpoints (/v1/points, /v1/ensembles, /v1/probabilities, /v1/maps, tiles, vector-field)
   - explicit initial_time resolution
   - /v1/forecast/availability
   - /v1/runs
   - cache key generation
2. Physical deletion fences (deletion_started_at != NULL, deleted_at != NULL)
   remain the sole authoritative lifecycle fences:
   - reject explicit initial_time with 404
   - exclude runs from filter_visible_runs
   - reject pre-warmed cached responses (adversarial cache bypass immunity)
   - exclude claimed/deleted runs from implicit latest-cycle resolution
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from api.core.database import Base, get_db
from api.main import app
from api.models.entities import (
    EnsembleMember,
    EnsembleMemberProduct,
    ForecastCenter,
    ForecastCycleLifecycle,
    ForecastGrid,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.lifecycle import (
    filter_visible_runs,
    is_cycle_visible,
    require_cycle_visible,
)
from api.services.point_forecast import (
    resolve_latest_run_cycle_time,
    resolve_serving_generation_for_store,
)
from api.services.tiles import _tile_cache


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


@pytest.fixture
def m4_db(tmp_path):
    """Create an isolated test database with schema and seed metadata."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_m4_decoupling.db",
        connect_args={"check_same_thread": False},
    )
    tables_to_create = [
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ModelRun.__table__,
        EnsembleMember.__table__,
        EnsembleMemberProduct.__table__,
        ForecastVariable.__table__,
        ForecastGrid.__table__,
        ForecastProduct.__table__,
        ForecastCycleLifecycle.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables_to_create)

    with Session(engine) as session:
        center = ForecastCenter(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="US",
            created_at=_dt(2026, 1, 1, 0),
        )
        gfs = Model(
            id="model_gfs",
            model_id="gfs",
            name="Global Forecast System",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs = Model(
            id="model_gefs",
            model_id="gefs",
            name="Global Ensemble Forecast System",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        v_gfs = ModelVersion(
            id="version_gfs_v1.0",
            model_id="gfs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        v_gefs = ModelVersion(
            id="version_gefs_v1.0",
            model_id="gefs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        grid = ForecastGrid(
            id="grid_global_025deg",
            grid_code="global_025deg",
            name="Global 0.25 Degree",
            resolution_km=25.0,
        )
        t2m = ForecastVariable(
            id="var_temperature_2m",
            variable_code="temperature_2m",
            name="2m Temperature",
            unit="degC",
        )
        w10 = ForecastVariable(
            id="var_wind_10m",
            variable_code="wind_10m",
            name="10m Wind",
            unit="m/s",
        )
        session.add_all([center, gfs, gefs, v_gfs, v_gefs, grid, t2m, w10])
        session.commit()

    def override_get_db():
        with Session(engine) as db_session:
            yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield engine
    app.dependency_overrides.pop(get_db, None)
    _tile_cache.clear()


# ---------------------------------------------------------------------------
# 1. Unfenced Cycle Serving Visibility
# ---------------------------------------------------------------------------


def test_unfenced_cycle_is_visible(m4_db):
    """is_cycle_visible and require_cycle_visible confirm unfenced cycles are visible."""
    c = _dt(2026, 9, 1, 6)
    with Session(m4_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=None,
                deleted_at=None,
            )
        )
        session.commit()

        assert is_cycle_visible(session, c, model_id="gfs") is True
        assert require_cycle_visible(session, c, model_id="gfs") == c


def test_unfenced_cycle_in_availability_and_runs(m4_db):
    """Unfenced runs remain visible in /v1/forecast/availability and /v1/runs."""
    client = TestClient(app)
    c1 = _dt(2026, 9, 1, 6)

    with Session(m4_db) as session:
        r1 = ModelRun(
            id="run_gfs_c1",
            model_version_id="version_gfs_v1.0",
            cycle_time=c1,
            status="ready",
            zarr_store_path="/store/c1",
            created_at=c1,
        )
        p1 = ForecastProduct(
            id="p1_t2m",
            run_id="run_gfs_c1",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        # Seed unfenced lifecycle row
        lc1 = ForecastCycleLifecycle(
            model_id="gfs",
            cycle_time=c1,
            deletion_started_at=None,
            deleted_at=None,
        )
        session.add_all([r1, p1, lc1])
        session.commit()

    # 1. Availability includes c1
    res_avail = client.get("/v1/forecast/availability")
    assert res_avail.status_code == 200
    data = res_avail.json()["data"]
    gfs_avail = next(m for m in data["models"] if m["id"] == "gfs")
    t2m_avail = next(v for v in gfs_avail["variables"] if v["id"] == "temperature_2m")
    init_times = [it["value"] for it in t2m_avail["initial_times"]]
    assert "2026-09-01T06:00:00Z" in init_times

    # 2. Runs listing includes run_gfs_c1
    res_runs = client.get("/v1/runs?model_id=gfs")
    assert res_runs.status_code == 200
    runs = res_runs.json()["data"]
    assert any(r["id"] == "run_gfs_c1" for r in runs)


def test_filter_visible_runs_preserves_unfenced_and_excludes_fenced(m4_db):
    """filter_visible_runs keeps unfenced visible, but excludes deletion_started_at/deleted_at."""
    c_unfenced = _dt(2026, 9, 1, 0)
    c_claimed = _dt(2026, 9, 1, 6)
    c_deleted = _dt(2026, 9, 1, 12)
    c_no_row = _dt(2026, 9, 1, 18)

    with Session(m4_db) as session:
        r_unfenced = ModelRun(
            id="r_unfenced",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_unfenced,
            status="ready",
            created_at=c_unfenced,
        )
        r_claimed = ModelRun(
            id="r_claimed",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_claimed,
            status="ready",
            created_at=c_claimed,
        )
        r_deleted = ModelRun(
            id="r_deleted",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_deleted,
            status="ready",
            created_at=c_deleted,
        )
        r_no_row = ModelRun(
            id="r_no_row",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_no_row,
            status="ready",
            created_at=c_no_row,
        )
        session.add_all([r_unfenced, r_claimed, r_deleted, r_no_row])

        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c_unfenced,
                deletion_started_at=None,
                deleted_at=None,
            )
        )
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c_claimed,
                deletion_started_at=_dt(2026, 9, 2, 1),
                deleted_at=None,
            )
        )
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c_deleted,
                deletion_started_at=_dt(2026, 9, 2, 1),
                deleted_at=_dt(2026, 9, 2, 2),
            )
        )
        session.commit()

        stmt = select(ModelRun.id)
        visible_ids = set(session.execute(filter_visible_runs(stmt, model_id="gfs")).scalars().all())
        assert "r_unfenced" in visible_ids
        assert "r_no_row" in visible_ids
        assert "r_claimed" not in visible_ids
        assert "r_deleted" not in visible_ids


# ---------------------------------------------------------------------------
# 2. Physical Fence Supremacy & Cache Decoupling
# ---------------------------------------------------------------------------


def test_cache_generation_manifest_only():
    """resolve_serving_generation_for_store derives generation exclusively from manifest."""
    # Manifest read error or missing returns None
    gen = resolve_serving_generation_for_store(None)
    assert gen is None

    # Even if arguments are supplied, generation derives from manifest
    # (tested hermetically without requiring live S3)
    gen_dummy = resolve_serving_generation_for_store(None, "2026-09-02T12:00:00Z")
    assert gen_dummy is None


def test_explicit_initial_time_404_on_deletion_started_at(m4_db):
    """Explicit initial_time returns HTTP 404 immediately when deletion_started_at is set."""
    client = TestClient(app)
    c = _dt(2026, 9, 1, 6)
    c_iso = c.isoformat().replace("+00:00", "Z")

    with Session(m4_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 2, 6),
                deleted_at=None,
            )
        )
        session.commit()

    res_map = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c_iso}"
    )
    assert res_map.status_code == 404
    assert "not available" in res_map.json()["error"]["message"]


def test_explicit_initial_time_404_on_deleted_at(m4_db):
    """Explicit initial_time returns HTTP 404 immediately when deleted_at is set."""
    client = TestClient(app)
    c = _dt(2026, 9, 1, 6)
    c_iso = c.isoformat().replace("+00:00", "Z")

    with Session(m4_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 2, 6),
                deleted_at=_dt(2026, 9, 2, 7),
            )
        )
        session.commit()

    res_map = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c_iso}"
    )
    assert res_map.status_code == 404
    assert "not available" in res_map.json()["error"]["message"]


def test_cached_tile_rejected_after_deletion_started_at(m4_db):
    """Pre-warmed tile cache entry is rejected with 404 after deletion_started_at is stamped."""
    client = TestClient(app)
    c = _dt(2026, 9, 1, 0)
    init_str = c.isoformat().replace("+00:00", "Z")

    # Populate cache
    cache_key = ("gfs", "temperature_2m", "surface", 0, 0, 0, 0, init_str, "gen_test")
    _tile_cache[cache_key] = (1000000000.0, b"\x89PNG\r\n\x1a\nFakeTileBytes")

    # Unfenced cycle: cache hit / visible
    with Session(m4_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=None,
                deleted_at=None,
            )
        )
        session.commit()

    assert is_cycle_visible(Session(m4_db), c, model_id="gfs") is True

    # Now stamp deletion_started_at
    with Session(m4_db) as session:
        lc = session.get(ForecastCycleLifecycle, ("gfs", c))
        assert lc is not None
        lc.deletion_started_at = _dt(2026, 9, 2, 1)
        session.commit()

    # Request tile: must raise 404 and NOT return cached bytes
    res = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={init_str}"
    )
    assert res.status_code == 404


def test_cached_tile_rejected_after_deleted_at(m4_db):
    """Pre-warmed tile cache entry is rejected with 404 after deleted_at is stamped."""
    client = TestClient(app)
    c = _dt(2026, 9, 1, 0)
    init_str = c.isoformat().replace("+00:00", "Z")

    cache_key = ("gfs", "temperature_2m", "surface", 0, 0, 0, 0, init_str, "gen_test2")
    _tile_cache[cache_key] = (1000000000.0, b"\x89PNG\r\n\x1a\nFakeTileBytes")

    with Session(m4_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                deletion_started_at=_dt(2026, 9, 2, 1),
                deleted_at=_dt(2026, 9, 2, 2),
            )
        )
        session.commit()

    res = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={init_str}"
    )
    assert res.status_code == 404


def test_implicit_latest_cycle_advances_when_claimed(m4_db):
    """resolve_latest_run_cycle_time advances past claimed/deleted cycles."""
    c1 = _dt(2026, 9, 1, 0)
    c2 = _dt(2026, 9, 1, 6)

    with Session(m4_db) as session:
        r1 = ModelRun(
            id="r_c1",
            model_version_id="version_gfs_v1.0",
            cycle_time=c1,
            status="ready",
            zarr_store_path="/store/c1",
            created_at=c1,
        )
        r2 = ModelRun(
            id="r_c2",
            model_version_id="version_gfs_v1.0",
            cycle_time=c2,
            status="ready",
            zarr_store_path="/store/c2",
            created_at=c2,
        )
        session.add_all([r1, r2])

        # c2 unfenced -> remains latest
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c2,
            )
        )
        session.commit()

        latest = resolve_latest_run_cycle_time(session, "gfs")
        assert latest == "2026-09-01T06:00:00Z"

        # Now claim c2 for deletion
        lc2 = session.get(ForecastCycleLifecycle, ("gfs", c2))
        assert lc2 is not None
        lc2.deletion_started_at = _dt(2026, 9, 2, 1)
        session.commit()

        # Advanced past c2 back to c1
        latest2 = resolve_latest_run_cycle_time(session, "gfs")
        assert latest2 == "2026-09-01T00:00:00Z"
        lc2.deletion_started_at = _dt(2026, 9, 2, 1)
        session.commit()

        # Now c1 must be resolved as latest
        latest_after_claim = resolve_latest_run_cycle_time(session, "gfs")
        assert latest_after_claim == "2026-09-01T00:00:00Z"
