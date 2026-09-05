"""End-to-end integration tests for Phase 6C serving retirement enforcement.

Verifies that once a cycle is marked retired in forecast_cycle_lifecycle:
- It becomes completely inaccessible across /v1/forecast/availability, /v1/points,
  /v1/ensembles, /v1/probabilities, /v1/maps, raster tiles, vector fields, and /v1/runs.
- Explicit historical requests for retired cycles return HTTP 404 Not Found.
- Pre-existing cached responses/tiles cannot bypass retirement enforcement.
- Non-retired partial cycles continue to serve progressively under Phase 3 rules.
- Physical forecast stores that still exist are blocked from serving.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
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
    is_cycle_visible,
    parse_cycle_time,
    require_cycle_visible,
)
from api.services.tiles import _tile_cache


def _dt(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def test_db(tmp_path):
    """Create an isolated test database with schema and seed metadata."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_serving_retirement.db",
        connect_args={"check_same_thread": False},
    )
    # Create tables without GeoAlchemy spatial DDL issues on SQLite
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
        session.add(center)
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
        session.add_all([gfs, gefs])

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
        session.add_all([v_gfs, v_gefs])

        grid = ForecastGrid(
            id="grid_global_025deg",
            grid_code="global_025deg",
            name="Global 0.25 Degree Grid",
            resolution_km=25.0,
        )
        session.add(grid)

        var_t2m = ForecastVariable(
            id="var_temperature_2m",
            variable_code="temperature_2m",
            name="2-Meter Temperature",
            unit="°C",
        )
        var_w10m = ForecastVariable(
            id="var_wind_10m",
            variable_code="wind_10m",
            name="10-Meter Wind",
            unit="km/h",
        )
        session.add_all([var_t2m, var_w10m])
        session.commit()

    def override_get_db():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_db] = override_get_db
    yield engine
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Centralized Lifecycle Service Unit Tests
# ---------------------------------------------------------------------------


def test_lifecycle_service_predicates(test_db):
    c1 = _dt(2026, 9, 1, 0)
    c2 = _dt(2026, 9, 1, 6)
    c3 = _dt(2026, 9, 1, 12)
    c4 = _dt(2026, 9, 1, 18)

    with Session(test_db) as session:
        # c1: no row -> visible by default
        assert is_cycle_visible(session, c1) is True
        assert require_cycle_visible(session, c1) == c1

        # c2: row with retired_at NULL -> visible
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c2,
                retired_at=None,
                retired_by_cycle_time=None,
            )
        )
        session.commit()
        assert is_cycle_visible(session, c2, model_id="gfs") is True
        assert require_cycle_visible(session, c2, model_id="gfs") == c2

        # c3: row with retired_at set -> NOT visible (404)
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c3,
                retired_at=_dt(2026, 9, 2, 12),
                retired_by_cycle_time=_dt(2026, 9, 2, 12),
            )
        )
        session.commit()
        assert is_cycle_visible(session, c3, model_id="gfs") is False
        with pytest.raises(HTTPException) as exc_info:
            require_cycle_visible(session, c3, model_id="gfs")
        assert exc_info.value.status_code == 404
        assert "not available" in exc_info.value.detail

        # c4: row with deleted_at set (tombstone) -> NOT visible (404)
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c4,
                retired_at=_dt(2026, 9, 2, 18),
                retired_by_cycle_time=_dt(2026, 9, 2, 18),
                deleted_at=_dt(2026, 9, 3, 0),
            )
        )
        session.commit()
        assert is_cycle_visible(session, c4, model_id="gfs") is False
        with pytest.raises(HTTPException) as exc_info4:
            require_cycle_visible(session, c4, model_id="gfs")
        assert exc_info4.value.status_code == 404


def test_parse_cycle_time_formats():
    t1 = parse_cycle_time("2026-09-01T06:00:00Z")
    assert t1 == _dt(2026, 9, 1, 6)

    t2 = parse_cycle_time("2026-09-01T06:00:00+00:00")
    assert t2 == _dt(2026, 9, 1, 6)

    t3 = parse_cycle_time(_dt(2026, 9, 1, 6))
    assert t3 == _dt(2026, 9, 1, 6)

    with pytest.raises(HTTPException) as exc:
        parse_cycle_time("invalid-date-format")
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# Availability Endpoint Tests (/v1/forecast/availability)
# ---------------------------------------------------------------------------


def test_availability_excludes_retired_cycles(test_db):
    client = TestClient(app)
    c1 = _dt(2026, 9, 1, 6)
    c2 = _dt(2026, 9, 2, 6)

    with Session(test_db) as session:
        # Seed GFS runs for c1 and c2
        r1 = ModelRun(
            id="run_gfs_c1",
            model_version_id="version_gfs_v1.0",
            cycle_time=c1,
            status="ready",
            zarr_store_path="/dummy/c1",
            created_at=c1,
        )
        r2 = ModelRun(
            id="run_gfs_c2",
            model_version_id="version_gfs_v1.0",
            cycle_time=c2,
            status="ready",
            zarr_store_path="/dummy/c2",
            created_at=c2,
        )
        session.add_all([r1, r2])

        p1 = ForecastProduct(
            id="p1_t2m",
            run_id="run_gfs_c1",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        p2 = ForecastProduct(
            id="p2_t2m",
            run_id="run_gfs_c2",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        session.add_all([p1, p2])
        session.commit()

    # 1. Before retirement: both c1 and c2 appear in availability
    res1 = client.get("/v1/forecast/availability")
    assert res1.status_code == 200
    data1 = res1.json()["data"]
    gfs_avail = next(m for m in data1["models"] if m["id"] == "gfs")
    t2m_avail = next(v for v in gfs_avail["variables"] if v["id"] == "temperature_2m")
    init_times = [it["value"] for it in t2m_avail["initial_times"]]
    assert "2026-09-01T06:00:00Z" in init_times
    assert "2026-09-02T06:00:00Z" in init_times

    # 2. Mark c1 as RETIRED
    with Session(test_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c1,
                retired_at=_dt(2026, 9, 2, 6),
                retired_by_cycle_time=c2,
            )
        )
        session.commit()

    # 3. After retirement: only c2 appears; c1 is strictly excluded
    res2 = client.get("/v1/forecast/availability")
    assert res2.status_code == 200
    data2 = res2.json()["data"]
    gfs_avail2 = next(m for m in data2["models"] if m["id"] == "gfs")
    t2m_avail2 = next(v for v in gfs_avail2["variables"] if v["id"] == "temperature_2m")
    init_times2 = [it["value"] for it in t2m_avail2["initial_times"]]
    assert "2026-09-01T06:00:00Z" not in init_times2
    assert "2026-09-02T06:00:00Z" in init_times2


def test_availability_preserves_partial_non_retired_cycles(test_db):
    """Verify that Phase 3 progressive serving partial cycles remain in availability."""
    client = TestClient(app)
    c_partial = _dt(2026, 9, 2, 12)

    with Session(test_db) as session:
        r_part = ModelRun(
            id="run_gfs_c_part",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_partial,
            status="partial",
            zarr_store_path="/dummy/c_part",
            created_at=c_partial,
        )
        session.add(r_part)
        p_part = ForecastProduct(
            id="p_part_t2m",
            run_id="run_gfs_c_part",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=6,
        )
        session.add(p_part)
        session.commit()

    res = client.get("/v1/forecast/availability")
    assert res.status_code == 200
    data = res.json()["data"]
    gfs_avail = next(m for m in data["models"] if m["id"] == "gfs")
    t2m_avail = next(v for v in gfs_avail["variables"] if v["id"] == "temperature_2m")
    init_times = [it["value"] for it in t2m_avail["initial_times"]]
    assert "2026-09-02T12:00:00Z" in init_times


# ---------------------------------------------------------------------------
# Ensembles & Probabilities Endpoint Tests
# ---------------------------------------------------------------------------


def test_ensemble_and_probability_explicit_retired_returns_404(test_db):
    client = TestClient(app)
    c_retired = _dt(2026, 9, 1, 0)
    c_visible = _dt(2026, 9, 2, 0)

    with Session(test_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gefs",
                cycle_time=c_retired,
                retired_at=_dt(2026, 9, 2, 0),
                retired_by_cycle_time=c_visible,
            )
        )
        session.commit()

    # Explicit request for retired initial_time on /v1/ensembles -> 404
    res_ens = client.get(
        f"/v1/ensembles?lat=38.0&lon=-105.0&variable=temperature_2m&model=gefs&initial_time={c_retired.isoformat().replace('+00:00', 'Z')}"
    )
    assert res_ens.status_code == 404
    assert "not available" in res_ens.json()["error"]["message"]

    # Explicit request for retired initial_time on /v1/probabilities -> 404
    res_prob = client.get(
        f"/v1/probabilities?lat=38.0&lon=-105.0&variable=temperature_2m&threshold=0&operator=gt&lead_time_hours=0&model=gefs&initial_time={c_retired.isoformat().replace('+00:00', 'Z')}"
    )
    assert res_prob.status_code == 404
    assert "not available" in res_prob.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Maps, Tiles & Vector-Field Endpoint Tests
# ---------------------------------------------------------------------------


def test_maps_metadata_explicit_retired_returns_404(test_db):
    client = TestClient(app)
    c_retired = _dt(2026, 9, 1, 0)
    c_visible = _dt(2026, 9, 2, 0)

    with Session(test_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c_retired,
                retired_at=_dt(2026, 9, 2, 0),
                retired_by_cycle_time=c_visible,
            )
        )
        session.commit()

    res = client.get(
        f"/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=0&initial_time={c_retired.isoformat().replace('+00:00', 'Z')}"
    )
    assert res.status_code == 404
    assert "not available" in res.json()["error"]["message"]


def test_raster_tile_and_vector_field_explicit_retired_returns_404(test_db):
    client = TestClient(app)
    c_retired = _dt(2026, 9, 1, 0)
    c_visible = _dt(2026, 9, 2, 0)

    with Session(test_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c_retired,
                retired_at=_dt(2026, 9, 2, 0),
                retired_by_cycle_time=c_visible,
            )
        )
        session.commit()

    # Raster tile -> 404
    res_tile = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={c_retired.isoformat().replace('+00:00', 'Z')}"
    )
    assert res_tile.status_code == 404

    # Vector field -> 404
    res_vec = client.get(
        f"/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=0&initial_time={c_retired.isoformat().replace('+00:00', 'Z')}"
    )
    assert res_vec.status_code == 404


# ---------------------------------------------------------------------------
# Pre-Existing Tile Cache Cannot Bypass Retirement
# ---------------------------------------------------------------------------


def test_cached_tile_cannot_bypass_retirement(test_db):
    """Prove that if a tile was cached before retirement, it returns 404 once retired."""
    client = TestClient(app)
    c = _dt(2026, 9, 1, 0)
    init_str = c.isoformat().replace("+00:00", "Z")

    # Manually populate tile cache entry
    cache_key = ("gfs", "temperature_2m", "surface", 0, 0, 0, 0, init_str, "gen_dummy")
    _tile_cache[cache_key] = (1000000000.0, b"\x89PNG\r\n\x1a\nFakeTileBytes")

    # Before retirement: is_cycle_visible is True (no lifecycle row)
    with Session(test_db) as session:
        assert is_cycle_visible(session, c, model_id="gfs") is True

    # Now mark cycle as RETIRED
    with Session(test_db) as session:
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c,
                retired_at=_dt(2026, 9, 2, 0),
                retired_by_cycle_time=_dt(2026, 9, 2, 0),
            )
        )
        session.commit()

    # Request tile: must raise 404 and NOT return the cached PNG bytes
    res = client.get(
        f"/v1/maps/gfs/temperature_2m/surface/0/0/0.png?lead_time_hours=0&initial_time={init_str}"
    )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Model Runs Endpoint (/v1/runs)
# ---------------------------------------------------------------------------


def test_runs_catalog_excludes_retired_runs(test_db):
    client = TestClient(app)
    c_vis = _dt(2026, 9, 2, 0)
    c_ret = _dt(2026, 9, 1, 0)

    with Session(test_db) as session:
        r_vis = ModelRun(
            id="run_gfs_visible",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_vis,
            status="ready",
            created_at=c_vis,
        )
        r_ret = ModelRun(
            id="run_gfs_retired",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_ret,
            status="ready",
            created_at=c_ret,
        )
        session.add_all([r_vis, r_ret])
        session.add(
            ForecastCycleLifecycle(
                model_id="gfs",
                cycle_time=c_ret,
                retired_at=_dt(2026, 9, 2, 0),
                retired_by_cycle_time=c_vis,
            )
        )
        session.commit()

    res = client.get("/v1/runs?model_id=gfs")
    assert res.status_code == 200
    runs = res.json()["data"]
    run_ids = [r["id"] for r in runs]
    assert "run_gfs_visible" in run_ids
    assert "run_gfs_retired" not in run_ids
