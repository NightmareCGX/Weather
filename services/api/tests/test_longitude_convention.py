"""Integration tests for 0-360 longitude-convention datasets in the serving path.

Native GFS grids store longitude in the ``[0, 360]`` convention (0..359.75).
The point/probability/ensemble serving path must align western-hemisphere
WGS84 coordinates (e.g. ``lat=39.19, lon=-106.82``) into that convention so
they interpolate instead of being rejected by ``RegularGrid.contains``.

This module is self-contained: it seeds its own deterministic 0-360 models,
versions, runs, and Zarr stores, so the shared ``conftest`` fixtures and the
catalog contract tests are untouched.
"""

import os
import sys
from datetime import datetime, timezone

# Ensure services/api/src is on sys.path for test execution (mirrors conftest).
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from domain.coverage import register_expected_members
from alembic import command
from api.core.database import get_db
from api.main import app
from api.models.entities import (
    EnsembleMember,
    EnsembleMemberProduct,
    ForecastCenter,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from tests.fixtures import (
    LAT_START,
    LEAD_TIMES,
    LON_START,
    MEMBER_COUNT,
    MEMBER_INDICES,
    ensemble_precipitation_at,
    ensemble_temperature_at,
    precipitation_at,
    temperature_at,
    write_ensemble_zarr_0_360,
    write_forecast_zarr_0_360,
)

#: Test point inside the 0-360 fixture grid (a western-hemisphere coordinate).
LAT = LAT_START + 0.125  # 38.125
LON = LON_START + 0.125  # -106.875
LEAD = 6


@pytest.fixture(scope="module")
def longitude_client(tmp_path_factory):
    """TestClient with its own migrated schema, 0-360 stores, and seed data.

    The schema is migrated to head, the two 0-360 Zarr stores are written to
    a temp directory, and catalog rows are seeded pointing at them. Every test
    in this module uses these fixtures, isolated from the shared ``conftest``.
    """
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            "PostgreSQL test instance not running or reachable; skipping integration tests."
        )

    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))

    command.upgrade(alembic_cfg, "head")

    base = tmp_path_factory.mktemp("zarr_0_360")
    gfs_store = str(base / "gfs_0_360")
    gefs_store = str(base / "gefs_0_360")
    write_forecast_zarr_0_360(gfs_store)
    write_ensemble_zarr_0_360(gefs_store)

    with Session(engine) as session:
        noaa = ForecastCenter(
            id="center_noaa",
            center_id="noaa",
            name="National Oceanic and Atmospheric Administration",
            country="USA",
        )
        gfs = Model(
            id="model_gfs_0_360",
            model_id="gfs_0_360",
            name="Global Forecast System (0-360 lon)",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
        )
        gefs = Model(
            id="model_gefs_0_360",
            model_id="gefs_0_360",
            name="Global Ensemble Forecast System (0-360 lon)",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
        )
        version_gfs = ModelVersion(
            id="version_gfs_0_360_v1", model_id="gfs_0_360", version_string="v1.0"
        )
        version_gefs = ModelVersion(
            id="version_gefs_0_360_v1", model_id="gefs_0_360", version_string="v1.0"
        )
        run_gfs = ModelRun(
            id="run_2026072100_gfs_0_360",
            model_version_id="version_gfs_0_360_v1",
            cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
            status="ready",
            zarr_store_path=gfs_store,
        )
        run_gefs = ModelRun(
            id="run_2026072100_gefs_0_360",
            model_version_id="version_gefs_0_360_v1",
            cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
            status="ready",
            zarr_store_path=gefs_store,
        )
        ensemble_members = [
            EnsembleMember(
                id=f"member_{i}_gefs_0_360",
                run_id="run_2026072100_gefs_0_360",
                member_index=i,
                member_name=f"gefs_0_360_member_{i}",
            )
            for i in MEMBER_INDICES
        ]
        temperature = ForecastVariable(
            id="var_temperature_2m",
            variable_code="temperature_2m",
            name="2-Meter Temperature",
            unit="°C",
        )
        precipitation = ForecastVariable(
            id="var_precipitation_rate",
            variable_code="precipitation_rate",
            name="Precipitation Rate",
            unit="mm/h",
        )
        register_expected_members("gefs_0_360", MEMBER_COUNT)
        ensemble_member_products = [
            EnsembleMemberProduct(
                id=f"emp_gefs_0_360_{i}_{lead}",
                run_id="run_2026072100_gefs_0_360",
                member_index=i,
                lead_time_hours=lead,
            )
            for i in MEMBER_INDICES
            for lead in LEAD_TIMES
        ]
        session.add_all(
            [
                noaa,
                gfs,
                gefs,
                version_gfs,
                version_gefs,
                run_gfs,
                run_gefs,
                *ensemble_members,
                *ensemble_member_products,
                temperature,
                precipitation,
            ]
        )
        session.commit()

    def override_get_db():
        db = Session(engine)
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)

    command.downgrade(alembic_cfg, "base")
    engine.dispose()


def _assert_envelope(body, object_type):
    assert body["object"] == object_type
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_points_western_hemisphere_on_0_360_store(longitude_client):
    # -106.875 must interpolate to the exact analytic value, not 404.
    resp = longitude_client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gfs_0_360")
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body, "point_forecast")
    data = body["data"]
    assert data["model"] == "gfs_0_360"
    assert data["location"]["latitude"] == pytest.approx(LAT)
    assert data["location"]["longitude"] == pytest.approx(LON)
    entry = next(e for e in data["forecasts"] if e["lead_time_hours"] == LEAD)
    assert entry["temperature_2m"] == pytest.approx(temperature_at(LAT, LON, LEAD))
    assert entry["precipitation_rate"] == pytest.approx(precipitation_at(LEAD))


def test_points_dateline_positive_longitude_on_0_360_store(longitude_client):
    # A positive longitude inside the [0, 340] axis is accepted as-is.
    resp = longitude_client.get("/v1/points?lat=38.125&lon=20.0&models=gfs_0_360")
    assert resp.status_code == 200
    assert resp.json()["data"]["model"] == "gfs_0_360"


def test_points_0_360_outer_longitude_404(longitude_client):
    # A WGS84 longitude in (-20, 0) aligns to (340, 360), outside the compact
    # [0, 340] fixture grid, and must 404 (not silently succeed).
    resp = longitude_client.get("/v1/points?lat=38.125&lon=-10.0&models=gfs_0_360")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_ensembles_western_hemisphere_on_0_360_store(longitude_client):
    resp = longitude_client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        f"&variable=temperature_2m&lead_time_hours={LEAD}"
        "&model=gefs_0_360"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body, "ensemble_statistics")
    data = body["data"]
    assert data["model"] == "gefs_0_360"
    assert data["member_count"] == MEMBER_COUNT

    members = [
        ensemble_temperature_at(member, LAT, LON, LEAD) for member in MEMBER_INDICES
    ]
    expected = {
        "mean": float(np.mean(members)),
        "median": float(np.median(members)),
        "spread": float(np.std(members, ddof=0)),
        "p10": float(np.percentile(members, 10, method="linear")),
        "p25": float(np.percentile(members, 25, method="linear")),
        "p50": float(np.percentile(members, 50, method="linear")),
        "p75": float(np.percentile(members, 75, method="linear")),
        "p90": float(np.percentile(members, 90, method="linear")),
    }
    stats = data["statistics"]
    for key, value in expected.items():
        assert stats[key] == pytest.approx(value)


def test_probabilities_western_hemisphere_on_0_360_store(longitude_client):
    resp = longitude_client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&threshold=4&operator=gt"
        f"&lead_time_hours={LEAD}&model=gefs_0_360"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body, "probability_forecast")
    data = body["data"]
    assert data["variable"] == "precipitation_rate"
    assert data["probability"] == pytest.approx(0.6)

    members = [ensemble_precipitation_at(member, LEAD) for member in MEMBER_INDICES]
    lower, upper = data["confidence_interval_95"]
    assert 0.0 <= lower <= data["probability"] <= upper <= 1.0
    assert len(members) == MEMBER_COUNT


def test_normalize_grid_longitudes_fully_western_0_360_axis():
    """A 0-360 axis confined to the western hemisphere maps to -180..180.

    cfgrib decodes every valid GRIB grid in the native 0..360 convention. A
    western-subset store (e.g. a small GFS grid covering lon 250..259) therefore
    arrives with an origin above 180, which RegularGrid cannot represent. The
    serving tier must shift such an axis into [-180, 180] while preserving
    order and spacing.
    """
    from api.services.point_forecast import _normalize_grid_longitudes

    western = [250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 256.0, 257.0, 258.0, 259.0]
    normalized = _normalize_grid_longitudes(western)
    assert normalized[0] == pytest.approx(-110.0)
    assert normalized[-1] == pytest.approx(-101.0)
    # Order and uniform spacing are preserved by the exact 360-degree shift.
    assert normalized == pytest.approx([-110.0, -109.0, -108.0, -107.0, -106.0,
                                        -105.0, -104.0, -103.0, -102.0, -101.0])


def test_normalize_grid_longitudes_leaves_other_conventions_unchanged():
    """Axes that are not fully western are not shifted.

    A global 0..360 axis (origin 0) and an already -180..180 axis must pass
    through unchanged so ``RegularGrid.align_longitude`` maps western queries
    into the 0..360 store as documented.
    """
    from api.services.point_forecast import _normalize_grid_longitudes

    global_axis = list(range(0, 360, 20))
    assert _normalize_grid_longitudes(global_axis) == global_axis
    minus_180_axis = [-107.0, -106.0, -105.0, -104.0]
    assert _normalize_grid_longitudes(minus_180_axis) == minus_180_axis
    # An empty axis is returned unchanged.
    assert _normalize_grid_longitudes([]) == []
