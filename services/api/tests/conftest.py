import os
import sys
from datetime import datetime, timezone

# Ensure services/api/src is on sys.path for test execution
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# The API depends on weather-platform-domain (which provides domain.locks for
# the reader-gate). In the workspace dev/test environment, point at the package
# source so the new domain.locks module is found without a reinstall. CI uses
# ``poetry install`` which installs the path dependency normally.
_domain_src = os.path.abspath(os.path.join(current_dir, "../../../packages/domain/src"))
if _domain_src not in sys.path:
    sys.path.insert(0, _domain_src)

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from geoalchemy2 import WKTElement
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from alembic import command
from api.core.database import get_db
from api.main import app
from api.models.entities import (
    City,
    EnsembleMember,
    ForecastCenter,
    ForecastGrid,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
    SkiResort,
    Station,
    VerificationObservation,
)
from tests.fixtures import (
    MEMBER_INDICES,
    write_ensemble_zarr,
    write_forecast_zarr,
)


@pytest.fixture(scope="session")
def db_engine():
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
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_db(db_engine):
    """Apply the Alembic schema to a clean database, following the existing
    ``test_migrations.py`` reset-and-upgrade convention.

    Module-scoped so that each DB-consuming test module re-migrates its own
    clean schema. ``test_migrations.py`` destructively drops and downgrades
    the shared schema to ``base``; keeping the fixtures module-scoped makes
    every module independent of that and immune to cross-module corruption.
    """
    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", str(db_engine.url))
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    with db_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))

    command.upgrade(alembic_cfg, "head")
    yield db_engine
    command.downgrade(alembic_cfg, "base")


@pytest.fixture(scope="session")
def tmp_zarr_stores(tmp_path_factory):
    """Write local on-disk Zarr fixture stores for the seeded model runs.

    Returns a mapping of model id to the local Zarr store directory. The
    stores are written once per session and referenced by the seeded
    ``model_runs.zarr_store_path`` rows so the point-forecast integration
    tests slice real datasets without requiring MinIO/S3.
    """
    base = tmp_path_factory.mktemp("zarr")
    gfs_store = str(base / "gfs")
    gefs_store = str(base / "gefs")
    write_forecast_zarr(gfs_store)
    # The gefs store is an ensemble dataset with a ``member`` dimension so the
    # /v1/ensembles and /v1/probabilities integration tests slice real
    # per-member data.
    write_ensemble_zarr(gefs_store)
    return {"gfs": gfs_store, "gefs": gefs_store}


def _seed_verification_observations(session: Session) -> None:
    """Seed forecast products and verification observations for the gfs run.

    The ready gfs run ``run_2026072100_gfs`` cycles at 2026-07-21T00:00Z and
    its fixture Zarr store carries lead times [0, 6, 12, 18]. Forecast product
    rows are seeded for ``temperature_2m`` and ``precipitation_rate`` across
    those leads so ``/v1/verifications`` can pair observations with forecast
    values. Each observation below has a valid time matching exactly one
    product (2026-07-21T06:00Z = 00Z+6h; 2026-07-21T18:00Z = 00Z+18h), so it
    yields a single forecast/observation pair.
    """
    lead_times = [0, 6, 12, 18]
    session.add_all(
        [
            ForecastProduct(
                id=f"product_gfs_temperature_2m_{lead}",
                run_id="run_2026072100_gfs",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
            )
            for lead in lead_times
        ]
    )
    session.add_all(
        [
            ForecastProduct(
                id=f"product_gfs_precipitation_rate_{lead}",
                run_id="run_2026072100_gfs",
                variable_id="precipitation_rate",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
            )
            for lead in lead_times
        ]
    )
    session.add_all(
        [
            ForecastProduct(
                id=f"product_gfs_precipitation_amount_3h_{lead}",
                run_id="run_2026072100_gfs",
                variable_id="precipitation_amount_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
            )
            for lead in lead_times
        ]
    )
    # The ready gefs run also has forecast products so the availability and map
    # metadata endpoints advertise it (a ready run without products is not
    # servable and must not be advertised).
    session.add_all(
        [
            ForecastProduct(
                id=f"product_gefs_temperature_2m_{lead}",
                run_id="run_2026072100_gefs",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
            )
            for lead in lead_times
        ]
    )
    session.add_all(
        [
            ForecastProduct(
                id=f"product_gefs_precipitation_rate_{lead}",
                run_id="run_2026072100_gefs",
                variable_id="precipitation_rate",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
            )
            for lead in lead_times
        ]
    )
    session.add_all(
        [
            ForecastProduct(
                id=f"product_gefs_precipitation_amount_3h_{lead}",
                run_id="run_2026072100_gefs",
                variable_id="precipitation_amount_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
            )
            for lead in lead_times
        ]
    )
    session.add_all(
        [
            VerificationObservation(
                id="obs_20260721_06z_temperature_2m",
                station_id="KASE",
                valid_time=datetime(2026, 7, 21, 6, 0, tzinfo=timezone.utc),
                variable_code="temperature_2m",
                observed_value=20.0,
            ),
            VerificationObservation(
                id="obs_20260721_18z_precipitation_rate",
                station_id="KASE",
                valid_time=datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc),
                variable_code="precipitation_rate",
                observed_value=12.0,
            ),
        ]
    )


def _seed_locations(session: Session) -> None:
    """Seed PostGIS location records used by the search and point tests.

    Coordinates are chosen inside the fixture Zarr grid
    (latitude 38.0-38.75, longitude -107.0 to -106.25).
    """
    session.add_all(
        [
            City(
                id="city_denver",
                city_name="Denver",
                region="Colorado",
                country="USA",
                population=700000,
                geom=WKTElement("POINT(-106.82 38.19)", srid=4326),
            ),
            City(
                id="city_aspen",
                city_name="Aspen",
                region="Colorado",
                country="USA",
                population=6700,
                geom=WKTElement("POINT(-106.82 38.19)", srid=4326),
            ),
            SkiResort(
                id="resort_aspen_mountain",
                resort_name="Aspen Mountain",
                region="Colorado",
                country="USA",
                summit_elevation_m=3417.0,
                geom=WKTElement("POINT(-106.82 38.19)", srid=4326),
            ),
            # A second resort at the exact same coordinates as Aspen Mountain
            # but with a different elevation, to exercise cache-key uniqueness
            # for same-type records that resolve to the same point.
            SkiResort(
                id="resort_aspen_buttermilk",
                resort_name="Buttermilk",
                region="Colorado",
                country="USA",
                summit_elevation_m=2450.0,
                geom=WKTElement("POINT(-106.82 38.19)", srid=4326),
            ),
            SkiResort(
                id="resort_denver_ski",
                resort_name="Denver Ski Area",
                region="Colorado",
                country="USA",
                summit_elevation_m=2500.0,
                geom=WKTElement("POINT(-106.5 38.5)", srid=4326),
            ),
            Station(
                id="station_aspen_co",
                station_code="KASE",
                name="Aspen Station",
                elevation_m=2380.0,
                geom=WKTElement("POINT(-106.82 38.22)", srid=4326),
            ),
        ]
    )


@pytest.fixture(scope="module")
def seed_data(migrated_db, tmp_zarr_stores):
    """Seed deterministic catalog and location rows used by the contract tests."""
    with Session(migrated_db) as session:
        noaa = ForecastCenter(
            id="center_noaa",
            center_id="noaa",
            name="National Oceanic and Atmospheric Administration",
            country="USA",
        )
        gfs = Model(
            id="model_gfs",
            model_id="gfs",
            name="Global Forecast System",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
        )
        gefs = Model(
            id="model_gefs",
            model_id="gefs",
            name="Global Ensemble Forecast System",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
        )
        version_gfs = ModelVersion(id="version_gfs_v1", model_id="gfs", version_string="v1.0")
        version_gefs = ModelVersion(id="version_gefs_v1", model_id="gefs", version_string="v1.0")
        # zarr_store_path points at local on-disk Zarr fixture stores so the
        # point-forecast integration tests run without MinIO/S3.
        run_gfs_00 = ModelRun(
            id="run_2026072100_gfs",
            model_version_id="version_gfs_v1",
            cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
            status="ready",
            zarr_store_path=tmp_zarr_stores["gfs"],
        )
        run_gfs_12 = ModelRun(
            id="run_2026072112_gfs",
            model_version_id="version_gfs_v1",
            cycle_time=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
            status="processing",
            zarr_store_path=None,
        )
        run_gefs_00 = ModelRun(
            id="run_2026072100_gefs",
            model_version_id="version_gefs_v1",
            cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
            status="ready",
            zarr_store_path=tmp_zarr_stores["gefs"],
        )
        ensemble_members = [
            EnsembleMember(
                id=f"member_{i}_gefs",
                run_id="run_2026072100_gefs",
                member_index=i,
                member_name=f"gefs_member_{i}",
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
        precip_amount_3h = ForecastVariable(
            id="var_precipitation_amount_3h",
            variable_code="precipitation_amount_3h",
            name="3-Hour Precipitation Amount",
            unit="mm",
        )
        global_grid = ForecastGrid(
            id="grid_global_025deg",
            grid_code="global_025deg",
            name="Global 0.25 Degree Grid",
            resolution_km=25.0,
        )
        downscaled_grid = ForecastGrid(
            id="grid_downscaled_3km",
            grid_code="downscaled_3km",
            name="AI Downscaled Local Grid",
            resolution_km=3.0,
        )
        session.add_all(
            [
                noaa,
                gfs,
                gefs,
                version_gfs,
                version_gefs,
                run_gfs_00,
                run_gfs_12,
                run_gefs_00,
                *ensemble_members,
                temperature,
                precipitation,
                precip_amount_3h,
                global_grid,
                downscaled_grid,
            ]
        )
        _seed_locations(session)
        _seed_verification_observations(session)
        session.commit()
    yield


@pytest.fixture(scope="module")
def client(migrated_db, seed_data):
    """TestClient bound to the test database via a ``get_db`` override.

    Module-scoped (matching ``migrated_db``/``seed_data``) so each test
    module gets its own clean migrated + seeded schema.
    """

    def override_get_db():
        db = Session(migrated_db)
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)
