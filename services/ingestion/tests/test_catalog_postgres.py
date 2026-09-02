"""Integration test: the ingestion catalog writer populates PostgreSQL, and the
API serving tier discovers the freshly ingested run.

Verifies the end-to-end contract (write Zarr -> record catalog -> serve
/v1/points) against real PostgreSQL and the migrated schema. Skipped when
PostgreSQL is unreachable, following the API suite convention.
"""

import os
import sys
from datetime import datetime, timezone

# Ensure services/api/src and services/ingestion/src are on sys.path.
_here = os.path.dirname(__file__)
_api_src = os.path.abspath(os.path.join(_here, "../../api/src"))
_ing_src = os.path.abspath(os.path.join(_here, "../src"))
for _p in (_api_src, _ing_src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest
import xarray as xr
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from alembic import command
from api.core.database import get_db
from api.main import app
from api.models.entities import ForecastVariable, ModelRun
from fastapi.testclient import TestClient
from ingestion.core.catalog import RunCatalogSpec, VariableSpec, record_run
from ingestion.core.zarr_writer import write_dataset

CYCLE = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
#: A point inside the fixture grid (38.0..38.75 lat, -107.0..-106.25 lon).
#: Deliberately distinct from the API suite's point-forecast test point
#: (38.125, -106.875) so the two suites never share a Redis cache key.
LAT = 38.375
LON = -106.625

GRIB_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                            "gfs.t00z.pgrb2.0p25.f006.grib2")


@pytest.fixture(autouse=True)
def _flush_redis():
    """Clear Redis before every test in this module.

    The point-forecast endpoints cache responses in Redis keyed by a SHA-256
    digest of the normalized request. Several tests here ingest a ``gfs`` run
    and query the same ``(lat, lon)`` point, so they share the exact same cache
    key; without clearing Redis, an earlier test's cached response can satisfy
    a later test without the later test's own ingestion path being exercised.
    Flushing Redis per test guarantees each test starts from a clean cache and
    genuinely executes its own pipeline (see the Milestone 1-11 remediation of
    the CLI production-entrypoint integration test).
    """
    import redis as redis_lib

    from api.core.config import settings

    client = redis_lib.from_url(settings.REDIS_URL)
    client.flushall()
    client.close()
    yield


def _build_dataset() -> xr.Dataset:
    """A deterministic dataset matching the API fixture grid geometry.

    Longitude is stored in the WGS84 [-180, 180] convention here (the API's
    ``RegularGrid`` also aligns 0-360 stores); the grid spans 38.0..38.75 lat
    and -107.0..-106.25 lon with lead times 0/6/12/18.
    """
    lat = np.array([38.0, 38.25, 38.5, 38.75], dtype=float)
    lon = np.array([-107.0, -106.75, -106.5, -106.25], dtype=float)
    lead = np.array([0, 6, 12, 18], dtype=float)
    lead_grid, lat_grid, lon_grid = np.meshgrid(lead, lat, lon, indexing="ij")
    temperature = (
        10.0 + 10.0 * (lat_grid - 38.0) + 10.0 * (lon_grid + 107.0) + 0.5 * lead_grid
    )
    precipitation = 0.5 * lead_grid
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                temperature,
            ),
            "precipitation_rate": (
                ("lead_time_hours", "latitude", "longitude"),
                precipitation,
            ),
        },
        coords={
            "lead_time_hours": lead,
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture(scope="module")
def catalog_db(tmp_path_factory):
    """A migrated PostGIS schema and a TestClient bound to it."""
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
            "PostgreSQL test instance not running or reachable; skipping catalog integration test."
        )

    api_dir = os.path.abspath(os.path.join(_api_src, ".."))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))

    command.upgrade(alembic_cfg, "head")

    def override_get_db():
        db = Session(engine)
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield {"engine": engine, "client": test_client}
    finally:
        app.dependency_overrides.pop(get_db, None)
        command.downgrade(alembic_cfg, "base")
        engine.dispose()


def _ingest_gfs_run(engine, store_dir: str, *, model_id: str = "gfs") -> str:
    """Deterministic ingestion: write the dataset to a local Zarr store, then
    record it in the PostgreSQL catalog as a ready run.

    ``model_id`` gives each integration test a distinct model identity so the
    three tests never share a run row or compete as cross-cycle candidates (a
    shared ``(model_version_id, cycle_time)`` run would be upserted and its
    ``zarr_store_path`` replaced by whichever test ran last, causing the
    cross-cycle lead-coordinate mismatch this module validates against).
    """
    store_path = store_dir
    dataset = _build_dataset()
    write_dataset(dataset, store_path)

    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="National Oceanic and Atmospheric Administration",
        center_country="USA",
        model_id=model_id,
        model_name="Global Forecast System",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=CYCLE,
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree Grid",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=store_path,
        variables=(
            VariableSpec("temperature_2m", "2-Meter Temperature", "°C"),
            VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h"),
        ),
    )
    with Session(engine) as session:
        record_run(session, spec, dataset)
    return store_path


def test_ingested_run_is_discoverable_and_served(catalog_db, tmp_path_factory) -> None:
    engine = catalog_db["engine"]
    client = catalog_db["client"]

    store_dir = str(tmp_path_factory.mktemp("catalog_it_gfs"))
    store_path = _ingest_gfs_run(engine, store_dir, model_id="gfs_disc")

    # The run is recorded as ready with the store path.
    with Session(engine) as session:
        run = (
            session.query(ModelRun)
            .join(ModelRun.model_version)
            .where(ModelRun.zarr_store_path == store_path)
            .one()
        )
        assert run.status == "ready"

    # /v1/runs lists the freshly ingested run.
    resp = client.get("/v1/runs")
    assert resp.status_code == 200
    runs = resp.json()["data"]
    assert any(r["id"] == run.id for r in runs)
    assert any(r["status"] == "ready" for r in runs)

    # /v1/points serves it (the API _resolve_run shape: ready + store path).
    resp = client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gfs_disc")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["model"] == "gfs_disc"
    assert data["generated_at"] == "2026-07-22T00:00:00Z"
    assert any(entry["lead_time_hours"] == 6 for entry in data["forecasts"])


def test_ingested_run_catalog_tables(catalog_db, tmp_path_factory) -> None:
    engine = catalog_db["engine"]

    _ingest_gfs_run(
        engine, str(tmp_path_factory.mktemp("catalog_it_gfs2")), model_id="gfs_catalog"
    )

    with Session(engine) as session:
        # Variables and grid are recorded by the writer (not test fixtures).
        var_codes = {
            code for (code,) in session.query(ForecastVariable.variable_code).all()
        }
        assert {"temperature_2m", "precipitation_rate"} <= var_codes


def test_cli_production_entrypoint_ingests_and_serves(
    catalog_db, tmp_path_factory, monkeypatch
) -> None:
    """The real production CLI entrypoint ingests and the API serves the run.

    This exercises the actual deployable call path: ``ingestion.cli:main``
    downloads (mocked at the connector boundary), parses the committed GRIB2
    fixture, writes Zarr, records the run in PostgreSQL as ``partial``
    (Phase 5B: lead 6 is one wave target of the canonical 81-lead horizon;
    partial runs remain serving-eligible), and the API discovers/serves it
    through ``/v1/points``.
    """
    engine = catalog_db["engine"]
    client = catalog_db["client"]

    store = str(tmp_path_factory.mktemp("cli_it") / "gfs.zarr")
    download_dir = str(tmp_path_factory.mktemp("cli_it_dl"))
    captured_variables: list[tuple[str, ...] | None] = []

    # Mock only the network download; run everything else for real.
    async def _fake_download(
        self,
        model,
        cycle_date,
        cycle_hour,
        lead_time_hours,
        destination,
        member=None,
        variables=None,
    ):
        captured_variables.append(variables)
        from pathlib import Path

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copyfile(GRIB_FIXTURE, destination)
        return destination

    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download",
        _fake_download,
    )

    from ingestion.cli import DEFAULT_VARIABLES, main

    code = main(
        [
            "ingest",
            "--model", "gfs",
            "--cycle-date", "2026-07-22",
            "--cycle-hour", "0",
            "--lead-time-hours", "6",
            "--store", store,
            # The test uses a local tmp_path store (no object storage in the
            # test harness), which is a non-derived path. The approved store-path
            # validation requires an explicit opt-in for such paths; this test
            # opts in while still exercising the real production CLI entrypoint.
            "--allow-custom-store",
            "--download-dir", download_dir,
        ]
    )
    assert code == 0
    assert captured_variables == [tuple(v.code for v in DEFAULT_VARIABLES)]

    # Phase 5B: a single-lead target of the canonical 81-lead horizon is
    # committed and served, but the run stays partial until the whole horizon
    # is committed.
    with Session(engine) as session:
        run = (
            session.query(ModelRun)
            .join(ModelRun.model_version)
            .where(ModelRun.zarr_store_path == store)
            .one()
        )
        assert run.status == "partial"
        # The run id is version-scoped (approved remediation): model gfs,
        # version v1.0, cycle 2026-07-22T00Z. The CLI only ingests gfs/gefs, so
        # this test's run uses model gfs. The other two tests use distinct model
        # ids (gfs_disc/gfs_catalog), so this CLI run never shares a run row
        # with them and never competes as a cross-cycle candidate.
        assert run.id == "run_version_gfs_v1.0_202607220000_gfs"

    # /v1/runs lists it.
    resp = client.get("/v1/runs")
    assert resp.status_code == 200
    assert any(r["id"] == "run_version_gfs_v1.0_202607220000_gfs" for r in resp.json()["data"])

    # /v1/points serves it. Use the distinct point (LAT, LON) that is inside
    # the GRIB fixture grid (lat 44..0, lon -120..30) but deliberately differs
    # from the API suite's point-forecast point (38.125, -106.875) so the two
    # suites never share a Redis cache key. The CLI-parsed dataset renames the
    # raw GRIB2 't2m' variable to temperature_2m.
    resp = client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gfs")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["model"] == "gfs"
    assert data["generated_at"] == "2026-07-22T00:00:00Z"
    entry = next(
        e for e in data["forecasts"] if e["lead_time_hours"] == 6
    )
    assert "temperature_2m" in entry
