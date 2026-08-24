"""Regression tests: tile rendering must NOT hold the DB connection across
the expensive Zarr materialize + PNG encode.

Root-cause context: ``render_tile_png`` previously received the request's
``db`` Session, resolved the run/store/product metadata, then opened and fully
materialized the Zarr store and encoded the PNG *while the Session (and its
QueuePool connection) was still checked out*. A browser viewport issues many
concurrent tile requests; each cold tile against a large S3-backed store held a
pool connection for seconds, so 15 simultaneous requests exhausted the default
``QueuePool`` (``pool_size=5`` + ``max_overflow=10``) and the 16th timed out
with ``sqlalchemy.exc.TimeoutError``.

The fix splits the render into a cheap catalog "metadata phase" (done while the
session is live) and a "render phase" (opening the store via the reader gate's
dedicated lock pool after the request's session/connection is closed). These
tests lock in that the DB connection is released *before* the store read, and
that a tight-pool concurrent burst no longer exhausts the pool.

The module is self-contained (seeds its own catalog + TestClient) so it is
independent of the shared conftest DB-shape fixtures. Skipped when PostgreSQL
is unreachable, following the existing convention.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# Ensure services/api/src is on sys.path (mirrors conftest).
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import pytest
import xarray as xr
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.main import app
from api.models.entities import (
    ForecastCenter,
    ForecastGrid,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)

#: The DB URL used for the module fixtures.
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://weather_user:weather_password@localhost:5432/weather_db",
)


def _build_tiny_dataset() -> xr.Dataset:
    """A deterministic tiny 4x4 grid dataset with temperature + lead coords."""
    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    lead = np.array([0, 6, 12, 18])
    lead_grid, lat_grid, lon_grid = np.meshgrid(lead, lat, lon, indexing="ij")
    temperature = (
        10.0
        + 10.0 * (lat_grid - 38.0)
        + 10.0 * (lon_grid + 107.0)
        + 0.5 * lead_grid
    )
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                temperature,
            )
        },
        coords={
            "lead_time_hours": lead,
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture(scope="module")
def tile_client(tmp_path_factory):
    """A migrated schema + seeded minimal run/store + TestClient bound to it."""
    engine = create_engine(DB_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable; skipping.")
        return None

    api_dir = os.path.abspath(os.path.join(current_dir, ".."))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", DB_URL)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
    command.upgrade(alembic_cfg, "head")

    store = str(tmp_path_factory.mktemp("tile_lifetime") / "cycle.zarr")
    _build_tiny_dataset().to_zarr(store, mode="w")

    with Session(engine) as session:
        session.add(ForecastCenter(id="center_noaa", center_id="noaa", name="NOAA", country="USA"))
        session.add(
            Model(
                id="model_gfs",
                model_id="gfs",
                name="Global Forecast System",
                center_id="noaa",
                is_ensemble=False,
                resolution_km=25.0,
            )
        )
        session.add(ModelVersion(id="version_gfs_v1", model_id="gfs", version_string="v1.0"))
        session.add(
            ModelRun(
                id="run_tile",
                model_version_id="version_gfs_v1",
                cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
                status="ready",
                zarr_store_path=store,
            )
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
        for lead in (0, 6, 12, 18):
            session.add(
                ForecastProduct(
                    id=f"product_tile_temperature_2m_{lead}",
                    run_id="run_tile",
                    variable_id="temperature_2m",
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=lead,
                )
            )
        session.commit()

    def override_get_db():
        db = Session(engine)
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)
        command.downgrade(alembic_cfg, "base")
        engine.dispose()


def _tile_img_url() -> str:
    return "/v1/maps/gfs/temperature_2m/surface/8/51/98.png?lead_time_hours=6"


def test_tile_connection_released_before_store_read(tile_client, monkeypatch):
    """The request's DB connection is returned before gated_read_dataset runs.

    ``render_tile_png`` is invoked directly with a Session bound to a dedicated
    engine, and ``gated_read_dataset`` is replaced with a probe that records
    ``engine.pool.checkedout()`` at the instant the store read begins.

    * Before the fix the Session (and its checked-out connection) was carried
      into the store read, so the pool reported 1 checked out.
    * After the fix the metadata phase closes the session first, so the pool
      reports 0 checked out when the store read begins.
    """
    from api.services.tiles import _tile_cache, render_tile_png

    eng = create_engine(DB_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)
    observed: dict[str, object] = {}

    def fake_gated(store_path: str) -> xr.Dataset:
        observed["checked_out"] = eng.pool.checkedout()
        observed["store_path"] = store_path
        return _build_tiny_dataset()

    monkeypatch.setattr("api.core.reader_gate.gated_read_dataset", fake_gated)

    _tile_cache.clear()
    db = Session(eng)
    try:
        png = render_tile_png(
            db,
            model="gfs",
            variable="temperature_2m",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            lead_time_hours=6,
            initial_time="2026-07-21T00:00:00Z",
        )
    finally:
        db.close()
    _tile_cache.clear()
    eng.dispose()

    assert png.startswith(b"\x89PNG"), "expected a real PNG (fix broke rendering)"
    assert observed["checked_out"] == 0, (
        f"DB connection was still checked out when the store read began: "
        f"{observed['checked_out']} (>=1 means the session/connection is held "
        f"across the expensive render -> QueuePool exhaustion under tile concurrency)"
    )


def test_concurrent_tiles_no_timeout_with_tight_pool(tile_client, monkeypatch):
    """A tight pool + slow reads must not raise QueuePool TimeoutError.

    Runs many concurrent ``render_tile_png`` calls against an engine whose
    QueuePool is deliberately small (``pool_size=2``, ``max_overflow=2``,
    ``pool_timeout=0.5``) with every store read slowed to ~600ms.

    On the original code each request held its connection across the whole
    read+render, so 8 concurrent requests against a 4-connection pool hold all
    four at once and the 6th+ request's checkout exceeds the 0.5s timeout ->
    ``TimeoutError``. After the fix each request releases its connection before
    the read, so the pool is only ever contended for the milliseconds of the
    metadata queries and every request completes.
    """
    from api.services.tiles import _tile_cache, render_tile_png

    tight_engine = create_engine(DB_URL, pool_size=2, max_overflow=2, pool_timeout=0.5, pool_pre_ping=True)

    def fake_slow_gated(store_path: str) -> xr.Dataset:
        time.sleep(0.6)  # simulate a slow cold Zarr materialize
        return _build_tiny_dataset()

    monkeypatch.setattr("api.core.reader_gate.gated_read_dataset", fake_slow_gated)

    _tile_cache.clear()
    errors: list[BaseException] = []
    lock = threading.Lock()

    def render_one(i: int) -> None:
        db = Session(tight_engine)
        try:
            png = render_tile_png(
                db,
                model="gfs",
                variable="temperature_2m",
                level="surface",
                zoom=4,
                x=1,
                y=2,
                lead_time_hours=6,
                initial_time="2026-07-21T00:00:00Z",
            )
            assert png.startswith(b"\x89PNG")
        except BaseException as exc:  # noqa: BLE001 - capture for assertion
            with lock:
                errors.append(exc)
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(render_one, range(8)))

    _tile_cache.clear()
    tight_engine.dispose()
    assert errors == [], f"concurrent pool exhaustion under tight pool: {errors!r}"