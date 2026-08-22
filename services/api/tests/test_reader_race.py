"""Reader-race tests: a reader that selected READY before a downgrade must
revalidate under the SHARED gate and refuse to read the updating store.

Uses real PostgreSQL (for the advisory-lock gate + fresh Core revalidation).
Skipped when PostgreSQL is unreachable.
"""

from __future__ import annotations

import os
import sys

import pytest
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../packages/domain/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ingestion/src")))

from api.core.config import settings
from api.core.reader_gate import _ReaderGateSession

DB_URL = settings.DATABASE_URL


def _pg_reachable() -> bool:
    try:
        eng = create_engine(DB_URL, pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_reachable(), reason="PostgreSQL test instance not reachable"
)


@pytest.fixture
def catalog_engine():
    """A PostgreSQL engine with a clean ingestion catalog schema per test."""
    from ingestion.core.catalog import CatalogBase

    eng = create_engine(DB_URL, pool_pre_ping=True)
    # Drop + recreate so each test starts from an empty catalog (isolated).
    CatalogBase.metadata.drop_all(eng)
    CatalogBase.metadata.create_all(eng)
    yield eng
    CatalogBase.metadata.drop_all(eng)
    eng.dispose()


def test_reader_revalidation_observes_downgrade(catalog_engine, tmp_path) -> None:
    """A reader that selected READY before a downgrade revalidates and sees
    ``partial`` -> it refuses to read the store (FileNotFoundError)."""
    from datetime import datetime, timezone

    from ingestion.core.catalog import (
        CenterRecord,
        GridRecord,
        ModelRecord,
        ModelRunRecord,
        ModelVersionRecord,
    )
    from sqlalchemy.orm import Session

    # Insert a READY run row pointing at a local store path.
    store_path = str(tmp_path / "cycle.zarr")
    with Session(catalog_engine) as db:
        db.add(CenterRecord(id="c", center_id="noaa", name="NOAA", country="USA"))
        db.flush()
        db.add(
            ModelRecord(
                id="m", model_id="gfs", name="GFS", center_id="noaa",
                is_ensemble=False, resolution_km=25.0,
            )
        )
        db.flush()
        db.add(ModelVersionRecord(id="v", model_id="gfs", version_string="v1.0"))
        db.flush()
        db.add(
            ModelRunRecord(
                id="r", model_version_id="v",
                cycle_time=datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc),
                status="ready", zarr_store_path=store_path,
            )
        )
        db.add(GridRecord(id="g", grid_code="global_025deg", name="g", resolution_km=25.0))
        db.commit()

    # Simulate the reader's in-flight selection: it holds the store path from a
    # prior SELECT (the run was READY). Before it acquires SHARED + revalidates,
    # the writer downgrades the run to partial.
    with Session(catalog_engine) as db:
        run = db.get(ModelRunRecord, "r")
        setattr(run, "status", "partial")
        db.commit()

    # The reader now acquires SHARED + revalidates on a fresh Connection.
    from api.core.reader_gate import ReaderLockPool

    pool = ReaderLockPool(
        DB_URL, pool_size=2, max_overflow=2, pool_timeout=2.0
    )
    session = _ReaderGateSession(pool, store_path)
    try:
        session.acquire(timeout_seconds=5.0)
        ok, path = session.revalidate(DB_URL)
        assert ok is False  # run is partial -> revalidation fails
        assert path == store_path
    finally:
        session.release()
        pool.dispose()


def test_reader_revalidation_ready_succeeds(catalog_engine, tmp_path) -> None:
    """A READY run whose store is stable passes fresh Core revalidation."""
    from datetime import datetime, timezone

    from ingestion.core.catalog import (
        CenterRecord,
        GridRecord,
        ModelRecord,
        ModelRunRecord,
        ModelVersionRecord,
    )
    from sqlalchemy.orm import Session

    store_path = str(tmp_path / "ready.zarr")
    with Session(catalog_engine) as db:
        db.add(CenterRecord(id="c2", center_id="noaa", name="NOAA", country="USA"))
        db.flush()
        db.add(
            ModelRecord(
                id="m2", model_id="gfs", name="GFS", center_id="noaa",
                is_ensemble=False, resolution_km=25.0,
            )
        )
        db.flush()
        db.add(ModelVersionRecord(id="v2", model_id="gfs", version_string="v1.0"))
        db.flush()
        db.add(
            ModelRunRecord(
                id="r2", model_version_id="v2",
                cycle_time=datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc),
                status="ready", zarr_store_path=store_path,
            )
        )
        db.add(GridRecord(id="g2", grid_code="global_025deg", name="g", resolution_km=25.0))
        db.commit()

    from api.core.reader_gate import ReaderLockPool

    pool = ReaderLockPool(DB_URL, pool_size=2, max_overflow=2, pool_timeout=2.0)
    session = _ReaderGateSession(pool, store_path)
    try:
        session.acquire(timeout_seconds=5.0)
        ok, path = session.revalidate(DB_URL)
        assert ok is True
        assert path == store_path
    finally:
        session.release()
        pool.dispose()
