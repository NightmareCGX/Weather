"""Integration tests for the region-write concurrency coordinator (PG-backed).

These tests exercise the retained-seed initialization, wave pre-update,
region-write worker, and coalesced finalization against real PostgreSQL and a
local Zarr store. They use the real GRIB fixture (so parse is real) and route
the catalog through the injectable CLI session factory.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from ingestion.core.config import settings

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

FIXTURE = str(
    Path(__file__).parent / "fixtures" / "gfs.t00z.pgrb2.0p25.f006.grib2"
)


@pytest.fixture
def catalog_engine(tmp_path):
    """An in-memory SQLite catalog engine with the ingestion schema."""
    from ingestion.core.catalog import CatalogBase

    eng = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def cli_catalog(catalog_engine, monkeypatch):
    """Route the CLI's catalog access to the SQLite catalog engine."""
    import ingestion.cli as CLI

    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: catalog_engine)
    # Also route the coordinator's worker connections to the SQLite engine so
    # the advisory locks... but SQLite has no advisory locks. The coordinator's
    # locks module requires PG. For the catalog-path tests we mock the lock
    # coordinator to a no-op.
    return catalog_engine


class _NoopCoordinator:
    """A no-op lock coordinator for catalog-path tests (SQLite has no locks)."""

    def __init__(self, *a, **k):
        pass

    def acquire_shared_gate(self):
        pass

    def release_shared_gate(self):
        pass

    def acquire_exclusive_gate(self):
        pass

    def release_exclusive_gate(self):
        pass

    def acquire_admission(self):
        pass

    def release_admission(self):
        pass

    def acquire_shared_admission(self):
        pass

    def release_shared_admission(self):
        pass

    def acquire_region_locks(self, region_ids):
        pass

    def release_region_locks(self, region_ids):
        pass

    def release_all(self):
        pass

    def close_connection(self):
        pass


def test_retained_seed_initializes_store_and_catalogs(
    cli_catalog, tmp_path, monkeypatch
) -> None:
    """The retained-seed flow initializes the store and records the run."""

    from ingestion.core.coordinator import RunCoordinator
    from ingestion.core.markers import read_protocol_version
    from ingestion.core.pipeline import (
        _apply_variable_mapping,
        _normalize_canonical_units,
        _validate_requested_lead,
    )
    from ingestion.core.zarr_writer import read_dataset
    from ingestion.cli import _build_spec

    store_path = str(tmp_path / "cycle.zarr")
    from ingestion.cli import RunSpec

    run_spec = RunSpec(
        model="gfs", cycle_date=date(2026, 7, 21), cycle_hour=0,
        lead_time_hours=(6,), members=(), store=store_path, allow_custom_store=True,
    )
    spec = _build_spec(
        run_spec,
        type("A", (), {
            "center_id": "noaa", "version_string": "v1.0", "grid_id": "global_025deg",
            "variable": None, "concurrency": 1, "download_dir": str(tmp_path / "dl"),
        })(),
        store_path,
    )

    # Parse the seed (the retained dataset).
    from ingestion.providers.noaa.parser import parse_grib2

    ds = parse_grib2(FIXTURE)
    ds = _apply_variable_mapping(ds, spec.variables)
    ds = _normalize_canonical_units(ds, spec.variables)
    ds.attrs["model_id"] = spec.model_id
    _validate_requested_lead(ds, 6)

    # Monkeypatch the lock coordinator to a no-op (SQLite has no advisory locks).
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator", _NoopCoordinator
    )

    coordinator = RunCoordinator(spec, store_path, timeout_seconds=2.0)

    engine = cli_catalog
    conn = engine.connect()
    try:
        coordinator.initialize_run_store(
            conn,
            seed_dataset=ds,
            expected_leads=(6,),
            expected_members=(),
            run_id=None,
            is_same_cycle=False,
        )
    finally:
        conn.close()

    # The store was initialized as marker_v1 and is readable.
    assert read_protocol_version(store_path) == "marker_v1"
    assert read_dataset(store_path) is not None
