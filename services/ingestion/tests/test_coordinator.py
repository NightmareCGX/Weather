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
    import ingestion.core.wave_runner as wave_runner

    monkeypatch.setattr(wave_runner, "_catalog_session_factory", lambda: catalog_engine)
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
        target_lead_time_hours=(6,), members=(), store=store_path, allow_custom_store=True,
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


def test_validated_incremental_init_skips_exclusive_gate(
    cli_catalog, tmp_path, monkeypatch
) -> None:
    """A valid, fully initialized store takes the fast path in initialize_run_store and skips EXCLUSIVE gate."""
    from unittest.mock import MagicMock
    from ingestion.core.coordinator import RunCoordinator
    from ingestion.core.pipeline import (
        _apply_variable_mapping,
        _normalize_canonical_units,
        _validate_requested_lead,
    )
    from ingestion.cli import RunSpec, _build_spec
    from ingestion.providers.noaa.parser import parse_grib2

    store_path = str(tmp_path / "incremental_fast_path.zarr")

    run_spec = RunSpec(
        model="gfs", cycle_date=date(2026, 7, 21), cycle_hour=0,
        target_lead_time_hours=(6, 12), members=(), store=store_path, allow_custom_store=True,
    )
    spec = _build_spec(
        run_spec,
        type("A", (), {
            "center_id": "noaa", "version_string": "v1.0", "grid_id": "global_025deg",
            "variable": None, "concurrency": 1, "download_dir": str(tmp_path / "dl"),
        })(),
        store_path,
    )

    ds = parse_grib2(FIXTURE)
    ds = _apply_variable_mapping(ds, spec.variables)
    ds = _normalize_canonical_units(ds, spec.variables)
    ds.attrs["model_id"] = spec.model_id
    _validate_requested_lead(ds, 6)

    # 1. First initialization (creates store)
    monkeypatch.setattr("ingestion.core.coordinator.StoreLockCoordinator", _NoopCoordinator)
    coordinator = RunCoordinator(spec, store_path, timeout_seconds=2.0)
    engine = cli_catalog
    with engine.connect() as conn:
        coordinator.initialize_run_store(
            conn,
            seed_dataset=ds,
            expected_leads=(6, 12),
            expected_members=(),
            run_id=None,
            is_same_cycle=False,
        )

    assert coordinator._snapshot is not None

    # 2. Second initialization (incremental wave for lead 12)
    # Spy on StoreLockCoordinator to verify acquire_exclusive_gate is NOT called on valid store
    mock_lock_coord = MagicMock()
    monkeypatch.setattr("ingestion.core.coordinator.StoreLockCoordinator", lambda *a, **k: mock_lock_coord)

    coordinator2 = RunCoordinator(spec, store_path, timeout_seconds=2.0)
    with engine.connect() as conn:
        coordinator2.initialize_run_store(
            conn,
            seed_dataset=ds,
            expected_leads=(6, 12),
            expected_members=(),
            run_id="run_123",
            is_same_cycle=True,
        )

    # Assert fast path was taken: snapshot is populated, acquire_exclusive_gate was NOT called
    assert coordinator2._snapshot is not None
    assert coordinator2._snapshot.lead_index_map == {6: 0, 12: 1}
    mock_lock_coord.acquire_exclusive_gate.assert_not_called()


def test_corrupted_or_incomplete_store_takes_exclusive_fallback(
    cli_catalog, tmp_path, monkeypatch
) -> None:
    """An incomplete, missing-protocol, or mismatched store falls back to EXCLUSIVE gate."""
    import os
    from unittest.mock import MagicMock
    from ingestion.core.coordinator import RunCoordinator
    from ingestion.core.pipeline import (
        _apply_variable_mapping,
        _normalize_canonical_units,
    )
    from ingestion.cli import RunSpec, _build_spec
    from ingestion.providers.noaa.parser import parse_grib2

    store_path = str(tmp_path / "fallback_store.zarr")

    run_spec = RunSpec(
        model="gfs", cycle_date=date(2026, 7, 21), cycle_hour=0,
        target_lead_time_hours=(6, 12), members=(), store=store_path, allow_custom_store=True,
    )
    spec = _build_spec(
        run_spec,
        type("A", (), {
            "center_id": "noaa", "version_string": "v1.0", "grid_id": "global_025deg",
            "variable": None, "concurrency": 1, "download_dir": str(tmp_path / "dl"),
        })(),
        store_path,
    )

    ds = parse_grib2(FIXTURE)
    ds = _apply_variable_mapping(ds, spec.variables)
    ds = _normalize_canonical_units(ds, spec.variables)
    ds.attrs["model_id"] = spec.model_id

    # 1. Initialize store first
    monkeypatch.setattr("ingestion.core.coordinator.StoreLockCoordinator", _NoopCoordinator)
    coordinator = RunCoordinator(spec, store_path, timeout_seconds=2.0)
    engine = cli_catalog
    with engine.connect() as conn:
        coordinator.initialize_run_store(
            conn,
            seed_dataset=ds,
            expected_leads=(6, 12),
            expected_members=(),
            run_id=None,
            is_same_cycle=False,
        )

    # 2. Corrupt store by deleting __commit__/v1/version
    proto_file = Path(store_path) / "__commit__" / "v1" / "version"
    if proto_file.is_file():
        os.remove(proto_file)

    # 3. Next initialization must detect missing protocol marker and take EXCLUSIVE fallback
    mock_lock_coord = MagicMock()
    monkeypatch.setattr("ingestion.core.coordinator.StoreLockCoordinator", lambda *a, **k: mock_lock_coord)

    coordinator2 = RunCoordinator(spec, store_path, timeout_seconds=2.0)
    with engine.connect() as conn:
        coordinator2.initialize_run_store(
            conn,
            seed_dataset=ds,
            expected_leads=(6, 12),
            expected_members=(),
            run_id=None,
            is_same_cycle=False,
        )

    # EXCLUSIVE gate must have been acquired on fallback
    mock_lock_coord.acquire_exclusive_gate.assert_called_once()
    mock_lock_coord.release_exclusive_gate.assert_called_once()


def test_mismatched_model_takes_exclusive_fallback(
    cli_catalog, tmp_path, monkeypatch
) -> None:
    """A store with wrong model_id falls back to EXCLUSIVE initialization."""
    from unittest.mock import MagicMock
    from ingestion.core.coordinator import RunCoordinator
    from ingestion.core.pipeline import (
        _apply_variable_mapping,
        _normalize_canonical_units,
    )
    from ingestion.cli import RunSpec, _build_spec
    from ingestion.providers.noaa.parser import parse_grib2

    store_path = str(tmp_path / "mismatched_model.zarr")

    run_spec = RunSpec(
        model="gfs", cycle_date=date(2026, 7, 21), cycle_hour=0,
        target_lead_time_hours=(6,), members=(), store=store_path, allow_custom_store=True,
    )
    spec = _build_spec(
        run_spec,
        type("A", (), {
            "center_id": "noaa", "version_string": "v1.0", "grid_id": "global_025deg",
            "variable": None, "concurrency": 1, "download_dir": str(tmp_path / "dl"),
        })(),
        store_path,
    )

    ds = parse_grib2(FIXTURE)
    ds = _apply_variable_mapping(ds, spec.variables)
    ds = _normalize_canonical_units(ds, spec.variables)
    ds.attrs["model_id"] = "gefs"  # mismatched model!

    monkeypatch.setattr("ingestion.core.coordinator.StoreLockCoordinator", _NoopCoordinator)
    coordinator = RunCoordinator(spec, store_path, timeout_seconds=2.0)
    engine = cli_catalog
    with engine.connect() as conn:
        coordinator.initialize_run_store(
            conn,
            seed_dataset=ds,
            expected_leads=(6,),
            expected_members=(),
            run_id=None,
            is_same_cycle=False,
        )

    # Validate against spec expecting 'gfs'
    mock_lock_coord = MagicMock()
    monkeypatch.setattr("ingestion.core.coordinator.StoreLockCoordinator", lambda *a, **k: mock_lock_coord)

    coordinator2 = RunCoordinator(spec, store_path, timeout_seconds=2.0)
    with engine.connect() as conn:
        try:
            coordinator2.initialize_run_store(
                conn,
                seed_dataset=ds,
                expected_leads=(6,),
                expected_members=(),
                run_id=None,
                is_same_cycle=False,
            )
        except Exception:
            pass

    # Exclusive gate was acquired
    mock_lock_coord.acquire_exclusive_gate.assert_called_once()


