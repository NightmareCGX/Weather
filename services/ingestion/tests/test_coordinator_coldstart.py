"""Regression tests for fresh/cold-start cycle ingestion (self-block fix).

A first-ever ingestion can fail *after* the store is initialized (e.g. a
region-write failure) and still record a ``model_runs`` placeholder row with
status ``processing``/``partial`` and an **empty** store (no committed
regions). A retry of the same cycle then aimed at ``initialize_run_store``'s
absent-store branch, where ``guard_full_overwrite`` saw the placeholder row and
refused to cold-start initialize the store — a self-block on a fresh cycle.

These tests pin the correct lifecycle semantics:

* a non-ready placeholder row + genuinely-absent store MUST allow cold-start
  initialization (fresh-cycle recovery);
* a ``ready`` run with a missing store (external shrink / corruption) MUST
  still be refused (Case B);
* a same-cycle re-ingestion of a run that owns committed store content MUST
  keep working (Case C).

SQLite-backed; the advisory-lock coordinator is stubbed (no PG needed).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from ingestion.core.base import LiveStoreOverwriteError
from ingestion.core.catalog import (
    CenterRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
)
from ingestion.core.coordinator import RunCoordinator
from ingestion.core.zarr_writer import read_dataset

FIXTURE = str(Path(__file__).parent / "fixtures" / "gfs.t00z.pgrb2.0p25.f006.grib2")


class _NoopCoordinator:
    """No-op advisory-lock coordinator (SQLite has no PG advisory locks)."""

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


@pytest.fixture
def catalog_engine(tmp_path):
    from ingestion.core.catalog import CatalogBase

    eng = create_engine(f"sqlite:///{tmp_path / 'catalog.sqlite'}")
    CatalogBase.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _restore_store_lock_coordinator():
    """Restore the real StoreLockCoordinator after a test stubs it."""
    import ingestion.core.coordinator as CO

    return CO.StoreLockCoordinator


def _seed_dataset():
    """Parse + normalize the committed GRIB fixture (decodes to lead 6)."""
    from ingestion.core.pipeline import (
        _apply_variable_mapping,
        _normalize_canonical_units,
        _validate_requested_lead,
    )
    from ingestion.providers.noaa.parser import parse_grib2

    ds = parse_grib2(FIXTURE)
    spec = _spec()
    ds = _apply_variable_mapping(ds, spec.variables)
    ds = _normalize_canonical_units(ds, spec.variables)
    ds.attrs["model_id"] = spec.model_id
    _validate_requested_lead(ds, 6)
    return ds


def _spec(
    *,
    model: str = "gfs",
    is_ensemble: bool = False,
    leads: tuple[int, ...] = (6,),
    members: tuple[int, ...] = (),
) -> "object":
    """A run spec for cycle 2026-08-22T00Z (deterministic or ensemble)."""
    from ingestion.cli import RunSpec, _build_spec

    run_spec = RunSpec(
        model=model, cycle_date=date(2026, 8, 22), cycle_hour=0,
        target_lead_time_hours=leads, members=members, store=None, allow_custom_store=False,
    )
    args = type("A", (), {
        "center_id": "noaa", "version_string": "v1.0", "grid_id": "global_025deg",
        "variable": None, "concurrency": 1, "download_dir": "/tmp/dl",
    })()
    store = f"s3://weather-data/{model}/2026-08-22/00/cycle.zarr"
    return _build_spec(run_spec, args, store)


def _add_run(
    engine, store_path: str, *, status: str, run_id: str = "run_placeholder"
) -> None:
    """Insert a model_runs row (plus minimal FK parents) referencing store_path."""
    from sqlalchemy.orm import Session

    with Session(engine) as db:
        _upsert_center(db)
        _upsert_model(db)
        _upsert_version(db)
        db.add(
            ModelRunRecord(
                id=run_id,
                model_version_id="version_gfs_v1.0",
                cycle_time=datetime(2026, 8, 22, 0, tzinfo=timezone.utc),
                status=status,
                zarr_store_path=store_path,
            )
        )
        db.commit()


def _upsert_center(db):
    if db.query(CenterRecord).filter_by(center_id="noaa").first() is None:
        db.add(CenterRecord(id="center_noaa", center_id="noaa", name="NOAA", country="USA"))


def _upsert_model(db):
    _upsert_center(db)
    center = db.query(CenterRecord).filter_by(center_id="noaa").one()
    if db.query(ModelRecord).filter_by(model_id="gfs").first() is None:
        db.add(
            ModelRecord(
                id="model_gfs", model_id="gfs", name="GFS", center_id=center.center_id,
                is_ensemble=False, resolution_km=25.0,
            )
        )


def _upsert_version(db):
    _upsert_model(db)
    if db.query(ModelVersionRecord).filter_by(model_id="gfs").first() is None:
        db.add(
            ModelVersionRecord(id="version_gfs_v1.0", model_id="gfs", version_string="v1.0")
        )


def _initialize(catalog_engine, store, *, run_id, is_same_cycle, seed_dataset):
    """Run the coordinator's retained-seed init under a stubbed lock gate."""
    import ingestion.core.coordinator as CO

    _orig = CO.StoreLockCoordinator
    CO.StoreLockCoordinator = _NoopCoordinator
    try:
        coordinator = RunCoordinator(_spec(), store, timeout_seconds=2.0)
        conn = catalog_engine.connect()
        try:
            coordinator.initialize_run_store(
                conn,
                seed_dataset=seed_dataset,
                expected_leads=(6,),
                expected_members=(),
                run_id=run_id,
                is_same_cycle=is_same_cycle,
            )
        finally:
            conn.close()
    finally:
        CO.StoreLockCoordinator = _orig


def test_nonready_placeholder_with_absent_store_initializes(
    catalog_engine, tmp_path
) -> None:
    """Cold-start retry over a processing/partial placeholder + absent store succeeds.

    This is the regression: a first-ever ingestion that created a non-ready
    placeholder row but no committed store content must be able to (re)initialize
    the store, not self-block at ``guard_full_overwrite``.
    """
    store = str(tmp_path / "cycle.zarr")
    _add_run(catalog_engine, store, status="processing", run_id="run_placeholder")

    _initialize(
        catalog_engine, store,
        run_id="run_placeholder", is_same_cycle=True, seed_dataset=_seed_dataset(),
    )

    # The store was initialized with marker_v1 and is readable.
    from ingestion.core.markers import read_protocol_version

    assert read_protocol_version(store) == "marker_v1"
    assert read_dataset(store) is not None


def test_ready_row_with_absent_store_still_refused(
    catalog_engine, tmp_path
) -> None:
    """A ready run whose store is genuinely missing must still be refused.

    External shrink / corruption that removes a ready run's store is a
    destructive-overwrite condition: cold-start initializing over it would
    silently replace the run's contents without reconciliation. Case B is
    preserved.
    """
    store = str(tmp_path / "ready_missing.zarr")
    _add_run(catalog_engine, store, status="ready", run_id="run_ready")

    import ingestion.core.coordinator as CO

    CO.StoreLockCoordinator = _NoopCoordinator
    try:
        coordinator = RunCoordinator(_spec(), store, timeout_seconds=2.0)
        conn = catalog_engine.connect()
        try:
            with pytest.raises(LiveStoreOverwriteError):
                coordinator.initialize_run_store(
                    conn,
                    seed_dataset=_seed_dataset(),
                    expected_leads=(6,),
                    expected_members=(),
                    run_id="run_ready",
                    is_same_cycle=True,
                )
        finally:
            conn.close()
    finally:
        CO.StoreLockCoordinator = _restore_store_lock_coordinator()


def test_same_cycle_reingest_with_existing_store_still_works(
    catalog_engine, tmp_path
) -> None:
    """Same-cycle re-ingestion of a run that owns committed store content works.

    A ready run with an existing store is validated (identity) and downgraded to
    partial before the wave region writes; the guard must not fire for an
    existing-store init.
    """
    store = str(tmp_path / "existing.zarr")
    _add_run(catalog_engine, store, status="ready", run_id="run_existing")

    # Write a real store up front so the coordinator takes the existing-store
    # branch (which validates identity + downgrades, never guards).
    from ingestion.core.pipeline import (
        _apply_variable_mapping,
        _normalize_canonical_units,
        _validate_requested_lead,
    )
    from ingestion.core.zarr_writer import prepare_run_store
    from ingestion.providers.noaa.parser import parse_grib2

    ds = parse_grib2(FIXTURE)
    spec = _spec()
    ds = _apply_variable_mapping(ds, spec.variables)
    ds = _normalize_canonical_units(ds, spec.variables)
    ds.attrs["model_id"] = spec.model_id
    _validate_requested_lead(ds, 6)
    prepare_run_store(ds, store, expected_lead_time_hours=(6,))

    import ingestion.core.coordinator as CO

    CO.StoreLockCoordinator = _NoopCoordinator
    try:
        coordinator = RunCoordinator(spec, store, timeout_seconds=2.0)
        conn = catalog_engine.connect()
        try:
            coordinator.initialize_run_store(
                conn,
                seed_dataset=ds,
                expected_leads=(6,),
                expected_members=(),
                run_id="run_existing",
                is_same_cycle=True,
            )
        finally:
            conn.close()
    finally:
        CO.StoreLockCoordinator = _restore_store_lock_coordinator()

    # The store is still readable (not destroyed).
    assert read_dataset(store) is not None


# --- Ensemble / batch cold-start and re-ingestion (scenarios 2, 3, 4, 7) ---


def _ensemble_seed(member: int, lead: int = 6):
    """A normalized single-member, single-lead ensemble dataset."""
    import numpy as np
    import xarray as xr

    coords = {
        "member": [member],
        "lead_time_hours": [lead],
        "time": np.datetime64("2026-08-22T00:00:00"),
        "latitude": [38.0, 38.25, 38.5, 38.75],
        "longitude": [-107.0, -106.75, -106.5, -106.25],
    }
    dims = ("member", "lead_time_hours", "latitude", "longitude")
    shape = (1, 1, 4, 4)
    data = np.full(shape, float(lead) + float(member), dtype=np.float32)
    ds = xr.Dataset(
        data_vars={"temperature_2m": (dims, data), "precipitation_rate": (dims, data + 1.0)},
        coords=coords,
        attrs={"model_id": "gefs", "cycle_time": "2026-08-22T00:00:00"},
    )
    ds["temperature_2m"].attrs["units"] = "°C"
    ds["precipitation_rate"].attrs["units"] = "mm/h"
    return ds


def _initialize_ens(
    catalog_engine, store, *, run_id, is_same_cycle, seed_dataset,
    leads=(6,), members=(1,),
):
    """Run the ensemble coordinator's retained-seed init under a stubbed gate."""
    import ingestion.core.coordinator as CO

    _orig = CO.StoreLockCoordinator
    CO.StoreLockCoordinator = _NoopCoordinator
    try:
        coordinator = RunCoordinator(
            _spec(model="gefs", is_ensemble=True, leads=leads, members=members),
            store,
            timeout_seconds=2.0,
        )
        conn = catalog_engine.connect()
        try:
            coordinator.initialize_run_store(
                conn,
                seed_dataset=seed_dataset,
                expected_leads=leads,
                expected_members=members,
                run_id=run_id,
                is_same_cycle=is_same_cycle,
            )
        finally:
            conn.close()
    finally:
        CO.StoreLockCoordinator = _orig


def test_ensemble_fresh_coldstart_with_nonready_placeholder_initializes(
    catalog_engine, tmp_path
) -> None:
    """Fresh GEFS first ingestion with a fresh DB/store must succeed (cold-start)."""
    store = str(tmp_path / "gefs_placeholder.zarr")
    _add_run(catalog_engine, store, status="partial", run_id="run_gefs_placeholder")

    _initialize_ens(
        catalog_engine, store,
        run_id="run_gefs_placeholder", is_same_cycle=True,
        seed_dataset=_ensemble_seed(1),
        leads=(6,), members=(1, 2, 3),
    )
    from ingestion.core.markers import read_protocol_version

    assert read_protocol_version(store) == "marker_v1"
    restored = read_dataset(store)
    assert "member" in restored.coords
    assert sorted(int(v) for v in restored.coords["member"].values) == [1, 2, 3]


def test_fresh_batch_single_lead_does_not_self_block(
    catalog_engine, tmp_path
) -> None:
    """Fresh GFS with multiple leads must not self-block on a placeholder retry.

    The coordinator initializes a multi-lead store over a non-ready placeholder.
    """
    store = str(tmp_path / "gfs_multi_lead.zarr")
    _add_run(catalog_engine, store, status="processing", run_id="run_multi_lead")

    from ingestion.providers.noaa.parser import parse_grib2
    from ingestion.core.pipeline import (
        _apply_variable_mapping,
        _normalize_canonical_units,
        _validate_requested_lead,
    )

    ds = parse_grib2(FIXTURE)
    spec = _spec(leads=(0, 6, 12, 18))
    ds = _apply_variable_mapping(ds, spec.variables)
    ds = _normalize_canonical_units(ds, spec.variables)
    ds.attrs["model_id"] = spec.model_id
    _validate_requested_lead(ds, 6)

    import ingestion.core.coordinator as CO

    CO.StoreLockCoordinator = _NoopCoordinator
    try:
        coordinator = RunCoordinator(spec, store, timeout_seconds=2.0)
        conn = catalog_engine.connect()
        try:
            coordinator.initialize_run_store(
                conn,
                seed_dataset=ds,
                expected_leads=(0, 6, 12, 18),
                expected_members=(),
                run_id="run_multi_lead",
                is_same_cycle=True,
            )
        finally:
            conn.close()
    finally:
        CO.StoreLockCoordinator = _restore_store_lock_coordinator()

    restored = read_dataset(store)
    assert sorted(int(v) for v in restored.coords["lead_time_hours"].values) == [0, 6, 12, 18]


def test_gefs_member_lead_batch_coldstart_does_not_self_block(
    catalog_engine, tmp_path
) -> None:
    """Fresh GEFS member x lead batch must not self-block (scenario 4)."""
    store = str(tmp_path / "gefs_multi.zarr")
    _add_run(catalog_engine, store, status="partial", run_id="run_gefs_multi")

    _initialize_ens(
        catalog_engine, store,
        run_id="run_gefs_multi", is_same_cycle=True,
        seed_dataset=_ensemble_seed(1, lead=0),
        leads=(0, 6, 12),
        members=(1, 2, 3),
    )
    restored = read_dataset(store)
    assert sorted(int(v) for v in restored.coords["lead_time_hours"].values) == [0, 6, 12]
    assert sorted(int(v) for v in restored.coords["member"].values) == [1, 2, 3]


def test_gefs_same_cycle_reingest_existing_store_still_works(
    catalog_engine, tmp_path
) -> None:
    """GEFS same-cycle member/lead re-ingestion into an existing store succeeds."""
    store = str(tmp_path / "gefs_existing.zarr")
    _add_run(catalog_engine, store, status="ready", run_id="run_gefs_existing")

    # Prepare a real 2-member x 2-lead ensemble store, then re-ingest.
    from ingestion.core.zarr_writer import prepare_run_store

    prepare_run_store(
        _ensemble_seed(1, lead=0),
        store,
        expected_lead_time_hours=(0, 6),
        expected_members=(1, 2),
    )

    _initialize_ens(
        catalog_engine, store,
        run_id="run_gefs_existing", is_same_cycle=True,
        seed_dataset=_ensemble_seed(1, lead=0),
        leads=(0, 6), members=(1, 2),
    )
    restored = read_dataset(store)
    assert "member" in restored.coords
    assert sorted(int(v) for v in restored.coords["member"].values) == [1, 2]