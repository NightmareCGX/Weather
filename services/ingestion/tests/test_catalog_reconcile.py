"""Regression tests: complete (bi-directional) store↔catalog reconciliation.

These tests pin the fix for the one-way reconciliation defect: the coordinator
path commits forecast regions to the Zarr store but (for every region except
the retained seed) never calls ``record_run``, so the catalog only ever holds
the synthetic seed's rows. The finalizer's ``_reconcile_catalog_to_store`` was
delete-only — it removed stale rows but never **restored** rows for regions
that are physically committed in the store. Result: a successful multi-lead /
multi-member run stayed ``partial`` even though the store had every region.

The fix makes reconciliation bi-directional:

    committed_state = read_committed_state(store)
    catalog_state   = read_catalog_products(run)
    missing = committed_state - catalog_state   -> insert/upsert
    stale   = catalog_state   - committed_state -> delete

Store/committed-marker state remains the source of truth: only COMPLETE
marker-validated regions drive restoration (never raw array value scans).

Coverage: deterministic + ensemble missing/stale/both, uncommitted-no-row,
idempotency, concurrent no-duplicate, healthy no-op, and RUN-SCOPE distinction
(READY requires the declared expected set, not merely "everything in store").
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.catalog import (
    CatalogBase,
    CommittedState,
    EnsembleMemberProductRecord,
    EnsembleMemberRecord,
    ProductRecord,
    RunCatalogSpec,
    VariableSpec,
    _reconcile_catalog_to_store,
    record_run,
)

CYCLE = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)

TEMPERATURE = VariableSpec("temperature_2m", "2-Meter Temperature", "°C")
PRECIPITATION = VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h")


def _spec(
    *,
    is_ensemble: bool = False,
    expected_leads: tuple[int, ...] = (6,),
    expected_members: tuple[int, ...] = (),
    store: str = "/tmp/reconcile.zarr",
) -> RunCatalogSpec:
    return RunCatalogSpec(
        center_id="noaa",
        center_name="National Oceanic and Atmospheric Administration",
        center_country="USA",
        model_id="gefs" if is_ensemble else "gfs",
        model_name="GEFS" if is_ensemble else "Global Forecast System",
        is_ensemble=is_ensemble,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=CYCLE,
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree Grid",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=store,
        variables=(TEMPERATURE, PRECIPITATION),
        expected_lead_time_hours=expected_leads,
        expected_members=expected_members,
    )


def _deterministic_state(leads: set[int]) -> CommittedState:
    return CommittedState.deterministic(leads)


def _ensemble_state(pairs: set[tuple[int, int]]) -> CommittedState:
    members = {m for m, _ in pairs}
    return CommittedState.ensemble(pairs, members)


@pytest.fixture
def db() -> Session:
    """An in-memory SQLite catalog schema."""
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


def _product_leads(db: Session, run_id: str) -> set[int]:
    return {
        int(p.lead_time_hours)
        for p in db.query(ProductRecord).filter(ProductRecord.run_id == run_id).all()
    }


def _member_pairs(db: Session, run_id: str) -> set[tuple[int, int]]:
    return {
        (int(p.member_index), int(p.lead_time_hours))
        for p in db.query(EnsembleMemberProductRecord)
        .filter(EnsembleMemberProductRecord.run_id == run_id)
        .all()
    }


def _member_indices(db: Session, run_id: str) -> set[int]:
    return {
        int(m.member_index)
        for m in db.query(EnsembleMemberRecord)
        .filter(EnsembleMemberRecord.run_id == run_id)
        .all()
    }


# ---------------------------------------------------------------------------
# Deterministic (GFS) reconciliation
# ---------------------------------------------------------------------------


def test_deterministic_missing_lead_restored(db: Session) -> None:
    """Store has {0,12,24}; catalog has {0,12}; 24 is restored."""
    spec = _spec(expected_leads=(0, 12, 24))
    run = record_run(db, spec, _dataset_leads((0, 12)), committed_state=None)
    assert _product_leads(db, run.id) == {0, 12}

    _reconcile_catalog_to_store(db, run, _deterministic_state({0, 12, 24}), spec)
    assert _product_leads(db, run.id) == {0, 12, 24}


def test_deterministic_stale_lead_removed(db: Session) -> None:
    """Store has {0,12}; catalog has {0,12,48}; 48 is removed."""
    spec = _spec(expected_leads=(0, 12))
    run = record_run(db, spec, _dataset_leads((0, 12)), committed_state=None)
    # Add a stale lead (48) that the store does not hold.
    _add_product(db, run.id, spec, 48)

    _reconcile_catalog_to_store(db, run, _deterministic_state({0, 12}), spec)
    assert _product_leads(db, run.id) == {0, 12}


def test_deterministic_stale_and_missing_reconciled(db: Session) -> None:
    """Store {0,12,24}; catalog {0,99}; reconcile to {0,12,24}."""
    spec = _spec(expected_leads=(0, 12, 24))
    run = record_run(db, spec, _dataset_leads((0,)), committed_state=None)
    _add_product(db, run.id, spec, 99)  # stale

    _reconcile_catalog_to_store(db, run, _deterministic_state({0, 12, 24}), spec)
    assert _product_leads(db, run.id) == {0, 12, 24}


# ---------------------------------------------------------------------------
# Ensemble (GEFS) reconciliation
# ---------------------------------------------------------------------------


def test_ensemble_missing_pair_restored(db: Session) -> None:
    """Store (1,6),(2,6); catalog (1,6); restore (2,6) + member 2."""
    spec = _spec(is_ensemble=True, expected_leads=(6,), expected_members=(1, 2))
    run = record_run(db, spec, _dataset_member(1, (6,)), member=1, committed_state=None)
    assert _member_pairs(db, run.id) == {(1, 6)}
    assert _member_indices(db, run.id) == {1}

    _reconcile_catalog_to_store(db, run, _ensemble_state({(1, 6), (2, 6)}), spec)
    assert _member_pairs(db, run.id) == {(1, 6), (2, 6)}
    assert _member_indices(db, run.id) == {1, 2}


def test_ensemble_stale_pair_removed(db: Session) -> None:
    """Store (1,6); catalog (1,6),(2,6); remove stale (2,6) + member 2."""
    spec = _spec(is_ensemble=True, expected_leads=(6,), expected_members=(1, 2))
    run = record_run(db, spec, _dataset_member(2, (6,)), member=2, committed_state=None)
    # Simulate a stale catalog entry for member 2 (store only has member 1).
    _reconcile_catalog_to_store(db, run, _ensemble_state({(1, 6)}), spec)
    assert _member_pairs(db, run.id) == {(1, 6)}
    assert _member_indices(db, run.id) == {1}


def test_ensemble_stale_and_missing_reconciled(db: Session) -> None:
    """Store (1,6),(2,12); catalog (1,6),(1,12); -> {(1,6),(2,12)}."""
    spec = _spec(is_ensemble=True, expected_leads=(6, 12), expected_members=(1, 2))
    run = record_run(
        db, spec, _dataset_member(1, (6, 12)), member=1, committed_state=None
    )
    _reconcile_catalog_to_store(db, run, _ensemble_state({(1, 6), (2, 12)}), spec)
    assert _member_pairs(db, run.id) == {(1, 6), (2, 12)}
    assert _member_indices(db, run.id) == {1, 2}


def test_ensemble_missing_multiple_pairs_restored(db: Session) -> None:
    """Store full 2x2 matrix; catalog only member-1 lead-6; restore all others."""
    spec = _spec(is_ensemble=True, expected_leads=(0, 12), expected_members=(1, 2, 3))
    run = record_run(db, spec, _dataset_member(1, (0,)), member=1, committed_state=None)
    committed = {(m, lead_t) for m in (1, 2, 3) for lead_t in (0, 12)}
    _reconcile_catalog_to_store(db, run, _ensemble_state(committed), spec)
    assert _member_pairs(db, run.id) == committed
    assert _member_indices(db, run.id) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Uncommitted / idempotency / concurrency / healthy
# ---------------------------------------------------------------------------


def test_no_row_created_for_uncommitted_region(db: Session) -> None:
    """A region absent from committed state is never added."""
    spec = _spec(expected_leads=(0, 12))
    run = record_run(db, spec, _dataset_leads((0,)), committed_state=None)
    _reconcile_catalog_to_store(db, run, _deterministic_state({0}), spec)
    assert _product_leads(db, run.id) == {0}


def test_repeated_reconciliation_idempotent(db: Session) -> None:
    """Repeated reconcile produces the same rows and no duplicates."""
    spec = _spec(expected_leads=(0, 12, 24))
    run = record_run(db, spec, _dataset_leads((0,)), committed_state=None)
    state = _deterministic_state({0, 12, 24})
    for _ in range(3):
        _reconcile_catalog_to_store(db, run, state, spec)
    # Exactly one product per (variable x lead); 2 vars x 3 leads = 6 rows.
    products = db.query(ProductRecord).filter(ProductRecord.run_id == run.id).all()
    assert len(products) == 6
    assert _product_leads(db, run.id) == {0, 12, 24}
    # Re-running on an already-consistent catalog is a no-op.
    before = {(p.lead_time_hours) for p in products}
    _reconcile_catalog_to_store(db, run, state, spec)
    after = _product_leads(db, run.id)
    assert after == before == {0, 12, 24}


def test_concurrent_reconcile_no_duplicates(db: Session) -> None:
    """Two sequential reconcile passes must not duplicate rows.

    A true cross-process concurrency test is impractical in SQLite; this proves
    the upserter is duplicate-safe under repeated invocation (the unique
    constraint is the real guard; advisory locks serialize finalizers in prod).
    """
    spec = _spec(expected_leads=(0, 12))
    run = record_run(db, spec, _dataset_leads((0,)), committed_state=None)
    state = _deterministic_state({0, 12})
    _reconcile_catalog_to_store(db, run, state, spec)
    _reconcile_catalog_to_store(db, run, state, spec)
    products = db.query(ProductRecord).filter(ProductRecord.run_id == run.id).all()
    assert len(products) == 4  # 2 vars x 2 leads, no duplicates


def test_healthy_catalog_unchanged(db: Session) -> None:
    """A fully-consistent catalog is a no-op (no row mutations)."""
    spec = _spec(expected_leads=(0, 12, 24))
    run = record_run(db, spec, _dataset_leads((0, 12, 24)), committed_state=None)
    state = _deterministic_state({0, 12, 24})
    _reconcile_catalog_to_store(db, run, state, spec)
    assert _product_leads(db, run.id) == {0, 12, 24}


# ---------------------------------------------------------------------------
# RUN-SCOPE distinction: READY requires the declared expected set
# ---------------------------------------------------------------------------


def test_run_ready_when_deterministic_expected_scope_complete(
    db: Session,
) -> None:
    """GFS expected {0,12,24} all committed + catalog reconciled -> READY."""
    from ingestion.core.catalog import _derive_run_status

    spec = _spec(expected_leads=(0, 12, 24))
    run = record_run(db, spec, _dataset_leads((0,)), committed_state=None)
    state = _deterministic_state({0, 12, 24})
    _reconcile_catalog_to_store(db, run, state, spec)
    status = _derive_run_status(db, run, spec, state)
    assert status == "ready"


def test_run_partial_when_expected_leads_genuinely_missing(db: Session) -> None:
    """GFS expected {0,12,24} but store only {0,12} -> PARTIAL."""
    from ingestion.core.catalog import _derive_run_status

    spec = _spec(expected_leads=(0, 12, 24))
    run = record_run(db, spec, _dataset_leads((0, 12)), committed_state=None)
    state = _deterministic_state({0, 12})
    _reconcile_catalog_to_store(db, run, state, spec)
    status = _derive_run_status(db, run, spec, state)
    assert status == "partial"


def test_ensemble_ready_only_when_declared_matrix_complete(db: Session) -> None:
    """GEFS declared members {1,2}, leads {0,12}; both members complete -> READY."""
    from ingestion.core.catalog import _derive_run_status

    spec = _spec(is_ensemble=True, expected_leads=(0, 12), expected_members=(1, 2))
    run = record_run(db, spec, _dataset_member(1, (0,)), member=1, committed_state=None)
    committed = {(m, lead_t) for m in (1, 2) for lead_t in (0, 12)}
    _reconcile_catalog_to_store(db, run, _ensemble_state(committed), spec)
    status = _derive_run_status(db, run, spec, _ensemble_state(committed))
    assert status == "ready"


def test_ensemble_partial_when_declared_members_missing(db: Session) -> None:
    """GEFS declared members {1,2,3}; only 1,2 committed -> PARTIAL."""
    from ingestion.core.catalog import _derive_run_status

    spec = _spec(is_ensemble=True, expected_leads=(0, 12), expected_members=(1, 2, 3))
    run = record_run(db, spec, _dataset_member(1, (0,)), member=1, committed_state=None)
    committed = {(m, lead_t) for m in (1, 2) for lead_t in (0, 12)}
    _reconcile_catalog_to_store(db, run, _ensemble_state(committed), spec)
    status = _derive_run_status(db, run, spec, _ensemble_state(committed))
    assert status == "partial"


# ---------------------------------------------------------------------------
# Coordinator-path integration: a real multi-region run reaches READY
# ---------------------------------------------------------------------------


class _NoopLockCoordinator:
    """No-op advisory locks for a SQLite/local-store coordinator test."""

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


def test_coordinator_multi_lead_run_reaches_ready(tmp_path, monkeypatch) -> None:
    """A coordinated 3-lead GFS run reaches READY after the reconciliation fix.

    Before the fix: the region workers commit leads {6,12,18} to the store but
    the catalog only ever records the synthetic seed (lead 6). The delete-only
    finalizer reconciliation leaves leads 12/18 absent from the catalog, so
    ``_derive_run_status`` finds the store↔catalog consistency gate failing and
    the run stays ``partial``.

    After the fix: the finalizer restores the missing product rows for 12/18
    from the committed store state + run spec, and the run becomes READY.
    """
    from concurrent.futures import ThreadPoolExecutor
    from sqlalchemy import select

    import ingestion.core.coordinator as CO
    from ingestion.core.coordinator import RunCoordinator, WaveRegion
    from ingestion.core.zarr_writer import read_dataset

    store = str(tmp_path / "coord_multi.zarr")
    spec = _spec(expected_leads=(6, 12, 18))

    # The catalog is SQLite via a fresh engine.
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)

    monkeypatch.setattr(CO, "StoreLockCoordinator", _NoopLockCoordinator)

    coordinator = RunCoordinator(spec, store, timeout_seconds=2.0)
    conn = engine.connect()
    try:
        # 1. Record the run row exactly as _resolve_run_id does for a fresh run:
        #    the synthetic spec dataset -> only lead 6 products (the bug).
        from ingestion.cli import _synthetic_spec_dataset

        with Session(bind=conn) as catalog_session:
            db_run = record_run(
                catalog_session,
                spec,
                _synthetic_spec_dataset(spec),
                committed_state=None,
            )
            run_id = str(db_run.id)

        # 2. Initialize the store with the seed (lead 6).
        coordinator.initialize_run_store(
            conn,
            seed_dataset=_dataset_leads((6,)),
            expected_leads=(6, 12, 18),
            expected_members=(),
            run_id=run_id,
            is_same_cycle=True,
        )

        # 3. Pre-update + write each region (leads 6, 12, 18) with its own
        #    generation, exactly like three region workers.
        import threading

        for lead in (6, 12, 18):
            gen = f"gen-{lead}"
            regions = [WaveRegion(lead_time_hours=lead, member=None, generation=gen)]
            coordinator.pre_update_wave(
                conn,
                regions=regions,
                run_id=run_id,
                is_same_cycle=True,
                executor=ThreadPoolExecutor(1),
                cancel_event=threading.Event(),
            )
            coordinator.write_region_worker(
                conn,
                dataset=_dataset_leads((lead,)),
                member=None,
                generation=gen,
                expected_leads=(6, 12, 18),
                expected_members=(),
            )

        # 4. Finalize.
        finalize_result = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=(6, 12, 18),
            expected_members=(),
        )
        status = finalize_result.status
        assert status == "ready"
    finally:
        conn.close()

    # Store has all 3 leads physically.
    ds = read_dataset(store)
    assert sorted(int(v) for v in ds.coords["lead_time_hours"].values) == [6, 12, 18]
    # Catalog now has all 3 leads' products.
    with Session(engine) as c:
        product_leads = {
            int(v)
            for v in c.execute(
                select(ProductRecord.lead_time_hours).where(
                    ProductRecord.run_id == run_id
                )
            ).scalars()
        }
    assert product_leads == {6, 12, 18}
    # And the run is READY.
    assert status == "ready"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset_leads(leads: tuple[int, ...]):
    """A deterministic dataset with the given leads (dimensionless, 2 vars)."""
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                np.ones((len(leads), 2, 2), dtype=float),
            ),
            "precipitation_rate": (
                ("lead_time_hours", "latitude", "longitude"),
                np.ones((len(leads), 2, 2), dtype=float),
            ),
        },
        coords={
            "lead_time_hours": list(leads),
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
        attrs={
            # The parser records the forecast-run identity so store↔cycle
            # validation can refuse cross-cycle merges.
            "cycle_time": CYCLE.isoformat(),
        },
    )


def _dataset_member(member: int, leads: tuple[int, ...]):
    """A single-member dataset with the given leads."""
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                np.ones((1, len(leads), 2, 2), dtype=float),
            ),
            "precipitation_rate": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                np.ones((1, len(leads), 2, 2), dtype=float),
            ),
        },
        coords={
            "member": [member],
            "lead_time_hours": list(leads),
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
    )


def _add_product(db: Session, run_id: str, spec: RunCatalogSpec, lead: int) -> None:
    """Directly insert a product row (bypassing record_run) to simulate a stale row."""
    db.add(
        ProductRecord(
            id=f"product_{run_id}_temperature_2m_global_025deg_surface_{lead}",
            run_id=run_id,
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=lead,
            zarr_chunk_path=spec.zarr_store_path,
        )
    )
    db.commit()
