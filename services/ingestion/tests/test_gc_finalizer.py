"""Focused tests for V3 Whole-Cycle End-of-Life Finalizer and M2 Concurrency Hardening."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from domain.horizon import (
    MODEL_CANONICAL_HORIZONS,
    MODEL_VERSION_HORIZONS,
    register_canonical_lead_horizon,
)
from domain.lifecycle import canonical_cycle_store_path
from domain.temporal import serving_start_valid_time
from ingestion.core.base import CycleTombstonedError
from ingestion.core.catalog import (
    CatalogBase,
    CenterRecord,
    ForecastCycleLifecycleRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    ReclamationQueueRecord,
    RunCatalogSpec,
    VariableSpec,
    _ensure_utc_datetime,
    ensure_lifecycle_row,
    reserve_run,
)
from ingestion.core.coordinator import RunCoordinator
from ingestion.gc.finalizer import (
    claim_fresh_candidate,
    enumerate_cycle_store_paths,
    finalize_cycle_eol,
    finalize_cycle_physical_and_queue,
    run_finalizer_pass,
)


@pytest.fixture()
def _restore_horizons():
    saved = dict(MODEL_CANONICAL_HORIZONS)
    saved_versions = dict(MODEL_VERSION_HORIZONS)
    yield
    MODEL_CANONICAL_HORIZONS.clear()
    MODEL_CANONICAL_HORIZONS.update(saved)
    MODEL_VERSION_HORIZONS.clear()
    MODEL_VERSION_HORIZONS.update(saved_versions)


class _NoopCoordinator:
    """Stub coordinator for offline SQLite test harnesses."""
    def __init__(self, *a, **k): pass
    def acquire_shared_gate(self): pass
    def release_shared_gate(self): pass
    def acquire_exclusive_gate(self): pass
    def release_exclusive_gate(self): pass
    def acquire_admission(self): pass
    def release_admission(self): pass
    def acquire_shared_admission(self): pass
    def release_shared_admission(self): pass
    def acquire_region_locks(self, r): pass
    def release_region_locks(self, r): pass


@pytest.fixture()
def catalog_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    CatalogBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _make_spec(
    model_id: str = "gfs",
    cycle_time: datetime | None = None,
    version_string: str = "v1.0",
    store_path: str | None = None,
) -> RunCatalogSpec:
    c_time = (
        cycle_time
        if cycle_time is not None
        else datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    )
    s_path = (
        store_path
        if store_path is not None
        else canonical_cycle_store_path(model_id, c_time)
    )
    return RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id=model_id,
        model_name=model_id.upper(),
        is_ensemble=(model_id == "gefs"),
        resolution_km=25.0,
        version_string=version_string,
        cycle_time=c_time,
        grid_id="global_025deg",
        grid_name="Global 0.25deg",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=s_path,
        variables=(
            VariableSpec(
                code="temperature_2m", name="2m Temperature", unit="K"
            ),
        ),
        expected_lead_time_hours=(0, 3, 6),
        expected_members=() if model_id != "gefs" else tuple(range(1, 31)),
    )


# ---------------------------------------------------------------------------
# A. Reservation Safety
# ---------------------------------------------------------------------------


def test_reserve_run_creates_row_before_physical_write(catalog_engine) -> None:
    spec = _make_spec(store_path="s3://weather-data/gfs/2026-08-01/00/cycle.zarr")
    with Session(catalog_engine) as session:
        run = reserve_run(session, spec)
        session.commit()
        run_id = run.id

    with Session(catalog_engine) as session:
        row = session.get(ModelRunRecord, run_id)
        assert row is not None
        assert row.status == "processing"
        assert (
            row.zarr_store_path
            == "s3://weather-data/gfs/2026-08-01/00/cycle.zarr"
        )
        assert _ensure_utc_datetime(row.cycle_time) == spec.cycle_time

        # Zero product rows created during reservation
        prods = (
            session.execute(
                select(ProductRecord).where(ProductRecord.run_id == run_id)
            )
            .scalars()
            .all()
        )
        assert len(prods) == 0


def test_reserve_run_idempotent_retry_reuses_row(catalog_engine) -> None:
    spec = _make_spec()
    with Session(catalog_engine) as session:
        run1 = reserve_run(session, spec)
        session.commit()
        id1 = run1.id

    with Session(catalog_engine) as session:
        run2 = reserve_run(session, spec)
        session.commit()
        id2 = run2.id

    assert id1 == id2


def test_reserve_run_rejects_conflicting_store_path(catalog_engine) -> None:
    spec1 = _make_spec(store_path="s3://weather-data/gfs/store1.zarr")
    with Session(catalog_engine) as session:
        reserve_run(session, spec1)
        session.commit()

    spec2 = _make_spec(store_path="s3://weather-data/gfs/store2.zarr")
    with Session(catalog_engine) as session:
        with pytest.raises(ValueError, match="Cannot change immutable store path"):
            reserve_run(session, spec2)


def test_reserve_run_rejects_if_lifecycle_claimed_or_deleted(
    catalog_engine,
) -> None:
    spec = _make_spec()
    now_utc = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

    # Claim the cycle
    with Session(catalog_engine) as session:
        ensure_lifecycle_row(session, spec.model_id, spec.cycle_time)
        lc = session.execute(
            select(ForecastCycleLifecycleRecord).where(
                ForecastCycleLifecycleRecord.model_id == spec.model_id,
                ForecastCycleLifecycleRecord.cycle_time == spec.cycle_time,
            )
        ).scalar_one()
        lc.deletion_started_at = now_utc
        session.commit()

    with Session(catalog_engine) as session:
        with pytest.raises(CycleTombstonedError, match="Refusing to reserve run"):
            reserve_run(session, spec)


# ---------------------------------------------------------------------------
# B. Reservation / Claim Serialization
# ---------------------------------------------------------------------------


def test_claim_fresh_candidate_sees_committed_reservation(
    catalog_engine,
) -> None:
    c_time = datetime(2026, 7, 15, 0, 0, tzinfo=timezone.utc)
    spec = _make_spec(
        cycle_time=c_time, store_path="s3://custom/gfs_custom.zarr"
    )

    with Session(catalog_engine) as session:
        reserve_run(session, spec)
        session.commit()

    now_utc = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    serving_start = serving_start_valid_time(now_utc)

    with Session(catalog_engine) as session:
        claimed = claim_fresh_candidate(
            session, "gfs", c_time, serving_start=serving_start, claim_time=now_utc
        )
        assert claimed is True

        stores = enumerate_cycle_store_paths(session, "gfs", c_time)
        assert "s3://custom/gfs_custom.zarr" in stores


def test_claim_fresh_candidate_missing_versions_fails_closed(
    catalog_engine,
) -> None:
    c_time = datetime(2026, 7, 15, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    serving_start = serving_start_valid_time(now_utc)

    # Zero runs exist in database!
    with Session(catalog_engine) as session:
        claimed = claim_fresh_candidate(
            session, "gfs", c_time, serving_start=serving_start, claim_time=now_utc
        )
        assert claimed is False


# ---------------------------------------------------------------------------
# C. Version-Aware Fresh Eligibility
# ---------------------------------------------------------------------------


def test_claim_fresh_candidate_multi_version_conservative_horizon(
    catalog_engine, _restore_horizons
) -> None:
    """v1 = 240h, v2 = 300h.

    Cycle at serving_start - 270h:
    Contains both v1 and v2 runs.
    Must NOT be claimed until v2 (300h) expires.
    """
    register_canonical_lead_horizon(
        "gfs", tuple(range(0, 241, 3)), version_string="v1.0"
    )
    register_canonical_lead_horizon(
        "gfs", tuple(range(0, 301, 3)), version_string="v2.0"
    )

    now_utc = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    serving_start = serving_start_valid_time(now_utc)
    c_270h = serving_start - timedelta(hours=270)

    # Reserve both v1 and v2 for c_270h
    spec_v1 = _make_spec(
        cycle_time=c_270h,
        version_string="v1.0",
        store_path="s3://bucket/v1.zarr",
    )
    spec_v2 = _make_spec(
        cycle_time=c_270h,
        version_string="v2.0",
        store_path="s3://bucket/v2.zarr",
    )
    with Session(catalog_engine) as session:
        reserve_run(session, spec_v1)
        reserve_run(session, spec_v2)
        session.commit()

    # 1. At 270h old: 270h > 240h (v1 expired), but 270h < 300h (v2 servable).
    # Conservative multi-version horizon is 300h -> NOT claimable!
    with Session(catalog_engine) as session:
        claimed = claim_fresh_candidate(
            session,
            "gfs",
            c_270h,
            serving_start=serving_start,
            claim_time=now_utc,
        )
        assert claimed is False

    # 2. Advance time so cycle is 301h old (> 300h, both expired) -> claimable!
    now_future = now_utc + timedelta(hours=35)
    serving_start_future = serving_start_valid_time(now_future)
    with Session(catalog_engine) as session:
        claimed2 = claim_fresh_candidate(
            session,
            "gfs",
            c_270h,
            serving_start=serving_start_future,
            claim_time=now_future,
        )
        assert claimed2 is True


def test_claim_fresh_candidate_strict_boundary(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    serving_start = serving_start_valid_time(now_utc)
    # Exact boundary: cycle + 240h == serving_start
    c_exact = serving_start - timedelta(hours=240)

    spec = _make_spec(cycle_time=c_exact)
    with Session(catalog_engine) as session:
        reserve_run(session, spec)
        session.commit()

    # Exact boundary: NOT expired (strict < required) -> NOT claimable
    with Session(catalog_engine) as session:
        claimed = claim_fresh_candidate(
            session,
            "gfs",
            c_exact,
            serving_start=serving_start,
            claim_time=now_utc,
        )
        assert claimed is False


# ---------------------------------------------------------------------------
# D. Writer-Gate TOCTOU (Under-Gate Recheck)
# ---------------------------------------------------------------------------


def test_under_gate_recheck_blocks_mutations_after_claim(
    catalog_engine, tmp_path, monkeypatch
) -> None:
    """Guarantee C test:

    If deletion_started_at is committed, RunCoordinator under-gate check raises
    CycleTombstonedError inside initialize_run_store, pre_update_wave, write_region_worker,
    finalize_run, and publish_settled_lead.
    """
    monkeypatch.setattr("ingestion.core.coordinator.StoreLockCoordinator", _NoopCoordinator)

    store_dir = tmp_path / "test_cycle.zarr"
    store_path = str(store_dir)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    spec = _make_spec(cycle_time=c_time, store_path=store_path)

    with Session(catalog_engine) as session:
        reserve_run(session, spec)
        session.commit()

    # Finalizer stamps claim
    now_utc = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    with Session(catalog_engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
        assert lc is not None
        lc.deletion_started_at = now_utc
        session.commit()

    coordinator = RunCoordinator(spec, store_path)
    with catalog_engine.connect() as conn:
        import xarray as xr

        dummy_ds = xr.Dataset()

        # 1. initialize_run_store raises under gate
        with pytest.raises(CycleTombstonedError, match="Refusing store initialization"):
            coordinator.initialize_run_store(
                conn,
                seed_dataset=dummy_ds,
                expected_leads=(0, 3),
                expected_members=(),
                run_id=None,
                is_same_cycle=False,
            )

        # 2. pre_update_wave raises under gate
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(CycleTombstonedError, match="Refusing wave pre-update"):
                coordinator.pre_update_wave(
                    conn,
                    regions=[],
                    run_id=None,
                    is_same_cycle=False,
                    executor=pool,
                    cancel_event=threading.Event(),
                )

        # 3. write_region_worker raises under gate
        with pytest.raises(CycleTombstonedError, match="Refusing region write"):
            coordinator.write_region_worker(
                conn,
                dataset=dummy_ds,
                member=None,
                generation="gen1",
                expected_leads=(0,),
            )

        # 4. finalize_run raises under gate
        with pytest.raises(CycleTombstonedError, match="Refusing run finalization"):
            coordinator.finalize_run(
                conn,
                run_id="run_dummy",
                spec=spec,
                expected_leads=(0,),
                expected_members=(),
            )


# ---------------------------------------------------------------------------
# E. Multi-Store Physical Deletion & Deduplication
# ---------------------------------------------------------------------------


def test_finalize_cycle_deletes_multiple_stores_sequentially(
    catalog_engine, tmp_path
) -> None:
    store1 = tmp_path / "store1.zarr"
    store2 = tmp_path / "store2.zarr"
    store1.mkdir(parents=True, exist_ok=True)
    store2.mkdir(parents=True, exist_ok=True)
    (store1 / "data.txt").write_text("store1")
    (store2 / "data.txt").write_text("store2")

    c_time = datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc)
    spec1 = _make_spec(
        cycle_time=c_time, version_string="v1.0", store_path=str(store1)
    )
    spec2 = _make_spec(
        cycle_time=c_time, version_string="v2.0", store_path=str(store2)
    )

    with Session(catalog_engine) as session:
        reserve_run(session, spec1)
        reserve_run(session, spec2)
        session.commit()

    now_utc = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    serving_start = serving_start_valid_time(now_utc)

    # Finalize cycle
    ok = finalize_cycle_eol(
        catalog_engine,
        "gfs",
        c_time,
        is_recovery=False,
        serving_start=serving_start,
        now=now_utc,
    )
    assert ok is True

    # Both physical stores must be deleted!
    assert not store1.exists()
    assert not store2.exists()

    # deleted_at must be committed
    with Session(catalog_engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
        assert lc is not None
        assert _ensure_utc_datetime(lc.deleted_at) == now_utc
        assert _ensure_utc_datetime(lc.deletion_started_at) == now_utc


# ---------------------------------------------------------------------------
# F. Queue Normalization Contract
# ---------------------------------------------------------------------------


def test_queue_normalization_all_existing_rows_atomic(catalog_engine) -> None:
    c_time = datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc)
    spec = _make_spec(cycle_time=c_time)

    with Session(catalog_engine) as session:
        run = reserve_run(session, spec)
        session.commit()
        run_id = run.id

    now_t0 = datetime(2026, 7, 10, 1, 0, tzinfo=timezone.utc)
    reclaimed_t0 = datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc)

    # Seed reclamation_queue in queued, deleting, failed, and deleted
    with Session(catalog_engine) as session:
        session.add(
            ReclamationQueueRecord(
                id="q_queued",
                run_id=run_id,
                model_id="gfs",
                cycle_time=c_time,
                lead_time_hours=0,
                variable_code="t2m",
                target_kind="det",
                member_index=0,
                valid_time=c_time,
                store_path=spec.zarr_store_path,
                physical_key="k1",
                status="queued",
                attempt_count=1,
                last_error=None,
                reclaimed_at=None,
                created_at=now_t0,
                updated_at=now_t0,
            )
        )
        session.add(
            ReclamationQueueRecord(
                id="q_deleting",
                run_id=run_id,
                model_id="gfs",
                cycle_time=c_time,
                lead_time_hours=3,
                variable_code="t2m",
                target_kind="det",
                member_index=0,
                valid_time=c_time,
                store_path=spec.zarr_store_path,
                physical_key="k2",
                status="deleting",
                attempt_count=2,
                last_error="in_flight",
                reclaimed_at=None,
                created_at=now_t0,
                updated_at=now_t0,
            )
        )
        session.add(
            ReclamationQueueRecord(
                id="q_failed",
                run_id=run_id,
                model_id="gfs",
                cycle_time=c_time,
                lead_time_hours=6,
                variable_code="t2m",
                target_kind="det",
                member_index=0,
                valid_time=c_time,
                store_path=spec.zarr_store_path,
                physical_key="k3",
                status="failed",
                attempt_count=3,
                last_error="s3_timeout",
                reclaimed_at=None,
                created_at=now_t0,
                updated_at=now_t0,
            )
        )
        session.add(
            ReclamationQueueRecord(
                id="q_deleted",
                run_id=run_id,
                model_id="gfs",
                cycle_time=c_time,
                lead_time_hours=9,
                variable_code="t2m",
                target_kind="det",
                member_index=0,
                valid_time=c_time,
                store_path=spec.zarr_store_path,
                physical_key="k4",
                status="deleted",
                attempt_count=1,
                last_error=None,
                reclaimed_at=reclaimed_t0,  # Original timestamp
                created_at=now_t0,
                updated_at=reclaimed_t0,
            )
        )
        session.commit()

    fin_time = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    finalize_cycle_physical_and_queue(
        catalog_engine, "gfs", c_time, finalization_time=fin_time
    )

    with Session(catalog_engine) as session:
        # All rows must be 'deleted'
        rows = {
            r.id: r
            for r in session.execute(
                select(ReclamationQueueRecord).where(
                    ReclamationQueueRecord.cycle_time == c_time
                )
            ).scalars()
        }
        assert len(rows) == 4

        # q_queued -> deleted, reclaimed_at set to fin_time
        assert rows["q_queued"].status == "deleted"
        assert _ensure_utc_datetime(rows["q_queued"].updated_at) == fin_time
        assert _ensure_utc_datetime(rows["q_queued"].reclaimed_at) == fin_time
        assert rows["q_queued"].attempt_count == 1

        # q_deleting -> deleted, reclaimed_at set to fin_time, last_error preserved
        assert rows["q_deleting"].status == "deleted"
        assert _ensure_utc_datetime(rows["q_deleting"].updated_at) == fin_time
        assert _ensure_utc_datetime(rows["q_deleting"].reclaimed_at) == fin_time
        assert rows["q_deleting"].last_error == "in_flight"

        # q_failed -> deleted, reclaimed_at set to fin_time, last_error preserved
        assert rows["q_failed"].status == "deleted"
        assert _ensure_utc_datetime(rows["q_failed"].updated_at) == fin_time
        assert _ensure_utc_datetime(rows["q_failed"].reclaimed_at) == fin_time
        assert rows["q_failed"].last_error == "s3_timeout"

        # q_deleted -> deleted, ORIGINAL reclaimed_at preserved via COALESCE!
        assert rows["q_deleted"].status == "deleted"
        assert _ensure_utc_datetime(rows["q_deleted"].updated_at) == fin_time
        assert _ensure_utc_datetime(rows["q_deleted"].reclaimed_at) == reclaimed_t0


# ---------------------------------------------------------------------------
# G. Crash Recovery & Catalog Retention
# ---------------------------------------------------------------------------


def test_crash_recovery_resumes_without_fresh_horizon_recheck(
    catalog_engine, tmp_path
) -> None:
    store_dir = tmp_path / "recovery_store.zarr"
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "chunk.bin").write_text("chunk")

    c_time = datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc)
    spec = _make_spec(cycle_time=c_time, store_path=str(store_dir))

    with Session(catalog_engine) as session:
        reserve_run(session, spec)
        session.commit()

    # Simulate crash state: claimed (deletion_started_at set), but deleted_at is NULL
    claim_t = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
    with Session(catalog_engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
        assert lc is not None
        lc.deletion_started_at = claim_t
        session.commit()

    now_utc = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    serving_start = serving_start_valid_time(now_utc)

    # Resume finalization as recovery candidate
    ok = finalize_cycle_eol(
        catalog_engine,
        "gfs",
        c_time,
        is_recovery=True,
        serving_start=serving_start,
        now=now_utc,
    )
    assert ok is True

    # Store deleted
    assert not store_dir.exists()

    # deleted_at committed, deletion_started_at preserved
    with Session(catalog_engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
        assert lc is not None
        assert _ensure_utc_datetime(lc.deleted_at) == now_utc
        assert _ensure_utc_datetime(lc.deletion_started_at) == claim_t

        # Catalog metadata is RETAINED (M3 owns purging it, M2 must not!)
        runs = (
            session.execute(
                select(ModelRunRecord).where(
                    ModelRunRecord.cycle_time == c_time
                )
            )
            .scalars()
            .all()
        )
        assert len(runs) == 1


def test_finalizer_discovers_and_finalizes_legacy_cycle_without_lifecycle_row(
    catalog_engine, tmp_path
) -> None:
    """Historical legacy cycle with model_runs/model_versions but NO forecast_cycle_lifecycle row.

    Proves:
    1. run_finalizer_pass discovers the cycle directly from catalog evidence (model_runs outerjoin).
    2. Lifecycle row is race-safely created and locked via ensure_lifecycle_row.
    3. Fresh horizon check passes, deletion_started_at is claimed.
    4. Physical store is deleted.
    5. deleted_at is committed atomically.
    6. ModelRunRecord remains retained in the catalog.
    7. Clean execution against contracted lifecycle schema.
    """
    legacy_store = tmp_path / "legacy_gfs.zarr"
    legacy_store.mkdir(parents=True, exist_ok=True)
    (legacy_store / "chunk.bin").write_text("legacy_data")

    # Historical cycle 720h ago (far beyond 240h horizon)
    c_time = datetime(2026, 7, 2, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)

    # 1. Seed model hierarchy and model_run directly WITHOUT reserve_run and WITHOUT lifecycle row
    with Session(catalog_engine) as session:
        center = CenterRecord(
            id="center_noaa", center_id="noaa", name="NOAA", country="USA"
        )
        model = ModelRecord(
            id="model_gfs",
            model_id="gfs",
            name="GFS",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
        )
        version = ModelVersionRecord(
            id="version_gfs_v1.0", model_id="gfs", version_string="v1.0"
        )
        run = ModelRunRecord(
            id="run_legacy_gfs_2026070200",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_time,
            status="ready",
            zarr_store_path=str(legacy_store),
        )
        session.add_all([center, model, version, run])
        session.commit()

    # 2. Assert lifecycle row explicitly does NOT exist before finalizer pass
    with Session(catalog_engine) as session:
        assert (
            session.get(ForecastCycleLifecycleRecord, ("gfs", c_time)) is None
        )

    assert legacy_store.exists()

    # 3. Execute normal production entry point: run_finalizer_pass
    pass_result = run_finalizer_pass(
        catalog_engine,
        models=("gfs",),
        dry_run=False,
        base_bucket="weather-data",
        timeout_seconds=5.0,
        now=now_utc,
    )

    # 4. Verify candidate discovery & finalization
    assert ("gfs", c_time) in pass_result.claimed_cycles
    assert ("gfs", c_time) in pass_result.finalized_cycles
    assert pass_result.failed_cycles == ()

    # 5. Verify physical store is absent
    assert not legacy_store.exists()

    # 6. Verify lifecycle row created and stamped with both claim and tombstone
    with Session(catalog_engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
        assert lc is not None
        assert _ensure_utc_datetime(lc.deletion_started_at) == now_utc
        assert _ensure_utc_datetime(lc.deleted_at) == now_utc

        # 7. Verify ModelRunRecord is RETAINED (M2 does not purge catalog metadata)
        legacy_run = session.get(ModelRunRecord, "run_legacy_gfs_2026070200")
        assert legacy_run is not None
        assert legacy_run.status == "ready"
        assert legacy_run.zarr_store_path == str(legacy_store)

