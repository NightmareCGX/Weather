"""Focused tests for 14-Day Detailed Metadata Retention Sweeper (Data Lifecycle V3 Milestone 3).

Covers the complete mandatory M3 test matrix:
1. Retention Boundary (now - 14d + 1s, exact 14d, now - 14d - 1s)
2. Permanent Tombstone Preservation (forecast_cycle_lifecycle untouched)
3. Detailed Metadata Purge (model_runs, products, members, queue deleted)
4. Younger-Than-14-Day Retention (deleted_at > cutoff retained)
5. Exact 14-Day Inclusive Boundary (deleted_at == cutoff eligible)
6. Non-Starvation / Batch Progress (already-swept tombstones do not starve subsequent batches)
7. Bounded Batch (batch_size finite, positive, normalized)
8. FK-Safe Purge (child-first deletion order under enforced FK constraints)
9. Per-Cycle Failure Isolation (failed cycle rolls back, prior and later cycles commit)
10. Retry (previously failed cycle swept on next pass)
11. Already-Swept Tombstone (not selected as actionable candidate)
12. No Physical Work (zero calls to S3 or store gate)
13. M2 Metadata Retention Compatibility (M2 finalizer preserves metadata, M3 sweeps at 14d)
14. Dry-run mode (zero mutations)
15. Multi-version cycle purge (all runs for the cycle purged together)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from domain.lifecycle import (
    METADATA_RETENTION_DAYS,
    canonical_cycle_store_path,
    is_metadata_purge_eligible,
)
from ingestion.core.catalog import (
    CenterRecord,
    EnsembleMemberProductRecord,
    EnsembleMemberRecord,
    ForecastCycleLifecycleRecord,
    GridRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    ReclamationQueueRecord,
    VariableRecord,
    _ensure_utc_datetime,
)
from ingestion.core.db import CatalogBase
from ingestion.gc.finalizer import finalize_cycle_physical_and_queue
from ingestion.gc.sweeper import (
    DEFAULT_SWEEPER_BATCH_SIZE,
    _normalize_batch_size,
    discover_sweeper_candidates,
    purge_cycle_metadata,
    run_metadata_sweeper_pass,
)


@pytest.fixture()
def catalog_engine(tmp_path):
    """SQLite engine with enforced foreign key constraints."""
    db_file = tmp_path / "catalog.db"
    engine = create_engine(f"sqlite:///{db_file}")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    CatalogBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _init_catalog_basics(
    session: Session,
    model_id: str = "gfs",
    version_string: str = "v1.0",
) -> ModelVersionRecord:
    """Seed foundational center, model, version, grid, and variable rows."""
    center = session.execute(
        select(CenterRecord).where(CenterRecord.center_id == "noaa")
    ).scalar_one_or_none()
    if not center:
        center = CenterRecord(id="noaa", center_id="noaa", name="NOAA", country="USA")
        session.add(center)
        session.flush()

    model = session.execute(
        select(ModelRecord).where(ModelRecord.model_id == model_id)
    ).scalar_one_or_none()
    if not model:
        model = ModelRecord(
            id=model_id,
            model_id=model_id,
            center_id="noaa",
            name=model_id.upper(),
            is_ensemble=(model_id == "gefs"),
            resolution_km=25.0,
        )
        session.add(model)
        session.flush()

    mv = session.execute(
        select(ModelVersionRecord).where(
            ModelVersionRecord.model_id == model_id,
            ModelVersionRecord.version_string == version_string,
        )
    ).scalar_one_or_none()
    if not mv:
        mv = ModelVersionRecord(
            id=f"{model_id}_{version_string}",
            model_id=model_id,
            version_string=version_string,
        )
        session.add(mv)
        session.flush()

    grid = session.execute(
        select(GridRecord).where(GridRecord.grid_code == "global_025")
    ).scalar_one_or_none()
    if not grid:
        grid = GridRecord(
            id="global_025",
            grid_code="global_025",
            name="Global 0.25deg",
            resolution_km=25.0,
        )
        session.add(grid)
        session.flush()

    var = session.execute(
        select(VariableRecord).where(VariableRecord.variable_code == "t2m")
    ).scalar_one_or_none()
    if not var:
        var = VariableRecord(
            id="t2m",
            variable_code="t2m",
            name="2m Temperature",
            unit="K",
        )
        session.add(var)
        session.flush()

    session.flush()
    return mv


def _seed_cycle(
    engine,
    *,
    model_id: str = "gfs",
    cycle_time: datetime,
    version_string: str = "v1.0",
    deleted_at: datetime | None = None,
    deletion_started_at: datetime | None = None,
    with_queue: bool = True,
    with_products: bool = True,
    with_members: bool = True,
) -> str:
    """Seed a complete lifecycle cycle with runs, products, members, and queue rows."""
    c_time = _ensure_utc_datetime(cycle_time)
    d_at = _ensure_utc_datetime(deleted_at) if deleted_at is not None else None
    ds_at = (
        _ensure_utc_datetime(deletion_started_at)
        if deletion_started_at is not None
        else d_at
    )

    with Session(engine) as session:
        mv = _init_catalog_basics(session, model_id=model_id, version_string=version_string)

        run_id = f"run_{model_id}_{version_string}_{int(c_time.timestamp())}"
        run = ModelRunRecord(
            id=run_id,
            model_version_id=mv.id,
            cycle_time=c_time,
            status="ready",
            zarr_store_path=canonical_cycle_store_path(model_id, c_time),
            created_at=c_time,
        )
        session.add(run)
        session.flush()

        if with_products:
            prod = ProductRecord(
                id=f"prod_{run_id}_0",
                run_id=run_id,
                variable_id="t2m",
                grid_id="global_025",
                product_type="surface",
                lead_time_hours=0,
                zarr_chunk_path=None,
            )
            session.add(prod)

        if with_members:
            mem = EnsembleMemberRecord(
                id=f"mem_{run_id}_1",
                run_id=run_id,
                member_index=1,
                member_name="gep01",
            )
            session.add(mem)
            emp = EnsembleMemberProductRecord(
                id=f"emp_{run_id}_1_0",
                run_id=run_id,
                member_index=1,
                lead_time_hours=0,
            )
            session.add(emp)

        if with_queue:
            q_row = ReclamationQueueRecord(
                id=f"q_{run_id}_0",
                run_id=run_id,
                model_id=model_id,
                cycle_time=c_time,
                lead_time_hours=0,
                variable_code="t2m",
                target_kind="det",
                member_index=0,
                valid_time=c_time,
                store_path=canonical_cycle_store_path(model_id, c_time),
                physical_key=f"{model_id}/{c_time.strftime('%Y-%m-%d/%H')}/cycle.zarr/t2m/0",
                status="deleted",
                attempt_count=1,
                last_error=None,
                reclaimed_at=d_at,
                created_at=c_time,
                updated_at=d_at or c_time,
            )
            session.add(q_row)

        lc = session.get(ForecastCycleLifecycleRecord, (model_id, c_time))
        if not lc:
            lc = ForecastCycleLifecycleRecord(
                model_id=model_id,
                cycle_time=c_time,
                deleted_at=d_at,
                deletion_started_at=ds_at,
                created_at=c_time,
                updated_at=c_time,
            )
            session.add(lc)
        else:
            setattr(lc, "deleted_at", d_at)
            setattr(lc, "deletion_started_at", ds_at)

        session.commit()
    return run_id


# ---------------------------------------------------------------------------
# 1. Retention Boundary Math & Inclusive Comparisons
# ---------------------------------------------------------------------------


def test_retention_boundary_exact_math() -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    exact_14d_ago = now_utc - timedelta(days=METADATA_RETENTION_DAYS)

    # 1. Younger than 14d by 1 second -> retained (False)
    just_under_14d = exact_14d_ago + timedelta(seconds=1)
    assert not is_metadata_purge_eligible(just_under_14d, now_utc=now_utc)

    # 2. Exactly 14d -> eligible (True)
    assert is_metadata_purge_eligible(exact_14d_ago, now_utc=now_utc)

    # 3. Older than 14d by 1 second -> eligible (True)
    older_than_14d = exact_14d_ago - timedelta(seconds=1)
    assert is_metadata_purge_eligible(older_than_14d, now_utc=now_utc)

    # 4. None -> not eligible (False)
    assert not is_metadata_purge_eligible(None, now_utc=now_utc)


def test_retention_boundary_candidate_discovery(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    exact_14d_ago = now_utc - timedelta(days=14)

    # Candidate A: younger by 1 second (retained)
    _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc),
        deleted_at=exact_14d_ago + timedelta(seconds=1),
    )
    # Candidate B: exact 14 days (eligible)
    _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc),
        deleted_at=exact_14d_ago,
    )
    # Candidate C: older by 1 second (eligible)
    _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        deleted_at=exact_14d_ago - timedelta(seconds=1),
    )

    with Session(catalog_engine) as session:
        cands = discover_sweeper_candidates(session, cutoff=exact_14d_ago)

    cycle_times = [c.cycle_time for c in cands]
    assert datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc) not in cycle_times
    assert datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc) in cycle_times
    assert datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc) in cycle_times
    assert len(cands) == 2


# ---------------------------------------------------------------------------
# 2. Permanent Tombstone Preservation
# ---------------------------------------------------------------------------


def test_permanent_tombstone_preservation(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    d_at = now_utc - timedelta(days=15)
    ds_at = now_utc - timedelta(days=15, minutes=1)

    _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        deleted_at=d_at,
        deletion_started_at=ds_at,
    )

    res = run_metadata_sweeper_pass(catalog_engine, now=now_utc)
    assert len(res.swept_cycles) == 1
    assert res.swept_cycles[0] == ("gfs", c_time)

    # Verify forecast_cycle_lifecycle survives unchanged
    with Session(catalog_engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
        assert lc is not None
        assert _ensure_utc_datetime(lc.deleted_at) == d_at
        assert _ensure_utc_datetime(lc.deletion_started_at) == ds_at
        assert lc.model_id == "gfs"
        assert _ensure_utc_datetime(lc.cycle_time) == c_time


# ---------------------------------------------------------------------------
# 3. Detailed Metadata Purge Across All Target Tables
# ---------------------------------------------------------------------------


def test_detailed_metadata_purge_all_tables(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    d_at = now_utc - timedelta(days=15)

    run_id = _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        deleted_at=d_at,
        with_queue=True,
        with_products=True,
        with_members=True,
    )

    # Before sweep, detailed rows exist
    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, run_id) is not None
        assert session.execute(select(ProductRecord).where(ProductRecord.run_id == run_id)).first() is not None
        assert session.execute(select(EnsembleMemberRecord).where(EnsembleMemberRecord.run_id == run_id)).first() is not None
        assert session.execute(select(EnsembleMemberProductRecord).where(EnsembleMemberProductRecord.run_id == run_id)).first() is not None
        assert session.execute(select(ReclamationQueueRecord).where(ReclamationQueueRecord.run_id == run_id)).first() is not None

    res = run_metadata_sweeper_pass(catalog_engine, now=now_utc)
    assert len(res.swept_cycles) == 1

    # After sweep, 100% of detailed rows are absent
    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, run_id) is None
        assert session.execute(select(ProductRecord).where(ProductRecord.run_id == run_id)).first() is None
        assert session.execute(select(EnsembleMemberRecord).where(EnsembleMemberRecord.run_id == run_id)).first() is None
        assert session.execute(select(EnsembleMemberProductRecord).where(EnsembleMemberProductRecord.run_id == run_id)).first() is None
        assert session.execute(select(ReclamationQueueRecord).where(ReclamationQueueRecord.run_id == run_id)).first() is None

        # Foundation rows MUST survive
        assert session.get(ModelRecord, "gfs") is not None
        assert session.get(ModelVersionRecord, "gfs_v1.0") is not None
        assert session.get(CenterRecord, "noaa") is not None
        assert session.get(ForecastCycleLifecycleRecord, ("gfs", c_time)) is not None


# ---------------------------------------------------------------------------
# 4. Younger-Than-14-Day Retention
# ---------------------------------------------------------------------------


def test_younger_than_14_day_retention(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 10, 0, 0, tzinfo=timezone.utc)
    # Physically finalized 5 days ago (less than 14 days)
    d_at = now_utc - timedelta(days=5)

    run_id = _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        deleted_at=d_at,
    )

    res = run_metadata_sweeper_pass(catalog_engine, now=now_utc)
    assert len(res.candidates) == 0
    assert len(res.swept_cycles) == 0

    # 100% of detailed metadata remains untouched
    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, run_id) is not None
        assert session.execute(select(ProductRecord).where(ProductRecord.run_id == run_id)).first() is not None
        assert session.execute(select(ReclamationQueueRecord).where(ReclamationQueueRecord.run_id == run_id)).first() is not None


# ---------------------------------------------------------------------------
# 5. Exact 14-Day Inclusive Boundary
# ---------------------------------------------------------------------------


def test_exact_14_day_inclusive_boundary(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)
    exact_cutoff = now_utc - timedelta(days=14)

    run_id = _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        deleted_at=exact_cutoff,
    )

    res = run_metadata_sweeper_pass(catalog_engine, now=now_utc)
    assert len(res.candidates) == 1
    assert len(res.swept_cycles) == 1
    assert res.swept_cycles[0] == ("gfs", c_time)

    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, run_id) is None
        assert session.get(ForecastCycleLifecycleRecord, ("gfs", c_time)) is not None


# ---------------------------------------------------------------------------
# 6. Critical Requirement: Non-Starvation & Batch Forward Progress
# ---------------------------------------------------------------------------


def test_non_starvation_and_batch_forward_progress(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 25, 0, 0, 0, tzinfo=timezone.utc)
    cutoff = now_utc - timedelta(days=14)

    # Seed 5 eligible cycles (C0, C1, C2, C3, C4) with ascending deleted_at
    base_t = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    cycles = [
        base_t + timedelta(hours=i * 6)
        for i in range(5)
    ]
    for i, c_time in enumerate(cycles):
        _seed_cycle(
            catalog_engine,
            model_id="gfs",
            cycle_time=c_time,
            deleted_at=cutoff - timedelta(hours=(5 - i)),
        )

    # Pass 1: batch_size=2 sweeps C0, C1
    res1 = run_metadata_sweeper_pass(catalog_engine, now=now_utc, batch_size=2)
    assert len(res1.candidates) == 2
    assert len(res1.swept_cycles) == 2
    assert res1.swept_cycles == (("gfs", cycles[0]), ("gfs", cycles[1]))

    # Pass 2: batch_size=2 sweeps C2, C3
    # CRITICAL: C0 and C1 tombstones MUST NOT starve C2, C3!
    res2 = run_metadata_sweeper_pass(catalog_engine, now=now_utc, batch_size=2)
    assert len(res2.candidates) == 2
    assert len(res2.swept_cycles) == 2
    assert res2.swept_cycles == (("gfs", cycles[2]), ("gfs", cycles[3]))

    # Pass 3: batch_size=2 sweeps C4
    res3 = run_metadata_sweeper_pass(catalog_engine, now=now_utc, batch_size=2)
    assert len(res3.candidates) == 1
    assert len(res3.swept_cycles) == 1
    assert res3.swept_cycles == (("gfs", cycles[4]),)

    # Pass 4: all swept, 0 candidates
    res4 = run_metadata_sweeper_pass(catalog_engine, now=now_utc, batch_size=2)
    assert len(res4.candidates) == 0
    assert len(res4.swept_cycles) == 0

    # All 5 tombstones survive
    with Session(catalog_engine) as session:
        for c_time in cycles:
            assert session.get(ForecastCycleLifecycleRecord, ("gfs", c_time)) is not None


# ---------------------------------------------------------------------------
# 7. Bounded Batch Semantics
# ---------------------------------------------------------------------------


def test_bounded_batch_semantics(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 25, 0, 0, 0, tzinfo=timezone.utc)
    cutoff = now_utc - timedelta(days=15)

    for i in range(10):
        _seed_cycle(
            catalog_engine,
            model_id="gfs",
            cycle_time=datetime(2026, 8, 1, i, 0, tzinfo=timezone.utc),
            deleted_at=cutoff,
        )

    # Pass with batch_size=3
    res = run_metadata_sweeper_pass(catalog_engine, now=now_utc, batch_size=3)
    assert len(res.candidates) == 3
    assert len(res.swept_cycles) == 3

    # Batch normalization guarantees
    assert _normalize_batch_size(None) == DEFAULT_SWEEPER_BATCH_SIZE
    assert _normalize_batch_size(25) == 25
    with pytest.raises(ValueError, match="positive integer"):
        _normalize_batch_size(0)
    with pytest.raises(ValueError, match="positive integer"):
        _normalize_batch_size(-5)


# ---------------------------------------------------------------------------
# 8. FK-Safe Child-First Purge Order
# ---------------------------------------------------------------------------


def test_fk_safe_purge_all_child_combinations(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)

    _seed_cycle(
        catalog_engine,
        model_id="gefs",
        cycle_time=c_time,
        deleted_at=now_utc - timedelta(days=16),
        with_queue=True,
        with_products=True,
        with_members=True,
    )

    # Purge directly under enforced SQLite foreign keys
    purge_res = purge_cycle_metadata(catalog_engine, "gefs", c_time)
    assert purge_res.success is True
    assert purge_res.model_runs_deleted == 1
    assert purge_res.forecast_products_deleted == 1
    assert purge_res.ensemble_members_deleted == 1
    assert purge_res.ensemble_member_products_deleted == 1
    assert purge_res.reclamation_queue_deleted >= 1


# ---------------------------------------------------------------------------
# 9. Per-Cycle Failure Isolation
# ---------------------------------------------------------------------------


def test_per_cycle_failure_isolation(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c1 = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    c2 = datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc)
    c3 = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    d_at = now_utc - timedelta(days=16)

    r1 = _seed_cycle(catalog_engine, model_id="gfs", cycle_time=c1, deleted_at=d_at)
    r2 = _seed_cycle(catalog_engine, model_id="gfs", cycle_time=c2, deleted_at=d_at)
    r3 = _seed_cycle(catalog_engine, model_id="gfs", cycle_time=c3, deleted_at=d_at)

    original_purge = purge_cycle_metadata

    def _failing_purge(engine, model_id, cycle_time):
        if cycle_time == c2:
            raise RuntimeError("Simulated transient PostgreSQL disk/network error on cycle 2")
        return original_purge(engine, model_id, cycle_time)

    with patch("ingestion.gc.sweeper.purge_cycle_metadata", side_effect=_failing_purge):
        res = run_metadata_sweeper_pass(catalog_engine, now=now_utc, batch_size=10)

    # Prior cycle c1 and later cycle c3 succeed; c2 fails
    assert ("gfs", c1) in res.swept_cycles
    assert ("gfs", c3) in res.swept_cycles
    assert ("gfs", c2) in res.failed_cycles

    with Session(catalog_engine) as session:
        # c1 swept
        assert session.get(ModelRunRecord, r1) is None
        # c2 retained due to transaction rollback
        assert session.get(ModelRunRecord, r2) is not None
        lc2 = session.get(ForecastCycleLifecycleRecord, ("gfs", c2))
        assert lc2 is not None
        assert _ensure_utc_datetime(lc2.deleted_at) == d_at
        # c3 swept
        assert session.get(ModelRunRecord, r3) is None


# ---------------------------------------------------------------------------
# 10. Retry After Transient Failure
# ---------------------------------------------------------------------------


def test_retry_after_transient_failure(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    d_at = now_utc - timedelta(days=16)

    run_id = _seed_cycle(catalog_engine, model_id="gfs", cycle_time=c_time, deleted_at=d_at)

    # Pass 1: failure simulated
    with patch("ingestion.gc.sweeper.purge_cycle_metadata", side_effect=RuntimeError("Lock timeout")):
        res1 = run_metadata_sweeper_pass(catalog_engine, now=now_utc)
        assert len(res1.failed_cycles) == 1

    # Pass 2: transient failure removed -> succeeds
    res2 = run_metadata_sweeper_pass(catalog_engine, now=now_utc)
    assert len(res2.swept_cycles) == 1
    assert res2.swept_cycles[0] == ("gfs", c_time)

    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, run_id) is None
        assert session.get(ForecastCycleLifecycleRecord, ("gfs", c_time)) is not None


# ---------------------------------------------------------------------------
# 11. Already-Swept Tombstone Not Selected
# ---------------------------------------------------------------------------


def test_already_swept_tombstone_not_selected(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    cutoff = now_utc - timedelta(days=14)

    # Insert tombstone only (no model_runs)
    with Session(catalog_engine) as session:
        _init_catalog_basics(session, "gfs")
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c_time,
                deleted_at=cutoff - timedelta(days=5),
                deletion_started_at=cutoff - timedelta(days=5),
                created_at=c_time,
                updated_at=c_time,
            )
        )
        session.commit()

    with Session(catalog_engine) as session:
        cands = discover_sweeper_candidates(session, cutoff=cutoff)
    assert len(cands) == 0


# ---------------------------------------------------------------------------
# 12. No Physical Storage Work
# ---------------------------------------------------------------------------


def test_no_physical_storage_work(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)

    _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        deleted_at=now_utc - timedelta(days=15),
    )

    with patch("ingestion.gc.reconciler.delete_physical_store_gated") as mock_gate, \
         patch("ingestion.gc.reconciler._delete_store_prefix") as mock_s3, \
         patch("ingestion.core.locks.StoreLockCoordinator.acquire_exclusive_gate") as mock_lock:
        res = run_metadata_sweeper_pass(catalog_engine, now=now_utc)
        assert len(res.swept_cycles) == 1

        mock_gate.assert_not_called()
        mock_s3.assert_not_called()
        mock_lock.assert_not_called()


# ---------------------------------------------------------------------------
# 13. M2 Metadata Retention Compatibility
# ---------------------------------------------------------------------------


def test_m2_metadata_retention_compatibility(catalog_engine) -> None:
    c_time = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    t0_finalized = datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc)

    # 1. Seed cycle with runs and queue rows
    run_id = _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        deleted_at=None,
        deletion_started_at=t0_finalized,
    )

    # 2. Run M2 physical and queue finalization
    finalize_cycle_physical_and_queue(
        catalog_engine, "gfs", c_time, finalization_time=t0_finalized
    )

    # 3. Before 14 days (t0 + 5 days): metadata retained
    t_early = t0_finalized + timedelta(days=5)
    res_early = run_metadata_sweeper_pass(catalog_engine, now=t_early)
    assert len(res_early.candidates) == 0
    assert len(res_early.swept_cycles) == 0

    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, run_id) is not None

    # 4. At 14 days (t0 + 14 days): M3 sweeps detailed metadata
    t_eligible = t0_finalized + timedelta(days=14)
    res_eligible = run_metadata_sweeper_pass(catalog_engine, now=t_eligible)
    assert len(res_eligible.candidates) == 1
    assert len(res_eligible.swept_cycles) == 1

    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, run_id) is None
        lc = session.get(ForecastCycleLifecycleRecord, ("gfs", c_time))
        assert lc is not None
        assert _ensure_utc_datetime(lc.deleted_at) == t0_finalized


# ---------------------------------------------------------------------------
# 14. Dry-Run Mode
# ---------------------------------------------------------------------------


def test_dry_run_mode(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)

    run_id = _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        deleted_at=now_utc - timedelta(days=15),
    )

    res = run_metadata_sweeper_pass(catalog_engine, now=now_utc, dry_run=True)
    assert res.dry_run is True
    assert len(res.candidates) == 1
    assert len(res.swept_cycles) == 0
    assert res.total_model_runs_deleted == 0

    # Zero mutations in dry run
    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, run_id) is not None
        assert session.get(ForecastCycleLifecycleRecord, ("gfs", c_time)) is not None


# ---------------------------------------------------------------------------
# 15. Multi-Version Cycle Purge
# ---------------------------------------------------------------------------


def test_multi_version_runs_same_cycle(catalog_engine) -> None:
    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    d_at = now_utc - timedelta(days=15)

    # Seed run for v1.0
    r1 = _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        version_string="v1.0",
        deleted_at=d_at,
    )
    # Seed second run for v2.0 for the SAME cycle
    r2 = _seed_cycle(
        catalog_engine,
        model_id="gfs",
        cycle_time=c_time,
        version_string="v2.0",
        deleted_at=d_at,
    )

    res = run_metadata_sweeper_pass(catalog_engine, now=now_utc)
    assert len(res.swept_cycles) == 1
    assert res.total_model_runs_deleted == 2

    # Both version runs are purged
    with Session(catalog_engine) as session:
        assert session.get(ModelRunRecord, r1) is None
        assert session.get(ModelRunRecord, r2) is None
        assert session.get(ForecastCycleLifecycleRecord, ("gfs", c_time)) is not None


# ---------------------------------------------------------------------------
# 16. Real PostgreSQL Integration Test
# ---------------------------------------------------------------------------


def test_sweeper_pass_real_postgres() -> None:
    import os
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://weather_user:weather_password@localhost:5432/weather_db",
    )
    pg_engine = create_engine(db_url)
    try:
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL not available")

    # Apply migrations
    alembic_cfg = Config("services/api/alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", "services/api/alembic")
    try:
        command.upgrade(alembic_cfg, "head")
    except Exception:
        pass

    now_utc = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)
    c_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    d_at = now_utc - timedelta(days=15)

    run_id = _seed_cycle(
        pg_engine,
        model_id="gfs",
        cycle_time=c_time,
        deleted_at=d_at,
        with_queue=True,
        with_products=True,
        with_members=True,
    )

    res = run_metadata_sweeper_pass(pg_engine, now=now_utc)
    assert ("gfs", c_time) in res.swept_cycles

    with Session(pg_engine) as session:
        assert session.get(ModelRunRecord, run_id) is None
        assert session.get(ForecastCycleLifecycleRecord, ("gfs", c_time)) is not None

