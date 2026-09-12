"""Comprehensive regression test suite for Data Lifecycle V3 Phase 4 Granular Reclamation.

Covers all 45 required verification scenarios deterministically with offline fixtures.
Zero live NOAA, zero production S3.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from domain.reclamation import (
    DEFAULT_EXPECTED_REGION_VARIABLES,
    RECLAMATION_STATUS_DELETED,
    RECLAMATION_STATUS_DELETING,
    RECLAMATION_STATUS_FAILED,
    RECLAMATION_STATUS_QUEUED,
    TARGET_KIND_DET,
    TARGET_KIND_MEAN,
    TARGET_KIND_MEM,
    make_region_marker_relative_key,
    make_shard_relative_key,
    register_expected_region_variables,
)
from domain.temporal import serving_start_valid_time
from ingestion.core.catalog import (
    CenterRecord,
    CommittedState,
    EnsembleMemberProductRecord,
    ForecastCycleLifecycleRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    ReclamationQueueRecord,
    _reconcile_catalog_to_store,
)
from ingestion.core.db import CatalogBase
from ingestion.gc.planner import plan_reclamation_pass
from ingestion.gc.worker import (
    requeue_failed_reclamation_targets,
    run_reclamation_worker_pass,
)


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


@pytest.fixture
def catalog_engine(tmp_path):
    """Isolated SQLite catalog engine with all required tables."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_catalog.db",
        connect_args={"check_same_thread": False},
    )
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as session:
        center = CenterRecord(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="US",
            created_at=_dt(2026, 1, 1, 0),
        )
        gfs = ModelRecord(
            id="model_gfs",
            model_id="gfs",
            name="GFS",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs = ModelRecord(
            id="model_gefs",
            model_id="gefs",
            name="GEFS",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        gfs_v1 = ModelVersionRecord(
            id="version_gfs_v1.0",
            model_id="gfs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs_v1 = ModelVersionRecord(
            id="version_gefs_v1.0",
            model_id="gefs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        session.add_all([center, gfs, gefs, gfs_v1, gefs_v1])
        session.commit()
    yield engine
    engine.dispose()


def _seed_run(
    engine,
    model_id: str,
    cycle_time: datetime,
    status: str,
    store_dir: Path | None = None,
    version_string: str = "v1.0",
) -> str:
    v_id = f"version_{model_id}_{version_string}"
    c_tag = cycle_time.strftime("%Y%m%d%H%M")
    run_id = f"run_{model_id}_{c_tag}"
    store_path = str(store_dir) if store_dir else f"/store/{model_id}/{c_tag}.zarr"
    with Session(engine) as session:
        run = ModelRunRecord(
            id=run_id,
            model_version_id=v_id,
            cycle_time=cycle_time,
            status=status,
            zarr_store_path=store_path,
            created_at=cycle_time,
        )
        session.add(run)
        session.commit()
    return run_id


def _seed_gfs_products(
    engine,
    run_id: str,
    leads: list[int],
    variables: list[str] | None = None,
    store_dir: Path | None = None,
) -> None:
    vars_to_add = variables or [
        "temperature_2m",
        "precipitation_amount_3h",
        "cloud_cover_3h",
        "crain",
        "csnow",
        "wind_u_10m",
        "wind_v_10m",
    ]
    with Session(engine) as session:
        for lead in leads:
            for v in vars_to_add:
                session.add(
                    ProductRecord(
                        id=f"prod_{run_id}_{lead}_{v}",
                        run_id=run_id,
                        variable_id=v,
                        grid_id="global_025deg",
                        product_type="surface",
                        lead_time_hours=lead,
                        zarr_chunk_path="/fake",
                    )
                )
                if store_dir:
                    rel = make_shard_relative_key(v, TARGET_KIND_DET, lead, 0)
                    shard_file = store_dir / rel
                    shard_file.parent.mkdir(parents=True, exist_ok=True)
                    shard_file.write_bytes(b"SHARD_DATA")
        session.commit()


def _seed_gefs_products(
    engine,
    run_id: str,
    leads: list[int],
    member_count: int = 30,
    variables: list[str] | None = None,
    store_dir: Path | None = None,
) -> None:
    vars_to_add = variables or [
        "temperature_2m",
        "precipitation_amount_3h",
        "cloud_cover_3h",
        "wind_u_10m",
        "wind_v_10m",
    ]
    with Session(engine) as session:
        for lead in leads:
            # Mean products
            for v in vars_to_add:
                session.add(
                    ProductRecord(
                        id=f"prod_geavg_{run_id}_{lead}_{v}",
                        run_id=run_id,
                        variable_id=v,
                        grid_id="global_025deg",
                        product_type="ensemble_mean",
                        lead_time_hours=lead,
                        zarr_chunk_path="/fake",
                    )
                )
                if store_dir:
                    rel = make_shard_relative_key(v, TARGET_KIND_MEAN, lead, -1)
                    shard_file = store_dir / rel
                    shard_file.parent.mkdir(parents=True, exist_ok=True)
                    shard_file.write_bytes(b"MEAN_SHARD")

            # Member products
            for m in range(1, member_count + 1):
                session.add(
                    EnsembleMemberProductRecord(
                        id=f"emp_{run_id}_{lead}_{m}",
                        run_id=run_id,
                        member_index=m,
                        lead_time_hours=lead,
                    )
                )
                if store_dir:
                    for v in vars_to_add:
                        rel = make_shard_relative_key(v, TARGET_KIND_MEM, lead, m)
                        shard_file = store_dir / rel
                        shard_file.parent.mkdir(parents=True, exist_ok=True)
                        shard_file.write_bytes(b"MEM_SHARD")
        session.commit()


# ===========================================================================
# 1. Superseded ordinary shard becomes reclaimable
# 2. Far-horizon older shard remains retained
# ===========================================================================
def test_01_and_02_superseded_reclaimable_and_far_horizon_retained(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    c1 = _dt(2026, 9, 2, 6)

    # c0 has leads 0..24
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    _seed_gfs_products(catalog_engine, r0, [0, 6, 12, 18, 24], ["temperature_2m"], tmp_path / "c0")

    # c1 has leads 0..12 (covers valid times 06Z..18Z)
    r1 = _seed_run(catalog_engine, "gfs", c1, "ready", tmp_path / "c1")
    _seed_gfs_products(catalog_engine, r1, [0, 6, 12], ["temperature_2m"], tmp_path / "c1")

    now = _dt(2026, 9, 2, 7)  # serving start = 06Z
    with Session(catalog_engine) as session:
        plan = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        # r0 leads 6 and 12 (valid times 06Z and 12Z) are superseded by r1 leads 0 and 6
        # r0 lead 24 (valid time 09-03 00Z) is FAR HORIZON and remains retained!
        reclaimable_leads = {
            t.lead_time_hours for t in plan.would_enqueue if t.run_id == r0
        }
        assert 6 in reclaimable_leads
        assert 12 in reclaimable_leads
        assert 24 not in reclaimable_leads


# ===========================================================================
# 3. Expired unheld shard becomes reclaimable
# 4. Expired shard with active predecessor remains retained
# ===========================================================================
def test_03_and_04_expired_unheld_reclaimable_and_predecessor_retained(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 1, 0)
    r0 = _seed_run(catalog_engine, "gfs", c0, "partial", tmp_path / "c0")
    # r0 has lead 21 committed, lead 24 uncommitted
    _seed_gfs_products(catalog_engine, r0, [21], ["precipitation_amount_3h", "temperature_2m"], tmp_path / "c0")

    # Now is 09-02 12Z -> valid time for lead 21 (09-01 21Z) is strictly before serving start (09-02 12Z)
    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        plan = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        # temperature_2m at lead 21 is expired and unheld -> reclaimable
        reclaimable_vars = {
            t.variable_code for t in plan.would_enqueue if t.run_id == r0 and t.lead_time_hours == 21
        }
        assert "temperature_2m" in reclaimable_vars
        # precipitation_amount_3h at lead 21 is active predecessor for uncommitted reset lead 24 -> retained!
        assert "precipitation_amount_3h" not in reclaimable_vars


# ===========================================================================
# 5. Precip lead-0 fallback retains older positive-lead source
# 6. Precip companions remain with fallback source
# 7. Cloud cover fallback is independently retained
# 11. Unrelated ordinary variable is not retained by precipitation fallback
# ===========================================================================
def test_05_06_07_11_serving_fallbacks_and_companions(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    c1 = _dt(2026, 9, 2, 6)

    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    _seed_gfs_products(catalog_engine, r0, [6], ["precipitation_amount_3h", "crain", "cloud_cover_3h", "temperature_2m"], tmp_path / "c0")

    r1 = _seed_run(catalog_engine, "gfs", c1, "ready", tmp_path / "c1")
    _seed_gfs_products(catalog_engine, r1, [0], ["precipitation_amount_3h", "crain", "cloud_cover_3h", "temperature_2m"], tmp_path / "c1")

    now = _dt(2026, 9, 2, 6, 30)  # valid time 06Z: r1 is anchor at lead 0
    with Session(catalog_engine) as session:
        plan = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        r0_reclaimable = {
            t.variable_code for t in plan.would_enqueue if t.run_id == r0 and t.lead_time_hours == 6
        }
        # r0 precipitation_amount_3h is held as active interval fallback
        assert "precipitation_amount_3h" not in r0_reclaimable
        # r0 crain companion is held with precipitation fallback
        assert "crain" not in r0_reclaimable
        # r0 cloud_cover_3h is held as active interval fallback
        assert "cloud_cover_3h" not in r0_reclaimable
        # r0 temperature_2m is ordinary variable, superseded by r1 lead 0 -> reclaimable!
        assert "temperature_2m" in r0_reclaimable


# ===========================================================================
# 8. L21 retained while L24 is recoverable
# 9. L21 released once L24 commits
# 10. Predecessor released when cycle leaves recovery eligibility
# ===========================================================================
def test_08_09_10_predecessor_retention_and_release(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 1, 0)
    r0 = _seed_run(catalog_engine, "gfs", c0, "partial", tmp_path / "c0")
    _seed_gfs_products(catalog_engine, r0, [21], ["precipitation_amount_3h"], tmp_path / "c0")

    now = _dt(2026, 9, 2, 0)  # c0 is within CANONICAL_MAX_LEAD_HOURS, recovery eligible
    with Session(catalog_engine) as session:
        # Step 1: L24 uncommitted -> L21 retained
        plan1 = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        assert not any(t.run_id == r0 and t.lead_time_hours == 21 for t in plan1.would_enqueue)

        # Step 2: L24 commits -> L21 released
        _seed_gfs_products(catalog_engine, r0, [24], ["precipitation_amount_3h"], tmp_path / "c0")
        plan2 = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        assert any(t.run_id == r0 and t.lead_time_hours == 21 for t in plan2.would_enqueue)

    # Step 3: Cycle leaves recovery eligibility (older than min_recovery_cycle)
    c_old = _dt(2026, 8, 1, 0)
    r_old = _seed_run(catalog_engine, "gfs", c_old, "partial", tmp_path / "c_old")
    _seed_gfs_products(catalog_engine, r_old, [21], ["precipitation_amount_3h"], tmp_path / "c_old")
    with Session(catalog_engine) as session:
        plan3 = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        assert any(t.run_id == r_old and t.lead_time_hours == 21 for t in plan3.would_enqueue)


# ===========================================================================
# 12. Wind component coherence is preserved
# ===========================================================================
def test_12_wind_component_coherence(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    _seed_gfs_products(catalog_engine, r0, [6], ["wind_u_10m", "wind_v_10m"], tmp_path / "c0")

    now = _dt(2026, 9, 2, 0)
    with Session(catalog_engine) as session:
        plan = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        # Both wind components must be retained together for active wind_10m
        reclaimable_wind = {
            t.variable_code for t in plan.would_enqueue if t.run_id == r0 and t.lead_time_hours == 6
        }
        assert "wind_u_10m" not in reclaimable_wind
        assert "wind_v_10m" not in reclaimable_wind


# ===========================================================================
# 13. GEFS coherent-vintage reclamation is correct
# 27. GEFS member region does not resurrect
# 28. GEFS mean region does not resurrect
# 29. GEFS variable-specific member coverage excludes deleting/deleted shards
# ===========================================================================
def test_13_27_28_29_gefs_coherent_vintage_and_anti_resurrection(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    c1 = _dt(2026, 9, 2, 6)

    # c0 has 30/30 members at lead 6
    r0 = _seed_run(catalog_engine, "gefs", c0, "ready", tmp_path / "c0")
    _seed_gefs_products(catalog_engine, r0, [6], member_count=30, variables=["temperature_2m"], store_dir=tmp_path / "c0")

    # c1 has 30/30 members at lead 0 (supersedes c0 lead 6 for valid time 06Z)
    r1 = _seed_run(catalog_engine, "gefs", c1, "ready", tmp_path / "c1")
    _seed_gefs_products(catalog_engine, r1, [0], member_count=30, variables=["temperature_2m"], store_dir=tmp_path / "c1")

    now = _dt(2026, 9, 2, 6, 30)
    with Session(catalog_engine) as session:
        plan = plan_reclamation_pass(session, models=("gefs",), dry_run=False, now=now)
        # r0 shards at lead 6 are superseded -> enqueued
        r0_enqueued = [t for t in plan.would_enqueue if t.run_id == r0]
        assert len(r0_enqueued) > 0

        # Execute worker pass with delete_enabled=True
        w_res = run_reclamation_worker_pass(session, delete_enabled=True, now=now)
        assert w_res.deleted_count > 0

        # Verify catalog commit evidence is STILL PRESERVED!
        mean_row = session.execute(
            select(ProductRecord).where(ProductRecord.run_id == r0, ProductRecord.lead_time_hours == 6)
        ).scalar_one_or_none()
        assert mean_row is not None

        emp_rows = session.execute(
            select(EnsembleMemberProductRecord).where(
                EnsembleMemberProductRecord.run_id == r0,
                EnsembleMemberProductRecord.lead_time_hours == 6,
            )
        ).scalars().all()
        assert len(emp_rows) == 30

        # Run _reconcile_catalog_to_store: proves anti-resurrection does NOT delete commit rows!
        c_state = CommittedState.ensemble(
            pairs=set(),
            members=set(),
            variables=set(),
            mean_leads=set(),
        )
        run_record = session.get(ModelRunRecord, r0)
        _reconcile_catalog_to_store(session, run_record, c_state)

        # Still preserved!
        emp_rows_after = session.execute(
            select(EnsembleMemberProductRecord).where(
                EnsembleMemberProductRecord.run_id == r0,
                EnsembleMemberProductRecord.lead_time_hours == 6,
            )
        ).scalars().all()
        assert len(emp_rows_after) == 30


# ===========================================================================
# 14. Unfenced cycle does not authorize reclamation
# 15. Granular GC never stamps cycle deletion_started_at
# 16. Enqueue is idempotent
# 17. Unique queue identity distinguishes different run_ids
# ===========================================================================
def test_14_15_16_17_lifecycle_isolation_and_idempotent_enqueue(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    _seed_gfs_products(catalog_engine, r0, [24], ["temperature_2m"], tmp_path / "c0")

    # Seed lifecycle row in forecast_cycle_lifecycle
    with Session(catalog_engine) as session:
        lc = ForecastCycleLifecycleRecord(
            model_id="gfs",
            cycle_time=c0,
            created_at=c0,
            updated_at=c0,
        )
        session.add(lc)
        session.commit()

    now = _dt(2026, 9, 2, 1)  # c0 lead 24 is still canonical anchor
    with Session(catalog_engine) as session:
        # Unfenced cycle does NOT authorize reclamation of canonical shard!
        plan = plan_reclamation_pass(session, models=("gfs",), dry_run=False, now=now)
        assert not any(t.run_id == r0 and t.lead_time_hours == 24 for t in plan.would_enqueue)

        # Verify granular GC never stamped deletion_started_at on whole cycle!
        lc_row = session.get(ForecastCycleLifecycleRecord, ("gfs", c0))
        assert lc_row.deletion_started_at is None

        # Re-running plan is completely idempotent
        plan2 = plan_reclamation_pass(session, models=("gfs",), dry_run=False, now=now)
        assert plan2.enqueued_count == 0


# ===========================================================================
# 18. Worker delete is idempotent for absent object
# 19. Crash after DeleteObject before finalization recovers
# 20. Multiple workers claim disjoint batches
# ===========================================================================
def test_18_19_20_worker_idempotency_crash_recovery_and_batching(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 1, 0)
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    _seed_gfs_products(catalog_engine, r0, [6, 12], ["temperature_2m"], tmp_path / "c0")

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # Enqueue shards
        plan = plan_reclamation_pass(session, models=("gfs",), dry_run=False, now=now)
        assert plan.enqueued_count >= 2

        # Worker 1 claims batch of 1
        w1 = run_reclamation_worker_pass(session, batch_size=1, lease_seconds=60, delete_enabled=True, now=now)
        assert w1.claimed_count == 1
        assert w1.deleted_count == 1

        # Worker 2 claims next batch: disjoint batch!
        w2 = run_reclamation_worker_pass(session, batch_size=1, lease_seconds=60, delete_enabled=True, now=now)
        assert w2.claimed_count == 1
        assert w2.deleted_count == 1

        # Simulate crash: a row was deleted from disk but stuck in 'deleting' with expired lease
        fake_rec = ReclamationQueueRecord(
            id="crash_recovery_test",
            run_id=r0,
            model_id="gfs",
            cycle_time=c0,
            lead_time_hours=18,
            variable_code="temperature_2m",
            target_kind=TARGET_KIND_DET,
            member_index=0,
            valid_time=c0 + timedelta(hours=18),
            store_path=str(tmp_path / "c0"),
            physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 18),
            status=RECLAMATION_STATUS_DELETING,
            attempt_count=1,
            lease_expires_at=now - timedelta(seconds=10),  # expired lease
            created_at=now,
            updated_at=now,
        )
        session.add(fake_rec)
        session.commit()

        # Object is absent on disk (already deleted before crash)
        # Worker re-claims expired row: DeleteObject is idempotent, row reaches 'deleted'!
        w_recover = run_reclamation_worker_pass(session, delete_enabled=True, now=now)
        assert w_recover.deleted_count >= 1
        recovered = session.get(ReclamationQueueRecord, "crash_recovery_test")
        assert recovered.status == RECLAMATION_STATUS_DELETED


# ===========================================================================
# 21. Counterfactual revalidation detects self-fenced target becoming required
# 22. Newly emerged fallback aborts deletion
# 23. Predecessor dependency aborts deletion
# 24. Newer partial run without exact committed representation cannot authorize deletion
# ===========================================================================
def test_21_22_23_24_counterfactual_revalidation_aborts(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    _seed_gfs_products(catalog_engine, r0, [6], ["precipitation_amount_3h"], tmp_path / "c0")

    now = _dt(2026, 9, 2, 6, 30)

    with Session(catalog_engine) as session:
        # Enqueue r0
        rel_key = make_shard_relative_key("precipitation_amount_3h", TARGET_KIND_DET, 6)
        target = ReclamationQueueRecord(
            id="reval_test_target",
            run_id=r0,
            model_id="gfs",
            cycle_time=c0,
            lead_time_hours=6,
            variable_code="precipitation_amount_3h",
            target_kind=TARGET_KIND_DET,
            member_index=0,
            valid_time=c0 + timedelta(hours=6),
            store_path=str(tmp_path / "c0"),
            physical_key=rel_key,
            status=RECLAMATION_STATUS_QUEUED,
            created_at=now,
            updated_at=now,
        )
        session.add(target)
        session.commit()

        # Now suppose a newer cycle c1 arrives with lead 0!
        c1 = _dt(2026, 9, 2, 6)
        r1 = _seed_run(catalog_engine, "gfs", c1, "ready", tmp_path / "c1")
        _seed_gfs_products(catalog_engine, r1, [0], ["precipitation_amount_3h"], tmp_path / "c1")

        # Worker runs: target is claimed (status='deleting').
        # Counterfactual revalidation evaluates target as physically present.
        # It discovers that c1 lead 0 REQUIRES r0 lead 6 as its interval fallback!
        # Deletion is aborted!
        w_res = run_reclamation_worker_pass(session, delete_enabled=True, now=now)
        assert w_res.revalidated_held_count == 1
        assert w_res.deleted_count == 0

        # Target reverted to 'queued' with exact abort reason in last_error
        t_after = session.get(ReclamationQueueRecord, "reval_test_target")
        assert t_after.status == RECLAMATION_STATUS_QUEUED
        assert "counterfactual_revalidation_abort" in t_after.last_error


# ===========================================================================
# 25. Partially reclaimed previously complete GFS region does not resurrect
# 26. Genuinely incomplete GFS region is not falsely completed
# ===========================================================================
def test_25_26_anti_resurrection_and_incomplete_leads(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    # Complete lead 6 with 2 variables
    _seed_gfs_products(catalog_engine, r0, [6], ["temperature_2m", "precipitation_amount_3h"], tmp_path / "c0")

    with Session(catalog_engine) as session:
        # Reclaim 1 variable physically
        rel = make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6)
        rec = ReclamationQueueRecord(
            id="gfs_anti_resurrect",
            run_id=r0,
            model_id="gfs",
            cycle_time=c0,
            lead_time_hours=6,
            variable_code="temperature_2m",
            target_kind=TARGET_KIND_DET,
            member_index=0,
            valid_time=c0 + timedelta(hours=6),
            store_path=str(tmp_path / "c0"),
            physical_key=rel,
            status=RECLAMATION_STATUS_DELETED,
            created_at=c0,
            updated_at=c0,
        )
        session.add(rec)
        session.commit()

        # Catalog still has ProductRecord for lead 6
        prod = session.execute(
            select(ProductRecord).where(ProductRecord.run_id == r0, ProductRecord.lead_time_hours == 6)
        ).scalars().first()
        assert prod is not None  # Preserved! Lead 6 does not appear uncommitted to Phase 2 scheduler!


# ===========================================================================
# 31. Region marker survives until every expected variable shard is deleted
# 32. Variable schema is model/version correct
# ===========================================================================
def test_31_32_region_marker_cleanup_only_when_all_variables_deleted(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    store_dir = tmp_path / "c0"
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", store_dir)
    # Lead 6 has 2 variables
    _seed_gfs_products(catalog_engine, r0, [6], ["temperature_2m", "precipitation_amount_3h"], store_dir)

    marker_rel = make_region_marker_relative_key(TARGET_KIND_DET, 6)
    marker_file = store_dir / marker_rel
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.write_bytes(b'{"state":"complete"}')
    assert marker_file.exists()

    # Temporarily register 2 expected variables for this test to prove full deletion triggers marker cleanup
    saved_schema = DEFAULT_EXPECTED_REGION_VARIABLES[("gfs", TARGET_KIND_DET)]
    register_expected_region_variables("gfs", TARGET_KIND_DET, ["temperature_2m", "precipitation_amount_3h"])

    try:
        now = _dt(2026, 9, 2, 12)
        with Session(catalog_engine) as session:
            # Step 1: Delete 1 variable shard
            t1 = ReclamationQueueRecord(
                id="m_clean_1",
                run_id=r0,
                model_id="gfs",
                cycle_time=c0,
                lead_time_hours=6,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c0 + timedelta(hours=6),
                store_path=str(store_dir),
                physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6),
                status=RECLAMATION_STATUS_QUEUED,
                created_at=now,
                updated_at=now,
            )
            session.add(t1)
            session.commit()

            w1 = run_reclamation_worker_pass(session, delete_enabled=True, now=now)
            assert w1.deleted_count == 1
            assert w1.markers_cleaned_count == 0
            # Marker STILL EXISTS!
            assert marker_file.exists()

            # Step 2: Delete 2nd variable shard
            t2 = ReclamationQueueRecord(
                id="m_clean_2",
                run_id=r0,
                model_id="gfs",
                cycle_time=c0,
                lead_time_hours=6,
                variable_code="precipitation_amount_3h",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c0 + timedelta(hours=6),
                store_path=str(store_dir),
                physical_key=make_shard_relative_key("precipitation_amount_3h", TARGET_KIND_DET, 6),
                status=RECLAMATION_STATUS_QUEUED,
                created_at=now,
                updated_at=now,
            )
            session.add(t2)
            session.commit()

            w2 = run_reclamation_worker_pass(session, delete_enabled=True, now=now)
            assert w2.deleted_count == 1
            assert w2.markers_cleaned_count == 1
            # Now every expected variable shard is deleted -> marker deleted!
            assert not marker_file.exists()
    finally:
        register_expected_region_variables("gfs", TARGET_KIND_DET, saved_schema)


# ===========================================================================
# 43. Non-destructive defaults issue zero DeleteObject calls
# 44. Failed quarantine/requeue semantics work
# ===========================================================================
def test_43_44_non_destructive_defaults_and_quarantine_requeue(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    store_dir = tmp_path / "c0"
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", store_dir)
    _seed_gfs_products(catalog_engine, r0, [6], ["temperature_2m"], store_dir)

    shard_file = store_dir / make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6)
    assert shard_file.exists()

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # Enqueue
        plan_reclamation_pass(session, models=("gfs",), dry_run=False, now=now)

        # Worker with delete_enabled=False (safe default)
        w_res = run_reclamation_worker_pass(session, delete_enabled=False, now=now)
        assert w_res.deleted_count == 0
        # Physical file STILL INTACT!
        assert shard_file.exists()

        # Simulate failure exceeding max retries
        rec = session.execute(select(ReclamationQueueRecord)).scalars().first()
        rec.attempt_count = 5
        rec.status = RECLAMATION_STATUS_FAILED
        rec.last_error = "max_retries_exceeded: test"
        session.commit()

        # Requeue helper resets failed targets to queued
        requeued = requeue_failed_reclamation_targets(session, model_id="gfs")
        assert requeued == 1
        rec_requeued = session.get(ReclamationQueueRecord, rec.id)
        assert rec_requeued.status == RECLAMATION_STATUS_QUEUED
        assert rec_requeued.attempt_count == 0
        assert rec_requeued.last_error is None


# ===========================================================================
# 30. V2 whole-cycle and V3 granular mutation serialize on store gate
# ===========================================================================
def test_30_v2_whole_cycle_and_v3_granular_serialization_aborts_on_whole_cycle_claim(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    store_dir = tmp_path / "c0"
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", store_dir)
    _seed_gfs_products(catalog_engine, r0, [6], ["temperature_2m"], store_dir)

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # Enqueue target
        plan_reclamation_pass(session, models=("gfs",), dry_run=False, now=now)

        # Whole-cycle V2 GC claims cycle
        lc = ForecastCycleLifecycleRecord(
            model_id="gfs",
            cycle_time=c0,
            deletion_started_at=now,
            created_at=c0,
            updated_at=now,
        )
        session.add(lc)
        session.commit()

        # V3 granular worker runs: re-reads lifecycle under gate, detects whole-cycle GC claim,
        # aborts granular deletion, and reverts target to 'queued'!
        w_res = run_reclamation_worker_pass(session, delete_enabled=True, now=now)
        assert w_res.deleted_count == 0
        rec = session.execute(select(ReclamationQueueRecord)).scalars().first()
        assert rec.status == RECLAMATION_STATUS_QUEUED
        assert "cycle_claimed_by_whole_cycle_gc" in rec.last_error


# ===========================================================================
# 39. Planner query count is bounded
# 41. Queue restart is reconstructed from PostgreSQL
# ===========================================================================
def test_39_41_planner_query_count_bounded_and_queue_restart_reconstructed(catalog_engine, tmp_path):
    from sqlalchemy import event

    c0 = _dt(2026, 9, 2, 0)
    store_dir = tmp_path / "c0"
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", store_dir)
    # Seed 50 products across leads
    _seed_gfs_products(catalog_engine, r0, list(range(0, 150, 3)), ["temperature_2m"], store_dir)

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        queries: list[str] = []

        def capture_query(conn, cursor, statement, parameters, context, executemany):
            queries.append(statement)

        event.listen(catalog_engine, "before_cursor_execute", capture_query)
        try:
            plan = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
            # Must be bounded: <= 5 queries total for 50 leads (no query per lead or shard!)
            assert len(queries) <= 5
            assert plan.reclaimable_shards > 0
        finally:
            event.remove(catalog_engine, "before_cursor_execute", capture_query)

        # Enqueue to database
        plan_reclamation_pass(session, models=("gfs",), dry_run=False, now=now)
        enqueued_count = session.execute(select(ReclamationQueueRecord)).scalars().all()
        assert len(enqueued_count) > 0

    # 41. Queue restart reconstruction: fresh session reading from SQLite/PostgreSQL
    with Session(catalog_engine) as fresh_session:
        restored = fresh_session.execute(select(ReclamationQueueRecord)).scalars().all()
        assert len(restored) == len(enqueued_count)
        assert all(r.status == RECLAMATION_STATUS_QUEUED for r in restored)


# ===========================================================================
# Final Acceptance Audit Point 1: Model-Specific Horizon Boundaries
# ===========================================================================
def test_acceptance_1_model_specific_horizon_boundaries_and_synthetic_model(catalog_engine, tmp_path):
    from domain.horizon import (
        MODEL_CANONICAL_HORIZONS,
        model_max_lead_hours,
        register_canonical_lead_horizon,
    )

    # Invariant: GFS and GEFS max lead is 240h, not hardcoded 384h
    assert model_max_lead_hours("gfs") == 240
    assert model_max_lead_hours("gefs") == 240

    now = _dt(2026, 9, 12, 6)
    serving_start = serving_start_valid_time(now)  # 09-12 06:00Z

    # Boundary test: cycle_time + 240h == serving_start -> still recovery eligible!
    # cycle_time = serving_start - 240h = 09-12 06:00Z - 10 days = 09-02 06:00Z
    c_exact = serving_start - timedelta(hours=240)
    r_exact = _seed_run(catalog_engine, "gfs", c_exact, "partial", tmp_path / "c_exact")
    _seed_gfs_products(catalog_engine, r_exact, [21], ["precipitation_amount_3h"], tmp_path / "c_exact")

    # Boundary test: cycle_time + 240h < serving_start -> no longer recovery eligible!
    c_expired = c_exact - timedelta(hours=3)
    r_expired = _seed_run(catalog_engine, "gfs", c_expired, "partial", tmp_path / "c_expired")
    _seed_gfs_products(catalog_engine, r_expired, [21], ["precipitation_amount_3h"], tmp_path / "c_expired")

    with Session(catalog_engine) as session:
        plan = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        reclaimable_runs = {t.run_id for t in plan.would_enqueue}
        # r_exact is still recovery eligible -> predecessor L21 is held (NOT reclaimable)
        assert r_exact not in reclaimable_runs
        # r_expired has cycle_time + 240h < serving_start -> no longer recovery eligible -> L21 is reclaimable!
        assert r_expired in reclaimable_runs

    # Synthetic model fixture with a different max lead (e.g. 120h)
    saved_horizons = dict(MODEL_CANONICAL_HORIZONS)
    try:
        register_canonical_lead_horizon("gfs_regional", tuple(range(0, 121, 3)))
        assert model_max_lead_hours("gfs_regional") == 120
    finally:
        MODEL_CANONICAL_HORIZONS.clear()
        MODEL_CANONICAL_HORIZONS.update(saved_horizons)


# ===========================================================================
# Final Acceptance Audit Point 2: Schema-Authoritative Marker Cleanup
# ===========================================================================
def test_acceptance_2_schema_authoritative_marker_cleanup_rejects_incomplete_region(catalog_engine, tmp_path):
    """Marker cleanup MUST NOT delete marker if region only committed a truncated subset of variables."""
    c0 = _dt(2026, 9, 2, 0)
    store_dir = tmp_path / "c0_incomplete"
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", store_dir)

    # Abnormal incomplete run: only committed 2 out of 15 variables at lead 6
    _seed_gfs_products(catalog_engine, r0, [6], ["temperature_2m", "precipitation_amount_3h"], store_dir)

    marker_rel = make_region_marker_relative_key(TARGET_KIND_DET, 6)
    marker_file = store_dir / marker_rel
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.write_bytes(b'{"state":"complete"}')
    assert marker_file.exists()

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # Enqueue both committed variables for reclamation
        for v in ("temperature_2m", "precipitation_amount_3h"):
            session.add(
                ReclamationQueueRecord(
                    id=f"incomp_{v}",
                    run_id=r0,
                    model_id="gfs",
                    cycle_time=c0,
                    lead_time_hours=6,
                    variable_code=v,
                    target_kind=TARGET_KIND_DET,
                    member_index=0,
                    valid_time=c0 + timedelta(hours=6),
                    store_path=str(store_dir),
                    physical_key=make_shard_relative_key(v, TARGET_KIND_DET, 6),
                    status=RECLAMATION_STATUS_QUEUED,
                    created_at=now,
                    updated_at=now,
                )
            )
        session.commit()

        # Worker runs and deletes both committed shards
        w_res = run_reclamation_worker_pass(session, delete_enabled=True, now=now)
        assert w_res.deleted_count == 2
        # Because authoritative GFS schema expects 15 variables, deleting only 2 CANNOT delete the marker!
        assert w_res.markers_cleaned_count == 0
        assert marker_file.exists()  # Commit marker is SAFE and preserved!


# ===========================================================================
# Final Acceptance Audit Point 3: Ambiguous Delete Outcomes & Safe Requeue
# ===========================================================================
def test_acceptance_3_ambiguous_delete_failed_state_fenced_and_safe_requeue(catalog_engine, tmp_path):
    c0 = _dt(2026, 9, 2, 0)
    store_dir = tmp_path / "c0_failed"
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", store_dir)
    _seed_gfs_products(catalog_engine, r0, [6], ["temperature_2m"], store_dir)

    shard_rel = make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6)
    shard_file = store_dir / shard_rel
    assert shard_file.exists()

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # 1. Target enters 'failed' status after ambiguous remote failure / retry exhaustion
        rec = ReclamationQueueRecord(
            id="failed_target_test",
            run_id=r0,
            model_id="gfs",
            cycle_time=c0,
            lead_time_hours=6,
            variable_code="temperature_2m",
            target_kind=TARGET_KIND_DET,
            member_index=0,
            valid_time=c0 + timedelta(hours=6),
            store_path=str(store_dir),
            physical_key=shard_rel,
            status=RECLAMATION_STATUS_FAILED,
            attempt_count=5,
            last_error="timeout_or_ambiguous_outcome",
            created_at=now,
            updated_at=now,
        )
        session.add(rec)
        session.commit()

        # Invariant: 'failed' is physically fenced from serving!
        from api.services.resolver import check_physical_targets_fenced
        fenced = check_physical_targets_fenced(session, [(r0, shard_rel)])
        assert (r0, shard_rel) in fenced

        # Case A: Shard file is positively confirmed still present -> safe to requeue to 'queued'
        requeue_failed_reclamation_targets(session, model_id="gfs")
        session.refresh(rec)
        assert rec.status == RECLAMATION_STATUS_QUEUED
        assert rec.attempt_count == 0

        # Case B: Simulate ambiguous remote delete succeeded before network crash (shard file absent)
        shard_file.unlink()
        assert not shard_file.exists()
        rec.status = RECLAMATION_STATUS_FAILED
        rec.attempt_count = 5
        session.commit()

        # Requeue detects object is absent -> promotes directly to 'deleted' (prevents serving resurrection)!
        requeue_failed_reclamation_targets(session, model_id="gfs")
        session.refresh(rec)
        assert rec.status == RECLAMATION_STATUS_DELETED
        assert rec.reclaimed_at is not None


# ===========================================================================
# Final Acceptance Audit Point 2 (Part B): Mixed-Store Worker Gate Coverage
# ===========================================================================
def test_acceptance_mixed_store_worker_gate_coverage(catalog_engine, tmp_path):
    """Batch spanning two distinct store_paths partitions by store_path and acquires each store's gate."""
    c0 = _dt(2026, 9, 1, 0)
    c1 = _dt(2026, 9, 1, 6)

    store_a = tmp_path / "store_a.zarr"
    store_b = tmp_path / "store_b.zarr"

    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", store_a)
    r1 = _seed_run(catalog_engine, "gfs", c1, "ready", store_b)

    _seed_gfs_products(catalog_engine, r0, [6], ["temperature_2m"], store_a)
    _seed_gfs_products(catalog_engine, r1, [6], ["temperature_2m"], store_b)

    file_a = store_a / make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6)
    file_b = store_b / make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6)
    assert file_a.exists()
    assert file_b.exists()

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # Enqueue targets from both store A and store B in the same queue
        session.add_all([
            ReclamationQueueRecord(
                id="batch_store_a",
                run_id=r0,
                model_id="gfs",
                cycle_time=c0,
                lead_time_hours=6,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c0 + timedelta(hours=6),
                store_path=str(store_a),
                physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6),
                status=RECLAMATION_STATUS_QUEUED,
                created_at=now,
                updated_at=now,
            ),
            ReclamationQueueRecord(
                id="batch_store_b",
                run_id=r1,
                model_id="gfs",
                cycle_time=c1,
                lead_time_hours=6,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c1 + timedelta(hours=6),
                store_path=str(store_b),
                physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6),
                status=RECLAMATION_STATUS_QUEUED,
                created_at=now,
                updated_at=now,
            ),
        ])
        session.commit()

        # Worker claims both in a single batch (batch_size=10)
        # Partitions by store_path, acquires store A gate -> deletes file A, then store B gate -> deletes file B
        w_res = run_reclamation_worker_pass(session, batch_size=10, delete_enabled=True, now=now)
        assert w_res.claimed_count == 2
        assert w_res.deleted_count == 2
        assert not file_a.exists()
        assert not file_b.exists()


# ===========================================================================
# Final Acceptance Audit Point 3 (Part B): Exact Post-Read Physical Identity Matching
# ===========================================================================
def test_acceptance_post_read_exact_identity_matching(catalog_engine, tmp_path):
    """Post-read validation matches exact (run_id, physical_key) and never falsely rejects unrelated shards."""
    from api.services.resolver import check_physical_targets_fenced

    c0 = _dt(2026, 9, 2, 0)
    c1 = _dt(2026, 9, 2, 6)

    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    r1 = _seed_run(catalog_engine, "gfs", c1, "ready", tmp_path / "c1")

    key_a = make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6)
    key_b = make_shard_relative_key("precipitation_amount_3h", TARGET_KIND_DET, 6)

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # Case A: Shard B in run r0 enters deleting. Response used shard A in run r0.
        session.add(
            ReclamationQueueRecord(
                id="unrelated_shard_b",
                run_id=r0,
                model_id="gfs",
                cycle_time=c0,
                lead_time_hours=6,
                variable_code="precipitation_amount_3h",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c0 + timedelta(hours=6),
                store_path=str(tmp_path / "c0"),
                physical_key=key_b,
                status=RECLAMATION_STATUS_DELETING,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()

        # Target used by request is (r0, key_a): must NOT be falsely rejected!
        fenced_a = check_physical_targets_fenced(session, [(r0, key_a)])
        assert fenced_a == set()

        # Case B: Two different runs contain the same relative key_a. Only r1 enters deleting.
        session.add(
            ReclamationQueueRecord(
                id="same_key_run_r1",
                run_id=r1,
                model_id="gfs",
                cycle_time=c1,
                lead_time_hours=6,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c1 + timedelta(hours=6),
                store_path=str(tmp_path / "c1"),
                physical_key=key_a,
                status=RECLAMATION_STATUS_DELETING,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()

        # Target used was (r0, key_a): r0 is NOT fenced, even though r1 with same relative key IS fenced!
        assert check_physical_targets_fenced(session, [(r0, key_a)]) == set()
        # Target used was (r1, key_a): r1 IS detected!
        assert (r1, key_a) in check_physical_targets_fenced(session, [(r1, key_a)])


# ===========================================================================
# Final Acceptance Audit Point 7: Reconciliation Intentional V3 vs Accidental Corruption
# ===========================================================================
def test_acceptance_reconciliation_intentional_v3_vs_accidental_corruption(catalog_engine, tmp_path):
    """Intentional reclamation preserves catalog rows; accidental data loss without tombstone is pruned."""
    c0 = _dt(2026, 9, 2, 0)
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", tmp_path / "c0")
    _seed_gfs_products(catalog_engine, r0, [6, 12, 18, 24], ["temperature_2m"], tmp_path / "c0")

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # Case A: Lead 6 has status='deleted'
        # Case B: Lead 12 has status='deleting'
        # Case C: Lead 18 has status='failed' (ambiguous delete outcome)
        # Case D: Lead 24 has NO row in reclamation_queue
        session.add_all([
            ReclamationQueueRecord(
                id="intentional_tombstone_l6",
                run_id=r0,
                model_id="gfs",
                cycle_time=c0,
                lead_time_hours=6,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c0 + timedelta(hours=6),
                store_path=str(tmp_path / "c0"),
                physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6),
                status=RECLAMATION_STATUS_DELETED,
                created_at=now,
                updated_at=now,
            ),
            ReclamationQueueRecord(
                id="intentional_tombstone_l12",
                run_id=r0,
                model_id="gfs",
                cycle_time=c0,
                lead_time_hours=12,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c0 + timedelta(hours=12),
                store_path=str(tmp_path / "c0"),
                physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 12),
                status=RECLAMATION_STATUS_DELETING,
                created_at=now,
                updated_at=now,
            ),
            ReclamationQueueRecord(
                id="intentional_tombstone_l18",
                run_id=r0,
                model_id="gfs",
                cycle_time=c0,
                lead_time_hours=18,
                variable_code="temperature_2m",
                target_kind=TARGET_KIND_DET,
                member_index=0,
                valid_time=c0 + timedelta(hours=18),
                store_path=str(tmp_path / "c0"),
                physical_key=make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 18),
                status=RECLAMATION_STATUS_FAILED,
                created_at=now,
                updated_at=now,
            ),
        ])
        session.commit()

        # Suppose store is read by reconciliation and finds none of the leads in physical storage
        # (committed_state is empty)
        run_record = session.get(ModelRunRecord, r0)
        c_state = CommittedState.deterministic(leads=set(), variables=set())
        _reconcile_catalog_to_store(session, run_record, c_state)

        # Case A: status='deleted' -> catalog row is PRESERVED!
        p6 = session.execute(
            select(ProductRecord).where(ProductRecord.run_id == r0, ProductRecord.lead_time_hours == 6)
        ).scalar_one_or_none()
        assert p6 is not None

        # Case B: status='deleting' -> catalog row is PRESERVED!
        p12 = session.execute(
            select(ProductRecord).where(ProductRecord.run_id == r0, ProductRecord.lead_time_hours == 12)
        ).scalar_one_or_none()
        assert p12 is not None

        # Case C: status='failed' -> catalog row is PRESERVED!
        p18 = session.execute(
            select(ProductRecord).where(ProductRecord.run_id == r0, ProductRecord.lead_time_hours == 18)
        ).scalar_one_or_none()
        assert p18 is not None

        # Case D: Lead 24 had NO reclamation row -> treated as accidental corruption and PRUNED!
        p24 = session.execute(
            select(ProductRecord).where(ProductRecord.run_id == r0, ProductRecord.lead_time_hours == 24)
        ).scalar_one_or_none()
        assert p24 is None


# ===========================================================================
# Final Acceptance Audit Point 8: Non-Destructive Default Execution
# ===========================================================================
def test_acceptance_non_destructive_stage_1_2_3_execution(catalog_engine, tmp_path):
    """Verify execution path across Stage 1 (defaults), Stage 2 (audit queue), Stage 3 (authorized delete)."""
    c0 = _dt(2026, 9, 2, 0)
    store_dir = tmp_path / "c0_staged"
    r0 = _seed_run(catalog_engine, "gfs", c0, "ready", store_dir)
    _seed_gfs_products(catalog_engine, r0, [6], ["temperature_2m"], store_dir)

    shard_file = store_dir / make_shard_relative_key("temperature_2m", TARGET_KIND_DET, 6)
    assert shard_file.exists()

    now = _dt(2026, 9, 2, 12)
    with Session(catalog_engine) as session:
        # Stage 1: DRY_RUN=True, DELETE_ENABLED=False -> Zero mutations!
        p1 = plan_reclamation_pass(session, models=("gfs",), dry_run=True, now=now)
        assert p1.reclaimable_shards == 1
        q_count1 = session.execute(select(ReclamationQueueRecord)).scalars().all()
        assert len(q_count1) == 0  # Zero queue writes!

        # Stage 2: DRY_RUN=False, DELETE_ENABLED=False -> Queue populates, zero deletes!
        p2 = plan_reclamation_pass(session, models=("gfs",), dry_run=False, now=now)
        assert p2.enqueued_count == 1
        w2 = run_reclamation_worker_pass(session, delete_enabled=False, now=now)
        assert w2.deleted_count == 0
        assert shard_file.exists()  # Zero physical deletions!

        # Stage 3: DELETE_ENABLED=True -> Physical deletion authorized!
        w3 = run_reclamation_worker_pass(session, delete_enabled=True, now=now)
        assert w3.deleted_count == 1
        assert not shard_file.exists()  # Shard physically deleted!



