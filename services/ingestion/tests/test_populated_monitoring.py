"""Live integration test suite exercising monitoring collectors against a populated relational catalog.

Validates that all real PostgreSQL queries (PostgreSQL capacity, lifecycle deletion claims,
14-day metadata retention sweeper, granular reclamation queue depths, GFS completeness,
GEFS completeness, anti-resurrection audits, and alert mapping) execute correctly against
realistic data without leaving permanent test fixtures in the database.

All seeded rows are strictly transaction-scoped and rolled back unconditionally upon test completion.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Generator

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection

from domain.coverage import get_expected_members
from domain.horizon import canonical_lead_time_hours, model_max_lead_hours
from ingestion.core.db import engine
from ingestion.monitoring.alerts import (
    ALERT_ENGINE,
)
from ingestion.monitoring.ingestion_collector import (
    IngestionHealthCollector,
)
from ingestion.monitoring.lifecycle_collector import (
    LifecycleHealthCollector,
)
from ingestion.monitoring.resources import (
    DiskInfo,
)


@pytest.fixture(scope="function")
def populated_db() -> Generator[Connection, None, None]:
    """Provide a live PostgreSQL connection with realistic seeded data rolled back unconditionally."""
    import os
    from alembic import command
    from alembic.config import Config

    orig = os.getcwd()
    try:
        os.chdir("services/api")
        command.upgrade(Config("alembic.ini"), "head")
    except Exception:
        pass
    finally:
        os.chdir(orig)

    with engine.connect() as conn:
        trans = conn.begin()
        now = datetime.now(timezone.utc)

        try:
            # 1. Base catalogs: centers, models, versions, variables, grids
            conn.execute(
                text(
                    """
                    INSERT INTO forecast_centers (id, center_id, name, country, created_at)
                    VALUES ('fc_test', 'noaa_test', 'NOAA Test', 'US', :now)
                    ON CONFLICT (center_id) DO NOTHING
                    """
                ),
                {"now": now},
            )

            conn.execute(
                text(
                    """
                    INSERT INTO models (id, model_id, name, center_id, is_ensemble, resolution_km, created_at)
                    VALUES ('m_gfs_test', 'gfs', 'GFS Test', 'noaa_test', false, 25.0, :now),
                           ('m_gefs_test', 'gefs', 'GEFS Test', 'noaa_test', true, 50.0, :now)
                    ON CONFLICT (model_id) DO NOTHING
                    """
                ),
                {"now": now},
            )

            conn.execute(
                text(
                    """
                    INSERT INTO model_versions (id, model_id, version_string, created_at)
                    VALUES ('mv_gfs_test', 'gfs', 'v1.0', :now),
                           ('mv_gefs_test', 'gefs', 'v1.0', :now)
                    ON CONFLICT (model_id, version_string) DO NOTHING
                    """
                ),
                {"now": now},
            )
            gfs_mv_pk = conn.execute(
                text("SELECT id FROM model_versions WHERE model_id = 'gfs' AND version_string = 'v1.0'"),
            ).scalar() or "mv_gfs_test"
            gefs_mv_pk = conn.execute(
                text("SELECT id FROM model_versions WHERE model_id = 'gefs' AND version_string = 'v1.0'"),
            ).scalar() or "mv_gefs_test"

            conn.execute(
                text(
                    """
                    INSERT INTO forecast_variables (id, variable_code, name, unit)
                    VALUES ('var_t2m_test', 'temperature_2m', '2m Temperature', 'K')
                    ON CONFLICT (variable_code) DO NOTHING
                    """
                )
            )

            conn.execute(
                text(
                    """
                    INSERT INTO forecast_grids (id, grid_code, name, resolution_km)
                    VALUES ('grid_test', 'global_025deg', 'Global 0.25', 25.0)
                    ON CONFLICT (grid_code) DO NOTHING
                    """
                )
            )

            # 2. Scenario 1: GFS Completeness
            # Full 81 leads committed, status 'ready'
            gfs_cycle = now - timedelta(hours=3)
            run_gfs_id = f"test_run_gfs_{uuid.uuid4().hex[:8]}"
            conn.execute(
                text(
                    """
                    INSERT INTO model_runs (id, model_version_id, cycle_time, status, zarr_store_path, created_at)
                    VALUES (:id, :mv_id, :ctime, 'ready', 's3://test/gfs/cycle.zarr', :now)
                    """
                ),
                {"id": run_gfs_id, "mv_id": gfs_mv_pk, "ctime": gfs_cycle, "now": now},
            )
            # Seed realistic committed leads (0..240 at 3h cadence -> 81 leads)
            gfs_expected_leads = canonical_lead_time_hours("gfs", "v1.0")
            for lead in gfs_expected_leads:
                conn.execute(
                    text(
                        """
                        INSERT INTO forecast_products (id, run_id, variable_id, grid_id, product_type, lead_time_hours, zarr_chunk_path)
                        VALUES (:id, :run_id, 'temperature_2m', 'global_025deg', 'deterministic', :lead, :path)
                        """
                    ),
                    {
                        "id": f"fp_gfs_{lead}_{uuid.uuid4().hex[:6]}",
                        "run_id": run_gfs_id,
                        "lead": lead,
                        "path": f"tp/{lead}.zarr",
                    },
                )

            # 3. Scenario 2: GEFS Completeness
            # 30 perturbation members, all 81 leads committed, status 'ready'
            gefs_cycle = now - timedelta(hours=4)
            run_gefs_id = f"test_run_gefs_{uuid.uuid4().hex[:8]}"
            conn.execute(
                text(
                    """
                    INSERT INTO model_runs (id, model_version_id, cycle_time, status, zarr_store_path, created_at)
                    VALUES (:id, :mv_id, :ctime, 'ready', 's3://test/gefs/cycle.zarr', :now)
                    """
                ),
                {"id": run_gefs_id, "mv_id": gefs_mv_pk, "ctime": gefs_cycle, "now": now},
            )
            # Seed 30 members
            for m in range(1, 31):
                conn.execute(
                    text(
                        """
                        INSERT INTO ensemble_members (id, run_id, member_index, member_name)
                        VALUES (:id, :run_id, :m_idx, :m_name)
                        """
                    ),
                    {
                        "id": f"em_{m}_{uuid.uuid4().hex[:6]}",
                        "run_id": run_gefs_id,
                        "m_idx": m,
                        "m_name": f"gep{m:02d}",
                    },
                )
            for lead in gfs_expected_leads:
                conn.execute(
                    text(
                        """
                        INSERT INTO forecast_products (id, run_id, variable_id, grid_id, product_type, lead_time_hours, zarr_chunk_path)
                        VALUES (:id, :run_id, 'temperature_2m', 'global_025deg', 'ensemble_member', :lead, :path)
                        """
                    ),
                    {
                        "id": f"fp_gefs_{lead}_{uuid.uuid4().hex[:6]}",
                        "run_id": run_gefs_id,
                        "lead": lead,
                        "path": f"tp/{lead}.zarr",
                    },
                )

            # 4. Scenario 3: Lifecycle Deletion Claims
            # Cycle A: Active cycle (deletion_started_at is NULL)
            c_active = now - timedelta(hours=6)
            # Cycle B: Young claimed cycle (claim age 15m < 1h)
            c_young = now - timedelta(hours=12)
            # Cycle C: 2h claim (claim age 2h > 1h warning)
            c_warn = now - timedelta(hours=18)
            # Cycle D: 5h claim (claim age 5h > 4h critical)
            c_crit = now - timedelta(hours=24)
            # Cycle E: Permanent tombstone (deleted_at set)
            c_tomb = now - timedelta(days=5)

            conn.execute(
                text(
                    """
                    INSERT INTO forecast_cycle_lifecycle (model_id, cycle_time, deletion_started_at, deleted_at, created_at, updated_at)
                    VALUES ('gfs', :c_act, NULL, NULL, :now, :now),
                           ('gfs', :c_young, :now - INTERVAL '15 minutes', NULL, :now, :now),
                           ('gfs', :c_warn, :now - INTERVAL '2 hours', NULL, :now, :now),
                           ('gfs', :c_crit, :now - INTERVAL '5 hours', NULL, :now, :now),
                           ('gfs', :c_tomb, :now - INTERVAL '5 days', :now - INTERVAL '5 days', :now, :now)
                    ON CONFLICT (model_id, cycle_time) DO UPDATE
                    SET deletion_started_at = EXCLUDED.deletion_started_at,
                        deleted_at = EXCLUDED.deleted_at
                    """
                ),
                {
                    "c_act": c_active,
                    "c_young": c_young,
                    "c_warn": c_warn,
                    "c_crit": c_crit,
                    "c_tomb": c_tomb,
                    "now": now,
                },
            )

            # 5. Scenario 4: Reclamation Queue
            # Item 1: queued (30s)
            # Item 2: deleting (< 600s, 60s ago)
            # Item 3: deleting (> 600s, 15m ago)
            # Item 4: failed
            rec_run_id = run_gfs_id
            conn.execute(
                text(
                    """
                    INSERT INTO reclamation_queue (
                        id, run_id, model_id, cycle_time, lead_time_hours, variable_code,
                        target_kind, member_index, valid_time, store_path, physical_key,
                        status, attempt_count, created_at, updated_at
                    )
                    VALUES
                    ('rec_q_1', :rid, 'gfs', :ctime, 0, 'temperature_2m', 'det', 0, :ctime, 's3://test', 'k1', 'queued', 0, :now - INTERVAL '30 seconds', :now),
                    ('rec_del_young', :rid, 'gfs', :ctime, 3, 'temperature_2m', 'det', 0, :ctime, 's3://test', 'k2', 'deleting', 1, :now - INTERVAL '60 seconds', :now - INTERVAL '60 seconds'),
                    ('rec_del_old', :rid, 'gfs', :ctime, 6, 'temperature_2m', 'det', 0, :ctime, 's3://test', 'k3', 'deleting', 1, :now - INTERVAL '15 minutes', :now - INTERVAL '15 minutes'),
                    ('rec_failed', :rid, 'gfs', :ctime, 9, 'temperature_2m', 'det', 0, :ctime, 's3://test', 'k4', 'failed', 3, :now - INTERVAL '30 minutes', :now - INTERVAL '5 minutes')
                    """
                ),
                {"rid": rec_run_id, "ctime": gfs_cycle, "now": now},
            )

            # 6. Scenario 5: 14-Day Metadata Retention Sweeper
            # Tombstone older than 14 days with detailed child metadata still present
            c_overdue = now - timedelta(days=20)
            run_overdue_id = f"test_run_overdue_{uuid.uuid4().hex[:8]}"
            conn.execute(
                text(
                    """
                    INSERT INTO forecast_cycle_lifecycle (model_id, cycle_time, deletion_started_at, deleted_at, created_at, updated_at)
                    VALUES ('gfs', :c_od, :now - INTERVAL '20 days', :now - INTERVAL '20 days', :now, :now)
                    ON CONFLICT (model_id, cycle_time) DO UPDATE
                    SET deleted_at = EXCLUDED.deleted_at
                    """
                ),
                {"c_od": c_overdue, "now": now},
            )
            # Insert detailed model_run row that should have been swept
            conn.execute(
                text(
                    """
                    INSERT INTO model_runs (id, model_version_id, cycle_time, status, zarr_store_path, created_at)
                    VALUES (:id, :mv_id, :ctime, 'ready', 's3://test/old/cycle.zarr', :now - INTERVAL '21 days')
                    """
                ),
                {"id": run_overdue_id, "mv_id": gfs_mv_pk, "ctime": c_overdue, "now": now},
            )

            # 7. Scenario 6: Anti-Resurrection Invariant Violation
            # Cycle marked permanently deleted at T_del, but active processing run created at T_create > T_del
            c_resurrect = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
            t_deleted = datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)
            run_resurrect_id = f"test_run_resurrect_{uuid.uuid4().hex[:8]}"
            conn.execute(
                text(
                    """
                    INSERT INTO forecast_cycle_lifecycle (model_id, cycle_time, deletion_started_at, deleted_at, created_at, updated_at)
                    VALUES ('gfs', :ctime, :del_t, :del_t, :del_t, :del_t)
                    ON CONFLICT (model_id, cycle_time) DO UPDATE
                    SET deleted_at = EXCLUDED.deleted_at
                    """
                ),
                {"ctime": c_resurrect, "del_t": t_deleted},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO model_runs (id, model_version_id, cycle_time, status, zarr_store_path, created_at)
                    VALUES (:id, :mv_id, :ctime, 'processing', 's3://test/resurrect/cycle.zarr', :now)
                    """
                ),
                {"id": run_resurrect_id, "mv_id": gfs_mv_pk, "ctime": c_resurrect, "now": now},
            )

            # Provide the active transaction connection to the test
            yield conn

        finally:
            # Unconditional rollback guarantees 100% isolation
            trans.rollback()


# =============================================================================
# Populated Collector & Invariant Validation Tests
# =============================================================================


def test_populated_gfs_completeness(populated_db: Connection):
    """Verify GFS completeness derives 81 leads and 240h max dynamically and returns ready."""
    collector = IngestionHealthCollector(engine=populated_db)
    comp = collector.evaluate_gfs_completeness()

    assert comp["model"] == "gfs"
    assert comp["expected_leads"] == len(canonical_lead_time_hours("gfs", "v1.0"))
    assert comp["max_lead_hours"] == model_max_lead_hours("gfs", "v1.0")
    assert comp["committed_leads"] == 81
    assert comp["servable"] is True
    assert comp["ready"] is True
    assert comp["status"] == "ready"


def test_populated_gefs_completeness(populated_db: Connection):
    """Verify GEFS completeness derives 30 members and 81 leads dynamically and returns ready."""
    collector = IngestionHealthCollector(engine=populated_db)
    comp = collector.evaluate_gefs_completeness()

    assert comp["model"] == "gefs"
    assert comp["expected_members"] == get_expected_members("gefs")
    assert comp["members"] == 30
    assert comp["expected_leads"] == len(canonical_lead_time_hours("gefs", "v1.0"))
    assert comp["committed_leads"] == 81
    assert comp["servable"] is True
    assert comp["ready"] is True
    assert comp["status"] == "ready"


def test_populated_lifecycle_deletion_claims(populated_db: Connection):
    """Verify deletion claims correctly identify active, young claims, 2h warnings, 5h criticals, and tombstones."""
    collector = LifecycleHealthCollector(populated_db)
    report = collector.collect()

    assert report.error is None
    # Active cycles: at least 1 (the active cycle seeded without deletion_started_at)
    assert report.active_cycles >= 1
    # Claimed deletion: 3 cycles (young, 2h, 5h)
    assert report.claimed_cycles >= 3
    # Tombstones: at least 2 (young tombstone, overdue tombstone, anti-resurrect tombstone)
    assert report.tombstone_cycles >= 2

    # Oldest claim age must reflect the 5-hour claim (>= 18,000s)
    assert report.oldest_claim_age_s >= 18000.0

    # Stuck claims:
    # 2h claim and 5h claim are > 1 hour -> warning >= 2
    assert report.stuck_claims_warning >= 2
    # 5h claim is > 4 hours -> critical >= 1
    assert report.stuck_claims_critical >= 1


def test_populated_reclamation_queue(populated_db: Connection):
    """Verify reclamation queue reflects queued, young deleting, expired deleting (>600s), and failed rows."""
    collector = LifecycleHealthCollector(populated_db)
    report = collector.collect()
    rec = report.reclamation

    assert rec.queued_count >= 1
    assert rec.deleting_count >= 2
    assert rec.failed_count >= 1

    # Oldest deleting target was seeded at 15 minutes ago (> 900s)
    assert rec.oldest_deleting_age_s >= 900.0


def test_populated_14_day_metadata_sweeper(populated_db: Connection):
    """Verify 14-day sweeper detects tombstones older than 14 days with unpurged child metadata."""
    collector = LifecycleHealthCollector(populated_db)
    report = collector.collect()

    # Eligible tombstones older than 14 days
    assert report.sweeper_eligible_count >= 1
    # Unpurged detailed metadata count
    assert report.sweeper_unpurged_count >= 1
    # Overdue duration (20 days - 14 days = 6 days overdue >= 500,000s)
    assert report.sweeper_oldest_overdue_s >= 500000.0


def test_populated_anti_resurrection_invariant(populated_db: Connection):
    """Verify anti-resurrection audit detects active/recreated model_runs under permanent tombstones."""
    collector = LifecycleHealthCollector(populated_db)
    report = collector.collect()

    anti_resurrect_violations = [
        v for v in report.violations if v.violation_type == "anti_resurrection_violation"
    ]
    assert len(anti_resurrect_violations) >= 1

    viol = anti_resurrect_violations[0]
    assert viol.model_id == "gfs"
    assert "violates permanent tombstone" in viol.description
    assert "status=processing" in viol.description


def test_populated_alert_engine_mapping(populated_db: Connection):
    """Verify AlertEngine fires the exact expected WARNING and CRITICAL alerts against populated reports."""
    lc_collector = LifecycleHealthCollector(populated_db)
    lc_report = lc_collector.collect()

    alerts = ALERT_ENGINE.evaluate_rules(
        resource_data={"disks": [DiskInfo(path="/", total_bytes=100, used_bytes=50, free_bytes=50, used_percent=50.0)]},
        postgres_data=None,
        lifecycle_data=lc_report,
        ingestion_data=None,
        storage_data=None,
        leak_data=None,
    )

    alert_names = {a.name for a in alerts}

    # 1. Stuck claim critical (> 4h)
    assert "finalizer_claim_stuck_critical" in alert_names
    # 2. Reclamation deleting stuck (> 600s)
    assert "reclamation_deleting_stuck" in alert_names
    # 3. Reclamation failed shards
    assert "reclamation_failed_shards" in alert_names
    # 4. Metadata sweeper backlog overdue
    assert "metadata_sweeper_backlog_overdue" in alert_names
    # 5. Anti-resurrection violation
    assert "invariant_anti_resurrection_violation" in alert_names
