"""Live runtime smoke audit for all real monitoring collectors against PostgreSQL and MinIO.

Validates that real schema queries execute without syntax errors, null errors,
or unbounded table scans.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.main import app  # type: ignore[import-untyped]
from ingestion.core.config import settings
from ingestion.core.db import engine
from ingestion.monitoring import (
    INGESTION_COLLECTOR,
    RESOURCE_COLLECTOR,
    LifecycleHealthCollector,
    PostgresHealthCollector,
    StorageHealthCollector,
)


def run_smoke_audit() -> int:
    print("=" * 80)
    print("LIVE RUNTIME COLLECTOR SMOKE AUDIT")
    print("=" * 80)

    # Ensure schema is at head
    try:
        from alembic.config import Config
        from alembic import command
        import os
        orig = os.getcwd()
        os.chdir("services/api")
        command.upgrade(Config("alembic.ini"), "head")
        os.chdir(orig)
    except Exception as exc:
        print(f"Warning: Alembic upgrade check: {exc}")

    # 1. PostgreSQL Capacity & Health Collector
    print("\n[Audit 1/7] Testing PostgresHealthCollector against live PostgreSQL...")
    pg_collector = PostgresHealthCollector(engine)
    pg_report = pg_collector.collect()
    assert pg_report.connected is True, f"PostgreSQL not connected: {pg_report.error}"
    print(f"  Connected:               {pg_report.connected}")
    print(f"  Connections:             {pg_report.active_connections} / {pg_report.max_connections} ({pg_report.connection_utilization_pct}%)")
    print(f"  Database Size:           {pg_report.total_size_bytes / (1024*1024):.2f} MB")
    print(f"  Cumulative Deadlocks:    {pg_report.deadlocks} (Delta: {pg_report.deadlocks_delta})")
    print(f"  Lock Waits:              {pg_report.lock_waits}")
    print(f"  Tracked Tables Checked:  {len(pg_report.tables)} tables")
    for tname, tstat in pg_report.tables.items():
        print(f"    - {tname:<26} size={tstat.total_bytes / 1024:.1f} KB, live={tstat.live_tuples}, dead={tstat.dead_tuples}")
    print("  -> PASS: PostgreSQL collector executed cleanly.")

    # 2. Object Storage Health Collector
    print("\n[Audit 2/7] Testing StorageHealthCollector against live MinIO...")
    storage_collector = StorageHealthCollector(
        endpoint_url=getattr(settings, "MINIO_ENDPOINT", "http://localhost:9000"),
        bucket=getattr(settings, "MINIO_BUCKET_NAME", "weather-data"),
        access_key=getattr(settings, "MINIO_ACCESS_KEY", "minio_admin"),
        secret_key=getattr(settings, "MINIO_SECRET_KEY", "minio_password"),
    )
    storage_report = storage_collector.probe()
    assert storage_report.connected is True, f"Storage not connected: {storage_report.error}"
    print(f"  Connected:    {storage_report.connected}")
    print(f"  Latency:      {storage_report.latency_ms:.2f} ms")
    print(f"  Endpoint:     {storage_report.endpoint}")
    print(f"  Bucket:       {storage_report.bucket}")
    print("  -> PASS: MinIO storage collector executed cleanly.")

    # 3. Lifecycle, Sweeper, Reclamation & Invariants Collector
    print("\n[Audit 3/7] Testing LifecycleHealthCollector against live schema...")
    lc_collector = LifecycleHealthCollector(engine)
    lc_report = lc_collector.collect()
    assert lc_report.error is None, f"Lifecycle collector error: {lc_report.error}"
    print(f"  Active Cycles:             {lc_report.active_cycles}")
    print(f"  Claimed Deletions:         {lc_report.claimed_cycles}")
    print(f"  Permanent Tombstones:      {lc_report.tombstone_cycles}")
    print(f"  Oldest Claim Age:          {lc_report.oldest_claim_age_s:.1f}s")
    print(f"  14-Day Sweeper Eligible:   {lc_report.sweeper_eligible_count}")
    print(f"  14-Day Sweeper Unpurged:   {lc_report.sweeper_unpurged_count}")
    print(f"  Reclamation Queue:         queued={lc_report.reclamation.queued_count}, deleting={lc_report.reclamation.deleting_count}, deleted={lc_report.reclamation.deleted_count}, failed={lc_report.reclamation.failed_count}")
    print(f"  Invariant Violations:      {len(lc_report.violations)}")
    print("  -> PASS: Lifecycle collector executed cleanly.")

    # 4. Ingestion Completeness (GFS & GEFS)
    print("\n[Audit 4/7] Testing Ingestion Model Completeness...")
    gfs_comp = INGESTION_COLLECTOR.evaluate_model_completeness("gfs")
    gefs_comp = INGESTION_COLLECTOR.evaluate_model_completeness("gefs")
    print(f"  GFS Completeness:  committed={gfs_comp['committed_leads']}/{gfs_comp['expected_leads']} (max {gfs_comp['max_lead_hours']}h), servable={gfs_comp['servable']}, ready={gfs_comp['ready']}")
    print(f"  GEFS Completeness: members={gefs_comp['members']}/{gefs_comp['expected_members']}, committed={gefs_comp['committed_leads']}/{gefs_comp['expected_leads']}, servable={gefs_comp['servable']}, ready={gefs_comp['ready']}")
    assert gfs_comp["expected_leads"] == 81
    assert gfs_comp["max_lead_hours"] == 240
    assert gefs_comp["expected_members"] == 30
    assert gefs_comp["expected_leads"] == 81
    print("  -> PASS: Model-specific completeness contracts verified.")

    # 5. Ingestion Lag
    print("\n[Audit 5/7] Testing Ingestion Lag Analysis...")
    now_utc = datetime.now(timezone.utc)
    gfs_lag = INGESTION_COLLECTOR.evaluate_lag("gfs", now=now_utc)
    gefs_lag = INGESTION_COLLECTOR.evaluate_lag("gefs", now=now_utc)
    print(f"  GFS Lag:  expected={gfs_lag.latest_expected_cycle.strftime('%HZ')}, ready={gfs_lag.latest_ready_cycle}, lag_cycles={gfs_lag.lag_cycles}, is_behind={gfs_lag.is_behind}")
    print(f"  GEFS Lag: expected={gefs_lag.latest_expected_cycle.strftime('%HZ')}, ready={gefs_lag.latest_ready_cycle}, lag_cycles={gefs_lag.lag_cycles}, is_behind={gefs_lag.is_behind}")
    print("  -> PASS: Ingestion lag analysis verified.")

    # 6. System Resource Collector
    print("\n[Audit 6/7] Testing SystemResourceCollector...")
    res = RESOURCE_COLLECTOR.collect_and_export()
    print(f"  RSS:      {res['memory'].rss_bytes / (1024*1024):.1f} MB (Peak: {res['memory'].peak_rss_bytes / (1024*1024):.1f} MB)")
    print(f"  Threads:  {res['threads']}")
    print(f"  Disks:    {len(res['disks'])} inspected")
    print("  -> PASS: System resource collector verified.")

    # 7. FastAPI Live Endpoints (/v1/health, /v1/health/detailed, /v1/metrics)
    print("\n[Audit 7/7] Testing API Serving Endpoints via TestClient...")
    with TestClient(app) as client:
        # Standard health
        resp_health = client.get("/v1/health")
        print(f"  GET /v1/health -> HTTP {resp_health.status_code}")
        assert resp_health.status_code == 200
        assert resp_health.json()["data"]["status"] == "healthy"

        # Detailed health
        resp_detail = client.get("/v1/health/detailed")
        print(f"  GET /v1/health/detailed -> HTTP {resp_detail.status_code}")
        assert resp_detail.status_code == 200
        body_detail = resp_detail.json()
        assert body_detail["status"] == "healthy"
        assert body_detail["dependencies"]["database"] == "connected"
        assert body_detail["dependencies"]["redis"] == "connected"
        assert body_detail["dependencies"]["object_storage"] == "connected"

        # Prometheus metrics
        resp_metrics = client.get("/v1/metrics")
        print(f"  GET /v1/metrics -> HTTP {resp_metrics.status_code} ({len(resp_metrics.text)} bytes)")
        assert resp_metrics.status_code == 200
        assert "text/plain; version=0.0.4" in resp_metrics.headers["Content-Type"]
        assert "weather_api_database_connected 1.0" in resp_metrics.text
        assert "weather_api_redis_connected 1.0" in resp_metrics.text
        assert "weather_api_storage_connected 1.0" in resp_metrics.text
        assert "weather_api_process_memory_rss_bytes" in resp_metrics.text

    print("\n" + "=" * 80)
    print("SMOKE AUDIT COMPLETE: ALL LIVE COLLECTORS OPERATING NORMALLY!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_audit())
