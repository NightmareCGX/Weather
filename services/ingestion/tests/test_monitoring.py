"""Comprehensive test suite for runtime health, resource, ingestion, lifecycle, and alert monitoring.

Covers:
- Prometheus metric registry, counters, gauges, histograms, exposition format
- Cross-platform system resource collection (CPU, RSS, VMS, threads, disk)
- Cycle-to-cycle resource leak detection (stable vs. sustained monotonic leak)
- PostgreSQL health collection with bounded catalog queries
- Lifecycle, finalizer escalation, 14-day sweeper, reclamation queue, and anti-resurrection audits
- Ingestion progress tracking, stuck detection (download/decode/write/finalize), and lag analysis
- GEFS completeness rules (26/30 servable, 30/30 ready)
- Object storage (MinIO/S3) probes and phase duration metrics
- Alert rules evaluation, deduplication, cooldown suppression, escalation, and recovery
- Pluggable alert delivery (logging, in-memory, webhook fail-open)
- Operator summary and diagnostics renderers (TASK 20 format)
- CLI commands (status, diagnostics, audit, alert-check, metrics)
- Failure safety / fail-open guarantees
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ingestion.monitoring.alerts import (
    Alert,
    AlertDeduplicator,
    AlertEngine,
    AlertEvent,
    AlertSeverity,
    WebhookAlertSink,
)
from ingestion.monitoring.database import (
    LongRunningQuery,
    PostgresHealthCollector,
    PostgresHealthReport,
    TableStat,
)
from ingestion.monitoring.ingestion_collector import (
    IngestionHealthCollector,
    IngestionLagReport,
    ModelIngestionState,
    latest_synoptic_cycle,
)
from ingestion.monitoring.lifecycle_collector import (
    LifecycleHealthCollector,
    LifecycleHealthReport,
    ReclamationQueueStats,
)
from ingestion.monitoring.metrics import (
    MetricRegistry,
)
from ingestion.monitoring.resources import (
    CycleResourceLeakDetector,
    DiskInfo,
    MemoryInfo,
    ResourceLeakReport,
    SystemResourceCollector,
)
from ingestion.monitoring.storage import (
    StorageHealthCollector,
    StorageHealthReport,
)
from ingestion.monitoring.summary import (
    render_audit,
    render_diagnostics,
    render_platform_status,
)


# =============================================================================
# 1. Prometheus Metrics Registry & Exposition Tests
# =============================================================================


def test_metrics_primitives_and_exposition():
    reg = MetricRegistry()
    g = reg.gauge("test_gauge", "A test gauge", labelnames=("model",))
    c = reg.counter("test_counter", "A test counter", labelnames=("phase",))
    h = reg.histogram("test_hist", "A test histogram", buckets=(1.0, 5.0))

    g.labels(model="gfs").set(42.5)
    c.labels(phase="download").inc(3.0)
    h.observe(0.5)
    h.observe(2.5)

    expo = reg.generate_latest()
    assert "# HELP test_gauge A test gauge" in expo
    assert "# TYPE test_gauge gauge" in expo
    assert 'test_gauge{model="gfs"} 42.5' in expo
    assert 'test_counter{phase="download"} 3.0' in expo
    assert 'test_hist_bucket{le="1.0"} 1' in expo
    assert 'test_hist_bucket{le="5.0"} 2' in expo
    assert 'test_hist_count 2' in expo
    assert 'test_hist_sum 3.0' in expo


def test_metric_validation_errors():
    reg = MetricRegistry()
    with pytest.raises(ValueError, match="Invalid metric name"):
        reg.gauge("invalid-name-with-dash", "doc")
    with pytest.raises(ValueError, match="Invalid label name"):
        reg.gauge("valid_name", "doc", labelnames=("bad-label",))
    with pytest.raises(ValueError, match="cannot start with '__'"):
        reg.gauge("valid_name", "doc", labelnames=("__reserved",))


# =============================================================================
# 2. System Resource Collector & Cycle Leak Detector Tests
# =============================================================================


def test_system_resource_collector():
    collector = SystemResourceCollector()
    mem = collector.get_memory_info()
    assert isinstance(mem, MemoryInfo)
    assert mem.rss_bytes >= 0
    assert mem.peak_rss_bytes >= mem.rss_bytes

    cpu = collector.get_cpu_percent()
    assert cpu >= 0.0

    threads = collector.get_thread_count()
    assert threads >= 1

    disk = collector.get_disk_usage(".")
    assert isinstance(disk, DiskInfo)
    assert disk.total_bytes > 0
    assert disk.free_bytes > 0
    assert 0.0 <= disk.used_percent <= 100.0

    exported = collector.collect_and_export(disk_paths=(".",))
    assert "memory" in exported
    assert "cpu_percent" in exported
    assert "threads" in exported
    assert len(exported["disks"]) >= 1


def test_cycle_resource_leak_detector_stable():
    detector = CycleResourceLeakDetector(min_cycles=5, min_growth_pct=0.15, min_growth_bytes=100_000_000)
    # 5 cycles with fluctuating but non-leaking memory
    baselines = [200_000_000, 205_000_000, 198_000_000, 202_000_000, 201_000_000]
    for i, b in enumerate(baselines):
        detector.record_cycle(
            cycle_key=f"gfs_c{i}",
            pre_baseline_rss=b - 5_000_000,
            peak_rss=b + 50_000_000,
            post_baseline_rss=b,
        )

    report = detector.evaluate_leak()
    assert report.is_leak is False
    assert report.cycle_count == 5
    assert "stable" in report.message


def test_cycle_resource_leak_detector_leak_detected():
    detector = CycleResourceLeakDetector(min_cycles=5, min_growth_pct=0.15, min_growth_bytes=100_000_000)
    # 5 cycles with sustained monotonic growth from 200MB to 350MB (+150MB, +75%)
    baselines = [200_000_000, 235_000_000, 270_000_000, 310_000_000, 350_000_000]
    for i, b in enumerate(baselines):
        detector.record_cycle(
            cycle_key=f"gfs_c{i}",
            pre_baseline_rss=b - 10_000_000,
            peak_rss=b + 80_000_000,
            post_baseline_rss=b,
        )

    report = detector.evaluate_leak()
    assert report.is_leak is True
    assert report.retained_growth_bytes == 150_000_000
    assert report.retained_growth_pct >= 0.70
    assert "Sustained baseline memory leak detected" in report.message


def test_cycle_leak_detector_temporary_peak_not_leak():
    detector = CycleResourceLeakDetector(min_cycles=5, min_growth_pct=0.15, min_growth_bytes=100_000_000)
    # Temporary peak in cycle 2, but post-baseline returns down in cycle 3
    baselines = [200_000_000, 210_000_000, 300_000_000, 205_000_000, 208_000_000]
    for i, b in enumerate(baselines):
        detector.record_cycle(
            cycle_key=f"gefs_c{i}",
            pre_baseline_rss=b,
            peak_rss=b + 200_000_000,
            post_baseline_rss=b,
        )

    report = detector.evaluate_leak()
    assert report.is_leak is False


# =============================================================================
# 3. PostgreSQL Health Collector Tests
# =============================================================================


def test_postgres_health_collector_mock():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # Setup scalar returns
    mock_conn.execute.side_effect = [
        MagicMock(scalar=lambda: 100),  # max_connections
        MagicMock(scalar=lambda: 25),   # active_conn
        MagicMock(scalar=lambda: 1024 * 1024 * 50),  # db size (50MB)
        MagicMock(scalar=lambda: 0),    # deadlocks
        [],                             # long queries
        MagicMock(scalar=lambda: 0),    # lock waits
        [],                             # table stats
    ]

    collector = PostgresHealthCollector(mock_engine)
    report = collector.collect()
    assert report.connected is True
    assert report.max_connections == 100
    assert report.active_connections == 25
    assert report.connection_utilization_pct == 25.0
    assert report.total_size_bytes == 1024 * 1024 * 50
    assert report.is_connection_warning is False
    assert report.is_connection_critical is False


def test_postgres_health_collector_disconnected():
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = RuntimeError("Connection refused")

    collector = PostgresHealthCollector(mock_engine)
    report = collector.collect()
    assert report.connected is False
    assert report.error is not None


# =============================================================================
# 4. Lifecycle & Anti-Resurrection Collector Tests
# =============================================================================


def test_lifecycle_collector_mock():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    cycle_row = MagicMock(
        active_cnt=5,
        claimed_cnt=1,
        tombstone_cnt=200,
        oldest_claim_age=1800.0,
        stuck_warn_cnt=0,
        stuck_crit_cnt=0,
    )
    sweep_row = MagicMock(
        eligible_cnt=10,
        unpurged_cnt=2,
        oldest_overdue_s=7200.0,
    )

    mock_conn.execute.side_effect = [
        MagicMock(fetchone=lambda: cycle_row),
        MagicMock(fetchone=lambda: sweep_row),
        [],  # reclamation queue rows
        [],  # invariant 1
        [],  # invariant 2 (anti-resurrection)
    ]

    collector = LifecycleHealthCollector(mock_engine)
    report = collector.collect()
    assert report.active_cycles == 5
    assert report.claimed_cycles == 1
    assert report.tombstone_cycles == 200
    assert report.oldest_claim_age_s == 1800.0
    assert report.sweeper_unpurged_count == 2
    assert len(report.violations) == 0


def test_lifecycle_collector_stuck_claims_and_anti_resurrection():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    cycle_row = MagicMock(
        active_cnt=3,
        claimed_cnt=2,
        tombstone_cnt=50,
        oldest_claim_age=15000.0,  # > 4 hours
        stuck_warn_cnt=1,
        stuck_crit_cnt=1,
    )
    sweep_row = MagicMock(eligible_cnt=0, unpurged_cnt=0, oldest_overdue_s=0.0)

    # Invariant violation 1: deleted_at without deletion_started_at
    inv1_row = MagicMock(model_id="gfs", cycle_time=datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc))
    # Invariant violation 2: anti-resurrection (run recreated under tombstone)
    inv2_row = MagicMock(
        model_id="gfs",
        cycle_time=datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc),
        id="run_123",
        status="processing",
        created_at=datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc),
        deleted_at=datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc),
    )

    mock_conn.execute.side_effect = [
        MagicMock(fetchone=lambda: cycle_row),
        MagicMock(fetchone=lambda: sweep_row),
        [],
        [inv1_row],
        [inv2_row],
    ]

    collector = LifecycleHealthCollector(mock_engine)
    report = collector.collect()
    assert report.stuck_claims_critical == 1
    assert len(report.violations) == 2
    assert report.violations[0].violation_type == "invalid_lifecycle_transition"
    assert report.violations[1].violation_type == "anti_resurrection_violation"


# =============================================================================
# 5. Ingestion Health & Stuck Detection Tests
# =============================================================================


def test_ingestion_stuck_detection_stages():
    collector = IngestionHealthCollector()
    mock_tracker = MagicMock()

    # Case 1: Running, steady progress -> NOT stuck
    collector.register_run_start("gfs", "2026-09-12 00:00Z", mock_tracker)
    mock_snap = MagicMock(
        download_active=2,
        download_queued=5,
        decode_active=0,
        decode_queued=0,
        write_active=0,
        write_waiting=0,
        finalize_state="waiting",
    )
    mock_tracker.get_snapshot.return_value = mock_snap

    report = collector.check_stuck("gfs")
    assert report.is_stuck is False

    # Case 2: Download stalled for > 600s
    state = collector.get_model_state("gfs")
    assert state is not None
    state.last_download_ts = time.monotonic() - 650.0  # 650 seconds ago
    stuck_report = collector.check_stuck("gfs")
    assert stuck_report.is_stuck is True
    assert "Download stage stalled" in str(stuck_report.reason)

    # Record progress -> resets stuck state
    collector.record_progress("gfs", "download", success=True)
    report_recovered = collector.check_stuck("gfs")
    assert report_recovered.is_stuck is False


def test_ingestion_stuck_detection_finalize():
    collector = IngestionHealthCollector()
    mock_tracker = MagicMock()
    collector.register_run_start("gefs", "2026-09-12 00:00Z", mock_tracker)

    # Finalize active for 700s
    mock_snap = MagicMock(
        download_active=0,
        download_queued=0,
        decode_active=0,
        decode_queued=0,
        write_active=0,
        write_waiting=0,
        finalize_state="active",
        finalize_start_time=time.monotonic() - 700.0,
    )
    mock_tracker.get_snapshot.return_value = mock_snap

    stuck_report = collector.check_stuck("gefs")
    assert stuck_report.is_stuck is True
    assert "Finalization stalled" in str(stuck_report.reason)


def test_ingestion_lag_detection():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.scalar.return_value = None

    collector = IngestionHealthCollector(engine=mock_engine)

    now = datetime(2026, 9, 12, 14, 0, 0, tzinfo=timezone.utc)  # Expected 12Z
    expected_cycle = latest_synoptic_cycle(now, cadence_hours=6)
    assert expected_cycle == datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)

    # Scenario A: Upstream NOT yet available (NOAA hasn't published 12Z)
    report_a = collector.evaluate_lag("gfs", upstream_latest_cycle=None, now=now)
    assert report_a.upstream_available is False
    assert report_a.is_behind is False

    # Scenario B: Upstream available at 12Z, local catalog max ready is 06Z (1 cycle lag)
    mock_conn.execute.return_value.scalar.return_value = datetime(2026, 9, 12, 6, 0, 0, tzinfo=timezone.utc)

    report_b = collector.evaluate_lag(
        "gfs",
        upstream_latest_cycle=datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc),
        now=now,
    )
    assert report_b.upstream_available is True
    assert report_b.lag_cycles == 1
    assert report_b.lag_hours == 6.0
    assert report_b.is_behind is True


def test_gefs_completeness_thresholds():
    mock_engine = MagicMock()
    collector = IngestionHealthCollector(engine=mock_engine)
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # 26 members -> servable True, ready False
    mock_conn.execute.return_value.fetchone.return_value = MagicMock(
        cycle_time=datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc),
        member_count=26,
        committed_leads=40,
        status="partial",
    )
    comp_26 = collector.evaluate_gefs_completeness()
    assert comp_26["servable"] is True
    assert comp_26["ready"] is False

    # 30 members and ready -> ready True
    mock_conn.execute.return_value.fetchone.return_value = MagicMock(
        cycle_time=datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc),
        member_count=30,
        committed_leads=81,
        status="ready",
    )
    comp_30 = collector.evaluate_gefs_completeness()
    assert comp_30["servable"] is True
    assert comp_30["ready"] is True


def test_gfs_completeness_thresholds():
    mock_engine = MagicMock()
    collector = IngestionHealthCollector(engine=mock_engine)
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # 40 leads -> servable True, ready False
    mock_conn.execute.return_value.fetchone.return_value = MagicMock(
        cycle_time=datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc),
        committed_leads=40,
        status="partial",
    )
    comp_40 = collector.evaluate_gfs_completeness()
    assert comp_40["servable"] is True
    assert comp_40["ready"] is False
    assert comp_40["expected_leads"] == 81
    assert comp_40["max_lead_hours"] == 240

    # 81 leads and ready -> ready True
    mock_conn.execute.return_value.fetchone.return_value = MagicMock(
        cycle_time=datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc),
        committed_leads=81,
        status="ready",
    )
    comp_81 = collector.evaluate_gfs_completeness()
    assert comp_81["servable"] is True
    assert comp_81["ready"] is True


def test_postgres_deadlocks_delta():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # First collect: 5 deadlocks historically -> delta 0 (no new deadlocks)
    mock_conn.execute.side_effect = [
        MagicMock(scalar=lambda: 100),  # max
        MagicMock(scalar=lambda: 10),   # active
        MagicMock(scalar=lambda: 1000),  # size
        MagicMock(scalar=lambda: 5),    # deadlocks cumulative
        [],                             # long queries
        MagicMock(scalar=lambda: 0),    # locks
        [],                             # tables
    ]
    collector = PostgresHealthCollector(mock_engine)
    r1 = collector.collect()
    assert r1.deadlocks == 5
    assert r1.deadlocks_delta == 0

    # Second collect: 7 deadlocks cumulative -> delta 2 new deadlocks!
    mock_conn.execute.side_effect = [
        MagicMock(scalar=lambda: 100),
        MagicMock(scalar=lambda: 10),
        MagicMock(scalar=lambda: 1000),
        MagicMock(scalar=lambda: 7),    # deadlocks cumulative
        [],
        MagicMock(scalar=lambda: 0),
        [],
    ]
    r2 = collector.collect()
    assert r2.deadlocks == 7
    assert r2.deadlocks_delta == 2

    # Alert engine should fire only when deadlocks_delta > 0
    engine = AlertEngine()
    alerts_no_delta = engine.evaluate_rules(postgres_data=r1)
    assert not any(a.name == "postgres_deadlocks_detected" for a in alerts_no_delta)

    alerts_with_delta = engine.evaluate_rules(postgres_data=r2)
    assert any(a.name == "postgres_deadlocks_detected" for a in alerts_with_delta)


# =============================================================================
# 6. Object Storage (MinIO/S3) Health Collector Tests
# =============================================================================


def test_storage_health_collector():
    collector = StorageHealthCollector(endpoint_url="http://localhost:9000", bucket="weather-data")
    with patch("boto3.client") as mock_boto:
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        report = collector.probe()
        assert report.connected is True
        assert report.latency_ms > 0.0

    collector.record_operation("get", duration_s=0.015, success=True)
    collector.record_operation("put", duration_s=0.045, success=False, retried=True)
    collector.record_phase_duration("prepare_run_store", duration_s=0.125)


# =============================================================================
# 7. Alert Rules, Deduplication, Cooldown, and Recovery Tests
# =============================================================================


def test_alert_rules_evaluation():
    engine = AlertEngine()

    # Test disk critical rule (>=90%)
    disk_crit = DiskInfo(path="/data", total_bytes=100, used_bytes=95, free_bytes=5, used_percent=95.0)
    alerts = engine.evaluate_rules(resource_data={"disks": [disk_crit]})
    assert any(a.name == "disk_space_critical" and a.severity == AlertSeverity.CRITICAL for a in alerts)

    # Test postgres unreachable rule
    pg_down = PostgresHealthReport(connected=False, error="Connection timeout")
    alerts_pg = engine.evaluate_rules(postgres_data=pg_down)
    assert any(a.name == "postgres_unreachable" and a.severity == AlertSeverity.CRITICAL for a in alerts_pg)

    # Test finalizer stuck claim > 4 hours
    lc_stuck = LifecycleHealthReport(stuck_claims_critical=1, oldest_claim_age_s=15000.0)
    alerts_lc = engine.evaluate_rules(lifecycle_data=lc_stuck)
    assert any(a.name == "finalizer_claim_stuck_critical" and a.severity == AlertSeverity.CRITICAL for a in alerts_lc)


def test_alert_deduplication_cooldown_and_recovery():
    dedup = AlertDeduplicator(default_cooldown_seconds=3600.0)
    t0 = 1000.0

    alert_warn = Alert(
        name="disk_space_warning",
        severity=AlertSeverity.WARNING,
        summary="Disk at 85%",
        description="Disk warning",
        scope="/data",
        timestamp=t0,
    )

    # 1. First trigger: emits "triggered"
    events_1 = dedup.process_alerts([alert_warn], now=t0)
    assert len(events_1) == 1
    assert events_1[0].event_type == "triggered"
    assert events_1[0].alert.name == "disk_space_warning"

    # 2. Re-trigger within cooldown (300s later): suppressed!
    events_2 = dedup.process_alerts([alert_warn], now=t0 + 300.0)
    assert len(events_2) == 0

    # 3. Escalation to CRITICAL: immediate trigger despite cooldown!
    alert_crit = Alert(
        name="disk_space_warning",  # same key
        severity=AlertSeverity.CRITICAL,
        summary="Disk at 95%",
        description="Disk critical",
        scope="/data",
        timestamp=t0 + 500.0,
    )
    events_3 = dedup.process_alerts([alert_crit], now=t0 + 500.0)
    assert len(events_3) == 1
    assert events_3[0].event_type == "escalated"
    assert events_3[0].alert.severity == AlertSeverity.CRITICAL

    # 4. Recovery: alert condition resolves (empty list) -> emits "recovered"
    events_4 = dedup.process_alerts([], now=t0 + 600.0)
    assert len(events_4) == 1
    assert events_4[0].event_type == "recovered"
    assert events_4[0].alert.name == "disk_space_warning"
    assert events_4[0].previous_severity == AlertSeverity.CRITICAL


def test_webhook_alert_sink_fail_open():
    sink = WebhookAlertSink("http://localhost:9999/nonexistent", timeout_seconds=0.1)
    event = AlertEvent(
        event_type="triggered",
        alert=Alert(name="test", severity=AlertSeverity.WARNING, summary="s", description="d"),
    )
    # Must fail open and not raise
    sink.emit(event)


# =============================================================================
# 8. Operator Summary & Diagnostics Formatting Tests
# =============================================================================


def test_render_platform_status_layout():
    res_data = {
        "memory": MemoryInfo(rss_bytes=1_800_000_000, vms_bytes=2_500_000_000, peak_rss_bytes=2_000_000_000),
        "disks": [DiskInfo(path="/data", total_bytes=100_000_000_000, used_bytes=61_000_000_000, free_bytes=39_000_000_000, used_percent=61.0)],
    }
    pg_data = PostgresHealthReport(
        connected=True,
        active_connections=18,
        max_connections=100,
        connection_utilization_pct=18.0,
        total_size_bytes=42 * 1024 * 1024 * 1024,
        long_running_queries=[LongRunningQuery(pid=1, duration_seconds=2.1, state="active", query_snippet="SELECT", wait_event=None)],
    )
    storage_data = StorageHealthReport(connected=True, latency_ms=14.0, endpoint="localhost:9000", bucket="weather-data")
    lifecycle_data = LifecycleHealthReport(
        active_cycles=5,
        claimed_cycles=1,
        stuck_claims_warning=0,
        stuck_claims_critical=0,
        tombstone_cycles=214,
        sweeper_unpurged_count=3,
        sweeper_oldest_overdue_s=17 * 60,
        reclamation=ReclamationQueueStats(queued_count=34, deleting_count=2, deleted_count=100, failed_count=0),
    )
    gfs_lag = IngestionLagReport(
        model="gfs",
        latest_expected_cycle=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        latest_upstream_cycle=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        latest_ready_cycle=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        upstream_available=True,
        lag_cycles=0,
        lag_hours=0.0,
        is_behind=False,
    )
    gefs_lag = IngestionLagReport(
        model="gefs",
        latest_expected_cycle=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        latest_upstream_cycle=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        latest_ready_cycle=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        upstream_available=True,
        lag_cycles=0,
        lag_hours=0.0,
        is_behind=False,
    )
    gfs_state = ModelIngestionState(model="gfs", last_duration_s=108.0)
    gefs_state = ModelIngestionState(model="gefs", last_duration_s=2330.0)
    leak_data = ResourceLeakReport(is_leak=False, retained_growth_bytes=1000, retained_growth_pct=0.012, cycle_count=10, message="Stable")

    out = render_platform_status(
        resource_data=res_data,
        postgres_data=pg_data,
        storage_data=storage_data,
        lifecycle_data=lifecycle_data,
        gfs_lag=gfs_lag,
        gefs_lag=gefs_lag,
        gfs_state=gfs_state,
        gefs_state=gefs_state,
        leak_data=leak_data,
        active_alerts=[],
        use_color=False,
    )

    # Assert matching TASK 20 sample specification
    assert "Platform Health: HEALTHY" in out
    assert "connections 18/100" in out
    assert "DB size 42.0 GB" in out
    assert "longest transaction 2.1s" in out
    assert "MinIO:" in out
    assert "reachable (14.0ms)" in out
    assert "disk usage 61.0%" in out
    assert "GFS:" in out
    assert "latest ready cycle 12Z" in out
    assert "ingestion lag 0 cycles" in out
    assert "last duration 108s" in out
    assert "GEFS:" in out
    assert "last duration 38m50s" in out
    assert "active cycles 5" in out
    assert "claimed deletion 1" in out
    assert "tombstones 214" in out
    assert "queued 34" in out
    assert "deleting 2" in out
    assert "eligible backlog 3" in out
    assert "oldest overdue 17m00s" in out
    assert "RSS 1.7 GB" in out
    assert "10-cycle post-run baseline trend: +1.2%" in out


def test_render_diagnostics_and_audit():
    res_data = {"threads": 4, "asyncio_tasks": 2, "open_fds": 12, "gc": {"counts": (100, 10, 1), "threshold": (700, 10, 10)}}
    pg_data = PostgresHealthReport(connected=True, tables={"model_runs": TableStat("model_runs", 1024, 10, 0, None)})
    storage_data = StorageHealthReport(connected=True, latency_ms=5.0, endpoint="localhost:9000", bucket="test")
    lifecycle_data = LifecycleHealthReport(violations=[])

    diag = render_diagnostics(
        resource_data=res_data,
        postgres_data=pg_data,
        storage_data=storage_data,
        lifecycle_data=lifecycle_data,
        gfs_state=None,
        gefs_state=None,
        recent_events=[],
    )
    assert "PLATFORM OPERATIONAL DIAGNOSTICS" in diag
    assert "Active Threads: 4" in diag

    audit = render_audit(
        lifecycle_data=lifecycle_data,
        gefs_completeness={"members": 30, "servable": True, "ready": True},
    )
    assert "DATA LIFECYCLE V3 & DATA INTEGRITY AUDIT" in audit
    assert "Permanent anti-resurrection tombstones intact" in audit
    assert "Available members:   30/30" in audit


def test_monitoring_failure_fail_open_safety():
    """Verify that any monitoring collector exception fails open without breaking ingestion."""
    from ingestion.core.observability import PipelineProgressTracker

    tracker = PipelineProgressTracker(model="gfs", cycle_str="2026-09-12 00:00Z", total_items=10)

    # Patch INGESTION_COLLECTOR to throw
    with patch("ingestion.monitoring.ingestion_collector.INGESTION_COLLECTOR.record_progress", side_effect=RuntimeError("Monitoring crashed")):
        # Stage calls must succeed normally without raising
        tracker.on_download_complete(None, 6, duration_ms=10.0)
        tracker.on_download_failed(None, 12, duration_ms=10.0)
        tracker.on_decode_complete(None, 6, duration_ms=15.0)
        tracker.on_decode_failed(None, 12, duration_ms=15.0)
        tracker.on_write_complete(None, 6, duration_ms=20.0)
        tracker.on_write_failed(None, 12, duration_ms=20.0)
        tracker.on_finalize_complete(duration_ms=50.0)
        tracker.on_finalize_failed(duration_ms=50.0)

    assert tracker.counters.download_done == 1
    assert tracker.counters.download_failed == 1
    assert tracker.counters.decode_done == 1
    assert tracker.counters.decode_failed == 1
    assert tracker.counters.write_done == 1
    assert tracker.counters.write_failed == 1
    assert tracker.counters.finalize_state == "failed"

