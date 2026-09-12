"""Live runtime verification script for platform monitoring, alerting, and recovery.

Demonstrates:
1. Live health and resource telemetry collection.
2. Synthetic alert generation.
3. Alert deduplication and cooldown suppression.
4. Alert escalation from WARNING to CRITICAL.
5. Alert recovery notification.
"""

from __future__ import annotations

import time

from ingestion.monitoring.alerts import (
    Alert,
    AlertEngine,
    AlertSeverity,
)
from ingestion.monitoring.resources import RESOURCE_COLLECTOR


def main() -> int:
    print("=" * 80)
    print("STARTING LIVE RUNTIME MONITORING & ALERT VALIDATION")
    print("=" * 80)

    # 1. Live system resource collection
    print("\n[Step 1] Collecting live host resource metrics...")
    res = RESOURCE_COLLECTOR.collect_and_export()
    print(f"  CPU Utilization:   {res['cpu_percent']}%")
    print(f"  Process RSS:       {res['memory'].rss_bytes / (1024*1024):.2f} MB")
    print(f"  Peak RSS:          {res['memory'].peak_rss_bytes / (1024*1024):.2f} MB")
    print(f"  Active Threads:    {res['threads']}")
    for d in res["disks"]:
        print(f"  Disk ({d.path}): {d.used_percent}% used ({d.free_bytes / (1024**3):.1f} GB free)")

    # 2. Setup AlertEngine with Memory Sink for validation
    print("\n[Step 2] Setting up AlertEngine...")
    engine = AlertEngine(cooldown_seconds=3600.0)

    t0 = time.time()

    # 3. Simulate Alert Trigger (Warning)
    print("\n[Step 3] Simulating synthetic WARNING alert: disk_space_warning on /data...")
    warn_alert = Alert(
        name="disk_space_warning",
        severity=AlertSeverity.WARNING,
        summary="Disk space warning on /data (85.0%)",
        description="Disk utilization is 85.0% on /data.",
        scope="/data",
        value=85.0,
        threshold=80.0,
        timestamp=t0,
    )
    events_1 = engine.deduplicator.process_alerts([warn_alert], now=t0)
    for ev in events_1:
        engine.log_sink.emit(ev)
        engine.memory_sink.emit(ev)
    assert len(events_1) == 1
    assert events_1[0].event_type == "triggered"
    assert events_1[0].alert.severity == AlertSeverity.WARNING
    print("  -> Verified: WARNING alert triggered and dispatched.")

    # 4. Simulate Repeated Trigger within Cooldown (Suppression)
    print("\n[Step 4] Simulating repeated WARNING alert 60s later (within cooldown)...")
    events_2 = engine.deduplicator.process_alerts([warn_alert], now=t0 + 60.0)
    for ev in events_2:
        engine.log_sink.emit(ev)
        engine.memory_sink.emit(ev)
    assert len(events_2) == 0
    print("  -> Verified: Duplicate alert correctly suppressed by cooldown.")

    # 5. Simulate Alert Escalation to CRITICAL (Immediate bypass of cooldown)
    print("\n[Step 5] Simulating alert escalation to CRITICAL: disk reaches 95%...")
    crit_alert = Alert(
        name="disk_space_warning",  # Same key
        severity=AlertSeverity.CRITICAL,
        summary="Disk space critical on /data (95.0%)",
        description="Disk utilization has reached 95.0% on /data.",
        scope="/data",
        value=95.0,
        threshold=90.0,
        timestamp=t0 + 120.0,
    )
    events_3 = engine.deduplicator.process_alerts([crit_alert], now=t0 + 120.0)
    for ev in events_3:
        engine.log_sink.emit(ev)
        engine.memory_sink.emit(ev)
    assert len(events_3) == 1
    assert events_3[0].event_type == "escalated"
    assert events_3[0].alert.severity == AlertSeverity.CRITICAL
    print("  -> Verified: Alert escalated to CRITICAL immediately.")

    # 6. Simulate Condition Resolution & Recovery Notification
    print("\n[Step 6] Simulating condition resolution (disk freed, empty active alerts)...")
    events_4 = engine.deduplicator.process_alerts([], now=t0 + 180.0)
    for ev in events_4:
        engine.log_sink.emit(ev)
        engine.memory_sink.emit(ev)
    assert len(events_4) == 1
    assert events_4[0].event_type == "recovered"
    assert events_4[0].alert.name == "disk_space_warning"
    print("  -> Verified: RECOVERY event emitted successfully.")

    # Check memory sink
    recent = engine.memory_sink.get_recent()
    print(f"\n[Step 7] Total events logged in memory ring buffer: {len(recent)}")
    for r in recent:
        print(f"  - [{r.event_type.upper()}] {r.alert.severity.value}: {r.alert.summary}")

    print("\n" + "=" * 80)
    print("ALL RUNTIME VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
