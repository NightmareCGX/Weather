"""Operator-facing runtime health summary, diagnostics, and audit formatting.

Provides clean, structured, and human-friendly CLI status output for operators
(weather-ingest status, weather-ingest diagnostics, weather-ingest audit),
with Rich terminal styling when available and plain-text fallback.
"""

from __future__ import annotations

import sys
from typing import Any

from ingestion.monitoring.alerts import AlertSeverity


def _format_bytes(num_bytes: int | float) -> str:
    """Format bytes into B, KB, MB, GB, TB."""
    val = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(val) < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PB"


def _format_seconds(seconds: float) -> str:
    """Format seconds into readable string (e.g. 108s or 38m50s or 2.1h)."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 300:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = seconds / 3600.0
    return f"{hours:.1f}h"


def render_platform_status(
    *,
    resource_data: dict[str, Any],
    postgres_data: Any,
    storage_data: Any,
    lifecycle_data: Any,
    gfs_lag: Any,
    gefs_lag: Any,
    gfs_state: Any,
    gefs_state: Any,
    leak_data: Any,
    active_alerts: list[Any],
    use_color: bool = True,
) -> str:
    """Render the authoritative operator runtime health summary (TASK 20 format)."""
    # 1. Determine overall platform status
    has_critical = any(a.severity == AlertSeverity.CRITICAL for a in active_alerts)
    has_warning = any(a.severity == AlertSeverity.WARNING for a in active_alerts)

    ansi_color = ""
    ansi_reset = ""
    if not postgres_data.connected or not storage_data.connected or has_critical:
        platform_status = "CRITICAL"
        if use_color:
            ansi_color = "\033[91m"  # Red
            ansi_reset = "\033[0m"
    elif has_warning:
        platform_status = "DEGRADED"
        if use_color:
            ansi_color = "\033[93m"  # Yellow
            ansi_reset = "\033[0m"
    else:
        platform_status = "HEALTHY"
        if use_color:
            ansi_color = "\033[92m"  # Green
            ansi_reset = "\033[0m"

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  GLOBAL PROBABILISTIC WEATHER PLATFORM — RUNTIME HEALTH SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Platform Health: {ansi_color}{platform_status}{ansi_reset}")
    lines.append("")

    # PostgreSQL section
    lines.append("PostgreSQL:")
    if postgres_data.connected:
        lines.append(f"    connections {postgres_data.active_connections}/{postgres_data.max_connections} ({postgres_data.connection_utilization_pct}%)")
        lines.append(f"    DB size {_format_bytes(postgres_data.total_size_bytes)}")
        if postgres_data.long_running_queries:
            longest = postgres_data.long_running_queries[0].duration_seconds
            lines.append(f"    longest transaction {longest:.1f}s")
        else:
            lines.append("    longest transaction 0.0s")
        if postgres_data.lock_waits > 0:
            lines.append(f"    lock waits {postgres_data.lock_waits}")
    else:
        lines.append(f"    status: DISCONNECTED ({postgres_data.error})")
    lines.append("")

    # MinIO section
    lines.append("MinIO:")
    if storage_data.connected:
        lines.append(f"    reachable ({storage_data.latency_ms:.1f}ms)")
        # Look for storage disk if available
        disks = resource_data.get("disks", [])
        if disks:
            lines.append(f"    disk usage {disks[0].used_percent}% ({_format_bytes(disks[0].free_bytes)} free)")
        else:
            lines.append("    disk usage normal")
    else:
        lines.append(f"    status: DISCONNECTED ({storage_data.error})")
    lines.append("")

    # GFS section
    lines.append("GFS:")
    gfs_ready = gfs_lag.latest_ready_cycle.strftime("%HZ") if (gfs_lag and gfs_lag.latest_ready_cycle) else "none"
    gfs_exp = gfs_lag.latest_expected_cycle.strftime("%HZ") if gfs_lag else "unknown"
    gfs_dur = _format_seconds(gfs_state.last_duration_s) if gfs_state else "none"
    lines.append(f"    latest ready cycle {gfs_ready}")
    lines.append(f"    expected latest {gfs_exp}")
    lines.append(f"    ingestion lag {gfs_lag.lag_cycles if gfs_lag else 0} cycles ({gfs_lag.lag_hours if gfs_lag else 0.0}h)")
    lines.append(f"    last duration {gfs_dur}")
    lines.append("")

    # GEFS section
    lines.append("GEFS:")
    gefs_ready = gefs_lag.latest_ready_cycle.strftime("%HZ") if (gefs_lag and gefs_lag.latest_ready_cycle) else "none"
    gefs_exp = gefs_lag.latest_expected_cycle.strftime("%HZ") if gefs_lag else "unknown"
    gefs_dur = _format_seconds(gefs_state.last_duration_s) if gefs_state else "none"
    lines.append(f"    latest ready cycle {gefs_ready}")
    lines.append(f"    expected latest {gefs_exp}")
    lines.append(f"    ingestion lag {gefs_lag.lag_cycles if gefs_lag else 0} cycles ({gefs_lag.lag_hours if gefs_lag else 0.0}h)")
    lines.append(f"    last duration {gefs_dur}")
    lines.append("")

    # Lifecycle section
    lines.append("Lifecycle:")
    if lifecycle_data.error is None:
        lines.append(f"    active cycles {lifecycle_data.active_cycles}")
        lines.append(f"    claimed deletion {lifecycle_data.claimed_cycles}")
        lines.append(f"    stuck claims {lifecycle_data.stuck_claims_warning + lifecycle_data.stuck_claims_critical}")
        lines.append(f"    tombstones {lifecycle_data.tombstone_cycles}")
    else:
        lines.append(f"    error: {lifecycle_data.error}")
    lines.append("")

    # Reclamation section
    lines.append("Reclamation:")
    if lifecycle_data.error is None:
        rec = lifecycle_data.reclamation
        lines.append(f"    queued {rec.queued_count}")
        lines.append(f"    deleting {rec.deleting_count}")
        lines.append(f"    deleted {rec.deleted_count}")
        lines.append(f"    failed {rec.failed_count}")
    else:
        lines.append("    unavailable")
    lines.append("")

    # Metadata retention section
    lines.append("Metadata retention:")
    if lifecycle_data.error is None:
        lines.append(f"    eligible backlog {lifecycle_data.sweeper_unpurged_count}")
        oldest_overdue_str = _format_seconds(lifecycle_data.sweeper_oldest_overdue_s)
        lines.append(f"    oldest overdue {oldest_overdue_str}")
    else:
        lines.append("    unavailable")
    lines.append("")

    # Memory & Process section
    mem = resource_data.get("memory")
    lines.append("Memory:")
    if mem is not None:
        lines.append(f"    RSS {_format_bytes(mem.rss_bytes)} (Peak: {_format_bytes(mem.peak_rss_bytes)})")
        lines.append(f"    VMS {_format_bytes(mem.vms_bytes)}")
    if leak_data is not None:
        lines.append(f"    {leak_data.cycle_count}-cycle post-run baseline trend: {leak_data.retained_growth_pct * 100:+.1f}%")
        if leak_data.is_leak:
            lines.append(f"    WARNING: {leak_data.message}")
    lines.append("")

    # Active Alerts section
    if active_alerts:
        lines.append("-" * 80)
        lines.append(f"ACTIVE ALERTS ({len(active_alerts)}):")
        for alert in active_alerts:
            lines.append(f"  [{alert.severity.value}] {alert.name} ({alert.scope}): {alert.summary}")
            if alert.runbook_anchor:
                lines.append(f"       Runbook: docs/RUNBOOKS.md{alert.runbook_anchor}")
        lines.append("-" * 80)
    else:
        lines.append("Active Alerts: None (all systems nominal)")
        lines.append("-" * 80)

    return "\n".join(lines)


def render_diagnostics(
    *,
    resource_data: dict[str, Any],
    postgres_data: Any,
    storage_data: Any,
    lifecycle_data: Any,
    gfs_state: Any,
    gefs_state: Any,
    recent_events: list[Any],
) -> str:
    """Render deep operational diagnostics including locked sessions and stage latencies."""
    lines: list[str] = [
        "=" * 80,
        "                    PLATFORM OPERATIONAL DIAGNOSTICS",
        "=" * 80,
    ]

    # Process internals
    lines.append("1. Process & Runtime Internals:")
    lines.append(f"   Python Version: {sys.version.split()[0]} on {sys.platform}")
    lines.append(f"   Active Threads: {resource_data.get('threads')}")
    lines.append(f"   Asyncio Tasks: {resource_data.get('asyncio_tasks')}")
    lines.append(f"   Open FDs/Handles: {resource_data.get('open_fds')}")
    gc_stats = resource_data.get("gc", {})
    lines.append(f"   GC Generations: {gc_stats.get('counts')} (thresholds: {gc_stats.get('threshold')})")
    lines.append("")

    # PostgreSQL catalog tables breakdown
    lines.append("2. PostgreSQL Relational Catalog Statistics:")
    if postgres_data.connected:
        lines.append(f"   {'Table Name':<28} {'Total Size':<14} {'Live Rows':<12} {'Dead Tuples':<12}")
        lines.append("   " + "-" * 66)
        for tname, stat in postgres_data.tables.items():
            lines.append(f"   {tname:<28} {_format_bytes(stat.total_bytes):<14} {stat.live_tuples:<12} {stat.dead_tuples:<12}")
        lines.append("")
        if postgres_data.long_running_queries:
            lines.append("   Active Long-Running Queries:")
            for q in postgres_data.long_running_queries:
                lines.append(f"     PID {q.pid} ({q.duration_seconds}s, {q.state}): {q.query_snippet}")
        else:
            lines.append("   No long-running queries detected.")
    else:
        lines.append(f"   PostgreSQL disconnected: {postgres_data.error}")
    lines.append("")

    # Pipeline Durations Breakdown
    lines.append("3. Recent Ingestion Pipeline Milestones:")
    for model_name, st in [("GFS", gfs_state), ("GEFS", gefs_state)]:
        if st is not None:
            lines.append(f"   {model_name}:")
            lines.append(f"     Cold-start delay:    {st.cold_start_s:.2f}s")
            lines.append(f"     Prepare run store:   {st.prep_store_s:.2f}s")
            lines.append(f"     Pre-update markers:  {st.pre_update_s:.2f}s")
            lines.append(f"     Finalization:        {st.finalize_s:.2f}s")
            lines.append(f"     Total run duration:  {st.last_duration_s:.2f}s")
    lines.append("")

    # Recent Alert History
    lines.append("4. Recent Alert Events History:")
    if recent_events:
        for ev in recent_events[-10:]:
            lines.append(f"   [{ev.event_type.upper()}] {ev.alert.severity.value} - {ev.alert.name} ({ev.alert.scope}): {ev.alert.summary}")
    else:
        lines.append("   No recent alert transitions recorded.")
    lines.append("=" * 80)

    return "\n".join(lines)


def render_audit(
    *,
    lifecycle_data: Any,
    gefs_completeness: dict[str, Any],
    gfs_completeness: dict[str, Any] | None = None,
) -> str:
    """Render deep invariant, tombstone anti-resurrection, and store consistency audit."""
    lines: list[str] = [
        "=" * 80,
        "             DATA LIFECYCLE V3 & DATA INTEGRITY AUDIT",
        "=" * 80,
    ]

    # Invariants
    lines.append("1. Lifecycle Contract Invariants & Anti-Resurrection:")
    if not lifecycle_data.violations:
        lines.append("   [PASS] No lifecycle state transition violations detected.")
        lines.append("   [PASS] Permanent anti-resurrection tombstones intact (0 resurrected runs).")
    else:
        lines.append(f"   [FAIL] {len(lifecycle_data.violations)} VIOLATIONS DETECTED:")
        for v in lifecycle_data.violations:
            lines.append(f"     - [{v.violation_type}] Model: {v.model_id}, Cycle: {v.cycle_time}: {v.description}")
    lines.append("")

    # GFS completeness
    if gfs_completeness is not None:
        lines.append("2. GFS Canonical Horizon Completeness Audit:")
        lines.append(f"   Latest active cycle: {gfs_completeness.get('cycle_time', 'N/A')}")
        lines.append(f"   Committed leads:     {gfs_completeness.get('committed_leads', 0)}/{gfs_completeness.get('expected_leads', 0)} (max lead: {gfs_completeness.get('max_lead_hours', 0)}h)")
        lines.append(f"   Runtime servable:    {'YES' if gfs_completeness.get('servable') else 'NO'}")
        lines.append(f"   Formal ready:        {'YES' if gfs_completeness.get('ready') else 'NO'}")
        lines.append("")

    # GEFS completeness
    lines.append("3. GEFS Perturbation Completeness Audit:" if gfs_completeness else "2. GEFS Perturbation Completeness Audit:")
    lines.append(f"   Latest active cycle: {gefs_completeness.get('cycle_time', 'N/A')}")
    lines.append(f"   Available members:   {gefs_completeness.get('members', 0)}/{gefs_completeness.get('expected_members', 30)}")
    lines.append(f"   Committed leads:     {gefs_completeness.get('committed_leads', 0)}/{gefs_completeness.get('expected_leads', 0)} (max lead: {gefs_completeness.get('max_lead_hours', 0)}h)")
    lines.append(f"   Runtime servable:    {'YES' if gefs_completeness.get('servable') else 'NO'}")
    lines.append(f"   Formal ready:        {'YES' if gefs_completeness.get('ready') else 'NO'}")
    lines.append("=" * 80)

    return "\n".join(lines)
