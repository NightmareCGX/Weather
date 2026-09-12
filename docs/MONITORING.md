# Global Probabilistic Weather Platform — Runtime Monitoring Specification

This document is the authoritative reference for the runtime health, resource monitoring, ingestion metrics, Data Lifecycle V3 observability, data integrity verification, and alerting subsystem.

---

## 1. Monitoring Architecture & Design Principles

The monitoring system provides continuous, non-intrusive observability across the entire platform without altering data lifecycle semantics, adding query overhead, or breaking ingestion on monitoring failures.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PRODUCER SERVICES                               │
├──────────────────────────────┬──────────────────────────────────────────────┤
│      services/ingestion      │                 services/api                 │
│  • SystemResourceCollector   │  • Process Memory & CPU Probes               │
│  • CycleResourceLeakDetector │  • ReaderLockPool Metrics                    │
│  • PostgresHealthCollector   │  • Database / Redis / S3 Connectivity        │
│  • StorageHealthCollector    │  • /v1/metrics (Prometheus Format)           │
│  • LifecycleHealthCollector  │  • /v1/health & /v1/health/detailed          │
│  • IngestionHealthCollector  │                                              │
│  • AlertEngine & Deduplicator│                                              │
└──────────────┬───────────────┴──────────────────────┬───────────────────────┘
               │                                      │
               ▼                                      ▼
┌──────────────────────────────┐       ┌──────────────────────────────────────┐
│  OPERATOR STATUS CLI         │       │    PROMETHEUS SCRAPING / EXPORTER    │
│  • weather-ingest status     │       │  • curl http://<API>/v1/metrics      │
│  • weather-ingest diagnostics│       │  • weather-ingest metrics [--port P] │
│  • weather-ingest audit      │       └──────────────────┬───────────────────┘
│  • weather-ingest alert-check│                          │
└──────────────┬───────────────┘                          ▼
               │                       ┌──────────────────────────────────────┐
               ▼                       │          GRAFANA DASHBOARD           │
┌──────────────────────────────┐       │  (Platform Overview, Resources,      │
│   ALERT DISPATCH SINKS       │       │   Ingestion, Postgres, MinIO,        │
│  • Structured Python Logging │       │   Lifecycle, Sweeper, Reclamation)   │
│  • Webhook (Slack / Discord) │       └──────────────────────────────────────┘
│  • In-Memory Event Ledger    │
└──────────────────────────────┘
```

### Core Operational Guarantees:
1. **Zero External Metric Dependencies:** Built using lightweight, thread-safe primitives that export standard Prometheus text format (`version=0.0.4`) directly without third-party C-extensions or external wheels.
2. **Fail-Open Safety (TASK 27):** All monitoring collectors and alert sinks are wrapped in exception barriers. A database, MinIO, or webhook failure **never** cancels an ingestion wave, interrupts finalization, or blocks serving.
3. **Cardinality Safety (TASK 26):** Metric labels are strictly bounded to small enums (e.g. `model="gfs"|"gefs"`, `phase="download"|"decode"|"write"|"finalize"`, `status="queued"|"deleting"|"deleted"|"failed"`). Unbounded dimensions (cycle timestamps, run IDs, S3 paths) are excluded from Prometheus metric labels and routed exclusively to diagnostic snapshots and structured logs.
4. **Cost-Class Separation (TASK 25):**
   - **Fast Metrics (15–60s):** Process CPU, RSS, threads, tasks, connection pool checked-out counts, in-flight pipeline progress counters.
   - **Medium Health Checks (2–5m):** Ingestion lag, stuck pipeline detectors, database table sizes and dead tuples, deletion claim ages, retention sweeper backlogs.
   - **Deep Audits (15–30m or on-demand):** Anti-resurrection cross-catalog audits, invariant checks, store consistency validation.

### 1.1 Process-Local Prometheus Scraping Topology

Prometheus metrics are organized into two distinct physical exposition surfaces matching service boundaries:

1. **API Serving Process (`services/api`):**
   - **Endpoint:** `GET http://<API_HOST>:8000/v1/metrics`
   - **Exposed Data:** Process-local API resources (CPU, RSS, VMS, threads), API `ReaderLockPool` connection pool stats, and live dependency reachability probes (`database`, `redis`, `storage`).
   - **Topology Rule:** The API process **never** claims to aggregate or proxy process-local ingestion worker metrics.

2. **Ingestion Worker & Daemon Processes (`services/ingestion`):**
   - **Endpoint:** `GET http://<INGESTION_HOST>:<PORT>/metrics` (via `weather-ingest metrics --port <PORT>`).
   - **Exposed Data:** Ingestion pipeline stage counters, throughput, stuck warnings, model completeness, cycle durations, and multi-cycle baseline memory leak indicators. Also observes shared PostgreSQL capacity and MinIO latency.
   - **Multi-Worker Scrapes:** In deployments with multiple independent ingestion workers, Prometheus scrapes each worker instance endpoint directly (adding standard Prometheus `instance` labels).

> **Deployment Boundary Note:** Prometheus server collection, Alertmanager routing, and Grafana hosting are external infrastructure components. The platform provides standard exposition endpoints and an authoritative dashboard specification (`monitoring/grafana/weather_platform_dashboard.json`), rather than bundling containerized Prometheus/Grafana servers.

---

## 2. Complete Metric Catalog

### 2.1 Process & Host Resource Metrics

| Metric Name | Type | Description | Labels | Cost Class |
| :--- | :--- | :--- | :--- | :--- |
| `weather_process_cpu_percent` | Gauge | Current process CPU utilization percentage (0–100+) | - | Fast |
| `weather_process_memory_rss_bytes` | Gauge | Process resident set size in bytes | - | Fast |
| `weather_process_memory_vms_bytes` | Gauge | Process virtual memory size in bytes | - | Fast |
| `weather_process_memory_peak_rss_bytes` | Gauge | Peak recorded RSS in bytes | - | Fast |
| `weather_process_active_threads` | Gauge | Number of active Python threads | - | Fast |
| `weather_process_asyncio_tasks` | Gauge | Active tasks in current asyncio event loop | - | Fast |
| `weather_process_open_file_descriptors` | Gauge | Open file descriptors / handles (-1 if unsupported) | - | Fast |
| `weather_disk_total_bytes` | Gauge | Total filesystem capacity in bytes | `path` | Fast |
| `weather_disk_used_bytes` | Gauge | Used filesystem space in bytes | `path` | Fast |
| `weather_disk_free_bytes` | Gauge | Free filesystem space in bytes | `path` | Fast |
| `weather_disk_used_percent` | Gauge | Filesystem utilization percentage (0–100) | `path` | Fast |
| `weather_memory_leak_warning` | Gauge | 1 if sustained post-cycle baseline memory growth detected, 0 otherwise | - | Medium |

### 2.2 PostgreSQL Database Metrics

| Metric Name | Type | Description | Labels | Cost Class |
| :--- | :--- | :--- | :--- | :--- |
| `weather_postgres_connected` | Gauge | PostgreSQL connectivity status (1=up, 0=down) | - | Fast |
| `weather_postgres_active_connections` | Gauge | Active connections from `pg_stat_activity` | - | Medium |
| `weather_postgres_max_connections` | Gauge | Configured `max_connections` limit | - | Medium |
| `weather_postgres_connection_utilization_percent`| Gauge | Active / max connections percentage | - | Medium |
| `weather_postgres_pool_checked_out` | Gauge | Checked-out connections in SQLAlchemy QueuePool | - | Fast |
| `weather_postgres_pool_size` | Gauge | Configured base QueuePool size | - | Fast |
| `weather_postgres_pool_overflow` | Gauge | Current burst overflow connection count | - | Fast |
| `weather_postgres_total_size_bytes` | Gauge | Total database size from `pg_database_size` | - | Medium |
| `weather_postgres_table_size_bytes` | Gauge | Relation size (table + indexes) | `table_name` | Medium |
| `weather_postgres_table_dead_tuples` | Gauge | Estimated dead tuples from `pg_stat_user_tables` | `table_name` | Medium |
| `weather_postgres_long_running_transactions` | Gauge | Active transactions open longer than threshold | - | Medium |
| `weather_postgres_lock_waits` | Gauge | Number of sessions blocked waiting on locks | - | Medium |
| `weather_postgres_deadlocks_total` | Gauge | Cumulative server deadlocks count | - | Medium |

### 2.3 Object Storage (MinIO / S3) Metrics

| Metric Name | Type | Description | Labels | Cost Class |
| :--- | :--- | :--- | :--- | :--- |
| `weather_storage_connected` | Gauge | Object storage connectivity (1=connected, 0=disconnected) | - | Fast |
| `weather_storage_probe_latency_milliseconds` | Gauge | Health probe latency in milliseconds | - | Fast |
| `weather_storage_operations_total` | Counter | Total storage operations | `operation` (`get`, `put`, `head`, `list`, `delete`) | Fast |
| `weather_storage_operation_failures_total` | Counter | Total failed storage operations | `operation` | Fast |
| `weather_storage_operation_retries_total` | Counter | Total retried storage operations | `operation` | Fast |
| `weather_storage_operation_duration_seconds` | Histogram| Operation latency histogram | `operation` | Fast |
| `weather_storage_phase_duration_seconds` | Gauge | Duration of sensitive storage phases | `phase` (`prepare_run_store`, `marker_put`, `manifest_write`, `prefix_deletion`) | Medium |

### 2.4 Ingestion & Pipeline Health Metrics

| Metric Name | Type | Description | Labels | Cost Class |
| :--- | :--- | :--- | :--- | :--- |
| `weather_ingestion_last_start_timestamp_seconds` | Gauge | Unix timestamp when ingestion last started | `model` (`gfs`, `gefs`) | Fast |
| `weather_ingestion_last_success_timestamp_seconds`| Gauge | Unix timestamp when ingestion last succeeded | `model` | Fast |
| `weather_ingestion_last_failure_timestamp_seconds`| Gauge | Unix timestamp when ingestion last failed | `model` | Fast |
| `weather_ingestion_cycle_duration_seconds` | Gauge | Total elapsed duration of last completed cycle | `model` | Fast |
| `weather_ingestion_cold_start_duration_seconds` | Gauge | Delay from seed download start to 1st non-seed download | `model` | Fast |
| `weather_ingestion_prepare_store_duration_seconds`| Gauge | Duration to initialize Zarr store and metadata | `model` | Fast |
| `weather_ingestion_pre_update_duration_seconds` | Gauge | Duration to stage UPDATING markers | `model` | Fast |
| `weather_ingestion_finalization_duration_seconds` | Gauge | Duration of marker validation, manifest write & catalog reconciliation | `model` | Fast |
| `weather_ingestion_items_completed_total` | Counter | Completed target regions | `model`, `phase` (`download`, `decode`, `write`, `finalize`) | Fast |
| `weather_ingestion_items_failed_total` | Counter | Failed target regions | `model`, `phase` | Fast |
| `weather_ingestion_stuck_warning` | Gauge | 1 if pipeline has zero progress beyond timeout, 0 otherwise | `model` | Fast |
| `weather_ingestion_lag_cycles` | Gauge | Lag in 6-hour cycles relative to available upstream data | `model` | Medium |
| `weather_ingestion_lag_hours` | Gauge | Lag in hours relative to available upstream data | `model` | Medium |
| `weather_model_committed_leads_count` | Gauge | Number of committed distinct leads for latest active cycle | `model` | Medium |
| `weather_model_expected_leads_count` | Gauge | Authoritative expected distinct leads count for model and version | `model` | Medium |
| `weather_model_max_lead_hours` | Gauge | Authoritative maximum lead time in hours for model and version | `model` | Medium |
| `weather_model_ready_status` | Gauge | 1 if fully committed and run is ready, 0 otherwise | `model` | Medium |
| `weather_gefs_available_members_count` | Gauge | Committed perturbation members (1..30) for latest active cycle | - | Medium |
| `weather_gefs_servable_status` | Gauge | 1 if $\ge 26/30$ members committed, 0 otherwise | - | Medium |
| `weather_gefs_ready_status` | Gauge | 1 if $30/30$ members committed and run is ready, 0 otherwise | - | Medium |

### 2.5 Lifecycle, Sweeper, Reclamation & Invariant Metrics

| Metric Name | Type | Description | Labels | Cost Class |
| :--- | :--- | :--- | :--- | :--- |
| `weather_lifecycle_active_cycles` | Gauge | Active cycles not yet claimed for deletion | - | Medium |
| `weather_lifecycle_claimed_cycles` | Gauge | Cycles claimed (`deletion_started_at`) awaiting physical store deletion | - | Medium |
| `weather_lifecycle_tombstone_cycles` | Gauge | Permanent anti-resurrection tombstones (`deleted_at`) | - | Medium |
| `weather_lifecycle_oldest_claim_age_seconds` | Gauge | Age in seconds of oldest in-flight deletion claim | - | Medium |
| `weather_lifecycle_stuck_claims_warning` | Gauge | Deletion claims older than 1 hour | - | Medium |
| `weather_lifecycle_stuck_claims_critical` | Gauge | Deletion claims older than 4 hours | - | Medium |
| `weather_metadata_sweeper_eligible_tombstones` | Gauge | Tombstones older than 14-day retention window | - | Medium |
| `weather_metadata_sweeper_unpurged_metadata_count`| Gauge | Tombstones older than 14 days still retaining detailed child metadata | - | Medium |
| `weather_metadata_sweeper_oldest_overdue_seconds` | Gauge | Age in seconds of oldest metadata overdue past 14-day deadline | - | Medium |
| `weather_reclamation_queue_count` | Gauge | Shard reclamation queue row counts | `status` (`queued`, `deleting`, `deleted`, `failed`) | Fast |
| `weather_reclamation_oldest_queued_age_seconds` | Gauge | Age in seconds of oldest queued reclamation target | - | Medium |
| `weather_reclamation_oldest_deleting_age_seconds` | Gauge | Age in seconds of oldest leased deleting target | - | Medium |
| `weather_reclamation_oldest_failed_age_seconds` | Gauge | Age in seconds of oldest quarantined failed target | - | Medium |
| `weather_lifecycle_invariant_violations_count` | Gauge | Number of detected lifecycle state transition violations | - | Deep |
| `weather_anti_resurrection_violations_count` | Gauge | Number of active or recreated runs under permanent tombstones | - | Deep |

---

## 3. Operator CLI Reference

The `weather-ingest` CLI provides four dedicated monitoring and diagnostic subcommands:

### 3.1 `weather-ingest status`
Prints the human-friendly operational health dashboard matching the approved format:
```bash
weather-ingest status
# Or output as JSON for automated scrapers:
weather-ingest status --json
```
**Example Output:**
```text
================================================================================
  GLOBAL PROBABILISTIC WEATHER PLATFORM — RUNTIME HEALTH SUMMARY
================================================================================
Platform Health: HEALTHY

PostgreSQL:
    connections 18/100 (18.0%)
    DB size 42.1 GB
    longest transaction 2.1s

MinIO:
    reachable (14.0ms)
    disk usage 61.0% (39.0 GB free)

GFS:
    latest ready cycle 12Z
    expected latest 12Z
    ingestion lag 0 cycles (0.0h)
    last duration 108s

GEFS:
    latest ready cycle 12Z
    expected latest 12Z
    ingestion lag 0 cycles (0.0h)
    last duration 38m50s

Lifecycle:
    active cycles 5
    claimed deletion 1
    stuck claims 0
    tombstones 214

Reclamation:
    queued 34
    deleting 2
    deleted 100
    failed 0

Metadata retention:
    eligible backlog 3
    oldest overdue 17m00s

Memory:
    RSS 1.8 GB (Peak: 2.0 GB)
    VMS 2.5 GB
    10-cycle post-run baseline trend: +1.2%

Active Alerts: None (all systems nominal)
--------------------------------------------------------------------------------
```

### 3.2 `weather-ingest diagnostics`
Outputs deep internal telemetry, active threads, asyncio tasks, GC generation stats, table size breakdown, long-running query snippets, stage latency breakdowns, and recent alert transition history.
```bash
weather-ingest diagnostics
```

### 3.3 `weather-ingest audit`
Executes deep invariant and anti-resurrection audits against the relational database and GEFS perturbation completeness:
```bash
weather-ingest audit
```
Exits with code `0` if all invariants pass, `1` if any consistency violation is detected.

### 3.4 `weather-ingest alert-check`
Evaluates all rules, dispatches notifications to configured sinks, and exits with standard monitoring exit codes:
```bash
weather-ingest alert-check
```
* `0`: All systems nominal / healthy.
* `1`: One or more `WARNING` alerts active.
* `2`: One or more `CRITICAL` alerts active.

### 3.5 `weather-ingest metrics`
Outputs the complete Prometheus metric catalog in standard exposition format (`text/plain; version=0.0.4`) to stdout:
```bash
weather-ingest metrics
# Or start a lightweight standalone Prometheus HTTP scraper on a port:
weather-ingest metrics --port 9100
```

---

## 4. Alert Rules & Severity Matrix

Alerts are classified as `INFO`, `WARNING`, or `CRITICAL`. Every alert links directly to a specific response section in `docs/RUNBOOKS.md`.

| Alert Rule Name | Severity | Trigger Condition | Runbook Anchor |
| :--- | :--- | :--- | :--- |
| `disk_space_warning` | WARNING | Staging/host disk used $\ge 80\%$ | `#disk-space-warning` |
| `disk_space_critical` | CRITICAL | Staging/host disk used $\ge 90\%$ | `#disk-space-critical` |
| `memory_leak_warning` | WARNING | Baseline RSS grows monotonically over 5+ cycles by $\ge 15\%$ and $\ge 150\text{ MB}$ | `#memory-leak` |
| `postgres_unreachable` | CRITICAL | PostgreSQL connection probe fails | `#database-unreachable` |
| `postgres_connections_warning` | WARNING | Active connections $\ge 80\%$ of `max_connections` | `#database-connections` |
| `postgres_connections_critical`| CRITICAL | Active connections $\ge 95\%$ of `max_connections` | `#database-connections` |
| `postgres_lock_waits` | WARNING | Blocked sessions waiting on locks $> 0$ | `#database-lock-waits` |
| `postgres_deadlocks_detected` | WARNING | New deadlocks occurred during collection interval (delta $> 0$) | `#database-deadlocks` |
| `minio_unreachable` | CRITICAL | Object storage health probe fails | `#minio-unreachable` |
| `ingestion_pipeline_stuck` | CRITICAL | Active/queued work present with zero progress for $> 10\text{ minutes}$ | `#ingestion-stuck` |
| `ingestion_lag_warning` | WARNING | 1 cycle behind available NOAA upstream | `#ingestion-lag` |
| `ingestion_lag_critical` | CRITICAL | 2+ cycles behind available NOAA upstream | `#ingestion-lag` |
| `finalizer_claim_stuck_warning`| WARNING | Physical deletion claim age $> 1\text{ hour}$ | `#stuck-deletion-claim` |
| `finalizer_claim_stuck_critical`| CRITICAL | Physical deletion claim age $> 4\text{ hours}$ | `#stuck-deletion-claim` |
| `metadata_sweeper_backlog_overdue`| WARNING| Tombstones older than 14 days retain metadata for $> 1\text{ day}$ overdue | `#sweeper-backlog` |
| `reclamation_failed_shards` | WARNING | Any records in `reclamation_queue` with `status='failed'` | `#reclamation-failures` |
| `reclamation_deleting_stuck` | WARNING | Reclaimed target in `deleting` status for $> 600\text{s}$ | `#reclamation-stuck` |
| `invariant_invalid_lifecycle_transition` | CRITICAL | Cycle has `deleted_at` set without prior `deletion_started_at` | `#anti-resurrection-violation` |
| `invariant_anti_resurrection_violation` | CRITICAL | Run recreated or active under permanent `deleted_at` tombstone | `#anti-resurrection-violation` |

---

## 5. Deduplication, Cooldown & Recovery

Alert spam is eliminated by the `AlertDeduplicator` state machine:
* **Initial Trigger:** Emits `triggered` event on first occurrence.
* **Cooldown Suppression:** While an alert remains in `WARNING` or `CRITICAL`, repeated notifications are suppressed for the cooldown period (default: `3600 seconds` / 1 hour).
* **Immediate Escalation:** If an alert escalates from `WARNING` to `CRITICAL`, the cooldown is bypassed and an `escalated` event is dispatched immediately.
* **Recovery Notification:** When an alert condition clears, a `recovered` event is emitted automatically, confirming resolution to operators.

### 5.1 Restart Semantics of In-Memory Monitoring State

Monitoring state is deliberately designed with clean, process-local restart semantics without requiring an external monitoring database:

1. **`AlertDeduplicator` State:**
   - Kept in-memory. On daemon or worker restart, active alert tracking initializes empty.
   - If an alertable condition (e.g. disk $> 80\%$) remains active upon restart, it triggers once on the first evaluation pass and enters cooldown suppression.
   - If an alert cleared while the service was down, no stale recovery notification is emitted.

2. **`MemoryAlertSink` Ring Buffer:**
   - The in-memory buffer of recent alert transitions (capacity: 100) resets on restart.
   - Structured log records (`[ALERT:...]`) remain durable in stdout/syslog and survive restarts.

3. **`CycleResourceLeakDetector` History:**
   - Process memory (RSS/VMS) is strictly local to an individual operating system process lifetime.
   - On worker restart, PID and address space reset; the leak detector reinitializes and requires 5 newly completed cycles in the new process before evaluating sustained leak trends. This prevents false positive leak alarms caused by comparing memory across distinct OS processes.

---

## 6. Alert Delivery Channels

1. **Structured Logging (Default):**
   Always active across all services. Emits tagged log records:
   - `[ALERT:WARNING] <name> (<scope>): <description>`
   - `[ALERT:CRITICAL] <name> (<scope>): <description>`
   - `[ALERT:RECOVERY] <name> (<scope>): <summary> recovered`
2. **HTTP Webhook (Optional):**
   Configured via environment variable:
   ```bash
   ALERT_WEBHOOK_URL="https://alerts.example.com/webhook"
   ```
   Dispatches JSON payloads containing event type, severity, summary, value, threshold, and runbook URL. The webhook sink executes with a 2-second timeout and fails open on any connection or transport error.
3. **In-Memory Event Ledger:**
   Stores the last 100 alert events for immediate inspection via `weather-ingest diagnostics`.
