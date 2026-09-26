# Weather Platform — Operational Runbooks & Deployment Framework

This document is the authoritative operational runbook framework for deploying, operating, diagnosing, and recovering the Global Probabilistic Weather Platform.

## Operator Utility Scripts (`scripts/`)

| Script | Purpose | Typical invocation |
|---|---|---|
| `shadow_v2.py` | sharded_v2 rollout gate: validate a canonical v1 store against its shadow-v2 re-encoding (numerical equivalence verdict), or clean up shadow-namespace stores. See docs/DEPLOYMENT.md §3.2. | `uv run --no-sync python scripts/shadow_v2.py validate ...` |
| `hydrate_city_elevations.py` | Offline batch backfill of `cities.elevation_m` from Open-Meteo (pairs with migration 006). | `uv run --no-sync python scripts/hydrate_city_elevations.py [--batch-size 100] [--limit 1000] [--dry-run]` |
| `smoke_collectors_runtime.py` | Live smoke audit that the monitoring collectors return real values against the running PostgreSQL/MinIO. | `uv run --no-sync python scripts/smoke_collectors_runtime.py` |
| `verify_monitoring_runtime.py` | Live verification of the alert engine, metrics, and recovery behaviors end to end. | `uv run --no-sync python scripts/verify_monitoring_runtime.py` |
| `cleanup_test_pollution.py` | One-time cleanup of pytest pollution rows in a live catalog (see docs/lifecycle-v3-architecture.md). | `uv run --no-sync python scripts/cleanup_test_pollution.py` |
| `check_dependency_alignment.py` | CI gate asserting shared runtime dependency alignment across workspace pyprojects (run automatically in CI; safe to run manually). | `uv run --no-sync python scripts/check_dependency_alignment.py` |

---

## 1. Parameter Placeholder Conventions

This framework is cloud- and topology-agnostic. Specific hardware sizing, IP addresses, worker counts, and credentials are intentionally parameterized with placeholders to be finalized during **Stage 8 Server Deployment**:

| Placeholder | Meaning / Governing Source |
| :--- | :--- |
| `<PRODUCTION_DATABASE_URL>` | Full PostgreSQL connection string with production user and credentials |
| `<PRODUCTION_REDIS_URL>` | Full Redis connection string |
| `<OBJECT_STORAGE_ENDPOINT>` | Production S3-compatible endpoint (e.g. `s3.us-east-1.amazonaws.com` or custom S3 FQDN) |
| `<OBJECT_STORAGE_BUCKET>` | Target S3 bucket holding forecast cycle stores (e.g. `weather-platform-prod`) |
| `<OBJECT_STORAGE_ACCESS_KEY>`| Authenticated access key ID or IAM role |
| `<OBJECT_STORAGE_SECRET_KEY>`| Authenticated secret access key |
| `<API_SERVICE_URL>` | Internal/external URL where the FastAPI service is reachable |
| `<API_WORKERS>` | Number of Uvicorn worker processes per host (`FINALIZE IN STAGE 8`) |
| `<POSTGRES_MAX_CONNECTIONS>` | Total PostgreSQL server connection ceiling (`FINALIZE IN STAGE 8`) |
| `<CONTAINER_STOP_GRACE>` | Container runtime stop timeout in seconds ($\ge 50\text{s}$) |
| `<RETENTION_POLICY>` | Number of historical forecast cycles retained before GC deletion |

---

## 2. Logical Production Topology

```text
       ┌────────────────────────────────────────────────────────┐
       │                   End Users / Clients                  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                     Reverse Proxy                      │
       │           (TLS Termination / Load Balancing)           │
       └──────────────┬──────────────────────────┬──────────────┘
                      │                          │
                      │ (Static & SSR)           │ (/v1/* API Proxy)
                      ▼                          ▼
       ┌──────────────────────────────┐ ┌──────────────────────────────┐
       │      services/frontend       │ │         services/api         │
       │   (Next.js Standalone Node)  │ │   (FastAPI / Uvicorn Server) │
       └──────────────────────────────┘ └──────────────┬───────────────┘
                                                       │
                      ┌────────────────────────────────┼────────────────────────────────┐
                      │                                │                                │
                      ▼                                ▼                                ▼
       ┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
       │         PostgreSQL 18        │ │           Redis 7            │ │     Object Storage (S3)      │
       │      (+ PostGIS Extension)   │ │    (Hot-Cache Acceleration)  │ │   (s3://<BUCKET>/<model>/)   │
       │  • Relational Catalog & Runs │ └──────────────────────────────┘ └──────────────▲───────────────┘
       │  • Advisory Lock Gates       │                                                 │
       └──────────────▲───────────────┘                                                 │
                      │                                                                 │
                      ├────────────────────────────────┬────────────────────────────────┘
                      │                                │
                      ▼                                ▼
       ┌──────────────────────────────┐ ┌──────────────────────────────┐
       │      Ingestion Service       │ │          GC Daemon           │
       │  • weather-ingest realtime   │ │   • weather-ingest gc        │
       │  • NOAA Upstream Download    │ │   • Cycle Reconciler         │
       │  • DecodePool & Sharded Write│ │   • Deletion Fencing         │
       └──────────────────────────────┘ └──────────────────────────────┘
```

---

## 3. Pre-Deployment Verification Checklist

Before initiating any deployment:

- [ ] **1. CI Verification:** The target Git commit/tag has passed `ci-required` in GitHub Actions.
- [ ] **2. Database Connectivity & Capacity:**
  - Verify PostgreSQL is reachable via `pg_isready -d <PRODUCTION_DATABASE_URL>`.
  - Verify PostGIS extension is installed (`SELECT PostGIS_Version();`).
  - Verify `max_connections` meets or exceeds the Stage 7F connection budget formula:
    $$\text{TOTAL\_DB\_CONNECTIONS} = W_{\text{API}} \times 39 + N_{\text{ING}} \times 16 + N_{\text{GC}} \times 16 + 12$$
- [ ] **3. Object Storage Access:**
  - Verify `<OBJECT_STORAGE_BUCKET>` exists.
  - Verify credentials have `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, and `s3:ListBucket` permissions.
  - Verify `MINIO_SECURE=true` for all external/public S3 endpoints.
- [ ] **4. Frontend Build Target:**
  - Verify `weather-frontend` Docker image was built with `--build-arg API_PROXY_TARGET=<API_SERVICE_URL>`.
- [ ] **5. Container Grace Period:**
  - Verify container runtime stop timeout is set to at least 50 seconds ($\ge \text{API\_SHUTDOWN\_DRAIN\_TIMEOUT\_SECONDS} + 10\text{s}$).
- [ ] **6. Ingestion Disk Staging:**
  - Verify local staging directory (`downloads/`) has sufficient temporary disk capacity for multi-lead wave downloads.

---

## 4. Fresh Deployment & Initial Bootstrap Runbook

Execute these steps in strict sequence for a brand-new production deployment:

### Step 1: Initialize Database Schema
Run Alembic migrations against the production database:
```bash
cd services/api
DATABASE_URL="<PRODUCTION_DATABASE_URL>" uv run --no-sync alembic upgrade head
```
*Verification:* Confirm Alembic head revision:
```bash
DATABASE_URL="<PRODUCTION_DATABASE_URL>" uv run --no-sync alembic current
```

### Step 2: Seed Spatial Reference Data (If Configured)
The repository ships no SQL seed dumps — the `seeds/` directory is not part of
the repo. Reference cities, ski resorts, and stations used by the test suites
are provisioned from the fixtures documented in
`services/api/tests/fixtures/README.md`. If a deployment carries a
site-provided SQL seed file for reference locations, apply it manually:
```bash
psql "<PRODUCTION_DATABASE_URL>" -f <SEED_FILE>.sql
```

### Step 3: Start Core API Serving Service
Launch the FastAPI serving tier:
```bash
cd services/api
DATABASE_URL="<PRODUCTION_DATABASE_URL>" \
REDIS_URL="<PRODUCTION_REDIS_URL>" \
MINIO_ENDPOINT="<OBJECT_STORAGE_ENDPOINT>" \
MINIO_ACCESS_KEY="<OBJECT_STORAGE_ACCESS_KEY>" \
MINIO_SECRET_KEY="<OBJECT_STORAGE_SECRET_KEY>" \
MINIO_SECURE="true" \
uv run --no-sync uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers <API_WORKERS>
```

### Step 4: Start Frontend Service
Launch the Next.js frontend standalone server:
```bash
cd services/frontend
PORT=3000 node server.js
```

### Step 5: Perform Initial Full-Cycle Baseline Bootstrap
> **Operational Note on Known Technical Debt:** The realtime scheduler operates on incremental lead waves and does not automatically perform a full cold-start historical bulk baseline. A fresh deployment **requires** a manual baseline ingestion of the latest complete model runs before launching the realtime scheduler.

```bash
cd services/ingestion
# Ingest baseline GFS deterministic run (00Z or 12Z cycle, full 0–240h canonical horizon):
# (In POSIX bash, `$(seq 0 3 240)` can be used for the lead list):
DATABASE_URL="<PRODUCTION_DATABASE_URL>" \
MINIO_ENDPOINT="<OBJECT_STORAGE_ENDPOINT>" \
MINIO_ACCESS_KEY="<OBJECT_STORAGE_ACCESS_KEY>" \
MINIO_SECRET_KEY="<OBJECT_STORAGE_SECRET_KEY>" \
MINIO_SECURE="true" \
MINIO_BUCKET_NAME="<OBJECT_STORAGE_BUCKET>" \
uv run --no-sync weather-ingest ingest \
  --model gfs \
  --cycle-date <LATEST_CYCLE_DATE> \
  --cycle-hour <LATEST_CYCLE_HOUR> \
  --lead-time-hours 0 3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60 63 66 69 72 75 78 81 84 87 90 93 96 99 102 105 108 111 114 117 120 123 126 129 132 135 138 141 144 147 150 153 156 159 162 165 168 171 174 177 180 183 186 189 192 195 198 201 204 207 210 213 216 219 222 225 228 231 234 237 240
```

### Step 6: Start Realtime Lead-Wave Scheduler
Start the background realtime scheduling daemon:
```bash
cd services/ingestion
REALTIME_ENABLED="true" \
DATABASE_URL="<PRODUCTION_DATABASE_URL>" \
MINIO_ENDPOINT="<OBJECT_STORAGE_ENDPOINT>" \
MINIO_ACCESS_KEY="<OBJECT_STORAGE_ACCESS_KEY>" \
MINIO_SECRET_KEY="<OBJECT_STORAGE_SECRET_KEY>" \
MINIO_SECURE="true" \
MINIO_BUCKET_NAME="<OBJECT_STORAGE_BUCKET>" \
uv run --no-sync weather-ingest realtime
```

### Step 7: Start Retention Garbage Collection (GC) Daemon
Launch the background storage retention reconciler:
```bash
cd services/ingestion
DATABASE_URL="<PRODUCTION_DATABASE_URL>" \
MINIO_ENDPOINT="<OBJECT_STORAGE_ENDPOINT>" \
MINIO_ACCESS_KEY="<OBJECT_STORAGE_ACCESS_KEY>" \
MINIO_SECRET_KEY="<OBJECT_STORAGE_SECRET_KEY>" \
MINIO_SECURE="true" \
MINIO_BUCKET_NAME="<OBJECT_STORAGE_BUCKET>" \
uv run --no-sync weather-ingest gc --interval-seconds 1800
```

---

## 5. Database Migration Runbook

### Preconditions & Safety:
1. Capture a database snapshot/backup before running migrations.
2. Confirm no active DDL locks are present.

### Execution Procedure:
```bash
cd services/api

# 1. Check pending revisions:
DATABASE_URL="<PRODUCTION_DATABASE_URL>" uv run --no-sync alembic current

# 2. Apply migrations to head:
DATABASE_URL="<PRODUCTION_DATABASE_URL>" uv run --no-sync alembic upgrade head

# 3. Verify schema state:
DATABASE_URL="<PRODUCTION_DATABASE_URL>" uv run --no-sync alembic current
```

### Migration Failure Handling:
* If `alembic upgrade head` fails with an SQL error, Alembic will roll back the active transaction.
* Inspect PostgreSQL logs for lock timeouts or constraint violations.
* **Do NOT execute blind downgrades** on production databases without verifying table data dependencies.

---

## 6. Service Startup & Logical Dependency Order

```text
1. Backing Infrastructure: PostgreSQL 18 (PostGIS) ──► Redis 7 ──► S3 Object Storage
                                │
                                ▼
2. Core Serving Tier:      FastAPI (services/api)
                                │
                                ▼
3. User Interface:         Next.js (services/frontend)
                                │
                                ▼
4. Background Ingestion:   Realtime Scheduler (services/ingestion) ──► GC Daemon
```

### Resilient Startup Behavior:
* If the API starts before Redis is reachable, the API starts normally and logs a cache connection warning; `/v1/health` reports `status: degraded` (HTTP 503) until Redis connects.
* If Ingestion starts before PostgreSQL is reachable, it retries with exponential backoff.

---

## 7. Realtime Scheduler Operational Runbook

### Starting the Scheduler:
```bash
cd services/ingestion
uv run --no-sync weather-ingest realtime
```

### Operational Invariants & Leadership:
* The scheduler acquires a session-level PostgreSQL advisory lock (`scheduler_leader_key`).
* Exactly one active leader process executes discovery, wave planning, and dispatch per deployment identity.
* Standby scheduler instances will fail non-blocking `try_lock` and sleep until the leader releases the lock or disconnects.

### Verifying Operational Health:
1. **Check Leadership in PostgreSQL:**
   ```sql
   SELECT pid, application_name, query, state 
   FROM pg_stat_activity 
   WHERE pid IN (
       SELECT pid FROM pg_locks WHERE locktype = 'advisory'
   );
   ```
2. **Inspect Wave Dispatch Logs:**
   - Look for log line: `realtime leadership acquired (advisory key ...)`
   - Look for wave dispatch: `realtime wave planned for cycle <YYYY-MM-DDTHHZ> (leads: [0, 3, 6])`
   - Look for finalization: `coalesced finalization committed manifest generation <UUID>`

### Restarting the Scheduler:
* Send `SIGINT` or `SIGTERM` to the process.
* The scheduler sets its stop event, allows any in-flight wave to complete its finalizer commit, releases the advisory lock cleanly, and exits.
* *Cold Backlog Catch-Up Note:* If the scheduler was down for several hours, it will sequentially discover and dispatch pending backlogged leads upon restart.

---

## 8. Retention Garbage Collection (GC) Runbook

### Invocation Modes:
1. **Continuous Daemon Mode (Recommended):**
   ```bash
   uv run --no-sync weather-ingest gc --interval-seconds 1800 --bucket <OBJECT_STORAGE_BUCKET>
   ```
   Each daemon pass runs the full Lifecycle V3 pipeline: bookkeeping ->
   planner -> worker -> sweeper, gated by two-level authorization flags (see
   below), and — by default every 24 hours — ends with the scheduled
   store<->catalog orphan inventory stage (see below). The daemon holds the
   GC advisory lock; no cron wiring is required.
2. **Single-Pass Mode (Cron / Manual Execution):**
   ```bash
   uv run --no-sync weather-ingest gc --once --bucket <OBJECT_STORAGE_BUCKET>
   ```
   Runs one full pipeline pass (with the same authorization flags) and exits.
   The scheduled inventory stage never runs in `--once` mode.
3. **Dry-Run Inspection Mode:**
   ```bash
   uv run --no-sync weather-ingest gc --once --dry-run
   ```

### Pipeline Wiring (V3 planner -> worker -> bookkeeping mainline):
Without authorization flags the daemon only runs lifecycle bookkeeping (zero
physical storage operations). The reclamation mainline is staged in two
levels so deletion is always an explicit operator decision:

| Stage | Flag | Env fallback | Effect |
|---|---|---|---|
| Planner | `--enable-planner` | `RECLAMATION_PLANNER_ENABLED=true` | Enqueue reclaimable shard targets into `reclamation_queue` each pass (queue writes only, no physical side effects). |
| Worker | `--enable-delete` | `RECLAMATION_DELETE_ENABLED=true` | Physically delete enqueued shard targets each pass. **DANGEROUS.** |
| Sweeper | `--enable-sweeper` | `RECLAMATION_SWEEPER_ENABLED=true` | Include the M3 14-day metadata retention pass each pass. |

Recommended rollout: enable planner only, observe
`GC pass: ... planner enqueued=N ...` summaries for at least one full
interval, then enable `--enable-delete`. Each pass prints a structured
summary (`bookkeeping ...; planner ...; worker ...; sweeper ...`); stages are
failure-isolated so a transient DB/S3 fault never kills the daemon.

### Scheduled Orphan Inventory (store <-> catalog reconciliation, architecture doc §9):
In daemon mode the GC automatically runs one orphan inventory pass every
`--inventory-interval-hours` (default 24; `0` disables; env fallback
`GC_INVENTORY_INTERVAL_HOURS`). The first scheduled inventory runs one full
interval after daemon start. When a pass is due, the inventory summary is
appended to that pass's stdout summary:

```
GC pass: bookkeeping ...; sweeper ...; inventory discovered=42 orphans=1 beyond_frontier=0 errors=0
```

* **Reconciliation is discovery and reporting only.** The scheduled stage
  invokes `run_orphan_inventory(reap=False)` — no scheduled code path may
  physically delete orphan prefixes. Physical orphan cleanup remains a
  deliberate operator decision via the one-shot
  `weather-ingest gc --inventory [--inventory-reap]` command (unchanged).
* **Reading results:** `orphans=N` is the number of physical cycle stores
  with no catalog identity; `beyond_frontier=K` counts those whose whole
  serving horizon has expired (catalog recovery impossible, reap-eligible).
  Orphans *within* the frontier should be investigated for catalog recovery
  from COMPLETE marker evidence — never auto-reaped.
* **Metrics & alerts:** the scheduled stage publishes
  `weather_gc_inventory_orphans{beyond_frontier="true|false"}`,
  `weather_gc_inventory_last_success_timestamp`, and
  `weather_gc_inventory_errors_total` on the GC daemon's `--metrics-port`
  endpoint (default off, e.g. `9114`). A within-frontier orphan count > 0
  raises the `gc_orphan_stores_detected` webhook/log alert.
* **Manual reap procedure (unchanged):** run `weather-ingest gc --inventory`
  to list orphans with their `REAPABLE` / `recoverable-frontier` classification,
  review each path, then `weather-ingest gc --inventory --inventory-reap`
  to delete beyond-frontier prefixes under the exclusive store gate (fail-closed
  sanity guards refuse when a lifecycle row or a raced catalog run exists).

### Operational Invariants:
* **V3 Granular Reclamation:** physical deletion happens exclusively through
  the reclamation planner + worker at (variable, valid_time) granularity; the
  bookkeeping pass itself never deletes storage. Whole-cycle prefix deletion
  is retired.
* **Deletion Fencing:** the planner/worker honour the serving fence
  (`deletion_started_at`) on `forecast_cycle_lifecycle`; claimed cycles
  contribute no canonical candidates and are fully reclaimable.
* **Crash Safety:** worker claims use leases; a crashed worker's stale claims
  are recoverable via `weather-ingest reclamation requeue` (which now also
  recovers stuck `queued`/`deleting` rows over already-deleted physical
  objects, unblocking tombstone terminality).

---

## 9. API Serving & Graceful Shutdown Runbook

### Health Endpoint Verification:
Check system health via HTTP GET:
```bash
curl -i http://<API_SERVICE_URL>/v1/health
```
* **Expected Response (Healthy):**
  ```json
  HTTP/1.1 200 OK
  {
    "object": "health_check",
    "data": {
      "status": "healthy",
      "version": "1.1.0",
      "database": "connected",
      "redis": "connected",
      "object_storage": "connected"
    }
  }
  ```
* **Degraded Response (HTTP 503):** If any backing service (Redis, DB, S3) is unreachable, `status` reports `degraded` and the failing dependency reports `disconnected`.

### Graceful Shutdown Sequence:
1. Ingress load balancer stops sending new requests to the instance.
2. `SIGTERM` signal sent to the Uvicorn process.
3. FastAPI lifespan executes `reader_lifecycle.begin_shutdown()`, rejecting new gated reads immediately (`ReaderGateClosing`).
4. `reader_lifecycle.wait_drained()` waits up to `API_SHUTDOWN_DRAIN_TIMEOUT_SECONDS` (default 40s) for active S3 chunk reads to complete.
5. `ReaderLockPool.dispose()` closes all physical connections cleanly.

---

## 10. Frontend Deployment Runbook

### Build-Time Configuration:
The Next.js frontend **must** be built with the production API URL as a build argument:
```bash
cd services/frontend
docker build -f docker/Dockerfile.frontend \
  --build-arg API_PROXY_TARGET="<API_SERVICE_URL>" \
  -t weather-frontend:latest .
```

> **Operational Warning:** Changing `API_PROXY_TARGET` via environment variables at container runtime has **no effect**. If the API hostname changes, the frontend container image must be rebuilt and redeployed.

---

## 11. Infrastructure Outage Recovery Runbooks

### 11.1 PostgreSQL Outage Recovery
1. **Symptom:** API returns 503 / 500; Ingestion wave fails; `pg_stat_activity` drops.
2. **Behavior During Outage:** All session advisory locks drop server-side when client connections are severed. Realtime and GC leadership are lost immediately.
3. **Recovery Steps:**
   - Restore PostgreSQL service.
   - API `pool_pre_ping` automatically reconnects on incoming traffic.
   - Realtime scheduler detects connection loss and reacquires leadership on a fresh session.
   - Verify health via `curl http://<API_SERVICE_URL>/v1/health`.

### 11.2 Redis Outage Recovery
1. **Symptom:** `/v1/health` returns HTTP 503 `status: degraded` with `redis: disconnected`.
2. **Behavior During Outage:** Point forecasts and vector tiles continue serving directly from S3/Zarr. No forecast data is lost.
3. **Recovery Steps:**
   - Restore Redis instance.
   - API automatically resumes caching on subsequent requests; `/v1/health` recovers to 200 `healthy`.

### 11.3 S3 / Object Storage Outage Recovery
1. **Symptom:** Ingestion waves fail; API point/tile queries fail with 404/500; `/v1/health` reports `object_storage: disconnected`.
2. **Recovery Steps:**
   - Restore S3 bucket / network uplink.
   - Ingestion automatically retries pending lead waves on next poll.
   - **Do NOT manually fabricate manifest files or S3 keys.** Allow the ingestion pipeline's coalesced finalizer to commit fresh manifests.

---

## 12. Ingestion & Storage Failure Recovery Runbooks

### 12.1 Failed Ingestion / Interrupted Lead Wave
* **State after Interruption:** Unfinished shards and staging markers remain in temporary prefixes. Target run in PostgreSQL remains in `partial` status. Existing committed leads remain fully serviceable.
* **Recovery Action:** None required manually. The next wave iteration automatically re-downloads missing leads, overwrites physical shard keys, stages markers, and finalizes the manifest.

### 12.2 Same-Cycle Re-Ingestion (In-Place Overwrite)
* **Semantics:** Ingestion of an existing cycle overwrites physical shard files in-place under `s3://<BUCKET>/<model>/<date>/<hour>/cycle.zarr/`.
* **Correctness Guard:** Ingestion writer acquires `EXCLUSIVE` store gate during manifest commit; API readers hold `SHARED` store gate across all chunk reads.
* **Execution:** Re-run `weather-ingest ingest` with desired leads/members; catalog reconciliation automatically updates product rows.

### 12.3 Broken / Corrupted Store Diagnostic Decision Tree
```text
API reports Store Unreadable (FileNotFoundError / ManifestReadError)
    │
    ▼
Is a previous valid READY run available in catalog?
    ├── YES ──► API automatically falls back to previous cycle (Serving degraded but safe)
    │
    └── NO  ──► Execute Diagnostic Checks:
                  1. Check if S3 prefix exists: aws s3 ls s3://<BUCKET>/<model>/<date>/<hour>/
                  2. Check if __commit__/v1/manifest.json exists and is valid JSON.
                  3. Check if cycle is fenced: SELECT * FROM forecast_cycle_lifecycle WHERE cycle_time = ...;
                  4. If store is unrecoverable: re-ingest run via `weather-ingest ingest`.
```

---

## 13. Read-Only Operational Database Diagnostic Queries

Execute these queries for operational inspection:

### 1. View Latest Ingested Model Runs & Status
```sql
SELECT 
    mr.id,
    m.model_id,
    mr.cycle_time,
    mr.status,
    mr.zarr_store_path,
    mr.created_at
FROM model_runs mr
JOIN model_versions mv ON mr.model_version_id = mv.id
JOIN models m ON mv.model_id = m.model_id
ORDER BY mr.cycle_time DESC
LIMIT 10;
```

### 2. Check Committed Product Counts per Run
```sql
SELECT 
    run_id,
    count(*) AS committed_lead_count,
    min(lead_time_hours) AS min_lead,
    max(lead_time_hours) AS max_lead
FROM forecast_products
GROUP BY run_id
ORDER BY run_id DESC
LIMIT 10;
```

### 3. Check Cycle Deletion Fences & Tombstones
```sql
SELECT 
    model_id,
    cycle_time,
    deletion_started_at,
    deleted_at,
    created_at,
    updated_at
FROM forecast_cycle_lifecycle
ORDER BY cycle_time DESC
LIMIT 10;
```

### 4. Inspect Active Advisory Locks & Connection Owners
```sql
SELECT 
    l.pid,
    a.usename,
    a.application_name,
    a.client_addr,
    l.mode,
    l.granted,
    ((l.classid::bigint << 32) | (l.objid::bigint & 4294967295)) AS advisory_key,
    to_hex((l.classid::bigint << 32) | (l.objid::bigint & 4294967295)) AS advisory_key_hex
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE l.locktype = 'advisory'
ORDER BY l.pid;
```

---

## 14. Incident Evidence Capture Checklist

When opening an incident or preparing an escalation report, capture:
1. **API & Ingestion Service Logs** (last 500 lines).
2. **Current Health Status:** `curl -s http://<API_SERVICE_URL>/v1/health`.
3. **Database Diagnostic Output:** Output of Queries 1, 3, and 4 from Section 13.
4. **Target Store S3 Listing:** `aws s3 ls s3://<BUCKET>/<model>/<date>/<hour>/cycle.zarr/__commit__/v1/`.
5. **Realtime Scheduler State:** Output of leadership query.

---

## 15. Rollback Semantics & Warnings

### Application Code Rollback:
* Deploying a previous container image (`weather-api:<PREV_TAG>`, `weather-frontend:<PREV_TAG>`) is safe and supported.

### Database Schema Rollback:
* Alembic downgrades (`alembic downgrade -1`) should **only** be executed if explicitly verified in staging. Do not run destructive schema downgrades if production forecast tables carry active data.

### Data Rollback Warning (Mandatory Operational Invariant):
> **CRITICAL WARNING:** **Logical serving generation is NOT an immutable physical snapshot.**  
> You cannot assume a previous `serving_generation` implies recoverable physical shard data. Same-cycle re-ingestion overwrites physical shard keys in place. Rolling back a data error requires re-ingesting the correct GRIB2 data from upstream NOAA sources.

---

## 16. Operational Impact of Known Technical Debts

| Technical Debt | Operational Impact | Operator Procedure |
| :--- | :--- | :--- |
| **Fresh Deployment Baseline Bootstrap** | Realtime scheduler does not bulk-ingest historical baseline on empty DB. | Run manual `weather-ingest ingest` for initial full cycle before starting scheduler. |
| **Cold Backlog Catch-Up Latency** | If scheduler is down for hours, it processes accumulated leads sequentially. | Allow scheduler to catch up; wave runner will process pending leads in bounded batches. |
| **GEFS Cold-Serving Range GET Latency** | Cold ensemble queries fetch 30 member chunks without concurrent batching. | Redis hot-cache mitigates repeat queries; initial query may take ~500ms–1s. |
| **Stale `forecast_products` Re-ingest Acceptance** | Stale products from shrunk runs self-heal on next write. | None required; automatic reconciliation handles catalog cleanup on write. |

---

## 17. Stage 8 Deployment-Specific Sizing Placeholders

The following settings remain **TBD** until physical server provisioning in Stage 8:

| Configuration Setting | Parameter Placeholder | Governing Sizing Criterion | Status |
| :--- | :--- | :--- | :---: |
| **Host Compute & RAM** | `<HOST_CPU_CORES>`, `<HOST_RAM_GB>` | Chosen server instance size | `FINALIZE IN STAGE 8` |
| **API Worker Count** | `<API_WORKERS>` | Measured worker RSS and CPU core allocation | `FINALIZE IN STAGE 8` |
| **PostgreSQL Max Connections**| `<POSTGRES_MAX_CONNECTIONS>` | $\ge \text{Calculated TOTAL\_DB\_CONNECTION\_BUDGET}$ | `FINALIZE IN STAGE 8` |
| **Ingestion Decode Workers** | `<MAX_DECODE_CONCURRENCY>` | $\le \text{Host Cores} - \text{Colocated Reserves}$ | `FINALIZE IN STAGE 8` |
| **Redis Max Memory** | `<REDIS_MAXMEMORY>` | Host RAM minus API and Ingestion budgets | `FINALIZE IN STAGE 8` |
| **S3 Storage Provider & FQDN**| `<OBJECT_STORAGE_ENDPOINT>` | Cloud provider S3 endpoint | `FINALIZE IN STAGE 8` |
| **Staging Disk Allocation** | `<STAGING_DISK_GB>` | Sized for peak concurrent wave downloads | `FINALIZE IN STAGE 8` |
| **Retention Window Policy** | `<RETENTION_POLICY>` | Number of historical cycles to retain (e.g. 4 cycles) | `FINALIZE IN STAGE 8` |

---

## 18. Native ARM64 Production Host Acceptance Runbook

While GitHub Actions CI verifies ARM64 image construction, dependency resolution, and basic CLI invocation under Buildx/QEMU, final acceptance for deploying to a native ARM64 host (e.g., AWS Graviton, Ampere Altra, Apple Silicon server) requires validating runtime behavior on bare metal without emulation.

### Step 1: Verify Native Architecture & Kernel
Verify the host runs native 64-bit ARM Linux:
```bash
uname -m
# Expected output: aarch64 (or arm64)
```

### Step 2: Validate Backing Infrastructure
Verify that backing services run natively without architecture emulation:
```bash
docker compose up -d
docker inspect weather_postgres --format '{{.Architecture}}'  # Expected: arm64
docker inspect weather_redis --format '{{.Architecture}}'     # Expected: arm64
docker inspect weather_minio --format '{{.Architecture}}'     # Expected: arm64
```
Ensure PostgreSQL 18 is initialized with the dedicated `postgres18_data` volume:
```bash
docker exec weather_postgres psql -U weather_user -d weather_db -c "SELECT version(); SELECT PostGIS_Full_Version();"
```

### Step 3: Run Native Application Container Smokes
Verify native execution of API and Ingestion images:
```bash
# Ingestion CLI help smoke
docker run --rm weather-ingestion:latest --help

# Ingestion native ecCodes / GRIB decoding import smoke
docker run --rm --entrypoint python weather-ingestion:latest -c "import cfgrib; print('Native ecCodes OK')"

# API serving tier imports & dependencies smoke
docker run --rm weather-api:latest python -c "import numpy, pandas, psycopg2, zarr, numcodecs, api.main; print('Native API OK')"
```

### Step 4: Validate Real GRIB2 Decode & Zarr Write
Execute a one-lead test ingestion against live NOAA data to verify native ecCodes GRIB2 parsing, SIMD floating-point operations, and Zarr sharded storage:
```bash
docker run --rm \
  --network host \
  -e DATABASE_URL="postgresql://weather_user:weather_password@localhost:5432/weather_db" \
  -e MINIO_ENDPOINT="localhost:9000" \
  -e MINIO_ACCESS_KEY="minio_admin" \
  -e MINIO_SECRET_KEY="minio_password" \
  -e MINIO_SECURE="false" \
  -e WEATHER_TEST_MINIO="1" \
  weather-ingestion:latest ingest \
    --model gfs \
    --cycle-date "$(date -u +%Y-%m-%d)" \
    --cycle-hour 0 \
    --lead-time-hours 0 \
    --dry-run
```

### Step 5: Validate API Serving Endpoints
Start the API container on the native host and execute an end-to-end healthcheck:
```bash
curl -sf http://localhost:8000/v1/health | jq .
curl -sf http://localhost:8000/v1/models | jq .
```
Verify that JSON serialization, Redis caching, and coordinate projections execute without architecture-dependent regressions.

---

## 19. Runtime Health, Resource & Alert Response Runbooks

### 19.1 Ingestion Pipeline Stuck (`#ingestion-stuck`)
* **Symptom:** Alert `ingestion_pipeline_stuck` triggers (`CRITICAL`). A model run remains in `processing` or `partial` with zero forward progress in download, decode, write, or finalize for $> 10\text{ minutes}$.
* **Diagnostic Procedure:**
  1. Run `weather-ingest diagnostics` to identify which stage is stalled (`download_active`, `decode_active`, `write_active`, or `finalize_state`).
  2. Inspect active threads and system logs:
     - If download stalled: check NOAA upstream reachability (`curl -I https://nomads.ncep.noaa.gov` or AWS S3 open data).
     - If decode stalled: check `DecodePool` processes. A corrupted upstream GRIB2 message can crash ecCodes native parser.
     - If write stalled: check `pg_locks` to see if a writer is waiting for a database advisory lock or store gate.
     - If finalize stalled: check MinIO latency and marker read responses.
* **Remediation:**
  - If a worker process hung, restart the ingestion worker service. Wave tasks are transactional and idempotent; the restarted runner will resume with clean state.
  - If NOAA upstream is rate-limiting, verify `ENABLE_NOMADS_FALLBACK` and AWS Open Data configuration.

### 19.2 Ingestion Lag vs. Upstream Availability (`#ingestion-lag`)
* **Symptom:** Alert `ingestion_lag_warning` (1 cycle behind) or `ingestion_lag_critical` (2+ cycles behind) triggers.
* **Important Operational Distinction:**
  - **Upstream Unavailable:** If NOAA has not yet published the cycle, this is an external delay, not an ingestion failure. The lag alert only triggers when upstream data is confirmed available.
* **Diagnostic Procedure:**
  1. Run `weather-ingest status` to check `latest ready cycle`, `expected latest`, and `ingestion lag`.
  2. Query NOAA discovery endpoint:
     ```bash
     weather-ingest realtime --once --dry-run
     ```
* **Remediation:**
  - If the scheduler fell behind due to downtime, it will automatically process backlogged waves sequentially.
  - For manual catch-up, dispatch the missing cycle explicitly:
    ```bash
    weather-ingest ingest --model <model> --cycle-date <YYYY-MM-DD> --cycle-hour <HH> --lead-time-hours 0 3 6 ... 240
    ```

### 19.3 Process Memory Growth & Sustained Leak (`#memory-leak`)
* **Symptom:** Alert `memory_leak_warning` triggers (`WARNING`). Process post-cycle baseline RSS has grown monotonically over 5+ consecutive cycles by $\ge 15\%$ and $\ge 150\text{ MB}$.
* **Diagnostic Procedure:**
  1. Run `weather-ingest diagnostics` to inspect Python heap and garbage collection stats (`gc.get_count()`).
  2. Note that temporary RSS peaks during GEFS 30-member writes are expected; only sustained post-cycle baseline growth constitutes a leak.
  3. Inspect open file descriptors / handles (`weather_process_open_file_descriptors`).
* **Remediation:**
  - If `DecodePool` native C memory is leaking from repetitive ecCodes handles, restart the ingestion worker daemon. Worker process recycling ensures complete C-heap release.

### 19.4 PostgreSQL Connection Saturation, Locks & Deadlocks (`#database-connections`, `#database-lock-waits`, `#database-deadlocks`, `#database-unreachable`)
* **Symptom:** Alert `postgres_connections_warning` ($\ge 80\%$), `postgres_connections_critical` ($\ge 95\%$), `postgres_lock_waits`, or `postgres_deadlocks_detected` triggers.
* **Deadlock Semantics:** Note that `pg_stat_database.deadlocks` is cumulative; the alert only fires when **new deadlocks occur within the observation interval** (`deadlocks_delta > 0`). Historical deadlocks that resolved never cause ongoing alerts.
* **Diagnostic Procedure:**
  1. Run `weather-ingest diagnostics` to view active connection counts, locked sessions, and long-running queries.
  2. Inspect connection allocation across services:
     ```sql
     SELECT application_name, count(*) FROM pg_stat_activity GROUP BY application_name;
     ```
  3. Check for ungranted locks and blocked sessions:
     ```sql
     SELECT * FROM pg_locks WHERE NOT granted;
     ```
* **Remediation:**
  - Terminate hung long-running sessions via `SELECT pg_terminate_backend(<PID>);`.
  - If connection saturation occurs under high traffic, verify that `API_READER_LOCK_POOL_SIZE` and `DB_POOL_SIZE` obey the Stage 7F connection formula.
  - If deadlocks repeat across concurrent ingestion waves, verify that writers sort catalog row insertions in canonical primary key order.

### 19.5 Disk Storage & MinIO Volume Full (`#disk-space-warning`, `#disk-space-critical`, `#minio-unreachable`)
* **Symptom:** Alert `disk_space_warning` ($\ge 80\%$) or `disk_space_critical` ($\ge 90\%$) triggers on temporary staging directory or object storage volume.
* **Diagnostic Procedure:**
  1. Check disk utilization: `df -h` or `weather-ingest status`.
  2. Inspect staging directory `downloads/` for abandoned partial downloads from crashed runs.
* **Remediation:**
  - Purge orphaned temporary downloads: `rm -rf downloads/tmp_*`.
  - Check whether the GC finalizer is running. If retired cycles are not being purged, verify `weather-ingest gc` daemon status.

### 19.6 Stuck Physical Deletion Claims (`#stuck-deletion-claim`)
* **Symptom:** Alert `finalizer_claim_stuck_warning` ($> 1\text{h}$) or `finalizer_claim_stuck_critical` ($> 4\text{h}$) triggers. A cycle has `deletion_started_at` set but `deleted_at` is null.
* **Diagnostic Procedure:**
  1. Query the stuck cycle:
     ```sql
     SELECT model_id, cycle_time, deletion_started_at
     FROM forecast_cycle_lifecycle
     WHERE deletion_started_at IS NOT NULL AND deleted_at IS NULL;
     ```
  2. Check if a GC worker process crashed while deleting S3 keys.
  3. Verify object storage responsiveness: slow S3 prefix deletion can cause lease timeouts.
* **Remediation:**
  - Run a single-pass GC reconciliation to resume recovery candidates:
    ```bash
    weather-ingest gc --once
    ```
  - The finalizer will detect existing `deletion_started_at` claims, resume store deletion without re-evaluating horizon eligibility, and commit `deleted_at`.

### 19.7 14-Day Metadata Sweeper Backlog Overdue (`#sweeper-backlog`)
* **Symptom:** Alert `metadata_sweeper_backlog_overdue` triggers (`WARNING`). Tombstones older than 14 days retain detailed `model_runs` metadata.
* **Diagnostic Procedure:**
  1. Check unpurged tombstones count:
     ```sql
     SELECT count(*) FROM forecast_cycle_lifecycle l
     JOIN model_runs r ON r.cycle_time = l.cycle_time
     WHERE l.deleted_at <= NOW() - INTERVAL '14 days';
     ```
* **Remediation:**
  - Trigger a manual metadata sweeper pass:
    ```bash
    weather-ingest gc --once --sweep-metadata --batch-size 100
    ```
  - The sweeper will purge child records (`model_runs`, `forecast_products`, `ensemble_member_products`) while preserving the `forecast_cycle_lifecycle` tombstone row.

### 19.8 Granular Reclamation Failures & Lease Expiry (`#reclamation-failures`, `#reclamation-stuck`)
* **Symptom:** Alert `reclamation_failed_shards` or `reclamation_deleting_stuck` triggers.
* **Diagnostic Procedure:**
  1. Inspect non-terminal reclamation records (including stale-lease
     `deleting` claims and legacy stuck `queued` rows):
     ```sql
     SELECT id, model_id, cycle_time, variable_code, lead_time_hours, status, last_error
     FROM reclamation_queue
     WHERE status IN ('failed', 'deleting', 'queued');
     ```
* **Remediation:**
  - If failures were caused by transient S3 reachability — or rows are stuck
    in `queued`/`deleting` over already-deleted physical objects (blocking the
    cycle tombstone) — recover them:
    ```bash
    weather-ingest reclamation requeue
    ```
    Rows holding an active worker lease are skipped automatically; rows whose
    physical object is confirmed absent are promoted to terminal `deleted`,
    rows still present are reset to `queued` for worker retry.
  - Re-run the reclamation worker pass:
    ```bash
    weather-ingest reclamation work --delete --batch-size 100
    ```

### 19.9 Data Lifecycle Invariant & Anti-Resurrection Violations (`#anti-resurrection-violation`)
* **Symptom:** Alert `invariant_anti_resurrection_violation` or `invariant_invalid_lifecycle_transition` triggers (`CRITICAL`).
* **Diagnostic Procedure:**
  1. Run `weather-ingest audit` immediately to display the exact offending rows:
     ```bash
     weather-ingest audit
     ```
  2. Verify if a writer attempted to ingest or recreate a cycle whose tombstone `deleted_at` was already committed.
* **Remediation:**
  - Under no circumstances should a permanent tombstone be bypassed. Check scheduler and manual ingestion logs to identify the unauthorized source.
  - Delete any resurrected uncommitted run rows in `model_runs` that violate the permanent tombstone.

### 19.10 Readiness Not Promoted for a Due Cycle (`#model-ready-not-promoted`)
* **Symptom:** Alert `model_ready_not_promoted` triggers (`WARNING`). The newest cycle that should already be complete — its `model_runs.created_at` (the f000 ingest anchor) plus the fill budget `INGESTION_FILL_IN_GRACE_SECONDS` (default 2.5h) has passed — is still not `ready`.
* **Important Operational Distinction:**
  - This is **not** an ingestion-shortfall signal, and it is deliberately **not** gated on `weather_model_ready_status`. That gauge reflects the *newest* run, which is 0 throughout every normal fill window (a fresh cycle is expected to be `partial` while it ingests), so gating on it would fire on a healthy platform once per cycle.
  - The verdict is taken from the *due* cycle instead. A cycle can be fully ingested and serving traffic (`weather_ingestion_lag_cycles = 0`) while its status was never promoted, which no other view surfaces.
* **Diagnostic Procedure:**
  1. Run `weather-ingest status --json` and inspect `gfs`/`gefs`: `lag_target`, `lag_target_ready`, `cycles_complete_not_ready`, `latest_servable` vs `latest_ready`.
     - If `latest_servable` is at or ahead of `lag_target` while `latest_ready` lags behind it, the data is there and only the status is stale.
  2. Confirm the affected cycle directly:
     ```sql
     SELECT r.cycle_time, r.status, r.created_at
     FROM model_runs r
     JOIN model_versions v ON v.id = r.model_version_id
     WHERE v.model_id = '<model>'
     ORDER BY r.cycle_time DESC LIMIT 4;
     ```
  3. Check whether a wave is mid-flight for that cycle (the pre-update fence sets `partial` on purpose): look for `pre_update_start` without a matching `finalize_complete` in the realtime logs.
* **Remediation:**
  - No manual action is normally required: backlog recovery now dispatches a finalize-only status repair for a cycle that has no ingest work left but is not durably complete, so a stale `partial` is re-derived on the next poll.
  - If it persists, confirm the backlog path is enabled (`REALTIME_BACKLOG_ENABLED`) and look for `realtime repairing durable status for <cycle>` in the logs.
  - When the store↔catalog gate legitimately rejects promotion, the cause is a real catalog/store divergence: check for intentional reclamation rows before assuming corruption.
    ```sql
    SELECT status, count(*) FROM reclamation_queue
    WHERE run_id = '<run_id>' GROUP BY status;
    ```

### 19.11 Cycles Complete in the Catalog But Not Promoted (`#cycles-complete-not-ready`)
* **Symptom:** Alert `cycles_complete_not_ready` triggers (`WARNING`), driven by `weather_ingestion_cycles_complete_not_ready{model} > 0`. One or more cycles hold every expected lead (and, for GEFS, the full expected member matrix) in the catalog while their run status is not `ready`, and the oldest has been past its fill deadline for at least one fill budget.
* **Diagnostic Procedure:**
  1. Identify the cycles:
     ```sql
     SELECT r.cycle_time, r.status, r.created_at,
            (SELECT count(DISTINCT lead_time_hours) FROM forecast_products p WHERE p.run_id = r.id) AS leads
     FROM model_runs r
     JOIN model_versions v ON v.id = r.model_version_id
     WHERE v.model_id = '<model>' AND r.status <> 'ready'
     ORDER BY r.cycle_time DESC;
     ```
  2. Compare `leads` against the canonical horizon (81 leads, 0–240h step 3). A cycle at 81 leads with a non-`ready` status is this defect.
  3. Distinguish it from the harmless transient: immediately after a wave's finalization the cycle is briefly complete-and-then-promoted, and a wave's pre-update fence downgrades a same-cycle re-ingest to `partial` on purpose. Both are shorter than the fill budget and therefore do not alert.
* **Remediation:**
  - Expect this to clear itself within one poll interval: backlog recovery re-derives the status through the finalizer (no re-download). Verify with `weather-ingest status --json` (`cycles_complete_not_ready` returning to 0) and re-check `model_runs`.
  - If a cycle stays complete-but-unpromoted after several polls, the status derivation is failing rather than being skipped: grep the realtime logs for `catalog_reconcile` / `finalize` errors for that cycle, and check whether the store's committed evidence disagrees with the catalog.
  - Do **not** hand-edit `model_runs.status`; the status is derived, and a manual value would be overwritten by the next derivation while hiding the underlying divergence.


