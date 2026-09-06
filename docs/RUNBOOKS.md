# Weather Platform — Operational Runbooks & Deployment Framework

This document is the authoritative operational runbook framework for deploying, operating, diagnosing, and recovering the Global Probabilistic Weather Platform.

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
       │         PostgreSQL 16        │ │           Redis 7            │ │     Object Storage (S3)      │
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
DATABASE_URL="<PRODUCTION_DATABASE_URL>" poetry run alembic upgrade head
```
*Verification:* Confirm Alembic head revision:
```bash
DATABASE_URL="<PRODUCTION_DATABASE_URL>" poetry run alembic current
```

### Step 2: Seed Spatial Reference Data (If Configured)
If reference cities or stations are deployed from SQL seed dumps:
```bash
psql "<PRODUCTION_DATABASE_URL>" -f seeds/reference_locations.sql
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
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers <API_WORKERS>
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
poetry run weather-ingest ingest \
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
poetry run weather-ingest realtime
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
poetry run weather-ingest gc --interval-seconds 1800
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
DATABASE_URL="<PRODUCTION_DATABASE_URL>" poetry run alembic current

# 2. Apply migrations to head:
DATABASE_URL="<PRODUCTION_DATABASE_URL>" poetry run alembic upgrade head

# 3. Verify schema state:
DATABASE_URL="<PRODUCTION_DATABASE_URL>" poetry run alembic current
```

### Migration Failure Handling:
* If `alembic upgrade head` fails with an SQL error, Alembic will roll back the active transaction.
* Inspect PostgreSQL logs for lock timeouts or constraint violations.
* **Do NOT execute blind downgrades** on production databases without verifying table data dependencies.

---

## 6. Service Startup & Logical Dependency Order

```text
1. Backing Infrastructure: PostgreSQL 16 (PostGIS) ──► Redis 7 ──► S3 Object Storage
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
poetry run weather-ingest realtime
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
   poetry run weather-ingest gc --interval-seconds 1800 --bucket <OBJECT_STORAGE_BUCKET>
   ```
2. **Single-Pass Mode (Cron / Manual Execution):**
   ```bash
   poetry run weather-ingest gc --once --bucket <OBJECT_STORAGE_BUCKET>
   ```
3. **Dry-Run Inspection Mode:**
   ```bash
   poetry run weather-ingest gc --once --dry-run
   ```

### Operational Invariants:
* **Lifecycle V2 Cadence Retention:** For each model M with cadence C_M (e.g. 6h for GFS/GEFS), let T be the latest run with status == 'ready'. Cycles >= T - C_M are retained (T and T - C_M); cycles < T - C_M are deletion eligible. Models advance retention and GC independently.
* **Deletion Fencing:** GC sets `deletion_started_at = NOW()` on `forecast_cycle_lifecycle` for `(model_id, cycle_time)` **before** deleting physical S3 keys.
* **Crash Safety:** If GC crashes mid-deletion, the durable fence prevents new ingestion writers from resurrecting the cycle; the next GC run detects the incomplete deletion and purges remaining objects.

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

### 3. Check Cycle Retirement & Deletion Fences
```sql
SELECT 
    cycle_time,
    retired_at,
    retired_by_cycle_time,
    deletion_started_at,
    deleted_at
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
