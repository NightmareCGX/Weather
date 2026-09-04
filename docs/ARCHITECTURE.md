# System Architecture Specification

## 1. Executive Summary

The Global Probabilistic Weather Platform is a high-throughput, cloud-native meteorological data processing and serving engine. It ingests numerical weather prediction (NWP) model outputs from the National Oceanic and Atmospheric Administration (NOAA), normalizes physical meteorological fields into standardized units, packs tensor grids into byte-range queryable sharded Zarr containers on object storage, tracks progressive forecast lifecycle states in a relational PostgreSQL catalog, and serves low-latency point forecasts, ensemble statistics, and interactive map layers via a FastAPI backend and Next.js frontend.

---

## 2. System Overview

```text
                               ┌──────────────────────────────────────────────┐
                               │           Upstream Data Providers            │
                               │  NOAA NOMADS (HTTP) / AWS Open Data (S3)     │
                               │  GFS (0.25° Det) & GEFS (0.5° 30-Mem Ens)    │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │     Ingestion Engine (services/ingestion)    │
                               │  • Selective .idx Byte-Range GRIB2 Download  │
                               │  • Multiprocess DecodePool (cfgrib/ecCodes)  │
                               │  • Unit Normalization & Derivations          │
                               │  • Sharded v1 Zarr Encoding                  │
                               │  • Realtime Lead-Wave Scheduler & Discovery  │
                               │  • Retention Garbage Collector (GC Engine)   │
                               └──────────────┬────────────────┬──────────────┘
                                              │                │
                        Shard Files & Manifests│                │ Relational Catalog,
                                              │                │ Lifecycle & Advisory Locks
                                              ▼                ▼
                     ┌────────────────────────────────┐ ┌────────────────────────────────┐
                     │    Object Storage (S3/MinIO)   │ │    PostgreSQL 16 + PostGIS     │
                     │  s3://weather-data/{model}/    │ │  • Model runs, variables, grids│
                     │  {date}/{hour}/cycle.zarr/     │ │  • Progressive product catalog │
                     │  • Canonical shard containers  │ │  • Cycle lifecycle & fences    │
                     │  • __markers__/v1/ (staging)   │ │  • Stations, cities, resorts   │
                     │  • __commit__/v1/manifest.json │ │  • 64-bit advisory lock gates  │
                     └───────────────┬────────────────┘ └───────────────┬────────────────┘
                                     │                                  │
                                     └────────────────┬─────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │        Serving Tier (services/api)           │
                               │  • FastAPI REST Framework                    │
                               │  • Distributed PostgreSQL SHARED Reader Gate │
                               │  • Sharded v1 Byte-Range Chunk Reader        │
                               │  • Spatial Bilinear Grid Interpolation       │
                               │  • Probabilities, Percentiles & PDFs         │
                               │  • Dynamic Map Tile & Vector Field Server    │
                               │  • Redis 7 Response & Tile Hot-Cache         │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │       Frontend Tier (services/frontend)      │
                               │  • Next.js 14 (App Router, Standalone Build) │
                               │  • MapLibre GL Weather Map Visualizations    │
                               │  • Recharts Meteograms & Ensemble Charts     │
                               │  • Location Search & Forecast Dashboard      │
                               └──────────────────────────────────────────────┘
```

### Component Roles & Responsibilities

| Component | Repository Path | Core Technologies | Primary Responsibilities |
| :--- | :--- | :--- | :--- |
| **Ingestion Engine** | `services/ingestion` | Python 3.12, asyncio, `cfgrib`, `xarray`, `numcodecs`, `boto3`, `s3fs`, SQLAlchemy | Downloads GRIB2 messages, parses raw binary buffers, normalizes units, writes `sharded_v1` Zarr stores, schedules lead waves, and executes retention GC. |
| **Serving Tier** | `services/api` | Python 3.12, FastAPI, Uvicorn, SQLAlchemy, GeoAlchemy2, `xarray`, `s3fs`, Redis | Exposes REST endpoints, validates requested models/coordinates, acquires `SHARED` advisory locks, performs granular Range GETs against Zarr stores, and computes domain outputs. |
| **Frontend UI** | `services/frontend` | TypeScript, Next.js 14, React 18, MapLibre GL, Recharts, Tailwind CSS | Browser-based user interface for interactive map exploration, point meteograms, ensemble spaghetti/PDF charts, and location search. |
| **Domain Logic** | `packages/domain` | Python 3.12, NumPy, `xarray` | Pure, dependency-free mathematical domain models: grid coordinates, bilinear interpolation, ensemble statistics, precipitation phase classification, verification metrics, and advisory lock key derivation. |
| **Relational Database** | Infrastructure | PostgreSQL 16 + PostGIS 3.4 | Stores catalog hierarchy, progressive product availability, lifecycle fences, spatial point reference data, and coordinates distributed advisory locks. |
| **Object Store** | Infrastructure | AWS S3 / MinIO | Primary repository for all multidimensional meteorological raster data formatted in `sharded_v1` Zarr layout. |
| **Response Cache** | Infrastructure | Redis 7 | Caches interpolated point forecast payloads and vector field grids by generation key to minimize repeated object storage reads. |

---

## 3. Ingestion Architecture

The ingestion tier (`services/ingestion`) processes numerical weather prediction data from NOAA upstream sources into platform storage.

### 3.1 Data Pipeline Flow

```text
NOAA NOMADS / AWS S3
       │
       ▼
[ NOAAConnector ] ──► (HTTP .idx Range GET or S3 GET)
       │
       ▼
[ Staged GRIB2 Files ] (Local disk staging directory)
       │
       ▼
[ DecodePool ] (Multiprocessing Pool executing cfgrib / libeccodes)
       │
       ▼
[ Normalization ] ──► • Convert Kelvin to Celsius (°C)
                      • Convert kg/(m²·s) to mm/h
                      • De-accumulate total precipitation increments
                      • Derive wind speed (km/h) & direction (0–360°)
       │
       ▼
[ Coordinator & Lock Acquisition ] ──► Acquire SHARED Store Gate & EXCLUSIVE Region Lock
       │
       ▼
[ ShardedV1Writer ] ──► Pack 120 spatial chunks into binary shard containers
       │
       ▼
[ Staging Marker PUT ] ──► Upload __markers__/v1/{region_id}.json to S3
       │
       ▼
[ Settled-Lead Publication ] ──► Upsert forecast_products & ensemble_member_products (PostgreSQL)
       │
       ▼
[ Coalesced Finalization ] ──► • Acquire EXCLUSIVE Admission & Store Gate
                               • Commit __commit__/v1/manifest.json (new serving_generation)
                               • Update model_runs.status ('partial' or 'ready')
```

### 3.2 Ingestion Operational Modes

The `weather-ingest` CLI (`services/ingestion/src/ingestion/cli.py`) supports three execution modes:

1. **Manual Batch Ingestion (`weather-ingest ingest`):**
   - Explicitly ingests specific models, dates, cycle hours, and forecast leads.
   - Anti-Cartesian design: Multiple models/dates/hours are aligned 1:1 into run specifications rather than generating accidental cross-products.
2. **Realtime Lead-Wave Scheduler (`weather-ingest realtime`):**
   - Polls NOAA NOMADS / AWS S3 discovery endpoints for newly published forecast leads.
   - Uses a session-level PostgreSQL advisory lock (`scheduler_leader_key`) to ensure exactly one active scheduler instance per deployment.
   - Dispatches bounded lead waves (up to `REALTIME_WAVE_MAX_LEADS`, default 8) to the wave runner.
   - Progressively publishes settled leads to the catalog as they complete.
3. **Retention Garbage Collector (`weather-ingest gc`):**
   - Reconciles retired forecast cycles according to retention policies.
   - Sets durable deletion fences (`deletion_started_at`) on `forecast_cycle_lifecycle` records.
   - Deletes retired S3 stores sequentially under `EXCLUSIVE` store gates.

---

## 4. Storage Architecture (`sharded_v1`)

Forecast grid data is stored using the **Weather Platform Sharded v1 (`sharded_v1`)** binary layout.

### 4.1 Canonical Store Path Convention
Each model run cycle is stored under a deterministic object storage prefix derived from the forecast identity:
```text
s3://weather-data/{model}/{YYYY-MM-DD}/{HH}/cycle.zarr/
```
* Example GFS: `s3://weather-data/gfs/2026-09-03/00/cycle.zarr/`
* Example GEFS: `s3://weather-data/gefs/2026-09-03/00/cycle.zarr/`

### 4.2 Binary Shard Container Specification
Instead of storing tens of thousands of individual chunk files in S3, each 2D field (variable × lead × member) is stored in a single binary `.shard` container file.

* **Shard Naming Convention:**
  * **Deterministic (GFS):** `{variable}/shard.det_L{lead_time_hours:04d}.shard`
    * Example: `temperature_2m/shard.det_L0006.shard`
  * **Ensemble (GEFS):** `{variable}/shard.mem{member_index:03d}_L{lead_time_hours:04d}.shard`
    * Example: `temperature_2m/shard.mem003_L0006.shard`
* **Grid Chunking:**
  * NOAA GFS (721 × 1440 grid) is divided into 100 × 100 spatial chunks.
  * 8 latitude rows × 15 longitude columns = **120 spatial chunks** per 2D field.
* **Container Binary Layout:**
  ```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Chunk 0 (Zstd compressed float32[100, 100])                            │
  ├────────────────────────────────────────────────────────────────────────┤
  │ Chunk 1 (Zstd compressed float32[100, 100])                            │
  ├────────────────────────────────────────────────────────────────────────┤
  │ ...                                                                    │
  ├────────────────────────────────────────────────────────────────────────┤
  │ Chunk 119 (Zstd compressed float32[100, 100])                          │
  ├────────────────────────────────────────────────────────────────────────┤
  │ Index Table (120 entries × 16 bytes = 1920 bytes)                      │
  │ Entry format: <uint64 offset, uint64 length> (Little Endian)           │
  ├────────────────────────────────────────────────────────────────────────┤
  │ Trailer (12 bytes):                                                    │
  │ uint32 num_chunks (120) | uint32 index_bytes (1920) | uint32 0x53484152 │
  └────────────────────────────────────────────────────────────────────────┘
  ```

### 4.3 Manifest & In-Place Overwrite Invariant
* **Staging Markers:** As each region writes, a JSON marker is placed at `__markers__/v1/{region_id}.json`.
* **Committed Manifest:** At finalization, `__commit__/v1/manifest.json` is written containing a unique `serving_generation` (UUID4 string), committed leads, and committed members.
* **CRITICAL INVARIANT:**
  > **Logical generation is NOT an immutable physical generation.**  
  > Same-cycle re-ingestion overwrites the **same physical shard keys** in-place in object storage. Serving readers must hold the PostgreSQL `SHARED` advisory store gate across all chunk Range GETs to prevent reading partially overwritten containers.

---

## 5. Serving Architecture

The serving layer (`services/api`) handles incoming HTTP requests from users and the frontend.

```text
HTTP Request (e.g. GET /v1/points?lat=40.0&lon=-105.0)
       │
       ▼
[ FastAPI Router ] (api/routers/points.py)
       │
       ▼
[ Location Resolution ] (PostGIS spatial lookup or coordinate validation)
       │
       ▼
[ Redis Cache Lookup ] ──► (Cache Hit: return cached envelope)
       │ (Cache Miss)
       ▼
[ Candidate Run Resolution ] ──► Select newest READY or PARTIAL ModelRun
       │
       ▼
[ gated_read_dataset_with_selector ] (api/core/reader_gate.py)
       ├── 1. Acquire SHARED Admission Turnstile
       ├── 2. Acquire SHARED Store Gate (PostgreSQL advisory lock)
       ├── 3. Release SHARED Admission Turnstile
       ├── 4. Fresh Core Revalidation: verify model_runs row is valid in PostgreSQL
       ├── 5. Manifest Reader: probe __commit__/v1/manifest.json for serving_generation
       ├── 6. StoreHandleCache: retrieve or open lazy xarray.Dataset
       ├── 7. Bounded Selection: locate 2×2 neighborhood chunks (100×100) on grid
       ├── 8. ShardedV1Reader: perform S3 Range GET for chunk byte slices
       └── 9. Materialize numpy array & interpolate under the gate
       │
       ▼
[ Release SHARED Store Gate ]
       │
       ▼
[ Unit Conversion & Response Formatting ]
       │
       ▼
[ Populate Redis Cache & Return HTTP Response ]
```

### 5.1 Endpoint Capabilities
1. **Point Forecasts (`/v1/points`):** Returns hourly time-series interpolated bilinearly to requested latitude/longitude coordinates.
2. **Ensemble Statistics & PDFs (`/v1/ensembles/statistics`, `/v1/ensembles/pdf`):** Calculates mean, median, standard deviation, spread, interquartile range, P10–P90 percentiles, and empirical probability density functions across 30 GEFS members.
3. **Map Tiles (`/v1/maps/layers/{layer_id}/tiles/{z}/{x}/{y}.png`):** Generates 256×256 dynamic PNG tiles with meteorological color palettes.
4. **Vector Wind Fields (`/v1/maps/layers/{layer_id}/vector-field`):** Serves gridded $u/v$ wind components for GPU-accelerated client particle animations.
5. **System Health (`/v1/health`):** Live connectivity probes against PostgreSQL, Redis, and MinIO/S3. Returns 200 `healthy` or 503 `degraded`.

---

## 6. Concurrency and Locking Protocol

Distributed concurrency control is coordinated using PostgreSQL session-level 64-bit advisory locks (`domain.locks`, `ingestion.core.locks`, `api.core.reader_gate`).

```text
64-Bit Advisory Lock Key Space:
┌───────────┬─────────────────────────────────────────────────────────────┐
│ 4-bit NS  │ 60-bit Hash Payload (BLAKE2b digest of canonical identity)  │
└───────────┴─────────────────────────────────────────────────────────────┘
```

### 6.1 Advisory Lock Namespaces

| Namespace Nibble | Name | Function | Usage |
| :--- | :--- | :--- | :--- |
| `0x0000000000000000` | **Store Gate** | `store_gate_key(store_path)` | `SHARED` for active readers & region writers; `EXCLUSIVE` for store init, manifest finalization, and GC deletion. |
| `0x1000000000000000` | **Region Conflict** | `region_key(store_path, region_id)` | `EXCLUSIVE` held by writer on a specific `(lead, member)` slice to prevent concurrent writes to the same region. |
| `0x2000000000000000` | **Admission Turnstile**| `admission_key(store_path)` | `EXCLUSIVE` held briefly by finalizer to queue new readers and prevent writer starvation. |
| `0x3000000000000000` | **Scheduler Leader** | `scheduler_leader_key(identity)`| Session lock held by active realtime scheduler process for leader election. |
| `0x4000000000000000` | **GC Leader** | `gc_leader_key(identity)` | Session lock held by active GC retention reconciler process. |

### 6.2 Reader/Writer Safety Rules
1. **Blocking Acquisition with Bounded Timeout:**
   - Locks are acquired with `SET LOCAL lock_timeout = :ms` followed by blocking acquisition statements (`pg_advisory_lock` / `pg_advisory_lock_shared`).
   - Waiting writers enter the PostgreSQL FIFO queue, preventing reader starvation.
2. **Session Persistence & Safe Invalidation:**
   - Advisory locks survive transaction `COMMIT`.
   - Ingestion workers and API readers release locks explicitly in `finally` blocks.
   - If an unlock query fails or a connection is interrupted, the physical connection is **invalidated** immediately, causing PostgreSQL to automatically drop all held session locks.
3. **Full Reader Materialization Scope:**
   - The reader `SHARED` store gate is held continuously throughout all S3/Zarr Range GETs and chunk decompression until the bounded slice is fully materialized into memory.

---

## 7. Data and Catalog Architecture

The PostgreSQL database tracks platform metadata and progressive publication state across 12 primary tables.

```text
[forecast_centers] 1 ──< [models] 1 ──< [model_versions] 1 ──< [model_runs]
                                                                  │
                                           ┌──────────────────────┴──────────────────────┐
                                           │ (1:N)                                       │ (1:N)
                                           ▼                                             ▼
                                  [ensemble_members]                            [forecast_products]
                                           │
                                           ▼ (1:N)
                             [ensemble_member_products]

[forecast_cycle_lifecycle] (cycle_time PK, retired_at, deletion_started_at, deleted_at)
[forecast_variables]       (variable_code PK, name, unit)
[forecast_grids]           (grid_code PK, name, resolution_km)
[stations], [cities], [ski_resorts] (PostGIS spatial reference tables)
```

### Table Responsibilities:
* `model_runs`: Primary record of a model cycle (`status`, `zarr_store_path`, `cycle_time`). Status transitions: `processing` → `partial` → `ready` / `failed` → `retired`.
* `forecast_products`: Records committed lead times per variable and grid.
* `ensemble_member_products`: Records committed `(member_index, lead_time_hours)` pairs for GEFS.
* `forecast_cycle_lifecycle`: Tracks durable cycle supersession, retirement timestamps (`retired_at`), and deletion fences (`deletion_started_at`, `deleted_at`).
* `cities`, `stations`, `ski_resorts`: Geospatial tables with PostGIS `GEOMETRY(Point, 4326)` for autocomplete and point resolution.

---

## 8. Caching Strategy

The platform employs a multi-tiered caching strategy to maximize serving throughput:

1. **Redis Response Cache (`services/api/src/api/services/cache.py`):**
   - Caches completed point forecast JSON envelopes and vector field grids.
   - Cache keys incorporate the model, coordinates, variable list, and **`serving_generation`**.
   - When a new generation is committed, subsequent requests compute a new cache key, rendering old cache entries unreachable without manual cache invalidation.
2. **In-Memory Store Handle Cache (`api.core.store_cache.StoreHandleCache`):**
   - Reuses lazily-opened `xarray.Dataset` handles in the API process.
   - Keyed by `(store_path, serving_generation)`.
   - Skips consolidated metadata (`.zmetadata`) re-reading for warm cycles while ensuring zero cross-generation leakage.
3. **In-Memory Sharded Chunk & Index LRU Cache (`api.core.zarr.ShardedV1Reader`):**
   - Caches parsed shard container index tables (up to 4096 entries) and decompressed 100×100 float32 chunks (up to 2048 chunks) per process.

---

## 9. Failure and Recovery Overview

| Failure Scenario | Recovery Mechanism | System Impact |
| :--- | :--- | :--- |
| **API Process Restart** | Stateless restart; FastAPI lifespan re-establishes reader lock pools. | In-flight HTTP requests receive 502/reset; subsequent requests succeed immediately. |
| **Ingestion Worker Crash** | Idempotent region writes; uncommitted staging markers are superseded by the next wave. Target run remains in `partial` state. | Zero store corruption; next wave re-downloads and commits. |
| **Realtime Scheduler Crash** | Session advisory lock `scheduler_leader_key` drops on socket close. | Standby instance or restarted process acquires leadership and resumes discovery. |
| **PostgreSQL Outage** | SQLAlchemy `pool_pre_ping` detects disconnection and reconnects when DB recovers. Advisory locks drop on disconnection. | Serving and ingestion temporarily error; full recovery once DB is restored. |
| **Redis Outage** | Cache layer catches Redis connection errors and computes fresh from Zarr/DB. `/v1/health` reports 503 degraded. | API continues serving with slightly higher latency. |
| **Object Store Outage** | Ingestion waves fail and retry next poll. API Zarr reads return 500/404. `/v1/health` reports 503 degraded. | System resumes serving immediately when object store recovers. |
| **GC Deletion Failure** | `deletion_started_at` fence persists in PostgreSQL, blocking ingestion writers from resurrecting the cycle. | Next GC pass detects incomplete deletion and resumes removal. |
