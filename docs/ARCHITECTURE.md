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
                               │  • Sharded v1/v2 Zarr Encoding               │
                               │  • Realtime Lead-Wave Scheduler & Discovery  │
                               │  • Retention Garbage Collector (GC Engine)   │
                               └──────────────┬────────────────┬──────────────┘
                                              │                │
                        Shard Files & Manifests│                │ Relational Catalog,
                                              │                │ Lifecycle & Advisory Locks
                                              ▼                ▼
                     ┌────────────────────────────────┐ ┌────────────────────────────────┐
                     │    Object Storage (S3/MinIO)   │ │    PostgreSQL 18 + PostGIS     │
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
                               │  • Sharded v1/v2 Byte-Range Chunk Reader     │
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
| **Ingestion Engine** | `services/ingestion` | Python 3.12, asyncio, `cfgrib`, `xarray`, `numcodecs`, `boto3`, `s3fs`, SQLAlchemy | Downloads GRIB2 messages, parses raw binary buffers, normalizes units, writes `sharded_v1` (default) or opt-in `sharded_v2` Zarr stores, schedules lead waves, and executes retention GC. |
| **Serving Tier** | `services/api` | Python 3.12, FastAPI, Uvicorn, SQLAlchemy, GeoAlchemy2, `xarray`, `s3fs`, Redis | Exposes REST endpoints, validates requested models/coordinates, acquires `SHARED` advisory locks, performs granular Range GETs against Zarr stores, and computes domain outputs. |
| **Frontend UI** | `services/frontend` | TypeScript, Next.js 14, React 18, MapLibre GL, Recharts, Tailwind CSS | Browser-based user interface for interactive map exploration, point meteograms, ensemble spaghetti/PDF charts, and location search. |
| **Domain Logic** | `packages/domain` | Python 3.12, NumPy, `xarray` | Pure, dependency-free mathematical domain models: grid coordinates, bilinear interpolation, ensemble statistics, precipitation phase classification, verification metrics, and advisory lock key derivation. |
| **Relational Database** | Infrastructure | PostgreSQL 18.6 + PostGIS 3.6.4 | Stores catalog hierarchy, progressive product availability, lifecycle fences, spatial point reference data, and coordinates distributed advisory locks. |
| **Object Store** | Infrastructure | AWS S3 / MinIO | Primary repository for all multidimensional meteorological raster data formatted in the sharded Zarr layout (`sharded_v1` default; `sharded_v2` opt-in — see §4.4). |
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
[ Sharded Writer ] ──► Pack 120 spatial chunks into binary shard containers
                        (encode_region_sharded dispatches on STORAGE_FORMAT_VERSION:
                         sharded_v1 default / sharded_v2 opt-in; identical geometry)
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

## 4. Storage Architecture (Sharded Container Formats)

Forecast grid data is stored in the **Weather Platform Sharded** binary container layout. Two shard-container formats are supported, selected by the ingestion setting `STORAGE_FORMAT_VERSION` (`ingestion/core/config.py`):

* **`sharded_v1`** — the frozen, live **default**. Every variable is stored as float32; historical byte behavior is frozen and must never change.
* **`sharded_v2`** — implemented and **opt-in, pending rollout** (not deprecated, and not yet the default). Shares the v1 container geometry but persists each variable in its authoritative per-variable native dtype (see §4.4).

Both formats share identical container geometry and store layout; only the chunk payload dtype differs.

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

The layout above is identical for `sharded_v1` and `sharded_v2`: the `0x53484152` ("SHAR") magic, 16-byte index entries, 12-byte trailer, Zstd level 5 compression, 120-chunk grid, and shard-key naming are all shared. In `sharded_v1` every chunk payload is float32; in `sharded_v2` the chunk dtype follows the per-variable matrix (§4.4).

### 4.3 Manifest & In-Place Overwrite Invariant
* **Staging Markers:** As each region writes, a JSON marker is placed at `__markers__/v1/{region_id}.json`.
* **Committed Manifest:** At finalization, `__commit__/v1/manifest.json` is written containing a unique `serving_generation` (UUID4 string), committed leads, and committed members.
* **Format Stamp:** The manifest records the committed `storage_format_version` durably at every publish/finalize, enabling per-store reader dispatch in the API (§4.4).
* **CRITICAL INVARIANT:**
  > **Logical generation is NOT an immutable physical generation.**  
  > Same-cycle re-ingestion overwrites the **same physical shard keys** in-place in object storage. Serving readers must hold the PostgreSQL `SHARED` advisory store gate across all chunk Range GETs to prevent reading partially overwritten containers.

### 4.4 Storage Format Versioning (`sharded_v1` / `sharded_v2`)

The ingestion setting `STORAGE_FORMAT_VERSION` (`ingestion/core/config.py`) selects the format for newly initialized forecast cycles: `"sharded_v1"` (default), `"sharded_v2"` (opt-in), or `"v2_unsharded"` (see below).

* **Per-cycle format freeze:** the first sharded commit of a cycle snapshots the format; a mid-cycle configuration flip fails loudly with `CycleFormatConflictError` (`ingestion/core/base.py`) instead of producing a half-v1/half-v2 store. The in-process writer snapshot covers flips between region commits, and the manifest's durable `storage_format_version` stamp (§4.3) covers flips across process restarts.
* **Per-variable native dtypes (`sharded_v2`):** resolved by the single source of truth `packages/domain/src/domain/storage_dtype.py::resolve_storage_dtype` — no module may keep its own copy of the matrix. Nine continuous variables are stored as little-endian float16 (`<f2`); `precipitation_amount_3h` and `cloud_ceiling` stay float32 (`<f4`) as *threshold-coupled semantic exceptions* (the exactly-0.10 mm dry/wet precipitation sentinel and the 19.99 km unlimited-ceiling sentinel); categorical precipitation flags (det/member roles) are uint8 (`u1`); ensemble-mean probability flags are float32. Pre-cast guards run before any cast: NaN is allowed, ±Inf and `|v| > 65504` are rejected, and flag domains are validated.
* **Rollout gate:** the shadow-validation tool (`ingestion/core/shadow.py`, operator CLI `scripts/shadow_v2.py` with `validate` / `cleanup`) re-encodes a canonical v1 store's committed shards into a dedicated `shadow-v2` namespace (never registered in the catalog, therefore never served) and asserts that f32-exception variables are byte-identical, zero threshold-predicate flips occur, and f16 differences stay within the 0.5 tolerance bound.
* **API reader dispatch (per store):** `api/core/zarr.py::get_sharded_reader` probes each store's committed manifest and returns a `ShardedV2Reader` (`api/core/zarr_v2.py`) when `storage_format_version == "sharded_v2"`, otherwise the frozen `ShardedV1Reader` (including for absent manifests). `ShardedV2Reader` subclasses `ShardedV1Reader` and overrides only `read_chunk` to decode and cache chunks in the native persisted dtype. Every serving call site (point forecasts, vector fields, map tiles, the ensemble-statistics readers, and the serving resolver) goes through `get_sharded_reader`.
* **`v2_unsharded` legacy fallback:** a third value naming the LEGACY unsharded-Zarr fallback for manifest-less stores, read through the xarray path. It is named confusingly close to `sharded_v2` but is entirely unrelated to it.
* **Rollback:** switching the setting back to `"sharded_v1"` returns new cycles to v1; already-committed v2 cycles keep serving through `ShardedV2Reader`. `sharded_v1` remains the current default; `sharded_v2` is pending rollout, not deprecated.
* **Numerical-equality scope:** the float16/float32 payload comparisons in the rollout gate (and any future golden-value layer) are defined **within a single platform run**; cross-architecture byte identity of float payloads is explicitly not a goal — consistency across x86_64/arm64 is guaranteed only at the quantized-output boundary (tiles, Int16 vector fields). See docs/TESTING.md, "Numerical-equality scope".

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
       ├── 8. get_sharded_reader: per-store manifest dispatch (ShardedV1Reader
       │      or ShardedV2Reader) performs S3 Range GET for chunk byte slices
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
3. **Map Tiles (`/v1/maps/{model}/{variable}/{level}/{z}/{x}/{y}.png`):** Generates 256×256 dynamic PNG tiles with meteorological color palettes. A strong ETag over the rendered bytes enables cheap HTTP 304 revalidation (`api/routers/maps.py`).
4. **Vector Wind Fields (`/v1/maps/{model}/wind_10m/vector-field`):** Serves gridded $u/v$ wind components for GPU-accelerated client particle animations.
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

[forecast_cycle_lifecycle] ((model_id, cycle_time) PK, deletion_started_at, deleted_at)
[reclamation_queue]        (id PK, run_id FK, status: queued/deleting/deleted/failed)
[forecast_variables]       (variable_code PK, name, unit)
[forecast_grids]           (grid_code PK, name, resolution_km)
[stations], [cities], [ski_resorts] (PostGIS spatial reference tables)
```

### Table Responsibilities:
* `model_runs`: Primary record of a model cycle (`status`, `zarr_store_path`, `cycle_time`). Status transitions: `processing` → `partial` → `ready` / `failed`.
* `forecast_products`: Records committed lead times per variable and grid.
* `ensemble_member_products`: Records committed `(member_index, lead_time_hours)` pairs for GEFS.
* `forecast_cycle_lifecycle`: Tracks durable physical lifecycle state: deletion claim fence (`deletion_started_at`) and anti-resurrection tombstone (`deleted_at`).
* `reclamation_queue`: Tracks granular variable shard reclamation lifecycle (`queued` → `deleting` → `deleted` / `failed`).
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
   - Caches parsed shard container index tables (up to 16,384 entries), decompressed 100×100 chunks (up to 512 chunks), and interpolated 2×2 corner values for point requests (up to 32,768 entries) per process.
   - For `sharded_v2` stores, `ShardedV2Reader` caches chunks in their native persisted dtype (f16 chunks stay f16, u8 stay u8); readers themselves are held in a bounded per-process LRU via `get_sharded_reader` (8 entries).

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

---

## 10. Multi-Architecture Platform Support

The Weather Platform is designed for native portability across standard server architectures:
* **Officially Supported Architectures:** `native linux/amd64` (x86_64) and `native linux/arm64` (aarch64).
* **Backing Services Multi-Arch:** PostgreSQL 18.6 with PostGIS 3.6.4 (`nickblah/postgis:18.6-trixie-postgis-3.6.4`), Redis 7 (`redis:7-alpine`), and MinIO (`alpine/minio:RELEASE.2025-10-15T17-29-55Z`, the community rebuild of official source releases after MinIO pulled its quay.io/Docker Hub community images) publish native multi-arch manifests, eliminating architecture pinning in Compose and deployment manifests.
* **C-Extension & Wheel Baseline:** Python scientific dependencies (`numpy`, `pandas`, `psycopg2-binary`, `zarr`, and `numcodecs 0.16.5`) resolve to prebuilt manylinux aarch64 wheels under CPython 3.12, avoiding host C compiler toolchain dependencies in multi-stage Docker builds.
* **Native ecCodes Decoding:** Ingestion images install Debian `libeccodes-dev` to provide runtime `libeccodes.so` for `cfgrib` on both x86_64 and aarch64.
* **Production Guardrails:** Production infrastructure strictly forbids `platform: linux/amd64` overrides or QEMU emulation dependencies. Build-time cross-architecture compatibility is continuously enforced via the `arm64-builds` CI pipeline job.

---

## 11. Location Discovery, Startup Coarse IP Localization & Presentation Timezones

The frontend application provides ambient geographic and temporal context on startup while preserving strict privacy invariants and keeping canonical user selections authoritative.

### 11.1 Startup Coarse IP Context vs. Canonical SelectedLocation

A fundamental invariant governs the location model:

$$\text{ApproximateStartupLocation} \neq \text{SelectedLocation}$$

```text
Hierarchy of Authority:
EXPLICIT USER INTENT  >  SELECTED LOCATION  >  STARTUP COARSE IP CONTEXT  >  DEFAULT CONUS / UTC FALLBACK
```

1. **`SelectedLocation` (Canonical Point Selection):**
   - Represents an explicit user action: Search selection, Map Click, or successful "Locate Me" browser geolocation.
   - Places a pinpoint selection marker on the map.
   - Opens the Forecast Dashboard sidebar.
   - Triggers point forecast queries (`/v1/points`), elevation lookup (`/v1/elevation`), and point-specific ensemble statistics (`/v1/ensembles`).
   - Participates in the monotonic race guard (`selectionGeneration`).
2. **`ApproximateStartupLocation` (Ambient Regional Context):**
   - Derived asynchronously on application mount via a best-effort, non-blocking request to `GET /v1/locate`.
   - **Never** sets `SelectedLocation` (`selectedLocation === null` on startup).
   - **Never** places a map marker or implies GPS-level precision.
   - **Never** opens the Forecast Dashboard or triggers point forecast / elevation queries.
   - **Never** increments or interacts with `selectionGeneration`.
   - Safely degrades silently to default CONUS (`[-106.8, 39.2]`, zoom 5) and UTC if `/v1/locate` is unavailable, disabled, or fails.

### 11.2 Dual-Use Architecture of `/v1/locate`

The infrastructure endpoint `GET /v1/locate` serves two distinct purposes in the frontend lifecycle:

```text
                     ┌───────────────────┐
                     │   GET /v1/locate  │
                     └─────────┬─────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
               ▼                               ▼
    [ Startup Lifecycle ]             [ Locate Me Fallback ]
  • Automatic, best-effort on mount • Explicit user click on Locate Me
  • Ambient context only            • Invoked ONLY on POSITION_UNAVAILABLE or TIMEOUT
  • NO SelectedLocation             • Commits canonical SelectedLocation
  • Regional easeTo (5-9 tiles)      • Point flyTo (zoom 8) + opens forecast
  • Fails silently to CONUS/UTC     • NEVER called on PERMISSION_DENIED (Privacy Invariant)
```

1. **Passive Startup Coarse Context:**
   - On application startup, `useStartupLocation()` and `useDisplayTimezone()` subscribe to `getCachedStartupLocation()`.
   - Request is executed once and cached in a module-level Promise, ensuring React Strict Mode double-mount safety without duplicate fetches or aborted requests.
2. **Explicit Browser Geolocation Fallback (`useGeolocation`):**
   - Triggered only by user click on "Locate Me" (`LocateMeButton`).
   - Requests native `navigator.geolocation.getCurrentPosition()`.
   - If geolocation fails with `POSITION_UNAVAILABLE` or `TIMEOUT`, falls back to `/v1/locate` and commits an approximate `SelectedLocation`.
   - **Privacy Invariant:** If the user denies browser permission (`PERMISSION_DENIED`), the explicit fallback is strictly aborted. `/v1/locate` is **never** called as a substitute for denied permission.

### 11.3 Infrastructure Trust Boundary & Production Deployment Requirements

`GET /v1/locate` consults two sources in order and answers the same HTTP 404 (`Location unavailable`) whenever neither can place the visitor: the frontend's silent degradation to the CONUS/UTC defaults therefore behaves identically no matter which branch failed.

1. **Trusted infrastructure headers (Cloudflare).** Extracts visitor coordinates from Cloudflare Managed Transforms headers (`cf-iplatitude`, `cf-iplongitude`, `cf-ipcity`, `cf-region`, `cf-ipcountry`).
   - **Default Safety:** The backend setting `TRUST_CLOUDFLARE_LOCATION_HEADERS` defaults to `False`. When disabled, the API returns HTTP 404 (`Location unavailable`) for all `/v1/locate` requests.
   - **Production Deployment Requirement:** Enabling `TRUST_CLOUDFLARE_LOCATION_HEADERS=True` is safe **only** when the API origin is strictly firewalled to Cloudflare IP ranges or authenticated using Cloudflare Authenticated Origin Pulls (mTLS). In this architecture, Cloudflare edge transforms sanitize client requests and inject authoritative geolocation headers; direct client-to-origin connections with spoofed `cf-*` headers are rejected at the network edge.
   - A malformed or absent header pair falls through to the second source rather than failing the request.
2. **Local GeoLite2 database (self-hosted origins).** With `LOCATE_PROVIDER=maxmind`, the visitor address is read off the request and resolved against `LOCATE_GEOIP_DB_PATH` (provisioned per DEPLOYMENT.md §7). This path exists for deployments where no CDN injects the headers above, so the `cf-*` branch can never produce a value.
   - **Address Selection (`LOCATE_PROXY_MODE`):** the serving tier always sits behind the edge gateway, so the socket peer alone would resolve every visitor to the gateway's own datacenter. The default `auto` therefore trusts the nginx-set `X-Real-IP` **only when the socket peer is not a routable address** — the case for anything arriving through the gateway — and ignores it when the peer is public, so a caller that reaches the published API port directly cannot forge a viewport. `always` trusts the header unconditionally (safe only with the API port firewalled to the gateway) and `never` always uses the socket peer. `X-Forwarded-For` is deliberately not consulted: nginx appends to it, making its leftmost entry caller-supplied.
   - **Usable addresses only:** loopback, private, link-local, CGNAT and documentation ranges are rejected, so local development and same-host requests degrade to 404 instead of resolving to whatever the database records for that range.
   - **Reader lifecycle:** the database is opened with `MODE_MMAP`, never `MODE_MEMORY`. Resident cost is therefore the pages actually touched (a few MB), shared across the uvicorn workers rather than duplicated per worker — the API container runs near its memory limit. The reader is reopened when the file's mtime/size changes, so the monthly refresh needs no restart. Results are not cached: a lookup costs microseconds, less than a Redis round trip, and skipping the cache removes both the stale-entry and per-IP key growth concerns.

### 11.4 Startup Camera Transition & Race Protection

When startup coarse IP coordinates arrive, the map gently transitions from the default CONUS view (`[-106.8, 39.2]`, zoom 5) to the approximate regional viewport (duration 800ms).

The regional viewport is sized by a **tile budget** rather than a fixed zoom: `startupViewZoom()` (`src/lib/map/startupView.ts`) frames `STARTUP_TILE_SPAN` (3) tiles of the `STARTUP_TILE_ZOOM` (z8) Web-Mercator grid across the measured map width, which on a landscape viewport is a 3x2 region — about 6 tiles, roughly 360 km of ground across at mid-latitudes — and keeps that width on any display while the height follows the viewport aspect. Sizing the ground area rather than the zoom is deliberate: MapLibre renders the world at `512 * 2^zoom` CSS px, so one fixed zoom frames a different amount of ground on every viewport, and the number of tiles on screen is a function of viewport pixels alone. Unmeasurable viewports (hidden container, no layout) fall back to the tile grid's nominal zoom.

Startup auto-centering is a **one-time opportunity** guarded by `startupAutoCenterEligibleRef`. It permanently expires when ANY of the following occurs:
1. The startup IP camera transition (`map.easeTo`) is successfully applied.
2. The user manually interacts with the map (drag, pan, rotate, pitch, or wheel/touch gesture).
3. A canonical `SelectedLocation` is ever committed (e.g. search result chosen before IP returns).
4. A "Locate Me" attempt is initiated, regardless of whether permission is granted, denied, or fails.

**Clearing Selection Invariant:** If a user selects a location (e.g. Tokyo) and subsequently closes the forecast panel (`selectedLocation` becomes null), the map camera **remains at its current position**. It does **not** ease back to the startup IP location.

### 11.5 Timezone Resolution Precedence

All backend meteorological datasets store timestamps in canonical UTC (ISO 8601 strings ending in `Z`). The frontend resolves presentation times using a three-tier precedence model:

$$\text{SelectedLocation timezone} > \text{Startup coarse IP timezone} > \text{UTC}$$

```typescript
displayTimezone = selectedLocationTimezone ?? startupCoarseIpTimezone ?? null;
```

- **Selected Location Active:** Forecast valid times in the adjacent Valid label, Hourly Meteograms, and Ensemble Statistics display in the IANA timezone derived from `SelectedLocation` coordinates (via client-side `@photostructure/tz-lookup`).
- **No Selection + Startup IP Available:** General forecast valid-time labels display in the coarse IP timezone (e.g. `Aug 13, 00:00 MDT`).
- **No Selection + Startup IP Unavailable (or 404):** Falls back to UTC (`Aug 13, 06:00 UTC`).
- **Valid Time Dropdown Invariant:** The Valid Time selector dropdown options **always remain canonical UTC** (`{formatDayHourUtc(vt)} UTC`) across all application states.


