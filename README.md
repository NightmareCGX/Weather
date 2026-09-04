# Global Probabilistic Weather Platform

A high-performance spatiotemporal weather data ingestion, storage, and serving platform for global numerical weather prediction (NWP) models.

The platform currently ingests, normalizes, stores, and serves operational forecasts from:
* **NOAA GFS** (Global Forecast System) — 0.25° deterministic global forecast (0–240h canonical horizon).
* **NOAA GEFS** (Global Ensemble Forecast System) — 0.5° 30-member ensemble forecast (0–240h canonical horizon, members `gep01`–`gep30`).

---

## 1. System Overview & Runtime Dataflow

```text
       ┌────────────────────────────────────────────────────────┐
       │             Upstream Meteorological Source             │
       │       NOAA NOMADS / AWS Open Data S3 (.idx + GRIB2)    │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │           services/ingestion (weather-ingest)          │
       │  Selective Byte-Range Download → Multiprocess Decode   │
       │  Unit Normalization → Sharded v1 Zarr Encoding         │
       └──────────────┬──────────────────────────┬──────────────┘
                      │                          │
                      ▼ (shards & manifests)     ▼ (catalog metadata & locks)
       ┌──────────────────────────────┐ ┌──────────────────────────────┐
       │   Object Storage (S3/MinIO)  │ │ PostgreSQL 16 + PostGIS      │
       │   s3://weather-data/{model}/ │ │ Relational Catalog & Runs    │
       │   {date}/{hour}/cycle.zarr/  │ │ Advisory-Lock Concurrency    │
       └──────────────┬───────────────┘ └──────────────┬───────────────┘
                      │                                │
                      └───────────────┬────────────────┘
                                      │
                                      ▼
       ┌────────────────────────────────────────────────────────┐
       │                     services/api                       │
       │  FastAPI Serving Layer + SHARED Advisory Reader Gate   │
       │  Sharded v1 Reader + Bilinear Interpolation            │
       │  Ensemble Stats/PDFs + Map Tiles + Redis Hot Cache     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                  services/frontend                     │
       │  Next.js 14 / React 18 UI                              │
       │  MapLibre GL Weather Map + Recharts Forecast Dashboard │
       └────────────────────────────────────────────────────────┘
```

### Core Pipeline Capabilities:
1. **Selective Ingestion:** Fetches only requested meteorological fields using HTTP `.idx` byte-range GETs or direct S3 downloads, decoding GRIB2 files with multiprocessing worker pools (`cfgrib`/`ecCodes`).
2. **Sharded v1 Storage Layout:** Stores tensor forecast data in canonical binary shard containers (120 spatial chunks of 100×100 per container) indexed at the tail, enabling granular single-cell or bounding-box Range GETs without reading entire forecast fields.
3. **Progressive Availability:** Ingests forecasts in lead waves. Leads are published progressively to the PostgreSQL catalog (`partial` status) until the full canonical horizon is committed (`ready` status).
4. **Concurrency & Serving Correctness:** Concurrency between concurrent writers and serving readers is coordinated via PostgreSQL 64-bit advisory locks (`SHARED` reader gate vs `EXCLUSIVE` finalizer/GC gate).
5. **High-Performance Serving:** FastAPI serves point forecasts (bilinearly interpolated), ensemble summary statistics and PDF distributions, dynamic raster map tiles (PNG), and vector wind fields.
6. **Data-Driven Frontend:** Next.js application with interactive MapLibre meteorological overlays, meteograms, ensemble spaghetti/distribution charts, and location search.

---

## 2. Repository Structure

```text
Weather/
├── services/
│   ├── api/             # FastAPI serving service, Alembic migrations, reader gate
│   ├── ingestion/       # Ingestion engine, GRIB2 parser, realtime scheduler, GC
│   └── frontend/        # Next.js 14 web client (React 18, MapLibre GL, Recharts)
├── packages/
│   ├── domain/          # Pure domain logic (meteorology, grid math, ensemble stats, locks)
│   ├── contracts/       # Skeleton placeholder package (not an active runtime dependency)
│   └── config/          # Skeleton placeholder package (not an active runtime dependency)
├── docker/              # Production multi-stage Dockerfiles (API, Ingestion, Frontend)
├── docs/                # Architecture, API specifications, database design, runbooks
├── docker-compose.yml   # Local development infrastructure (PostgreSQL, Redis, MinIO)
└── pyproject.toml       # Root Poetry workspace definition
```

> **Note on Shared Packages:** `packages/contracts` and `packages/config` currently exist as placeholder packages reserved for future contract extraction. At present, `services/api` and `services/ingestion` maintain their own internal configuration modules (`api.core.config` and `ingestion.core.config`), and both import `packages/domain` for pure domain models and locking identity derivation.

---

## 3. Local Development Quickstart

### Prerequisites
* **Python:** 3.12+
* **Poetry:** 2.4.1+
* **Node.js:** 20+ (with `npm`)
* **Docker & Docker Compose**
* **System Library (Linux only):** `libeccodes-dev` (GRIB2 decoding; on Windows the Python `eccodes` wheel bundles the native library)

---

### Step 1: Start Local Infrastructure
Launch PostgreSQL 16 (PostGIS 3.4), Redis 7, and MinIO:
```bash
docker-compose up -d
```

Services are exposed at:
* **PostgreSQL:** `localhost:5432` (`weather_user` / `weather_password` / `weather_db`)
* **Redis:** `localhost:6379`
* **MinIO API:** `localhost:9000` (`minio_admin` / `minio_password`)
* **MinIO Console:** `http://localhost:9001`

---

### Step 2: Install Python & Frontend Dependencies

```bash
# Install domain package
cd packages/domain && poetry install && cd ../..

# Install API service
cd services/api && poetry install && cd ../..

# Install Ingestion service
cd services/ingestion && poetry install && cd ../..

# Install Frontend dependencies
cd services/frontend && npm ci && cd ../..
```

---

### Step 3: Run Database Migrations
Apply schema migrations (001–004) to PostgreSQL:
```bash
cd services/api && poetry run alembic upgrade head && cd ../..
```

---

### Step 4: Start Applications

#### Start FastAPI Serving Tier (Port 8000)
```bash
cd services/api
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation is available at `http://localhost:8000/docs`.

#### Start Next.js Frontend (Port 3000)
In a separate terminal:
```bash
cd services/frontend
npm run dev
```
Open `http://localhost:3000` to access the weather explorer interface.

---

## 4. Ingestion & Operational Commands

The ingestion service provides the `weather-ingest` CLI (`services/ingestion/src/ingestion/cli.py`).

### Manual Batch Ingestion
Download, parse, and commit specific forecast cycles and leads into object storage and catalog:

```bash
# Ingest GFS deterministic leads 0, 3, 6, 9, 12 for 00Z cycle:
cd services/ingestion
poetry run weather-ingest ingest --model gfs --cycle-date 2026-09-03 --cycle-hour 0 --lead-time-hours 0 3 6 9 12

# Ingest GEFS ensemble perturbation members 1, 2, 3 for leads 0, 3, 6:
poetry run weather-ingest ingest --model gefs --cycle-date 2026-09-03 --cycle-hour 0 --lead-time-hours 0 3 6 --member 1 2 3
```

### Realtime Lead-Wave Scheduler
Polls upstream NOAA for new publication activity and dispatches wave ingestion under distributed PostgreSQL advisory leader locks:

```bash
cd services/ingestion
# Run a single discovery and wave iteration:
poetry run weather-ingest realtime --once

# Run continuous realtime scheduling daemon:
poetry run weather-ingest realtime
```

### Retention Garbage Collection (GC)
Reconciles retired cycles, sets durable deletion fences, and deletes expired S3 stores sequentially under exclusive store gates:

```bash
cd services/ingestion
# Run a single GC reconciliation pass:
poetry run weather-ingest gc --once
```

---

## 5. Testing, Linting & Quality Checks

```bash
# Domain tests (100% coverage gate required)
cd packages/domain && poetry run pytest && cd ../..

# API integration tests (requires PostgreSQL + Redis)
cd services/api && poetry run pytest && cd ../..

# Ingestion integration tests (requires PostgreSQL + Redis + MinIO)
cd services/ingestion && poetry run pytest && cd ../..

# Frontend unit tests
cd services/frontend && npm test && cd ../..

# Code style and type checking
cd packages/domain && poetry run ruff check . && poetry run mypy && cd ../..
cd services/api && poetry run ruff check . && poetry run mypy && cd ../..
cd services/ingestion && poetry run ruff check . && poetry run mypy && cd ../..
cd services/frontend && npm run lint && npm run typecheck && npm run format:check && cd ../..
```

---

## 6. Current Project Status

* **Implemented & Verified:**
  * NOAA GFS (0.25°) and GEFS (0.5° 30-member) operational ingestion pipelines.
  * `sharded_v1` binary Zarr container storage and granular Range GET readers.
  * Realtime lead-wave scheduler with automated discovery and progressive publication.
  * PostgreSQL advisory-lock coordination (`SHARED` reader gate vs `EXCLUSIVE` writer/GC gate).
  * FastAPI serving layer for point forecasts, ensemble statistics/PDFs, map tiles, and vector fields.
  * Next.js 14 interactive map and forecast dashboard frontend.
  * Cycle supersession lifecycle tracking and storage retention GC.
* **Under Active Development (Stage 7):**
  * Engineering baseline standardization, CI workflow redesign, and deployment runbooks.
* **Deferred Backlog Items:**
  * Additional model expansions (HRRR, ECMWF) and upper-air isobaric variable expansion.
  * Precomputed ensemble mean fields and multi-lead request batching optimizations.
  * Reader lock narrowing / generation pinning (deferred until immutable physical storage paths are introduced).

---

## 7. Documentation Index

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Detailed runtime architecture, storage format, and concurrency models.
* [`docs/API.md`](docs/API.md) — HTTP API endpoints, request parameters, and response schemas.
* [`docs/DATABASE.md`](docs/DATABASE.md) — PostgreSQL relational schema, migrations, and table ownership.
* [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Docker container builds and environment configuration.
* [`docs/RUNBOOKS.md`](docs/RUNBOOKS.md) — Operational runbooks, failure recovery procedures, and diagnostics.
* [`docs/MODELS.md`](docs/MODELS.md) — Meteorological model specifications and variable mapping contracts.
* [`docs/TESTING.md`](docs/TESTING.md) — Testing philosophy, coverage requirements, and CI matrix.
