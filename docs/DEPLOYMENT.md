# Deployment & Operations Guide

This document describes the currently implemented deployment architecture, container images, configuration requirements, and operational execution modes for the Weather Platform.

> **Operational Runbooks:** For step-by-step procedures covering fresh deployment bootstrapping, database migrations, failure recovery, leadership failover, and operational diagnostic queries, see [`docs/RUNBOOKS.md`](RUNBOOKS.md).

---

## 1. Local Development Infrastructure (`docker-compose.yml`)

For local development and integration testing, backing services run via Docker Compose:

* **PostgreSQL 16 + PostGIS 3.4 (`postgis/postgis:16-3.4`):**
  * Port: `5432`
  * Database: `weather_db`
  * Credentials: `weather_user` / `weather_password`
  * Stores relational catalog, forecast product availability, lifecycle fences, and coordinates PostgreSQL advisory locks.
* **Redis 7 (`redis:7-alpine`):**
  * Port: `6379`
  * In-memory cache for API point forecast JSON envelopes and vector field grids.
* **MinIO (`minio/minio:latest`):**
  * S3 API: `http://localhost:9000`
  * Console: `http://localhost:9001`
  * Credentials: `minio_admin` / `minio_password`
  * S3-compatible object storage bucket: `weather-data`

To launch:
```bash
docker-compose up -d
```

---

## 2. Container Images (`docker/`)

The repository provides multi-stage production Dockerfiles:

### 2.1 API Image (`docker/Dockerfile.api`)
* **Base:** `python:3.12-slim` (non-root `appuser`, UID 1001).
* **Builder Stage:** Installs Poetry 2.4.1, resolves path dependencies (`packages/domain`), and builds a self-contained in-project virtual environment (`--only main`).
* **Runtime:** Copies venv, API source, Alembic migrations, and configuration. Runs `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
* **Healthcheck:** Probes `http://127.0.0.1:8000/docs` (or `/v1/health`).
* **Build Command:**
  ```bash
  docker build -f docker/Dockerfile.api -t weather-api:latest .
  ```

### 2.2 Ingestion Image (`docker/Dockerfile.ingestion`)
* **Base:** `python:3.12-slim` (non-root `appuser`, UID 1001).
* **System Packages:** Installs `libeccodes-dev` (required for GRIB2 decoding on Linux).
* **Runtime:** Self-contained Poetry venv + `packages/domain` source + `services/ingestion` source.
* **Entrypoint:** `weather-ingest` CLI.
* **Build Command:**
  ```bash
  docker build -f docker/Dockerfile.ingestion -t weather-ingestion:latest .
  ```

### 2.3 Frontend Image (`docker/Dockerfile.frontend`)
* **Base:** `node:20-alpine` (non-root `nextjs`, GID/UID 1001).
* **Build Output:** Next.js `output: "standalone"` (`.next/standalone`, `.next/static`, `public`).
* **Build Argument:** `API_PROXY_TARGET` (defaults to `http://127.0.0.1:8000`). Note: Next.js bakes rewrite destinations into routes-manifest at build time.
* **Entrypoint:** `node server.js`.
* **Build Command:**
  ```bash
  docker build -f docker/Dockerfile.frontend --build-arg API_PROXY_TARGET=http://api:8000 -t weather-frontend:latest services/frontend
  ```

---

## 3. Production Configuration & Environment Variables

### 3.1 Common Infrastructure Settings
* `DATABASE_URL`: PostgreSQL connection string (e.g. `postgresql://user:pass@db-host:5432/weather_db`).
* `REDIS_URL`: Redis connection string (e.g. `redis://redis-host:6379/0`).
* `MINIO_ENDPOINT`: S3/MinIO host and port (e.g. `s3.us-east-1.amazonaws.com` or `minio:9000`).
* `MINIO_ACCESS_KEY`: S3 access key ID.
* `MINIO_SECRET_KEY`: S3 secret access key.
* `MINIO_SECURE`: `true` for HTTPS (production AWS S3 / CloudFlare R2), `false` for plain HTTP (local MinIO).
* `MINIO_BUCKET_NAME`: Target object storage bucket (default `weather-data`).

### 3.2 Ingestion-Specific Settings
* `NOAA_DOWNLOAD_SOURCE`: `aws_s3` (recommended for production) or `nomads`.
* `ENABLE_NOMADS_FALLBACK`: `true` (automatically fall back to NOMADS on AWS S3 404 publication lag).
* `ENABLE_SELECTIVE_DOWNLOAD`: `true` (enables `.idx` byte-range partial file download).
* `DB_POOL_SIZE`: Ingestion PostgreSQL connection pool size (default `10`).
* `MAX_WRITE_CONCURRENCY`: Parallel region write worker ceiling (must be $\le \text{DB\_POOL\_SIZE}$, default `6`).
* `GLOBAL_PUT_CONCURRENCY`: Shard object upload concurrency (default `64`).
* `REALTIME_ENABLED`: `true` to enable realtime polling daemon.
* `REALTIME_ACTIVE_POLL_SECONDS`: Cadence for active cycle tracking (default `600.0` seconds).
* `REALTIME_WAVE_MAX_LEADS`: Number of accumulated leads before dispatching a wave (default `8`).

### 3.3 API-Specific Settings
* `API_READER_LOCK_POOL_SIZE`: Dedicated connection pool size for PostgreSQL reader advisory locks (default `16`).
* `API_READER_LOCK_MAX_OVERFLOW`: Overflow connections for reader locks (default `8`).
* `API_READER_GATE_TIMEOUT_SECONDS`: Maximum wait time to acquire `SHARED` store gate (default `30.0` seconds).
* `SEARCH_PROVIDER`: `google` (Google Places API) or `mapbox` (Mapbox Geocoding).
* `GOOGLE_PLACES_API_KEY`: Server-side API key for Google Places (New).
* `ELEVATION_PROVIDER`: `dem` (local DEM raster) or `none`.
* `DEM_DATA_PATH`: Local or S3 path to global DEM Zarr/NetCDF dataset.

---

## 4. Operational Execution & Runbooks

### 4.1 Database Migration
Run database migrations before starting application services:
```bash
cd services/api
poetry run alembic upgrade head
```

### 4.2 Starting the Serving Tier
```bash
cd services/api
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4.3 Starting the Realtime Ingestion Daemon
The realtime scheduler automatically probes upstream NOAA publication, coordinates leadership via PostgreSQL advisory locks, and dispatches lead waves:
```bash
cd services/ingestion
poetry run weather-ingest realtime
```

### 4.4 Running Retention Garbage Collection (GC)
Execute the GC reconciler to reclaim expired S3 stores and set deletion tombstones:
```bash
cd services/ingestion
# Single-pass execution (e.g. from cron):
poetry run weather-ingest gc --once

# Continuous daemon mode:
poetry run weather-ingest gc --interval-seconds 1800
```

---

## 5. Production Orchestration Status

* **Currently Implemented:** Docker multi-stage container builds, local Docker Compose stack, database migrations via Alembic, CLI daemon entrypoints for API, Ingestion, and GC, and the complete operational runbook framework in [`docs/RUNBOOKS.md`](RUNBOOKS.md).
* **Stage 8 Server Deployment Handoff:**
  * Specific physical server sizing (CPU/RAM), PostgreSQL `max_connections`, API worker counts, and production supervisor definitions (systemd unit files, Docker Compose production profiles, or Kubernetes manifests) will be finalized during Stage 8 server deployment based on the parameter placeholders in `docs/RUNBOOKS.md`.
