# Global Probabilistic Weather Platform

A production-grade, global numerical weather prediction (NWP) data ingestion, storage, serving, and interactive visualization platform.

The platform ingests operational high-resolution global weather models from the National Oceanic and Atmospheric Administration (**NOAA GFS** deterministic and **NOAA GEFS** 30-member ensemble), parses multi-message GRIB2 streams using high-performance byte-range HTTP streaming and process-isolated decoding, normalizes meteorological variables into multidimensional **Zarr** datasets on S3-compatible object storage, and indexes metadata in **PostgreSQL 16 + PostGIS**.

Forecasts are served via a high-throughput **FastAPI** REST API protected by a PostgreSQL advisory-lock reader gate and cached via **Redis**, and visualized through an interactive **Next.js 14** web application powered by **MapLibre GL JS** (raster tiles and animated wind particle flows) and **Recharts** (meteograms, ensemble percentile spreads, probability distributions, precipitation phase evolution, and wind roses).

---

## Current Implemented Status

The repository currently implements the complete end-to-end data pipeline, API serving surface, and web application:

* **Ingestion Engine**: Automated NOAA GFS (0.25° deterministic) and GEFS (0.25° 30-member perturbation ensemble `gep01`..`gep30`) downloader with AWS S3 / NOMADS `.idx` byte-range selection (>97% bandwidth savings), process-isolated `cfgrib`/`ecCodes` decoding, 3-hour precipitation de-accumulation, cloud cover reconstruction, and atomic Zarr chunk writes on S3/MinIO with PostgreSQL catalog synchronization.
* **Domain Engine (`packages/domain`)**: Comprehensive pure-domain business logic featuring 100% test coverage: spatial bilinear/nearest/conservative interpolation, ensemble statistics (mean, median, spread, P10–P90 percentiles, Gaussian KDE probability density function estimation, threshold exceedance/subceedance probabilities with Wilson score confidence intervals), meteorological wind vector math, precipitation phase classification and transition state modeling, cloud ceiling height and coverage analysis, and advisory-lock key derivation.
* **Serving & API Layer (`services/api`)**: FastAPI `/v1` REST surface exposing catalog discovery, forecast availability, point forecast interpolation (coordinates, cities, ski resorts), place search autocomplete (Google Places API New integration), ensemble statistics, threshold probabilities, verification error metrics (RMSE, MAE, bias), server-side Web Mercator XYZ raster PNG tile rendering, and binary wind vector field streaming.
* **Storage & Caching Architecture**: Multidimensional Zarr datasets on S3/MinIO, PostgreSQL 16 + PostGIS spatial indexing, Redis response caching, and a robust advisory-lock reader gate preventing race conditions between concurrent readers and ingestion re-writes.
* **Web Client (`services/frontend`)**: Next.js 14 (App Router) client with MapLibre GL JS raster layer overlays, animated wind vector particle flow, place search, point forecast meteograms, ensemble spaghetti and percentile band charts, PDF distribution inspection, precipitation phase probabilities, and 16-sector wind roses.
* **CI/CD & Container Builds**: 8-job GitHub Actions matrix (`.github/workflows/ci.yml`) enforcing dual-platform Windows/Linux compatibility, 100% domain test coverage, Ruff/MyPy quality gates, Jest unit tests, Playwright E2E tests, and multi-stage production Docker container builds.

> **Roadmap Note**: ECMWF (IFS/AIFS) and ECCC (GDPS/GEPS) model providers, Neural AI downscaling (25km to 3km/1km), Multi-Model Ensemble (MME) calibration, and commercial subscription billing are future expansion phases and are not yet implemented.

---

## Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Upstream["Upstream NOAA Weather Data"]
        AWS["AWS Open Data S3 (Primary)"]
        NOMADS["NOAA NOMADS (Fallback)"]
    end

    subgraph Ingestion["Ingestion Service (weather-ingest CLI)"]
        IDX["Index Parser & Range Selector (.idx)"]
        DL["Async HTTP Range Downloader"]
        DECODE["ProcessPool ecCodes / cfgrib Decoder"]
        NORM["Normalization, De-accumulation & Mapping"]
        ZARR_W["Zarr Writer (Chunked Array Slices)"]
    end

    subgraph Storage["Storage & Relational Tier"]
        MINIO[("Object Storage / MinIO / S3\n(Zarr Datasets & Markers)")]
        POSTGRES[("PostgreSQL 16 + PostGIS\n(Catalog, Runs, Spatial Index)")]
        REDIS[("Redis 7\n(Query & Tile Caches)")]
    end

    subgraph API["API Serving Tier (FastAPI)"]
        GATE["Advisory Lock Reader Gate"]
        POINTS["Point Forecast & Interpolation Service"]
        TILES["Server-side PNG Tile & Vector Renderer"]
        PROB["Ensemble & Probability Engine"]
        SEARCH_SVC["Place Search Service (Google Places)"]
    end

    subgraph Frontend["Frontend Client (Next.js 14)"]
        MAP["MapLibre GL JS Map\n(Raster Layers & Wind Particle Flow)"]
        DASH["Forecast Dashboard & Meteograms"]
        ENS_UI["Ensemble PDF, Phase & Wind Rose Charts"]
        SEARCH_UI["Location Search Bar"]
    end

    AWS --> IDX
    NOMADS --> IDX
    IDX --> DL
    DL --> DECODE
    DECODE --> NORM
    NORM --> ZARR_W
    ZARR_W -->|Zarr Chunks & Manifests| MINIO
    ZARR_W -->|Catalog Records & Markers| POSTGRES

    POSTGRES <-->|Catalog & Locks| GATE
    MINIO <-->|Gated Slices| GATE
    GATE --> POINTS
    GATE --> TILES
    GATE --> PROB
    REDIS <-->|Cache Hit / Miss| POINTS
    SEARCH_SVC <-->|Geocoding / Places| POSTGRES

    POINTS -->|/v1/points| DASH
    PROB -->|/v1/probabilities & /v1/ensembles| ENS_UI
    TILES -->|/v1/maps/.../*.png & vector-field| MAP
    SEARCH_SVC -->|/v1/search| SEARCH_UI
```

---

## Repository Structure

```text
weather-platform/
├── .github/
│   └── workflows/
│       └── ci.yml                # 8-job GitHub Actions CI/CD pipeline
├── docker/
│   ├── Dockerfile.api            # Multi-stage production image for FastAPI
│   ├── Dockerfile.ingestion      # Multi-stage production image for weather-ingest CLI
│   └── Dockerfile.frontend       # Multi-stage standalone image for Next.js
├── docs/                         # Comprehensive architectural and technical documentation
│   ├── ARCHITECTURE.md           # System architecture, components, and data flow
│   ├── API.md                    # OpenAPI/REST contract specifications
│   ├── DATABASE.md               # PostgreSQL + PostGIS schema and spatial indexes
│   ├── MODELS.md                 # Meteorological model specifications and parameters
│   ├── TESTING.md                # Testing strategy, test fixtures, and coverage rules
│   ├── DEPLOYMENT.md             # Infrastructure deployment and container specs
│   └── CONTRIBUTING.md           # Developer guidelines and engineering standards
├── packages/                     # Shared workspace libraries
│   ├── domain/                   # Core domain logic, spatial math, ensemble statistics
│   │   ├── src/domain/           # coordinates, grid, interpolation, ensemble, wind, precip, cloud, locks
│   │   └── tests/                # 100% coverage unit test suite
│   ├── contracts/                # Shared Pydantic contract definitions (stub)
│   └── config/                   # Centralized configuration stubs
├── services/                     # Application services
│   ├── api/                      # FastAPI core backend service
│   │   ├── alembic/              # Database migration versions (001_initial_schema, 002_ensemble_member_products)
│   │   ├── src/api/              # routers, services (point_forecast, tiles, vector_field, elevation, places), core
│   │   └── tests/                # Integration and contract tests
│   ├── ingestion/                # Ingestion engine and weather-ingest CLI
│   │   ├── src/ingestion/        # CLI, NOAA connector, decode pool, coordinator, zarr writer, markers
│   │   └── tests/                # Integration tests (MinIO round-trip, concurrency, S3 lifecycle)
│   └── frontend/                 # Next.js 14 interactive web client
│       ├── src/app/              # Next.js App Router root layout and main page
│       ├── src/components/       # MapLibre map, layer controls, meteograms, ensemble charts, search
│       ├── src/hooks/            # API hooks (usePointForecast, useEnsemble, useVectorField, useSearch)
│       └── src/lib/              # Map layers, wind particle engine, API client, legend generators
├── docker-compose.yml            # Local backing infrastructure (PostgreSQL 16, Redis 7, MinIO)
├── .env.example                  # Environment configuration template
├── CLAUDE.md                     # AI engineering contract and platform guidelines
├── ENGINEERING_CONTRACT.md       # Mandatory architectural and quality contracts
├── IMPLEMENTATION_PLAN.md        # Master milestone roadmap and tracking
├── pyproject.toml                # Root Poetry workspace manifest
└── README.md                     # Root project documentation (this file)
```

---

## Implemented Product Capabilities

### 1. Interactive Weather Map
* **Web Mercator Raster Tiles**: Dynamic 256x256 PNG tiles rendered directly from Zarr datasets (`/v1/maps/{model}/{variable}/{level}/{z}/{x}/{y}.png`) with continuous color ramps for temperature, precipitation rate, 3-hour precipitation accumulation, relative humidity, wind gust, visibility, snow depth, cloud cover, and cloud ceiling.
* **Animated Wind Vector Particle Flow**: Real-time particle animation driven by server-streamed binary vector fields (`/v1/maps/{model}/wind_10m/vector-field`) encoding 10m U/V components for GFS and consensus mean vectors for GEFS.
* **Interactive Timeline & Controls**: Lead time scrubber (0h to 384h), automatic playback controls, model selection (GFS vs. GEFS), variable switching, and layer opacity adjustment.
* **Dynamic Legends**: Client-side legend rendering synchronizing color stops, units, and ranges directly with the active map layer.

### 2. Point Forecasts & Spatial Interpolation
* **Flexible Location Resolution**: Query forecasts by exact coordinates (`latitude`/`longitude`), city name, or ski resort identifier.
* **Spatial Interpolation**: Point extraction supporting `nearest`, `bilinear`, and `conservative` interpolation against regular global grids.
* **Elevation Correction**: Topographic elevation adjustments for surface temperature using local DEM data (Copernicus 30m) or geodetic station elevation.
* **Deterministic & Ensemble Timeseries**: Detailed hourly forecast time series covering temperature, precipitation, wind, humidity, visibility, snow depth, cloud cover, and cloud ceiling.

### 3. Probabilistic & Ensemble Analysis
* **Full Ensemble Member Ingestion**: Ingests all 30 individual GEFS perturbation members (`gep01`–`gep30`) preserving individual member trajectories across leads.
* **Statistical Reductions**: On-demand computation of ensemble mean, median, spread (standard deviation), and percentiles (P10, P25, P50, P75, P90).
* **Probability Density Function (PDF) Estimation**: Non-parametric Gaussian Kernel Density Estimation (KDE) with automated Scott's rule bandwidth calculation and empirical bounds.
* **Threshold Exceedance & Intervals**: Calculates probabilities of variables exceeding, falling below, or staying between thresholds (e.g. `Pr(precip > 5.0 mm)` or `Pr(temp < 0 °C)`), accompanied by Wilson score confidence intervals.

### 4. Advanced Meteorological Variable Models
* **Precipitation Phase & Evolution**: Joint amount and phase analysis classifying categorical precipitation into Rain, Snow, Freezing Rain, and Ice Pellets; calculates phase agreement across ensemble members and transition probability matrices across forecast leads.
* **3-Hour Precipitation De-accumulation**: Robust de-accumulation of GFS/GEFS 6-hour reset accumulation buckets (`tp`) into discrete 3-hour precipitation increments (`precipitation_amount_3h`).
* **10m Wind & Wind Rose**: U/V vector decomposition, meteorological wind direction derivation (0–360°), cardinal heading conversion, consensus ensemble vector derivation, and 16-sector wind rose speed/frequency distributions.
* **Cloud Cover & Ceiling**: Reconstructs 3-hour cloud cover intervals (`cloud_cover_3h`), evaluates cloud ceiling heights (`cloud_ceiling`), identifies unlimited ceiling sentinels (>19,990m), and calculates low-ceiling aviation risk probabilities.

### 5. Location Search & Place Autocomplete
* **Debounced Place Autocomplete**: Integration with Google Places API (New) supporting location search, structured address parsing, and coordinate resolution.
* **Fallback & Offline Geocoding**: Local PostGIS spatial database lookup for cities and ski resorts when external search providers are unconfigured.

### 6. Model Verification & Quality Metrics
* **Statistical Error Metrics**: Endpoints evaluating forecast accuracy against observed station measurements, computing Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and mean forecast bias.

---

## Supported Models & Data Coverage

| Model Identifier | Model Name | Type | Horizontal Resolution | Forecast Cycles | Forecast Leads | Members |
|---|---|---|---|---|---|---|
| `gfs` | Global Forecast System (NOAA) | Deterministic | 0.25° (~25 km) | 00Z, 06Z, 12Z, 18Z | 0h – 384h (3h / 6h steps) | 1 (deterministic) |
| `gefs` | Global Ensemble Forecast System (NOAA) | Ensemble | 0.25° (~25 km) | 00Z, 06Z, 12Z, 18Z | 0h – 384h (3h / 6h steps) | 30 perturbation members (`gep01`–`gep30`) |

### Supported Platform Variables

The platform normalizes incoming GRIB2 messages to the following 15 canonical surface variables:

| Variable Code | Name | Canonical Unit | Upstream GRIB Selector |
|---|---|---|---|
| `temperature_2m` | 2-Meter Temperature | °C | `shortName=2t`, `typeOfLevel=heightAboveGround`, `level=2` |
| `precipitation_rate` | Precipitation Rate | mm/h | `shortName=prate`, `typeOfLevel=surface`, `level=0`, `stepType=instant` |
| `precipitation_amount_3h` | 3-Hour Precipitation Amount | mm | `shortName=tp`, `typeOfLevel=surface`, `level=0`, `stepType=accum` (de-accumulated) |
| `crain` | Categorical Rain Flag | binary (0/1) | `shortName=crain`, `typeOfLevel=surface`, `level=0` |
| `csnow` | Categorical Snow Flag | binary (0/1) | `shortName=csnow`, `typeOfLevel=surface`, `level=0` |
| `cfrzr` | Categorical Freezing Rain Flag | binary (0/1) | `shortName=cfrzr`, `typeOfLevel=surface`, `level=0` |
| `cicep` | Categorical Ice Pellets Flag | binary (0/1) | `shortName=cicep`, `typeOfLevel=surface`, `level=0` |
| `relative_humidity_2m` | 2-Meter Relative Humidity | % | `shortName=2r`, `typeOfLevel=heightAboveGround`, `level=2` |
| `wind_gust` | Wind Gust | km/h | `shortName=gust`, `typeOfLevel=surface`, `level=0` |
| `visibility` | Visibility | m | `shortName=vis`, `typeOfLevel=surface`, `level=0` |
| `snow_depth` | Snow Depth | m | `shortName=sde`, `typeOfLevel=surface`, `level=0` |
| `wind_u_10m` | 10-Meter U Wind Component | m/s | `shortName=10u`, `typeOfLevel=heightAboveGround`, `level=10` |
| `wind_v_10m` | 10-Meter V Wind Component | m/s | `shortName=10v`, `typeOfLevel=heightAboveGround`, `level=10` |
| `cloud_cover_3h` | 3-Hour Cloud Cover | % | `shortName=tcc`, `typeOfLevel=atmosphere`, `level=0` |
| `cloud_ceiling` | Cloud Ceiling Height | m | `shortName=gh`, `typeOfLevel=cloudCeiling`, `level=0` |

---

## Technology Stack & Prerequisites

### System Requirements
* **Python**: `3.12.x`
* **Poetry**: `2.4.1`
* **Node.js**: `>=18.17.0` (Node `20.x` recommended)
* **Docker & Docker Compose**: For local PostgreSQL, Redis, and MinIO
* **C-Libraries (Linux)**: `libeccodes-dev` (required on Linux for `cfgrib`/`eccodes` GRIB2 decoding; on Windows, the pre-built `eccodes` wheel bundles the binary library).

---

## Local Development Setup

### 1. Clone & Environment Configuration

```bash
git clone https://github.com/your-org/weather-platform.git
cd weather-platform

# Copy the environment template
cp .env.example .env
```

### 2. Start Backing Infrastructure

Start local PostgreSQL (PostGIS 16), Redis 7, and MinIO using Docker Compose:

```bash
docker compose up -d
```

Verify service health:
* **PostgreSQL / PostGIS**: `localhost:5432` (`weather_user` / `weather_password`, database `weather_db`)
* **Redis**: `localhost:6379`
* **MinIO S3 API**: `http://localhost:9000` (`minio_admin` / `minio_password`)
* **MinIO Web Console**: `http://localhost:9001`

### 3. Install Dependencies

Install workspace packages and services:

```bash
# Shared domain package
cd packages/domain
poetry install

# Backend API service
cd ../../services/api
poetry install

# Ingestion service & CLI
cd ../ingestion
poetry install

# Frontend web client
cd ../frontend
npm ci
cd ../..
```

### 4. Database Migrations

Apply Alembic migrations to create all database tables, foreign keys, and PostGIS spatial indexes:

```bash
cd services/api
poetry run alembic upgrade head
cd ../..
```

---

## Running the Platform

### Running Services Locally

Open three terminal sessions to run the development stack:

#### 1. Backend API (FastAPI)
```bash
cd services/api
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
* API Documentation (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/v1/health](http://localhost:8000/v1/health)

#### 2. Frontend Client (Next.js)
```bash
cd services/frontend
npm run dev
```
* Web Application: [http://localhost:3000](http://localhost:3000) (Proxies `/v1/*` requests to `http://127.0.0.1:8000`)

---

## Data Ingestion CLI (`weather-ingest`)

The ingestion engine is executed via the `weather-ingest` CLI or Python module (`python -m ingestion.cli`).

### Usage & Examples

#### 1. Ingest GFS Deterministic Forecast
Download, decode, store, and catalog GFS surface fields for a cycle:

```bash
cd services/ingestion
poetry run weather-ingest ingest \
    --model gfs \
    --cycle-date 2026-08-31 \
    --cycle-hour 0 \
    --lead-time-hours 0 3 6 9 12 18 24 \
    --concurrency 4
```

#### 2. Ingest GEFS Ensemble Forecast (Subset of Members or Leads)
Ingest perturbation members for GEFS:

```bash
cd services/ingestion
poetry run weather-ingest ingest \
    --model gefs \
    --cycle-date 2026-08-31 \
    --cycle-hour 0 \
    --lead-time-hours 0 3 6 \
    --member 1 2 3 4 5 \
    --concurrency 4
```

*(Omitting `--member` for GEFS defaults to ingesting the complete 30-member ensemble `gep01`–`gep30`).*

#### 3. Dry-Run Verification
Inspect resolved run specifications and canonical Zarr store destinations without downloading data:

```bash
poetry run weather-ingest ingest \
    --model gfs \
    --cycle-date 2026-08-31 \
    --cycle-hour 0 \
    --lead-time-hours 6 12 \
    --dry-run
```

#### 4. Batch Manifest Ingestion
Execute complex multi-run ingestion workflows using a JSON manifest file:

```bash
poetry run weather-ingest ingest --manifest ingest-manifest.json
```

*Example `ingest-manifest.json`:*
```json
{
  "runs": [
    {
      "model": "gfs",
      "cycle_date": "2026-08-31",
      "cycle_hour": 0,
      "lead_time_hours": [0, 3, 6, 9, 12]
    },
    {
      "model": "gefs",
      "cycle_date": "2026-08-31",
      "cycle_hour": 0,
      "lead_time_hours": [0, 6, 12],
      "members": [1, 2, 3]
    }
  ]
}
```

### CLI Options Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | `choice(gfs, gefs)` | *(Required)* | Weather model identifier(s) to ingest. |
| `--cycle-date` | `ISO Date (YYYY-MM-DD)` | *(Required)* | UTC model initialization date(s). |
| `--cycle-hour` | `choice(0, 6, 12, 18)` | *(Required)* | UTC cycle initialization hour(s). |
| `--lead-time-hours` | `int [int ...]` | *(Required)* | Forecast lead time offsets in hours (0–384). |
| `--member` | `int [int ...]` | `1..30` (GEFS) | GEFS perturbation member identities (1–30). Ignored for GFS. |
| `--manifest` | `path` | `None` | Path to JSON manifest for explicit batch jobs. |
| `--dry-run` | `flag` | `False` | Print resolved run specs without downloading or writing. |
| `--concurrency` | `int` | `4` | Maximum concurrent forecast files fetched and processed per run. |
| `--download-dir` | `path` | `downloads` | Staging directory for temporary GRIB2 downloads. |
| `--keep-downloads` | `flag` | `False` | Retain downloaded `.grib2` and `.idx` files after ingestion. |
| `--no-progress` | `flag` | `False` | Disable live terminal UI and output plain log lines. |
| `--store` | `str` | `None` (derived) | Custom Zarr store destination URL (e.g. `s3://...`). |
| `--allow-custom-store` | `flag` | `False` | Permit non-canonical Zarr store paths. |

---

## Testing & Quality Gates

The platform enforces strict quality, linting, typechecking, and test coverage standards across Windows and Linux CI environments.

### 1. Python Quality & Formatting (Ruff & MyPy)

```bash
# Ruff linting
(cd packages/domain && ruff check .)
(cd services/api && ruff check .)
(cd services/ingestion && ruff check .)

# MyPy strict type checking
(cd packages/domain && mypy)
(cd services/api && mypy)
(cd services/ingestion && mypy)
```

### 2. Domain Unit Tests (100% Coverage Gate)

Domain calculation logic must pass with zero external dependencies and 100% code coverage:

```bash
cd packages/domain
poetry run pytest
```

### 3. API Integration Tests (PostgreSQL + Redis)

```bash
cd services/api
poetry run pytest
```

### 4. Ingestion Integration Tests (PostgreSQL + Redis + MinIO S3)

```bash
cd services/ingestion
poetry run pytest
```

*(To enable real MinIO S3 Zarr round-trip tests, ensure MinIO is running and set `WEATHER_TEST_MINIO=1`).*

### 5. Frontend Unit & Type Checking

```bash
cd services/frontend

# Jest unit tests
npm run test

# TypeScript typecheck
npm run typecheck

# ESLint & Prettier format checks
npm run lint
npm run format:check

# Production standalone build validation
npm run build
```

### 6. Frontend End-to-End Tests (Playwright)

```bash
cd services/frontend
npx playwright install --with-deps chromium
npm run e2e
```

---

## Production Container Builds

Multi-stage Dockerfiles are provided for building self-contained, non-root Linux production images:

```bash
# 1. API Production Image
docker build -f docker/Dockerfile.api -t weather-api:latest .

# 2. Ingestion Worker Image
docker build -f docker/Dockerfile.ingestion -t weather-ingestion:latest .

# 3. Frontend Standalone Image
docker build -f docker/Dockerfile.frontend --build-arg API_PROXY_TARGET=http://127.0.0.1:8000 -t weather-frontend:latest services/frontend
```

---

## Configuration & Environment Variables

Environment variables are configured via `.env` (see `.env.example`). Key configuration parameters include:

### Database & Caching
* `DATABASE_URL`: PostgreSQL connection string (`postgresql://weather_user:weather_password@localhost:5432/weather_db`).
* `REDIS_URL`: Redis connection string (`redis://localhost:6379/0`).

### Object Storage (MinIO / S3)
* `MINIO_ENDPOINT`: S3 endpoint host and port (`localhost:9000`).
* `MINIO_ACCESS_KEY`: S3 access key ID (`minio_admin`).
* `MINIO_SECRET_KEY`: S3 secret access key (`minio_password`).
* `MINIO_BUCKET_NAME`: Target S3 bucket name (`weather-data`).
* `MINIO_SECURE`: `false` for HTTP, `true` for HTTPS.

### Ingestion Engine
* `NOAA_DOWNLOAD_SOURCE`: Primary upstream provider (`aws_s3` or `nomads`; default `aws_s3`).
* `ENABLE_NOMADS_FALLBACK`: Enable automatic fallback to NOMADS on publication lag (`true`).
* `ENABLE_SELECTIVE_DOWNLOAD`: Enable `.idx` byte-range selective download (`true`).
* `MAX_DOWNLOAD_CONCURRENCY`: Network I/O concurrency ceiling (default `24`).
* `MAX_DECODE_CONCURRENCY`: ProcessPool decode concurrency ceiling (default `8`).
* `MAX_WRITE_CONCURRENCY`: DB and storage write concurrency ceiling (default `6`).

### Search & Geocoding
* `SEARCH_PROVIDER`: Autocomplete provider (`google` or `mapbox`; default `google`).
* `GOOGLE_PLACES_API_KEY`: API key for Google Places API (New).

### Topography & Elevation
* `ELEVATION_PROVIDER`: Elevation lookup source (`dem`, `google`, or `none`; default `dem`).
* `DEM_DATA_PATH`: Path to Copernicus 30m DEM Zarr/NetCDF dataset.

---

## Documentation Index

For detailed architectural specifications and design documents, refer to:

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Detailed system architecture, component breakdown, and data pipeline.
* [docs/API.md](docs/API.md) — REST API endpoints, request/response models, and error envelopes.
* [docs/DATABASE.md](docs/DATABASE.md) — PostgreSQL + PostGIS schema, table definitions, and spatial indexes.
* [docs/MODELS.md](docs/MODELS.md) — Meteorological model parameters, vertical levels, and grid geometries.
* [docs/TESTING.md](docs/TESTING.md) — Testing strategy, test fixtures, and coverage requirements.
* [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Infrastructure deployment, Docker Compose, and container topologies.
* [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — Development workflow, code style, and pull request procedures.
* [ENGINEERING_CONTRACT.md](ENGINEERING_CONTRACT.md) — Core engineering rules and quality gates.
* [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Master implementation roadmap and completed milestones.

---

## License

Proprietary & Confidential. All rights reserved.
