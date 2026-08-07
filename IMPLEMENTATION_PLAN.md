# Master Implementation Plan (IMPLEMENTATION_PLAN.md)

This document is the master execution roadmap for the Global Probabilistic Weather Forecasting Platform. Implementation proceeds strictly one milestone at a time, following vertical slices from initial project bootstrap through commercial readiness and multi-model expansion.

---

## Milestone 1: Project Bootstrap & Workspace Setup
- **Goal**: Establish the monorepo structure, Python workspace configuration, dependency management, and local developer environment.
- **Scope**: Root workspace files, `pyproject.toml`, shared package directories (`packages/domain`, `packages/contracts`, `packages/config`), and basic service stubs.
- **Out of Scope**: Database migrations, API routers, ingestion workers, and frontend code.
- **Deliverables**: 
  - Monorepo directory structure matching REPOSITORY.md.
  - Root `pyproject.toml` with Poetry workspace configuration.
  - Initial stub packages and modules.
- **Dependencies**: None.
- **Acceptance Criteria**: `poetry lock` and `poetry install` succeed cleanly without resolution errors across all packages.
- **Testing Requirements**: Workspace configuration dry-run test.
- **Estimated Complexity**: Low

---

## Milestone 2: Development Environment & Infrastructure
- **Goal**: Spin up local containerized infrastructure (PostgreSQL 16 + PostGIS, Redis, MinIO S3 object storage) via Docker Compose.
- **Scope**: `docker-compose.yml`, health checks, persistent volume mounts, and environment configuration templates (`.env.example`).
- **Out of Scope**: Kubernetes manifests and Terraform cloud deployment scripts.
- **Deliverables**:
  - `docker-compose.yml` configured with `weather_postgres`, `weather_redis`, and `weather_minio`.
  - `.env.example` file defining all required environment variables.
- **Dependencies**: Milestone 1.
- **Acceptance Criteria**: `docker compose up -d` starts all services successfully and health checks pass for PostgreSQL, Redis, and MinIO.
- **Testing Requirements**: Container connectivity smoke tests.
- **Estimated Complexity**: Low

---

## Milestone 3: Database Migrations & Schema Initialization
- **Goal**: Establish relational persistence by applying the frozen database schema via Alembic migrations.
- **Scope**: SQLAlchemy base configuration, Alembic migration setup, and initial migration script for the 13 database tables defined in `docs/DATABASE.md` (the 14-table relationship map in DATABASE.md is a high-level illustration; the implemented schema and its constraints are authoritative).
- **Out of Scope**: Application-level DB queries and seed data.
- **Deliverables**:
  - `services/api/src/api/core/database.py` (SQLAlchemy engine & session factory).
  - Alembic configuration (`alembic.ini` and `env.py`).
  - Initial migration script creating all tables, indexes, and GIST spatial constraints.
- **Dependencies**: Milestone 2.
- **Acceptance Criteria**: `alembic upgrade head` runs successfully against the local PostgreSQL instance, creating all tables and PostGIS spatial indexes.
- **Testing Requirements**: Migration upgrade and downgrade smoke tests.
- **Estimated Complexity**: Medium

---

## Milestone 4: NOAA Ingestion Connector (GFS & GEFS)
- **Goal**: Implement the provider-agnostic ingestion client for downloading operational GRIB2 files from NOAA NOMADS / S3.
- **Scope**: `services/ingestion/src/providers/noaa/` adapter, network request handling, and upstream mocking via `respx`.
- **Out of Scope**: GRIB2 parsing and Zarr writing.
- **Deliverables**:
  - `services/ingestion/src/ingestion/core/base.py` (Base model connector interface).
  - `services/ingestion/src/providers/noaa/connector.py` (NOAA NOMADS client).
  - Unit tests with mocked HTTP responses.
- **Dependencies**: Milestone 1.
- **Acceptance Criteria**: Ingestion client successfully queries NOAA catalog endpoints and retrieves target GRIB2 file URLs.
- **Testing Requirements**: Unit tests using `respx` to mock NOAA HTTP endpoints.
- **Estimated Complexity**: Medium

---

## Milestone 5: GRIB2 Parsing & Zarr Storage Layer
- **Goal**: Parse raw GRIB2 binary data using `cfgrib` and `xarray`, normalizing coordinates and writing compressed chunks to Zarr object storage.
- **Scope**: `services/ingestion/src/providers/noaa/parser.py`, NetCDF/Zarr translation, and MinIO/S3 object storage persistence.
- **Out of Scope**: FastAPI integration and AI downscaling.
- **Deliverables**:
  - GRIB2 decoder using `cfgrib` and `xarray`.
  - Zarr writer persisting chunked datasets to MinIO.
  - Integration tests using sample GRIB2 fixture files.
- **Dependencies**: Milestone 2, Milestone 4.
- **Acceptance Criteria**: Sample GRIB2 files are correctly decoded into `xarray` datasets and successfully written to and read from MinIO Zarr stores.
- **Testing Requirements**: Integration tests verifying fixture decoding and Zarr round-trip persistence.
- **Estimated Complexity**: High

---

## Milestone 6: Internal Forecast Domain Model
- **Goal**: Implement core meteorological business logic, coordinate transformation helpers, and forecast data structures in `packages/domain`.
- **Scope**: `packages/domain/src/domain/models/`, `packages/domain/src/domain/geo/`, and spatial interpolation helpers.
- **Out of Scope**: HTTP handlers and database queries.
- **Deliverables**:
  - Domain dataclasses for forecast points (`ForecastPoint`, `GridPoint`) and regular grid definitions (`RegularGrid`).
  - Spatial coordinate mapping utilities (`domain/geo/coordinates.py`, `domain/geo/grid.py`).
  - Bilinear interpolation helper (`domain/geo/interpolation.py`).
  - Comprehensive unit tests for domain math.
  - *Note (approved scope refinement): the `ForecastVariable` catalog vocabulary is deferred and may be introduced later.*
- **Dependencies**: Milestone 1.
- **Acceptance Criteria**: All unit tests in `packages/domain` pass with zero external dependencies.
- **Testing Requirements**: 100% unit test coverage for domain calculation modules.
- **Estimated Complexity**: Medium

---

## Milestone 7: Probability & Ensemble Calculation Engine
- **Goal**: Implement statistical post-processing, ensemble spread calculators, mean, median, percentiles (P10, P50, P90), and exceedance probability math in `packages/domain`.
- **Scope**: `packages/domain/src/ensemble/` calculation modules.
- **Out of Scope**: API routing and database persistence.
- **Deliverables**:
  - Ensemble statistical functions (mean, spread, percentiles).
  - Threshold exceedance probability calculators.
  - Unit tests using synthetic ensemble member arrays.
- **Dependencies**: Milestone 6.
- **Acceptance Criteria**: Ensemble spread and percentile calculations return mathematically verified outputs against known test arrays.
- **Testing Requirements**: Numerical unit tests using pytest.
- **Estimated Complexity**: Medium

---

## Milestone 8: FastAPI Core & Catalog Endpoints
- **Goal**: Establish the thin FastAPI application and implement all Catalog domain endpoints (`/v1/centers`, `/v1/models`, `/v1/runs`, `/v1/variables`, `/v1/grids`).
- **Scope**: `services/api/src/main.py`, `services/api/src/routers/catalog.py`, dependencies (`deps.py`), and error handling middleware.
- **Out of Scope**: Complex spatial point interpolations.
- **Deliverables**:
  - FastAPI application entrypoint with RFC 7807 error handling and response envelopes.
  - Catalog routers querying PostgreSQL metadata.
  - Contract and integration tests.
- **Dependencies**: Milestone 3, Milestone 6.
- **Acceptance Criteria**: `GET /v1/models`, `GET /v1/centers`, etc., return correctly structured JSON responses matching `docs/API.md`.
- **Testing Requirements**: FastAPI `TestClient` integration tests against PostgreSQL test DB.
- **Estimated Complexity**: Medium

---

## Milestone 9: Point Forecast & Search Endpoints
- **Goal**: Implement `/v1/points` and `/v1/search` endpoints backed by PostGIS spatial indexing and Zarr dataset slicing.
- **Scope**: `services/api/src/routers/points.py`, `services/api/src/routers/search.py`, PostGIS spatial query logic, and Redis-primary cache layer (`point_query_fallback_audit`).
- **Out of Scope**: Probability and map tile generation.
- **Deliverables**:
  - Spatial geocoding search router (`/v1/search`).
  - Point forecast router (`/v1/points`) querying Zarr stores and PostGIS.
  - Redis caching and PostgreSQL fallback implementation.
- **Dependencies**: Milestone 5, Milestone 8.
- **Acceptance Criteria**: `GET /v1/points` successfully resolves coordinates, cities, or ski resorts, slices Zarr data, and returns hourly forecasts adhering to `docs/API.md`.
- **Testing Requirements**: End-to-end integration tests combining PostGIS, MinIO Zarr stores, and FastAPI.
- **Estimated Complexity**: High

> **Milestone 9 approved scope notes**: As delivered, `/v1/points` serves a **single model** per request (default `gfs`); the multi-model response contract is not defined and multi-model requests are rejected with `422`. Address geocoding is **not implemented**; clients resolve locations through `/v1/search` first and then query `/v1/points` with coordinates or a platform id. See `docs/API.md` section 2.1.

---

## Milestone 10: Probability, Maps, and Ensemble Endpoints
- **Goal**: Implement remaining core API endpoints: `/v1/probabilities`, `/v1/maps`, and `/v1/ensembles`.
- **Scope**: `services/api/src/routers/probabilities.py`, `services/api/src/routers/maps.py`, and `services/api/src/routers/ensembles.py`.
- **Out of Scope**: Frontend UI and verification reporting.
- **Deliverables**:
  - Probability calculation endpoint.
  - Map tile metadata endpoint.
  - Ensemble statistics and spread endpoint.
  - Integration tests for all three routers.
- **Dependencies**: Milestone 7, Milestone 9.
- **Acceptance Criteria**: All probability, map tile, and ensemble endpoints return valid responses matching contract specifications.
- **Testing Requirements**: API contract and integration tests.
- **Estimated Complexity**: Medium

---

## Milestone 11: Verification & Administration Endpoints
- **Goal**: Implement `/v1/verifications` and `/v1/health` endpoints to complete the MVP API surface.
- **Scope**: `services/api/src/routers/verifications.py`, `services/api/src/routers/admin.py`.
- **Out of Scope**: Frontend UI.
- **Deliverables**:
  - Verification error metrics router.
  - Health check router verifying PostgreSQL, Redis, and MinIO connectivity.
- **Dependencies**: Milestone 8.
- **Acceptance Criteria**: `GET /v1/health` returns healthy status when all dependencies are running. `GET /v1/verifications` returns model verification metrics.
- **Testing Requirements**: Integration tests with mocked and live container dependencies.
- **Estimated Complexity**: Low

---

## Milestone 12: Next.js Frontend Foundation & Map Viewer
- **Goal**: Initialize the Next.js / React client application, configure Tailwind CSS, and implement MapLibre GL JS map rendering with meteorological overlays.
- **Scope**: `services/frontend/`, MapLibre canvas integration, API client library, and core layout.
- **Out of Scope**: Advanced charting and administrative dashboards.
- **Deliverables**:
  - Next.js application setup with TypeScript and Tailwind CSS.
  - MapLibre GL JS map component rendering base tiles and weather raster/vector layers.
  - API client communicating with FastAPI endpoints.
- **Dependencies**: Milestone 10.
- **Acceptance Criteria**: Frontend starts successfully, renders the interactive map, and successfully fetches model metadata and map layers from the backend API.
- **Testing Requirements**: Frontend component tests (Jest/React Testing Library).
- **Estimated Complexity**: High

---

## Milestone 13: Frontend Point Forecast Dashboard & Meteograms
- **Goal**: Implement user-facing point forecast inspection, interactive meteogram charts, ensemble spread boxplots, and location search UI.
- **Scope**: Location search bar, interactive point inspector panel, time-series meteogram charts, and probability distribution widgets.
- **Out of Scope**: Multi-model comparison advanced tooling.
- **Deliverables**:
  - Search autocomplete component (`/v1/search`).
  - Point forecast meteogram charts (`/v1/points`).
  - Ensemble spread visualization.
- **Dependencies**: Milestone 12.
- **Acceptance Criteria**: Users can search for a city or ski resort, click any map point, and view rich hourly forecast meteograms and ensemble spreads.
- **Testing Requirements**: End-to-end UI tests (Playwright).
- **Estimated Complexity**: High

---

## Milestone 14: End-to-End Integration & CI/CD Pipelines
- **Goal**: Implement comprehensive GitHub Actions CI workflows for linting, type-checking, automated testing, and container builds across all packages and services.
- **Scope**: `.github/workflows/ci.yml`, Docker multi-stage builds (`docker/Dockerfile.*`).
- **Out of Scope**: Cloud Kubernetes deployments.
- **Deliverables**:
  - GitHub Actions workflow running Ruff, MyPy, Pytest, and Jest.
  - Multi-stage Dockerfiles for API, Ingestion worker, and Frontend.
- **Dependencies**: Milestone 1 through 13.
- **Acceptance Criteria**: GitHub Actions workflow passes successfully on push and pull requests, verifying code quality and test suites.
- **Testing Requirements**: CI pipeline execution verification.
- **Estimated Complexity**: Medium

---

## Milestone 15: Future Expansion - ECMWF & Canada Integration (Phase 2)
- **Goal**: Extend the ingestion and normalization pipeline to support ECMWF (IFS/AIFS) and ECCC Canada (GDPS/GEPS) models.
- **Scope**: `services/ingestion/src/providers/ecmwf/` and `services/ingestion/src/providers/canada/` connectors and normalization parsers.
- **Out of Scope**: AI downscaling.
- **Deliverables**:
  - ECMWF CDS/MARS ingestion adapter.
  - ECCC Datamart ingestion adapter.
  - Multi-model normalization tests.
- **Dependencies**: Milestone 5.
- **Acceptance Criteria**: ECMWF and Canadian model GRIB2 files are successfully ingested, decoded, and stored in Zarr format conforming to unified internal schemas.
- **Testing Requirements**: Provider-specific unit and integration tests.
- **Estimated Complexity**: High

---

## Milestone 16: Future Expansion - Multi-Model Ensemble (MME) Calibration (Phase 3)
- **Goal**: Implement Bayesian Model Averaging (BMA) and empirical bias correction to blend NOAA, ECMWF, and Canadian forecasts into a calibrated MME system.
- **Scope**: `services/processing/ensemble/` calibration engine and database calibration tables.
- **Out of Scope**: AI neural downscaling.
- **Deliverables**:
  - Bias correction (Quantile Mapping) pipeline.
  - Bayesian Model Averaging (BMA) weighting module.
  - MME forecast product generation tasks.
- **Dependencies**: Milestone 15.
- **Acceptance Criteria**: MME calibration jobs successfully blend multi-center inputs into a unified probabilistic forecast product.
- **Testing Requirements**: Numerical calibration verification tests.
- **Estimated Complexity**: High

---

## Milestone 17: Future Expansion - AI Downscaling Engine (Phase 4)
- **Goal**: Build the custom neural downscaling training pipeline and GPU inference worker to bridge coarse global models (~25km) down to high-resolution local grids (3km/1km).
- **Scope**: `services/processing/downscale/` (PyTorch conditional UNet, static DEM/LULC conditioning tensors, GPU inference workers).
- **Out of Scope**: Commercial subscription billing.
- **Deliverables**:
  - PyTorch neural downscaling model architecture & training pipeline.
  - Automated inference worker persisting 3km/1km Zarr tiles.
  - Frontend integration for high-res downscaled layers.
- **Dependencies**: Milestone 5, Milestone 16.
- **Acceptance Criteria**: Coarse global grids combined with high-res DEM tensors successfully pass through the neural inference worker, producing verified 3km/1km high-resolution Zarr outputs.
- **Testing Requirements**: Model inference smoke tests and spatial error metric evaluations.
- **Estimated Complexity**: High

---

## Milestone 18: Commercial Weather Platform Readiness (Phase 6)
- **Goal**: Harden the platform for commercial production scale, including API key management, OAuth2 authentication, rate limiting, and subscription billing.
- **Scope**: API security middleware, rate limiting (`redis`-backed), Stripe billing webhooks, and Prometheus/Grafana observability.
- **Out of Scope**: Further feature development.
- **Deliverables**:
  - API key authentication and tiered rate-limiting middleware.
  - Stripe subscription tier integration.
  - Prometheus metrics instrumentation and Grafana dashboards.
- **Dependencies**: Milestone 14.
- **Acceptance Criteria**: API enforces rate limits based on API keys, rejects unauthorized requests, and successfully processes billing tiers.
- **Testing Requirements**: Security and rate-limiting integration tests.
- **Estimated Complexity**: Medium

---

## Milestone Approval & Execution Rule

Per **`CLAUDE.md`**, implementation begins **strictly with Milestone 1**. After completing Milestone 1, execution must pause, and results must be presented for review before proceeding to Milestone 2.

Would you like to begin implementation of **Milestone 1: Project Bootstrap & Workspace Setup**?