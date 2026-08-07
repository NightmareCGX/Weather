# Testing Strategy & Guidelines

This document outlines the testing strategy, layers, and fixtures for the Global Probabilistic Weather Platform.

---

## 1. Testing Pillars

### 1.1 Unit Tests (`pytest` & `Jest`)
- **Location**: `packages/*/tests/`, `services/*/tests/`
- **Scope**: Pure functions, mathematical operations (ensemble statistics, percentile calculations, interpolation), and data parsers.
- **Rule**: Unit tests must run offline without requiring external network access, databases, or object storage.

### 1.2 Integration Tests (`pytest` + `Docker Compose`)
- **Location**: `services/api/tests/`, `services/ingestion/tests/`
- **Scope**: End-to-end flows involving FastAPI test clients, PostgreSQL + PostGIS spatial queries, Zarr dataset reads/writes, and the ingestion→catalog→serving pipeline (`services/ingestion/tests/test_catalog_postgres.py` writes a run to the catalog and serves it through `/v1/points`, including a test that drives the real `weather-ingest` CLI production entrypoint with a mocked download).
- **Environment**: Executed against local containers (`postgres`, `redis`, `minio`) via Docker Compose. The API and ingestion suites **skip** gracefully when their backing services are unreachable; the S3/MinIO Zarr round-trip is additionally gated behind `WEATHER_TEST_MINIO=1`.

### 1.3 Contract Tests
- **Scope**: Validates that all FastAPI response payloads adhere strictly to the response envelope and resource schemas defined in `docs/API.md` (the API service owns its response schemas in `api/schemas.py`; the `packages/contracts` package is currently a placeholder).
- **Mechanism**: Automated assertions in `services/api/tests/` comparing serialized API responses against the documented `docs/API.md` shapes.

---

## 2. Fixtures & Mocking Strategy

- **Upstream Network Mocking**: All NOAA NOMADS / S3 HTTP requests in ingestion tests are mocked using `respx` / `httpx` transport mocks to isolate tests from upstream API outages.
- **Sample GRIB2 Fixtures**: Small, anonymized GRIB2 subset files are stored in `services/ingestion/tests/fixtures/` to test `cfgrib` decoding without downloading multi-gigabyte operational files.
- **Temporary Zarr Stores**: Storage tests utilize `pytest`'s built-in `tmp_path` fixture to create isolated local Zarr hierarchies.
