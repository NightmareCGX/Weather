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
- **Scope**: End-to-end flows involving FastAPI test clients, PostgreSQL + PostGIS spatial queries, and Zarr dataset reads/writes.
- **Environment**: Automatically executed against local ephemeral containers (`postgres`, `redis`, `minio`) via Docker Compose.

### 1.3 Contract Tests
- **Scope**: Validates that all FastAPI response payloads adhere strictly to shared Pydantic schemas in `packages/types`.
- **Mechanism**: Automated assertions comparing API responses against OpenAPI specifications.

---

## 2. Fixtures & Mocking Strategy

- **Upstream Network Mocking**: All NOAA NOMADS / S3 HTTP requests in ingestion tests are mocked using `respx` / `httpx` transport mocks to isolate tests from upstream API outages.
- **Sample GRIB2 Fixtures**: Small, anonymized GRIB2 subset files are stored in `services/ingestion/tests/fixtures/` to test `cfgrib` decoding without downloading multi-gigabyte operational files.
- **Temporary Zarr Stores**: Storage tests utilize `pytest`'s built-in `tmp_path` fixture to create isolated local Zarr hierarchies.
