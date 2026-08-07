# API Test Fixtures

This directory contains helpers for generating deterministic test data; it
does **not** contain committed binary fixtures.

## Zarr forecast stores

The point-forecast integration tests write a tiny, deterministic
`xarray.Dataset` (see `__init__.py`) to a **local on-disk Zarr store** at
test time via the API-local test writer `tests._zarr_writer.write_dataset`
(kept separate so the API service never imports the ingestion package).
Seeded `model_runs.zarr_store_path` rows point at these local
stores, so the full end-to-end slice + interpolate path runs against the
local test PostgreSQL **without** requiring MinIO/S3.

S3/MinIO coverage is optional and opt-in, gated by `WEATHER_TEST_MINIO=1`
(matching the ingestion service convention). It is not a dependency of the
default test suite.

## Location records

Search and point-resolution tests seed PostGIS `cities`, `ski_resorts`, and
`stations` rows with `POINT` geometries (`EPSG:4326`) whose coordinates lie
inside the fixture Zarr grid so point forecasts can be interpolated.
