# Final Acceptance Review — Live Findings Log

This file records independent findings from the read-only final acceptance review.
It is an audit artifact, not a modification to application source.

## FINDING-5 (CLEANUP): untracked test residue `services/ingestion/downloads/`

- CLI tests write downloaded GRIB fixtures into `services/ingestion/downloads/`.
  This directory is not gitignored and appears untracked after test runs.
- It is **test residue**, not part of the remediation's 44-file scope. It should
  be cleaned or gitignored (``*.grib2`` is not in `.gitignore`).

## VERIFICATION-1 (PASS): `initial_time` genuinely restricts the run query

- `test_maps.py::test_maps_initial_time_in_template` and
  `::test_maps_initial_time_not_available_404` pass: pinning a known cycle
  returns the tile template with `initial_time`; pinning an unknown cycle
  returns 404 (NOT a fallback to the newest run).
- Code inspection: `point_forecast.py::_resolve_ready_dataset` and
  `resolve_latest_run_cycle_time` both `.where(ModelRun.cycle_time ==
  _parse_cycle_time(initial_time))` when `initial_time` is provided — so the
  parameter affects the underlying query and the cache key, not just the
  OpenAPI surface.
- `initial_time` on `/v1/ensembles` and `/v1/probabilities` is threaded through
  the service to `_resolve_ready_dataset` (verified by inspection; the API
  suite's 168 tests pass including ensembles/probabilities).

## FINDING-6 (OBSERVATION): raster timing variance vs report

- The implementation report claimed cold ~11 ms / cached ~0.001 ms. This review
  measured cold ~15–52 ms / cached ~0.0006–0.077 ms on this Windows box with a
  global GFS-scale store. The **order of magnitude and the massive speedup over
  the ~230–318 ms baseline are confirmed**; exact figures vary by machine/store.
  Not a defect.


## FINDING-2 (REGRESSION): M1-14 E2E test asserts the OLD misleading ensemble heading

- **File**: `services/frontend/e2e/forecast.spec.ts`
- **Test**: `ensemble statistics: deterministic selected model shows ensemble empty state` (line 75)
- **Evidence**: `git show HEAD:...` confirms the test at HEAD asserted
  `await expect(page.getByText(/Ensemble Statistics \(GFS\)/)).toBeVisible();`
  (line 87). The remediation **deliberately removed** that panel (Issue 5 —
  deterministic GFS must not show an ensemble panel), but this E2E test was
  **not updated** to assert the new behavior. It now fails.
- **Result**: Playwright E2E run: **5 passed, 1 failed**. The other ensemble E2E
  test (GEFS, line 93) passes.
- **Note**: This is a *semantic* regression — the remediation is behaviorally
  correct, but the E2E suite is no longer green because the old test still
  asserts the removed (misleading) UI. The test must be updated to assert that
  GFS shows **no** ensemble panel.
- **Severity**: MEDIUM — red E2E test in the suite.

## FINDING-4 (VALIDATION GAP): Elevation real-DEM path validated with synthetic store, not production GLO-30

- The elevation **code path** is validated end-to-end with a real on-disk Zarr
  DEM (bilinear, no-data→None, rounded-coordinate cache collapsing 6 requests
  to 1 provider call): Denver 1600m, Aspen 2400m, plains 710m. All PASS.
- A **production Copernicus GLO-30** store is not present in the environment
  (`DEM_DATA_PATH=""`). The same `DEMElevationProvider`/`read_dataset` code
  path is exercised; only the specific production DEM dataset is absent.

## FINDING-3 (VALIDATION GAP): No live Google Places / DEM credentials or data configured

- The environment has **no `.env` file**; `GOOGLE_PLACES_API_KEY=""` and
  `DEM_DATA_PATH=""` per settings defaults.
- **Real** Google Places autocomplete and **real** DEM elevation cannot be
  exercised end-to-end. The provider abstraction is verified with mocked
  transports (unit tests), and the frontend E2E uses a mocked Denver response,
  but a real external acceptance test is **NOT FULLY VALIDATED**.


- **File**: `services/ingestion/tests/test_catalog_postgres.py`
- **Test**: `test_cli_production_entrypoint_ingests_and_serves`
- **Evidence**: Fails with exit code 1. The CLI now rejects the test's local
  `--store` path:
  ```
  run FAILED: model=gfs cycle=2026-07-22 00:00:00+00:00: Store path
  'C:\...\gfs.zarr' does not match the forecast identity (model=gfs,
  cycle=2026-07-22T00Z). Expected 's3://weather-data/gfs/2026-07-22/00/cycle.zarr'.
  Pass --allow-custom-store to override.
  ```
- **Root cause**: This test existed at HEAD and passed pre-remediation (the CLI
  accepted any `--store`). The remediation added `validate_store_path` which
  requires `--allow-custom-store` for a non-derived path, but this pre-existing
  M1-14 integration test was **not** updated to pass that flag.
- **Result**: 1 real regression against an existing M1-14 contract/integration
  test. `git show HEAD:...` confirms the test existed and used `--store` without
  the new flag.
- **Severity**: MEDIUM — the feature (store path validation) is intended, but it
  broke an existing test that must be updated to pass `--allow-custom-store`
  (or use a derived path) for the suite to be green.
