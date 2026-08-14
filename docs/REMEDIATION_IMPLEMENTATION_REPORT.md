# M1–14 Remediation Implementation Report

**Date:** 2026-08-13
**Status:** Implementation complete, uncommitted in the working tree.
**Head:** `164c020 feat(explorer): make the forecast explorer fully database-driven`

---

## A. Implementation Summary

### Issue 2 — Forecast run / cycle identity (highest priority)

The confirmed defect — `_merge_lead` attempting to merge a dataset from one cycle
into a Zarr store of another, surfacing `xarray.MergeError: conflicting values
for variable 'time'` — is fixed with **fail-fast cycle validation** (never
`compat="override"`):

- `ingestion/core/base.py`: new domain errors `CycleStoreMismatchError` and
  `StoreSchemaMismatchError`.
- `ingestion/providers/noaa/parser.py`: the parser now records `cycle_time` (the
  GRIB reference time) as a dataset attribute, so the store is self-describing.
- `ingestion/core/pipeline.py`: `_merge_lead` validates the incoming dataset's
  cycle against the existing store's cycle (`_validate_store_identity`) and the
  structural schema (`_validate_lead_schema`) **before** merging. A mismatch
  raises `CycleStoreMismatchError` with both cycles and the refusal; the store
  is untouched.
- `ingestion/core/zarr_writer.py`: `write_dataset_atomic` stages the write to a
  sibling and swaps it into place so a partially-written store is never served.
- `ingestion/cli.py`: `derive_store_path` is the single source of truth for the
  store layout; `validate_store_path` derives the path when `--store` is omitted
  and rejects a supplied path that contradicts `model/cycle-date/cycle-hour`
  (unless `--allow-custom-store`).

**The exact previously-failing case is verified:** same-cycle leads `[0,6,12,18]`
merge into one 00Z store; a 12Z file merged into that store is refused with
`CycleStoreMismatchError`.

### Issue 2B — Batch / multi-run ingestion

- CLI `--model`/`--cycle-date`/`--cycle-hour`/`--lead-time-hours` accept multiple
  values with **aligned-zipped expansion** (never a Cartesian product); a single
  model/date/hour with many leads is "all leads of one cycle".
- `--manifest` supplies an explicit run list; `--dry-run` prints resolved specs;
  `--max-runs` guards accidental huge jobs.
- Per-run failure isolation: a failing run reports and the process exits non-zero
  without aborting successful runs.
- Preserved the existing single-lead invocation (backward compatible; `--store`
  still accepted with `--allow-custom-store` for local/custom layouts).

### Issue 1 — Raster performance

- **Vectorized** the per-pixel color loop (was ~175 ms/tile) and longitude
  alignment (was ~22 ms/tile via `np.vectorize`).
- Added a **server-side tile LRU cache** keyed by the full forecast identity
  `(model, variable, level, zoom, x, y, lead, initial_time)`, so identical tile
  requests are served from memory and distinct cycles/leads never share entries.
- Fixed a latent unbounded loop in the original `_native_lon` for small native
  regions (a pixel longitude far outside a small region oscillated forever); the
  new `_align_longitudes` is bounded and masks out-of-region pixels as no-data.
- **Measured:** cached tile ~0.001 ms vs ~230–318 ms cold; new cold render is
  byte-identical to the reference on both variables.

### Issue 3 — Location autocomplete

- New `api/services/places.py`: `PlaceAutocompleteProvider` interface +
  `GooglePlacesAutocompleteProvider` (Places API (New), proxied server-side,
  session-token aware, `includedPrimaryTypes` to avoid Text-Search billing) +
  `MapboxGeocodingProvider` alternative. Uses stdlib HTTP (no new runtime dep).
- `/v1/search?type=place` returns ranked suggestions (`object: "place"` with a
  `place_id`, no coordinates yet); `GET /v1/search/places/{place_id}` resolves
  the canonical place (name, lat/lon, country, region).
- Frontend: `useSearch` uses `type=place` + a per-session token; `LocationSearch`
  resolves the selected place before updating the map/forecast. Debounce/abort/
  stale-guard/keyboard nav preserved.

### Issue 4 — Elevation

- New `api/services/elevation.py`: `ElevationProvider` interface +
  `DEMElevationProvider` (local/server-side DEM read via xarray, bilinear
  interpolation in NumPy — no scipy dependency) + `GoogleElevationProvider`
  (alternative, not the default) + `RoundedElevationCache` (3-decimal rounding).
- `resolve_location` populates `elevation_m` for raw coordinates and cities via
  the provider; no-data/ocean/unconfigured → `None` (never `0`, never a guess).
- Frontend `SelectedLocationSummary` renders "unavailable" instead of `—`.

### Issue 5 — Ensemble capability

- `ForecastDashboard` renders the ensemble section **only** when the selected
  model is ensemble-capable (`is_ensemble`); a deterministic model (GFS) shows
  no misleading "Ensemble Statistics (GFS)" panel. An ensemble model with no
  data shows a distinct "Ensemble data is not yet available for this forecast."

### Cross-cutting (API/cache identity)

- All three cache-key builders (`point`, `probability`, `ensemble`) now include
  the resolved run's `cycle_time` — a cache entry for GFS 00Z never satisfies a
  GFS 12Z request even at the same lead/location.
- `/v1/ensembles` and `/v1/probabilities` accept an optional `initial_time` to
  pin the forecast run (GAP-2).
- Run id is version-scoped (GAP-4): two versions of the same model at the same
  cycle get distinct run ids.

---

## B. Forecast Identity (final canonical model)

```
forecast run  = model + cycle_time          (model_runs.UNIQUE(model_version_id, cycle_time))
forecast lead = forecast run + lead_time_hours
valid time    = cycle_time + lead_time_hours
```

Flow through the system:

- **Ingestion CLI**: a run spec is `(model, cycle_date, cycle_hour, leads)`; the
  store path is derived from the identity and validated.
- **Zarr**: the store is self-describing (`model_id`, `cycle_time` attrs); the
  `time` coordinate is kept; `lead_time_hours` is the dimension.
- **S3**: `s3://weather-data/{model}/{cycle_date}/{cycle_hour}/cycle.zarr` (derived,
  single source of truth).
- **DB**: `model_runs` unique on `(model_version_id, cycle_time)`; run id is
  version-scoped; `forecast_products` unique on
  `(run_id, variable_id, grid_id, product_type, lead_time_hours)`.
- **API**: `/v1/points`, `/v1/ensembles`, `/v1/probabilities` resolve the newest
  ready run (optionally pinned via `initial_time` on ensembles/probabilities);
  `/v1/maps` pins via `initial_time`.
- **Cache**: all forecast-data cache keys include `cycle_time`.
- **Frontend**: selection carries `model, variable, initialTime, leadTimeHours`;
  `validTime = initialTime + leadTimeHours`.

---

## C. Files Changed

### Backend (API)
- `services/api/src/api/core/config.py` — new provider/DEM settings.
- `services/api/src/api/routers/ensembles.py` — `initial_time` param + cache cycle.
- `services/api/src/api/routers/points.py` — cache cycle.
- `services/api/src/api/routers/probabilities.py` — `initial_time` + cache cycle.
- `services/api/src/api/routers/search.py` — `type=place`, `session_token`, place resolve endpoint, 502 handling.
- `services/api/src/api/schemas.py` — `SearchResultOut.place_id`.
- `services/api/src/api/services/cache.py` — cycle in all cache keys.
- `services/api/src/api/services/ensemble_data.py` — `initial_time` threading.
- `services/api/src/api/services/point_forecast.py` — `resolve_latest_run_cycle_time`, `_parse_cycle_time`, elevation wiring, `_elevation_for`, `_resolve_ready_dataset(initial_time)`.
- `services/api/src/api/services/search.py` — place search/resolve.
- `services/api/src/api/services/tiles.py` — vectorized render, `_align_longitudes`, tile LRU cache.
- **New** `services/api/src/api/services/places.py` — place provider.
- **New** `services/api/src/api/services/elevation.py` — elevation provider + cache.

### Backend (Ingestion)
- `services/ingestion/src/ingestion/cli.py` — `derive_store_path`, `validate_store_path`, `RunSpec`, `expand_run_specs`, `--manifest/--dry-run/--max-runs`, batch loop, per-run failure reporting.
- `services/ingestion/src/ingestion/core/base.py` — `CycleStoreMismatchError`, `StoreSchemaMismatchError`.
- `services/ingestion/src/ingestion/core/catalog.py` — version-scoped run id.
- `services/ingestion/src/ingestion/core/pipeline.py` — `_validate_store_identity`, `_validate_lead_schema`, `_resolve_cycle_time`, atomic write.
- `services/ingestion/src/ingestion/core/zarr_writer.py` — `write_dataset_atomic`.
- `services/ingestion/src/ingestion/providers/noaa/parser.py` — `cycle_time` attr.

### Frontend
- `services/frontend/src/components/forecast/ForecastDashboard.tsx` — ensemble capability UI.
- `services/frontend/src/components/forecast/SelectedLocationSummary.tsx` — "unavailable".
- `services/frontend/src/components/search/LocationSearch.tsx` — place resolution on selection.
- `services/frontend/src/hooks/useSearch.ts` — `type=place`, session token, min length 2.
- `services/frontend/src/lib/api/client.ts` — `type=place`, `session_token`, `resolvePlace`.
- `services/frontend/src/lib/api/types.ts` — `place` object, `place_id`.
- `services/frontend/src/lib/forecast/selection.ts` — place → SelectedLocation.

### Database/migrations
- No schema migration required. Run-id format change is write-path only (existing rows keep old ids). `docs/DATABASE.md` documents the version-scoped run id.

### Configuration
- `.env.example` — Google Places + DEM env vars.

### Documentation
- `docs/API.md` — search place, ensembles/probabilities/maps `initial_time`, elevation note.
- `docs/DATABASE.md` — run-id version scoping.
- `docs/DEPLOYMENT.md` — Google Places + DEM operational prerequisites.
- `docs/TESTING.md` — provider-mock rules.
- `docs/ACCEPTANCE_REMEDIATION_PLAN.md` — marked as implemented.

### Tests
- `services/ingestion/tests/test_pipeline.py` — cycle validation, schema mismatch, self-describing store.
- `services/ingestion/tests/test_cli.py` — store derivation/validation, batch (multi-lead/cycle/model, manifest, dry-run, anti-Cartesian, max-runs, partial failure).
- `services/ingestion/tests/test_catalog.py` — version-scoped run ids.
- `services/api/tests/test_points.py` — cache-key cycle/model/lead separation.
- `services/api/tests/test_tiles.py` — tile cache identity (cycle/lead/tile) + reuse.
- `services/api/tests/test_search.py` — place search/resolve (mocked provider).
- **New** `services/api/tests/test_places.py` — provider unit tests (mocked transport).
- **New** `services/api/tests/test_elevation.py` — DEM/cache tests (mocked DEM).
- `services/frontend/src/components/search/__tests__/LocationSearch.test.tsx`, `hooks/__tests__/useSearch.test.ts`, `components/forecast/__tests__/ForecastDashboard.test.tsx` — updated for new behavior.

---

## D. Behavior Changes

- **Ingestion CLI**: store path is derived unless `--store --allow-custom-store`;
  batch (multi-lead/cycle/model) supported; a cross-cycle merge now raises a
  clear domain error (fail-fast) instead of a raw `MergeError`.
- **API**: cache keys now include the forecast-run cycle (no cross-cycle cache
  leak); `/v1/ensembles` and `/v1/probabilities` accept optional `initial_time`;
  `/v1/search` supports `type=place` + place resolution; `elevation_m` is
  populated for coordinates/cities; `/v1/search/places/{place_id}` is a new
  endpoint.
- **Frontend**: search uses place autocomplete (min length 2, session token);
  elevation shows "unavailable" (not `—`); deterministic models show no
  ensemble panel; ensemble-model-no-data shows a distinct message.
- **Raster**: tiles are cached server-side (identity-correct key); rendering is
  vectorized (same output, much faster).

---

## E. Validation

| Check | Result | Command |
|---|---|---|
| Ingestion suite | **PASS** (85 passed, 4 skipped) | `cd services/ingestion && poetry run pytest tests/` |
| API offline tests (places/elevation/tiles/png/imports/points cache-keys) | **PASS** (41 passed, 27 skipped) | `cd services/api && pytest tests/test_places.py tests/test_elevation.py tests/test_tiles.py tests/test_png.py tests/test_imports.py tests/test_points.py` |
| API full suite | 52 passed, 111 skipped, **5 FAIL** (Redis-unavailable, pre-existing environment) | `cd services/api && pytest tests/` |
| Domain suite | **PASS** (296 passed, 100% coverage) | `cd packages/domain && poetry run pytest` |
| Frontend suite | **PASS** (109 passed across 19 suites) | `cd services/frontend && npm run test` |
| Frontend typecheck | **PASS** | `cd services/frontend && npx tsc --noEmit` |
| Frontend prettier | **PASS** | `cd services/frontend && npx prettier --check` |
| ruff (ingestion + api) | **PASS** | `python -m ruff check services/ingestion/src services/ingestion/tests services/api/src services/api/tests` |
| mypy (ingestion) | **PASS** | `cd services/ingestion && poetry run mypy` |
| mypy (api) | **PASS** | `cd services/api && poetry run mypy` |
| Cross-cycle regression (exact failing case) | **PASS** | Direct `_merge_lead` smoke: same-cycle leads [0,6,12,18] merge; 12Z into 00Z store refused |
| Raster differential + timing | **PASS** | old == new byte-identical; cached tile ~0.001 ms |

**Note on the 5 API FAILs:** `test_points.py` cache tests require a **live Redis**
(the `cache` fixture binds `PointCache()` to `redis://localhost:6379/0`). Redis is
not running in this environment, so they fail with `redis_read_unavailable`. These
are pre-existing (they fail identically without my changes) and are an environment
condition, not a code regression. The `test_cache_key_distinguishes_*` tests I
added are pure functions and pass.

---

## F. Performance

Measured on a real GFS-scale store and the test fixture:

| Scenario | Before | After |
|---|---|---|
| Tile render (cold, no cache) | ~230–318 ms | ~11 ms (byte-identical output) |
| Tile render (server cache hit) | n/a (no cache) | **~0.001 ms** |
| 200 cached renders | n/a | 0.001 ms/tile |

The 5 `test_points.py` cache tests that require Redis are the only backend
failures and are environment-gated (no Redis running).

---

## G. External Services

- **Google Places API (New)**: enable the Places API in the Google Cloud project;
  create a server API key (IP + API restricted). Configured via
  `GOOGLE_PLACES_API_KEY` (server-side; never in the browser). `SEARCH_PROVIDER`
  selects `google` (default) or `mapbox`. **Pricing figures from the plan should
  be re-verified against the live pricing page before committing to volume.**
- **DEM**: `DEM_DATA_PATH` points at a global xarray-readable DEM (Zarr/NetCDF)
  with `latitude`/`longitude` coords and an `elevation` variable in meters.
  Recommended source: Copernicus DEM GLO-30 (public COGs) prepared into the
  platform's storage. `ELEVATION_PROVIDER` selects `dem` (default), `google`,
  or `none`. The exact bucket/URL should be verified against the current AWS/
  Planetary Computer distribution at deployment time.
- **Environment variables**: see `.env.example` (all optional with defaults).

---

## H. Compatibility

- **Existing S3 stores**: compatible — the target layout is the same convention
  (`s3://weather-data/{model}/{date}/{hour}/cycle.zarr`); the change is derived +
  validated rather than manual.
- **Zarr datasets**: existing stores lack a `cycle_time` attr; the merge guard
  uses the `time` coordinate as a fallback, so legacy stores remain validable.
  New writes carry the attr.
- **DB records**: run-id format changes on the write path only (existing rows keep
  old ids). No migration required. If a legacy store already contains mixed-cycle
  data (the old bug), the new guard fails fast — the operator re-ingests that cycle.
- **API**: no breaking changes to `/v1/`. All additions are optional params
  (`type=place`, `session_token`, `initial_time` on ensembles/probabilities,
  `/v1/search/places/{id}`) or additive fields (`place_id`, populated
  `elevation_m`).
- **Frontend**: `SearchResult.place_id` is optional; `MIN_QUERY_LENGTH` 1→2;
  elevation "unavailable" text; ensemble heading behavior. Types updated
  accordingly.
- **Configuration**: new env vars (all optional with defaults).

---

## I. Known Limitations

- **Google Places / DEM live verification**: the plan's pricing figures and the
  exact DEM bucket URL were researched under web rate-limits and are flagged as
  "verify at deployment." The provider abstraction means swapping the backend is
  low-cost.
- **DEM interpolation precision**: bilinear at the DEM's native resolution; a
  30 m/90 m source yields city-level accuracy (not survey-grade). No-data/ocean
  cells return `None` (rendered `unavailable`).
- **Tile cache is process-local LRU** (not Redis); a multi-worker deployment
  would benefit from a shared Redis tile cache (documented as a future option).
- **Run-id format change** means new runs have a different id than pre-change
  runs; existing rows are unaffected.
- **`_resolve_ready_dataset` still re-opens the Zarr store per request** for
  point/ensemble/probability (tile path now uses the cache). A kept-open dataset
  cache was noted in the plan but not implemented (lower priority than the tile
  cache; the tile path is the interactive bottleneck).

---

## J. Git State

- **No commit created.** **No push performed.**
- All changes remain **uncommitted** in the working tree (`git status --short`
  shows the modified/untracked files listed in Section C).
- Untracked files are the deliverables: `docs/ACCEPTANCE_REMEDIATION_PLAN.md`,
  `services/api/src/api/services/{places,elevation}.py`,
  `services/api/tests/test_{places,elevation}.py`. Test-residue (`downloads/`)
  was removed.
