# M1–14 Second-Round Acceptance Remediation Plan

**Status:** PLANNED AND IMPLEMENTED (2026-08-13)
**Date:** 2026-08-13
**Scope:** Issues 1–5 + cross-cutting forecast-identity audit from the M1–14 second-round acceptance validation.

> **Implementation note (2026-08-13):** This plan has been fully implemented per the approved
> remediation. The read-only investigation that produced this document is complete; the
> implementation is uncommitted in the working tree (see the implementation report for the
> complete changed-file list and validation). Sections A–O below document the plan and the
> implemented behavior.

---

## A. Executive Summary

| # | Finding | Verdict | One-line explanation |
|---|---------|---------|----------------------|
| 1 | Weather raster rendering is slow | **CONFIRMED** | Per-tile requests re-open the Zarr store from object storage, re-resolve the run via SQL, and pay a ~0.2s per-pixel Python loop; there is **no server-side tile cache**. Measured ~230–318 ms/tile on a real GFS-scale store with no caching. |
| 2 | Ingestion fails when another forecast cycle is ingested | **CONFIRMED (reproduced)** | `_merge_lead` merges into the cycle store with **no cycle/identity guard**; a different-cycle file surfaces `xarray.MergeError: conflicting values for variable 'time'` — reproduced verbatim. The existing tests use `time`-less synthetic datasets so the path is untested. |
| 2B | Multi-model / multi-cycle / multi-lead ingestion | **DESIGN ENHANCEMENT** | The CLI is one-lead-per-invocation with a manually-supplied `--store`; no batch semantics, no store-path derivation/validation, no manifest. |
| 3 | Location autocomplete / typeahead | **PARTIALLY CONFIRMED** | The frontend `LocationSearch` already implements a debounced, abortable combobox, but the backend `/v1/search` is a PostGIS `ILIKE` over `cities`/`ski_resorts`/`stations` only. There is **no Google Places stack** in the repo; the product requires a true place-autocomplete provider. |
| 4 | Elevation resolved from coordinates | **PARTIALLY CONFIRMED** | `/v1/points` already returns `elevation_m` (nullable) and the frontend shows `—`; but raw-coordinate and city selections always resolve `elevation_m=None` (cities have no elevation column). No elevation provider exists. |
| 5 | Ensemble Statistics incorrectly identifies GFS as ensemble | **PARTIALLY CONFIRMED** | The backend already rejects `/v1/ensembles?model=gfs` with 422, and the dashboard already shows the empty-state for deterministic models — but the panel **heading still renders "Ensemble Statistics (GFS)"** for a deterministic model, which is the misleading text. |

### Section summary of the five findings

1. **Raster rendering (Issue 1)** — confirmed slow; the dominant costs are (a) re-opening the Zarr store per tile request (fresh `xr.open_zarr` against object storage), (b) re-resolving the run via SQL per request, and (c) a 65,536-iteration Python pixel loop (~0.17 s/tile) plus `np.vectorize` (~22 ms/tile). PNG encoding is fast (~2 ms). No server-side tile cache; `Cache-Control: public, max-age=300` only lets the *browser* cache.

2. **Ingestion cycle merge (Issue 2)** — confirmed and reproduced. `services/ingestion/src/ingestion/core/pipeline.py::_merge_lead` reads the existing store, keeps leads `!= new_lead`, and `xr.concat`s with `coords="minimal"`. The parser keeps `time` (the GRIB reference time) as a coordinate; when a different-cycle file is merged into a same-cycle store, `time` conflicts and `xr.concat` raises `MergeError`. The CLI never validates that `--store` matches the `--model/--cycle-date/--cycle-hour`.

3. **Batch ingestion (2B)** — design enhancement: current CLI is strictly one lead per invocation; no multi-model/cycle/lead semantics, no manifest, no store-path derivation or validation, no partial-failure/idempotency guarantees beyond the existing single-run upsert.

4. **Autocomplete (Issue 3)** — the frontend combobox is already well-built (debounce 300 ms, min length 1, abort, stale-guard, keyboard nav, empty/error/loading states). The gap is entirely the **data source**: PostGIS `ILIKE` over a tiny seeded table cannot provide global place autocomplete. There is no Google Maps/Places integration anywhere in the repo.

5. **Elevation (Issue 4)** — `elevation_m` is already part of the `/v1/points` contract (nullable) and the frontend already renders it (as `—` when null). The gap is the **source**: raw coordinates and cities always resolve `elevation_m=None`; no elevation provider or cache exists.

6. **Ensemble capability (Issue 5)** — the capability flag `is_ensemble` is already modeled in `models`, exposed by `/v1/models` and `/v1/forecast/availability`, enforced by `/v1/ensembles` (422 for non-ensemble), and consumed by the dashboard to show the empty-state for deterministic models. The remaining defect is cosmetic-but-misleading: the panel heading always prints `Ensemble Statistics (<model>)` even when the model is deterministic.

---

## B. Root Cause Analysis

### Issue 1 — Raster rendering slow

**Path:** browser → MapLibre raster tile → `/v1/maps/{model}/{variable}/{level}/{z}/{x}/{y}.png` → `render_tile_png` → `_resolve_run_and_field` (SQL + `xr.open_zarr`) → `_slice_field` (Zarr chunk read) → per-pixel color loop → `encode_rgba_png`.

Measured on a real GFS-scale store (1440×720, 3 leads, Zstd-compressed):

| Stage | Location | Measured cost |
|---|---|---|
| `xr.open_zarr` (metadata) | `api/core/zarr.py:read_dataset` | ~84 ms/store open |
| DB run resolution | `tiles.py::_resolve_run_and_field` (SQL) | ~sub-ms–few ms |
| Zarr chunk read (kept-open dset) | `tiles.py::_slice_field` | ~0.01 ms/read |
| **Fresh open + read per tile** | `tiles.py::_resolve_run_and_field` → `read_dataset` | **~2.8 ms/tile** |
| `np.vectorize` lon alignment | `tiles.py::_native_lon` | ~22 ms/tile |
| **Per-pixel Python loop (65,536 px)** | `tiles.py::render_tile_png` (lines 308–318) | **~175 ms/tile** |
| PNG encode | `api/core/png.py` | ~2 ms |
| **Total, no cache** | — | **~230–318 ms/tile** |

**Why it's slow:**
- **No server-side tile cache.** `render_tile_png` recomputes the full PNG on every request. `Cache-Control: public, max-age=300` only caches in the browser; identical tiles requested by different clients (or the same client after a cache eviction) recompute from scratch.
- **Re-opening the Zarr store per request.** `_resolve_run_and_field` (tiles.py:482) calls `read_dataset(run.zarr_store_path)` per request — a fresh `xr.open_zarr` against object storage per tile. With a global store this is ~2.8 ms/tile of S3 metadata + the store open; with a slow/remote store it dominates.
- **Re-resolving the run via SQL per request.** `_resolve_run_and_field` runs a `select(ModelRun)...order_by(cycle_time.desc())` per request (tiles.py:504). Cheap but redundant per tile.
- **Per-pixel Python loop.** The color-mapping loop (tiles.py:308–318) iterates 256×256=65,536 pixels one at a time, each calling `_interpolate_color` (Python function) and building a `bytearray` one pixel at a time. This is the single largest cost (~175 ms/tile) and is trivially vectorizable with NumPy (LUT + `np.take`).
- **`np.vectorize`** for longitude alignment (tiles.py:293) adds ~22 ms/tile; it's a Python-level loop disguised as vectorized.
- **No chunk-aligned read minimization.** `_slice_field` does slice to the tile's bounds (so only overlapping chunks are read), but because the store is re-opened per request and the per-pixel color loop dominates, the effective work per tile is far larger than necessary.

**Why it matters:** a MapLibre viewport fetches ~4–16 tiles per pan/zoom. At ~230–318 ms/tile with no cache, an interactive session quickly becomes unusable; each pan re-fetches and re-renders from scratch.

---

### Issue 2 — Ingestion fails when another forecast cycle is ingested

**Root cause:** `services/ingestion/src/ingestion/core/pipeline.py::_merge_lead` (lines 277–311) merges a single-lead dataset into a cycle store with **no cycle-identity validation**. It reads the existing store, keeps leads `!= new_lead`, and `xr.concat`s with `coords="minimal"`. The parser (`providers/noaa/parser.py::normalize`, line 160) **keeps** the `time` coordinate (the GRIB reference/analysis time) and only drops `step`/`valid_time`. When a file from a *different* cycle (e.g. 12Z) is merged into a store built for *another* cycle (e.g. 00Z), the scalar `time` coordinate values differ and `xr.concat` raises:

```
xarray.core.merge.MergeError:
conflicting values for variable 'time' on objects to be combined.
```

**Reproduced verbatim** with the real `_merge_lead` + `time`-carrying datasets:

```
wrote 00Z store with leads [6]
same-cycle merge OK; leads now [6, 12]
wrong-store merge raised: MergeError
  conflicting values for variable 'time' on objects to be combined. You can skip this check by specifying compat='override'.
```

**Why the existing tests miss it:** the `_merge_lead` unit tests (`services/ingestion/tests/test_pipeline.py`) build synthetic datasets (`_dataset_for_lead`) that carry **no `time` coordinate**, so a same-cycle merge "just works" and a cross-cycle merge "works" too (silently — the concat would drop `time` or take one value), never raising. The `time`-carrying real-parser output is never exercised in the merge path.

**The CLI gap:** `ingestion/cli.py` requires `--store` and never validates it against `--model/--cycle-date/--cycle-hour`. The `_merge_lead` merge happens *after* parse, and the store path is not derived from the forecast identity. So the failure surfaces as a confusing `MergeError` deep in the pipeline, and — worse — in cases where `time` happens not to collide (or if `compat="override"` were naively added), a wrong-cycle store would be silently contaminated.

**Related identity defect (from the cross-cutting audit):** the run id is not version-scoped (`catalog.py:359`: `run_{cycle_time:%Y%m%d%H%M}_{model_id}`), so two versions of the same model at the same cycle collide on the PK even though the schema's `(model_version_id, cycle_time)` uniqueness would allow both.

---

### Issue 2B — Multi-model / multi-cycle / multi-lead ingestion

**Root cause (design gap, not a bug):** the CLI (`ingestion/cli.py`) is strictly one-lead-per-invocation: `--model`, `--cycle-date`, `--cycle-hour`, `--lead-time-hours`, `--store`. There is no batch semantics, no store-path derivation or validation, no manifest, and no multi-lead/multi-cycle/multi-model orchestration. The prior commit `606e8f2` explicitly documents `--store` as the shared cycle store and requires the operator to pass the same store for every lead.

---

### Issue 3 — Location autocomplete

**Root cause:** the entire "search" surface is a PostGIS `ILIKE` substring match over three small tables (`cities`, `ski_resorts`, `stations`) in `services/api/src/api/services/search.py`. The frontend `LocationSearch` is already a polished combobox (debounce 300 ms, min length 1, abort, stale-guard, keyboard nav, empty/error/loading states), but it can only suggest rows already present in the seed data. There is **no Google Maps/Places integration anywhere in the repo** — the requirement text assumed a Google stack that does not exist here.

---

### Issue 4 — Elevation

**Root cause:** `elevation_m` is already in the `/v1/points` contract (`ForecastLocationOut.elevation_m`, nullable) and `ResolvedLocation` carries `elevation_m`. But:
- raw coordinates → `elevation_m=None` (`point_forecast.py:156-161`),
- cities → `elevation_m=None` (`point_forecast.py:184-187`, no elevation column),
- only ski resorts carry `summit_elevation_m` (`point_forecast.py:201-207`).

The frontend `SelectedLocationSummary` renders `—` when null. There is no elevation provider, no cache, and no server-side elevation resolution.

---

### Issue 5 — Ensemble capability (GFS identified as ensemble)

**Root cause:** the capability flag is **already correctly modeled** end-to-end:
- `models.is_ensemble` column (entities.py:44), set by ingestion (`catalog.py`), exposed by `/v1/models` (catalog.py:101) and `/v1/forecast/availability`,
- `/v1/ensembles` rejects non-ensemble models with 422 (`ensemble_data.py:210-224`; confirmed by `test_ensembles_non_ensemble_model_422`),
- the dashboard derives `selectedModelIsEnsemble` from availability and shows the empty-state / skips ensemble requests for deterministic models (`ForecastDashboard.tsx`).

**The remaining defect is the panel heading.** `ForecastDashboard.tsx:126-128` renders:

```tsx
<h3>Ensemble Statistics{selectedModel !== null ? ` (${selectedModel.toUpperCase()})` : ""}</h3>
```

For a deterministic selected model (e.g. `gfs`), `selectedModel` is non-null, so the heading prints **"Ensemble Statistics (GFS)"** above the empty-state message "No ensemble data available for the selected forecast." This is the misleading text. It conflates the two distinct states:
1. "this model is not an ensemble model" (deterministic GFS), and
2. "this is an ensemble model but data for this forecast is currently unavailable."

The dashboard test `ForecastDashboard.test.tsx` asserts the empty-state message but does **not** assert the heading, so the misleading text is not caught.

---

## C. Forecast Identity Audit

This section traces the forecast identity (`model`, `cycle/run`, `lead`, `variable`, `valid time`) through every layer and records each discovered gap. A forecast must never be identified merely as `"GFS + lead 18"` if multiple cycles exist, and raster/cache entries must not leak across cycles/models/leads.

### C.1 Identity by layer

| Layer | Identity carrier(s) | Cycle captured? | Variable/lead captured? | Notes |
|---|---|---|---|---|
| CLI (`ingestion/cli.py`) | `--model`, `--cycle-date`, `--cycle-hour`, `--lead-time-hours`, `--store` | Yes (cycle-date+hour) | Yes (lead) | `--store` is **manual**, never derived or validated against the cycle. |
| xarray dataset (`parser.py::normalize`) | `time` coord (kept), `lead_time_hours` (scalar), `member` (renamed from `number`), lat/lon | `time` kept; `cycle_time` **not** written as a named coord/attr | lead yes; variable via data-var codes | `step`/`valid_time` dropped. The store is **not** self-describing about its cycle. |
| Zarr store | data vars + coords + per-var `units` attrs | No explicit `cycle_time` attr/coord | lead via `lead_time_hours` coord | `zarr_writer.write_dataset` writes only per-var encoding; no dataset attrs. |
| S3 path | `--store` (operator-supplied), convention `s3://weather-data/{model}/{date}/{hour}/cycle.zarr` | Convention only; **not enforced** | n/a | A wrong-store path is not detected. |
| DB (`model_runs`) | `id = run_{cycle_time:%Y%m%d%H%M}_{model_id}`, `cycle_time`, `status`, `zarr_store_path` | Yes (cycle_time) | n/a (run-scoped) | Run id is **not version-scoped** → version collision risk. |
| DB (`forecast_products`) | `(run_id, variable_id, grid_id, product_type, lead_time_hours)` unique | via run_id | yes | One row per (var × lead). |
| API `/v1/points` | `models` (single), lat/lon, lead window | **newest run only — no cycle pinning** | via `variables`, lead window | `generated_at`/`valid_time` derive from the run's `cycle_time`. |
| API `/v1/ensembles` | `model`, lat/lon, `lead_time_hours` | **newest run only — no `initial_time` param** | via `variable`, lead | Cannot pin a cycle. |
| API `/v1/probabilities` | `model`, lat/lon, `variable`, lead | **newest run only** | yes | Cannot pin a cycle. |
| API `/v1/maps` | `model`, `variable`, `level`, `lead_time_hours`, `initial_time` (optional) | **optional `initial_time` pins the run** | yes | The only endpoint that supports cycle pinning. |
| Cache keys (`cache.py`) | `model`, lat/lon, `resolved_via`/`location_id`, variables, units, lead window | **MISSING** | lead window yes | `build_point_cache_key`, `build_probability_cache_key`, `build_ensemble_cache_key` contain **no cycle/run identity**. |
| Frontend selection | `{model, variable, initialTime, leadTimeHours}` | Yes (initialTime) | yes | `resolveValidTime` = `initialTime + leadTimeHours` (correct UTC). |

### C.2 Discovered identity gaps

1. **GAP-1 — Cache keys omit run identity (HIGH).** `build_point_cache_key` (cache.py:291-301), `build_probability_cache_key` (cache.py:323-332), `build_ensemble_cache_key` (cache.py:354-361) all key on `model` but **not** on `cycle_time`/run id. Two cycles of the same model at the same lat/lon/lead share a cache key. Because `/v1/points`, `/v1/probabilities`, `/v1/ensembles` expose no `initial_time`, the cached payload is whatever run was newest at first compute; a newer cycle ingested afterward is masked until the Redis TTL expires (point/ensemble 1800 s, probability 3600 s). The `point_query_fallback_audit` ledger inherits the same cycle-less key.

2. **GAP-2 — `/v1/ensembles` and `/v1/probabilities` cannot pin a cycle (MEDIUM).** Only `/v1/maps` accepts `initial_time`. Ensemble/probability requests always resolve the newest ready run, so a user cannot compare two cycles, and the cycle-less cache key (GAP-1) compounds this.

3. **GAP-3 — Zarr store is not self-describing about its cycle (MEDIUM).** The only cycle identity in the store is the un-renamed `time` coordinate; there is no explicit `cycle_time` attr/coord. Anything that decouples the store from its `model_runs` row (a copied store, a repair that drops the DB row) loses its run identity.

4. **GAP-4 — Run id not version-scoped (LOW-MEDIUM).** `catalog.py:359` `run_{cycle_time:%Y%m%d%H%M}_{model_id}` omits `version_string`; two versions of the same model at the same cycle collide on the PK (schema allows both via `(model_version_id, cycle_time)`).

5. **GAP-5 — `--store` is not validated against forecast identity (HIGH).** The CLI accepts any `--store`; `_merge_lead` never checks that the existing store's `time` matches the requested cycle. This is the direct cause of Issue 2.

6. **GAP-6 — `_merge_lead` silently depends on `coords="minimal"` for `time` (HIGH).** A same-cycle merge works because `time` is scalar and equal; a cross-cycle merge raises `MergeError`; a naive `compat="override"` would silently contaminate. Neither is an acceptable outcome — the merge must be gated on an explicit identity check.

7. **GAP-7 — The `time` coordinate is scalar after `expand_dims` (LOW, but root of GAP-6).** `parser.normalize` keeps `time` scalar; `expand_dims("lead_time_hours")` leaves it scalar; `xr.concat` then compares scalars. The merge logic has no identity semantics around `time`.

8. **GAP-8 — `initial_time` is not in the documented `/v1/maps` contract (LOW).** It works (the template carries it and MapLibre preserves the query string) but is undocumented; it should be added to API.md for the contract to be honest.

---

## D. Proposed Target Architecture

The goal is a capability- and identity-driven system where every dimension is explicit and no two layers disagree about which forecast is being served.

### D.1 Desired relationship model

```
model  ──  forecast_type (deterministic | ensemble)          (models.is_ensemble already exists)
  │
  ├── forecast run (cycle) = model + cycle_time               (model_runs, UNIQUE(model_version_id, cycle_time))
  │        ├── S3 store = {model}/{cycle_date}/{cycle_hour}/cycle.zarr   (derived, not manual)
  │        ├── Zarr coords: time (cycle), lead_time_hours (dim), member (ensemble), lat/lon
  │        └── DB run row (id, cycle_time, status, zarr_store_path)
  │
  ├── lead = lead_time_hours within a run                     (forecast_products row per var × lead)
  │
  └── variable = forecast_variables.variable_code             (temperature_2m, precipitation_rate)
```

- **Identity invariant:** a forecast is uniquely identified by `(model, cycle_time, lead_time_hours, variable)`. No layer may collapse to `(model, lead, variable)`.
- **Store invariant:** one S3 store = one forecast cycle. The store path is **derived** from `(model, cycle_date, cycle_hour)` and **validated** against the requested identity; a mismatched store is a hard error, never a silent merge.
- **Cache invariant:** every cache key must carry the resolved run identity (`cycle_time`), not just `model`.
- **API invariant:** endpoints that serve a specific forecast must be able to pin the run (at least optionally), and the default must be well-defined (newest ready run) and consistent with the cache key.
- **Capability invariant:** deterministic vs ensemble is a model capability (`is_ensemble`); it drives both what endpoints accept (already true for `/v1/ensembles`) and what the frontend renders.

### D.2 Where the current system matches

- `model_runs.UNIQUE(model_version_id, cycle_time)` — one run per cycle ✓
- `/v1/forecast/availability` returns real `initial_time`s (cycle times) ✓
- `/v1/maps` supports `initial_time` pinning ✓
- `is_ensemble` is modeled and enforced ✓
- Frontend selection carries `initialTime` and derives `validTime` correctly ✓

### D.3 Where the current system must change

- **Cache keys** must include `cycle_time`/run identity (GAP-1).
- **`/v1/ensembles` and `/v1/probabilities`** should accept an optional `initial_time` (GAP-2) so they are cycle-consistent with `/v1/maps` and `/v1/points`.
- **Zarr stores** should carry an explicit `cycle_time` (and `model_id`) in attrs (GAP-3), making the store self-describing and enabling store-vs-request validation.
- **Run id** should be version-scoped (GAP-4) or the PK collision resolved.
- **CLI/store path** must be derived and validated (GAP-5, GAP-6) — the core of Issue 2.
- **`/v1/maps` `initial_time`** should be documented (GAP-8).

---

## E. Multi-Forecast Ingestion Design

### E.1 CLI semantics — single-command, batch-ready

The current CLI is one lead per invocation. The target is a CLI that can describe **one or more forecast-run specifications** without an accidental Cartesian product. The design keeps the common single-lead case ergonomic and adds a batch form for multi-lead / multi-cycle / multi-model.

**Design principle: an invocation describes a *set of forecast-run specifications*, not a Cartesian product.** Concretely:

- `--model`, `--cycle-date`, `--cycle-hour` each accept **repeatable** values (or a single value).
- `--lead-time-hours` accepts **repeatable** values (or a single value).
- The expansion rule is **per-run, not cross-product**: a specification is formed by *zipping* the model/cycle/lead lists **only when the user passes them as aligned tuples**, OR by an explicit `--spec`/manifest. To avoid accidental huge jobs, the CLI:
  - **requires** at least one model, cycle date, cycle hour, and lead;
  - **defaults** to a small, safe expansion (e.g. if exactly one model/date/hour are given and multiple leads, it's "all leads of that one cycle" — not a product across models);
  - **rejects** ambiguous combinations (e.g. multiple models + multiple dates + multiple hours with a single lead) unless a manifest is supplied.

### E.2 Concrete CLI examples

**Common case (one cycle, many leads):**

```powershell
python -m ingestion.cli ingest `
  --model gfs `
  --cycle-date 2026-08-13 `
  --cycle-hour 00 `
  --lead-time-hours 0 6 12 18 24
```

This is a **single forecast-run specification**: `{model: gfs, cycle: 2026-08-13T00Z, leads: [0,6,12,18,24]}`. The store path is **derived** as `s3://weather-data/gfs/2026-08-13/00/cycle.zarr`.

**Multiple cycles of one model:**

```powershell
python -m ingestion.cli ingest `
  --model gfs `
  --cycle-date 2026-08-13 `
  --cycle-hour 00 12 `
  --lead-time-hours 0 6 12 18 24
```

This is **two** run specifications: `{gfs, 00Z, leads}` and `{gfs, 12Z, leads}`. The expansion is per-cycle (each cycle gets its own derived store), **not** a cross-product that would multiply leads × cycles × models.

**Multiple models:**

```powershell
python -m ingestion.cli ingest `
  --model gfs gefs `
  --cycle-date 2026-08-13 `
  --cycle-hour 00 `
  --lead-time-hours 0 6 12 18 24
```

This is **two** run specifications: `{gfs, 00Z, leads}` and `{gefs, 00Z, leads}`. Each resolves to its own store.

**Explicit multi-run (manifest) for complex jobs:**

```powershell
python -m ingestion.cli ingest --manifest ingest-manifest.json
```

```json
{
  "runs": [
    {
      "model": "gfs",
      "cycle_date": "2026-08-13",
      "cycle_hour": "00",
      "lead_time_hours": [0, 6, 12, 18, 24]
    },
    {
      "model": "gfs",
      "cycle_date": "2026-08-13",
      "cycle_hour": "12",
      "lead_time_hours": [0, 6, 12, 18, 24]
    },
    {
      "model": "gefs",
      "cycle_date": "2026-08-13",
      "cycle_hour": "00",
      "lead_time_hours": [0, 6, 12, 18, 24]
    }
  ]
}
```

### E.3 Anti-Cartesian guards

- **Aligned-zipped lists:** when multiple values are given for more than one identity dimension (model, cycle-date, cycle-hour), the CLI requires the lengths to match and **zips** them into aligned specifications. It must **not** broadcast each model across every cycle × hour.
- **Manifest is the escape hatch:** complex/combinatorial jobs are expressed as an explicit `runs` list in a manifest, so the operator states intent rather than the CLI inferring it.
- **Dry-run:** `--dry-run` prints the resolved run specifications (model, cycle_time, derived store path, leads) without downloading or writing, so an operator can inspect exactly what a batch will do before running it.
- **Limit guard:** a batch that expands to more than a configurable `--max-runs` (default e.g. 16) requires explicit confirmation or a manifest.

### E.4 Store-path generation & validation

**Derive the path (default):** `s3://weather-data/{model}/{cycle_date:%Y-%m-%d}/{cycle_hour:02d}/cycle.zarr`. This makes the store a pure function of the forecast identity, so the same model/cycle always maps to the same store and different cycles can never collide.

**Validate an explicitly-supplied path (back-compat):** keep `--store` for the legacy single-run case, but **validate** it before merging:
1. derive the canonical path from `(model, cycle_date, cycle_hour)`;
2. if `--store` is supplied and differs from the derived path, warn (or, for a fresh store, accept an explicit override with a `--allow-custom-store` flag);
3. **before merging into an existing store, read its `time` coordinate** (or a stored `cycle_time` attr) and confirm it equals the requested `cycle_time`; otherwise **fail fast** with a clear error.

**Fail-fast behavior:** a store whose `time`/`cycle_time` does not match the requested cycle → abort before any write. This is the direct fix for Issue 2 and is the **only** acceptable response — never `compat="override"`.

### E.5 Batch semantics — partial failure, idempotency, atomicity

- **Per-run atomicity:** each forecast-run specification (one cycle) is processed independently. A failure in one run (e.g. a bad lead file) does **not** corrupt other runs; the batch reports per-run status.
- **Lead idempotency (already present):** re-ingesting the same lead replaces it in the cycle store (`_merge_lead` keep-`!= new_lead` + concat). This is preserved.
- **Store-write atomicity:** for a cycle store, write to a **staged sibling** (`cycle.zarr.tmp-<run>-<ts>`) then atomically rename/swap into `cycle.zarr` when the write is known-good, so a partially-written store is never served. (The current code writes `mode="w"` to the live path; this must change for batch safety.)
- **DB transaction boundary:** the catalog write (`record_run`) is already one transaction; it must commit **after** the store is fully written and verified, so a failed store write never leaves a `ready` run pointing at a partial store.
- **Concurrent ingestion:** keep the existing `record_ingested_dataset` retry on `IntegrityError`; for S3, prefer a per-store lock (object `lock` marker or a conditional write) so two workers cannot interleave writes to the same cycle store.
- **Retries:** per-run, bounded retries with backoff on transient upstream/network errors (already in `connector.download`); a retried lead is idempotent.
- **Catalog consistency:** `forecast_products` rows are upserted per (var × lead) under the run; after a partial batch, the run reflects exactly the leads written (because `record_run` rebuilds products from the dataset's `lead_time_hours`).
- **Re-ingesting an existing lead:** replaces that lead's slice and updates the product row (idempotent).
- **Incompatible variable/grid schemas between leads:** the merge must validate that incoming leads share the same grid (lat/lon axes), variable set, and member axis as the existing store; a mismatch fails fast rather than corrupting the store.

### E.6 S3 layout (target)

```
s3://weather-data/
  gfs/
    2026-08-13/
      00/cycle.zarr
      06/cycle.zarr
      12/cycle.zarr
      18/cycle.zarr
  gefs/
    2026-08-13/
      00/cycle.zarr
```

This matches the current convention (the CLI docstring already shows `s3://weather-data/gfs/2026-07-21/00/cycle.zarr`), so the target layout is **compatible with existing stores** — the change is that the path becomes *derived and validated* rather than manual.

---

## F. Raster Performance Plan

### F.1 Current bottleneck evidence

Measured on a real GFS-scale store (1440×720, 3 leads, Zstd): **~230–318 ms/tile**, dominated by the per-pixel Python loop (~175 ms/tile) and `np.vectorize` (~22 ms/tile), with a fresh store-open (~2.8 ms/tile) and no server-side tile cache. PNG encode is ~2 ms. A MapLibre viewport fetches ~4–16 tiles per pan/zoom, so an interactive session is quickly unusable.

### F.2 Proposed changes (in priority order)

1. **Vectorize the color mapping** (`tiles.py::render_tile_png`). Replace the per-pixel loop + `_interpolate_color` calls with a NumPy lookup table (`np.interp` over the stops' value→RGB, then `np.take`), and build the scanlines with NumPy `tobytes()`/`tobytes()` into a single bytearray. This eliminates ~175 ms/tile. **Expected:** ~5–10 ms/tile for the color stage.
2. **Vectorize longitude alignment** (`tiles.py::_native_lon` via `np.vectorize`). Replace with a vectorized `np.mod`/`np.where` expression (the alignment is a pure arithmetic on arrays). **Expected:** ~22 ms → sub-ms.
3. **Add a server-side tile cache.** A process-local LRU (or Redis) keyed by `(model, variable, level, zoom, x, y, lead_time_hours, initial_time)` storing the PNG bytes, with a TTL aligned to the model update cadence (the tile `Cache-Control: max-age=300` already implies this). **Expected:** identical tiles served from memory (~0 ms) instead of re-rendering.
4. **Keep the Zarr store open across tile requests for a run** (process-lifetime cache of the open dataset for the resolved run), instead of `read_dataset` per request. This requires invalidating on ingestion of a new run — a simple "store path + file mtime/version" key. **Expected:** removes ~2.8 ms/store-open + S3 metadata latency per tile.
5. **Preserve correctness:** the cache key must include the **run identity** (GAP-1) so tiles never leak across cycles/models/leads; the vectorized color mapping must reproduce the exact stop interpolation (test against the current per-pixel loop).
6. **Frontend:** the layer template already includes `lead_time_hours` and (when selected) `initial_time`, so distinct cycles already produce distinct tile URLs. No frontend change is required for caching; but confirm MapLibre's `{z}/{x}/{y}` substitution preserves the query string (it does). Optionally reduce redundant tile fetches by setting `minzoom`/`maxzoom` and avoiding re-apply on every selection (the `useMapLayer` hook already cancels stale requests).

### F.3 Caching strategy

- **Layer:** server-side LRU/Redis tile cache keyed by the full identity tuple including `initial_time` (run). TTL ~ model update cadence (300–3600 s).
- **Browser:** keep `Cache-Control: public, max-age=300` for tiles; add `ETag`/`If-None-Match` if cheap.
- **Cache-key correctness:** the key MUST include `(model, variable, level, zoom, x, y, lead_time_hours, initial_time)`. This is the same identity discipline as GAP-1 — a tile for cycle 00Z must never satisfy a request for cycle 12Z.

### F.4 Expected impact

| Change | Expected per-tile cost | vs today |
|---|---|---|
| Vectorized color + lon | ~5–15 ms | ~230–318 ms → ~20–30 ms |
| + server tile cache (hit) | ~0–1 ms | ~230–318 ms → ~1 ms |
| + kept-open dataset | removes S3 metadata latency | further margin |

### F.5 Correctness risks

- **Cache-key leakage across runs** is the top risk — mitigated by including `initial_time`/run identity in the key (GAP-1 fix).
- **Vectorized color must exactly reproduce the per-pixel interpolation** (stop positions, clamping, no-data transparency) — mitigated by differential tests against the current loop on synthetic and real fields.
- **Kept-open dataset staleness** — a newly-ingested run must invalidate the cached dataset; mitigated by keying the dataset cache on store path + store mtime/version, and by falling back to a fresh open if the store changes.
- **Cache invalidation on re-ingest** — a re-ingested lead must not leave a stale tile; the run-identity cache key plus TTL bounds staleness, and the store-version key invalidates it.

---

## G. Location Autocomplete Design

### G.1 Current state (verified)

The frontend `LocationSearch.tsx` is already a production-quality combobox:
- Debounce 300 ms (`useSearch.ts`), min query length 1, abort of in-flight requests, stale-response guard (`queryRef` check),
- Keyboard navigation (ArrowUp/Down, Enter, Escape), ARIA combobox/listbox semantics,
- Loading / empty / error states, click-outside close, disabled state.

The **only** gap is the data source: `/v1/search` is a PostGIS `ILIKE` substring match over the tiny seeded `cities`/`ski_resorts`/`stations` tables. It cannot provide a real global place autocomplete (e.g. typing "den" → "Denver, Colorado, United States"). **There is no Google Maps/Places stack anywhere in the repo.**

### G.2 Provider / API — the decision

**Recommended: Google Places API (New) — Autocomplete + Place Details, proxied through the FastAPI backend.**

Current product facts (mid-2026; pricing should be re-verified against the live pricing page before finalizing the numbers):

- **Endpoint — Autocomplete (New):** `POST https://places.googleapis.com/v1/places:autocomplete` with header `X-Goog-Api-Key`. Body fields: `input` (required), `locationBias` (circle/rectangle/origin — bias toward a region without hard-limiting), `regionCode`, `languageCode`, `includedPrimaryTypes` (e.g. `["locality","address"]` — **critical:** omitting this can return `queryPrediction` results billed as the more expensive **Text Search (New)**), `sessionToken`.
- **Response:** `suggestions[]`, each a `placePrediction { placeId, text, structuredFormat { mainText, secondaryText }, types, distanceMeters }` or a `queryPrediction` (text-only, no placeId, billed as Text Search).
- **Endpoint — Place Details (New):** `GET|POST https://places.googleapis.com/v1/places/{placeId}` with `X-Goog-Api-Key`; for POST use `X-Goog-FieldMask` to select exactly the fields you need (controls billing "usage" costing). Canonical fields: `location {latitude, longitude}`, `displayName`, `formattedAddress` / `shortFormattedAddress`, `addressComponents` (country = `types:["country"]`, region = `types:["administrativeAreaLevel1"]`).
- **Session tokens:** generate a fresh UUIDv4 when the user focuses the search box; reuse the **same** token for every Autocomplete keystroke and the subsequent Place Details call; discard after selection. **Billing effect:** with a session token, a whole typeahead session is billed as **1 Autocomplete + 1 Place Details**; without one, **every keystroke is billed individually**. This is the single biggest cost lever.
- **Pricing (verify — volatile):** Autocomplete (New) ≈ $2.83/1000; Place Details (New) ≈ $17/1000; Text Search (New) ≈ $32/1000; a recurring ~$200/month Google Cloud credit applies. Re-confirm at `https://developers.google.com/maps/billing-and-pricing/pricing`.
- **API key security:** **proxy through FastAPI.** The backend holds a server key (IP-restricted + API-restricted to the Places API); the Next.js client calls `/v1/search` (already proxied to FastAPI via `next.config.mjs`). The key **never reaches the browser**; you gain per-user rate limiting, caching, and centralized billing. (Browser-direct with a referrer-restricted key + Places SDK for Web is possible but weaker for a weather app that already has a backend.)
- **Alternatives (if a Google dependency is undesired):** Mapbox Geocoding API (`api.mapbox.com/geocoding/v5/mapbox.places/{query}.json`, `autocomplete=true`, pairs naturally with MapLibre); Nominatim (OSM, free, ~1 req/s policy, ODbL attribution); Photon (komoot, free OSM-based); Pelias (self-hostable). These avoid Google billing but have their own rate/usage limits.

**Recommendation:** use **Google Places API (New)** for the initial implementation, proxied through FastAPI, because it is the current recommended product, has clean session-token billing semantics, and provides both ranked autocomplete and canonical place resolution. Keep the provider abstraction so Mapbox/OSM remains a drop-in alternative. **(Design decision — see Section N.)**

Working design (provider-agnostic):

- **Provider abstraction** (backend): `PlaceAutocompleteProvider` with `suggest(text, session_token) -> list[PlaceSuggestion]` and `resolve(place_id, session_token) -> ResolvedPlace { display_name, lat, lon, country, region }`. This mirrors the `ElevationProvider` abstraction and keeps the product vendor-independent.
- **Backend boundary**: the FastAPI service calls the provider. The frontend talks only to `/v1/search`. Because the frontend already proxies `/v1/*` to FastAPI via Next.js rewrites (`next.config.mjs`), **no browser-exposed API key is needed** — the provider key lives server-side.

### G.3 Request lifecycle (backend)

`/v1/search` (already existing) gains autocomplete semantics:
1. Client types → debounced `GET /v1/search?q=den&type=place`.
2. Backend calls the `PlaceAutocompleteProvider` (e.g. Google Places Autocomplete) with a **session token**.
3. Provider returns ranked suggestions; backend maps them to the existing `SearchResultOut` shape (`name`, `region`, `country`, `latitude`, `longitude`, `object`).
4. When the user **selects** a suggestion, the client resolves the canonical place (provider `resolve`) → lat/lon → map recenters → point forecast updates. This resolution can be a second provider call with the same session token (for Google billing semantics) or a client-side mapping.

### G.4 Request lifecycle (frontend)

The existing `useSearch` hook already debounces and aborts. The addition is the **selection flow**:
- `LocationSearch` currently calls `onSelect` with a `SearchResult`. For a place suggestion that has no lat/lon yet (only a `place_id`), the selection must first **resolve** the canonical place (provider `resolve`), then update the shared `SelectedLocation` (name, lat/lon, region, country).
- This maps onto the existing `searchResultToSelectedLocation` in `lib/forecast/selection.ts` — extend it to accept a resolved place.

### G.5 Debounce / session / stale handling

- **Debounce**: keep 300 ms; min query length 2 (Google publishes no formal minimum; the ecosystem convention is 2–3 chars; 2 is a good balance). With a session token, extra keystrokes are nearly free (one session), so debounce is mainly about server load.
- **Session tokens** (Google): the backend (or frontend) generates a fresh UUIDv4 when the user focuses the search box; reuses it for every Autocomplete keystroke; reuses it again for the Place Details call on selection; discards after selection. The token is carried in the `/v1/search` call (and the place-resolution call) so Google bills one session per completed selection, not per keystroke.
- **Cancellation/stale**: already handled (`useSearch` aborts + `queryRef` guard). Ensure the same guard applies to the resolution step.

### G.5b Quota / cost control (Google)

- Session tokens (one billable session per completed selection) are the primary lever.
- **`includedPrimaryTypes`** must be set to avoid `queryPrediction` results being billed at the higher Text Search rate.
- Per-IP / per-key rate limit on `/v1/search` (FastAPI middleware or a Redis counter) prevents abuse of the server key.
- Server-side caching of repeated autocomplete queries (debounced, TTL-bounded) further reduces provider calls.

### G.6 Result model

Extend the provider suggestion → `SearchResultOut` mapping. Place suggestions should be ranked by the provider. The existing `SearchResultOut` already has `name`, `region`, `country`, `latitude`, `longitude`, `elevation_m`, `object` — add a `place_id`/`provider_id` field (additive, non-breaking) so selection can resolve the canonical place.

### G.7 Coordinate selection flow

```
type "den"
→ debounced /v1/search?q=den (provider autocomplete, session token)
→ suggestions: Denver, Colorado, United States; Denver International Airport; ...
→ user selects "Denver, Colorado, United States"
→ resolve(place_id, session_token) → { display_name, lat: 39.7392, lon: -104.9842, country: US, region: Colorado }
→ update SelectedLocation → map recenters → point forecast updates → elevation resolution (Section H)
```

### G.8 Security / quota considerations

- **API key**: lives server-side (FastAPI). Never in the browser bundle. Use a **server key** restricted by **IP address** and restricted to the **Places API** service. Browser-direct would require a referrer-restricted key (inherently visible in client JS), which is weaker — the proxy is the recommended pattern.
- **Quota/cost control**: debounce + session tokens (one billable session per completed selection) + `includedPrimaryTypes` (avoid Text Search billing) + a per-IP/per-key rate limit on `/v1/search` + server-side caching of repeated queries.
- **Error handling**: provider errors degrade gracefully — the combobox shows its existing error state; no crash.
- **No-results**: existing empty state "No matching locations." reused.
- **Tests**: the provider abstraction is mocked; tests never call live Google services (matching the existing `respx`/`httpx` mock pattern and `docs/TESTING.md` network isolation rule).

---

## H. Elevation Architecture

### H.1 Requirement

After a location is resolved (autocomplete or map click), the Coordinates UI must show an elevation in meters (e.g. "Elevation: 1,609 m"), sourced from an authoritative terrain dataset — not guessed from forecast variables. The value must be explicit (`elevation_m`, already in the `/v1/points` contract), cached, and resilient to lookup failure.

### H.2 Strategy comparison

#### Strategy A — Network elevation API

| Option | Data source | Coverage | Resolution | Latency | Cost | Auth | Reliability | Caching/ToS | Production fit |
|---|---|---|---|---|---|---|---|---|---|
| **Google Elevation API** | SRTM+ (Google) | Global | ~30 m | ~50–200 ms | $5/1k req after $200/mo credit | API key + billing | High | **ToS restricts caching (≤30 days), cannot persist** | Limited — the caching restriction makes it poor for a cache-first product |
| **Open-Elevation** (open-elevation.com) | SRTM | Global (±60°) | 30 m | ~100–500 ms (unreliable) | Free | None | **Low (single academic server)** | Allowed | Not alone |
| **OpenTopoData** (public) | SRTM/ASTER/etc | Dataset-dependent | 30–90 m | ~50–300 ms | Free | None | Medium (~1 req/s/IP) | Allowed | Fallback only |
| **AWS Terrarium/Skadi** (public S3) | SRTM+GMTED | ±60° / global | ~30 m (z14) | ~50–200 ms + decode | Free (S3 egress) | None | High | Allowed | **Good with a local cache** |

#### Strategy B — Local / server-side DEM

| Dataset | Resolution | Coverage | Accuracy | Storage | Licensing |
|---|---|---|---|---|---|
| **SRTM v3** | 30 m (1 arc-sec) / 90 m | ±60° lat | ~8–16 m | ~600 GB raw / 100–150 GB COG (30 m); ~70 GB raw / 10–20 GB COG (90 m) | **Public domain** (NASA/NGA) |
| **Copernicus GLO-30** | 30 m (and 90 m) | 84°N–90°S (true global incl. poles) | ~4 m spec | ~100–200 GB COG (30 m) | **Free, Copernicus Programme, attribution required** |

Both are available as public AWS Open Data COGs (`s3://elevation-tiles-prod/skadi/` for SRTM; `s3://copernicus-dem-30m/` for Copernicus) and via Microsoft Planetary Computer. Server-side point query (rasterio `ds.sample`, or xarray/rioxarray bilinear `interp`) is **microsecond–~5 ms** per point — vastly cheaper than any network API and fully self-hosted.

### H.3 Recommendation

**Primary: local/server-side DEM (Strategy B), using Copernicus GLO-30 as the source**, with an on-demand COG-tile fetch + per-tile LRU cache for arbitrary coordinates, and **bilinear interpolation** at 30 m. Rationale:

- **Correctness/coverage**: GLO-30 is truly global (84°N–90°S), more accurate (~4 m spec) than SRTM, and available as public COGs on AWS/Planetary Computer — no per-call cost, no external billing, no rate limits.
- **Operational independence**: no dependency on Google billing/ToS; elevation is effectively static per coordinate, so a local DEM + cache eliminates both cost and network failure modes.
- **Caching freedom**: unlike Google Elevation (whose ToS restricts caching/persistence), a local DEM allows indefinite caching — exactly what a static-per-coordinate elevation needs.
- **Fits the existing stack**: the platform already stores gridded data in Zarr/COG-style stores; a DEM is a natural fit.

**Alternative (if zero storage footprint is preferred):** AWS Terrarium/Skadi (public S3, no auth, SRTM-based) as the fetch-on-demand source feeding the same local LRU cache — same data, no Google. **Avoid** Google Elevation (ToS caching restriction) and Open-Elevation (reliability).

### H.4 Provider abstraction

Introduce a backend `ElevationProvider` interface (application-level, testable):

```
ElevationProvider
    get_elevation(latitude: float, longitude: float) -> float | None
```

Concrete implementations:
- `DEMElevationProvider` — reads a local/cached COG tile (Copernicus GLO-30 or SRTM), bilinear interpolation, LRU tile cache. The initial implementation.
- `GoogleElevationProvider` — wraps the Google Elevation API (kept behind the same interface for future/fallback, but **not** the initial implementation).

This matches the ENGINEERING_CONTRACT "provider isolation" pattern already used for ingestion (`providers/noaa/`). The interface returns `None` for no-data/ocean rather than raising, so the UI can show "unavailable."

### H.5 Caching

Elevation is effectively static per coordinate. Cache design:

- **Round coordinates** to a precision that does not materially degrade accuracy. At 30 m, rounding to **~3 decimal degrees (~100 m)** is safe for a city-level display (e.g. 39.739 → 39.739); at 90 m, rounding to 3 decimals is still fine. A rounded key avoids repeated lookups for nearby clicks.
- **Layer**: process-local LRU (rounded-lat/lon → elevation), plus optionally a DB cache table keyed by the rounded coordinate. Since the DEM is local, the LRU is the primary cache and the DB is optional.
- **Deterministic**: the same rounded coordinate always yields the same elevation, so cache behavior is deterministic and testable.

### H.6 API / schema changes

- **No schema change required** — `elevation_m` already exists on `ForecastLocationOut` (`/v1/points`) and `SearchResultOut`. The change is that elevation is now **populated** for raw coordinates and cities (currently only ski resorts have it).
- **Where it belongs**: `ResolvedLocation.elevation_m` is populated by `resolve_location` via the `ElevationProvider` when the resolved record does not already define an elevation. The provider is called server-side (FastAPI), so no client key is exposed.
- **Frontend**: `SelectedLocationSummary` already renders `elevation_m` (as `—` when null). Change the null rendering from `—` to `unavailable` per the requirement, and consider an imperial display later (out of scope).

### H.7 UI behavior

- After autocomplete/map-click resolution, the `SelectedLocation` carries coordinates; `resolve_location` (server-side) resolves elevation via the provider and returns it in `/v1/points`.
- The UI shows "Elevation: 1,609 m" (or `unavailable` when the provider returns `None`/the location is ocean/no-data).
- The point forecast panel and the location summary both reflect the resolved elevation.

### H.8 Fallback behavior

- Provider returns `None` → UI shows `unavailable` (never a crash, never `—`).
- Provider network/DEM read error → the lookup is skipped gracefully (elevation stays `None`), the forecast still renders.
- Cache miss → lookup from DEM; cache hit → return cached.
- No terrain value (ocean, SRTM void) → `None` → `unavailable`.

---

## I. Ensemble Capability Fix

### I.1 The defect

`ForecastDashboard.tsx:126-128` renders the panel heading `Ensemble Statistics (<MODEL>)` for **any** selected model, including a deterministic one (GFS). When `ensembleModel === null` (deterministic selected model), the body already shows the honest empty-state "No ensemble data available for the selected forecast." — but the heading above it says "Ensemble Statistics (GFS)", which is semantically wrong: GFS is not an ensemble model, and there is no "ensemble data" to be unavailable for it.

The two distinct product states are currently conflated by the heading:
1. **"This model is not an ensemble model"** (deterministic GFS) — there is no ensemble capability at all.
2. **"This is an ensemble model but data for this forecast is currently unavailable"** (ensemble-capable GEFS with no ready run/lead) — the capability exists but the data is missing.

### I.2 Target representation (capability-driven)

The capability is already modeled as `models.is_ensemble` and flows to the frontend via `/v1/forecast/availability` (`ModelAvailability.is_ensemble`) and `/v1/models` (`is_ensemble`). The fix is to make the **UI** capability-driven rather than hard-coded:

- **GFS selected (deterministic):** do **not** render an "Ensemble Statistics" panel at all. The deterministic forecast is the point forecast; there is no ensemble section, so no misleading heading.
- **GEFS selected (ensemble):** render the ensemble panel with the heading `Ensemble Statistics (GEFS)`. If the ensemble data is genuinely unavailable (no ready run / lead for the point), show a *distinct* empty state: **"Ensemble data is not yet available for this forecast."** — not the deterministic-model message.
- Differentiate the two states in both the heading and the body.

### I.3 Implementation

In `ForecastDashboard.tsx`:
- Render the ensemble section **only when `selectedModelIsEnsemble` is true** (the `ensembleModel !== null` branch already exists).
- When `ensembleModel === null` (deterministic selected model): render **no** ensemble section, OR a collapsed/absent panel — never a "Ensemble Statistics (GFS)" heading.
- When `ensembleModel !== null` but the ensemble hooks return no data / an error: show the distinct "data unavailable" message (already partially present via the `error`/`empty` branches), but ensure the heading only appears for ensemble models.

The exact heading logic becomes:

```tsx
{selectedModelIsEnsemble && (
  <section aria-label="Ensemble statistics">
    <h3>Ensemble Statistics ({selectedModel.toUpperCase()})</h3>
    {/* ensembleModel !== null here; body shows chart, empty "unavailable", or error */}
  </section>
)}
```

And the "not an ensemble model" state is simply the absence of the section — the capability drives the UI.

### I.4 Distinguish the two empty states

| State | Heading | Body |
|---|---|---|
| Deterministic model (GFS) | (no ensemble section) | — |
| Ensemble model, data present | `Ensemble Statistics (GEFS)` | chart |
| Ensemble model, data unavailable | `Ensemble Statistics (GEFS)` | **"Ensemble data is not yet available for this forecast."** |
| Ensemble model, request error | `Ensemble Statistics (GEFS)` | scoped error (already present) |

### I.5 Backend consistency

The backend already enforces the capability (`/v1/ensembles?model=gfs` → 422). No backend change is required for the capability itself. However, the **`/v1/ensembles` cycle-pinning gap (GAP-2)** is relevant: an ensemble request always resolves the newest run, so "ensemble data unavailable" can occur when a lead is missing in the newest run even though an older run has it. That is a separate identity issue (fixed in the API/cache plan), not a capability issue.

---

## J. File-by-File Implementation Plan

> Every change below is additive or a targeted fix; none redesigns the approved architecture. "New file" is marked explicitly.

### J.1 Ingestion identity & batch (Issues 2, 2B)

#### `services/ingestion/src/ingestion/core/base.py`
- **Why:** add a new domain exception for store/cycle mismatch.
- **Change:** add `CycleStoreMismatchError(IngestionError)` ("the requested forecast cycle does not match the store at <path>").
- **Tests affected:** new unit tests.

#### `services/ingestion/src/ingestion/providers/noaa/parser.py`
- **Why:** make the store self-describing about its cycle (GAP-3) and give `_merge_lead` an authoritative identity to validate against.
- **Change:** in `normalize`, also assign a `cycle_time` attribute (and keep the `time` coordinate) derived from the `time` coordinate: `dataset.attrs["cycle_time"] = str(time)` and `dataset.attrs["model_id"]` via a param. Keep dropping `step`/`valid_time`; do **not** drop `time`.
- **Tests affected:** `test_parser.py` — add assertion that `cycle_time` attr is set and equals the `time` value.

#### `services/ingestion/src/ingestion/core/zarr_writer.py`
- **Why:** persist the self-describing identity (GAP-3) and support staged/atomic writes.
- **Change:** `write_dataset` to also write dataset attrs (xarray already persists `attrs` via `to_zarr`); add `write_dataset_atomic` (write to a staged sibling then rename/swap) for batch safety (Section E.5).
- **Tests affected:** `test_zarr_roundtrip.py` — assert attrs survive a round-trip.

#### `services/ingestion/src/ingestion/core/pipeline.py`
- **Why:** the core fix for Issue 2 (GAP-5/GAP-6) and batch safety.
- **Change:**
  - `_merge_lead(dataset, store_path)` gains a **store-identity guard**: before reading the existing store, derive the requested `cycle_time` from `dataset` (the `time` coord or `cycle_time` attr); open the existing store; if its `time`/`cycle_time` differs from the requested cycle, raise `CycleStoreMismatchError` (fail-fast, **never** `compat="override"`).
  - Validate grid/variable/member compatibility between the incoming lead and the existing store (same lat/lon axes, variable set, member axis) — fail fast on mismatch.
  - Route the store write through `write_dataset_atomic` for cycle stores (when `store_path` resolves to a cycle store) so a partial write is never served.
- **Tests affected:** `test_pipeline.py` — add wrong-cycle merge test (expects `CycleStoreMismatchError`, not `MergeError`), same-cycle multi-lead test (existing), grid-schema-mismatch test.

#### `services/ingestion/src/ingestion/cli.py`
- **Why:** derive/validate the store path, add batch/multi-run semantics (Issue 2B), keep back-compat.
- **Change:**
  - Add `derive_store_path(model, cycle_date, cycle_hour) -> str` producing `s3://weather-data/{model}/{cycle_date:%Y-%m-%d}/{cycle_hour:02d}/cycle.zarr`.
  - When `--store` is omitted, derive it; when supplied, validate it against the derived path (allow explicit override only with `--allow-custom-store`).
  - Add repeatable `--model`, `--cycle-hour`, `--lead-time-hours` (and `--cycle-date` stays single, or repeatable) with **aligned-zipped** expansion (Section E.3) — never a Cartesian product.
  - Add `--manifest <path>` for explicit multi-run specs; add `--dry-run` to print resolved run specs; add `--max-runs` guard.
  - Loop each resolved run-spec through the existing `_run_ingest` logic; report per-run status; non-fatal per-run failures do not abort the whole batch.
- **Tests affected:** `test_cli.py` — add multi-lead (same cycle), multi-cycle, multi-model, store-derivation, store-mismatch, dry-run, manifest, anti-Cartesian tests.

#### `services/ingestion/src/ingestion/core/catalog.py`
- **Why:** fix the run-id version collision (GAP-4) and keep `record_run` batch-safe.
- **Change:** scope the run id to the version: `run_{version_id}_{cycle_time:%Y%m%d%H%M}` (or include `version_string`). Keep the `(model_version_id, cycle_time)` unique constraint. Ensure `record_ingested_dataset` still retries on concurrent collisions and records the effective store path.
- **Tests affected:** `test_catalog.py` — add two-versions-same-cycle test (distinct run ids, both rows).

#### `services/ingestion/src/ingestion/core/config.py`
- **Why:** configure batch defaults and DEM/Google credentials later.
- **Change:** add `MAX_BATCH_RUNS` (default 16), `STORE_PATH_TEMPLATE`, and (later) `GOOGLE_PLACES_API_KEY` / `ELEVATION_PROVIDER` settings (Section G/H).
- **Tests affected:** config import smoke tests.

### J.2 API / cache identity (GAP-1, GAP-2, GAP-8)

#### `services/api/src/api/services/cache.py`
- **Why:** cache keys must include run identity (GAP-1) so tiles/points/ensembles never leak across cycles.
- **Change:** add an optional `cycle_time: datetime | None` (and/or resolved `run_id`) parameter to `build_point_cache_key`, `build_probability_cache_key`, `build_ensemble_cache_key`, folded into the canonical payload. The routers pass the resolved run's `cycle_time`.
- **Tests affected:** `test_points.py`, `test_ensembles.py`, `test_probabilities.py` — add "two cycles → distinct cache keys" tests.

#### `services/api/src/api/services/point_forecast.py`
- **Why:** carry run identity to the cache key and (optionally) support cycle pinning.
- **Change:** `build_point_forecast` returns/exposes the resolved run's `cycle_time`; the router folds it into the cache key. Optionally add an `initial_time` parameter to `resolve_location`/`build_point_forecast` to pin the run (mirroring `/v1/maps`).
- **Tests affected:** `test_points.py` — add cache-key-differs-by-cycle test.

#### `services/api/src/api/routers/points.py`, `routers/ensembles.py`, `routers/probabilities.py`
- **Why:** fold run identity into cache keys; add optional `initial_time` to ensembles/probabilities (GAP-2).
- **Change:** resolve the run (via the service) and pass its `cycle_time` into the cache-key builders; add optional `initial_time` query param to `/v1/ensembles` and `/v1/probabilities` (additive, non-breaking) to pin the run.
- **Tests affected:** corresponding router tests.

#### `services/api/src/api/routers/maps.py`
- **Why:** document `initial_time` (GAP-8) and keep tile cache keys identity-correct.
- **Change:** no functional change (tiles already carry `initial_time`); add the parameter to the docstring/OpenAPI description.
- **Tests affected:** none (or assert the OpenAPI description).

#### `services/api/src/api/core/zarr.py`
- **Why:** enable kept-open dataset caching (raster performance, F.2.4) with identity-safe invalidation.
- **Change:** add an optional process-lifetime `get_dataset_cached(store_path, version_key)` that caches the open dataset keyed by store path + store mtime/version; invalidate when the store changes.
- **Tests affected:** new unit test (store changed → cache invalidated).

### J.3 Raster performance (Issue 1)

#### `services/api/src/api/services/tiles.py`
- **Why:** the core raster slowness.
- **Change:**
  - Vectorize longitude alignment (replace `np.vectorize` with `np.mod`/`np.where`).
  - Vectorize the color mapping (replace the per-pixel loop + `_interpolate_color` with a NumPy LUT + `np.take`); keep the exact stop interpolation and no-data transparency.
  - Add a server-side tile cache (process-local LRU or Redis) keyed by `(model, variable, level, zoom, x, y, lead_time_hours, initial_time)` storing the PNG bytes with a TTL ~ the tile `Cache-Control` (300 s).
  - Use the kept-open dataset cache (J.2 `core/zarr.py`) instead of `read_dataset` per request where safe.
- **Tests affected:** `test_tiles.py` — existing correctness tests must still pass (differential test: vectorized output == per-pixel output); add cache-key-leakage test (two cycles → distinct cache entries); add repeat-request cache-hit test.

#### `services/api/src/api/core/png.py`
- **Why:** possibly optimize encoding if needed.
- **Change:** no change required (measured ~2 ms). Keep as-is unless profiling shows otherwise.

### J.4 Location autocomplete (Issue 3)

#### `services/api/src/api/services/search.py` (or new `services/places.py`)
- **Why:** add provider-backed place autocomplete to `/v1/search`.
- **Change (new `api/services/places.py`):** `PlaceAutocompleteProvider` interface + `GooglePlacesAutocompleteProvider` (calls Places API (New) Autocomplete + Place Details, session-token aware) + `MapboxGeocodingProvider` (alternative). Add `suggest`/`resolve`. Wire `/v1/search` to call the provider when a new `type=place` (or a `provider` param) is requested; keep the existing PostGIS search for `city`/`resort`/`station`.
- **Tests affected:** new unit tests with a mocked provider (no live Google); `test_search.py` — add `type=place` contract tests with the mock.

#### `services/api/src/api/routers/search.py`
- **Why:** expose place autocomplete + resolution.
- **Change:** add `type=place` handling; add `place_id`/`session_token` params; extend `SearchResultOut` with an optional `place_id` field.
- **Tests affected:** `test_search.py`.

#### `services/api/src/api/schemas.py`
- **Why:** additive, non-breaking fields for place identity.
- **Change:** add `place_id: str | None = None` to `SearchResultOut`.
- **Tests affected:** schema tests.

#### `services/api/src/api/core/config.py`
- **Why:** hold the provider key/credentials server-side.
- **Change:** add `GOOGLE_PLACES_API_KEY`, `GOOGLE_PLACES_API_BASE`, optional `MAPBOX_TOKEN`, `SEARCH_PROVIDER` (default `google`), `PLACES_MIN_QUERY_LENGTH` (2), `PLACES_SESSION_TTL`.
- **Tests affected:** config import smoke tests.

#### `services/frontend/src/lib/api/client.ts` + `types.ts`
- **Why:** expose `type=place` + `place_id`/`session_token` on the search client.
- **Change:** extend `searchLocations` params and `SearchResult` type with `place_id`/`session_token`.
- **Tests affected:** `client.test.ts`.

#### `services/frontend/src/lib/forecast/selection.ts`
- **Why:** resolve a place suggestion to a `SelectedLocation` (lat/lon) before updating the map.
- **Change:** add `resolvePlaceToSelectedLocation` (calls `/v1/search` place-resolution or a new `/v1/places/:id`), and wire it into `LocationSearch` selection.
- **Tests affected:** `selection.test.ts`.

#### `services/frontend/src/hooks/useSearch.ts` / `components/search/LocationSearch.tsx`
- **Why:** min query length, session token, place resolution on selection.
- **Change:** raise `MIN_QUERY_LENGTH` to 2; generate/attach a session token per search session; on selection, resolve the place (via the new client call) before `onSelect`. Keep the existing debounce/abort/stale-guard.
- **Tests affected:** `LocationSearch.test.tsx`, `useSearch.test.ts`.

### J.5 Elevation (Issue 4)

#### `services/api/src/api/services/elevation.py` (new)
- **Why:** the provider abstraction + cache.
- **Change:** `ElevationProvider.get_elevation(lat, lon) -> float | None`; `DEMElevationProvider` (Copernicus GLO-30 or SRTM COG, bilinear, per-tile LRU); optional `GoogleElevationProvider` (same interface, not initial). `RoundedElevationCache` keyed on 3-decimal rounded coords.
- **Tests affected:** new unit tests (mock the DEM; no network).

#### `services/api/src/api/services/point_forecast.py`
- **Why:** populate `elevation_m` for raw coordinates and cities.
- **Change:** `resolve_location` calls the `ElevationProvider` for the resolved lat/lon when the record does not already define an elevation; set `ResolvedLocation.elevation_m`.
- **Tests affected:** `test_points.py` — add elevation-resolution tests (provider mocked).

#### `services/api/src/api/core/config.py`
- **Why:** elevation provider config.
- **Change:** add `ELEVATION_PROVIDER` (default `dem`), `DEM_COG_PATH`/`DEM_TILE_BUCKET`, `ELEVATION_CACHE_MAX`.
- **Tests affected:** config smoke tests.

#### `services/frontend/src/components/forecast/SelectedLocationSummary.tsx`
- **Why:** show `unavailable` instead of `—` when null.
- **Change:** render "unavailable" (localized) when `elevation_m` is null.
- **Tests affected:** `SelectedLocationSummary` (or dashboard) test.

### J.6 Ensemble capability (Issue 5)

#### `services/frontend/src/components/forecast/ForecastDashboard.tsx`
- **Why:** the misleading "Ensemble Statistics (GFS)" heading.
- **Change:** render the ensemble section **only** when `selectedModelIsEnsemble`; when deterministic, render no ensemble section (capability-driven). For an ensemble model with no data, show the distinct "Ensemble data is not yet available for this forecast." empty state.
- **Tests affected:** `ForecastDashboard.test.tsx` — add assertions: deterministic model → **no** "Ensemble Statistics" heading; ensemble model → heading present; ensemble model with no data → distinct message.

### J.7 Docs

#### `docs/API.md`
- **Why:** document `initial_time` on `/v1/maps` (GAP-8), `type=place` on `/v1/search`, `place_id` on search results, optional `initial_time` on `/v1/ensembles`/`/v1/probabilities`, and the `elevation_m` population.
- **Tests affected:** n/a.

#### `docs/DATABASE.md`
- **Why:** document the run-id version-scoping and (optionally) an elevation cache table.
- **Tests affected:** n/a.

#### `docs/ARCHITECTURE.md` / `docs/TESTING.md`
- **Why:** document the new providers (places, elevation) and testing rules (mock, no live Google/DEM network).
- **Tests affected:** n/a.

---

## K. Test Plan

### K.1 Ingestion

| Test | File | Verifies |
|---|---|---|
| Same model + same cycle + multiple leads succeeds | `test_pipeline.py` | `_merge_lead` merges leads 6/12 into one store (already present — extend to `time`-carrying datasets) |
| **Same model + different cycle stays isolated** | `test_pipeline.py` (new) | merging a 12Z file into a 00Z store raises `CycleStoreMismatchError`; the 00Z store is **unchanged** |
| **Different models stay isolated** | `test_pipeline.py` (new) | GFS 00Z and GEFS 00Z resolve to distinct stores |
| **Cycle/store mismatch fails fast** | `test_pipeline.py`, `test_cli.py` (new) | a `--store` that doesn't match the requested cycle → `CycleStoreMismatchError` before any write |
| **Duplicate lead behavior is deterministic** | `test_pipeline.py` | re-ingesting lead 6 replaces it (already present — keep) |
| **Batch lead ingestion works** | `test_cli.py` (new) | `--lead-time-hours 0 6 12` → one run, leads [0,6,12] |
| **Multi-cycle ingestion works** | `test_cli.py` (new) | `--cycle-hour 00 12` → two runs, two stores |
| **Multi-model ingestion works** | `test_cli.py` (new) | `--model gfs gefs` → two runs, two stores |
| **Partial failure does not corrupt completed runs** | `test_cli.py` (new) | a failing lead in one cycle does not affect another cycle's store |
| **S3 paths reflect forecast identity** | `test_cli.py` (new) | `derive_store_path` produces `{model}/{date}/{hour}/cycle.zarr`; `--store` mismatch rejected |
| Run-id version-scoped | `test_catalog.py` (new) | two versions of the same model at the same cycle → distinct run ids, no PK collision |

### K.2 Raster performance / correctness

| Test | File | Verifies |
|---|---|---|
| Raster identity includes model/cycle/lead/variable | `test_tiles.py` (new) | tile cache key carries the full identity tuple |
| **Cache cannot leak across forecast runs** | `test_tiles.py` (new) | two `initial_time` cycles → distinct cache entries |
| Repeated identical requests benefit from caching | `test_tiles.py` (new) | second request served from cache (hit) |
| Existing raster correctness unchanged | `test_tiles.py` | existing tests still pass; differential test: vectorized output == per-pixel output |
| Tile cache key includes zoom/x/y | `test_tiles.py` (new) | different tile coords → distinct keys |

### K.3 Location autocomplete

| Test | File | Verifies |
|---|---|---|
| Partial input (e.g. `den`) produces suggestions via the provider mock | `test_search.py` / `test_places.py` (new) | provider mock returns ranked suggestions; `/v1/search?q=den&type=place` returns them |
| **Stale autocomplete responses cannot overwrite newer searches** | `useSearch.test.ts` | existing abort/stale-guard tested; add a resolve-step guard |
| Selecting a result updates canonical coordinates | `LocationSearch.test.tsx`, `selection.test.ts` | place resolution → `SelectedLocation` lat/lon |
| No-results state works | `LocationSearch.test.tsx` | existing empty state |
| Provider errors degrade gracefully | `test_search.py` / `LocationSearch.test.tsx` | provider 500/network error → combobox error state |
| Tests do not require live Google credentials | `test_places.py` | provider mocked; no live network |

### K.4 Elevation

| Test | File | Verifies |
|---|---|---|
| Selected coordinates trigger elevation resolution | `test_points.py` (new) | `/v1/points?lat&lon` returns non-null `elevation_m` (provider mocked) |
| Returned elevation is explicit in meters | `test_points.py` | `elevation_m` is a number of meters |
| Elevation lookup failure handled gracefully | `test_points.py` | provider `None`/error → `elevation_m: null`, forecast still returns |
| Cache behavior is deterministic | `test_elevation.py` (new) | same rounded coord → same cached value |
| Coordinates with no terrain value do not crash the UI | `SelectedLocationSummary` test | `null` → "unavailable" (not `—`, not crash) |
| External provider is mockable | `test_elevation.py` | `DEMElevationProvider` mocked; no DEM network read |

### K.5 Ensemble capability

| Test | File | Verifies |
|---|---|---|
| GFS is not presented as ensemble | `ForecastDashboard.test.tsx` (new) | deterministic model → **no** "Ensemble Statistics" heading |
| GEFS is recognized as ensemble-capable | `ForecastDashboard.test.tsx` | ensemble model → heading present |
| "unsupported model capability" and "ensemble data unavailable" are distinct states | `ForecastDashboard.test.tsx` | deterministic (no section) vs ensemble-no-data (distinct message) |
| Existing ensemble calculations remain correct | `domain/tests/test_ensemble_*.py` | unchanged |

---

## L. Migration / Compatibility Plan

### L.1 Existing S3 stores

The target layout (`s3://weather-data/{model}/{date}/{hour}/cycle.zarr`) **matches the current convention** (the CLI docstring already uses it). Existing stores are compatible — the change is that the path becomes *derived and validated* rather than manual. No store relocation is required.

### L.2 Zarr datasets

- Existing stores lack an explicit `cycle_time` attr. The fix adds the attr **on new writes**; for **legacy stores**, the `time` coordinate (kept by the parser) is the fallback identity source. `_merge_lead` should validate against `time` first, then the `cycle_time` attr if present. No rewrite of existing stores is required for correctness (the merge guard works off `time`); adding the attr to legacy stores is optional and safe.
- **Back-compat note:** if a legacy store already contains mixed-cycle data (contaminated by the old silent bug), a new-cycle merge would fail fast — this is **desirable** (it surfaces the contamination) but may require a manual re-ingest of the affected cycle. Flag this in the rollout.

### L.3 DB records

- **Run-id version-scoping (GAP-4):** changing `run_{cycle_time}_{model_id}` → includes version. This is a **write-path** change (new runs get the new id format). Existing run rows keep their old ids. The unique constraint `(model_version_id, cycle_time)` is unchanged. No migration is strictly required; a new migration is only needed if we add an elevation cache table (below).
- **Elevation cache table (optional):** if a DB cache for elevation is added, create an Alembic migration (`002_elevation_cache.py`) — additive, no breaking change.

### L.4 API compatibility

- **No breaking changes to `/v1/`.** All additions are optional query params (`type=place`, `place_id`, `session_token`, `initial_time` on ensembles/probabilities) or additive response fields (`place_id`, populated `elevation_m`). Per API.md section 1.3, additive params/fields are allowed.
- **`elevation_m`** was already nullable; populating it for coordinates/cities is a value change, not a schema change — backward-compatible.
- **Cache-key change (GAP-1):** keys gain run identity; this only affects cache entries (Redis), which are ephemeral. No client-visible change.

### L.5 Frontend compatibility

- `SearchResultOut` gains `place_id` (additive). `SelectedLocation`/`SearchResult` TS types gain `place_id` (additive).
- `MIN_QUERY_LENGTH` 1 → 2 is a behavior change for single-char queries (fewer/no results) — intentional per provider cost/best-practice; verify no test depends on 1-char matching.
- `SelectedLocationSummary` null elevation renders "unavailable" instead of `—` — a UI text change; update any snapshot/test.
- `ForecastDashboard` ensemble heading change — behavior change; update dashboard tests.

### L.6 Configuration / environment / Docker

- New env vars (all optional with defaults): `GOOGLE_PLACES_API_KEY`, `GOOGLE_PLACES_API_BASE`, `MAPBOX_TOKEN`, `SEARCH_PROVIDER`, `PLACES_MIN_QUERY_LENGTH`, `PLACES_SESSION_TTL`, `ELEVATION_PROVIDER`, `DEM_COG_PATH`/`DEM_TILE_BUCKET`, `ELEVATION_CACHE_MAX`, `MAX_BATCH_RUNS`, `STORE_PATH_TEMPLATE`. Add to `.env.example` and the Dockerfiles (runtime env passthrough) as needed.
- **Google API configuration:** enabling the Places API (New) in the Google Cloud project + a server key (IP-restricted, API-restricted) is an operational prerequisite, not a code change.
- **Docker:** the API image needs the DEM COG path/bucket reachable (network or volume); if a local DEM is bundled, it must be added to the image (size tradeoff — see Section N).

---

## M. Implementation Order

Dependency-aware sequence (each step lands independently and is testable):

1. **Forecast identity / storage invariants (GAP-3, GAP-5, GAP-6):** parser `cycle_time` attr, `_merge_lead` store-identity guard + grid-schema validation, atomic store write. Tests: wrong-cycle merge fails fast; same-cycle multi-lead works; legacy `time`-based validation.
2. **Ingestion correctness (GAP-4):** run-id version-scoping. Tests: two-versions-same-cycle distinct ids.
3. **Batch ingestion (Issue 2B):** `derive_store_path`, CLI repeatable/zipped lists, `--manifest`, `--dry-run`, `--max-runs`, per-run failure isolation. Tests: multi-lead/multi-cycle/multi-model/manifest/anti-Cartesian.
4. **API / cache identity (GAP-1, GAP-2, GAP-8):** cache keys gain run identity; optional `initial_time` on ensembles/probabilities; document `/v1/maps` `initial_time`. Tests: cache-key-differs-by-cycle.
5. **Raster performance (Issue 1):** vectorize color + lon, server tile cache (identity-correct key), kept-open dataset cache. Tests: differential correctness, cache-hit, cache-leakage.
6. **Location autocomplete (Issue 3):** provider abstraction + Google Places backend, `/v1/search` `type=place`, session tokens, frontend selection resolution. Tests: provider mocked, no live Google.
7. **Elevation (Issue 4):** `ElevationProvider` + DEM, `resolve_location` population, rounded-coord cache, UI `unavailable`. Tests: provider mocked.
8. **Ensemble capability / UI (Issue 5):** dashboard heading fix + distinct empty states. Tests: deterministic → no heading; ensemble-no-data → distinct message.
9. **Integration / E2E validation:** run all suites (pytest, Jest, Playwright), the CI gates, and a manual acceptance pass against the acceptance criteria (Section O).

> Rationale: identity/storage invariants (1) must precede ingestion correctness (2) and batch (3), because batch semantics depend on correct store identity and atomicity. Cache identity (4) must precede raster caching (5) so caches never leak. Autocomplete (6) and elevation (7) are independent of the forecast data path and can proceed in parallel once the identity work (1–4) is stable. Ensemble UI (8) is a small, independent fix. Integration (9) is the final gate.

---

## N. Risks / Open Decisions

Only genuine unresolved decisions are listed; each includes the recommended choice.

1. **Place provider: Google Places API (New) vs Mapbox/OSM.**
   - Google offers the cleanest session-token billing + ranked place resolution, but introduces a billing dependency and the volatile pricing figures. Mapbox pairs naturally with MapLibre but is an added paid dependency; Nominatim/Photon are free but rate-limited and ODbL-attribution-bound.
   - **Recommendation:** Google Places API (New) proxied through FastAPI, with a provider abstraction (so Mapbox/OSM is a drop-in). Confirm the live pricing before committing (research was rate-limited on the price page).

2. **Elevation source: Copernicus GLO-30 local DEM vs AWS Terrarium/Skadi on-demand vs Google Elevation.**
   - Local GLO-30 is truly global, accurate, cache-friendly, but adds storage (100–200 GB COG or a per-tile LRU). AWS Terrarium/Skadi is free/no-auth but SRTM-only (no poles) and network-dependent. Google Elevation is easy but ToS restricts caching (poor for a cache-first product).
   - **Recommendation:** local/server-side **Copernicus GLO-30** (bilinear, per-tile LRU, precompute known stations/cities at ingest), with AWS Skadi as a no-storage fallback. Google Elevation reserved as a future fallback behind the same `ElevationProvider` interface.

3. **`/v1/ensembles` + `/v1/probabilities` cycle pinning: add `initial_time` param?**
   - Adding it makes these endpoints cycle-consistent with `/v1/maps` and fixes GAP-2. It's additive (non-breaking). The cost is a slightly wider API surface.
   - **Recommendation:** add the optional `initial_time` param and fold the resolved `cycle_time` into cache keys (fixes GAP-1 and GAP-2 together).

4. **Server-side tile cache: process-local LRU vs Redis.**
   - Redis scales across API workers and matches the existing cache layer; a process-local LRU is simpler and sufficient for a single-instance MVP.
   - **Recommendation:** start with a process-local LRU (simplest, correct), and move to Redis only if multi-worker scaling is needed. Either way the cache key must be identity-complete.

5. **Kept-open dataset cache invalidation.**
   - Caching the open Zarr dataset per run risks serving stale data after a re-ingest. The safe invalidation is keying on store path + store version/mtime, with a fresh-open fallback.
   - **Recommendation:** implement the version-keyed dataset cache (J.2 `core/zarr.py`) with TTL + store-version invalidation, and keep `read_dataset` as the fallback path on any uncertainty.

6. **Batch CLI: repeatable/zipped flags vs manifest-only.**
   - Zipped flags are convenient for common cases but risk ambiguity with multiple models×cycles×hours. A manifest is unambiguous but heavier for simple jobs.
   - **Recommendation:** support both — zipped-repeatable flags for common cases (with anti-Cartesian guards + `--dry-run` + `--max-runs`) and a manifest for complex jobs.

7. **Legacy contaminated stores (the old silent-merge bug).**
   - If a store already contains mixed-cycle data from the previous behavior, the new fail-fast guard will reject new merges — surfacing the contamination. Whether to auto-repair (re-ingest the cycle) or leave to operators is a rollout decision.
   - **Recommendation:** fail fast on detection, document it, and re-ingest the affected cycle cleanly (the CLI's per-run idempotency makes this safe).

8. **DEM storage footprint.**
   - Bundling a full global GLO-30 COG (~100–200 GB) in the API image is heavy. The lightweight option is per-tile on-demand fetch + LRU (tens of MB).
   - **Recommendation:** start with per-tile on-demand fetch + LRU (no image bloat); precompute known stations/cities into the DB at ingest to minimize runtime lookups.

---

## O. Acceptance Criteria

Objective, testable criteria for the final validation pass.

### O.1 Ingestion (Issue 2, 2B)

- [ ] `_merge_lead` merges same-model + same-cycle + multiple leads into one store; `forecast_products` has one row per (var × lead).
- [ ] A different-cycle file merged into an existing store raises `CycleStoreMismatchError` **before any write**; the store is byte-identical afterward.
- [ ] GFS 00Z and GFS 12Z are distinct stores and distinct runs; serving never mixes them.
- [ ] GFS and GEFS are distinct stores and distinct runs.
- [ ] A `--store` that does not match the requested `--model/--cycle-date/--cycle-hour` fails fast (no silent contamination; no `compat="override"`).
- [ ] Re-ingesting the same lead is deterministic (replaces, does not duplicate).
- [ ] Batch lead ingestion (`--lead-time-hours 0 6 12`) produces one run with leads [0,6,12].
- [ ] Multi-cycle (`--cycle-hour 00 12`) and multi-model (`--model gfs gefs`) ingestion produce the expected run/store sets with no accidental Cartesian product (verified via `--dry-run`).
- [ ] A failing lead/cycle in a batch does not corrupt already-completed runs.
- [ ] `derive_store_path` yields `{model}/{cycle_date}/{cycle_hour}/cycle.zarr`; explicitly-supplied mismatched paths are rejected (or require `--allow-custom-store`).
- [ ] Two versions of the same model at the same cycle produce distinct run ids (no PK collision).

### O.2 Raster performance / correctness (Issue 1)

- [ ] `render_tile_png` produces byte-identical PNGs to the current implementation on the existing fixture grids (differential test).
- [ ] Tile cache key includes `(model, variable, level, zoom, x, y, lead_time_hours, initial_time)`.
- [ ] Identical tile requests hit the cache (2nd request substantially faster than the 1st).
- [ ] Two `initial_time` cycles of the same model never share a tile cache entry.
- [ ] End-to-end tile latency on a GFS-scale store is < ~50 ms/tile (from ~230–318 ms today).
- [ ] `/v1/maps` metadata and tile endpoints return correct `Cache-Control` headers.

### O.3 Location autocomplete (Issue 3)

- [ ] Typing `den` returns ranked place suggestions (Denver, Denver International Airport, …) via the provider mock — no live Google in tests.
- [ ] A stale autocomplete response cannot overwrite a newer search (abort + stale-guard).
- [ ] Selecting a suggestion resolves the canonical place (name + lat/lon + region/country) and updates the map/forecast.
- [ ] No-results and provider-error states render gracefully (no crash).
- [ ] Session-token semantics are correct (one token per search session; reused for autocomplete + place resolution).
- [ ] The Google key never appears in the browser bundle or client code.
- [ ] `/v1/search?type=place` is backward-compatible with the existing `city`/`resort`/`station` search.

### O.4 Elevation (Issue 4)

- [ ] Resolving a location (autocomplete or map click) triggers elevation resolution; `/v1/points` returns a non-null `elevation_m` in meters for land coordinates.
- [ ] Elevation is returned as an explicit `elevation_m` field (already in the schema), never as frontend-only state.
- [ ] Elevation lookup failure / no-data (ocean) returns `elevation_m: null` and does not crash the forecast UI; the UI renders `unavailable` (not `—`).
- [ ] The elevation cache is deterministic (same rounded coordinate → same value).
- [ ] The external provider is mocked in tests (no live DEM/network).
- [ ] The `ElevationProvider` abstraction is in place (GoogleElevationProvider/DEMElevationProvider interchangeable).

### O.5 Ensemble capability (Issue 5)

- [ ] Selecting GFS (deterministic) does **not** show an "Ensemble Statistics (GFS)" panel.
- [ ] Selecting GEFS (ensemble) shows the ensemble panel with the correct heading.
- [ ] "This model is not an ensemble model" (no panel) and "ensemble data unavailable" (panel with distinct message) are clearly distinct states.
- [ ] Existing ensemble math and `/v1/ensembles` behavior are unchanged (all existing ensemble tests pass).

### O.6 Cross-cutting identity

- [ ] Point/probability/ensemble cache keys include the resolved run identity (`cycle_time`), so two cycles of the same model never share a cached response.
- [ ] `/v1/maps` `initial_time` is documented in API.md.
- [ ] Zarr stores carry a `cycle_time` attr (new writes) and `_merge_lead` validates against it (with `time` fallback for legacy stores).
- [ ] The full pipeline — CLI → parse → Zarr → S3 → DB → API → cache → frontend — preserves `(model, cycle_time, lead_time_hours, variable)` end-to-end; no layer collapses to `(model, lead, variable)`.

### O.7 Regression / CI

- [ ] All existing pytest, Jest, and Playwright suites pass.
- [ ] Ruff + mypy pass for the changed packages.
- [ ] The CI gates (domain/api/ingestion/frontend/container-builds) pass.
- [ ] No breaking changes to the `/v1/` API contract (all additions are optional params/fields).

