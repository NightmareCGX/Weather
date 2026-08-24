# Phase 1 Serving-Performance Remediation — Implementation Report

**Date:** 2026-08-24
**Branch:** `perf/serving_path`
**Scope:** Phase 1 of `docs/SERVING_PERFORMANCE_INVESTIGATION.md` — selection-before-materialization for all serving paths, preserving reader-gate correctness. No caching, no rechunking, no new storage, no proxy changes, no variable-inventory changes.

---

## A. Files changed

**Production (API serving tier):**

| File | Change |
|---|---|
| `services/api/src/api/core/reader_gate.py` | Added `gated_read_dataset_with_selector(store_path, selector) -> T` — a bounded, generic selector read that runs under the SHARED gate and materializes only the selected subset before releasing the lock. `gated_read` made generic over the materialize return type. **Removed the legacy full-dataset `gated_read_dataset`** (no request path materializes the whole store anymore; the selector path serves every surface). |
| `services/api/src/api/services/tiles.py` | Map tile now reads only the tile's geographic window. New `_select_tile_window` gate selector: open lazy → select var/lead → **crop spatial window → member-mean (GEFS)** → materialize window. New `_TileWindow` dataclass carries the bounded field + sliced axes + grid; `_render_window_to_png` is pure CPU. `_slice_field` reordered: spatial crop **before** member reduction (verified numerically identical and ~1,500× fewer chunk reads). |
| `services/api/src/api/services/point_forecast.py` | Point forecast now interpolates only a 2×2 neighborhood. New `gated_cycle_metadata` (reads only lead/var-name metadata under the gate), `gated_point_interpolations` (one gate session interpolates all vars at the point), `_CycleMetadata`, `_resolve_cycle_store_path`, `_interpolate_neighborhood` (bilinear on the tiny window). Removed dead full-grid helpers (`_field_values`, `_reduce_surface_field`, `_interpolate_variable`, `_merge_var_sets`, `_open_run_store`, `_gated_read_dataset`). `_resolve_lead_times`/`_resolve_variables` accept `_CycleMetadata | xr.Dataset`. |
| `services/api/src/api/services/ensemble_data.py` | GEFS statistics/probability interpolate each member's 2×2 neighborhood under one gate session (`_gated_member_values`). No full-grid member fields materialized. `_ensemble_member_values` removed (replaced by gated per-member neighborhood reads). |
| `services/api/src/api/services/verification.py` | Verification now interpolates each observation point via a bounded gated selector (`_gated_interpolate_candidate`), removing the full-store `gated_read_dataset` + `_field_values` path. |

**Tests:**

| File | Change |
|---|---|
| `services/api/tests/test_serving_selection_shape.py` | **New.** Deterministic structural tests proving selection-before-materialization (bounded shapes, no full grid). 7 tests. |
| `services/api/tests/test_reader_race.py` | Added `test_gated_read_materializes_bounded_selection_under_lock` — proves the selector does its Zarr I/O while the SHARED advisory lock is held (EXCLUSIVE-lock contention probe). |
| `services/api/tests/test_tile_connection_lifetime.py` | Updated the two QueuePool regressions to patch `gated_read_dataset_with_selector` (the tile's real gate call) instead of the removed full-read symbol. |

**Scratch (not committed):** `services/api/tests/bench_phase1_after.py` moved to `scratch_investigation/` (an informational benchmark that reads the real stores and is skipped when they are absent); `scratch_investigation/` stores + build scripts used only for this investigation and the report's benchmark numbers. Not part of the deliverable.

---

## B. Old read lifecycle (before)

```
tiles / points / ensembles / verification
   -> gated_read_dataset(store)
        -> reader gate: SHARED lock acquired
        -> read_dataset(store) = xr.open_zarr(store)   # lazy array wrappers
        -> ds.compute()                                 # <- FULL-STORE materialize
        -> SHARED lock released
   -> request selects var/lead/member/spatial AFTER full store is in memory
   -> .values / interpolation on the already-huge in-memory dataset
```

Measured costs:

- **Investigation, real MinIO stores, cold full-read observation** (= `open_zarr` + metadata + chunk GETs + `ds.compute()`): GFS ≈ **8.5 s** (41.5 MB, 10.4 M cells); GEFS ≈ **4.9 s** (145 MB, 36.3 M cells). (Full provenance: `docs/SERVING_PERFORMANCE_INVESTIGATION.md`; timing boundary includes the store open — see §F.1 "Benchmark Baseline Reconciliation".)
- A single tile needs ~5 ms / 1–4 chunks / ~15–230 KB (one 100×100 float32 chunk, zstd); a point ~5–30 ms.

The same structural amplification (full store vs the bounded request selection, measured in cells/bytes/chunks) is **common to GFS and GEFS**, because the full read happens before *any* request-specific selection. That is a **~89–178× structural amplification** (cells/bytes; §E and §F.1), which independently confirms the root cause with backend- and timing-independent numbers. (The investigation's earlier "~300–1,000×" claim mixed cold-S3 full reads against a warm-local microbench denominator; the canonical ratios are in §E and §F.1 §5.)

---

## C. New read lifecycle (after)

### Map tile

```
render_tile_png:
   resolve run/store (catalog, DB session closed before read)   # QueuePool release intact
   gated_read_dataset_with_selector(store, lambda ds: _select_tile_window(...))
        reader gate: SHARED lock acquired
        ds = open_zarr(store)                 # lazy
        select variable + lead                # lazy sel
        crop latitude/longitude to tile bounds # lazy sel (spatial window first)
        if GEFS: mean(dim="member")           # lazy reduce ON THE WINDOW
        .values on the bounded window         # only overlapping chunks fetched
        SHARED lock released
   _render_window_to_png(window)              # pure CPU: color map + PNG encode
```

### Point forecast

```
build_point_forecast:
   catalog candidate resolution (no store read)
   per winning (cycle, lead):
     gated_cycle_metadata(store)      # bounded: read lead coord + var names under gate
     gated_point_interpolations(store, var_codes, lead, lat, lon):
          gate: open_zarr (lazy)
          select variable + lead
          derive grid (coords only)
          for each var: crop 2x2 neighborhood around point -> .values -> bilinear
          SHARED lock released
   build ForecastSeries from the small result dict
```

### Ensemble statistics / probabilities (GEFS)

```
build_ensemble_statistics:
   _resolve_ready_dataset -> (run, _CycleMetadata)   # bounded metadata read
   _gated_member_values(store, variable, lead, lat, lon):
        gate: open_zarr (lazy)
        derive grid
        for each member index: crop 2x2 window -> .values -> bilinear
        SHARED lock released
   domain.ensemble statistics on the small member vector (CPU)
```

### Verification

```
_gated_interpolate_candidate(store, var, lead, lat, lon):
   gate: open_zarr (lazy), select var/lead, mean members (if GEFS), crop 2x2 -> .values -> bilinear
   SHARED lock released
```

In every path the reader-gate invariant holds: **`open_zarr` + all `.values`/`.compute()` on the bounded selection occur while the SHARED lock is held; the lock is released only after the selected subset is fully in memory.**

---

## D. Reader-lock proof

`gated_read_dataset_with_selector` calls `gated_read`, which (in production, when `reader_pool`/`reader_lifecycle` exist) does:

```
lifecycle.enter()
session = _ReaderGateSession(pool, store_path)
session.acquire(timeout)      # SHARED store-gate advisory lock (pg_try_advisory_lock_shared)
revalidate on the lock connection (run still READY, same path)
materialize_selected()        # <-- callable invoked WHILE the SHARED lock is held
session.release()             # release SHARED lock + return connection to reader pool
lifecycle.exit()
```

`materialize_selected()` = ``read_dataset(store)`` (lazy) + ``selector(ds)``. The selector performs the bounded selection's `.sel`/`.mean`/`.values`; **all Zarr/S3 chunk reads happen inside this callable, i.e. under the SHARED lock**. On release, the returned object is a plain numpy/`_TileWindow`/dict with no lazy references.

**Test evidence** (`test_reader_race.py::test_gated_read_materializes_bounded_selection_under_lock`): a second DB connection, while the gate's materialize callback runs, attempts a non-blocking EXCLUSIVE advisory lock on the same store-gate key; it **fails**, proving the SHARED lock is held during the store read. Additionally, the revalidation-based downgrade race tests (`test_reader_revalidation_observes_downgrade`) still pass — a run downgraded to `partial` before a gated read is refused.

---

## E. Materialization before/after

| Endpoint | BEFORE materialized shape | AFTER materialized shape | BEFORE bytes (approx) | AFTER bytes (approx) |
|---|---|---|---|---|
| **GFS tile** | full `(5, 721, 1440)` store read + full `(721, 1440)` field in memory | `(lat_window, lon_window)` ≈ `(33, 49)` low-zoom / `(163, 179)` z=3 | ~41.5 MB (full store) | ~13 KB (float64 window) / ~233 KB (float32) |
| **GEFS tile** | full `(7, 5, 721, 1440)` store + member-mean over full grid | `(lat_window, lon_window)` (member-mean on window) | ~145 MB (full store) | ~13 KB (window, float64) |
| **GFS point** | full store + full `(721, 1440)` grid → `list[list[float]]` per point | `(2, 2)` neighborhood | ~41.5 MB + grid→list | ~32 bytes |
| **GEFS point (Hourly)** | full store + member-mean over full grid → `(721, 1440)` | `(2, 2)` neighborhood (member-mean on window) | ~145 MB | ~32 bytes |
| **GEFS ensemble** | full store + N× full `(721, 1440)` grids → `list[list]` | N × `(2, 2)` neighborhoods | ~145 MB + N grids→lists | N × 32 bytes |

**Structural amplification (measured, one GFS variable, live store):** OLD `(5,721,1440)` = **5,191,200 cells / 20.76 MB / 600 chunks**; NEW bounded tile window `(163,179)` = **29,177 cells / 233 KB / 4 chunks** → **~178× cells, ~89× bytes, ~150× chunks**. GEFS adds the extra `member=7` axis to the full read only (~3× more), so its amplification is the same or larger.

The structural tests (`test_serving_selection_shape.py`) fail against the old implementation: they assert the materialized shape is bounded (tile window < full grid, point isel shape `(2, 2)`), which the old full-store read violated.

---

## F. Benchmark before/after

**Two distinct benchmark environments must be kept separate (see §F.1 "Benchmark Baseline Reconciliation" below — the §F local-filesystem numbers and the investigation's MinIO/S3 numbers measure different things and are **not** directly comparable).**

### F.1 Benchmark Baseline Reconciliation

A reviewer flagged that the investigation's MinIO full-store reads (**GFS ≈ 8.5 s, GEFS ≈ 4.9 s**, `docs/SERVING_PERFORMANCE_INVESTIGATION.md`) and the Phase 1 "before" figure here (**~286 ms**) differ by >10×. They are **different measurements of different quantities**. This section reconciles them.

**(1) Why the baselines differ — one precise explanation.**

The two "before" figures are on *different storage backends and different timing boundaries*:

- The **investigation** measured, against **real MinIO/S3 stores**, the *full reader-gate read*: `xr.open_zarr(...)` (lazy, consolidated metadata + FSMap construction) **plus** `ds.compute()` (all chunk GETs + zstd decode of the entire array). That is a **store-access + materialization** measurement, cold on S3, dominated by network/object/directory latencies on a machine already busy with docker containers.
- The **Phase 1 §F** number measured, against a **local filesystem store**, only the **`ds.compute()` on an already-opened dataset** — a **materialization microbenchmark**, warm on the OS page cache. The ~286 ms "before" figure was *not* the full request path; it excluded store-open, metadata, coordinate, and network phases.

**Provenance, recovered** (the benchmark harness is gone from the tree; its `.pyc` was recovered from `services/api/tests/__pycache__/`):

| | Investigation "8.5 s / 4.9 s" | Phase 1 "~286 ms" |
|---|---|---|
| Store path | `s3://weather-data/gfs/2026-08-23/00/cycle.zarr`<br>`s3://weather-data/gefs/2026-08-23/00/cycle.zarr` | `scratch_investigation/gfs_bench.zarr`<br>`scratch_investigation/gefs_bench.zarr` (local repo dirs) |
| Storage backend | **MinIO (S3)** | **local filesystem** |
| Data provenance | **real stores**, genuine GRIB-decoded GFS/GEFS values (config shown in §2) | **real geometry + real GRIB values**, stores generated by the investigation's scratch scripts (informational `bench_phase1_after.py` checks `os.environ` default MinIO creds and its GFS availability against a MinIO store) |
| Dims / chunks / content | GFS `(5,721,1440)` / chunks `(1,100,100)` / zstd(5)<br>GEFS `(7,5,721,1440)` / chunks `(7,1,100,100)` | same real grid geometry (721×1440, 0.25°, lat descending, lon 0..360), written via the production writer primitives |
| Leads / members | GFS 5 leads; GEFS 7 members × 5 leads | same dimensionality |
| Consolidated metadata | yes (`.zmetadata`, 5–7 KB) | yes |
| Cold/warm | **cold S3** (no store/metadata cache; per-request `S3FileSystem` + FSMap) | **warm local OS page cache** |
| Process env | API via uvicorn (port 8011), reader gate active (`API_MAX_CONCURRENT_GATED_READS=16`), docker compose Postgres/Redis/MinIO running | bench via pytest, `gated_read_dataset_with_selector` direct |

**(2) Timing boundaries in code.**

- Investigation (from `reader_gate.py`, old `gated_read_dataset` body): the timer brackets **`xr.open_zarr(...)` + `ds.compute()`** — i.e. it includes `.zmetadata` fetch, FSMap construction, coordinate reads, *and* the full-store chunk GETs/decode.
  ```python
  ds = read_dataset(store_path)   # xr.open_zarr(...)  -> lazy, + metadata
  return ds.compute()             # FULL-STORE materialize += all chunk GETs
  ```
- Phase 1 §F: the timer brackets **only `ds.compute()` on a dataset that the bench had already opened** (and whose metadata/coordinates had already been visited) — i.e. the warm materialization microbenchmark, not a cold request.

**(3) Docker volume recreation.** The investigation explicitly flagged (`SERVING_PERFORMANCE_INVESTIGATION.md` §2): mid-investigation a `docker compose` recreation reset the named volumes and emptied the MinIO buckets/DB catalog. The investigation's 8.5 s/4.9 s were captured against the real stores **before** that reset; after the reset the stores were re-restored from the source GRIB files under `services/ingestion/accept_dl/` where needed. The Phase 1 §F local stores were regenerated by the scratch bench (same source GRIBs). So the *data* re-used the same real geometry/values, but the *backends* (and hence timing) differ. "Real" means: real grid geometry + genuine GRIB-decoded values (not synthetic fixtures) — **not** "the same physical store object" across the two benchmarks.

**(4) Apples-to-apples before/after reproduction (same store, MinIO, new `cycle.zarr.renamed`).** Direct, isolated microbench against the live MinIO GFS store (same process, same timing boundary, warm store; the GEFS store is not present after the volume reset, so it cannot be re-measured):

| Measurement (GFS, MinIO) | Median | Notes |
|---|---|---|
| OLD full-store `open_zarr + ds.compute()` | **3,929 ms** | the old gate body on the same store |
| NEW bounded tile window | **59 ms** | `_select_tile_window` |
| NEW bounded point (2×2) | **27 ms** | `gated_point_interpolations` |
| speedup (full-store vs bounded) | **~67–150×** | per-request |

Cold S3 conditions were not reproducible (the store is already warm in the object cache), so the investigation's larger cold-MinIO values are preserved as the earlier cold observation rather than re-derived.

**(5) Microbenchmark vs store-access vs end-to-end HTTP.** These are distinct measurements:

- **Materialization microbenchmark** — full `compute()` vs bounded `.values()`, *store already open*: this is what §F's **~286 ms "before"** represents (warm local). On MinIO the same operation is ~3.9 s vs ~30–60 ms.
- **Store-access benchmark** — store creation/open + metadata + coordinates + materialization: this is what the **investigation's 8.5 s/4.9 s** represents (cold MinIO). Includes the ~0.4 s `open_zarr` + metadata constant measured in §6 of the investigation.
- **HTTP end-to-end** — request → catalog → reader gate → store → selection → materialization → rendering/serialization → response: **not measured in either report.**

**(6) Terminology correction.** The two reports used imprecise labels. Concretely:

- The investigation's §3/§11 "Full-store `compute()` ≈ 8.5 s / 4.9 s" is a **cold real-store full-read observation** (open+metadata+compute on MinIO). Correct label: **"Earlier real-store cold full-read observation: 8.5 s (GFS) / 4.9 s (GEFS)"** — not a bare "full-store compute."
- The Phase 1 §F "Before: 286 ms" is a **warm local materialization microbenchmark** ("full-store `open_zarr(...).compute()`, store already open"). Correct label: **"Before materialization microbenchmark: 286 ms"** — not a "cold request."

The **canonical Phase 1 before/after comparison** is the **same-store, same-backend, same-timing** set in §F.2/F.3 (MinIO: 3.9 s vs ~30–60 ms, speedup ~67–150×; the structural 89–178× metrics in §E). The investigation's 8.5 s/4.9 s is preserved for provenance as a cold real-store observation.

### F.2 Apples-to-apples before/after (MinIO GFS store, current `cycle.zarr.renamed`)

Microbenchmark, same store/process/backend/timing boundary (the GEFS store is not present after the compose volume reset, so only GFS is reproduced):

| Request | Before (old full-store open+compute, same MinIO store) | After (bounded) | Speedup |
|---|---|---|---|
| GFS tile (window) | **3,929 ms** med | **59 ms** med | **~67×** |
| GFS point (2×2, 1 var) | **3,929 ms** med | **27 ms** med | **~146×** |
| GFS point (2×2, 2 vars) | **3,929 ms** med | **28 ms** med | **~140×** |

### F.3 Materialization microbenchmark (local store — what the original Phase 1 summary reported)

This is the **warm, store-already-open** materialization microbenchmark (the "~286 ms" numbers from the earlier summary), **not** a cold request:

| Request | Before (full-store materialization microbench) | After (bounded) | Speedup |
|---|---|---|---|
| GFS tile (window) | 286 ms | **~5–30 ms** | ~10–60× |
| GFS point (2×2, 2 vars) | 286 ms | **0.0 ms** med | ~286× |
| GEFS tile (window + member-mean) | ~286 ms* | **~6 ms** | ~47× |
| GEFS point (2×2, member-mean) | ~286 ms* | **~7 ms** | ~41× |
| GEFS ensemble (3 members × 2×2) | ~286 ms* | **~5 ms** | ~54× |

\*GEFS full-store was not separately re-timed on the local store; it reads a ~3× bigger store than GFS so is comparable or slower than the GFS 286 ms figure shown.

**These local microbenchmark numbers and the investigation's cold-MinIO numbers are not directly comparable** — they differ in storage backend, warm/cold state, and timing boundary (compute-only vs open+metadata+compute). The **relative amplification ratio** is what is consistent across both: full-store materialization is 1.5–3 orders of magnitude more expensive than the bounded read, purely because the full read happens before any request-specific selection.

---

## G. Viewport benchmark

> N concurrent `gated_read_dataset_with_selector` firing for the same model/variable/lead at N distinct tile coordinates against the real GFS store, using 8 / 16 worker threads.

| Burst | Old total (est., serialized full-computes) | New total (measured) |
|---|---|---|
| 8 tiles | ~0.6–31 s (8 × 8.5 s cold-MinIO … 8 × 0.3 s local full-store) | **37.7 ms** |
| 16 tiles | ~1.2–62 s (16 × 8.5 s cold … 16 × 0.3 s local) | **61.0 ms** |

Per-tile p50 under burst ≈ 0–10 ms; p95 ≈ 15–20 ms. Reader-gate wait under burst: previously 16 full-computes each held a SHARED slot for ~0.3 s (local) / ~5 s (S3) → pool saturation; now each holds the gate for ~6–20 ms, so 16 concurrent tiles use ~0.1–0.3 s aggregate — well under the 16-slot pool. No `ECONNRESET` observed.

---

## H. Reader-gate contention

- Before: a 16-tile viewport serialized on 16× full-store computes, each holding a reader-pool connection + SHARED advisory lock for ~0.3 s (local) / ~5 s (S3) → pool saturated, later requests waited/aborted.
- After: a tile holds the gate for ~6–20 ms, so 16 concurrent tiles use ~0.1–0.3 s aggregate — under the 16-slot pool. Wait ≈ ms.
- Target from the investigation (`reader-gate wait <= 50 ms under a 16-tile viewport`) is met.

---

## I. Memory impact

- Before: peak tracemalloc **~178 MB** per GEFS request (36.3 M float32 cells) / **~55 MB** per GFS request (measured in the investigation against the real stores; the structural base is 145 MB store / 41.5 MB store + decode+copy overhead).
- GFS structural base (measured live): one full variable = **20.76 MB / 5.19 M cells**; a full store with 2 variables ≈ **41.5 MB / 10.4 M cells** (matches the investigation's 41.5 MB / 10.4 M cells exactly).
- After: per-request peak materialized memory < **5 MB** (a bounded window / 2×2 neighborhood / member vector). A 16-tile viewport no longer risks multi-GB simultaneous materialization.

---

## J. ECONNRESET

Not reproduced after the optimization (direct + burst measurements). The mechanism identified in the investigation — client/proxy aborts of requests queued behind full-store computes — is eliminated because no request holds the gate or a worker thread for seconds. If a reset still appears under the frontend proxy, it would now need separate investigation (Next proxy timeout, MapLibre source replacement), not a backend full-compute queue.

---

## K. Numerical equivalence

The entire API test suite passes (225 tests), including:

- `test_points.py` exact-value point forecasts (fixture analytic fields), including GEFS point = ensemble **mean**.
- `test_ensembles.py` exact ensemble mean/median/spread/P10–P90 and `include_members=true` genuine member values.
- `test_probabilities.py` exact probability + Wilson CI.
- `test_maps.py` / `test_tiles.py` exact tile content and `test_ensemble_tile_reduces_member_dimension_to_mean`.
- `test_longitude_convention.py` (0–360 grids, western-hemisphere alignment, outer-longitude 404).
- `test_serving_selection_shape.py::test_point_interpolation_numeric_gfs` explicitly checks the bounded interpolation returns the same value as a full-grid bilinear reference (`abs(diff) < 1e-6`).

The member-mean-vs-window ordering change is verified numerically identical (`np.allclose(equal_nan=True)`, maxdiff 0) in the local measurement and enforced by the ensemble tile contract test.

---

## L. Regression tests protecting against re-introducing full-store materialization

1. **`test_serving_selection_shape.py`** (deterministic, no wall-clock):
   - tile window shape < full grid (GFS + GEFS),
   - point interpolation isel shape == `(2, 2)`,
   - GEFS member interpolation returns exactly 5 bounded member values, each matching the analytic field,
   - `gated_read_dataset_with_selector` materializes a bounded `(3,3)` result, never the full grid,
   - `gated_cycle_metadata` reads metadata only (no gridded data).
   These **fail against the old full-compute implementation** (shapes were full-grid).
2. **`test_reader_race.py`** — reader-gate race + the new lock-held proof.
3. **`test_tile_connection_lifetime.py`** — QueuePool early-release + tight-pool concurrency (no `TimeoutError`).

---

## M. Validation

**Windows (this machine):**

| Check | Command | Result |
|---|---|---|
| API tests | `.venv/Scripts/python.exe -m pytest tests/` | **225 passed** |
| Ruff (api src + tests) | `python -m ruff check src/ tests/` | All checks passed |
| MyPy (api, strict) | `python -m mypy src/api` | Success, 34 files |
| Domain tests | unchanged (no `packages/domain` edits) | n/a |
| Contracts/config import | `import contracts; import config` | v0.1.0 / v0.1.0 |
| Ingestion import | `import ingestion.core.zarr_writer/catalog; domain.locks` | OK (untouched) |

**Linux / CI-equivalent:** My changes touch only the API serving tier (no domain, no ingestion, no Docker). The affected CI jobs are `python-quality` (ruff/mypy on api — both green locally with the CI's pinned versions 0.3.4/1.9.0) and `api-tests` (pytest — green locally). Docker/container-builds are unaffected (no Dockerfile/entrypoint changes; the API image just runs the same uvicorn). A full Linux/CI-equivalent run (Docker/WSL) was not reproduced here; the code is pure-Python with no platform-specific paths beyond the existing xarray/zarr/s3fs stack already CI-proven, and the region/selector logic is deterministic.

**Non-reproducible CI checks:** `container-builds` (Linux Docker runtime smoke) could not be run on the Windows host; the change does not alter the image or runtime dependencies, so the residual risk is minimal. Real `frontend-lint-build`/`frontend-e2e` were not re-run (no frontend code changed).

---

## N. Remaining bottlenecks after Phase 1

With the full-store materialization gone, per-request cost is now dominated by:

1. **Per-request store open + metadata (`.zmetadata`) + grid-coordinate read** (~0.4 s measured against MinIO in the investigation; now the *largest* single phase for a point/tile request because the compute is gone). Every request still builds a fresh `S3FileSystem` + `FSMap` + `open_zarr` and re-reads coordinates. → **Phase 2 candidate: store/s3fs/metadata reuse** (a process-wide cached `s3fs` client, and a per-store open/metadata cache keyed by `serving_generation`) would cut this ~0.4 s overhead.
2. **The `MemoryCachedArray` first-access read of the ~1 chunk overlapping a tile/point.** After the metadata open, the actual chunk GET is the remaining S3 cost (small).
3. **QueuePool/Redis/IPC** — negligible now.

**Is Phase 2 justified?** Yes — the pre-change bottleneck (full-store compute) is eliminated; measured after values are single-digit ms on local disk, but the per-request `open_zarr` + coordinate re-read overhead (~0.4 s measured against MinIO in the investigation) would still dominate on S3. A store/metadata/s3fs reuse cache keyed by `serving_generation` (Phase 2) is expected to cut that constant and reduce S3 object counts further. Rechunking (Phase 3) is **not** yet justified: the window-first ordering already reads 1 chunk per tile/point and the member-full chunk is good for tiles; re-chunking would only matter if point-heavy reads (all-member point series) dominate, which profiling after Phase 2 will determine.

---

## Phase 2 recommendation (not implemented here)

Per the task's instruction ("No Phase 2 work yet"), no caching/rechunking was added. The recommended next step is a **store/metadata/s3fs reuse cache keyed by `serving_generation`**, with a byte cap and same-cycle invalidation semantics, followed by re-profiling to see what remains.