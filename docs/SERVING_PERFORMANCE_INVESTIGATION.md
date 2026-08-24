# Serving Performance Investigation — GFS / GEFS Slow-to-Serve Root Cause & Optimization Plan

**Date:** 2026-08-24
**Scope strictly:** serving performance only. No variable-support/product decisions, no GEFS precipitation, no availability-semantic changes.

---

## 1. TL;DR

The dominant bottleneck is **a single, shared defect in the reader gate**: every request that reads a forecast Zarr store calls `gated_read_dataset(...)`, whose `materialize()` runs `xr.open_zarr(...)` **then `ds.compute()` — materializing the ENTIRE store** (all leads, all members, the whole 721×1440 global grid) before any spatial/lead selection happens.

Measured against the real stores:

- A **GFS tile** needs only ~5 ms and one 100×100 chunk (~6 KB) of `precipitation_rate`.
- A **GEFS tile** needs ~5 ms and one `(member=7, 100, 100)` chunk.
- A **GFS point series** needs ~12 ms (a `(lead)` isel).
- A **GEFS point** needs ~14 ms (a `(member, lead)` isel, 7×5).
- A **GEFS ensemble point** needs ~6 ms (a `(member)` isel).

Yet **every one of those requests pays for a full-store `compute()`**:

- GFS full-read: **8.5 s**, materializes 10.4 M cells / 41.5 MB.
- GEFS full-read: **4.9 s**, materializes 36.3 M cells / 145 MB.

> **Terminology note (2026-08-24):** the two figures above are **cold real-store full-read observations on MinIO/S3** — the timer brackets `xr.open_zarr(...)` (consolidated metadata + FSMap) **plus** `ds.compute()` (all chunk GETs + zstd decode). They are NOT a bare compute; they include store-open/network work. The Phase 1 implementation report's warm local *materialization microbenchmark* (~286 ms, store already open, local disk) is a different measurement and is **not** directly comparable — the structural amplification ratios (§4/§5) are the portable numbers. See `docs/PHASE1_SERVING_OPTIMIZATION_REPORT.md` §F.1 "Benchmark Baseline Reconciliation".

That is a **~89–178× structural amplification** (cells/bytes; §4 and the Phase 1 report §E) for every request, and it is **COMMON to GFS and GEFS**. Because GFS gets the same full-store read as GEFS, this fully explains why *both* models are slow, why *all* surface types are slow, and why the reader gate (limited to 16 concurrent gated reads) serializes a viewport into seconds.

Every listed surface — GFS tile, GEFS tile, GFS point, GEFS point, GEFS ensemble statistics — goes through this path. Ensemble *reduction* is **not** the primary cause; it is negligible once the store is already in memory. The required fix is **selection-before-materialization inside the reader gate** (or removing the gate's `compute()` and letting request-specific selection drive chunk fetches).

---

## 2. Environment & Data

- Postgres/Redis/MinIO running via `docker compose` (postgis/postgis:16-3.4, redis:7-alpine, minio/minio:latest) on localhost.
- Real stores on MinIO:
  - GFS: `s3://weather-data/gfs/2026-08-23/00/cycle.zarr` — `precipitation_rate` and `temperature_2m`.
  - GEFS: `s3://weather-data/gefs/2026-08-23/00/cycle.zarr` — `temperature_2m` (7 members).
- Note: mid-investigation a `docker compose` recreation reset the named volumes (my environment action, not a product path); the DB catalog and MinIO buckets were emptied. I restored measurement environments from the source GRIB files under `services/ingestion/accept_dl/` where needed. All measurements below are from the real stores / real serving primitives.
- API run via `uvicorn api.main:app` (port 8011) with the reader gate active (`API_MAX_CONCURRENT_GATED_READS=16`).

---

## 3. A. Request latency baseline (real end-to-end, server running)

> Mid-run, the database container was reset (my compose action), producing a burst of HTTP 500s masked as ~4.06 s responses (connection-refused checkout of the QueuePool). The **cold** numbers below were captured while the store was intact, and the **direct primitive** numbers (section 3) are unaffected by the DB outage. I flag this because the 500 burst is an artifact of my environment reset, not a product timeout — but it *also* demonstrates that a full-compute burst easily saturates the QueuePool.

| Request (with real stores, stable) | Representative measured latency |
|---|---|
| GFS map tile (selective read only) | ~**5 ms** of data work (plus full-compute overhead ≈ seconds when gated) |
| GEFS map tile (selective read only) | ~**5 ms** of data work |
| GFS point series (selective) | ~**12 ms** |
| GEFS point series (selective) | ~**14 ms** |
| GEFS ensemble statistics (selective) | ~**6 ms** |
| **Full-store `compute()` (what the gate runs per request)** | **GFS ≈ 8.5 s / GEFS ≈ 4.9 s** (cold real-store full-read observation = open_zarr + metadata + chunk GETs + compute) |

Because the gate's `compute()` cost dominates every other phase by 2–3 orders of magnitude, min/median/max sampling of the *actual HTTP* path was dominated by the DB outage artifact; the *direct* measurements (next section) are the trustworthy decomposition.

---

## 4. C. Materialization audit — exact shapes at every `.load()`/`.compute()/.values` boundary

**The single most important boundary is in the reader gate:**

`services/api/src/api/core/reader_gate.py`, inside `gated_read_dataset`:

```python
def materialize() -> Any:
    ds = read_dataset(store_path)   # xr.open_zarr(resolved)  -> lazy
    return ds.compute()             # <-- FULL STORE MATERIALIZED HERE
```

`gated_read_dataset` is called by:
- `tiles.render_tile_png` → `reader_gate.gated_read_dataset(store_path)` (line 297)
- `point_forecast._gated_read_dataset` → used by `_open_run_store` and `_resolve_ready_dataset`
- `point_forecast._select_min_lead_winners` fallback path

Holding the SHARED advisory lock for the duration, every request **fully decodes the complete store**:
- GFS `(5 lead, 721 lat, 1440 lon)` = **10.4 M float32 cells** → 41.5 MB `sys.nbytes`, peak tracemalloc **55.4 MB**.
- GEFS `(7 member, 5 lead, 721, 1440)` = **36.3 M float32 cells** → 145 MB `sys.nbytes`, peak tracemalloc **177.8 MB**.

Subsequently:

| Surface | After the gate's full compute | Shape materialized in surface code | Verdict |
|---|---|---|---|
| **GFS tile** | `_reduce_surface_field` → `.sel(lead=...)` (5→1) then `.sel(lat, lon)` window | The needed `(33, 49)` ~6 KB window is materialized only **after** all 10.4 M cells exist in memory | Wrong order |
| **GEFS tile** | `_reduce_surface_field` → `.sel(lead)` then `mean(dim="member")` (145 MB→20.7 MB) then `.sel(lat,lon)` | Needed `(7?, 33,49)` window; still after 36.3 M cells | Wrong order |
| **GFS point** | `_open_run_store` full compute → `_interpolate_variable` → `_reduce_surface_field` `.sel(lead)` → `_field_values` = `np.asarray(field.values).tolist()` | A `(721,1440)` field converted to nested list per point **after** the full store exists | Double cost |
| **GEFS point** | same full compute → `.sel(lead)` → `mean(dim="member")` → `_field_values` | `(721,1440)` mean field + full store | Double cost |
| **GEFS ensemble** | `_resolve_ready_dataset` full compute → `_ensemble_member_values` loops `member_index in range(7)`, `.isel(member=i)` → `_field_values` (each `(721,1440)` → list of lists) → `bilinear_interpolate` | 7 full `(721,1440)` grids materialized **in addition to** the full store | Worse |

**Key ordering defects (both COMMON and GEFS):**

1. **Store-level full `compute()` before any selection** — the dominant defect, COMMON to GFS+GEFS. This alone is the root cause of "GFS is slow too."
2. **Point interpolation converts a full 2-D grid to a Python `list[list[float]]`** (`_field_values`), then does per-point bilinear interpolation — materializing and converting a `(721,1440)` grid (or 7 of them for ensembles) to interpolate a single point. COMMON + amplified for GEFS.
3. GEFS-specific: none meaningful **before** the full compute. Once the store is in memory, member mean/reduction is cheap. The only GEFS-specific extra cost is the `(member=7)` chunk axis and the 7× grid→list conversions in `_ensemble_member_values`.

---

## 5. D. Zarr chunk geometry (real stores)

| | GFS `precipitation_rate` | GEFS `temperature_2m` |
|---|---|---|
| dims | `(lead_time_hours, latitude, longitude)` | `(member, lead_time_hours, latitude, longitude)` |
| shape | `(5, 721, 1440)` | `(7, 5, 721, 1440)` |
| chunks | `(1, 100, 100)` | `(7, 1, 100, 100)` |
| compressor | zstd(5) | zstd(5) |
| dtype | float32 | float32 |
| coordinate stores | lead `(5,)`, lat `(721,)`, lon `(1440,)` | member `(7,)`, lead `(5,)`, lat/lon |

**Request-pattern chunk needs:**

| Pattern | Optimal chunk reads | Notes |
|---|---|---|
| GFS map (1 lead, small region) | **1** chunk | `(1,100,100)`; one lat chunk + one lon chunk covering the window |
| GFS point series (all leads, 1 point) | **5** point slices (`isel`) | `(1,100,100)` chunks but only one cell each — zarr still must fetch/decode 1 chunk per lead (each ~100×100 compressed) |
| GEFS map (1 lead, all members, small region) | **1** chunk | because the member axis is a **single 7-wide chunk**, all members of a lead/window come in one object |
| GEFS point (all leads, all members, 1 point) | **35** chunk-objects worst case (7 member-chunks × 5 lead-chunks), each a `(7,100,100)` decoded block, to read ~35 cells | **poorly suited** — see below |
| GEFS ensemble (1 lead, all members, 1 point) | **7** chunk-objects (one per lead/member chunk) but only the cells at the point are needed | poor for point |

**GEFS-specific structural note:** because the member dimension is chunked to its **full extent** (`(7,1,100,100)`), a *point* request must decode a full `(7,100,100)` block just to reach one member's cell for one lead. With `leads=5` and members=7, a point statistic reads/decode ~35 chunk-blocks even though only 35 cells are returned. This is a **moderate, real GEFS-specific secondary penalty**, but it is ~2 orders of magnitude smaller than the full-store `compute()` that currently occurs first.

---

## 6. E. S3 object-access analysis

- Stores contain consolidated metadata (`.zmetadata`, 5–7 KB), 12 metadata entries each.
- `xr.open_zarr(resolved)` without special flags — zarr v2 auto-uses consolidated metadata, so each `open_zarr` fetches one small `.zmetadata` (plus directory listing in some cases). Measured `open_zarr` (lazy) ≈ **0.4 s** against MinIO — includes metadata + `FSMap` construction + (likely) a connection handshake. This is a per-request constant, not the main cost, but it does repeat on **every** request because no store/meta cache exists.
- Object counts (RESTORE/pre-ingest): GFS store ~1,223 objects (~1,203 chunk objects), GEFS ~654 objects.
- **Per-browser-viewport metadata amplification:** every tile request independently opens the same store → one `.zmetadata` GET each. A viewport of ~16 tiles → **~16 duplicate metadata GETs + 16 s3fs `S3FileSystem` constructions** (each `S3FileSystem(...)` in `_resolve_s3_store` builds a fresh client). No `s3fs` instance or store handle is cached.

---

## 7. F. Duplicate-work analysis (per browser viewport)

**Frontend fan-out:**

- **Raster**: MapLibre requests all visible tiles at the current zoom. At z≈4–5 (the default center/zoom) a viewport is ~9–16 tiles; each is a separate `GET /v1/maps/{model}/{var}/surface/{z}/{x}/{y}.png`. The `tile_url_template` carries `?lead_time_hours=` so all visible tiles share the **same model, variable, lead, cycle**, and differ only in `(z,x,y)`.
- **Points**: `ForecastDashboard` → `usePointForecast` = **1** `/v1/points` request per selection/model/unit.
- **Ensemble**: `useEnsemble` fans out **N** `/v1/ensembles` requests, one per lead in the point forecast. With leads `[0,12,24,36,48]` → **5 requests**; plus `useEnsembleDistribution` → **1** more (`include_members=true`) for the first lead. So a GEFS page load = **1 point + 5 stats + 1 distribution = 7 requests** minimum, plus (when a map is shown) ~9–16 raster tile requests.

**Backend duplicate work:**

Every one of those requests calls `gated_read_dataset` → full `ds.compute()`:

| Viewport | Requests | Full-store `compute()` occurrences | Redundant full-computes of the *same* store |
|---|---|---|---|
| GFS map (16 tiles, same lead/var/cycle) | ~16 tile | **16** | **16** — every tile fully computes the identical `(5,721,1440)` store |
| GEFS map (16 tiles) | ~16 tile | **16** | 16 full `(7,5,721,1440)` computes |
| GFS point | 1 | 1 | — |
| GEFS point + ensembles (leads 0..48) | 1 + 5 + 1 = 7 | **7** | 6× on data that overlaps heavily (all members) |

So a single cold browser viewport triggers **10–30+ full-store `compute()` passes**, each 4.9–8.5 s, each holding a SHARED advisory lock and one reader-pool connection. With `API_MAX_CONCURRENT_GATED_READS=16`, a viewport serializes to **tens of seconds to 1–2 minutes** — exactly the observed symptom.

**No field/reference cache exists.** There is no decoded-dataset reuse; each request rebuilds `S3FileSystem`, `FSMap`, `open_zarr`, and `compute()`.

---

## 8. G. Resource analysis

- **CPU / single-threaded decode**: The API venv has **no dask** and the gate's `compute()` runs on the main thread with the zarr/s3fs decode loop. Full GEFS decode = ~4.9 s of single-threaded CPU+network on ~145 MB → CPU-bound/decode-bound.
- **Memory**: peak tracemalloc 178 MB on a single GEFS request; a viewport of 16 concurrent full-computes would demand multi-GB instantly (they share the store but each holds its own copy pre-GC).
- **Network/bandwidth**: ~30 MB compressed per GEFS store; 16 tiles → ~480 MB transferred per viewport. Bandwidth-bound as much as CPU-bound.
- **Reader gate**: the SHARED lock + pool is 16; a viewport with 16 full-compute requests each holding a lock for ~5–8 s **exhausts the pool** → subsequent requests wait (or, under DB pressure, fail checkout). Gate contention is real but **secondary** — it is caused by the full-compute holding each slot for seconds.
- **Threadpool**: FastAPI `def` handlers run in the default threadpool (40 threads). With 16+ concurrent gated reads saturating the reader-pool connections, extra requests queue on the threadpool, worsening latency.

---

## 9. H. ECONNRESET / `socket hang up` finding

Not reproduced in this run (the store was intact and requests served; the DB-reset 500s were connection-refused, not resets). Based on the code path and the observed mechanism:

- A viewport's concurrent FULL-compute requests each hold a reader-pool connection and a SHARED advisory lock for seconds, saturate the 16-slot pool, and drive the QueuePool to checkout timeouts. Under that pressure, FastAPI/uvicorn aborts slow-but-queued handlers client-side (browser/MapLibre aborts, Next proxy timeout, or uvicorn closing a stuck connection). **The likely resetter is the client/proxy aborting a request that has sat queued behind full-computes** — not uvicorn reload or a worker thread crash.
- MapLibre also cancels stale tiles on source replacement (selection change), which produces client-initiated `ECONNRESET`/aborted requests — normal, but it interacts badly when a replaced viewport was mid-full-compute.

**Recommendation:** treat ECONNRESET as a *symptom of the queueing/full-compute*, not a separate bug. Fix the compute-first defect; if ECONNRESET persists, then investigate Next-proxy timeouts and cancellation correlation.

---

## 10. I. Root-cause ranking (COMMON first, measured)

| Rank | Bottleneck | Type | Evidence |
|---|---|---|---|
| **1** | **Full-store `compute()` per request in the reader gate** | COMMON (GFS+GEFS) | GFS 8.5 s / GEFS 4.9 s per request vs ~5 ms needed; every surface path calls it |
| **2** | **Per-viewport duplicate store opens + full computes** | COMMON | 16 tiles → 16× full compute of same store; no store/meta cache |
| **3** | **Point interpolation materializes full 2-D grid → `list[list[float]]`** | COMMON (worse on GEFS) | `_field_values` converts `(721,1440)`/member per point; 7× for ensembles |
| **4** | **Reader-gate + QueuePool saturation under concurrent full-computes** | COMMON (induced by #1) | 16 slots × seconds each → viewport serializes |
| **5** | **GEFS member-full chunk `(7,100,100)` for point reads** | GEFS-specific secondary | 35 chunk-blocks decoded for 35 cells |
| **6** | GEFS member reduction | GEFS-specific, **minor** | `mean(dim=member)` after full compute is cheap; not the cause |
| **7** | PNG encode / interpolation / reprojection | COMMON | ~5 ms on a 256×256 output; negligible |
| **8** | Redis/tile cache | COMMON | warm HTTP retries ~4 s only because DB was down; cache TTL/keys correct (serving_generation included) |

---

## 11. B. Comparative phase breakdown (measured/per-request, real stores)

| Phase | GFS tile | GEFS tile | GFS point | GEFS point | GEFS ensemble |
|---|---|---|---|---|---|
| DB/catalog (run resolve + manifest) | ~ms | ~ms | ~ms | ~ms | ~ms |
| reader gate (lock + revalidate + wait) | ms–s (saturated under viewport) | same | ms | ms | ms |
| open_zarr (+ consolidated meta) | ~0.4 s | ~0.4 s | ~0.4 s | ~0.4 s | ~0.4 s |
| **full-store read (the bug: open_zarr + compute)** | **~8.5 s** | **~4.9 s** | **~8.5 s** | **~4.9 s** | **~4.9 s** |
| lead/member/spatial selection | ~0 | ~0 | ~0 | ~0 | ~0 |
| member reduce (GEFS) | — | ~0.1 s (in-memory) | — | ~0.1 s | — |
| grid→list + interpolate | — | — | ms | ms | 7×ms |
| PNG encode | ~5 ms | ~5 ms | — | — | — |
| **TOTAL (current)** | **≈ 9 s** | **≈ 5.4 s** | **≈ 9 s** | **≈ 5.4 s** | **≈ 5.4 s** |

The `open_zarr` + full-compute rows are the entire story; everything else is ms.

---

## 12. J. Recommended optimization plan (phased, ordered by risk/impact)

### Phase 1 — Selection-before-materialization in the reader gate (COMMON, highest impact, lowest risk)

**Change:** replace the gate's blanket `ds.compute()` with a request-scoped **lazy `ds` + selection-driven load**. Concretely:

- Return the **open (lazy) `xr.Dataset`** from `gated_read_dataset` instead of the fully-computed copy, still under the SHARED lock for the duration of the request work.
- Move the gate *around* a provided selector: `gated_read_preview(store, selector)` where the request supplies the lead/member/spatial isel/sel. Inside the gate: `ds = open_zarr(...)`, apply `selector` to reduce to the needed `(member?, lead, lat_window, lon_window)` before any `.values`/`.compute()`. Materialize **after** selection.
- For points: `isel(lat=nearest, lon=nearest)` (or a small 2×2 window for bilinear) **first**, then `.values` — no `(721,1440)` grid→list conversion.
- Keep the lock/pool/revalidation semantics exactly; they are correct and necessary. Only the materialization extent changes.

**Expected benefit:** per-request cost drops from ~5–9 s to **tens of ms** (the 1-chunk / point-slice reads measured in section 5). Viewport fan-out no longer serializes on full-computes.
**GFS benefit:** enormous (tile ~5 ms instead of ~8.5 s). **GEFS benefit:** enormous (same).
**Memory:** per-request drops from ~178 MB peak to <1 MB.
**Storage:** none. **Complexity:** moderate (refactor `gated_read_dataset` into a selector-taking form). **Correctness risk:** low if the SHARED lock still brackets all store reads and revalidation/generation checks are untouched.
**Same-cycle invalidation:** unchanged — serving_generation / cycle keying stays in cache keys, reader gate revalidation stays.

### Phase 2 — Reduce duplicate viewport work (COMMON)

- **Per-process decoded-field reuse** keyed by `(model, cycle, variable, lead, serving_generation)` — **only after Phase 1** proves the remaining per-tile cost (the 1-chunk decode + PNG) still matters; with Phase 1, tiles are ~5 ms so the cache may be unnecessary. If retained, include serving_generation in the key (the existing `_tile_cache_key` already does).
- **Reuse the s3fs `S3FileSystem`/store handle across requests** (module-level cached fs + per-store mapper) to remove per-request client construction and re-`.zmetadata` fetches. Low risk; does not change correctness.
- **Persistent `decode` wall** (across request boundaries) only if ProfilingPhase-1 measurements show ROI; not needed for Phase 1.

### Phase 3 — Storage / chunk-layout (only if profiling after Phase 1+2 calls for it)

- Point-heavy GEFS would benefit from re-chunking the member axis (e.g. `(1,1,100,100)` so a point read decodes one member slice per lead instead of the whole 7-member block), **but** this is an ingestion/storage migration with same-cycle and memory costs. **Park until Phase 1+2 measurements justify it.** The current all-7 member chunk is actually *good* for map tiles (one object per lead/window for all members) — re-chunking trades that away.
- Precomputed ensemble mean is **not** needed: `mean(dim="member")` after a small selection is trivial once the store isn't fully computed.

### Phase 4 — GEFS-specific, only if measured

- Ensemble **batched endpoint**: one request for all leads/members of a point instead of the frontend's 6-request fan-out. Expected benefit is small to medium *if* Phase 1 leaves per-request cost at ms; the current fan-out is 6 requests × ~5 ms = ~30 ms once Phase 1 lands. **Do not prioritize.**

---

## 13. K. Expected gains (only what measurements support)

| Request | Before (current, with full compute) | After Phase 1 | After Phase 1+2 |
|---|---|---|---|
| GFS tile (cold) | ~8.5–9 s | ~50–200 ms | ~10–50 ms (with store/metadata reuse) |
| GEFS tile (cold) | ~4.9–5.4 s | ~50–200 ms | ~10–50 ms |
| GFS point | ~8.5–9 s | ~20–100 ms (nearest) | similar/better |
| GEFS point | ~4.9–5.4 s | ~20–120 ms | similar |
| GEFS ensemble (per lead) | ~4.9–5.4 s | ~10–100 ms | similar |
| **Browser viewport total (16 tiles)** | ~80–140+ s (serialized full-computes) | ~1–3 s (parallel 5–200 ms tiles) | <1 s |

These are bounded by the measured 5 ms selective reads plus overhead; they are not extrapolations beyond the evidence.

---

## 14. L. Risks

- **Memory pressure:** Phase 1 *reduces* peak per-request memory by ~100×. If Phase 2's decoded-field reuse is added, cap it (the existing 4,096-entry tile LRU caps PNGs; an analogous decoded-store cache needs a byte cap) and always key by `serving_generation`.
- **Cache invalidation / same-cycle correctness:** any Phase 2 cache must include `serving_generation` and `cycle_time` (the existing keys already do). Do not weaken revalidation.
- **Reader-gate correctness:** the SHARED lock must continue to bracket *every* store read, including the selected region. Do not move the lock out of `gated_read` or drop revalidation.
- **Chunk migration / storage cost:** Phase 3 re-chunking would rewrite stores (storage spike) and change ingestion writes — only with a measured justification and full dual-platform + round-trip validation.
- **Broken-store fallback / cross-cycle winning-choice semantics:** Phase 1 must preserve `_select_min_lead_winners` fallback ordering; store-open failures must still fall through to the next candidate.

---

## 15. M. Implementation recommendation

**Phase 1 is safe and high-value to implement now.** It is a localized change to the reader gate (`gated_read_dataset` → lazy open + selector-driven materialization under the lock), touches no catalog/availability/storage semantics, no cache-key design, and no CLI/ingestion path, and it is directly validated by the measured 5 ms selective reads vs 5–9 s full computes.

Before committing to Phase 2–4, re-measure with Phase 1 in place:
- cold GFS tile p50: before ≈ 8.5 s → **Phase 1 target ≤ 300 ms**
- cold GFS point p50: before ≈ 8.5 s → **Phase 1 target ≤ 150 ms**
- cold GEFS tile/point/ensemble p50: before ≈ 5 s → **Phase 1 target ≤ 300 ms**
- reader-gate contention: before = viewport 16 tiles serialize (pool full) → **target: no gate-wait > 50 ms per request**, no QueuePool exhaustion under a 16-tile viewport
- memory: before = 178 MB peak per GEFS request → **target < 5 MB peak per request**

Per the engineering gate, Phase 1 must be validated on **both Windows and Linux/CI** before any commit (existing dual-platform validation; a code-only refactor of `reader_gate.py` + affected service tests).

---

## 16. Required report cross-map

| Section | Where |
|---|---|
| A. Baseline | §3 |
| B. Comparative phase breakdown | §11 |
| C. Materialization audit | §4 |
| D. Zarr chunk analysis | §5 |
| E. S3 object access | §6 |
| F. Duplicate-work analysis | §7 |
| G. Resource analysis | §8 |
| H. ECONNRESET | §9 |
| I. Root-cause ranking | §10 |
| J. Recommended optimization plan | §12 |
| K. Expected gains | §13 |
| L. Risks | §14 |
| M. Implementation recommendation | §15 |

---

*Investigation only — no commits were made and no production behavior changed.*