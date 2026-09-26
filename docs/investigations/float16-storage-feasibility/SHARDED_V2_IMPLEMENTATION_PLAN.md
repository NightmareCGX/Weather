# Weather Platform Sharded V2 Final Validation & Implementation Plan

**Nature**: Goal A(cloud_ceiling semantic validation close-out)+ Goal B(directly actionable Sharded V2 Implementation Plan). This round makes zero production code changes. Evidence from the previous two rounds is in [`REPORT.md`](./REPORT.md) and [`QUANTIZATION_BENCHMARK.md`](./QUANTIZATION_BENCHMARK.md); this document does not re-expand the benchmark.

---

## 1. Executive Summary

- **cloud_ceiling final conclusion: Decision B — keep float32(semantic compatibility exception)**. As measured on real data, all 4 fields(GFS f006/f120, GEFS 0.25° member gep01, GEFS 0.25° geavg mean)**all showed real sentinel flips**(122/112/140/203 grid points, 0.011–0.020% of finite, direction all unlimited→finite). In contrast to precipitation, here**the baseline is the policy-correct side**(true value 19.991 km ≥ 19.99 → unlimited, exactly retained by f32), and f16 is the deviating side; product output for affected grid points would change from "unlimited" to "a finite cloud base height of 19.984 km", which is unacceptable. Storage cost ≈ +0.1–0.2% over the full cycle.
- **sharded_v2 representation formally frozen**(§4/§5): continuous variables plain `<f2` + two f32 semantic exceptions(precipitation_amount_3h, cloud_ceiling)+ det/mem flags `u1` + mean flags `<f4`;Zstd L5; index/trailer/object layout exactly identical to v1; the version is distinguished only by the manifest `storage_format_version: "sharded_v2"`.
- **Ready to enter implementation**. This plan gives a file/function-level change matrix(§19), 8 phases(§20), 5 GO/NO-GO gates(§15), and a rollback contract with no data migration(§16). **No remaining technical blocker**(§21/§22): every blocker identified in the previous two rounds is either already closed out(ceiling decision, flags short-circuit fix scheme, writer guard design), or was always a product decision item with a default path already given.

## 2. Frozen Decisions from Prior Investigations

| Decision | Basis |
|---|---|
| continuous = plain float16(`<f2`)+ Zstd L5 | QUANTIZATION_BENCHMARK: plain f16 is on the Pareto frontier (full cycle −22.4%); quantized variants ≤+3.6% relative and carry a permanent policy cost; any quantization inside an f32 container ≤8.3% (G11) |
| precipitation_amount_3h = f32 | exactly-0.10mm threshold:GEFS decimal packing produces a huge number of grid points exactly equal to f32(0.1) (12.813%), and f16 would flip wet→dry in the production float64 comparison domain |
| **cloud_ceiling = f32 (added this round)** | see §3 |
| det/mem flags = u8; mean flags = f32 | flags are a 0/1 category; mean flags are member-mean probability (0..1) |
| Not adopted | decimal quantization, variable-specific quantization, quantized f16, mantissa grooming, fixed-point, global round/truncate |
| compute domain unchanged | f16 arithmetic 8.7× slower + reduction overflow; float64 contracts(deaccum/cloud/stats) retained |
| chunk cache = native persisted dtype | only a native cache can deliver the RAM savings(20,000 vs 40,000 B/chunk) |
| v1 byte contract fully readable; format switching forbidden within a cycle; no DB migration; no bulk conversion | manifest mechanism + per-cycle store isolation(REPORT.md §16/§20) |

## 3. Cloud Ceiling Targeted Validation(Goal A)

### 3.1 Authoritative Chain and Comparison Points (file:function:line, measured code)

| Layer | Location | Semantics |
|---|---|---|
| GRIB source | `parser.py:143-149`(`gh`,`typeOfLevel=cloudCeiling`) | gpm |
| normalize | `pipeline.py:269-273`(gpm→km, ÷1000, float32-domain division) | km |
| constants | `cloud.py:28-29`:`THRESHOLD_KM=19.99`,`SENTINEL_KM=20.0` | |
| domain classification | `cloud.py:253` `classify_cloud_ceiling`:`value_km >= threshold` → unlimited | Python float(f64) |
| point serving | `point_forecast.py:588-597`:`val_km = float(raw_ceil)`;`val_km >= 19.99` → `ceiling=None, unlimited=True` | **float64 domain** |
| tiles rendering | `tiles.py:766`:`finite = isfinite & valid & (values < 19.99)`(float64 window);unlimited → transparent pixel | float64 |
| tiles legacy member path | `tiles.py:1263`:`raw_members >= 19990.0`(**unit bug: values are km, always False**);`tiles.py:1307`: the wrap branch uses `19.99`(km, correct) | float64;**the two branches disagree** |
| ensemble summary | `cloud.py:340-343`(`val = float(m); val >= threshold` → unlimited count), `:354`(conditional-quantile finite set), `cloud.py:383-423`(low-ceiling probability) | float64 |
| ensemble PDF | `ensemble_data.py:610`:`[m for m in members if m < 19.99]` | float64 |
| frontend | `frontend/src/lib/forecast/labels.ts:114`:`19.99 * 3280.84`(ft) | JS Number(f64) |

**Comparison domain determination**: every active comparison is in the float64 domain after the `float()` promotion at the reader boundary — fully isomorphic to precipitation 0.10, with no ambiguous np.float32-domain comparison point; the risk is concentrated in **f16 quantization moves a value across f64's 19.99**.

### 3.2 Sentinel Semantics (code is authoritative)

`>= 19.99 km → unlimited` (strictly greater-or-equal; `== 19.99` also counts as unlimited). No `> 19.99` or `== 20.0` variant. f32(19.99)=19.9899997711 in the float64 domain is **< 19.99 → finite**(the baseline's true classification of "exactly 19.99 km" is finite — a second-order effect of the f32 expansion landing below the threshold, confirmed by measurement).

### 3.3 Real Data (minimal download this round: gfs_f006/f120 + GEFS 0.25° gep01/geavg f120 ceiling;GEFS 0.5° pgrb2a has no such field, the platform actually uses pgrb2sp25 0.25°)

| field | km min..max | ≥19.99 grid points | [19.989,19.995] in-band values | **f16 flips** | direction |
|---|---|---|---|---|---|
| gfs f006 | 0.009..19.9999 | 491,517 | 354 | **122(0.0118%)** | unlimited→finite |
| gfs f120 | 0.009..20.0001 | 462,858 | 325 | **112(0.0108%)** | unlimited→finite |
| GEFS gep01 (0.25°) | 0.010..19.9999 | 577,017 | 472 | **140(0.0135%)** | unlimited→finite |
| GEFS geavg (0.25°) | 0.215..20.0001 | 231,247 | 594 | **203(0.0196%)** | unlimited→finite |

GRIB ceiling packing quantizes at ≈0.4 m, taking dense values just below the sentinel(19.900, 19.9006, …); the flip window = source values falling in **[19.9900000149, 19.9921875) km** (half a cell of the f16 grid {19.984375, 20.0}}), i.e. **≈2.19 m wide, and real data hits it** (the gpm 19990–19992 band).

### 3.4 Adversarial Boundary Replay (production float64 domain, value by value)

gpm 19985–19990 → both finite; **19991/19992 → baseline UNLIM / f16 finite (FLIP)**; 19993–20001, 20.0/20.01/20.02 → both UNLIM (f16 rounds up to the exactly representable 20.0); under python-float semantics, per-point replay of 19.95–20.02 shows flips only in the 19991/19992 band. There is no reverse finite→unlimited flip (rounding f16 up across the threshold requires a true value ≥19.9921875 > 19.99, and the baseline is already unlimited).

### 3.5 Ensemble Semantic Impact

The unlimited count of `cloud_ceiling_ensemble_summary`, `N_finite`, the conditional-quantile member set, the valid denominator of `compute_low_ceiling_probability`, and the PDF filter (`ensemble_data.py:610`) are all driven by the same `>= 19.99` predicate — member-level flips (gep01, 140 grid points) directly shift the `unlimited_probability` and the conditional-quantile member set of the affected cell.

### 3.6 Decision: **B — semantic exception, cloud_ceiling = float32**

Rationale: within this window the baseline is the policy-correct side (`>= 19.99 → unlimited`, satisfied by the true value 19.991), and f16 is the deviating side; affected grid points change output from "unlimited=true, ceiling=null" to "ceiling=19.984 km (≈65,545 ft)" — product-visible. This does not constitute Decision C (the baseline has no float noise error and needs no product-semantics follow-up; but **the `tiles.py:1263` 19990.0 unit bug is an independent bug**,listed in the §18 documentation/fix inventory, outside v2 scope).

**General rule (to be written into the v2 dtype policy)**: any variable whose product predicate compares directly against the stored value(threshold-coupled) must satisfy one of the two:(a) stored as f32; (b) empirically confirm an isolation band of >1 f16 ulp between the real value distribution and the threshold. Neither precipitation (0.10) nor cloud_ceiling (19.99) satisfies (b) → f32.

## 4. Final Variable/Dtype Matrix (v2 frozen)

| Variable / field class | v1 stored dtype | **v2 stored dtype** | product_role difference | Reason |
|---|---|---|---|---|
| temperature_2m | f32 | **f16** | none | safe continuous |
| relative_humidity_2m | f32 | **f16** | none | safe continuous |
| wind_u_10m / wind_v_10m | f32 | **f16** | none | safe continuous |
| wind_gust | f32 | **f16** | none | safe continuous |
| precipitation_rate | f32 | **f16** | none (GEFS has no such variable) | safe continuous |
| cloud_cover_3h | f32 | **f16** | none | safe continuous (reconstruction error ≤0.1pp, REPORT.md §7) |
| snow_depth | f32 | **f16** | none | safe continuous |
| visibility | f32 | **f16** | none | safe continuous |
| **precipitation_amount_3h** | f32 | **f32** | none | exactly-0.10mm threshold compatibility |
| **cloud_ceiling** | f32 | **f32** | none | 19.99km sentinel compatibility (§3) |
| crain / csnow / cfrzr / cicep | f32 (actual bytes) | **u8** | **det/mem** | categorical |
| crain / csnow / cfrzr / cicep | f32 | **f32** | **ensemble_mean** | member-mean probability 0..1 |
| shard index | `<u8` ×2 | **`<u8` (unchanged)** | — | dtype-irrelevant |
| trailer | `<u8 ×2 + <u4 ×1` | **unchanged** | — | dtype-irrelevant |

## 5. Sharded V2 Storage Contract (frozen)

1. **Payload dtype**: explicitly little-endian — f16=`<f2`, f32 exceptions=`<f4`, categorical=`u1`. Ambiguous platform-native endianness is forbidden (v1's `np.float32` implied native; v2 fixes this).
2. **Compression**: Zstd level 5, each 100×100 chunk compressed independently (same as v1, `zarr_writer.py:628`).
3. **Chunk geometry**: 100×100; edge chunks keep v1's `min(100, size)` no-padding special case (`zarr_writer.py:642-645`).
4. **Index/trailer/object layout**: **kept exactly as v1** — magic `0x53484152` unchanged, 16B index entry unchanged, 12B trailer unchanged, object key `{var}/shard.{det|mean|mem%03d}_L%04d.shard` unchanged. Rationale: index/trailer math is dtype-irrelevant (offset/length come from the runtime index); changing the magic would only break inventory/tool compatibility with no benefit.
5. **Version distinction**: only via the manifest `storage_format_version: "sharded_v2"`(written at `coordinator.py:987,1256`; passed through at `manifest_reader.py:171-184`, no whitelist, naturally supported). Note the naming risk: the legacy fallback string `v2_unsharded` is similar to the new `sharded_v2` but completely different — the legacy string is frozen and unchanged, to be flagged in a code comment.
6. **Metadata consistency**: `prepare_run_store` (`zarr_writer.py:373`) uses the resolver-resolved **real storage dtype** when pre-allocating `.zarray` (fixing v1's inconsistency where the flags metadata is uint8 while the bytes are f32). **No per-variable dtype manifest field is added**: dtype is deterministically derived by the frozen `(format_version, variable, product_role)` resolver, and the resolver is versioned together with the format constants (the "resolver is the metadata" principle); `.zarray` serves as the consistency assertion surface (see §19 tests).

## 6. Dtype Resolver Design(product-role aware)

Reuse the existing product identity enum **`TARGET_KIND_DET / TARGET_KIND_MEAN / TARGET_KIND_MEM`** (`domain/reclamation.py:19-21`, values `"det"/"mean"/"mem"`, isomorphic to the DB CHECK `entities.py:366-370`) — no new concept is invented.

```python
# packages/domain/src/domain/storage_dtype.py (new, pure domain, importable from both sides)
class ProductRole(StrEnum):        # directly aligned with TARGET_KIND_*
    DETERMINISTIC = "det"
    ENSEMBLE_MEMBER = "mem"
    ENSEMBLE_MEAN = "mean"

#: sharded_v2 frozen matrix (one-to-one with §4 in this document)
_V2_FLOAT16: frozenset[str] = frozenset({...9 continuous variables...})
_V2_F32_EXCEPTIONS: frozenset[str] = frozenset({"precipitation_amount_3h", "cloud_ceiling"})
_FLAG_VARIABLES: frozenset[str] = frozenset({"crain", "csnow", "cfrzr", "cicep"})

def resolve_storage_dtype(
    format_version: str, variable: str, product_role: ProductRole
) -> np.dtype:
    """v1 → always float32 (frozen); v2 → matrix lookup; unknown variable → fail-closed raise."""
```

Design points: the v1 path **always returns float32** (protecting old cycles); unknown variables are fail-closed (a new variable must be entered into the table explicitly — this is exactly where the "future variable maintenance cost" lands, a single-point decision); ingestion (`zarr_writer`), serving (`zarr_v2 reader`), inventory, and self-read share the same function.

## 7. Writer Safety Contract

Per-chunk validation **before** `astype("<f2")` (the new guard at `zarr_writer.py:658`):

```text
NaN          allowed (legitimate missing semantics), counted separately, not rejected
±Inf         reject (regardless of source)
|value| > 65504.0            reject (f16 max finite; prevents a silent → inf)
flags(det/mem)              value range ⊆ {0,1} assertion (naturally true once u8-ized, kept as a contract assertion)
mean flags                   value range ⊆ [0,1] assertion (probability semantics)
```

- Violation behavior: **raise** `F16RangeViolationError(IngestionError)`(new, `core/base.py`), handled by the existing region/cycle failure path (same pattern as `DeaccumulationError`) — **no silent clamping, no silent NaN-ization**.
- The error message must contain: model, cycle_time, lead, variable, product_role (det/mean/mem), region key, violation count, NaN count, min/max.
- Check order: `isfinite` mask separation → NaN count → Inf check (outside NaN) → magnitude check (on finite values) → astype.
- Why before astype: 70000 → astype(f16) → inf passes silently and is discarded downstream by `isfinite` as invalid, a silent data loss (as measured, REPORT.md §17).

## 8. Reader Architecture

```text
ShardedV1Reader(api/core/zarr.py:213)      → frozen: only severe bugs fixed, no extension
ShardedV2Reader(api/core/zarr_v2.py, new)   → reuse/compose: _ChunkPlacement, _chunk_placements,
                                              _bilinear, index/point cache patterns, executors
get_sharded_reader_for_format(...)          → factory: select the class by manifest_storage_format
```

- v2 differences are concentrated in one point: `read_chunk`'s `np.frombuffer(raw, dtype=resolve_storage_dtype(...))` + `astype`/NaN-fill use the native dtype; index/trailer/placement logic is reused line by line (v1's `zarr.py:316-372` math unchanged).
- 5 serving branch points (`point_forecast.py:781,1386`, `tiles.py:986-987`, `ensemble_data.py:932-935,1052-1055,1203-1206`, `vector_field.py:232-233`) change from `== "sharded_v1"` to calling the factory/predicate; the legacy `v2_unsharded` branch is kept as is.
- `get_sharded_reader` (`zarr.py:709`) LRU key remains store_path (a store has only one format, decided by the manifest).

## 9. Native-Dtype Chunk Cache Design(RAM contract)

`_chunk_cache` caches the **native dtype array after decoding**, not "convert the whole block to f32 right after reading":

```text
f16 continuous   → cache f16(20,000 B/chunk)
u8 flags         → cache u8(10,000 B/chunk)
f32 exceptions   → cache f32(40,000 B/chunk)
```

- Typical mix (full v2 variable set) per-reader 512 chunks: ≈ **10.4 MiB** (v1 19.58 MiB, −47%); point/index caches unchanged (total reader volume −10~15%, REPORT.md §10).
- Compute transients (f32 window, float64 tile) are decoupled from the cache layering — see §10.
- Prohibited: `read_chunk` upgrading the whole block with `.astype(np.float32)` before returning.

## 10. Compute Promotion Boundaries

| path | boundary |
|---|---|
| Point interpolation | chunk cache is native; **only the 4 corner scalars** are promoted via `float()` to the current compute scalar domain (f64) before `_bilinear`(the current `zarr.py:508-511,133-149` pattern kept as is); casting the whole block for 4 values is forbidden |
| Window/map | `read_window` copies chunk by chunk into a **float32 working window** when assembling (the current `zarr.py:612` semantics; the window is a request-level transient and does not affect cache RAM); the float64 promotion on the tiles side is unchanged (`tiles.py:718-720`) |
| Ensemble statistics | member native chunk → float()/f32 after reading points → existing domain statistics (`_validation.py:43` f64 coerce unchanged); the mean flags f32 path is unchanged |
| Vector field | u/v promotion points same as point; `encode_vector_field_int16` input domain unchanged (`wind.py:629-665`) |

## 11. Ingestion / Writer Change Map

| File | Function | Change |
|---|---|---|
| `packages/domain/src/domain/storage_dtype.py` | new module | resolver + v2 matrix (§6) |
| `services/ingestion/src/ingestion/core/base.py` | new exception | `F16RangeViolationError` |
| `core/config.py:236` | `STORAGE_FORMAT_VERSION` | default stays `"sharded_v1"`; switching only via env; new comment: frozen at cycle start |
| `core/zarr_writer.py:261` | `prepare_run_store` | pre-allocate dtype = the resolver result (fixing `.zarray` consistency) |
| `core/zarr_writer.py:615` | `encode_region_sharded_v1` | new v2 encoding path (or parameterized): per-chunk native buffer + writer guard (§7) + explicit `<f2`/`<f4`/`u1` tobytes; `build_sharded_v1_container` reused unchanged |
| `core/zarr_writer.py:408,539` | `commit_region` ×2 | format branch gains `"sharded_v2"` (goes through v2 encoding) |
| `core/zarr_writer.py:900,1043` | `_populate_sharded_data` / `read_slice` | **dtype-aware frombuffer** (key point: cloud_cover_3h's fallback predecessor read is f16; the precipitation predecessor stays f32 — the resolver distinguishes by variable, so there is no misjudgment) |
| `core/coordinator.py:987,1256` | manifest ×2 | inherit the setting; new in-cycle format freeze assertion (snapshot at the first commit, later commits verify consistency, a mismatch fails) |
| `core/inventory.py:606-628` | `region_expected_object_keys` | format branch gains `"sharded_v2"` (same 14/15 shard count; magic validation `:530,564` unchanged) |
| `core/pipeline.py:766-768` | `_normalize_canonical_units` | flags short-circuit fix (§Part 8): when `target_unit == "flag"`, force `np.asarray(..., dtype=np.uint8)` |
| `core/wave_runner.py:1358-1375` | predecessor state | **zero changes** (in-memory f32 predecessor; with precipitation staying f32, the fallback `read_slice` is f32 as well) |
| `core/markers.py` | — | zero changes (markers are dtype-irrelevant) |

## 12. API / Serving Change Map

| File | Change |
|---|---|
| `api/core/zarr_v2.py` | new `ShardedV2Reader` (§8) |
| `api/core/manifest_reader.py:171-184` | no whitelist → automatic passthrough; only a docstring and a `sharded_v2` test case are added |
| 5 branch points (§8 list) | `== "sharded_v1"` → `is_sharded_store_format(v)` predicate + factory |
| `api/services/point_forecast.py / tiles.py / ensemble_data.py / vector_field.py` | numeric paths unchanged (the reader already outputs float()/f32 transients); the `tiles.py:1263` legacy unit bug is listed separately (§18) and is not opportunistically fixed within v2 scope |
| `api/schemas.py` | zero changes |

## 13. Test Migration Strategy

**V1 exact contracts frozen (not relaxed)**: `test_sharded_storage.py` (f32 fixture written at `:51`), `test_serving_chunk_equivalence.py` (`abs=1e-9`, `:18-27` fixtures), `test_serving_selection_shape.py:210` (np.float64 exact). Marked `v1_format_only`, fixtures stay f32.

**New V2 contracts** (tolerances are the measured upper bounds from the two rounds):

| Class | Test cases |
|---|---|
| Serialization | f16 round-trip; f32 exception round-trip; u8 round-trip; NaN passthrough; ±Inf reject (raise + message fields); 65504 boundary (65504 passes / 65505 rejected); explicit `<f2`/`<f4`/`u1` endianness; mixed-dtype read from the same store |
| Metadata | `.zarray` dtype == resolver dtype == actual bytes; manifest `sharded_v2` write/read; v1+v2 coexisting in one process |
| Precipitation | in v2 the precip shard bytes are still f32; exactly-0.10 behavior unchanged (production float64 domain); production in-memory predecessor path; fallback predecessor (f32); reset boundary; the three clamp segments; NaN |
| Cloud ceiling | sentinel regression: injecting 19990/19991/19992/19993/20000 gpm → classification matches the baseline grid point by grid point (the f32 exception makes this always true); unlimited_probability/conditional quantiles are unaffected by storage |
| Cloud reconstruction | f16 stored predecessor read back → the reconstruction result vs the f32 baseline ≤0.1pp; the three ±5% guard segments |
| GEFS | 30 members f16 → mean/median/std/P10/P25/P50/P75/P90 (tolerance ≤0.03°C / ≤0.03 pp); phase support (member u8 flags + mean f32 flags); mean flags are not u8-ized |
| Point/Map | f16 storage → interpolation tolerance in the current compute domain (temperature ≤0.03°C etc., bench1 upper bound×1.2); window assembled as f32; tile PNG rendering equivalent (within tolerance) |
| Mixed format | v1 cycle + v2 cycle serving simultaneously; cross-cycle fallback (predecessor/serving selection/valid-time/lifecycle interaction) |
| Writer guard | NaN count; Inf raise; >65504 raise (message field assertion); flags ∈{0,1}; mean flags ∈[0,1] |

## 14. Shadow Validation Plan

- **Minimal implementation**: `canonical = s3://{bucket}/{model}/{date}/{HH}/cycle.zarr` (unchanged); `shadow = s3://{bucket}/shadow-v2/{model}/{date}/{HH}/cycle.zarr`. The same decoded Dataset goes through the ingestion library path (`pipeline.ingest_grib_file` encode/commit section) to write shadow, **without going through `catalog.reserve_run`** — no `model_runs` row → invisible to both canonical serving (`catalog` queries) and GC/reclamation; the shadow lifecycle is managed by a script (deleted after validation) and does not enter the tombstone system.
- Dual write of the same source cycle: canonical v1 (current production) + shadow v2.
- Comparison: (a) Storage: bytes/ratio/PUT bytes; (b) Numerical: per-variable max/MAE/P95/P99/threshold flips(precip exactly-0.1 and the ceiling sentinel must show zero flips — the f32 exception makes this always true); (c) Serving: instantiate v1/v2 readers directly for the same query and compare point/hourly/map/ensemble/phase/ceiling responses; (d) Runtime: writer CPU, API RSS, Range GET bytes, cold latency (only no material regression is required).

## 15. Rollout Gates

| Gate | Content | Pass criteria |
|---|---|---|
| **GO/NO-GO 1 — Dual Reader Ready** | v1 tests all green (v1 contracts frozen); v2 unit tests all green; mixed-format all green; the ceiling decision closed (this document §3); dtype metadata contract frozen (§5-6) | all green + freeze sign-off |
| **GO/NO-GO 2 — Shadow Passed** | a full GFS shadow cycle + a full GEFS shadow cycle | numeric thresholds met; zero semantic regression; storage savings close to the benchmark (±3pp); no RSS regression; no material regression in writer throughput |
| **GO/NO-GO 3 — GFS Production** | canonical writer switched to v2, GFS det only | at least one full lifecycle/serving cycle observed |
| **GO/NO-GO 4 — GEFS Mean** | add GEFS mean | same as above |
| **GO/NO-GO 5 — GEFS Members** | add 30 members (largest footprint / largest blast radius) | same as above; capacity savings realization confirmed |

## 16. Rollback Contract

- Rollback after enabling canonical v2 = switching env back to `sharded_v1`: **new cycles go back to v1; already-committed v2 cycles keep being served by the v2 reader; zero rewrite, zero DB migration, zero bulk conversion**. Architecture confirmed: the reader branches on the per-store manifest (§8) and is decoupled from the writer setting ✓.
- In-cycle format freeze: `coordinator.py` snapshots the setting at the first commit and fails on any inconsistency in later commits (§11); the manifest 30s forward cache only affects observation latency, not correctness.
- Rollback does not touch lifecycle/reclamation (reclamation is based on physical_key and the `store_generation` baseline, format-irrelevant).

## 17. Lifecycle Compatibility (verification only)

Item-by-item confirmation of format-transparency: canonical selection (by the newest serviceable run in the catalog, format-unaware)✓; retirement (`deletion_started_at`/`deleted_at` fence)✓; reclamation queue (`physical_key` + `store_generation` baseline, `worker.py:184-232`)✓; physical tombstone ✓; anti-resurrection (`catalog.py:215-223` + `wave_runner.py:701-710`)✓; recovery/predecessor holds (`gc/worker.py:577-601`, variable-level rather than dtype-level)✓; store generation ✓. **The only hardcoded place needing a code action**: the format branch in `inventory.py:628` (§11). No other `sharded_v1` whitelist was found blocking lifecycle.

## 18. Documentation Updates (release/docs phase, not changed this round)

1. README GEFS resolution (0.5° → actually pgrb2sp25 0.25°); 2. `docs/ARCHITECTURE.md:320` cache size (2048→512 + v2 native dtype note); 3. `gc/finalizer.py:20` "14-day" → 1 day; 4. `connector.py:71-72` "geavg out of scope" comment deleted; 5. new v2 storage contract document (§5); 6. reason for the precipitation f32 exception; 7. reason for the cloud_ceiling f32 exception (§3); 8. categorical flags native dtype; 9. **Independent bug ticket**: the `tiles.py:1263` legacy member path 19990.0 unit error (a km value compared against meters, unlimited never triggers) — to be fixed separately outside v2 scope; 10. the `pipeline.py:252` float64 comment is inaccurate; 11. naming caution for `v2_unsharded` vs `sharded_v2`.

## 19. File-Level Implementation Matrix

| Phase | File | Function/Class | Change | Risk | Tests |
|---|---|---|---|---|---|
| 0 | `ingestion/core/pipeline.py` | `_normalize_canonical_units:766` | flags short-circuit forced to u8 | low | `test_pipeline.py` flags test cases + new short-circuit test cases |
| 0 | `packages/domain/.../storage_dtype.py` | new | resolver + frozen matrix | low | resolver unit tests (all variables × 3 roles × 2 formats) |
| 0 | `ingestion/core/base.py` | new exception | `F16RangeViolationError` | low | — |
| 1 | `ingestion/core/config.py:236` | setting | comment + cycle freeze note | low | `test_config.py` |
| 1 | `ingestion/core/coordinator.py:987,1256` | manifest | format freeze assertion | medium | freeze-violation raise test cases |
| 2 | `api/core/zarr_v2.py` | `ShardedV2Reader` | new reader | medium | all the §13 v2 reader test cases |
| 2 | `api/core/zarr.py` | `get_sharded_reader` | factory wiring | low | factory dispatch test cases |
| 2 | 5 branch points (§8) | dispatch | predicate replacement | medium | mixed-format test cases |
| 2 | `api/tests/fixtures/__init__.py` | fixture writer | v2 fixture writing (u8/f16/f32 mixed) | medium | fixture self-check |
| 3 | `ingestion/core/zarr_writer.py:261,615,408,539,900,1043` | five places | dtype-aware write/read + guard | **high** | §13 Writer guard + round-trip + predecessor |
| 3 | `ingestion/core/inventory.py:606-628` | `region_expected_object_keys` | v2 branch | medium | inventory v2 test cases |
| 4 | shadow script (new, scripts/) | shadow dual write + comparison | does not enter the catalog | low | shadow report assertions |
| 5-7 | deployment configuration | env | `STORAGE_FORMAT_VERSION=sharded_v2` per-environment gradual rollout | low | — |

## 20. Phase-by-Phase Plan

**Phase 0 — Semantic/Hardening Prerequisites** (no byte changes): flags short-circuit fix; resolver module; exception class; dtype matrix frozen (this document §4 sign-off). *Acceptance*: all tests green; resolver coverage 100%. *Rollback*: pure addition, revert directly.

**Phase 1 — V2 Core Types/Constants**: `STORAGE_FORMAT_VERSION` supported-value extension (still defaults to v1); manifest freeze assertion; `.zarray` dtype wiring (effective only in the v2 branch). *Acceptance*: v1 all green (bytes unchanged). *Rollback*: revert.

**Phase 2 — Dual Reader**: `ShardedV2Reader` + factory + 5 branch points + v2 fixtures; the v1 path is marked frozen. *Acceptance*: GO/NO-GO 1. *Rollback*: the reader is a pure addition.

**Phase 3 — V2 Writer**: writer dtype path + guards + self-read dtype-aware + inventory branch; **canonical is still v1**. *Acceptance*: v2 writer unit tests + self write/read + fixture equivalence. *Rollback*: revert.

**Phase 4 — Shadow Validation**: full GFS + GEFS shadow cycle dual write and four-way comparison. *Acceptance*: GO/NO-GO 2. *Rollback*: delete the shadow storage.

**Phase 5/6/7 — GFS det → GEFS mean → GEFS members**: env gradual rollout, one full lifecycle observation per step. *Acceptance*: GO/NO-GO 3/4/5. *Rollback*: switch env back (§16).

**Phase 8 — Natural V1 Retirement**: old cycles are retired by lifecycle; whether the v1 reader stays is not decided this round.

## 21. Risks

- **Real blocker**: none. (the ceiling/precip semantic exceptions are closed; the writer guard design is complete; the format freeze mechanism is clear.)
- **Implementation detail**: test contract migration volume (v1 frozen + v2 newly created, the largest workload); `read_slice`/`_populate_sharded_data` dtype-aware rework (the critical path for predecessor correctness); consistency across the 5 dispatch points; dual-format fixtures.
- **Optional follow-up**: cleanup of unrounded JSON literal display (separate PR, REPORT.md §21); the `tiles.py:1263` legacy unit bug; a unified dispatch predicate refactor; the future of the v1 reader.

## 22. Final Go / No-Go

**Ready to enter implementation.** Q1: cloud_ceiling **cannot** be safely f16 (4 real fields, 122-203 grid points with sentinel flips) → f32 exception. Q2: dtype matrix = §4. Q3: **yes** — precip f32 remains the safest compatibility default (production in-memory predecessor + fallback predecessor + exactly-0.10 + future flexibility, a fourfold benefit at a cost of 0.4pp). Q4: the cache returns native arrays in `read_chunk` and `astype` happens only at the compute boundary (§9/§10). Q5: the hardcoding that must be changed = `zarr_writer.py:658/1022/1128`(dtype), `zarr_writer.py:373` (.zarray), `inventory.py:628`, the 5 serving `== "sharded_v1"` branches, `config.py:236` (supported values); the `4 bytes/value` assumption appears only once at `zarr.py:421` (payload; index PNG/wind i16's 4 bytes are irrelevant). Q6: frozen = all of `ShardedV1Reader`, `build_sharded_v1_container`/`parse_sharded_v1_index`, the v1 exact tests, the `v2_unsharded` string. Q7: the §13 table. Q8: **yes** — the dual-reader is fully inert for v1 cycles and can be deployed at any point before the writer (production validation must wait for the Phase 4 shadow write). Q9: shadow = an independent `shadow-v2/` prefix + library-path write + not entering the catalog (§14). Q10: **yes** — switching env back is the rollback; already-committed v2 cycles are served by the v2 reader with zero migration. Q11: **yes** — the increasing blast-radius order GFS (1 shard/lead/var) → GEFS mean (isomorphic) → members (30×) is still the lowest risk. Q12: no technical blocker; 5 implementation details; 3 optional follow-ups.
