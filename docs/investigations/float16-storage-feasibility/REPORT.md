# Weather Platform Float16 Storage / Float32 Compute Feasibility Report

**Nature of investigation**: investigation only, no implementation (whole repo, whole pipeline). All conclusions are based on the current repository implementation and measurement on real data (GFS 2026-09-23/18z 0.25°, GEFS 2026-09-23/00z members, 85 real decoded fields in total; the benchmark scripts and raw results are outside the repo at `%TEMP%\f16bench\`).

**Overall conclusion: conditionally safe** — provided float32 computation is kept unchanged, migrating the persistent storage of continuous-variable fields and the API decompression cache to float16, and migrating categorical fields to uint8, has zero substantive impact on the semantics of all products at measured precision; but the actual object storage/network saving is about **22% (not 50%)**, and there are 5 boundary conditions that must be handled before implementation (see §21).

---

## 1. Executive Summary

**Classification of conclusions: a technically safe core subset + a conditionally safe overall migration.**

- **The recommended target architecture is Architecture B** (§24): float32 decode/normalize/derived computation, float16 persistence of continuous variables and decompressed chunk cache, uint8 categorical (member/deterministic shard), float32 interpolation/statistics/reduction. **float16 computation is directly rejected by measured evidence** (on Zen 3 f16 arithmetic is 8.7 times slower and reduction overflows, §14/§15).
- **Precision safety (measured on real data)**:
  - Per-variable float16 quantization error: temperature max 0.027°C, RH/cloud cover max 0.03%, wind max 0.014 m/s, gust 0.061 km/h, visibility max 7.8 m, snow depth max ~1 mm, precipitation max 0.125 mm (only at the 306 mm extreme).
  - **All threshold flip rates are 0**: dry/wet boundary 0.10 mm, calm wind 0.5 m/s, fog 1 km, 95% overcast, 60 km/h gust, 0°C freezing line (mean/median).
  - de-accumulation: the production path (predecessor in memory as float32, `wave_runner.py:1358-1375`) has **zero semantic flips**, and the error is only the final increment quantization (max 0.0625 mm); the fallback path (predecessor read from storage) likewise has zero flips on real lead pairs; the adversarial corner case (true residual exactly at the −0.5 mm clamp boundary) occurs at ~1/10⁶ grid points.
  - GEFS 30-member statistics (30 real member fields): mean max 0.007°C, quantile max 0.0155°C, IQR max 0.029°C; at the 30°C threshold P90 flips 3/259,920 grid points (0.001%).
- **Benefits (measured, not theoretical)**:
  - Object storage: after Zstd-5 only **~22%** is saved (halving the payload ≠ halving after compression). GFS 0.83→0.64 GB per run, GEFS 23.7→18.4 GB per run; the active retention window (GFS ~40 runs / GEFS ~20 runs) totals ~507→394 GB, **saving ~113 GB**.
  - Range GET bytes: point request 7.8→6.4 KB (−18%), GEFS statistics request 235→191 KB (−19%); **request count unchanged**.
  - API chunk cache: the decompressed payload is exactly halved (19.59→9.83 MiB/reader), but the index/point caches within the reader's total cache are dtype-independent, so **the reader as a whole saves only ~10-15%**.
  - ingestion RSS: **almost no benefit** (with decode/normalize kept at float32, only the shard encoding buffer is halved).
- **Compatibility**: no DB migration needed — the manifest already carries `storage_format_version` (`coordinator.py:987`), serving already branches per store (`manifest_reader.py:171-184` + 5 call sites), and per-cycle coexistence of the old and new formats has a ready-made mechanism.
- **What is not recommended**: float16 computation (Architecture C), global round/truncate, converting uint8 flags to float16, assuming a 50% saving after compression.

---

## 2. Current Numerical Architecture (Numerical Architecture and Data Flow)

```text
NOAA S3/NOMADS
  │  HTTPS byte-range GET(httpx,bytes)connector.py:562-569
  ▼
[bytes] ── write to local staging file ──▶ ProcessPoolExecutor(decode=2 concurrency, decode_worker.py:89,97)
  │                                         only the file path is passed; returns a pickled xr.Dataset
  ▼
cfgrib decode(parser.py:209-214)━━━━━━━━━━━ outputs float32(test_parser.py:52 assertion)
  │                                            no astype anywhere
  ▼
raw-normalized Dataset(≈60 MB float32 / 15 vars × 721×1440)
  │  ① precipitation de-accumulation(pipeline.py:302-370)
  │     input explicitly cast float64(331-332)→ float64 subtraction → clamp/NaN guard → float32 output
  │     production path predecessor = in-memory np.copy(float32)(wave_runner.py:1358-1375)
  │     fallback path predecessor = re-read from committed storage(read_predecessor_precipitation, pipeline.py:373-420)
  │  ② cloud cover reconstruction(cloud.py:103-171)
  │     float64 computation(160-161),±5% guard(24), cast back to float32(188-189)
  │  ③ unit conversion(pipeline.py:251-299):K−273.15, ×3600, ×3.6, ÷1000 — scalar vs float32 array, stays float32
  │  ④ flags normalization:np.asarray(array, dtype=np.uint8)(pipeline.py:292-298)
  ▼
normalized Dataset(float32 continuous variables + uint8 flags)
  │  sharded_v1 encode(zarr_writer.py:615-672)
  │  ━━ ⚠ the only hard-coded dtype:chunk_buf = np.full(..., np.nan, dtype=np.float32)(zarr_writer.py:658)
  │  ━━ ⚠ uint8 flags are merged back into the float32 buffer here (while the .zarray metadata still records uint8, zarr_writer.py:373)
  │  each 100×100 chunk independently Zstd level 5(zarr_writer.py:628)
  ▼
Shard container:payload(120 chunks) + index(120×16B <u8) + trailer(12B,'SHAR')
  │  single PUT(zarr_writer.py:829), no multipart; object key {var}/shard.{det|mean|mem%03d}_L%04d.shard
  ▼
Object Storage(MinIO/S3) + PostgreSQL catalog(no format/dtype column)
  ══════════════════════════ serving ══════════════════════════
  ▼
ShardedV1Reader(api/core/zarr.py):tail Range GET reads index(316-372)→ per-chunk Range GET(413)
  │  np.frombuffer(raw, dtype=np.float32).reshape(100,100).copy()(421)━━ second hard-coded dtype
  ▼
_chunk_cache:512 × 100×100 float32 ndarray/reader(zarr.py:259;config.py:166)
  ├─▶ point:take the 4 corner values → float() immediately becomes a Python float(455,508-511)→ Python float bilinear(133-149)
  │     → round(x,1)(point_forecast.py:547,581)/ some variables are not rounded(603-606)→ JSON
  ├─▶ window:read_window float32 assembly(612)→ tiles immediately promoted to float64(tiles.py:718-720 etc.)→ np.interp color mapping → uint8 RGBA PNG(769)
  ├─▶ ensemble:per-member chunk GET(member executor 4 workers)→ domain statistics forced float64(_validation.py:43)→ Python float
  └─▶ vector:u/v float32 → hypot(tiles.py:1070)→ int16 scale=0.01 quantization(wind.py:664-665)→ Redis raw bytes
       flags(GEFS mean shard)= member-mean probability float32(0.65 etc.)→ round(x,4) output(point_forecast.py:575)
```

**Summary of dtype transition points**(per node:input→output, explicit/inherited, whether copied, whether compressed):

| # | Node | Location | Input→Output dtype | Explicit? | copy/allocation | Compression/serialization |
|---|---|---|---|---|---|---|
| 1 | Download | connector.py:619,668 | bytes | — | disk staging | no |
| 2 | cfgrib decode | parser.py:209-214 | GRIB→float32 | inherited(cfgrib) | Dataset allocation | no |
| 3 | cross-process return | wave_runner.py:1727-1729 | float32(pickle) | inherited | full pickle copy | pickle |
| 4 | de-accum | pipeline.py:331-339 | f32→**f64**→f32 | explicit | 2 array conversions + diff + mask | no |
| 5 | cloud reconstruction | cloud.py:160-189 | f32→**f64**→f32 | explicit | same as above | no |
| 6 | unit conversion | pipeline.py:255-298 | f32→f32 | inherited(weak scalar) | in-place assignment | no |
| 7 | flags normalization | pipeline.py:292-298 | f32→**uint8** | explicit | new array | no |
| 8 | shard encode | zarr_writer.py:658-660 | everything→**float32** | explicit(hard-coded) | chunk_buf | **Zstd-5/chunk** |
| 9 | shard read | zarr.py:413,421 | f32 bytes→ndarray | explicit(hard-coded) | .copy() | Zstd decompression |
| 10 | chunk cache | zarr.py:259 | f32 ndarray | — | held by LRU | — |
| 11 | point interpolation | zarr.py:455,508-511,133-149 | f32→**Python float(f64)** | explicit float() | scalar | JSON |
| 12 | window→tile | tiles.py:718-720,769 | f32→**f64**→uint8 RGBA | explicit | window copy | PNG(zlib-1) |
| 13 | domain statistics | _validation.py:43 | any→**float64** | explicit | array conversion | JSON |
| 14 | vector quantization | wind.py:629-665 | f32→f32(hypot)→**<i2** | explicit | copy | Redis bytes+gzip-6 |
| 15 | Redis response cache | cache.py:199 | Python float→JSON text | — | — | JSON text |

---

## 3. Current Storage Format(writer/reader/schema)

- **Layout**(writer `zarr_writer.py:575-601`, reader `zarr.py:316-372`):payload = the Zstd-5 compressed bytes of 120 100×100 chunks concatenated in order; index table 120×16B(`struct.pack("<QQ")`, little-endian uint64 offset/length); trailer 12B(num_chunks, index_size, magic 0x53484152).index read in a single tail Range GET(`zarr.py:342`), the API reader **does not validate the magic**(only the ingestion inventory validates it,`inventory.py:530`).
- **chunk geometry hard-coded**:100×100, 8×15=120 chunks(shard covers 721×1440); declared independently in three places:reader `zarr.py:385-389`, writer `zarr_writer.py:642-645`, inventory `inventory.py:519`.
- **dtype metadata:does not exist**.The manifest(`__commit__/v1/manifest.json`)has `storage_format_version` but **no dtype field**; the `.zarray` dtype comes from the preallocated dataset(`zarr_writer.py:373`)—**inconsistent with the shard's actual bytes**(flags metadata uint8, bytes float32).
- **Hard-coded dtype locations**(v2 must-change points):writer `zarr_writer.py:658`; reader `zarr.py:421`(the only payload decode point)+ NaN fill `zarr.py:387,402,408,612,666`; ingestion self-check read-back `zarr_writer.py:1022,1128,1146`.
- **Can old and new dtype coexist**:yes.The dispatch hook already exists—`manifest_storage_format()`(`manifest_reader.py:171-184`)branches at 5 serving call sites(point_forecast.py:781,1386; tiles.py:986-987; ensemble_data.py:932-935 etc.), and ingestion routes the write path by `STORAGE_FORMAT_VERSION`(`config.py:236`)(`zarr_writer.py:470-471`).Introducing a new version string(e.g. `sharded_v2`)+ a per-format dtype table is enough; the index/trailer math is dtype-independent(offset/length come from the runtime index,`zarr.py:406`).

**Evaluation of the three migration options**:
- **Option A**(keep v1 and add dtype metadata):although the manifest is a store-level JSON, the v1 read path's `np.frombuffer(dtype=np.float32)` is a metadata-free hard-coding—keeping the v1 string while changing its byte semantics would make an old reader that "reads as float32 whenever the manifest says sharded_v1" read the new store wrongly.**Rejected**.
- **Option B(recommended)**:a new version string `sharded_v2`(payload in per-variable native dtype:continuous variables f16, flags u8, mean-shard flags kept at f32 probability), written into the manifest's `storage_format_version`; for v2 the reader decodes by the dtype table and immediately unifies to the internal float32(or an f16 cache).An old cycle's manifest is still `sharded_v1` and takes the old path,**readable with zero changes**.
- **Option C**:per-chunk self-description(a dtype header prefixed to each chunk)—breaks the existing compact layout where "the index directly gives the offset" and wastes bytes per chunk.**Rejected**.

---

## 4. Current Variable Inventory (Real Variable List)

Authoritative definitions:`DEFAULT_VARIABLES`(`wave_runner.py:79-170`,15 of them), `SURFACE_FIELD_FILTERS`(`parser.py:44-150`), unit transforms `pipeline.py:251-299`.GEFS member count 30(`coverage.py:30-33`), horizon 0–240h/3h = 81 leads(`horizon.py:14-16`).

| Variable | GRIB | Source unit→internal unit | Range(measured) | ingestion derived | lead-0 | ensemble | Type | storage dtype current state |
|---|---|---|---|---|---|---|---|---|
| temperature_2m | `2t` | K→°C(−273.15) | −68.2..39.2 °C | no | instant | yes | continuous | f32 |
| relative_humidity_2m | `2r` | %→% | 5.7..100.0 %(no clipping) | no | instant | yes | continuous | f32 |
| precipitation_amount_3h | `tp` | kg m⁻²→mm | 0..306.4 mm | **yes**(deaccum) | **NaN** | yes | continuous | f32 |
| precipitation_rate | `prate` | kg m⁻² s⁻¹→mm/h(×3600) | 0..76.7 mm/h | no(conversion only) | instant | **GEFS unsupported**(`idx_parser.py:726-732`) | continuous | f32 |
| wind_u_10m / wind_v_10m | `10u`/`10v` | m/s→m/s | ±61.5 m/s | no | instant | yes(vector) | continuous | f32 |
| wind_gust | `gust` | m/s→km/h(×3.6) | 0..230.8 km/h | no | instant | yes | continuous | f32 |
| visibility | `vis` | m→km(÷1000) | 0.02..24.1 km | no | instant | yes | continuous | f32 |
| snow_depth | `sde` | m→m | 0..2.38 m | no | instant | yes | continuous | f32 |
| cloud_cover_3h | `tcc` | %→%(clip 0-100) | 0..100 % | **yes**(reconstruction) | **NaN** | yes | continuous | f32 |
| cloud_ceiling | `gh`@cloudCeiling | gpm→km(÷1000) | 0..20 km(sentinel 19.99=unlimited) | no | instant | yes(unlimited probability + conditional quantile) | continuous + discrete sentinel | f32 |
| crain/csnow/cfrzr/cicep | same name | Code 4.222→**flag(uint8)** | 0/1 | no(lead-0 zeroed) | **all 0 uint8** | yes(phase companions) | **categorical** | f32(⚠ see §18) |
| wind_speed | — | — | — | — | — | — | **serving-side derived**(`point_forecast.py:844` hypot;`ensemble_data.py:301`) | not stored |

ingestion has only two derived variables(precip, cloud), with no wind speed/dew point/apparent temp computation(confirmed by grep).**Do not assume precipitation is the only special variable—cloud_cover_3h is likewise a reset/reconstruction variable**(§7).

---

## 5. Float16 Numerical Error Analysis (Per-Variable, Measured on Real Data)

IEEE-754 binary16 half ulp resolution table(verified by measurement):

| Value range | spacing | half ulp(max quantization error) |
|---|---|---|
| [0.001, 0.002) | 9.5e-7 | 4.8e-7 |
| [0.0625, 0.125)(contains the 0.10 threshold) | 6.1e-5 | 3.05e-5 |
| [1, 2) | 9.77e-4 | 4.9e-4 |
| [16, 32)(typical wind speed/precipitation) | 0.0156 | 0.0078 |
| [64, 128)(large accumulation) | 0.0625 | 0.03125 |
| [128, 256) | 0.125 | 0.0625 |
| [256, 512)(306mm extreme) | 0.25 | 0.125 |

Real fields(Canonical units, float32→float16→float32 round-trip):

| Variable(unit) | measured range | max abs | MAE | P99 | 1dp display flip | threshold flip |
|---|---|---|---|---|---|---|
| temperature(GFS)°C | −68.2..39.2 | 0.0273 | 0.0031 | 0.0148 | 24.8% | — |
| temperature(GEFS)°C | −60.3..40.2 | 0.0133 | 0.0030 | 0.0133 | 7.1% | — |
| RH(both models)% | 5.7..100.0 | 0.0301 | 0.0148 | 0.030 | 0% | 95%/50% both 0 |
| wind u/v(GFS)m/s | ±61.5 | 0.0145 | 0.0008 | 0.0036 | 0% | calm wind 0.5m/s:0 |
| wind u/v(GEFS)m/s | ±27.7 | 0.0078 | 0.0007 | 0.0037 | ≤2.2% | 0 |
| gust km/h | 0.03..230.8 | 0.0612 | 0.0052 | 0.0263 | 7.0% | 60km/h:0 |
| precipitation(GFS)mm | 0..306.4 | **0.125** | 1.2e-6 | **0** | 0.001% | 0.10mm:0 |
| precipitation(GEFS)mm | 0..136.3 | 0.050 | 1.0e-4 | 0.0016 | 0% | 0 |
| visibility km | 0.02..24.1 | 0.0078 | 0.0044 | 0.0066 | 0.40% | 1km fog:0 |
| snow_depth m | 0..2.38 | 0.00097 | 3.0e-5 | 0.00044 | 0.03% | — |
| cloud cover(GFS)% | 0..100 | 0.0250 | 0.0040 | 0.025 | 0% | 0 |

Comparison baselines:
- **Upstream GRIB packing precision**:measured GFS APCP packing quantization is **0.125 mm**(all values are integer multiples of 0.125, such as 306.375).float16's half ulp is ≤0.0625mm for ≤256mm,**below one upstream packing quantum**; the two are equal only in the [256,512) interval(0.125).
- **Display precision**:the API/frontend renders at 1 decimal place(°C, km/h, %)or 2(NumberFormat); the float16 error is 3-20 times smaller than the display granularity.
- The only "visible" effect is rounding-boundary flips of the 1dp display value(temperature at most 24.8% of grid points differing by 0.1°C)—the same order as the existing 0.1 display rounding, and it does not change semantics; it does not affect variables covered by the `round()` contract(e.g. unrounded JSON fields, see §19/§21).

---

## 6. Precipitation De-accumulation Analysis

**Implementation**(`pipeline.py:302-370`):`curr−pred` explicit float64(331-332)→ keep non-negative, clamp `[−0.50,0)` to 0(`DEACCUMULATION_CLAMP_BOUND_MM=0.50`,`base.py:91-92`), `<−0.50` set to NaN → float32 output(339).Predecessor:when lead%6==0 take `lead−3`.

**Predecessor source(key fact)**:the production wave path uses an **in-memory copy** `raw_precip_for_future = np.copy(ds["tp"].values)`(`wave_runner.py:1358-1375`, float32, bypassing storage);`read_predecessor_precipitation`(`pipeline.py:373-420`)is only the fallback/library path.**Therefore the float16 storage migration does not change the production path's subtraction inputs at all**—the only things changed are:① the storage quantization of the final increment;② the fallback path reading back an f16 predecessor.

**Reset semantics**:APCP resets every 6h, and both sides of the subtraction are 3h/6h **window** accumulations(not since-run totals), with an upper magnitude bound of an extreme 6h rainfall.Measured GFS maximum in a single run 306.4mm(hurricane), GEFS 136.3mm.

**Experiment design**(mirrored implementation, including real lead pairs + adversarial constructions):

| Scenario | Architecture | max err vs current state | MAE | Semantic flip |
|---|---|---|---|---|
| real f189→f192(130→237mm) | B1:f16 predecessor storage→f32 subtract→f16 store | 0.0625 mm | 1.8e-7 | **0**(P99=0) |
| real f003→f006(180→306mm) | B1 | 0.0625 mm | 3.6e-7 | 0 |
| the same two pairs | B2:f16 storage→f64 subtract→f16 store | 0.0625 | 3.6e-7 | 0 |
| real two pairs | C:f16 direct subtraction(negative reference) | 0.1875 | 1.4e-6 | 0(only 14 1dp display flips) |
| **production path**:predecessor in memory f32 → f64 subtract → f32 result → **only the final increment stored as f16** | D | 0.0625 mm | 1.8e-7 | **0**(1M grid points, dry/wet, 0→pos, NaN all zero) |
| adversarial:curr=pred+0.125mm, pred∈[32,131)mm | B1 | 0.0625 | 8.0e-4 | dry/wet 1 cell / ~460k wet cells(~2e-6):the f16 predecessor at [64,128) with half ulp=0.03125 erodes the 0.125−0.10 margin |
| adversarial:curr=pred(true increment 0) | B1/B2 | 0.0625 | 4.0e-4 | 0→pos 1 cell(stores 0.031mm, below the trace threshold, product still dry) |
| adversarial:residual exactly −0.5mm(clamp boundary) | B1/B2 | — | — | **valid→NaN 1 cell / 1e6**:`pred_stored=pred+0.03125` makes −0.5−δ<−0.5 |
| adversarial:600.125−600.0 / 1100.125−1100.0 | B1/B2 | 0 | 0 | 0 |
| same as above | C(negative) | 0.125 | 0.125 | **100% increment loss**(600.125 cannot be represented by f16→difference 0) |

**Conclusions**:
1. **float16 storage + float32/float64 subtraction is safe**.The production path(in-memory predecessor)zero flips; the fallback path zero flips on real data and, in 10⁶-scale adversarial sampling, shows 2 classes of boundary events at a ~1e-6 rate(dry/wet boundary flip, clamp-boundary NaN).
2. The dominant error term in subtracting large numbers is the **f16 predecessor quantization**(relative error 2⁻¹¹), not the subtraction itself; the computational precision contribution of float64→float32(<1e-5 mm)is negligible—B1 and B2 give identical results.
3. **Do not do the subtraction in float16**(architecture C loses small increments wholesale above 512mm).
4. The clamp-boundary event(true residual = −0.50 exactly + predecessor rounding upward)can turn "clamped to 0" into "NaN invalidated", at a rate of ~1e-6, affecting a single cell only; to eliminate it entirely, v2 could widen the clamp bound by one predecessor quantization step(e.g. −0.5−ulp_f16(pred)), which is an implementation-time decision and does not affect product semantics.

---

## 7. Other Derived / Reset Variables (cloud and Other Special Variables)

**cloud_cover_3h reconstruction**(`cloud.py:103-171` + `pipeline.py:588-688`):formula `C_3h = (R·C_R − (R−W)·C(L−W))/W`, with R=6, W=3 giving `2·C6 − C3`; inputs explicitly float64(`cloud.py:160-161`), guard ±5 percentage points(`CLOUD_COVER_RECONSTRUCTION_TOLERANCE_PERCENT=5.0`,`cloud.py:24`):[0,100] kept, [−5,0)→0, (100,105]→100, out of range NaN; cast back to float32(188-189).Predecessor = read back from storage(`read_predecessor_cloud_cover`, f32)or an in-memory copy(`wave_runner.py:1364-1368`).

**float16 propagation analysis**:error = |2·δ(curr)| + |δ(pred)| ≤ 2×0.031 + 0.031 ≈ **0.094 pp worst case**(f16's half ulp in [64,128)), while the guard band is ±5pp, display precision 0.1%, ensemble min_valid 21—**a safety margin of 4 orders of magnitude**.The clip(0,100) of the lead%6==3 direct branch is unaffected by f16(the boundary values 0/100 are exactly representable).**Conclusion:cloud reconstruction is entirely safe**, and the predecessor error is amplified only once(no recursion—the predecessor itself is a 3h direct value).

**Other reset/derived/threshold-sensitive variables, item by item**:
- precipitation_amount_3h lead-0 NaN(f32 NaN → f16 NaN faithful,§17); flags lead-0 uint8 zero—unchanged.
- cloud_ceiling:sentinel ≥19.99 km(`cloud.py:28`); f16's half ulp in [16,32)is 0.0078 km, so a 19.99±0.0078 boundary flip requires the true value to fall inside a 7.8m narrow band; the measured range is 0-24km, with no overflow.**Recommendation**:v2 keeps the 19.99 sentinel semantics unchanged(f16 can exactly represent 19.984375/20.0, note that 19.99 itself is not an f16-exact value—the comparison happens in the float64 domain(`point_forecast.py:589`), so after read-back it is compared consistently with the current state, no risk).
- wind_speed:serving-derived(hypot); u/v each have f16 error ≤0.014 m/s → speed error ≤~0.02 m/s(convex combination),§9 measured mean-speed max 0.0097 km/h.
- Phase classification(`precipitation.py:143-314`):inputs amount + uint8 flags(≥0.5 determination,`point_forecast.py:899`)+ t2m; the only sensitive point under f16 is the comparison of amount against TRACE 0.10mm—§6 measured flips ~2e-6(adversarial constructions only), and zero on real lead pairs.
- No other accumulated/reset variables(confirmed by grep).

---

## 8. GEFS Ensemble Statistics Analysis

**Architecture facts**:mean is an **upstream official geavg product**(not platform-computed,`connector.py:181-185`; member-wise nanmean is only a fallback with no callers,`zarr.py:651-689`).Domain statistics **force float64**(`_validation.py:43` `np.asarray(members, dtype=np.float64)`), std ddof=0, percentile method="linear"(`statistics.py:75,109`, fixed by tests `test_ensembles.py:38-48`).serving reads 30 members per request(member executor 4 workers,`zarr.py:567-581`),~1 chunk GET per member per variable.

**Experiment**(30 real member t2m fields, f120; member storage f32 vs f16; two computation paths f64 and f32):

| Statistic | f16 storage→f64 computation:max / MAE / P99(°C) | Note |
|---|---|---|
| mean | 0.00722 / 0.00053 / 0.00259 | |
| median | 0.01529 / 0.00200 / 0.0100 | |
| std(spread) | 0.00719 / 0.00052 / 0.00252 | |
| P10/P25/P75/P90 | ≤0.01548 / ~0.0027 / ≤0.0123 | |
| IQR | 0.02933 / 0.00318 / 0.0155 | |
| **f16→f32 computation**(tile fallback path) | difference vs f64 computation max 1.8e-5 | **computational precision contribution negligible** |

- **Classification flips**:0°C freezing line(mean/median)**0 flips**; P90≥30°C 3/259,920(0.001%); P10≤0°C 0; spread 1dp display flips 0.5%(one 0.1°C step).
- **Member ordering**:f16 quantization makes the sorted sequence have ties/transpositions at 100% of grid points—but the quantiles are self-consistent on the quantized sample, with an error upper bound = the quantization half ulp(measured ≤0.0155°C).**No percentile method shift**.
- **phase support / transition**:flags are a uint8/integer determination(≥0.5),**entirely unaffected** by storage precision; the joint phase in which amount participates(`compute_joint_amount_phase_support`)is covered by the threshold analysis of §6(0.10mm flips ~0).
- **PDF**:Gaussian KDE float64, 100-point grid(`pdf.py:41-119`); a member value perturbation of ≤0.015°C changes the density curve far less than the rendering resolution.There is no separate IQR product(P25/P75 only feed the bandwidth).
- **Conclusion**:with float16 member storage + float32/64 computation, the ensemble products are **materially equivalent**(all deviations ≪ ensemble spread and display precision).

---

## 9. Point and Map Serving Analysis

**Point interpolation**(`zarr.py:457-532`):2×2 neighborhood;1 chunk-GET within a chunk(+1 tail index GET), at most 4 across chunk boundaries; corner values are **immediately converted to Python float**(`float(arr[...])`,`zarr.py:508-511`), bilinear kernel pure Python float(`zarr.py:133-149`).**No need to promote a whole chunk's dtype—the read path naturally promotes only the 4 scalars to float64**; for an f16 chunk the corner error = half ulp, the bilinear weights sum to 1, and the convex combination does not amplify.

Measured(real fields,200k random points):

| Field | Interpolation error max / MAE / P99 | 1dp display flip |
|---|---|---|
| temperature(°C) | 0.0253 / 0.0021 / 0.0102 | 3.3% |
| precip(mm) | 0.0647 / 8.7e-7 / 0 | 0 |
| wind-u(m/s) | 0.0074 / 0.0005 / 0.0030 | — |

The point cache stores exactly those 4 Python floats(dtype-independent of storage).

**Map/Window**(`tiles.py:986-1124`):`read_window` allocates a float32 window(`zarr.py:612`), per-chunk GET in parallel(2 workers); tiles immediately promote at the boundary to float64(`tiles.py:718-720,1087,1101`)→ float64 `np.interp` color mapping(773-783)→ uint8 RGBA.**The deterministic map can stay float16-resident throughout until the float64 promotion in tiles**(color-band mapping needs a continuous domain, but a 0.03% input error is invisible at the color-band stop resolution).Wind tiles:float32 hypot×3.6 → float64(`tiles.py:1068-1070`).ensemble map:the official mean shard is read directly(after f16 likewise ≤0.03% color-band shift); member-stack fallback `read_ensemble_mean_window` uses float32 nanmean(§8 proves the f32 vs f64 computation difference is 1.8e-5).**Conclusion:deterministic windows can be f16-resident; the ensemble computation boundary must be promoted to float32/64—exactly the split of Architecture B**.

---

## 10. API Memory Analysis (Byte-Level)

Configuration:`API_READER_MAX_CACHED_CHUNKS=512`, `API_READER_MAX_CACHED_INDICES=16384`, `API_READER_MAX_CACHED_POINTS=32768`(all **per-reader**,`zarr.py:241-245`);`MAX_READERS=8`(`zarr.py:704`); uvicorn 4 workers(compose).

| Cache | Per-entry content | Current/reader | Candidate(f16)/reader | Change |
|---|---|---|---|---|
| `_chunk_cache`(512) | 100×100 ndarray:40,000B payload + ~96B object header | **19.58 MiB** | **9.83 MiB**(20,096B) | **−9.77 MiB(−50%)** |
| `_index_cache`(16384) | (120,2) uint64 = 1,920B + header | ~31.3 MiB | unchanged | 0 |
| `_point_cache`(32768) | 9-tuple key + 4 Python floats(~400-800B) | ~13-26 MiB | unchanged(Python float) | 0 |
| **reader total(est.)** | | **~64-77 MiB** | ~54-67 MiB | **−10~15%** |
| **per worker(×8 readers)** | | **~0.5-0.6 GiB** | ~0.44-0.53 GiB | **−78~95 MiB** |

- The existing comment "`~40 KB each`, ~20 MB/reader"(`config.py:157-166`)is still accurate; but the `~0.8 KB/entry` point cache estimate(`config.py:188-207`)and the "~2 KB" index estimate show that **the chunk payload accounts for only ~25-30% of the reader cache**—the claim "API cache memory halved" does not hold; the accurate statement is **chunk-payload halved, reader total cache −10~15%, worker total cache −78~95 MiB**.
- The "2048 chunks" in docs/ARCHITECTURE.md:320 is a stale value(the implementation is 512).
- The float64 assembly of window/tile is a request-lifecycle temporary and is unrelated to the dtype migration(under the candidate the input is halved, the temporary float64 window unchanged).

---

## 11. Ingestion Memory Analysis

Concurrency structure(`config.py:133-172`, `wave_runner.py:371`):download 8 / **decode 2** / write 4, staging upper bound = 14 items; wave ≤8 leads; GEFS wave = (1 mean + 30 members)×leads, lead-major.Cross-process handover is a **full pickle Dataset**(~60 MB float32/file,`wave_runner.py:1727-1729`).Predecessor copies tp+tcc ≈ 2×8.3MB.

**Under the candidate architecture(decode/normalize kept at float32)**:the big resident items(decoded Dataset, pickle, predecessor copies)are **all unchanged**; the only changes are the shard encoding buffer(120×100×100:4.61MB→2.30MB/shard, f16)and the PUT bytes.**ingestion RSS net saving <2-3%, not worth using as the migration motive**; nor should decode/normalize be turned float16 ahead of time to save memory(it would break the float64 subtraction contract of §6 and the float64 statistics inputs of §8).The constraint *do not f16-ify prematurely to save memory* is quantitatively confirmed here.

---

## 12. Compression Benchmark (Real Data, Zstd level 5 = Same numcodecs Settings as the Writer)

**Chunk granularity(the real Range GET unit,100×100)**:

| Field | f32 compressed/chunk | f16 compressed/chunk | Saving | f32 ratio | f16 ratio |
|---|---|---|---|---|---|
| temperature | 7.8 KB | 6.4 KB | **18.7%** | 5.1× | 3.1× |
| RH | 11.2 | 9.7 | 13.1% | 3.6× | 2.1× |
| wind u / v | 9.6 / 9.6 | 7.7 / 7.6 | 19.9% / 20.5% | 4.3× | 2.6× |
| gust | 9.0 | 8.0 | 11.2% | 4.6× | 2.5× |
| prate | 6.6 | 5.2 | 21.9% | 6.1× | 3.8× |
| precip(3h, GFS/GEFS) | 4.4 / 4.2 | 3.5 / 3.2 | 18.7% / **23.6%** | 9.5× | 5.4× |
| cloud(GFS/GEFS) | 8.6 / 6.9 | 6.8 / 5.7 | 21.1% / 17.6% | 4.7× | 3.1× |
| visibility | 8.0 | 4.4 | **45.4%** | 5.0× | 4.5× |
| snow_depth | 5.1 | 2.7 | **46.7%** | 8.1× | 7.7× |

**flags(currently hard-coded as f32,`zarr_writer.py:658`)**:crain f32 100.3KB/shard-chunkset → **uint8 68.5KB(−31.7%)**; csnow −26.6%; cfrzr/cicep near zero(-9.0%/-5.2%); the four flags total **−29.6%**.Turning flags into f16 saves only a further 5-11% and is semantically wrong—**must be uint8**.
(Whole-field granularity numbers show the same trend:temperature 0.86→0.71MB etc.; encode/decode times in §14.)

**Key counter-intuitive conclusion:halving the payload ≠ halving after compression**.f32 itself compresses well(4-9×), and the compressed size is dominated by the entropy of the mantissa noise; measured continuous-variable savings are **11-47%(mostly 13-24%)**, so capacity planning must not assume 50%.

---

## 13. Object Store and Range GET Savings

Geometry:GFS 15 vars × 81 leads × 1 det = 1,215 shards/cycle; GEFS 14 vars × 81 × (30 mem + 1 mean) = 35,154 shards/cycle;120 chunks per shard.Measured per-chunk means(bench4)weighted:

| Object | f32 current state | f16+u8 candidate | Saving |
|---|---|---|---|
| GFS / cycle | 0.83 GB | 0.64 GB | −22.2% |
| GEFS / cycle | 23.70 GB | 18.43 GB | −22.2% |
| Active window(GFS ~40 runs + GEFS ~20 runs, derived from the 0-240h horizon + canonical latest selection,`planner.py:1-20`) | **~507 GB** | **~394 GB** | **−113 GB(−22.2%)** |

**Range GET transfer**(request count unchanged, bytes down):
- point(GFS):1 chunk(7.8→6.4KB)+ index tail(1.92KB, unchanged)→ −18%;
- point(GEFS reads the mean shard):same as above;
- ensemble statistics:30 chunk GETs 235→191KB(−19%); precip phase:180 chunk GETs(~30 members×6 fields)≈ 1.2→0.97MB;
- map tile:a single chunk for z≥4, up to 120 chunks at low zoom;
- **The debt of GEFS cold-serving is fundamentally request-count and cold-index RTT, not bytes**—f16 reduces bytes per request by ~19% and the decompression CPU(payload halved), but the **Range GET count is completely unchanged**; one cannot claim latency −50%.For cold latency, the decode time(in §14 frombuffer f16 is actually 2× faster)and the transfer-byte benefit are second-order improvements.

---

## 14. CPU / x86 / ARM64 Analysis

Measured platform:AMD Ryzen 7 5800X(Zen 3, x86-64, AVX2,**no native FP16 arithmetic**; ARM64 NEON has native f16 vectors but NumPy is currently likewise non-vectorized).numpy 2.5.3:

| Operation(4M elements) | f32 | f16 | f16 storage→f32 computation | Conclusion |
|---|---|---|---|---|
| subtraction | 2.94 ms | **25.6 ms(8.7× slower)** | 3.48 ms | f16 computation unacceptable |
| multiplication | 2.55 | 25.0(9.8×) | — | same as above |
| mean | 2.38 | 6.79(2.9×)⚠**overflow** | 2.12 | f16 reduction numerically dangerous |
| std | 8.92 | **61.8(6.9×)** | — | same as above |
| percentile | 31.3 | 38.1(1.2×) | — | sort-dominated |
| astype f32→f16 | — | 10.9 ms | — | writer quantization cost ~370M elem/s |
| astype f16→f32 | — | 8.0 ms | — | reader promotion cost |
| frombuffer+copy | 2.34 ms | **1.18 ms(2× faster)** | — | **f16 chunk decode is faster**(bytes halved) |
| hypot | 12.8 | — | — | keep f32 |

- **Measured f16 reduction overflow**:on a 4M-element N(10,20)sequence,`a16.mean()` triggers `overflow encountered in reduce`(the f16 accumulator overflows to inf)—f16 computation is not only slow but **numerically destructive**.
- **The target architecture should clearly position float16 as a storage/cache/bandwidth optimization format, not a compute format**—this benchmark is direct evidence.NumPy 2.x's f16 path on ARM64 is likewise non-vectorized(until an f16-vectorized stack is adopted, the conclusion holds across architectures).

---

## 15. Float64 Audit

| float64 usage site | Location | Classification | Disposition |
|---|---|---|---|
| de-accumulation subtraction | `pipeline.py:331-332` | **Required**(numerical margin for the accumulation-reduction contract + the 0.5mm clamp boundary;§6 proves its output is the reference baseline) | **retain** |
| cloud reconstruction arithmetic | `cloud.py:160-161` | **Required**(same source as above; conditionally cast back to float32 at 188-189) | **retain** |
| domain unified coerce | `_validation.py:43` | **Required**(statistics contract, tests assert exactly with np.float64,`test_serving_selection_shape.py:210`) | **retain** |
| ensemble statistics/pdf/wind stats | `statistics.py`, `pdf.py`, `wind.py:221-506` | Required(same contract as above) | retain |
| tile window promoted to f64 | `tiles.py:718-720` and 10+ other places | Probably unnecessary for correctness(f32 suffices for color-band mapping)but **out of scope for this migration**, and changes would only enlarge the regression surface | retain unchanged |
| 2×2 interpolation window | `point_forecast.py:1856`(legacy path) | Probably unnecessary | retain |
| coordinates/geometry | `tiles.py:629-659`, grid math | Non-impacting(scalar/metadata magnitude) | retain |
| inaccurate type annotation | `pipeline.py:252` annotates float64 but it is actually float32 | documentation error | suggest fixing the comment in passing |
| JSON expansion | `point_forecast.py:603` float32→float64 text | Non-impacting but see §19 | see §21-2 |

No float64 was found that can be safely deleted(deletion benefit ≈0, regression risk >0).

---

## 16. Storage Format Versioning (Recommended Compatibility Architecture)

**Recommendation:Option B — `sharded_v2`, per-variable native dtype, version carried in the manifest.**

- `STORAGE_FORMAT_VERSION`(ingestion `config.py:236`, currently `"sharded_v1"`)→ writes `"sharded_v2"` into the manifest(`coordinator.py:987,1256`).**No DB migration whatsoever**(the catalog has no format column; per-cycle store isolation;`model_runs` has an immutable store path,`catalog.py` immutable).
- v2 semantics:**continuous variables f16(little-endian `<f2`, explicit endianness, fixing v1's native-endianness ambiguity), flags u8(det/mem shard), mean-shard flags kept f32(member-mean probability 0..1), NaN sentinel unchanged**; index/trailer/object key/lifecycle completely unchanged(geometry is dtype-independent,`zarr.py:406` takes the offset at runtime).
- reader:the `manifest_storage_format()` branch already exists(5 call sites);`ShardedV1Reader.read_chunk` selects the frombuffer dtype by format and **uniformly materializes to the internal dtype**(point/window immediately in the f32/f64 domain; the chunk cache may keep f16—see §10).Recommended is a new reader class `ShardedV2Reader` reusing `ShardedV1Reader`'s index/cache machinery, rather than adding an if inside the v1 class—the v1 path stays frozen, reducing the regression surface.
- **Old cycles stay readable as v1, new cycles write v2, old data is retired naturally by lifecycle—zero bulk migration**:canonical selection reads only the "latest serviceable cycle", and a v1 cycle is served by the v1 branch while inside its active window, until GC reclaims it(`planner.py`; the reclamation queue is fenced against the `physical_key` and `store_generation` baseline,`worker.py:184-232`).
- Precondition:`STORAGE_FORMAT_VERSION` **must not be switched** between the first write of a cycle and finalization(`coordinator.py:1256` stamps every per-lead; manifest 30s forward cache,`manifest_reader.py:39-43`).

---

## 17. NaN / Inf / Fill / Missing Semantics

- The platform has **no `_FillValue`**; missing = index length==0 → NaN fill chunk(`zarr.py:407-408`); writer NaN-fill buffer(`zarr_writer.py:658`); validity is always an `np.isfinite` mask; the domain rejects non-finite inputs(`_validation.py:55`); NaN→JSON null(`point_forecast.py:554`).
- **Measured f16 round-trip**:NaN faithful(bytes 0x007e), ±Inf faithful, ±0.0 sign faithful;**70000.0 → inf(RuntimeWarning overflow in cast), 65519→65504(saturation), <6e-8 underflows to 0,6e-6 faithful as subnormal**.
- Overflow audit against the current variable inventory:measured max(306.4mm, 230.8km/h, ±61.5m/s, 24.1km, 2.38m, 100%, 39.2°C)are **all ≪ 65504**; PRATE 83.9mm/h and ceiling 20km are safe.**But the f16 range check should be a writer-side assertion**(on upstream outliers/unit errors astype silently produces inf, and the `np.isfinite` validity check would discard inf as invalid—semantics can be preserved, but a warning should be emitted at encode, to prevent the silent surface from growing).
- Search for 4-byte/byte-pattern assumptions:only `zarr.py:421`(frombuffer f32, the v2 modification point), `png.py:63,76,99`(RGBA 4B/px, unaffected), `wind.py:722`(int16 vector payload ×2, unaffected), `client.ts:436-442`(vector header getFloat32, unaffected).index/trailer math is dtype-independent.

---

## 18. Categorical Variables (Separate Analysis)

- uint8 after normalization(`pipeline.py:292-298`); lead-0 uint8 zero(`pipeline.py:488-499`);**but the sharded writer merges them back into f32 storage**(`zarr_writer.py:658`), while the `.zarray` metadata records uint8(`zarr_writer.py:373`)—**metadata and bytes are inconsistent**.
- There is a second hole:if the GRIB units token is exactly `"flag"`, the value short-circuited by unit conversion **stays float32**(`pipeline.py:766-767` short-circuit logic)—the normalization dtype depends on an upstream attribute and is unstable.
- **Quantization(measured, f32→u8, Zstd-5)**:crain −31.7%, csnow −26.6%, cfrzr −9.0%, cicep −5.2%, the four shards total **−29.6%**; raw reduced by 75%.
- **Exception—the flags of the GEFS mean shard are member-mean probabilities**(0..1 float32, such as 0.65,`test_precipitation_phase_companions.py:268`; the phase products consume the fractional flags,`point_forecast.py:899` ≥0.5 determination).The mean shard's flags **must stay float**(f16 is also acceptable:in the 0..1 interval the half ulp is ≤0.00049, and after rounding to the 4dp output there is no difference; conservatively the first v2 version may keep mean-shard flags at f32).
- serving outputs `round(float(x), 4)`(`point_forecast.py:575`)+ frontend `>= 0.5` determination(`FE/lib/forecast/precipitation.ts`)—the u8/det branch storing 0/1 is fully faithful.

---

## 19. Redis / JSON / Frontend

- **Redis**(`cache.py:199`):the **JSON text** of `model_dump_json()`; float16 only changes the decimal literals in the text, while the cache mechanism, key(sha256 of canonical JSON), TTL, corruption detection(schema validate)are all dtype-independent.**There is no such thing as "float16→Redis automatically halved"**.vector L2 is int16-quantized raw bytes(`vector_field.py:198`), completely isolated from the storage dtype.
- **JSON(Pydantic)**:all schemas are `float | None`; the conversion points are all `float(...)`(no `.item()`).**Two classes of fields**:
  1. **API boundary already rounded**(dtype-independent):direction/cloud 1dp, flags/probability/coverage 4dp, consensus 2dp etc.(agent 4 §7 table);
  2. **Unrounded pass-through**:the temperature_2m/precipitation_rate/RH/wind_gust/visibility/snow_depth/cloud_ceiling of `point_forecast.py:603-606` are output as float32→float64 **expanded text**(the current state already has noise tails like `15.100000381469727`).After f16 storage these literals become f16 expansions(such as `15.1015625`).**JSON semantics unchanged, byte text changed**; no impact on the `Number.isFinite` gating and Intl rendering, but any test/client that golden-file-asserts responses will see a difference.
- **Frontend**:fetch JSON(`client.ts:71,86`); all decimal handling is render-time formatting(toFixed/Intl,`labels.ts:86-117`); the client statistics replicate ddof=0/linear percentile(`transform.ts:273-301`); flags `>=0.5`, dry `<=0.05`, calm wind `hypot<0.5`(int16 quantization domain,`windParticles.ts:14`)—**these threshold comparisons happen on JSON Numbers and are independent of the storage dtype**(except for the display-rounding item of §21-2).vector binary decoding(`client.ts:404-478`)is unrelated to f16.**There is no such thing as "frontend memory automatically halved"**.
- **presentation rounding and storage precision are already naturally separated**(API round + FE Intl); this round **introduces no global round/truncate**—§21-2 is the only display-layer item requiring a product decision, and keeping the current state is recommended.

---

## 20. Backward-Compatible Rollout Plan (Design Only, No Implementation)

**Phase 0 — upfront hardening(1 small PR worth)**
- writer-side f16 range assertion + pre-encode non-finite check(§17);
- fix the flags normalization short-circuit hole(`pipeline.py:766` forced to u8);
- (optional)parameterize the clamp-boundary compensation(§6-4).

**Phase 1 — dual reader(read first)**
- add `ShardedV2Reader`(or a dtype-table parameter on the v1 class), with the `manifest_storage_format()` branch fully covering the 5 call sites; the v1 path has zero changes;
- chunk cache materialized by manifest dtype(continuous variables cached as f16);
- tests:v1 fixtures all green + v2 fixture equivalence assertions(§18 test plan).

**Phase 2 — dual writer(write later)**
- `STORAGE_FORMAT_VERSION=sharded_v2` staged rollout:GFS first(det, lowest risk)→ GEFS mean → GEFS members;
- switching forbidden within the same cycle(settings snapshotted at startup);
- writer integrates the dtype table at `zarr_writer.py:658`(f16/u8/f32-mean-flags).

**Phase 3 — mixed serving validation**
- production shadow comparison(§19):the same request goes to a v1 old cycle and a v2 new cycle, per-variable diff thresholds(the measured upper bounds of the §18 table);
- monitoring:per-format shard counts, Range GET bytes, API RSS, clamped/invalidated counts(existing QC logs `pipeline.py:355-368`).

**Phase 4 — natural retirement**
- v1 cycles are GC-reclaimed as canonical retires them(`planner.py`, reclamation fenced against the `store_generation` baseline); no bulk migration, no format-conversion job;
- the v1 read branch is **retained permanently**(historical tombstones/cycles may be long-lived)or removed after N releases per team policy.

**Rollback**:zero risk before Phase 2; after Phase 2, switching the env back to `sharded_v1` stops v2 production(already-written v2 cycles keep being served via the Phase 1 reader); no data writeback needed.

---

## 21. Required Tests / 19. Real-Data Validation (Merged into a Validation Plan)

**Unit**(new):
- f16/u8/f32 serialize round-trip(including NaN, ±Inf rejection policy, ±0.0, 65504 saturation, underflow); v2 index/trailer invariance; manifest `sharded_v2` read/write; mixed-dtype reader(v1+v2 in the same process).
- **Numerical regression(per-variable, asserting the measured upper bounds of this report)**:for each continuous variable, the f16 round-trip error ≤ the §5 table max value ×1.2.
- **Precipitation**:normal deaccum; tiny increments(1 quantum); large accumulation + small increment(three levels 64/128/256mm); reset boundaries(lead 3/6/9); three negative-residual segments(≥0 / clamped / NaN);**predecessor read from f16 storage**(fallback path)+ **predecessor in memory f32**(production path)both branches; adversarial AC1-AC5 reproduction(1e-6-level events allowed only at the clamp boundary, and must not occur in batches).
- **Cloud**:reconstruction ±5pp three segments; f16 predecessor propagation ≤0.1pp; tolerance boundary values.
- **Ensemble**:mean/median/std/P10/P25/P75/P90/IQR against the f32 baseline max ≤0.03°C(30 members); phase support uint8 invariance; joint amount-phase threshold flip rate 0.
- **Interpolation**:2×2 analytic field f32 vs f16 input max ≤ half ulp×1.2; the exact float64 contract of `test_serving_selection_shape.py:210` **holds for v1 only**, and v2 defines a new contract.
- **Mixed-format**:v1 cycle + v2 cycle served mixed within the same request(extension of the cross-cycle fallback paths `test_cross_cycle*.py`).
- **Contract updates(must be synchronized)**:float32 fixtures(`test_serving_chunk_equivalence.py:18-27` etc.), `abs=1e-9` bit equivalence(changed to v1-only or per-format), `abs=1e-5` deaccum/interpolation tolerance(v2 relaxed to the quantization step + 1e-5).

**Real-data validation(plan, using this report's pipeline)**:for 1 complete GFS + 1 complete GEFS cycle, output per-variable/per-lead max/MAE/P50/P95/P99 absolute error and max relative error; for GEFS products separately output mean/median/std/P10/P25/P50/P75/P90/IQR/phase-support%; check threshold/classification/member-ordering changes;**worst-case sample table**(variable/cycle/lead/member/grid index/baseline/candidate/downstream difference)—the measured tables of §5/§6/§8 of this report are the first run results of that pipeline, and it should be re-run on the target production hardware before implementation.

---

## 20. Quantified Expected Benefits (Summary of Measured Evidence)

| Dimension | raw theoretical | post-Zstd measured | whole-process actual |
|---|---|---|---|
| Object storage | −50%(payload) | **−22.2%**(GFS 0.83→0.64GB/cycle; GEFS 23.7→18.4GB/cycle; active ~507→394GB) | same as left(object storage is the final state) |
| Range GET bytes | −50% | **−18~19%**/request; request count −0% | cold latency:second-order improvement(transfer + decompression), not 50% |
| API chunk cache payload | −50% | −50%(19.58→9.83 MiB/reader) | reader total cache **−10~15%**; per worker −78~95 MiB(8 readers) |
| ingestion RSS | ~0(decode kept f32) | encoding buffer −50%(2.3MB/shard) | **<2-3%** |
| chunk decode CPU | — | frombuffer+copy **2× faster** | small gain on the serving hot path |
| writer CPU | — | astype f32→f16 +10.9ms/4M(~370M/s) | negligible increment for ingestion encoding |
| Network egress(MinIO→API) | −50% | ~−20% | scales with daily request volume |

---

## 22. Recommended Target Numerical Policy

```yaml
decode / normalize / derived (deaccum, cloud):
  float32 storage, float64 only inside the two derived functions(current state retained)
continuous persistent forecast fields (v2 shards):
  float16 (little-endian <f2)
categorical fields (det/mem shards):
  uint8 (native; fix the inconsistency currently merged back into f32)
ensemble mean-shard flags:
  float32 (member-mean probability 0..1; may later be lowered to f16)
decompressed API chunk cache:
  float16 / uint8 (native dtype, per manifest)
interpolation / ensemble statistics / reductions / tile color mapping:
  float32 domain to start from(the domain contract is actually float64, retained)
float64:
  only where the current state has proven it necessary(pipeline.py:331-332, cloud.py:160-161, _validation.py:43)
JSON / presentation:
  independent of storage dtype; no new round; keep the existing round contract
float16 arithmetic:
  forbidden(measured 8.7× slower + reduction overflow)
```

## 23. Go / No-Go Preconditions

1. The product side accepts 1dp display values flipping at f16 quantization boundaries(temperature at most ~25% of grid points differing by 0.1°C, mean square deviation ≤0.027°C)and the unrounded JSON literal text change(§19)—**real blocker(product decision)**.
2. f16 range/non-finite writer assertions land(§17)—**real blocker(prevent silent inf)**.
3. flags u8-ification + mean-shard flags exception + `pipeline.py:766` short-circuit fix—**implementation detail**.
4. Test contract synchronization(float32 fixtures, 1e-9 equivalence, 1e-5 tolerance separated by format)—**implementation detail, the bulk of the work**.
5. The 1e-6-level corner case at the clamp boundary −0.5 and trace 0.10:accept or parameterize compensation—**optional optimization**.
6. Dual reader/dual writer rollout in four phases per §20, GFS first—**implementation detail**.
7. Documentation fixes(in passing):README GEFS "0.5°" vs the implementation's pgrb2sp25 0.25°(`connector.py:186-188`); ARCHITECTURE.md:320 cache sizes; finalizer.py:20 "14-day" vs the actual 1-day retention; connector.py:71-72 "geavg out of scope" vs the actual ingestion of geavg.

---

## 24. Three Architectures Compared

| Dimension | A current state(f32+f32+Zstd) | **B recommended(f32 compute + f16/u8 storage + f16 cache)** | C f16 everywhere |
|---|---|---|---|
| Storage | — | **−22.2%**(measured) | ≈B(flags actually worse) |
| Network/GET | — | **−18~19% bytes**, request count unchanged | ≈B |
| API cache | — | payload −50%, reader total −10~15% | same as B |
| CPU | baseline | decode 2× faster, encode +small cost,**compute unchanged** | **arithmetic 8.7× slower + reduction overflow(measured)** |
| Precision | bit baseline | threshold flips 0(measured), max 0.027°C/0.0625mm | **destructive**(small increments 100% lost at 600mm, measured) |
| Complexity | — | medium(new format + dtype table + test contracts) | low(seemingly)but requires rewriting the entire numerical safety net |
| Compatibility/risk | — | manifest mechanism ready-made, zero DB migration, rollback possible | irreversible semantic damage |

**Reason for not considering a fourth architecture(such as bfloat16/zfp/bitshuffle)**:bfloat16's precision(8-bit mantissa)is worse than f16 and its ecosystem support is poor; zfp/lossy compression changes the "per-value interpretable" contract and is a larger change; bitshuffle/byte-plane reordering is an orthogonal compression optimization that can be evaluated independently on top of v2.

## 25. Decision Criteria (Checked Item by Item Against the Product Assumptions)

Sub-1°C temperature quantization ✅(0.027°C); RH/cloud ✅(0.03%); wind/visibility/cloud base ✅(0.014m/s / 7.8m / 0.0078km); precipitation storage error ✅(≤0.125mm and ≤1 upstream packing quantum); forecast uncertainty ≫ storage error ✅(GFS 2m temperature forecast error magnitude 1-3°C); product semantics unchanged ✅(all threshold flip rates 0, except the 1e-6 adversarial corner); deaccum/reset correct ✅(§6); ensemble materially equivalent ✅(§8); throughput/latency no obvious regression ✅(§12/§14:encode +small cost, decode faster, GET bytes down); backward compatible ✅(§16/§20).

## 26. Risks / Blocking Questions

**Real blockers**:① product acceptance of the unrounded JSON literal change(the only user-facing text difference);② silent inf from upstream extreme values >65504(writer assertion required).**Implementation details**:test contract migration(largest workload), flags metadata/byte inconsistency fix, v2 reader class boundary, `STORAGE_FORMAT_VERSION` per-cycle freeze.**Optional**:clamp-boundary compensation, byte-plane reordering of fields other than visibility/snow, mean-shard flags lowered to f16.

---

### Appendix: Evidence and Reproducibility
- Code evidence:all file:line references were cross-checked by 4 independent investigation agents(ingestion pipeline / storage and serving / variables and GEFS / repo-wide dtype), and key conclusions were directly re-verified by reading code by the lead investigator(`pipeline.py:302-420`, `cloud.py:103-235`, `zarr_writer.py:615-672`, `zarr.py:316-532` etc.).
- Data and benchmarks:85 real cfgrib decoded fields(GFS 20260923/18z f003/f006/f189/f192 + GEFS 20260923/00z all 30 members f120,0.5° pgrb2a proxying 0.25° pgrb2s); the benchmark scripts fetch/decode/bench1-5 are at `%TEMP%\f16bench\`, result JSON in `results\`; CPU=AMD Ryzen 7 5800X, numpy 2.5.3, numcodecs Zstd(level=5).
- Known deviations:the GEFS member benchmark uses 0.5° files(0.25° is slightly less smooth→ the compression-ratio estimate is conservative); cloud_ceiling was not sampled, so its compression estimate is by analogy with cloud; the active retention window is derived from the canonical retirement mechanism(GFS ~40 runs/GEFS ~20 runs), not a hard configuration.

---

## Addendum (post quantization benchmark round)

Follow-up benchmark ([`QUANTIZATION_BENCHMARK.md`](./QUANTIZATION_BENCHMARK.md)) refined two conclusions of this report:

1. **§5 GEFS trace-threshold flips**: the "zero flips" claim holds in float32 comparison semantics, but the production path compares after float64 promotion (`precipitation.py:184` `float(amount_curr)`). GEFS decimal packing produces a large mass of values exactly equal to `f32(0.1)` (12.813% of the f120 precipitation field); under plain float16 storage these flip wet→dry in phase products. `precipitation_amount_3h` therefore keeps **float32** storage in the recommended `sharded_v2` freeze (see QUANTIZATION_BENCHMARK.md §26).
2. **Compression**: plain float16 remains the recommended representation; variable-specific/fixed-point quantization variants were measured at ≤+3.6% relative on cycle-weighted totals and are rejected on complexity grounds.
