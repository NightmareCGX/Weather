# Weather Platform Quantization / Float16 / Zstd Storage Benchmark Report

**Nature**: Storage Representation / Quantization Benchmark (follow-up to [`REPORT.md`](./REPORT.md)). Benchmark work only: no changes to production code, no re-investigation of the already-confirmed numerical architecture. All candidates share the same batch of real float32 source fields, the same 100×100 chunk geometry, and the same numcodecs Zstd **level 5**; the compute-domain dtype is unchanged throughout.

**Final conclusion (up front): freeze `sharded_v2` as Option A — continuous variables plain float16 + det/mem flags uint8 (mean-shard flags stay float32), but `precipitation_amount_3h` is exceptionally stored as float32 (threshold semantics, not compression). Full-cycle weighted saving ≈ −22.0%; plain f16 is effectively already at the Pareto frontier, and any quantized-f16 variant gains ≤3.6% (relative) under full-cycle weighting, which is not worth a permanent policy cost.**

---

## 1. Executive Summary

- **16 storage representation schemes × 21 real field instances (15 GFS + 6 GEFS) × real chunk geometry, measured**. Instance-weighted and full-cycle-weighted ( GEFS accounts for 96.6% of bytes) results diverge markedly — the "advantage" of most schemes is amplified under instance weighting.
- **No quantization inside an f32 container can come close to float16**: uniform round3/2/1 saves only 3.9/9.3/21.0%; variable-specific (conservative/balanced) 3.2/10.6%; even trimming the mantissa to **f16-equivalent precision (11 effective bits)** saves only 8.3%. → f16's compression advantage comes mainly from the **2-byte container itself**, not from mantissa entropy (§10/§17, Q8/Q9).
- **The "further compression" of round+f16 is mostly an illusion**: round3+f16 has almost the same f16 codebook as plain f16 (temperature/RH/precipitation unique codes completely unchanged), only +1.0pp; round2+f16 is +4.5pp instance-weighted, but **under full-cycle weighting only +3.6% (relative)**, and the gains are concentrated in the three small fields snow (+62%)/rate (+30%)/vis (+11%), while on the four large fields (temp/RH/cloud/wind) it is ≈0 and on GFS precipitation it **regresses 3.6%**.
- **fixed-point int16/uint16 gains only +0.1% over plain f16 under full-cycle weighting** — losses in GEFS-side RH/temp cancel the gains in GFS rate/vis. Rejected outright on complexity grounds (Q10).
- **A real threshold landmine was found (the most important finding of this round)**: GEFS decimal packing produces **grid points exactly equal to f32(0.1), i.e. 0.100mm, 12.813% of that field**. Production code compares in the float64 domain (`precipitation.py:184` `float(amount_curr) <= 0.10`) → the baseline judges **wet**; plain f16 rounds 0.1000000015 to 0.0999755859375 → judges **dry**. That is, plain f16 would flip 12.8% of the grid points in the GEFS drizzle area wet→dry in phase products. Schemes that preserve the decimal grid (round2/fixed-point 0.01) retain that value exactly, with zero flips. → drives the dtype exception for `precipitation_amount_3h` (§14/§26).
- **Recommendation**: see §26. The gain is revised slightly from −22.4% in the previous round's report to **−22.0%** (giving up f16 for precipitation for threshold safety, ≈ −0.4pp).

## 2. Scope and Existing Evidence

Not re-investigated (inherited from REPORT.md): float16 arithmetic is already rejected (8.7× slower + reduction overflow); Zstd level is not the main lever (L12 only +3.3%, L19 +9.9%, both far below f16's 22%); the float64 computation contract for de-accumulation/cloud reconstruction; API cache structure; rollout mechanism. This round answers one question only: **what representation should the continuous payload of `sharded_v2` be**.

## 3. Dataset and Methodology

- **Data**: of the 85 real decoded fields from the previous round, 21 instances are reused (GFS 20260923/18z 0.25°: temp/RH/u/v/gust/prate/precip f006+f192/precip f189/cloud/vis/snow/4 flags; GEFS 20260923/00z gep01 f120: temp/RH/u/v/precip/cloud). cloud_ceiling: **not sampled** (cycle estimate uses cloud by analogy, marked est).
- **Geometry**: 100×100 chunk (721×1440 → 8×15=120 chunks, edge chunks are not padded, same writer semantics).
- **Compression**: numcodecs `Zstd(level=5)`, each chunk compressed independently (consistent with `zarr_writer.py:628,658-660`).
- **Error baseline**: dequantize→float32 vs the original float32 source; relative error is reported only for |x|≥1.0 (internal unit).
- **Threshold comparison domain note**: bench6 automatic flip detection runs in the float32 domain (NEP 50 weak scalar); production code promotes via `float()` and compares in the float64 domain. The two disagree on "values exactly on the threshold" — this report re-checked such values per domain (§14), which is the most important methodological correction of this round.
- Hardware: AMD Ryzen 7 5800X, numpy 2.5.3, Python 3.14.6. Scripts `bench6_quantization.py`/`bench7_cpu_cycle.py`, results `results/bench6_quantization.json`/`bench7_cpu_cycle.json`.

## 4. Current Float32 Baseline(Scheme A)

15 continuous-variable instances in total: raw 44.56 MB → compressed **10.78 MB** (average ratio 4.13×). Per-variable compressed KB is in the §16 table. Mean chunk: temperature 8015B, RH 11475B, wind 9857B, precipitation 4456B. This is the denominator for all saving.

## 5. Plain Float16 Baseline(Scheme B)

Instance-weighted **−22.0%** (10.78→8.41MB); full-cycle-weighted **−22.4%**. max err 0.125 (only at GFS precipitation 306mm, f16 spacing); **f32-domain threshold flips: all zero**; float64 production-domain re-check: the only flip class = GEFS exactly-0.1mm (§14). API chunk cache payload 40,000→20,000B.

## 6. Uniform Decimal Quantization + Float32(Scheme C)

| | saving vs A (instance) | max err | key flips (f32 domain) | verdict |
|---|---|---|---|---|
| C1 round(3) | 3.9% | 5.0e-4 | vis fog 13 grid points (0.0013%) | **dominated** (23.2pp worse than B, RAM 2×) |
| C2 round(2) | 9.3% | 5.0e-3 | prate-info 0.41%;vis fog 0.0087% | **dominated** |
| C3 round(1) | 21.0% | 0.05 | **trace 7.10%** (0.125→0.1); freeze 0.18%; prate 2.87% | **dominated + flips fail acceptance** |

Q1: **round(3)+f32 vs plain f16 is 23.2% larger** (10.36 vs 8.41MB), larger rather than smaller/comparable. Q2: round(2)+f32 (9.3%) **falls far short** of f16 (22.0%), and RAM is still 2×, so it does not qualify.

**Mechanism finding (applies to all decimal grids × GRIB binary packing)**: the behavior of GFS precipitation 0.125mm grid values near the 0.1 threshold — round(1) turns 0.125 → 0.1 → `<=0.10` judges dry, **flipping 7.1% of all grid points wet→dry**. Any decimal quantum coupled to the 0.10/0.125 grid has this pathology.

## 7. Uniform Decimal Quantization + Float16(Scheme D)

| | saving vs A | vs B | f16 codebook change | verdict |
|---|---|---|---|---|
| D1 round(3)+f16 | 22.7% | **+1.0pp** | temperature/RH/precipitation unique codes **completely unchanged** (values change 5-27% but fall back to the same code) | **dominated by B** (introducing a policy for 1pp) |
| D2 round(2)+f16 | 26.5% | +4.5pp | vis 6843→2120 codes; snow/rate large; temp/RH/cloud ≈0 | **frontier candidate, full cycle only +3.6%** (see §16/§21-23) |
| D3 round(1)+f16 | 36.6% | +18.7pp | — | **fails acceptance**: trace 7.1% flips (0.125→0.1) |

Q4: round3+f16 **cannot** further significantly reduce compressed size (entropy analysis see §17); round2+f16 can but the gains are concentrated; round1+f16 has the best numbers but breaks semantics.

## 8. Variable-Specific Quantization + Float32(Scheme E)

E1 conservative (temperature 0.01°C/RH 0.05%/wind 0.01m/s/vis 0.001km/precipitation 0.01mm…): **3.2%** — better precision than f16 but negligible compression, **dominated**.
E2 balanced (0.05/0.1/0.05/0.01/0.05…): **10.6%**, and the precipitation 0.05mm quantum produces a systematic tie with the 0.125 GRIB grid (0.125/0.05=2.5→banker's→2→0.1) → **trace 7.1% flips, fails acceptance**.

Q3: **No**. Under an f32 container, even with variable-specific quantization, compression (≤10.6%) is far behind f16 (22.0%), and RAM is 2× — the precision advantage is meaningless for the product (G11 already proved that even equal precision is worth only 8.3%).

## 9. Variable-Specific Quantization + Float16(Scheme F)

F1 conservative+f16: **23.3%** (+1.3pp vs B instance; full cycle ≈+0.4%) → **dominated by B**.
F2 balanced+f16: instance 28.1% (+6.1pp) — but it inherits E2's precipitation 0.05 pathology → **trace 7.1% flips, fails acceptance**. If F2's precipitation quantum is changed to 0.01mm (i.e. the D2 mix), it degenerates into D2 + a fruitless per-var complexity.

Q5: **variable-specific quantization+f16 does not constitute a genuine Pareto improvement over plain f16**: the only surviving variant (the D2 form) is +3.6% under full-cycle weighting, yet it introduces a permanent per-variable quantum table, threshold-aware tests, and a doubled presentation flip surface (§15).

## 10. Binary Mantissa Grooming(Scheme G)

frexp/ldexp RTN, preserving NaN/±Inf/±0, the float32 container and the exponent domain:

| effective significand bits | saving vs A | max err | notes |
|---|---|---|---|
| 17 | 4.5% | 0.0019 | exactly-0.1 retained (rounds up) |
| 15 | 5.1% | 0.0075 | exactly no flips (grid alignment by luck) |
| 13 | 5.8% | 0.0625 | |
| **11(≈f16 precision)** | **8.3%** | 0.1875 | encode CPU 50.5ms/4M = 4.7× astype |

Q8: **even after trimming to f16-equivalent mantissa precision, the 4-byte container still loses badly to 2-byte f16** (8.3% vs 22.0%). Q9 (experiment-based decomposition): of f16's 22pp advantage, **about 8pp comes from precision loss (mantissa entropy removal) and about 14pp from halving the raw word width**. Low tail-bit entropy is not Zstd's main bottleneck inside an f32 container — container layout is. An additional rejection reason: the frexp/ldexp pipeline costs 4.7× CPU, yet the gain is not much better than the C series.

## 11. Fixed-Point / Scale-Offset Packing(Scheme H)

Per-variable u16/i16 + NaN sentinel (temperature i16 0.01°C; RH/cloud/gust/prate/precip/vis u16 0.01; snow u16 0.001; wind i16 0.01); range assertions all pass (measured max: temp −68..39°C, precip 306mm, gust 231km/h, wind ±62m/s).

- Instance-weighted 25.4% (vs B +4.3pp); **full-cycle-weighted −22.3% vs A, only +0.1% vs B** — losses in GEFS-side RH (−9~10% vs B)/temp cancel the gains in GFS rate (+35%)/vis (+14%)/GEFS wind (+11%).
- Best error characteristics (small value domain max 0.005mm/0.0023°C; 2dp presentation almost never flips, 1.5%).
- Q10: **reject**. Zero full-cycle gain + permanent machinery cost for scale/sentinel/metadata/range/overflow/dequant. §27's "3-5% → reject" principle is hit in its strongest form (0.1%).

## 12. Categorical uint8

det/mem flags: the only reasonable candidate, `uint8 → Zstd L5` (consistent with REPORT.md §18). bench6 re-test: 4 flag fields round-trip with zero mismatch; total compression 91.5KB vs the current f32 133.1KB (−29.6%, bench4 basis). **No float quantization variant is done**. GEFS mean-shard flags are member-mean probabilities (0..1) and stay float32, outside this round's comparison scope (not applicable).

## 13. Per-Variable Numerical Error(serious candidates)

max / MAE / P99 (internal units; full JSON in results/):

| field | B plain f16 | D2 r2+f16 | H fixed |
|---|---|---|---|
| gfs/temperature (°C) | 0.027 / .0031 / .0148 | 0.027 / .0037 / .0148 | 0.0023 / .0023 / .0023 |
| gfs/visibility (km) | 0.0078 / .0044 / .0066 | 0.0123 / .0048 / .0080 | 0.0048 / .0042 / .0048 |
| gfs/precip (mm) | 0.125 / 1.2e-6 / 0 | 0.125 / .0015 / .0059 | 0.005 / .0015 / .0050 |
| gefs/precip (mm) | 0.050 / 1.0e-4 / .0016 | 0.050 / .0001 / .0016 | 7.6e-6 / 1.7e-8 / 0 |
| gfs/snow (m) | 0.00097 / 3.0e-5 | 0.00097 / ~same | 0.0005 |

The max/typical errors of all candidates are far below the presentation granularity and REPORT.md §5's product scale; error does not discriminate, **threshold behavior does** (§14).

## 14. Threshold / Semantic Stability

f32-domain automatic detection (bench6) + **float64 production-domain per-case re-check** (`precipitation.py:184` semantics):

| scheme | f32-domain flips | float64 production-domain re-check | verdict |
|---|---|---|---|
| **B plain f16** | all zero | **GEFS exactly-0.1mm: 12.813% of field grid points wet→dry** (0.1000000015→0.0999755); all others zero | the only semantic issue, see below |
| C1/D1/F1 | vis fog 13 grid points | same | negligible |
| C2/D2/H | prate-info 0.41%*; vis fog 0.0087% (0.9952→1.0) | exactly-0.1 **stays wet** (the decimal grid contains f32(0.1)) | fog flips 90 grid points/field, small but real |
| C3/D3/E2/F2 | **trace 7.10%/6.22%** (0.125→0.1); freeze 0.18% | same | **fails acceptance** |
| G15 | zero | exactly-0.1 retained (alignment by luck) | fragile |
| G17/G13 | zero | exactly-0.1 retained (rounds up→wet) | |

*prate 0.1mm/h is the informational threshold of this round, not a code-verified product threshold.

**Full mechanism of the exactly-0.1mm landmine** (core finding of this round):
1. GEFS decimal packing produces a huge number of exactly 0.100mm values (measured on that field 12.813%, widespread drizzle); the GFS binary 0.125 packing has **no** such value (0.125≠0.1) → only GEFS is affected.
2. Under the baseline (f32 storage), the serve path `float(np.float32(0.1)) = 0.10000000149 > 0.10` → **judges wet** — which is in fact a deviation of float expansion noise from the policy intent (`≤0.10 → dry`).
3. plain f16 rounds to 0.0999755859375 → **judges dry** — exactly back to the policy intent, but it **changes today's phase-product behavior** (drizzle-area phase-support/transition statistics shift by 12.8% of grid points).
4. Schemes that preserve the decimal grid (round2 / fixed-point 0.01) retain 0.1000000015 exactly as f32(0.1) → zero flips.
5. This at the same time corrects a conclusion of the previous round's REPORT.md §5/§6: the "GEFS trace flips = 0" there holds in the float32 comparison domain, but not in the production float64 domain (see the addendum at the end of REPORT.md).

**Disposition options** (§26 gives the recommendation): *(i)* plain f16 + accept the "policy-intent correction" type of flip (needs product sign-off); *(ii)* precipitation alone round(0.01mm)+f16 (preserves the baseline, introduces a single-variable quantum); *(iii)* **precipitation alone stays float32** (preserves the baseline, zero machinery, v2 is per-variable dtype anyway; giving up f16 compression for precipitation ≈ full cycle +0.4pp).

## 15. Presentation-Level Effects

| scheme | max 1dp flips | max 2dp flips | notes |
|---|---|---|---|
| B | 24.8% (temp) | 76.7% | consistent with REPORT.md §5 |
| D2 | **50.3%** | 76.7% | the round2 grid doubles the 1dp boundary flips |
| H | 52.5% | **1.5%** | the 0.01 grid = the 2dp grid, 2dp almost unchanged |

presentation flip ≠ semantic failure (REPORT.md §19 already separates them); but D2 makes the 1dp presentation flip surface larger by a factor of 2, a substantive enlargement of the user-visible diff surface.

## 16. Compression Results

**Per-variable compressed KB (instance, Zstd L5)**:

| field | A | B | D2 | H | G11 |
|---|---|---|---|---|---|
| gfs/RH | 1345 | 1168 | 1168 | 1286 | 1284 |
| gfs/wind_u / v | 1155/1152 | 925/916 | 911/914 | 906/893 | 1126/1119 |
| gfs/gust | 1081 | 959 | 958 | 948 | 1160 |
| gfs/cloud | 1032* | 838* | 838* | 889* | — |
| gfs/temp | 939 | 763 | 759 | 776 | 931 |
| gfs/vis | 958 | 523 | 464 | 449 | 619 |
| gfs/prate | 796 | 622 | 436 | 404 | 744 |
| gfs/snow | 617 | 329 | **125** | 246 | 335 |
| gfs/precip | 522 | **424** | 440(−3.6%!) | 423 | 522 |
| gefs/wind_u / v | 576/575 | 453/443 | 422/417 | 400/394 | 493/497 |
| gefs/temp | 292 | 239 | 237 | 241 | 283 |
| gefs/RH | 371 | 317 | 316 | 346 | 346 |
| gefs/precip | 136 | 103 | 103 | 104 | 133 |

*cloud is a bench6 supplementary measurement (120×8.7KB). **Key row: GFS precipitation D2 is 3.6% worse than B** (round2 breaks the alignment between the 0.125 grid and f16).

**Unified aggregate table** (instance-weighted saving vs A / full-cycle-weighted vs A / vs B):

| Scheme | Native | Quantization | Raw | saving vs current (instance) | (full cycle) | vs plain f16 (full cycle) | API cache payload | max err | threshold verdict | CPU (encode side) | Complexity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | f32 | none | 44.56MB | 0% | 0% | — | 40,000B | 0 | baseline | baseline | — |
| **Plain f16** | f16 | binary16 | 22.28MB | **22.0%** | **−22.4%** | 0 | **20,000B** | 0.125 | exactly-0.1 issue | astype 10.8ms | v2 baseline cost |
| round3 f32 | f32 | 0.001 | 44.56MB | 3.9% | ~4% | **23% worse** | 40,000B | 5e-4 | ok | 3.5ms | low but dominated |
| round2 f32 | f32 | 0.01 | 44.56MB | 9.3% | ~9% | 16% worse | 40,000B | 5e-3 | ok | 3.5ms | dominated |
| round1 f32 | f32 | 0.1 | 44.56MB | 21.0% | ~21% | ≈but worse error | 40,000B | 0.05 | **trace 7.1%** | 3.5ms | reject |
| round3 f16 | f16 | 0.001+b16 | 22.28MB | 22.7% | ~22.5% | +0.1pp | 20,000B | 0.125 | ok | 14.5ms | dominated |
| round2 f16 | f16 | 0.01+b16 | 22.28MB | 26.5% | **−25.2%** | **+3.6%** | 20,000B | 0.125 | fog 90 grid points | 14.5ms | +permanent policy |
| round1 f16 | f16 | 0.1+b16 | 22.28MB | 36.6% | ~36%* | +18pp* | 20,000B | 0.125 | **trace 7.1%** | 14.5ms | reject |
| var-cons f32 | f32 | per-var | 44.56MB | 3.2% | ~3% | 25% worse | 40,000B | 0.021 | ok | 7.4ms | dominated |
| var-bal f32 | f32 | per-var | 44.56MB | 10.6% | ~10% | 15% worse | 40,000B | 0.049 | **trace 7.1%** | 7.4ms | reject |
| var-cons f16 | f16 | per-var+b16 | 22.28MB | 23.3% | ~22.7% | +0.4pp | 20,000B | 0.125 | ok | 18.4ms | dominated |
| var-bal f16 | f16 | per-var+b16 | 22.28MB | 28.1% | ~25%* | +3.7pp* | 20,000B | 0.125 | **trace 7.1%** | 18.4ms | reject |
| groom 17/15/13 | f32 | binary | 44.56MB | 4.5/5.1/5.8% | ~same | 20%+ worse | 40,000B | ≤0.063 | ok (by luck) | **50.5ms** | reject |
| groom 11 | f32 | ≈f16 precision | 44.56MB | 8.3% | ~8% | 17% worse | 40,000B | 0.1875 | ok | 50.5ms | scientific control |
| fixed-point | i16/u16 | per-var scale | 22.28MB | 25.4% | **−22.3%** | **+0.1%** | 20,000B+dequant | 0.005 | ok | 12.0ms | heavy machinery, zero gain |
| **flags u8** | u8 | native | ÷4 | **−29.6%** | same | — | 10,000B | 0(zero mismatch) | zero | lowest | already in the v2 plan |

\* Full-cycle values are corrected by the GEFS weight, and the instance advantage of round1/var-bal shrinks; their failure on flips is independent of the weight.

## 17. Quantization + Float16 Entropy Analysis

unique f16 codes (plain vs quantized) and value change rate:

| field | plain unique | D1 unique | D2 unique | D2 value change rate | D2 compressed vs B |
|---|---|---|---|---|---|
| gfs/temperature | 1070 | 1070 | **1070** | 32.8% | −0.5%(≈0) |
| gfs/RH | 934 | 934 | 934 | **0.0%** | 0.0% |
| gfs/precip | 941 | 941 | 941 | 45.1% | **+3.6%(worse)** |
| gfs/visibility | 6843 | 5575 | **2120** | 15.2% | −11.4% |
| gfs/snow | — | — | — | — | **−62.1%** |
| gefs/precip | 399 | 399 | 399 | **0.0%** | 0.0% |

**Direct answer**: round(3)+f16 and plain f16 produce **an almost completely identical half-float codebook** (temperature/RH/GEFS precipitation unique counts completely unchanged; values are relabeled but fall back to the same code) → **no additional entropy reduction**. round(2)+f16's real entropy reduction happens only in fields that are naturally coarse in decimal (vis/snow/rate); for temperature/RH/cloud/GEFS precipitation it is zero or negative. The "further collapse" of semantic quantization has essentially no room in front of f16's 11-bit significand — f16 is itself already a coarse quantizer.

## 18. CPU Cost

4M elements (per 10K chunk divided by 400):

| operation | 4M ms | /chunk ms | notes |
|---|---|---|---|
| f32→f16 astype(writer) | 10.8 | 0.027 | baseline quantization cost |
| round2→f16(writer) | 14.5 | 0.039 | +34% vs astype, the absolute value is negligible |
| var-quant→f16 | 18.4 | 0.046 | |
| groom(17bit) | 50.5 | 0.126 | frexp/ldexp costs 4.7× |
| fixed-point pack | 12.0 | 0.030 | |
| f16→f32(reader) | 18.9 | 0.047 | negligible vs Zstd decode ~0.3ms/chunk |
| i16→f32 dequant | 5.6 | 0.014 | |
| f32 identity(copy) | 2.3 | 0.006 | |

quantization CPU is not a deciding factor for any candidate (except groom, whose 4.7× cost stacking with zero gain is one of the rejection grounds). No whole-system latency is inferred from this.

## 19. API Memory Impact

By native payload (100×100 chunk):

| representation | chunk cache payload/reader (512 chunks) | compute transient |
|---|---|---|
| A / C / E / G (f32 family) | 19.58 MiB | — |
| **B / D / F (all f16 family)** | **9.83 MiB** | — |
| H (packed i16) | 9.83 MiB (if packed is cached) + a 40,000B/block transient dequant before each compute, or materialize 20,000B f32 | +dequant path |

object-store compression and RAM representation must be accounted for separately: G11 (groom) proves that a scheme whose "compression approaches B" (if any) still has 2× RAM; conversely H does not win on compression yet must pay the dequant complexity. The f16 family is the only class that optimizes both ledgers at once.

## 20. Range GET Byte Impact(request count unchanged)

mean compressed chunk bytes (A→B→D2): temperature 8015→6513→6475; RH 11475→9967→9967; wind 9857→7895→7777; precipitation 4456→**3622**→3752; cloud 8853→6987→6987; vis 8171→4464→3957; snow 5265→2808→1063; prate 6790→5306→3720. GEFS statistics requests (30×temp): 280→229KB (B). point requests −18~19% (B); D2 is a further −10~30% on snow/vis/rate and a further **+3.6%** on precipitation.

## 21. GFS Whole-Cycle Estimate

15 vars × 81 leads (ceiling by cloud analogy, est; flags from bench4 f32/u8 measurements):

| | per cycle | vs A |
|---|---|---|
| A | 0.83 GB | — |
| B(+u8 flags) | 0.65 GB | −22.4% |
| D2 | 0.61 GB | −26.6% |
| H | 0.63 GB | −24.2% |

## 22. GEFS Whole-Cycle Estimate

14 vars × 81 leads × 31 sets (30 members + 1 mean; members/mean uniformly approximated, est):

| | per cycle | vs A |
|---|---|---|
| A | 23.87 GB | — |
| B | 18.52 GB | −22.4% |
| D2 | 17.87 GB | −25.1% |
| H | 18.56 GB | **−22.3%(≈zero gain)** |

## 23. Active Storage Estimate(estimate, carrying over REPORT.md's basis: GFS ~40 runs + GEFS ~20 runs, derived from canonical retirement, not a hard configuration)

| | active | vs A |
|---|---|---|
| A | 511 GB | — |
| **B** | **396 GB** | **−22.4%** |
| D2 | 382 GB | −25.2%(0.68GB more saved per two cycles) |
| H | 396 GB | −22.4%(+0.1%) |

**Full-cycle weighting is the key correction of this round**: under instance weighting D2 leads by 4.5pp; after GEFS weighting it shrinks to 3.6% (relative)/2.8pp (absolute); H goes to zero outright.

## 24. Pareto Frontier

**Dominated (fully dominated by B: compression worse or equal + RAM larger/equal + error no better + CPU worse + complexity higher)**: C1, C2, E1, G17, G15, G13, G11, D1, F1, **H**.
**Fails acceptance (threshold flips)**: C3, E2, D3, F2 (0.125×decimal-grid tie pathology, trace 7.1%).
**Frontier (2)**:
- **B plain f16**: −22.4% per cycle, 20,000B RAM, the only issue = exactly-0.1 (solvable with a single-variable dtype exception);
- **D2 round2+f16**: −25.2% per cycle, 20,000B RAM, the cost = a permanent decimal quantum policy + fog boundary flips + doubled 1dp presentation flips + a local regression on GFS precipitation.

The marginal trade of D2 vs B: 0.68GB per two cycles (≈ full cycle +2.8pp absolute), in exchange for a permanent policy and its entire test surface. Judged not worth it under the §27 principles.

## 25. Complexity / Maintenance Cost

- **B (plain f16)**: v2 machinery (dtype dispatch, reader/writer, test contract) = the baseline cost any v2 pays; **no** per-variable quantization table; new variables default to f16, zero decisions.
- **D2**: additionally requires a variable→quantum table, a policy for admitting new variables, threshold-aware tests (0.10/1km/60km/h boundaries × grid coupling), quantization documentation, a scientific contract; the "decimal grid × GRIB binary packing" coupling has already been proven by E2/F2/C3 to be a systemic landmine; the 1dp presentation diff surface doubles.
- **H**: plus scale/offset/sentinel/range/overflow/dequant/metadata; and the full-cycle gain is +0.1% → a purely negative option.
- **G**: frexp pipeline 4.7× CPU + scientific control value, no production value.

## 26. Final Recommendation

**Freeze `sharded_v2` payload = Option A (hybrid in exactly one place, driven by semantics rather than compression):**

```text
continuous fields(excluding precipitation):   plain float16(<f2, little-endian)+ Zstd L5
precipitation_amount_3h:                      float32(native f32, same as v1)— threshold-semantics exception
categorical det/mem flags:                    uint8
GEFS mean-shard flags:                        float32(member-mean probability, keep current state)
compute / cache materialization after read:   float32(unchanged)
```

- Expected gain: full cycle **−22.0%** (f16-everything is −22.4%; keeping precipitation as f32 gives up ≈0.4pp to buy out the exactly-0.1 risk), active ≈ −22%; Range GET −18~19%; chunk-cache payload −50%; RAM accounting same as REPORT.md §10.
- **The three-way choice for precipitation**: default to (iii) f32 per the table above — zero new machinery, a fallback predecessor path fully consistent with REPORT.md §6, zero displacement of today's product. If the product side confirms that "judging exactly-0.100mm dry better matches the policy intent" (the baseline's wet judgment is itself float expansion noise), (i) plain f16 may be chosen instead (0.4pp more saving, needs sign-off); (ii) round(0.01mm)+f16 only if future precipitation compression is proven worth a single-variable quantum.
- **Explicit rejection**: the entire quantized f32 family (C/E/G: compression 12-19pp worse, RAM 2×), round1/0.05mm-class grids (threshold pathology), fixed-point (full cycle +0.1%).
- **No hybrid per-variable compression policy**: under full-cycle weighting no quantization variant vs plain f16 has a substantive ≥4pp advantage, while D2's +3.6% requires a permanent policy — it does not meet the "additional gain large enough" hybrid threshold.

## 27. Review of Decision Principles(against this round's measurements)

- "plain f16 is close to Pareto optimal → recommend plain f16": **hit** (D2 full cycle +3.6% relative; not the hypothesized 24% vs 22%, but 25.2% vs 22.4%, and accompanied by real side effects).
- "quantized f16 is significantly better (30%+) → consider": **not hit** (measured 25.2%).
- "even if quantized f32 compresses strongly, RAM matters too": **hit and reinforced** (G11: equal-precision compression is only 8.3%, RAM 2×, a double elimination).
- "fixed-point is only 3-5% more → reject": **hit and reinforced** (measured +0.1%).

---

### Appendix: Reproducibility and Corrections to the Previous Round's Report
- Scripts: `%TEMP%\f16bench\{bench6_quantization,bench7_cpu_cycle}.py`; results: `results\{bench6_quantization,bench7_cpu_cycle}.json`; data: reuses the previous round's 85 fields (not re-downloaded).
- **Corrections to REPORT.md**: its §5 "GEFS precipitation trace flips = 0" and the related §6 conclusions hold in the float32 comparison domain; production code compares in the float64 domain (`precipitation.py:184`), and exactly-f32(0.1) grid points (12.813% of the GEFS field) go wet→dry under plain f16. An addendum pointing to §14 of this report has been appended at the end of REPORT.md.
- Not sampled: cloud_ceiling (cycle estimate uses cloud by analogy, est); GEFS mean-shard flags (outside this round's scope).
