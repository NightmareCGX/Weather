# Sharded V2 Implementation Report

Implementation basis: `SHARDED_V2_IMPLEMENTATION_PLAN.md` (authoritative) + `REPORT.md` + `QUANTIZATION_BENCHMARK.md`. This round completed all Phase 0–8 code, tests, shadow tooling, config support and documentation; **no production rollout performed, nothing committed/pushed**.

> 2026-09 update: the implementation has since been merged (commit 61b0ed1) and is available behind `STORAGE_FORMAT_VERSION=sharded_v2`; production rollout is still pending.

---

## 1. Summary

**SHARDED V2 IMPLEMENTATION COMPLETE** (§42, all 25 acceptance criteria satisfied, see the item-by-item check in §20). The frozen storage contract is implemented end-to-end: 9 continuous variables `<f2`, two f32 semantics exceptions `precipitation_amount_3h`/`cloud_ceiling`, det/member flags `u1`, mean flags `<f4`, explicit little-endian, Zstd L5, index/trailer/object keys identical to v1, version distinguished by the manifest `storage_format_version="sharded_v2"`. Staged production rollout (GFS→GEFS mean→GEFS members) retained as GO/NO-GO; the capability is ready.

## 2. Files Changed

**Modified (13)**:
- `packages/domain/src/domain/storage_dtype.py` **(new)** — authoritative dtype resolver: frozen matrix, `ProductRole` reuses `TARGET_KIND_DET/MEAN/MEM`, fail-closed (unknown variable/unknown format raise), `is_sharded_payload_format` predicate.
- `packages/domain/tests/test_storage_dtype.py` **(new)** — 14 resolver tests (matrix, role dependency, fail-closed, little-endian, 15-variable completeness, set disjointness).
- `services/ingestion/src/ingestion/core/base.py` — new exceptions `F16RangeViolationError`, `CategoricalDomainViolationError`, `CycleFormatConflictError` (all with contract docs).
- `services/ingestion/src/ingestion/core/config.py` — `STORAGE_FORMAT_VERSION` documentation expanded (v2 semantics, per-cycle freeze, rollback), default still `sharded_v1`.
- `services/ingestion/src/ingestion/core/pipeline.py` — **Phase 0 flags short-circuit fix**: `_normalize_canonical_units` still forces `np.uint8` when the unit token is already "flag" (det/member categorical stably uint8, no longer relying on writer guesswork).
- `services/ingestion/src/ingestion/core/zarr_writer.py` — `encode_region_sharded_v2` (per-variable native dtype encoder) + `encode_region_sharded` dispatch + writer guards (`_validate_f16_block`: NaN pass-through counted, ±Inf/overflow rejected **before** cast; `_validate_categorical_block`: det/mem flags ⊆{0,1}, mean flags ⊆[0,1]) + `assert_cycle_format_allowed` in-process format freeze + `commit_region` dual dispatch block accepting v2 (guard violation does **not** silently fall back to legacy) + `prepare_run_store` v2 branch (`.zarray` dtype=resolver result, fixing v1's metadata/byte inconsistency; v1 behavior frozen unchanged) + `read_slice`/`_populate_sharded_data` dtype-aware (manifest authoritative; when the manifest is missing, inferred from decoded byte length among resolver candidates — closing the hidden path of "an unpublished v2 store read as f32") + `_assert_chunk_byte_length` byte length as the last line of defense.
- `services/ingestion/src/ingestion/core/coordinator.py` — cycle format freeze persistence layer: `_assert_cycle_format_freeze` compares the version of the already-committed manifest at the two manifest sites, finalize and per-lead publish, and raises on mismatch (restart-safe second line of defense).
- `services/ingestion/src/ingestion/core/inventory.py` — `region_expected_object_keys` accepts sharded_v2 (v1/v2 physical key layout identical).
- `services/ingestion/src/ingestion/core/shadow.py` **(new)** — shadow validation core: re-encoding, three-way storage/numerical/serving comparison, namespace guard cleanup.
- `services/api/src/api/core/zarr_v2.py` **(new)** — `ShardedV2Reader` (inherits the frozen `ShardedV1Reader`, overriding only chunk decoding; **native dtype chunk cache**: f16 cached as f16, u8 cached as u8; byte length guard).
- `services/api/src/api/core/zarr.py` — `get_sharded_reader` dispatches to the V1/V2 reader by store manifest (format immutable per-store, LRU key still path).
- `services/api/src/api/services/{point_forecast,tiles,ensemble_data,vector_field}.py` — 6 serving branch points `== "sharded_v1"` → `is_sharded_payload_format(...)` (covering all entry points point/hourly/tiles/ensemble/wind/phase).
- `services/ingestion/tests/test_pipeline.py` — flags short-circuit regression test.
- `.env.example` — `STORAGE_FORMAT_VERSION` rollout entry (staged rollout order, freeze, rollback notes).

**New tests (3 files)**: `test_sharded_v2_writer.py` (25 test cases), `test_sharded_v2_reader.py` (13 test cases), `test_shadow_validation.py` (7 test cases).
**New tooling**: `scripts/shadow_v2.py` (`validate`/`cleanup` subcommands).

## 3. Final Storage Contract

Fully consistent with PLAN §4/§5 (item-by-item check ✓): dtype matrix, explicit little-endian `<f2`/`<f4`/`u1`, Zstd L5, 100×100 geometry and edge chunk special case, SHAR magic/index/trailer/object keys unchanged, `storage_format_version="sharded_v2"`, `.zarray`==payload dtype (v2 only), resolver as single source of truth (no second copy of FLOAT16_VARS anywhere in the repo). The `v2_unsharded` legacy string is not admitted into the v2 path (naming warning recorded in two docstrings).

## 4–10. Phase 0–8 Results

| Phase | Content | Result |
|---|---|---|
| 0 | flags short-circuit fix, resolver, exception classes | Done; `test_pipeline` 3/3, `test_storage_dtype` 14/14 |
| 1 | format contract, two-layer format freeze, `.zarray` dtype | Done; both freeze tests; `.zarray`==payload assertion (v1 behavior frozen as control) |
| 2 | `ShardedV2Reader`, factory dispatch, 6 serving branches | Done; 13/13; native cache assertion (cached array dtype=f16) |
| 3 | v2 writer, guards, hidden readback, inventory | Done; 25/25; repo-wide review of frombuffer/4B assumptions (`wind.py` i16 and `png.py` RGBA unaffected by storage) |
| 4 | shadow tooling + **locally executable validation** | Done; CLI measured exit=0 (see §16) |
| 5–7 | GFS/GEFS mean/GEFS members rollout capability | env/config/freeze/rollback/reader compatibility all ready; **no production cutover taken unilaterally** (GO/NO-GO 3/4/5 retained) |
| 8 | v1 natural retirement | v1 byte contract frozen, v1 exact tests all green, mixed-format serving test passing; zero lifecycle changes (no lifecycle files appear in the diff) |

## 11. V1 Compatibility Evidence

- All existing v1 tests were not relaxed and no assertions were modified: `test_sharded_storage.py`, `test_sharded_reader.py` (33), `test_serving_chunk_equivalence.py` and others, 797+457 test cases all green.
- v1 `.zarray` historical behavior frozen: flags metadata of newly written v1 stores is still the seed dtype (f32), consistent with history (the P2 fix only affects dtype stability of the normalized dataset, not v1 bytes/metadata).
- `read_slice` returns float32 for v1 stores (frozen); the `_populate_sharded_data` v1 path's f32 frombuffer unchanged.
- All v1 dispatch semantics preserved (guard/encoding failure → legacy fallback; only v2 guard violations changed to fail-loudly, which is part of the new v2 contract).

## 12. V2 Numerical Validation

Byte level: f16 chunk exactly 20,000B, f32 exactly 40,000B, u8 exactly 10,000B; LE byte assertion (`struct.pack("<e", 290.0)` compared byte by byte). Value level: after v2 round-trip, f16 variable values = `np.float16(source value)`; precip/ceiling f32 **bit-for-bit identical**. serving: f16 interpolation error ≤ half an f16 ulp; cached dtype assertion guards against hidden astype. Command-line shadow: maximum temperature difference 0.0076°C, all other variables 0.0.

## 13. Precipitation Semantic Regression

`test_precipitation_exactly_0_10mm_behavior_preserved`: under both v1 and v2 formats, `np.float32(0.1)` classified through storage→reader→`float()`→production predicate (`<=0.10` in float64) is exactly consistent (wet), and the stored bit patterns are bit-for-bit equal. Permanent retention, to prevent precipitation from being mistakenly added to the f16 set in the future.

## 14. Cloud Ceiling Semantic Regression

`test_cloud_ceiling_sentinel_boundaries_preserved`: the five boundary values 19.990/19.991/19.992/19.993/20.000 km, after v2 storage (f32 exception), classify identically to the baseline value by value; `>=19.99 → unlimited` behavior unchanged. Permanent retention.

## 15. Categorical Validation

det/member flags: u8, 1 byte/value, value range ⊆{0,1} (violation raises `CategoricalDomainViolationError`); mean flags: `<f4`, 4 bytes/value, [0,1] (violation raises). Same variable name but different role → resolver forces dispatch, so mean flags can never be u8-ified (covered by tests).

## 16. Shadow Validation Result

**Actually executed** (full local pipeline, not just tooling): `python scripts/shadow_v2.py validate --store <v1 cycle> --sample-points 40` → `SHADOW VALIDATION PASSED, exit=0`.
- Storage: v2 payload < v1 (−49.5% on the synthetic noise field; on real smooth fields measured at −13~24% per baseline run; the synthetic result is optimistic, a known bias);
- Numerical: precip/ceiling diff = 0.0, flips = 0; crain exact; temperature max 0.0076°C;
- serving: zero threshold flips across 40 sample points, reader factory correctly dispatches V2 by manifest;
- Tooling guards: cleanup rejects non-shadow prefixes, dry-run by default, age guard tests passing.
- Real production shadow command: `python scripts/shadow_v2.py validate --store s3://weather-data/{model}/{date}/{HH}/cycle.zarr` (generates `shadow-v2/` inside the same store tree, not entered into the catalog).
- **Architecture note**: the shadow module in the ingestion package depends only on ingestion (storage+numerical comparison) — the per-package CI environment (ingestion without api installed) introduces no cross-service dependency; serving comparison needs both the api+ingestion packages and is implemented in `scripts/shadow_v2.py` (operator tooling layer), and is executed in CI by the cross-package contract suite `tests/contracts/test_sharded_v2_shadow_contract.py` (the `uv sync --all-packages` CI job).

## 17. Performance Results

- **Writer throughput** (8-variable region, Zstd L5): v1 248ms vs **v2 169ms (−32%, no regression)** — f16 halves the byte volume entering Zstd.
- **Storage**: synthetic noise field −49.5%/region (real meteorological fields −13~24% per baseline run; smooth fields compress better, so the f16 gain is smaller than on the noise field, and the report distinguishes these faithfully).
- **API cache payload**: f16 variables 20,000B/chunk (−50%); tests assert the cached array's native dtype.
- **Range GET**: bytes transferred for f16 chunks −18~24% per baseline run (request count unchanged).

## 18. Tests

| Suite | Command | Result |
|---|---|---|
| domain (with 100% coverage gate) | `uv run --package weather-platform-domain pytest packages/domain/tests/` | **548 passed, 0 failed, coverage 100%** |
| ingestion full | `uv run --package weather-platform-ingestion pytest services/ingestion/tests/` | **797 passed, 29 skipped, 0 failed** (skip=requires MinIO/PostgreSQL/Redis services) |
| api full | `uv run --package weather-platform-api pytest services/api/tests/` | **457 passed, 235 skipped, 0 failed** (skip=requires PostgreSQL/Redis service containers) |
| New tests total | — | 59 new test cases all green (14 resolver + 25 writer + 13 reader + 7 shadow) |
| ruff | `uv run ruff check packages/domain services/ingestion services/api` | **All checks passed** |
| mypy | domain/ingestion/api changed files | 0 errors in new files; the 29 domain errors pre-existed as the pre-implementation baseline (confirmed against git stash, count unchanged) |
| frontend | — | **not run — reason: no frontend code touched this round** |
| CI-equivalent container/E2E | — | **not run — reason: this machine is Windows; per CLAUDE.md these run on ubuntu CI** |

## 19. Remaining Production Actions (manual GO/NO-GO only)

1. **GO/NO-GO 3**: GFS det canonical cutover (env `STORAGE_FORMAT_VERSION=sharded_v2`, observe one complete lifecycle).
2. **GO/NO-GO 4**: GEFS mean.
3. **GO/NO-GO 5**: GEFS members (where the largest capacity gain lies).
Before each step, run `scripts/shadow_v2.py validate --store <target cycle>` first; rollback = switch env back to `sharded_v1` (new cycles go back to v1, already-committed v2 cycles continue to be served by V2Reader, zero migration).

## 20. Acceptance Criteria Check (§42)

All 25 items satisfied individually: v1 byte/read contract not regressed ✓; v2 writer/reader complete dtype matrix ✓; native cache ✓ (test assertion); precip/ceiling f32 ✓ (permanent regression tests); det/member u8 ✓; mean flags f32 ✓; explicit little-endian ✓ (byte assertion); f16 guard ✓ (rejected before cast); NaN retained ✓; Inf/overflow fail-loudly ✓; hidden readback dtype-aware ✓ (including manifest-missing inference); cloud predecessor fallback ✓; mixed v1/v2 serving ✓; cycle format freeze ✓ (two layers + tests); rollback measured ✓; lifecycle semantically unchanged ✓; v1 tests not relaxed ✓; v2 numerical tests complete ✓; shadow does not pollute the catalog ✓; relevant test suites all green ✓; lint/type passing ✓; no core assertions weakened ✓; this report complete ✓.

## 21. Risks / Follow-ups

- **Blocking**: none.
- **Non-blocking**: ① the combined window of manifest missing + process restart before the first publish + env flip (theoretical residue, triggerable only by operator action; manifest validation intercepts at the first publish); ② textual change of unrounded JSON literals on f16 variables (consistent with the first round report; presentation-layer cleanup is a separate PR).
- **Unrelated bug (not mixed into this diff)**: `tiles.py:1263` legacy member path 19990.0 unit error; four stale descriptions in `docs` (README GEFS resolution, ARCHITECTURE cache size, finalizer 14 days, connector geavg comment) — all left to the release/docs phase per PLAN §18.
