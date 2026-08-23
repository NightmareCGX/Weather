# Engineering Report — Complete Store↔Catalog Reconciliation (Multi-Region Run Status Fix)

**Date:** 2026-08-23
**Branch:** `bug/concurrency_fix`
**Status:** Implemented, validated on Windows **and** Linux/CI-equivalent, real PG+MinIO acceptance PASSED (GFS 5/5 READY, GEFS 3×5 READY). **Not committed** (per task contract).

---

## 1. Exact Root Cause

The CLI coordinator path (`_ingest_one_run` → region workers → `RunCoordinator.write_region_worker`) **commits forecast regions to the Zarr store and writes committed markers, but never calls `record_run` per region.** Catalog rows for a run are created **once** by `_resolve_run_id`, which calls `record_run` with a **synthetic dataset** (`_synthetic_spec_dataset`) that covers **only the first expected lead** and (for GEFS) **zero member rows**.

The finalizer's `_reconcile_catalog_to_store` was **delete-only**: it removed catalog rows whose lead/member was absent from the store, but **never restored** rows for regions physically committed in the store. Therefore:

```
store committed = {0,12,24,36,48}   (5 regions, all real data)
catalog        = {0}                (only the synthetic seed's first lead)
```

`_derive_run_status` requires **both**:
- **completeness** — `expected ⊆ catalog`; AND
- **consistency gate** — `catalog == store`.

The catalog lag behind the store fails the consistency gate → the run stayed `partial` even though every region was physically committed.

## 2. Exact Successful-Ingestion → Finalizer → Catalog Flow

```
CLI coordinator path (_ingest_one_run, cli.py)
  ├─ _resolve_run_id (fresh run): record_run(db, spec, _synthetic_spec_dataset(spec))
  │     → creates model/version/grid/variables + forecast_products for FIRST expected lead
  │       (GEFS: NO member rows — the synthetic dataset has no member axis)
  ├─ initialize_run_store + pre_update_wave (EXCLUSIVE gate, UPDATING markers)
  ├─ region workers (write_region_worker):
  │     → Zarr region write + COMPLETE marker  (NO record_run — this is the gap)
  ├─ finalize_run (EXCLUSIVE gate):
  │     → read committed markers → CommittedState
  │     → _reconcile_catalog_to_store (was DELETE-ONLY)
  │     → _derive_run_status
```

## 3. Why Physically Committed Regions Can Be Missing From the Catalog

1. `record_run` is called **once** per run (via `_resolve_run_id`), not per region.
2. The synthetic spec dataset only carries the **first** expected lead's products (and no GEFS member axis).
3. The **coordinator region workers** write Zarr+markers only — they never upsert catalog rows.
4. The **finalizer reconciliation only deletes** stale rows; it never inserts missing ones.

So any run with **more than one region** (or any GEFS member beyond the synthetic seed's) leaves those regions' catalog rows permanently absent.

## 4. Why Existing Tests Did Not Catch It

- `test_coordinator.py` / `test_coordinator_coldstart.py` use **single-lead expected sets** and assert store init / no self-block — not multi-region READY through the coordinator.
- The only multi-region READY test (`test_ensemble_committed_pairs_detected_and_ready` in `test_parallel_region.py`) uses the **library path** (`pipeline.ingest_grib_file` → `record_ingested_dataset` per region), which DOES create per-region rows — and is guarded to non-live stores only.
- The **CLI coordinator path** — the real production path — was never exercised to READY with a multi-region run in the unit tests. The real acceptance (prior session) exposed it.

## 5. Exact Meaning of READY/PARTIAL for GFS and GEFS

`_derive_run_status` (`catalog.py:772`):

| State | Meaning |
|---|---|
| **READY** | `expected ⊆ committed_catalog` (completeness) AND `committed_catalog == committed_store` (consistency gate). |
| **PARTIAL** | Some expected region committed but not all, OR catalog ≠ store (stale-at-catalog OR catalog-lagging-store). |
| **PROCESSING** | Nothing committed yet, or no declared expectations. |

- **GFS**: `expected_leads` = the run's declared lead list. All declared leads committed + catalog == store → READY.
- **GEFS**: `expected_members × expected_leads` Cartesian product must all be committed + catalog == store → READY. A GEFS run that declares members `1..30` (CLI default) with only 1,2,3 committed is correctly PARTIAL.

## 6. Are Expected GEFS Members Invocation-Specific or Model-Global?

**Invocation-specific.** `_build_spec` sets `expected_members=tuple(spec.members)`, and `expand_run_specs._run_members` resolves:
- `--member 1 2 3` → `expected_members=(1,2,3)` (invocation-specific).
- no `--member` → `expected_members=(1..30)` (model-global **default**, not a hard contract).

So a GEFS batch declared with `--member 1 2 3` that commits all 3×5 regions **should** be READY (its declared scope is complete). This is the intended behavior the fix enables.

## 7. Files Changed

| File | Change |
|---|---|
| `services/ingestion/src/ingestion/core/catalog.py` | `_reconcile_catalog_to_store` gained a `spec` param and now **restores** missing `forecast_products`, `ensemble_members`, and `ensemble_member_products` (in addition to deleting stale). `record_run` passes its `spec`. |
| `services/ingestion/src/ingestion/core/coordinator.py` | `finalize_run` passes `spec` to `_reconcile_catalog_to_store`. |
| `services/ingestion/tests/test_catalog_reconcile.py` | **New.** 16 regression tests (Section 11). |
| `services/ingestion/src/ingestion/cli.py`, `.gitignore` | Unchanged in this task (carried from the process-isolation work). |

No changes to: `record_run`'s row-creation logic, `_derive_run_status`, `_store_consistency_holds`, `CommittedState`, `read_committed_state`, the region-write/locking/marker protocol, the decode architecture, or the parser.

## 8. Exact Reconciliation Algorithm: Before vs After

**Before (delete-only):**
```
committed = read_committed_state(store)
delete ensemble_member_products not in committed.pairs
delete ensemble_members not in committed.members
delete forecast_products not in committed.leads
derive_status(catalog, expected, committed)   # catalog still missing rows
```

**After (bi-directional):**
```
committed = read_committed_state(store)   # only COMPLETE marker evidence
# stale removal (unchanged)
delete ensemble_member_products not in committed.pairs
delete ensemble_members   not in committed.members
delete forecast_products  not in committed.leads
# missing restoration (new)
upsert ensemble_members   for each committed member
upsert ensemble_member_products for each committed (member, lead) pair
upsert forecast_products  for each committed lead × variable
derive_status(catalog, expected, committed)   # catalog now == store
```

## 9. How Missing Product Metadata Is Reconstructed Safely

Only **authoritative** metadata is used — no invented defaults:

| Field | Source |
|---|---|
| variable_codes | `spec.variables` (the run's declared catalog variables) |
| grid_code | `spec.grid_id` |
| product_type | `spec.product_type` |
| zarr_chunk_path | `spec.zarr_store_path` or `run.zarr_store_path` |
| member_name | `spec.model_id` + real member number |
| product/member IDs | exact deterministic convention from `record_run` (`product_{run}_{var}_{grid}_{type}_{lead}`, `member_{n}_{run}`, `member_product_{n}_{lead}_{run}`) |

If a future committed-marker schema lacks sufficient metadata, the requirement to report that explicitly is honored — today all required metadata is present in `spec`. No marker-schema change was needed.

## 10. Transaction / Concurrency / Idempotency Design

- **Atomicity**: `_reconcile_catalog_to_store` runs in the SAME transaction as status derivation. A failure rolls back both.
- **Concurrency safety**: production reconciliation runs under the advisory **EXCLUSIVE store gate** (`finalize_run` acquires EXCLUSIVE admission + gate before reconciling), so two finalizers on the same run serialize. The restoration uses `_get_or_create` which respects the DB unique constraints (`uq_forecast_product_coords`, `uq_ensemble_member_index`, `uq_ensemble_member_product`) — a concurrent duplicate insert hits the constraint and is handled by the existing retry/upsert semantics, not swallowed.
- **Idempotency**: deterministic IDs + `_get_or_create` → re-running reconciliation never duplicates, never mutates identity, never churns IDs. Verified by `test_repeated_reconciliation_idempotent` and `test_concurrent_reconcile_no_duplicates`.

## 11. New Regression Tests (`tests/test_catalog_reconcile.py`, 16 tests)

Deterministic:
1. `test_deterministic_missing_lead_restored` (Case C)
2. `test_deterministic_stale_lead_removed` (Case B)
3. `test_deterministic_stale_and_missing_reconciled` (Case D)
4. `test_healthy_catalog_unchanged` (Case A)

Ensemble:
5. `test_ensemble_missing_pair_restored` (Case E)
6. `test_ensemble_stale_pair_removed` (Case F)
7. `test_ensemble_stale_and_missing_reconciled`
8. `test_ensemble_missing_multiple_pairs_restored`

Safety / semantics:
9. `test_no_row_created_for_uncommitted_region` (Case G)
10. `test_repeated_reconciliation_idempotent` (Case H)
11. `test_concurrent_reconcile_no_duplicates` (Case I)
12. `test_run_ready_when_deterministic_expected_scope_complete`
13. `test_run_partial_when_expected_leads_genuinely_missing`
14. `test_ensemble_ready_only_when_declared_matrix_complete`
15. `test_ensemble_partial_when_declared_members_missing`

Coordinator integration:
16. `test_coordinator_multi_lead_run_reaches_ready` — a real 3-region coordinated run now reaches READY (proven to FAIL without the fix).

## 12. Full Validation Results

### Windows (development host)

| Suite | Result |
|---|---|
| Full ingestion suite + MinIO (`WEATHER_TEST_MINIO=1`) | **214 passed** |
| Reconciliation tests | **16 passed** |
| ruff | **PASS** |
| mypy | **PASS** |

### Linux / CI-equivalent (Docker `python:3.12-slim` + libeccodes, host PG/Redis/MinIO via `--network=host`, pytest 8)

| Suite | Result |
|---|---|
| Full ingestion suite + MinIO | **213 passed, 1 skipped** (Windows-only spawn probe) |
| ruff (CI-pinned 0.3.4) | **PASS** |
| mypy | **PASS** |

### GitHub CI Mapping

| CI job | Local equivalent | Result |
|---|---|---|
| `ingestion-tests` | full pytest + PG + Redis + MinIO (Windows + Linux) | PASS (214 / 213) |
| `python-quality` | ruff 0.3.4 + mypy (Windows + Linux) | PASS |
| `domain-tests`, `api-tests`, `frontend-*`, `container-builds` | not touched by this change | n/a |

### Non-reproducible checks
None material; the container-build decode smoke is reproducible and was re-validated with the rebuilt image.

## 13. Real GFS 5-Lead Acceptance

```
store committed leads : [0, 12, 24, 36, 48]
catalog committed leads: [0, 12, 24, 36, 48]
difference             : []
status                 : ready
```
Store dims: (lead=5, lat=721, lon=1440). 5,191,200/5,191,200 cells non-NaN (100%).

## 14. Real GEFS 3×5 Acceptance

```
store committed pairs : (1,0)(1,12)(1,24)(1,36)(1,48)(2,0)(2,12)(2,24)(2,36)(2,48)(3,0)(3,12)(3,24)(3,36)(3,48)   [15/15]
catalog committed pairs: (same 15 pairs)
difference (pairs)    : []
status                : ready
```
Store dims: (member=3, lead=5, lat=721, lon=1440), member coords `[1,2,3]`. 15,573,600/15,573,600 cells non-NaN (100%). Catalog `ensemble_members=[1,2,3]`, `ensemble_member_products=15`.

## 15. Store Set vs Catalog Set Comparison

| Run | Store committed | Catalog committed | Difference |
|---|---|---|---|
| GFS | `{0,12,24,36,48}` | `{0,12,24,36,48}` | `∅` |
| GEFS | 15 pairs `(m,l)` for m∈{1,2,3}, l∈{0,12,24,36,48} | same 15 pairs | `∅` |

## 16. Final Run Statuses and Justification

| Run | Declared expected scope | Committed store | Status | Justification |
|---|---|---|---|---|
| GFS | leads `(0,12,24,36,48)` | `{0,12,24,36,48}` | **READY** | every declared lead committed + catalog == store |
| GEFS | members `(1,2,3)` × leads `(0,12,24,36,48)` | full 3×5 matrix | **READY** | every declared pair committed + catalog == store |

Both runs are legitimately READY because their **invocation-specific declared scopes** are fully committed. (A GEFS run that declared `1..30` with only 1,2,3 would correctly be PARTIAL — that distinction is preserved.)

## 17. API Visibility Impact

Every API read path filters **`ModelRun.status == "ready"`**:
- `reader_gate.py:211` re-validates `status == "ready"` before serving a store (production reader gate).
- `availability.py:105`, `point_forecast.py` (multiple), `tiles.py:548/594`, `verification.py:220`, `ensemble_data.py:95/157` all select `status == "ready"` runs.

**Before the fix**: a successful multi-region run stayed `partial` → **invisible** to `/v1/runs`, `/v1/points`, `/v1/maps`, `/v1/ensembles`, `/v1/probabilities` — even though its store was 100% populated.

**After the fix**: complete runs reach READY → become selectable → the existing API serves them correctly. **No API behavior change was needed or made.**

## 18. Confirmation: Same-Cycle, Live-Store, Mixed-Model, Region-Write, Process-Isolation Unchanged

- **Same-cycle re-ingestion**: `test_coordinator_coldstart.py` (incl. GEFS batch re-ingest) passes; `_reconcile_catalog_to_store` still runs within the run's transaction and respects same-cycle PATCH semantics.
- **Live-store overwrite protection**: `guard_full_overwrite` / `is_ready_run_store` untouched; cold-start tests pass.
- **Mixed GFS/GEFS**: `test_mixed_model.py` passes unchanged.
- **Region-write concurrency**: `coordinator.py` lock/gate/marker/inventory logic untouched (only the one reconcile call now passes `spec`).
- **Process-isolated decode**: `test_decode_worker.py` (13 tests) passes unchanged.
- **Full-win tests**: 214 Windows + 213 Linux all pass, proving the fix composes cleanly with all prior guarantees.

## 19. Remaining Risks / Technical Debt

1. **The library path (`ingest_grib_file`) and coordinator path now both reconcile bi-directionally** — consistency preserved. But the library path is limited to non-live stores; a future library-path caller with a live store would still need the coordinator.
2. **CLI default GEFS expectation** (`1..30`) means a partial-member CLI run (without `--member`) is PARTIAL by design. Documented, not a bug.
3. **No marker-schema expansion was needed** — the committed state + run spec carry all required metadata. If a future store format drops variable metadata from `spec`, restoration would need to report rather than guess (documented in the function docstring).
4. **Empty `spec.variables`** skips `forecast_products` restoration (delete-only for products) — models with no declared variables can't reconstruct product rows. Today every supported model declares variables.
5. The acceptance wrote to local PG (`m5-rec/` rows) and left 3.3GB of staged real files in `dl_real/` (gitignored). Cleanup is optional.

## 20. Confirmation: No Unrelated Code Was Changed

The full `git diff` for this task is limited to:
- `services/ingestion/src/ingestion/core/catalog.py` (reconciliation bi-direction),
- `services/ingestion/src/ingestion/core/coordinator.py` (1-line spec pass-through),
- `services/ingestion/tests/test_catalog_reconcile.py` (new tests).

The `cli.py` and `decode_worker.py` changes shown in `git status` are **carried from the prior process-isolation task** (already validated, not modified here). No changes to the API, region-write/locking, markers/inventory, Zarr writer, parser, config, or DB schema.

---

## Final Verdict

**`READY FOR COMMIT`** (per the Windows + Linux/CI-equivalent gate). The patch:
- ✓ proves the delete-only reconciliation root cause and fixes it with **bi-directional** restoration,
- ✓ keeps the store/committed-marker state as the only source of truth (never scans raw array values),
- ✓ preserves stale-row deletion and the READY/PARTIAL completeness-vs-consistency semantics,
- ✓ passes 214 Windows + 213 Linux tests, ruff, mypy, and the real PG+MinIO mixed acceptance (GFS 5/5 READY, GEFS 3×5 READY, store == catalog, difference ∅),
- ✓ does not weaken READY criteria, does not bypass constraints, and does not change the API.