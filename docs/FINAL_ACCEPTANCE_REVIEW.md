# FINAL ENGINEERING ACCEPTANCE REVIEW — M1–14 REMEDIATION

**Date:** 2026-08-13
**Mode:** Strict read-only final validation pass. No source/test/config/doc modified
during this review (only two audit artifacts created: this report and
`docs/ACCEPTANCE_FINDINGS_LOG.md`).
**Source of truth:** `docs/ACCEPTANCE_REMEDIATION_PLAN.md`,
`docs/REMEDIATION_IMPLEMENTATION_REPORT.md`, and the actual working tree.

---

## A. Executive Summary

The M1–14 remediation was validated independently. The core identity work
(Issues 2, 2B, cross-cutting cache identity) is **correct and PASS**, verified by
live reproduction of the original failure and the new invariants. Raster
performance, elevation, and ensemble UI are **PASS** (with the noted validation
gaps on real external data). Autocomplete is **PASS with validation gap** (no
live Google credentials available; provider path mocked).

**Two real regressions were found** that the implementation report did not disclose
and the remediation did not fix:

1. **FINDING-1** — Pre-existing M1-14 integration test
   `test_catalog_postgres.py::test_cli_production_entrypoint_ingests_and_serves`
   is broken by the new store-path validation (needs `--allow-custom-store` but
   was not updated). **1 FAIL** in the ingestion suite.
2. **FINDING-2** — Pre-existing M1-14 Playwright E2E test
   `forecast.spec.ts:75` asserts the OLD misleading `Ensemble Statistics (GFS)`
   heading that the remediation correctly removed; it was not updated. **1 FAIL**
   in the E2E suite.

Both are **un-updated pre-existing tests**, not functional defects in the new
behavior — but they are **red tests** in the suites, so the remediation is not
fully green as delivered.

**Redis-backed API tests (the report's noted "5 failures") were re-run with Redis
running: all pass (168/168 API).** The report's environmental gap is resolved.

---

## B. Git / Scope Verification

- **38 modified + 6 new** files = 44, matching the reported scope.
- New files: `services/api/src/api/services/{places,elevation}.py`,
  `services/api/tests/test_{places,elevation}.py`, plus the two report docs.
- No deleted files. No unrelated changes found in the diff review.
- **Untracked test residue:** `services/ingestion/downloads/` (CLI test GRIB
  copies, not gitignored). Not part of the remediation scope; a cleanup item.
- **Audit artifacts created by this review:** `docs/ACCEPTANCE_FINDINGS_LOG.md`
  (findings) and this report. These are new untracked files.

Scope is consistent with the remediation plan; no accidental deletions or
unrelated edits.

---

## C. Issue 1 — Raster

| Requirement | Evidence | Result |
|---|---|---|
| Vectorize per-pixel loop + lon alignment | `tiles.py` uses `np.interp` per RGB channel + `_align_longitudes` (no `np.vectorize`) | PASS |
| Server tile cache with full identity key | `_tile_cache_key(model, variable, level, z, x, y, lead, initial_time)` | PASS |
| Measurable improvement | Cold ~15–52 ms (this Windows box, global store) vs ~230–318 ms baseline; cached ~0.0006–0.077 ms | PASS (order confirmed) |
| Output correctness | Differential: new render byte-identical to reference on temperature + precipitation | PASS |
| Cache separation (model/cycle/lead/tile/variable) | 3+ distinct keys verified; A-reuse hit verified | PASS |

**Verdict: PASS** (report's exact ms figures not byte-reproducible on this
machine — platform variance, FINDING-6 — but the magnitude and correctness are
confirmed).

---

## D. Issue 2 — Cycle Identity

**Live reproductions (all PASS):**

- **Repro A** — Same-cycle `[0,6,12,18]` into one 00Z store: all leads present,
  correct `time` coord, `model_id`+`cycle_time` attrs, reopen preserves identity.
- **Repro B** — 12Z file into 00Z store raises `CycleStoreMismatchError`; store
  leads **unchanged**, time **unchanged**, no leftover temp artifacts.
- **Repro C** — `--cycle-hour 12` + `--store .../00/cycle.zarr` rejected by
  `validate_store_path` (ValueError, no silent write).
- Zarr self-description: `model_id` + `cycle_time` attrs written on first write,
  survive additional-lead merge, reopen, and API read.
- Storage separation: `gfs/00`, `gfs/12`, `gefs/00` resolve to 3 distinct S3
  paths.
- DB identity: run id version-scoped; `UNIQUE(model_version_id, cycle_time)`;
  `UNIQUE(run_id, variable_id, grid_id, product_type, lead_time_hours)`.

**Verdict: PASS** — the original failure is fixed with fail-fast domain errors,
and the store is never partially modified on a refused merge.

---

## E. Issue 2B — Batch Ingestion

- **Aligned expansion:** `--model gfs gefs --cycle-date D1 D2 --cycle-hour 0`
  → exactly 2 runs (not 4); misaligned lists rejected. Anti-Cartesian PASS.
- **Manifest:** 2 distinct run specs expand correctly with distinct leads.
- **Dry-run:** exit 0, prints resolved specs, no writes.
- **max-runs:** exceeding limit → SystemExit with clear message.
- **Per-run failure isolation:** valid run → exit 0; invalid run → exit 1; mixed
  batch → exit 1 with the valid lead still ingested (failures surfaced, not
  silently swallowed).
- **Idempotency:** re-ingesting the same lead → still 1 lead (no duplicate).

**Verdict: PASS**

---

## F. Issue 3 — Autocomplete

- Provider abstraction + Google Places (New) backend implemented, proxied
  server-side (key never in browser), session-token aware, `includedPrimaryTypes`
  to avoid Text-Search billing.
- `/v1/search?type=place` returns ranked suggestions; `/v1/search/places/{id}`
  resolves canonical place. Unit tests (mocked transport) pass (11 in
  `test_places.py` + search tests).
- Frontend uses `type=place`, min length 2, per-session token; E2E search→forecast
  flow passes (mocked Denver).
- **Real external validation NOT performed:** no `.env` / `GOOGLE_PLACES_API_KEY`
  configured in this environment. The provider is verified only via mocks.

**Verdict: PASS WITH VALIDATION GAP** (FINDING-3) — implementation appears
correct, but live Google Places acceptance is not possible without credentials.

---

## G. Issue 4 — Elevation

- Provider abstraction + local DEM (`DEMElevationProvider`, bilinear via NumPy,
  no scipy) + `RoundedElevationCache` (3-decimal rounding).
- **Real on-disk Zarr DEM path validated:** Denver 1600 m, Aspen 2400 m, plains
  710 m; no-data → None (never 0); cache collapses 6 same-bucket requests to 1
  provider call.
- `resolve_location` populates `elevation_m` for coordinates/cities.
- UI: "unavailable" (not `—`) verified by frontend test.
- **Production Copernicus GLO-30 not present** (`DEM_DATA_PATH=""`), so the
  specific production dataset is unexercised (FINDING-4); the code path is
  identical.

**Verdict: PASS WITH VALIDATION GAP** (FINDING-4) — the provider and data path
are correct and exercised with a real Zarr DEM, but not the production GLO-30
store.

---

## H. Issue 5 — Ensemble Capability

- `ForecastDashboard` renders the ensemble section **only** for ensemble-capable
  models; deterministic GFS shows no panel. Unit tests confirm: no "Ensemble
  Statistics" for GFS; distinct "Ensemble data is not yet available" for GEFS-no-data.
- Backend `/v1/ensembles?model=gfs` → 422 (unchanged, correct).
- **FINDING-2:** the pre-existing Playwright E2E test at `forecast.spec.ts:75`
  still asserts the old `Ensemble Statistics (GFS)` heading and now **fails**.
  The remediation's behavior is correct; the E2E test was not updated.

**Verdict: PASS (behavior) with 1 red E2E test (FINDING-2)** — the UI fix is
correct, but the E2E suite is not green.

---

## I. Cross-Cutting Forecast Identity Audit

- **Cache keys:** `point`, `probability`, `ensemble` all embed `cycle_time` in
  the hashed payload (verified at `cache.py:301` and by distinct-key tests).
  GFS 00Z+18 ≠ GFS 12Z+18; gfs ≠ gefs; lead 6 ≠ lead 18.
- **Raster cache key:** full identity tuple (model/variable/level/z/x/y/lead/
  initial_time), separated by cycle/lead/tile/model, A-reuse hit.
- **`initial_time` genuinely affects the query:** `_resolve_ready_dataset` and
  `resolve_latest_run_cycle_time` both filter `cycle_time` when provided; maps
  tests confirm unknown-cycle → 404 (no newest-fallback).
- **Full pipeline:** CLI → parse → Zarr (self-describing) → S3 (derived path) →
  DB (version-scoped run) → API (cycle-selectable) → cache (cycle-keyed) →
  frontend (selection carries initialTime). No layer collapses to
  `(model, lead, variable)`.

**Verdict: PASS**

---

## J. Regression / Compatibility Review

- `/v1/points`, `/v1/probabilities`, `/v1/ensembles`, raster maps: all API tests
  pass (168/168 with Redis+Postgres).
- Single-run ingestion: preserved (`--store` still works; documented path
  unchanged; custom path requires `--allow-custom-store` — a documented behavior
  change).
- **FINDING-1 (regression):** `test_catalog_postgres.py::test_cli_production_entrypoint_ingests_and_serves`
  fails (CLI now rejects the test's local `--store` without `--allow-custom-store`).
- Model defaults (`gfs`/`gefs`) unchanged. Frontend selection unchanged.
- Existing GEFS ensemble functionality: E2E GEFS test passes.

---

## K. Test Matrix

| Command | Result | Reason |
|---|---|---|
| `pytest tests/` (ingestion) | **87 passed, 1 skip, 1 FAIL** | FINDING-1 (store-path regression) |
| `pytest tests/` (api) with Redis+PG | **168 passed** | all incl. previously-failing Redis cache tests |
| `pytest` (domain) | **296 passed, 100%** | |
| `jest` (frontend) | **109 passed (19 suites)** | |
| `playwright test` (e2e) | **5 passed, 1 FAIL** | FINDING-2 (old ensemble heading test) |
| `ruff check` (ingestion+api) | PASS | |
| `mypy` (ingestion) | PASS | 13 files |
| `mypy` (api) | PASS | 32 files |
| `tsc --noEmit` (frontend) | PASS | |
| `prettier --check` (frontend) | PASS | |
| Issue 2 repro A/B/C (live) | PASS | cycle merge / rejection / store-path |
| Raster benchmark + cache separation | PASS | cold ~15–52 ms, cached ~0.001–0.077 ms |
| Elevation real-DEM path | PASS | Denver/Aspen/plains/ocean/no-data/cache |

---

## L. Remaining Risks / Gaps

1. **FINDING-1 (ingestion test regression):** `test_cli_production_entrypoint_ingests_and_serves`
   must add `--allow-custom-store` (or use a derived path) to be green again.
2. **FINDING-2 (E2E regression):** `forecast.spec.ts:75` must be updated to
   assert GFS shows **no** ensemble panel (matching the corrected behavior).
3. **FINDING-3 (autocomplete real-data gap):** no Google Places credentials in
   the environment; real autocomplete not exercised.
4. **FINDING-4 (elevation real-data gap):** production Copernicus GLO-30 store
   not present; code path exercised with a synthetic-but-real Zarr DEM.
5. **FINDING-5 (cleanup):** untracked `services/ingestion/downloads/` test
   residue; not gitignored.
6. **S3 atomic-write caveat:** `write_dataset_atomic` deletes the target before
   copying the stage (S3 has no rename), leaving a brief absence window. The
   plan documented this as best-effort; the API's run-identity validation and
   caches bound staleness. Not a blocker but worth noting.

---

## M. Final Verdict

### Issue-by-issue

| Issue | Verdict |
|---|---|
| 1 — Raster performance | **PASS** |
| 2 — Cycle identity / cross-cycle ingestion safety | **PASS** |
| 2B — Batch ingestion | **PASS** |
| 3 — Google Places autocomplete | **PASS WITH VALIDATION GAP** (no live credentials) |
| 4 — Coordinate elevation | **PASS WITH VALIDATION GAP** (no production GLO-30) |
| 5 — Deterministic vs ensemble UI | **PASS (behavior); 1 red E2E test (FINDING-2)** |

### Critical acceptance gates

1. Cross-cycle ingestion safely rejected — **PASS** (live repro B).
2. Wrong-cycle store cannot contaminate — **PASS** (store unchanged after refusal).
3. API cache cycle separation — **PASS** (all cache keys embed `cycle_time`).
4. Raster cache no cross-identity leak — **PASS** (full identity key + separation tests).
5. Redis-backed API tests executed — **PASS** (168/168 with Redis running; report's 5 failures resolved).
6. GFS does not render misleading ensemble — **PASS** (behavior); **FINDING-2** E2E test not updated.
7. Real location selection resolves coordinates — **PASS WITH VALIDATION GAP** (provider mocked; no live Google).
8. Elevation not permanently null without environmental reason — **PASS** (provider + cache work; DEM must be configured).

### OVERALL: PASS WITH VALIDATION GAPS

The remediation is functionally correct for all five issues, with the original
cross-cycle failure and cache-identity gaps demonstrably fixed. It cannot receive
an unconditional PASS because:

- **FINDING-1** and **FINDING-2** are two un-updated pre-existing M1-14 tests now
  red (1 ingestion integration test, 1 Playwright E2E test). These are **not**
  functional defects — they assert pre-remediation behavior the remediation
  intentionally changed — but the delivered tree is not fully green.
- **FINDING-3**/**FINDING-4**: live Google Places and production DEM data were
  not available in this environment, so those two external-service surfaces are
  validated through mocks/synthetic data, not real external services.

The environment-blocked items (Redis-backed tests) are resolved: all pass. The
remaining gaps are (a) two stale tests that must be updated to the corrected
behavior and (b) live-credential external validation. Neither indicates a
functional defect in the remediation's intended behavior.
