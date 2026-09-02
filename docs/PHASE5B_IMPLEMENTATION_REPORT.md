# Phase 5B — Implementation Report

**Scope delivered:** Checkpoint 5B-1 (wave-target / cycle-horizon split), runner extraction, and
Checkpoint 5B-2 (read-only NOAA upstream discovery). No scheduler, polling, barrier, frontier,
supersession, retention, GC, serving, storage-format, or catalog-schema changes.

---

## A. Files changed

### New production files

| File | Why |
|---|---|
| `packages/domain/src/domain/horizon.py` | The authoritative **canonical cycle-horizon contract** (see §B): per-model registry `MODEL_CANONICAL_HORIZONS` (gfs/gefs → 81 leads, 0–240 h @ 3 h), `canonical_lead_time_hours()`, and `register_canonical_lead_horizon()` (test/startup injection, mirroring `domain.coverage.register_expected_members`). |
| `services/ingestion/src/ingestion/core/wave_runner.py` | The **reusable wave executor**, mechanically extracted from `ingestion/cli.py` (see §E). Owns `RunSpec` (wave targets), `ConcurrencyPlan` + stage-capacity derivation, staging/destination/cleanup helpers, decode-and-normalize parent-side plumbing, catalog-session plumbing (`_catalog_session_factory` injectable), `_build_spec`, `_resolve_run_id`, and `_run_wave`. Enforces the targets-vs-horizon split at wave start. |
| `services/ingestion/src/ingestion/providers/noaa/discovery.py` | **Read-only upstream discovery** (§F–G): anonymous S3 `ListObjectsV2` per cycle over product-scoped prefixes, paginated; immutable `CycleSnapshot` / `RegionArtifacts` / `ArtifactObservation`; data + `.idx` completeness predicate; snapshot fingerprint + `publication_changed()` diff helper; distinct error classes (`DiscoveryUnavailableError`, `DiscoveryInvalidResponseError`, `DiscoveryPaginationError`). |

### Modified production files

| File | Why |
|---|---|
| `services/ingestion/src/ingestion/cli.py` | Slimmed CLI: keeps argparse, batch/manifest expansion, store-path validation, `_run_ingest`/`_ingest_one_run` (delegating to the runner); re-exports the moved runner API so existing imports keep working (−1269 / +59 lines). Docstring and `--lead-time-hours`/`--member` help updated to describe wave targets. |
| `.env.example` | Documented the already-supported NOAA connector settings that were missing (`NOAA_DOWNLOAD_SOURCE`, `AWS_GFS_BASE_URL`, `AWS_GEFS_BASE_URL`, `ENABLE_NOMADS_FALLBACK`, `ENABLE_SELECTIVE_DOWNLOAD`). No new config keys were needed for this stage. |

### New tests

| File | Contents |
|---|---|
| `packages/domain/tests/test_horizon.py` | 9 tests: 81-lead 0–240 @ 3 h contract for both models, registry shape, unknown-model errors, default fallback, id normalization, reduced-horizon registration + validation errors (empty / negative / non-increasing). |
| `services/ingestion/tests/test_incremental_wave.py` | 6 tests — the required regression classes (§D below): `_build_spec` horizon wiring, out-of-horizon target refusal, GFS + GEFS disjoint-subset accumulation, GFS + GEFS full-horizon convergence (GEFS asserts the exact 30 × horizon `(member, lead)` pair set). Exercises the real runner, coordinator, markers, sharded_v1 local stores, settled-lead publication, and catalog reconciliation; downloads/decodes are stubbed and the horizon is reduced via the domain registry fixture contract. |
| `services/ingestion/tests/test_noaa_discovery.py` | 22 offline tests (§F/G): GFS + GEFS availability predicates, canonical vs observed lead views, pagination (2 pages and a >1000-object GEFS fixture), empty prefixes, unrelated/malformed key handling, `gec00`/`geavg`/`gespr` exclusion, member parsing, partial publication progression, snapshot fingerprint/diff (8/30 → 22/30 activity), transport/5xx/403-error-document/malformed-XML/truncated-without-token/runaway-pagination error semantics. All via `httpx.MockTransport` fixtures — zero live-upstream dependency. |

### Updated tests (mechanical + intentional)

| File | Change |
|---|---|
| `test_cli.py`, `test_cancel_stages.py`, `test_coordinator.py`, `test_coordinator_coldstart.py`, `test_deferred_db_checkout.py`, `test_decode_worker.py`, `test_observability.py`, `test_resident_dataset_bound.py`, `test_sharded_storage.py`, `test_small_pool_stress.py`, `test_write_backpressure.py`, `test_concurrency_derivation.py` | (1) `RunSpec(lead_time_hours=…)` → `RunSpec(target_lead_time_hours=…)`; (2) monkeypatch targets `ingestion.cli._catalog_session_factory` / `_decode_and_normalize` / `_validate_requested_lead` / `_detect_effective_cpus` repointed to `ingestion.core.wave_runner` (the implementation moved; the injectable is read there now); (3) CLI-path single-lead status expectations `ready` → `partial` (§I — the old expectation encoded the coupled behavior); (4) `test_mixed_model.py` multi-lead store-axis assertion updated to the canonical horizon with committed data still asserted exactly at the requested leads. |

---

## B. Canonical horizon ownership

The canonical 0–240 h / 3 h / 81-lead contract lives in **`packages/domain/src/domain/horizon.py`** as
`MODEL_CANONICAL_HORIZONS = {"gfs": (0, 3, …, 240), "gefs": (0, 3, …, 240)}`, read through
`canonical_lead_time_hours(model_id)`.

Why this location (per §4 of the task):

- The horizon is a **model/product contract** shared by big-batch and (future) realtime ingestion, so it belongs in the shared, dependency-free domain package — the same package that already owns the other fixed product contract, `domain.coverage.MODEL_EXPECTED_MEMBERS` (`gfs: 1`, `gefs: 30`). The two registries now sit side by side as the complete per-model product contract.
- **One authoritative source**: `_build_spec` (the only place big-batch turns a `RunSpec` into a `RunCatalogSpec`) calls `canonical_lead_time_hours(spec.model)`. No test, CLI flag, discovery code, or future scheduler re-declares `[0,3,…,240]`; the new tests assert *against* `canonical_lead_time_hours(...)`.
- It is deliberately **not** a `REALTIME_*` setting, per the Phase 5B correction: a product contract must not be silently mutable per deployment. `register_canonical_lead_horizon()` exists only for the test fixture contract (same pattern as `coverage.register_expected_members`, used by `test_case_j/k` style tests) — production keeps the registry literal. The 81-lead default is asserted explicitly in `test_build_spec_wires_canonical_horizon_not_targets`.
- Deliberate scope boundary: `domain.horizon` does not know about discovery/serving; the connector's 0–384 h URL-validation ceiling is unchanged (transport limit, not product horizon).

---

## C. Wave-vs-horizon split (data flow)

Every consumer of the old `RunSpec.lead_time_hours` was traced and classified before changing it:

**Wave-target concerns** (now read `RunSpec.target_lead_time_hours` / `RunSpec.members`):

```
CLI --lead-time-hours / --member / manifest
    → expand_run_specs / _parse_manifest        (cli.py — CLI surface unchanged)
    → RunSpec.target_lead_time_hours, RunSpec.members     [wave targets]
        → work items (member, lead) ordering + seed selection   (wave_runner._run_wave)
        → WaveRegion targets + per-region generations           (wave_runner._run_wave)
        → staging destination filenames + post-commit cleanup    (wave_runner._destination_for/_cleanup_sources)
        → settled-lead tracking (lead_pending) + publish_settled_lead triggers
        → predecessor (lead−3) in-wave coordination
        → progress tracker totals, dry-run print, failure accounting
```

**Cycle-horizon concerns** (read `RunCatalogSpec.expected_lead_time_hours` / `expected_members`,
built once in `_build_spec` from `domain.horizon`):

```
domain.horizon.canonical_lead_time_hours(model)        (+ gep01..gep30 for GEFS)
    → _build_spec → RunCatalogSpec.expected_*                    [cycle horizon]
        → initialize_run_store → prepare_run_store   (store lead/member axis pre-allocation)
        → finalize_run                               (readiness expectation)
            → _derive_run_status:  deterministic: expected ⊆ committed
                                   ensemble:     committed pairs == expected pairs (30 × horizon)
        → write_region_worker expected_* passthrough (used only by the fresh-store library path)
        → record_run (via _resolve_run_id) expected sets
```

**Unchanged / neutral:** catalog reconciliation is driven by actual committed marker evidence (independent of both concepts); `publish_settled_lead`'s member check uses the invocation's target members (subset-safe, idempotent upserts); the CLI manifest schema, flags, and exit codes are unchanged.

**Split enforcement (impossible to lose silently):** `_run_wave` raises `ValueError` when
`catalog_spec.expected_lead_time_hours` is empty (no horizon → no pre-allocation, no meaningful
readiness), when any target lead lies outside the canonical horizon, or when any target member lies
outside the required member contract. `RunSpec`'s field rename makes an accidental re-coupling a
visible call-site change rather than an invisible value flow. There is still **no store-resize
logic**: a store is pre-allocated once with the full horizon.

---

## D. Incremental-ingestion evidence (`test_incremental_wave.py`)

Reduced injected horizons are used for runtime (the domain registry's fixture contract); production
default (81 leads) is asserted separately. All four scenarios run the real runner/coordinator/
markers/sharded_v1 local stores/catalog reconciliation end to end.

| Required scenario | GFS | GEFS |
|---|---|---|
| **Test A — disjoint subset accumulation** (run1 `[0,3,6]`/`[0,3]`, run2 `[9,12]`/`[6,9]`) | ✅ run 2 succeeds, no `StoreSchemaMismatchError`; committed data = union `{0,3,6,9,12}` with per-lead values verified intact; store lead axis = full horizon; `forecast_products` = union; status `partial` | ✅ same; committed `(member, lead)` pairs = full union (30 × 4 leads) with member identity verified (`member 17 @ lead 6`); `ensemble_member_products` = union; status `partial` |
| **Test B — eventual full-horizon convergence** (waves unioning to the horizon) | ✅ 4 waves over 8-lead horizon → final status `ready`; committed set = horizon; products = horizon | ✅ 3 waves (30 members each) over 3-lead horizon → final status `ready`; pair set = exact required 30 × 3; `ensemble_member_products` complete |
| Refusal guard | ✅ target lead outside the canonical horizon → `ValueError` before any work | ✅ (same guard; member-contract guard shares the code path) |

Pre-split, run 2 in Test A raised `StoreSchemaMismatchError("lead N not found in store coordinate")`
(`coordinator.py`), and the GEFS Test B final wave stayed `partial` (exact-pairs check vs the
invoking subset). Both failure modes are now covered by passing tests that encode the accepted
architecture.

---

## E. Runner extraction

- Moved **verbatim** from `ingestion/cli.py` into `ingestion/core/wave_runner.py`: `RunSpec`
  (field renamed per the split), `ConcurrencyPlan`, `_detect_effective_cpus`,
  `_resolve_concurrency_plan`, `_destination_for`, `_cleanup_sources`/`_cleanup_source`,
  `_decode_and_normalize`, `_region_id_for`, `_new_generation`, `_catalog_session*` plumbing,
  `_resolve_run_id`, `_synthetic_spec_dataset`, `_build_spec`, `_run_wave`, and the
  `DEFAULT_VARIABLES` / center/model/grid metadata constants.
- `ingestion/cli.py` retains argparse/manifest expansion/store-path validation/`_run_ingest`/
  `_ingest_one_run` and delegates; it re-exports the runner API so existing imports keep working.
- `_run_wave`'s signature is unchanged (`spec, args, catalog_spec, store_path, concurrency,
  failures`); `args` is read only via `getattr` (`download_dir`, `no_progress`, `keep_downloads`,
  `lock_timeout`), so a Phase 5C scheduler can pass a plain namespace. Coordinator protocol,
  concurrency/semaphore stages, cancellation drain, progress/observability reporting, error
  propagation, and cleanup behavior are byte-for-byte the extracted originals — no coordinator
  redesign, no pipeline copy, no new abstraction.

---

## F. Discovery architecture

- **Prefixes** (product-scoped, per cycle — never the broad `atmos/` prefix):
  - GFS: `gfs.{date}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f` (data + `.idx` in one page);
  - GEFS: `gefs.{date}/{hh}/atmos/pgrb2sp25/` (shared across members — lowest request count per
    the task's preference; `geavg`/`gespr`/`gec00` keys are parsed out and ignored with a count +
    bounded sample for diagnostics).
- **Pagination**: standard ListObjectsV2 continuation-token loop at `max-keys=1000`, with a hard
  page budget (`_MAX_LIST_PAGES = 64`) so a broken loop fails with `DiscoveryPaginationError`
  instead of running away; a truncated page without a token is an invalid response.
- **Artifact predicate**: complete ⇔ data object **and** matching `.idx` sidecar listed in the same
  snapshot (`RegionArtifacts.is_complete`). Data-without-idx and idx-without-data are distinct,
  incomplete states.
- **Snapshot representation**: immutable `CycleSnapshot` keyed by `(member|None, lead)` →
  `RegionArtifacts(data, idx)`; each side carries key/size/etag/last-modified for future activity
  diffs. Query surface (per-model reality only, no barrier logic): `observed_leads()`,
  `highest_observed_lead()`, `is_artifact_complete(member, lead)`, `available_members(lead)`,
  `complete_member_count(lead)`, `missing_members(lead, expected)`, `available_leads(sequence)`,
  `available_member_leads(sequence)`, `fingerprint()`, plus `publication_changed(before, after)`.
- **Canonical lead filtering**: discovery observes everything matching the product grammar
  (including upstream-only GFS hourly f001/f002/…); sequence-aware helpers default to
  `domain.horizon.canonical_lead_time_hours(model)` so hourly leads can never enter platform
  interpretation. Filtering is a read-side helper — discovery still records upstream reality.
- **Key grammar**: full-key anchored regexes per product (`gfs.t{hh}z.pgrb2.0p25.f{ddd}(.idx)?`,
  `gep{NN}.t{hh}z.pgrb2s.0p25.f{ddd}(.idx)?`) with a defensive cycle-hour match; anything else is
  ignored (counted + sampled), including malformed keys that nearly match the grammar.
- **Network semantics**: bounded retries (existing `DOWNLOAD_RETRIES` / `RETRY_BACKOFF_SECONDS`
  conventions) for transport errors and 5xx → `DiscoveryUnavailableError`; non-200 (with S3 error
  document code extraction, e.g. `AccessDenied`), S3 error documents on 200, unparseable XML, and
  token-less truncation → `DiscoveryInvalidResponseError`; page-budget overrun →
  `DiscoveryPaginationError`. A successful empty listing is a valid empty snapshot. Network
  failures are never mapped to "nothing published".
- **Read-only guarantee**: the module imports only `ingestion.core.{base,config}`; nothing in the
  ingestion, serving, or CLI paths imports discovery (verified by grep) — the big-batch hot path
  performs zero discovery work.

## G. Remote-operation complexity

Per cycle snapshot (worst case): **GFS = 1** `ListObjectsV2` request; **GEFS = ⌈5346/1000⌉ = 6**
requests (shared product prefix, probed in Phase 5A). No HEAD/GET per artifact; no O(leads ×
members) probing. A hypothetical GFS+GEFS poll is ≤ 7 requests total.

---

## H. Test results

Environment: fresh local Windows venv (Python 3.12.13) built from `services/ingestion/poetry.lock`
pinned versions; `domain` installed as an editable path dependency (mirroring CI's poetry install).

| Gate | Command (cwd) | Result |
|---|---|---|
| Domain tests + 100% coverage gate | `packages/domain`: `python -m pytest` | **447 passed** (incl. 9 new horizon tests), coverage **100.00%** |
| Ingestion suite (full) | `services/ingestion`: `python -m pytest tests -q --ignore=tests/test_catalog_postgres.py` | **441 passed, 23 skipped, 0 failed**; 1 ERROR: `test_serving_contract_legacy_v2_unsharded_compatibility` — requires the CI PostgreSQL service container (`psycopg2.OperationalError: connection refused`); environment-only, unchanged by 5B |
| New Phase 5B tests | `tests/test_noaa_discovery.py tests/test_incremental_wave.py` | **28 passed** (22 discovery + 6 incremental) |
| Reproduction-set regression check | `tests/test_incremental_wave.py tests/test_observability.py tests/test_noaa_discovery.py` | 48 passed (see §I — the ordering bug this caught is fixed) |
| Domain lint | `packages/domain`: `python -m ruff check .` | All checks passed |
| Domain types | `packages/domain`: `python -m mypy` (strict) | no issues (22 files) |
| Ingestion lint | `services/ingestion`: `python -m ruff check .` | All checks passed |
| Ingestion types | `services/ingestion`: `python -m mypy` (strict, `src/ingestion`) | no issues (25 files) |
| API lint + types (untouched package sanity) | `services/api`: `python -m ruff check src`, `python -m mypy` | All checks passed / no issues (36 files) |

Non-reproducible on this Windows host (CI-only, unchanged by 5B): PostgreSQL/Redis/MinIO-backed
jobs (`api-tests`, `ingestion-tests` with `WEATHER_TEST_MINIO=1` real S3 round-trip,
`test_catalog_postgres.py`, the PG-dependent e2e test), `container-builds`. The 23 skips are the
MinIO/PG-gated integration tests; `libeccodes` availability matches CI via bundled Windows wheels.
No commit was created (engineering gate: Windows validation complete here; Linux/CI validation
happens on the CI runner).

## I. Regressions / unexpected findings

1. **Test-ordering pollution introduced and fixed (new tests).** The first full-suite run failed
   `test_observability.py::test_gefs_510_region_startup_delay_measurement`: my
   `test_gefs_full_horizon_converges_to_ready_exact_pairs` registered a reduced GEFS horizon
   without restoring the registry, so the later 510-region CLI test ran with horizon `0..6` and its
   targets were correctly refused by the new split guard ("Wave target lead(s) […] are outside the
   canonical cycle horizon 0..6"). The guard did exactly its job — the leak was in my fixture
   discipline. Fixed by giving that test the restoring `reduced_horizon` fixture; full suite is
   green. This is also a live demonstration of the split guard failing fast.
2. **Intentional expectation updates (old coupled behavior).** CLI-path tests asserting `ready`
   after a single-lead ingest (`test_cli.py` × 6) and the mixed-model store-axis assertion encoded
   the pre-split coupling; updated to `partial` / canonical-horizon axis with committed data still
   asserted at the requested leads (per §9 — the old expectations were genuinely invalid under the
   accepted architecture). The coordinator-level library test asserting `ready` with a
   caller-declared `expected_leads=(6,)` was left unchanged (caller-owned horizon contract).
3. **Mechanical test churn**: monkeypatch targets moved from `ingestion.cli` to
   `ingestion.core.wave_runner` (the implementation module now reads the injectables there).
4. No regressions were observed in coordinator, finalizer, marker, sharded_v1, catalog, connector,
   or decode behavior; no production hot-path code changed other than the CLI split/extraction.
5. Environment notes (not regressions): the PG-backed e2e ERROR and the 23 MinIO/PG skips require
   CI service containers, matching prior Windows-local behavior.

## J. Remaining Phase 5C prerequisites

Everything needed for the scheduler now exists; the following is the exact remaining work:

1. **Frontier planner** — pure functions over `CycleSnapshot` + durable committed state
   (catalog `forecast_products` / `ensemble_member_products`, or markers): contiguous complete-
   frontier walk over `canonical_lead_time_hours(model)`, committed-frontier reconstruction,
   pending-complete-lead set, next-blocked-lead + missing-artifact breakdown, and the predecessor
   eligibility rule (`lead % 6 == 0` requires lead−3 committed, else `MissingPredecessorLeadError`
   at write time).
2. **Shared GFS+GEFS completeness barrier** — scheduler-policy function combining
   `snapshot_gfs_cycle` + `snapshot_gefs_cycle` per lead (`complete(lead) = GFS artifact pair
   complete AND all 30 GEFS member artifact pairs complete`); must remain out of the discovery
   layer so future independent GFS/GEFS advance is a policy swap.
3. **Bounded wave batching** — `wave_max_leads` / `wave_max_wait` emission rule over pending
   complete leads (oldest-first age tracking in scheduler memory; correctness stays derivable from
   upstream + durable state each poll).
4. **Polling state machine** — ACTIVE (10 m) / PUBLISHING (2 m) / BACKOFF (30 m → 1 h) with jitter,
   using `publication_changed()` for activity (any snapshot change, not only frontier growth), and
   the §17 error taxonomy to distinguish "empty" from "unreachable" (backoff must not engage on
   discovery *failure* the same way as on idle).
5. **Scheduler process** — long-running loop calling `wave_runner._run_wave` (or a thin public
   wrapper around it) per emitted wave, with: cycle tracking/selection (minimal Phase 5 scope;
   supersession stays Phase 6), a PG-advisory leader lock (reuse `domain.locks` derivation) for
   double-start protection, graceful shutdown reusing the wave's non-abandoning cancel drain,
   scheduler-level structured logging (frontiers, poll phase, wave results), and the Phase 5C
   configuration block (`REALTIME_*` keys from the Phase 5A report §J — polling/wave settings were
   deliberately not added in 5B).
6. **Real-upstream E2E acceptance (5E)** — validate the live assumptions behind the discovery
   predicate (`.idx`-after-data timing on both buckets, NOMADS fallback caveat, bucket retention
   behavior) and tune `wave_max_leads`/`wave_max_wait` against measured publication windows
   (GFS ≈ 1.7 h, GEFS ≈ 3.5–5.5 h per cycle, probed 2026-09-02).
