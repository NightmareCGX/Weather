# Phase 5A — Repository Reality & Architecture Investigation Report

**Phase 5: Upstream-Driven Realtime Lead-Wave Ingestion — investigation and proposed architecture only.**
No production code was changed. Evidence classes are labeled throughout:

- **[FACT]** — verified repository fact (file/function cited) or live-upstream observation (probed 2026-09-02).
- **[DESIGN]** — recommended design, not yet implemented.
- **[ASSUMPTION]** — requires real-upstream validation before/during implementation (5E).

---

## A. Executive conclusion

**Phase 5 is architecturally straightforward with current repository primitives, with exactly one narrow structural extension required.**

The existing big-batch engine is already, de facto, an incremental lead-wave engine:

- The CLI accepts arbitrary lead subsets and repeated subsets of the same cycle; every `(member, lead)` is committed as an independent region under a PostgreSQL advisory-lock protocol; the finalizer reconciles the catalog to whatever is actually committed; and serving already exposes `partial`/`processing` runs per committed lead with 85% ensemble coverage gating (Phase 3).
- The durable state needed to derive `pending = upstream − committed` already exists (catalog rows + store markers), so restart/reconciliation needs **no** scheduler-owned truth.

The **one structural gap** is that the run's *requested wave leads* and the cycle's *expected horizon* are the same tuple today (`RunSpec.lead_time_hours` → `RunCatalogSpec.expected_lead_time_hours`, `cli.py::_build_spec`). This single coupling causes two concrete failures for multi-wave ingestion:

1. **Store lead axis is pre-allocated from the first invocation's leads** (`zarr_writer.prepare_run_store`). A later wave committing a lead outside that axis fails hard (`coordinator.py:write_region_worker` → `StoreSchemaMismatchError: "lead N not found in store coordinate"`; `zarr_writer._coordinate_index` raises `ValueError`). This is a pre-existing big-batch limitation too, not a realtime-only defect.
2. **Ensemble run status can never become `ready` under per-wave expectations**: `_derive_run_status` (`catalog.py`) requires committed `(member, lead)` pairs to **exactly equal** `expected_members × expected_leads` of the *invoking* spec. A wave-2 finalize whose spec declares only wave-2 leads sees the full store's pair set ≠ wave-2 pairs → `partial` forever. (Deterministic runs converge because they use a subset check.)

The fix is small and safe: introduce the platform's **canonical per-model lead horizon** (81 leads, 0…240 h @ 3 h, per `docs/investigations/gfs-gefs-variable-inventory/README.md`), pre-allocate every new store with it, and finalize every wave against the **cumulative horizon** while the wave's *targets* remain the planner's lead subset. Everything else — discovery, barrier, polling, waves — is new code that sits **above** the existing coordinator and reuses it unchanged.

Upstream reality is favorable: anonymous S3 `ListObjectsV2` works on both NOAA Open Data buckets (verified live), so per-poll availability costs **1 request (GFS) / 6 requests (GEFS)** instead of O(leads × members) HEAD probes, and the `.idx` sidecar (already required by the selective-download path) is a reliable per-file completeness signal.

**No second ingestion engine. No sharded_v1 changes. No Phase 3 changes. No Phase 6 work.**

---

## B. Repository reality map

### B.1 Ingestion entry points and execution lifecycle

```
weather-ingest ingest --model gfs|gefs --cycle-date D --cycle-hour H
                      --lead-time-hours [...] [--member 1..30] [--manifest runs.json]
  services/ingestion/src/ingestion/cli.py
    _build_parser() / expand_run_specs() / _parse_manifest()   → list[RunSpec]
    _run_ingest() → _ingest_one_run(spec, args)                (per run spec; failures isolated)
      validate_store_path()      s3://weather-data/{model}/{date}/{hh}/cycle.zarr  (cli.py STORE_PATH_TEMPLATE)
      _build_spec()              RunCatalogSpec (variables=DEFAULT_VARIABLES,
                                 expected_lead_time_hours=tuple(spec.lead_time_hours),   ← the coupling
                                 expected_members=range(1,31) for GEFS)
      asyncio.run(_run_wave(...))                                (cli.py:982)
```

`_run_wave` (cli.py:982–1680) — the wave executor:

| Step | Component | File:function |
|---|---|---|
| Staging | run-scoped `downloads/staging_{model}_{cycle}_{uuid}/` | cli.py:1022 |
| Seed download → decode | `NOAAConnector.download` (`.idx` selective range GETs, NOMADS fallback), `DecodePool` (process-isolated ecCodes) | providers/noaa/connector.py, core/decode_worker.py |
| Catalog identity | existing `model_runs` row lookup by `zarr_store_path` → `run_id`, `is_same_cycle` | cli.py:1198–1218 |
| Init under EXCLUSIVE gate | `RunCoordinator.initialize_run_store` — existing store: identity validate + `set_run_partial`; absent store: `guard_full_overwrite` + `prepare_run_store` + `marker_v1` sidecar | core/coordinator.py:296 |
| Wave pre-update | `pre_update_wave` — admission + EXCLUSIVE gate; run → `partial`; rolling UPDATING marker PUTs (generation per region) | coordinator.py:378 |
| Region workers | `write_region_worker` — SHARED gate → physical-conflict region locks → generation-ownership check → `_commit_region` (sharded_v1 shard PUTs) → COMPLETE marker (write-set fingerprint, required/omitted keys) | coordinator.py:486; core/pipeline.py:_commit_region; core/zarr_writer.py:commit_region/encode_region_sharded_v1 |
| Settled-lead publication | after all members of a lead settle: `publish_settled_lead` — EXCLUSIVE gate; upsert `ensemble_members` / `ensemble_member_products` / `forecast_products`; advance manifest generation | coordinator.py:872 |
| Coalesced finalization | `finalize_run` — EXCLUSIVE gate; list+validate ALL region markers (O(regions), no physical inventory scan); `_reconcile_catalog_to_store`; `_derive_run_status`; write committed manifest | coordinator.py:662; core/catalog.py:455,925 |

Concurrency protocol (frozen, Checkpoint 2H): admission turnstile → store gate (SHARED writers/readers, EXCLUSIVE init/finalize) → sorted region locks. Keys derived in `packages/domain/src/domain/locks.py` (`store_gate_key`, `admission_key`, `region_key`) from `canonical_storage_identity(store_path)`. Advisory-lock lifecycle in `core/locks.py::StoreLockCoordinator`. The API reader gate participates with the **same** keys (`services/api/src/api/core/reader_gate.py`).

Data commit order: `UPDATING marker → region data objects → COMPLETE marker → manifest → catalog commit` (coordinator.py module docstring).

Library path `pipeline.ingest_grib_file` exists but **refuses live stores** (`LiveStoreOverwriteError`); the coordinator path is the only live-store writer. A Phase 5 scheduler must use the coordinator path.

### B.2 Serving tier (unchanged by Phase 5)

- `services/api/src/api/core/reader_gate.py` — SHARED gate + revalidation; **valid statuses for reading: `ready`, `processing`, `partial`** (`_ReaderGateSession.revalidate`, reader_gate.py:196). Only `failed` is rejected.
- `services/api/src/api/core/zarr.py::open_serving_dataset` — manifest generation (`__commit__/v1/manifest.json` → `generation`) keys the `StoreHandleCache`; a new generation → fresh open → newly committed leads/shards visible immediately.
- `services/api/src/api/services/availability.py` — builds availability from `forecast_products` + `ensemble_member_products` for runs in `SERVING_ELIGIBLE_STATUSES = {ready, processing, partial}`; per-lead servability via `domain.coverage.is_lead_servable` with `MODEL_EXPECTED_MEMBERS = {gfs: 1, gefs: 30}` and `ENSEMBLE_MIN_COVERAGE_RATIO = 0.85`.
- Phase 3 tests pin this behavior: `services/api/tests/test_phase3_degraded_progressive_serving.py` (case E: in-flight horizon availability; case C: exact 85% threshold), `test_ready_run_fallback.py`.

### B.3 Upstream abstraction

- `providers/noaa/connector.py::NOAAConnector` — deterministic URL/key construction:
  - GFS: `gfs.YYYYMMDD/CC/atmos/gfs.tCCz.pgrb2.0p25.fXXX` on `AWS_GFS_BASE_URL` (default `https://noaa-gfs-bdp-pds.s3.amazonaws.com`)
  - GEFS: `gefs.YYYYMMDD/CC/atmos/pgrb2sp25/gepNN.tCCz.pgrb2s.0p25.fXXX` on `noaa-gefs-pds.s3.amazonaws.com`
  - Members: **`gep01..gep30` only**; `gec00`/`geavg`/`gespr` explicitly out of scope (connector.py:71–73, `_GEFS_MEMBER_MIN/MAX`).
  - Download = GET `{url}.idx` → parse → `select_records` → merged byte-range GETs with strict GRIB magic/length/terminator validation; fallbacks: full-file on same provider, then NOMADS on 404/5xx (`ENABLE_NOMADS_FALLBACK`, default on; `NOAA_DOWNLOAD_SOURCE` default `aws_s3`).
- There is **no** discovery/listing/inventory code today — the connector assumes requested artifacts are known targets. A 404 during download is the only "not yet published" signal, handled as `DownloadFailedError` → per-file failure (the wave stays partial; other items continue).

### B.4 Model / lead / member representation

- `RunSpec` (cli.py:171): `model, cycle_date, cycle_hour, lead_time_hours: tuple[int,...], members: tuple[int,...]`.
- Region identity: `domain.locks.logical_region_encoding` → `det_L0006` / `mem017_L0006`.
- Catalog: `ModelRunRecord` (one per `UNIQUE(model_version_id, cycle_time)`), `ProductRecord` (per run×variable×grid×type×**lead**), `EnsembleMemberProductRecord` (per run×**member×lead** pair), `EnsembleMemberRecord` (core/catalog.py:237–332).
- Storage: sharded_v1 shard object per `(variable, member?, lead)`: `{var}/shard.mem017_L0006.shard` / `{var}/shard.det_L0006.shard` (`zarr_writer.encode_region_sharded_v1`). New stores are strict `marker_v1` (`STORAGE_FORMAT_VERSION=sharded_v1` default).

---

## C. Current durable completion model

**[FACT]** Durable truth is spread over three coordinated stores; each answers a different question:

1. **PostgreSQL catalog** (`core/catalog.py`, `services/api/alembic/versions/*`):
   - `model_runs.status ∈ {processing, partial, ready}` — cycle-level lifecycle. `partial` is set by every wave pre-update (`set_run_partial`) and re-derived by every finalizer.
   - `forecast_products` — **lead-level completion** per (run, variable, grid, product_type, lead). This is what availability/serving query.
   - `ensemble_member_products` — **(member, lead)-pair completion**. This is what per-lead member counts (availability) and the exact-pairs readiness gate use.
   - `ensemble_members` — member presence.
2. **Store markers** (`core/markers.py`): `__commit__/v1/regions/{det_L0006|mem017_L0006}.json` — per-region `updating|complete` with generation + write-set evidence. The finalizer's source of truth for "what is committed in the store."
3. **Committed manifest** (`__commit__/v1/manifest.json`): serving generation + state fingerprints; the API's cache/visibility identity.

**Can the committed frontier be reconstructed from durable state? Yes.**

- From the catalog: `SELECT lead_time_hours FROM forecast_products WHERE run_id=…` (deterministic) or the pair table (ensemble) — authoritative for serving.
- From the store: `list_region_marker_keys` + marker validation (`finalize_run` Phase 1–2 does exactly this in O(regions)).
- `pipeline.read_committed_state` can additionally derive committed leads/pairs from non-NaN data regions, but is not needed by the scheduler (markers + catalog are sufficient and cheaper).

**[FACT]** There is no scheduler-owned state today and none is required: `pending = upstream_available − durably_committed` is computable from (upstream listing) − (catalog query). The `__commit__` sidecars and catalog are reconciled to each other by every finalizer run.

---

## D. Incremental-ingestion compatibility verdict

**Verdict: Safe With Narrow Changes.**

### Already safe (**[FACT]**, evidence)

| Concern | Evidence |
|---|---|
| Repeated same-cycle subset runs | `test_coordinator_coldstart.py::test_same_cycle_reingest_with_existing_store_still_works`, `::test_gefs_same_cycle_reingest_existing_store_still_works`; run lookup by `zarr_store_path` → `run_id`/`is_same_cycle` (cli.py:1198) |
| Region writes are independent | `commit_region` writes only the `(lead[, member])` shard; other regions never read/rewritten (zarr_writer.py:378); disjoint cross-process writes proven in `test_cross_process.py::test_cross_process_disjoint_deterministic_writes`; conflicting ensemble members serialize via shared physical chunk keys |
| Catalog PATCH both directions | `_reconcile_catalog_to_store` (catalog.py:455): deletes stale rows, restores missing rows; deterministic row IDs → idempotent re-runs |
| Per-lead catalog publication | `publish_settled_lead` upserts with `on_conflict_do_nothing` / `_get_or_create` (coordinator.py:947–1071); tested in `test_settled_lead_publication.py` |
| Predecessor de-accumulation across runs | 6 h-reset leads read the committed lead−3 slice from the store when not in-wave (`pipeline.read_predecessor_precipitation` / `read_predecessor_cloud_cover`) |
| Race assumptions | Single global order (admission → gate → region locks); unlock-verify-or-invalidate connection discipline (core/locks.py); catalog upsert retry on `uq_model_run_cycle` (catalog.py:1104) |
| Serving visibility of incremental commits | Reader gate accepts `partial`/`processing`; availability is product-row-driven; new manifest generation flushes the store-handle cache (B.2) |
| Finalizer repeated execution | `finalize_run` is idempotent: derives status from marker evidence + reconciled catalog; fingerprints suppress no-op manifest rewrites (coordinator.py:812–819) |

### Narrow changes required (**[FACT]** gaps + **[DESIGN]** fix)

1. **Wave targets vs cycle horizon split.** `RunSpec.lead_time_hours` drives both the wave's work items and `RunCatalogSpec.expected_lead_time_hours` (`cli.py::_build_spec:1860`). Consequences (verified by code reading):
   - Store axis pre-allocation = first invocation's leads (`prepare_run_store` ← `initialize_run_store(expected_leads=…)`). Later waves with leads outside the axis fail: `coordinator.py:517` `StoreSchemaMismatchError("lead N not found in store coordinate")`; library path `zarr_writer._coordinate_index` raises the equivalent `ValueError`. **This is a pre-existing big-batch limitation** (two disjoint-lead CLI invocations on one cycle already fail the same way); realtime merely makes it chronic.
   - Ensemble readiness exact-pairs equality (`catalog.py:1004` `committed_pairs != expected_pairs → partial`) means a wave finalized with only its own leads can never yield `ready`. Deterministic runs use a subset check (`expected ⊆ committed`) and do converge.
   - **[DESIGN]** Fix in Phase 5B: `RunCatalogSpec` gains the cycle's canonical horizon (`expected_lead_time_hours` = full model horizon; `expected_members` = full 1..30), while `RunSpec.lead_time_hours` stays the wave's targets. `initialize_run_store`/`finalize_run` receive the horizon; work-item generation keeps the wave targets. Big-batch behavior is unchanged when a batch's leads == horizon (today's default usage), and disjoint big-batch re-runs are *also* fixed.
2. **[DESIGN]** Full-horizon pre-allocation: first wave initializes the store with the canonical 81-lead axis (×30 members for GEFS) so every later wave lands inside the pre-allocated axis. No store resize logic is needed anywhere.
3. **[DESIGN]** No other structural change: no new marker states, no manifest schema change, no new catalog tables, no sharded_v1 layout change.

### Hidden whole-cycle assumptions inventory (**[FACT]**)

- `expected ⊆ committed` / exact-pairs checks assume the *spec* carries the whole expectation — fixed by (1).
- `publish_settled_lead`'s ensemble fallback `members_to_check = expected_members or range(1,31)` is already horizon-shaped.
- Staging-dir naming includes a uuid — repeated runs never collide.
- `_cleanup_sources` deletes only this wave's generation-proven-committed files — safe for partial waves.
- One `model_runs` row per (model_version, cycle) is an assumption Phase 5 keeps; wave state is not persisted anywhere (see H).

---

## E. Upstream discovery findings

### E.1 Artifact naming / discovery (**[FACT]** repo + **[FACT]** live probe 2026-09-02)

| | GFS 0.25° | GEFS 0.25° (pgrb2sp25) |
|---|---|---|
| Bucket | `noaa-gfs-bdp-pds.s3.amazonaws.com` | `noaa-gefs-pds.s3.amazonaws.com` |
| Key | `gfs.YYYYMMDD/CC/atmos/gfs.tCCz.pgrb2.0p25.fXXX` | `gefs.YYYYMMDD/CC/atmos/pgrb2sp25/gepNN.tCCz.pgrb2s.0p25.fXXX` |
| Sidecar | same key + `.idx` | same key + `.idx` |
| Upstream lead set (probed, 20260901/00) | **209 files**: f000–f120 **hourly** + f123–f384 @ 3 h | **81 files per member**: f000–f240 @ 3 h |
| Platform lead sequence (contract) | **81 leads: 0,3,…,240** (`docs/investigations/gfs-gefs-variable-inventory/README.md`: "0–240h forecast horizon at 3-hour cadence") | same: 81 leads @ 3 h |
| Members | deterministic (1) | `gep01..gep30` (30); `gec00`/`geavg`/`gespr` excluded by contract |

The CLI's `0–384` lead bound is a connector validation ceiling (`connector._MAX_LEAD_TIME_HOURS`), **not** the platform product horizon. The platform contract horizon is 0–240 h @ 3 h for both models. **[DESIGN]** The scheduler must use an explicit configured lead sequence (default the 81-lead contract sequence); it must not assume uniform 3 h spacing from upstream reality (GFS upstream is hourly to 120 h) nor the 384 h connector ceiling.

### E.2 How availability can be observed — cheapest authoritative method (**[FACT]**, live-verified)

Anonymous S3 `ListObjectsV2` (HTTPS GET on the bucket root with `list-type=2&prefix=…`) **works unauthenticated** on both buckets and returns per object: `Key`, `Size`, `ETag`, `LastModified`, `ChecksumAlgorithm/Type`.

Probed listing sizes for one cycle (2026-09-01/00):

| Prefix | Keys | LIST pages @1000/page |
|---|---|---|
| `gfs.20260901/00/atmos/gfs.t00z.pgrb2.0p25.f` (data+idx) | 418 (= 209 + 209) | **1** |
| `gefs.20260901/00/atmos/pgrb2sp25/` (all members incl. geavg/gespr, data+idx) | 5346 | **6** |
| `gefs.20260901/00/atmos/pgrb2sp25/gep01.` (one member) | 162 | 1 |

A product-scoped prefix (not cycle-scoped `atmos/`, which also carries `.nc` and other products and exceeds 1000 keys) keeps GFS to exactly one page. Per-lead × per-member presence for the whole cycle is therefore computable from **1 request (GFS) / ≤6 requests (GEFS)** per poll — versus O(2 × leads × members) = 162 (GFS) / 4860 (GEFS) HEAD/GET probes for a naïve per-artifact existence check.

**Answer to §12:** prefix listing is the cheapest reliable discovery mechanism in the current integration; individual HEADs are unnecessary. (The connector's httpx client is reusable for LIST calls, or `fsspec`; both are already dependencies.)

### E.3 Completeness of a single artifact (**[FACT]** live-verified behavior + S3 semantics)

- **S3 object visibility is atomic**: an object appears only after its upload completes (single PUT or completed multipart). Partial intra-file states are not observable on the S3 REST endpoint. Both buckets are synced from NCEP; LastModified values show files landing one-by-one over hours.
- **The `.idx` follows its data object**: probed LastModified deltas — GEFS `f240` data 17:31:21Z → `.idx` 17:31:22Z (+1 s); GFS `f384` data 05:17:17Z → `.idx` 05:17:46Z (+29 s). The `.idx` is generated from the completed GRIB2 file.
- **The `.idx` is independently required**: the selective-download path GETs `{url}.idx` first and treats 404 as "not ingestable this round" (`connector.py:443`).

**[DESIGN]** Per-artifact availability predicate: `available(artifact) = data key listed AND data.idx listed` (both from the same LIST snapshot). This costs nothing extra, is self-consistent with the download path, and provides a strong "file fully published" signal.

**[ASSUMPTION → validate in 5E]** (a) that `.idx`-present implies data-final on **NOMADS** too (the fallback source; historically yes, but not provable from this repo); (b) that bucket-side rollover/retention never removes a cycle's files mid-publication (probed cycles retained ≥ several days; GFS/GEFS Open Data retention is documented as days, so a multi-day outage can permanently lose a cycle — the reconciliation rule "committed ⇒ done" must tolerate upstream absence of already-committed leads); (c) that `LastModified`/`Size` stability across two polls is a sufficient "no longer growing" signal if we ever need one (we do not, given atomic visibility).

### E.4 Publication-in-progress detection (**[FACT]** semantics available from LIST)

Per poll, the discovery snapshot yields per-lead required-artifact counts (e.g. GEFS lead f015: `count_complete_members = 8/30`). Publication activity = **any change in the snapshot**: new keys, per-lead count growth, or (weakest) LastModified advance. The prompt's example (0 → 8 → 22 members at f015 with unchanged complete frontier) is directly computable. **[DESIGN]** Activity event: `snapshot differs from previous snapshot` for the tracked cycle prefixes.

### E.5 Expected polling cost (**[DESIGN]**, from E.2 measurements)

| Design | Remote ops per poll (GFS+GEFS, both tracked cycles) |
|---|---|
| Naïve per-artifact HEAD | ~162 + ~4860 = **~5000** |
| **Prefix LIST (recommended)** | **≤ 7** (1 GFS + 6 GEFS), plus catalog query (local PG) |
| With per-lead `.idx` probes | unnecessary (idx present in LIST) |

At the §9 cadences (10 min active / 2 min publishing), worst case ≈ 30 × 7 = 210 LIST requests/hour per model pair — negligible vs a single GRIB region write.

---

## F. Completeness barrier recommendation

**[FACT]** The current GEFS member/completeness contract, precisely:

- Ingestion set: `gep01..gep30` — 30 perturbation members (`connector.py:37–40,71–73`; CLI default `range(1,31)`; manifest default identical).
- Serving contract: `MODEL_EXPECTED_MEMBERS["gefs"] = 30` (`packages/domain/src/domain/coverage.py:30`); lead servable at ≥ `ENSEMBLE_MIN_COVERAGE_RATIO` (0.85 → ≥ 25.5 ⇒ 26 of 30) for *public availability*; `ready` run status requires the exact 30 × horizon pairs.
- The upstream control member `gec00` exists but is **not** part of the platform contract (explicitly excluded; docs/MODELS.md mentions "1 control + 30 perturbation" as upstream description, but no code ingests or serves gec00). **No silent redefinition: the barrier must require exactly the 30 perturbation members.**

**[DESIGN]** Validate the shared GFS+GEFS barrier for Phase 5 v1, implemented strictly as **scheduler policy**:

```
lead_complete(cycle, lead) =
    GFS:  gfs file + .idx present for lead                                   (1 artifact pair)
  AND
    GEFS: for all m in 1..30: gep{m} file + .idx present for lead             (30 artifact pairs)
```

- Rationale: it matches the product's coupling (ensemble products and deterministic products share the platform surface vocabulary and the same lead sequence; a shared frontier keeps map/point products mutually consistent and is the simplest correct v1).
- Placement: a pure function in the realtime scheduler module taking the two models' discovery snapshots + lead sequence. It must **not** live in `markers.py`/`inventory.py`/discovery primitives. Future independent GFS/GEFS advance = different policy function, zero storage/serving change.
- The asymmetry to document: the *serving* threshold (85%) is more lenient than the *ingestion barrier* (100% of 30 members). That is intentional and already the platform's shape (Phase 3 serves degraded; Phase 5 waves only ship fully complete leads).

---

## G. Frontier algorithm

**[DESIGN]** All sequences are indexed over the configured lead sequence `L = [l₀=0, l₁=3, …, l₈₀=240]` (contract sequence; configurable), **not** assumed uniform spacing.

Per tracked `(model, cycle)` — actually per tracked *cycle pair* (see F), computed from one discovery snapshot + one catalog query:

```
observed_frontier  = max { l ∈ L : ≥1 required artifact for l listed }        (evidence of upstream life)
lead_complete(l)   = barrier(l)                                               (§F; all required GFS+GEFS artifacts listed)
committed_set      = committed (member,lead) pairs / leads from catalog       (durable truth, §C)
committed_frontier = max { k : lead_complete-planning-prefix — i.e. l_k and all l_j (j<k) committed }   (contiguous prefix of L)
pending_complete   = { l ∈ L : committed(l) is false } ∩ { contiguous complete run after committed_frontier }
                   = walk L from the first uncommitted position; include l while lead_complete(l); stop at first incomplete lead
next_blocked_lead  = the first l that stops the walk (and its missing-artifact breakdown)
```

Worked example (prompt §5 semantics): committed {0}, upstream complete {3,6,9…} missing f009 → walk from l=3: 3 ✓, 6 ✓, 9 ✗ → pending {3,6}, blocked at 9. A later-appearing f012/f015 never enters the wave.

Additional eligibility rule (**[FACT]** domain constraint): a lead `l` with `l % 6 == 0, l > 0` requires `l−3` committed in the store for precipitation de-accumulation / cloud reconstruction (`pipeline._normalize_precipitation_increments` → `read_predecessor_precipitation` raises `MissingPredecessorLeadError` on all-NaN slice). The contiguous walk above satisfies this *between* waves, but the planner must additionally treat `lead_complete(l) ∧ ¬committed(l−3)` as **not eligible** (realtime joining mid-publication, or a barrier that skips ahead). Hourly GFS leads (1,2,4,5…) are outside the contract sequence and simply never planned.

`observed_frontier` beyond the blocked lead (e.g. f015 appearing) is retained only as the **activity signal** for polling (§I) and observability — never as wave content.

---

## H. Wave emission architecture

**[DESIGN]** Bounded batching exactly as §6, with restart-safety derived from durable state:

- **Emission rule**: after each poll, compute `pending_complete` (G). Emit a wave when `|pending_complete| ≥ wave_max_leads` **or** `oldest_pending_age ≥ wave_max_wait` (age = now − first-poll-observed-complete timestamp, tracked in scheduler memory per lead). Whichever first. A partially-published next lead is activity evidence only.
- **No durable wave state**: the pending set is recomputed every poll from upstream + catalog (cheap, §E.5). A restart mid-wave re-derives everything: committed regions are reconciled by the finalizer; the next poll re-plans only what is still missing. Scheduler memory (pending-first-seen timestamps, poll phase) is **optimization-only** — exactly the §7 requirement. Optionally persist a best-effort debug checkpoint (log line), never read back for correctness.
- **Wave execution**: reuse `_run_wave` unchanged per wave: targets = pending leads; spec.horizon = canonical horizon. Inside the wave, existing machinery applies: seed retention, UPDATING markers, stage semaphores, per-lead `publish_settled_lead` (visible while the wave runs), one coalesced `finalize_run` (cumulative expectation → correct status).
- **Sizing guidance (not final values)**: GEFS publishes ~81 leads over ~3.5–5.5 h (probed); GFS 81 contract-leads over ~2.5 h (0–240 h portion). `wave_max_leads` in the 4–12 range bounds per-wave EXCLUSIVE-gate windows and marker PUT bursts; `wave_max_wait` in the 10–30 min range bounds lead latency. Choose after one 5E rehearsal run; do not hardcode.
- **Failure containment**: unchanged from the CLI — per-file failures mark the wave partial and non-zero exit; the scheduler treats a failed wave as "targets still pending" (they are, by definition, uncommitted) and retries on the next poll with backoff. Retry-storm protection comes from the poll backoff, not from wave-level state.

---

## I. Polling state machine

**[DESIGN]** One scheduler loop; a small per-tracked-cycle state machine (per cycle, not global, so a stalled GFS doesn't freeze GEFS):

```
states: ACTIVE ──(publication activity)──► PUBLISHING
   ▲                                            │
   │        (no activity ≥ idle_after)          │ (activity)
   └──────────── BACKOFF ◄──────────────────────┘
                  │  30m → 1h (×idle_backoff_max), jittered
                  └── any activity → PUBLISHING (or ACTIVE if frontier advanced)
```

- **ACTIVE**: poll every `active_poll_interval` (default 10 min). Normal state when a cycle is young or mid-life.
- **PUBLISHING**: poll every `publication_poll_interval` (default 2 min) while **publication activity** is observed. Activity = the discovery snapshot changed in any way for the tracked cycle (new keys, per-lead member-count growth, complete-frontier growth, LastModified advance). Complete-frontier growth is *not* required — the prompt's §10 scenario (f015 member count 0→8→22) is activity.
- **BACKOFF**: no activity for `idle_after` (≈ one poll) → 30 min; each further idle poll doubles to `idle_backoff_max` (1 h). Any activity resets to PUBLISHING (frontier growth) or restarts backoff (cosmetic activity), per the events above.
- **Jitter**: ± `poll_jitter_fraction` (default 10%) on every interval, to avoid synchronized polling across instances.
- **Cycle rollover**: when the next cycle hour's prefix begins showing artifacts (probed: publication starts ~3.5 h after cycle time), the scheduler tracks the new cycle; the old cycle remains tracked until its horizon is committed or it ages out (`tracked_cycle_max_age`, guard rail). Full supersession/retention = Phase 6.
- All parameters configurable (§J). Shutdown at any point: the loop checks a cancel event between polls and during wave waits; the wave itself is drained non-abandoningly by the existing machinery (`core/cancel.py::await_all_workers_non_abandoning`).

---

## J. Configuration design

**[FACT]** Configuration conventions: `IngestionSettings` (pydantic-settings, env + `.env`, `core/config.py`), env-style UPPER_SNAKE keys, documented in `.env.example`; API has its own `api/core/config.py`. The scheduler is an ingestion-side concern → all Phase 5 settings belong in **`IngestionSettings`** (+ `.env.example`), no new config system.

**[DESIGN]** Proposed keys (names follow existing conventions; values = defaults for discussion, not final):

```python
# --- Realtime lead-wave ingestion (Phase 5) ---
REALTIME_ENABLED: bool = False                       # opt-in master switch
REALTIME_MODELS: str = "gfs,gefs"                    # tracked models
REALTIME_LEAD_SEQUENCE_HOURS: str = "0:3:240"        # canonical sequence: start:step:end → 81 leads
REALTIME_WAVE_MAX_LEADS: int = 8                     # §H
REALTIME_WAVE_MAX_WAIT_SECONDS: float = 1200.0       # §H
REALTIME_ACTIVE_POLL_SECONDS: float = 600.0          # §I 10 min
REALTIME_PUBLICATION_POLL_SECONDS: float = 120.0     # §I 2 min
REALTIME_IDLE_BACKOFF_INITIAL_SECONDS: float = 1800.0
REALTIME_IDLE_BACKOFF_MAX_SECONDS: float = 3600.0
REALTIME_POLL_JITTER_FRACTION: float = 0.10
REALTIME_SHARED_BARRIER: bool = True                 # §F policy switch (GFS+GEFS AND)
REALTIME_LEADER_LOCK_SECONDS: float = 600.0          # heartbeat renewal; double-start guard (§M)
```

Notes: `REALTIME_LEAD_SEQUENCE_HOURS` deliberately replaces the CLI's implicit "requested == expected" identity; per-model override can be added later only if the contract actually diverges (GFS and GEFS share the 81-lead contract today). `ENSEMBLE_MIN_COVERAGE_RATIO` already exists and is untouched.

Minor hygiene item found (not a blocker): `.env.example` does not yet list `NOAA_DOWNLOAD_SOURCE` / `AWS_GFS_BASE_URL` / `AWS_GEFS_BASE_URL` / `ENABLE_NOMADS_FALLBACK` / `ENABLE_SELECTIVE_DOWNLOAD`, although `core/config.py` supports them — fold into the Phase 5 config commit.

---

## K. Reuse vs new code

**Existing components to reuse unchanged**
- `NOAAConnector` (download, `download_idx`, URL/key building, retries, fallbacks) — providers/noaa/connector.py
- `DecodePool`, decode/normalize pipeline (`parse_grib2`, `_normalize_precipitation_increments`, `_normalize_cloud_cover_intervals`, `_apply_variable_mapping`, `_normalize_canonical_units`) — providers/noaa/parser.py, core/pipeline.py, core/decode_worker.py
- `RunCoordinator`: `initialize_run_store`, `pre_update_wave`, `write_region_worker`, `publish_settled_lead`, `finalize_run` — core/coordinator.py
- Advisory-lock stack — domain/locks.py, core/locks.py; marker/manifest protocol — core/markers.py; inventory evidence — core/inventory.py
- Catalog write/reconcile — core/catalog.py; zarr writer — core/zarr_writer.py; S3 fs lifecycle — core/s3.py; cancel drain — core/cancel.py; observability primitives — core/observability.py
- Entire serving tier (Phase 3 behavior) — no changes at all.

**Existing components requiring narrow extension**
- `RunSpec` / `RunCatalogSpec` / `_build_spec` / `_run_wave` / `_ingest_one_run` (cli.py): accept explicit wave targets vs cycle horizon (expected leads/members) instead of deriving both from one tuple. Keep the CLI flags' current meaning; add programmatic parameters. Files: services/ingestion/src/ingestion/cli.py, core/catalog.py (dataclass only).
- `initialize_run_store`/`prepare_run_store` call chain: pass the horizon for pre-allocation (already parameterized; only the caller's argument changes).
- Possibly extract `_run_wave` from cli.py into an importable runner module so the scheduler doesn't import a private function (mechanical move, no logic change).

**New Phase 5 components required**
- `providers/noaa/discovery.py` (new): S3-prefix inventory snapshotter (ListObjectsV2, paginated, product-scoped prefixes, anonymous HTTPS), returning per-cycle artifact presence + metadata; snapshot-diff helpers (activity detection). ~150–250 lines.
- `realtime/planner.py` (new): lead-sequence model, frontier walk (§G), barrier policy (§F), eligibility (predecessor rule), bounded-batch emission (§H). Pure functions → domain-testable offline.
- `realtime/scheduler.py` (new): poll loop + state machine (§I), cycle tracking, wave dispatch (calls the wave runner), leader lock (PG advisory key reuse), graceful shutdown, scheduler-level logging (structured lines: phase, intervals, frontiers, wave results).
- CLI subcommand `weather-ingest realtime` (cli.py): long-running foreground process; `--once` dry/planning mode for ops. Same Docker image (`docker/Dockerfile.ingestion`); deployed as a docker-compose service (deployment wiring can follow the existing compose patterns; a compose entry is config, not architecture).
- Tests: discovery (mocked XML + optional live-gated), planner (pure), scheduler (fake clock + fake discovery + real coordinator against SQLite/MinIO-style fixtures), failure modes (§15 matrix).

**Explicitly not built** (avoid speculative abstraction): no generic multi-provider scheduler framework, no Celery/queue introduction, no new metrics stack, no new config system, no second ingestion engine, no store-format work.

---

## L. Phase 5 / Phase 6 boundary

**[DESIGN]**
Phase 5 implements **only**:
- Cycle **selection**: for each model, the tracked cycle = latest `(cycle_date, cycle_hour)` whose product prefix shows artifacts, subject to `tracked_cycle_max_age`. At most one active tracked cycle per model (the pair GFS+GEFS per §F). Handover: when a newer cycle's artifacts appear, start tracking it; keep finishing the older cycle's remaining horizon under the existing backoff (both may transiently be tracked; each still one `model_runs` row).
- Everything in §E–K.

Phase 6 retains (explicitly out of scope now):
- Supersession policy (what happens to the old cycle's store/rows when the new one takes over), old-cycle retirement, retention windows, deletion/GC, crash-safe GC, multi-cycle lifecycle management, serving-side cross-cycle stitching changes (Phase 3 behavior untouched).
- Any change to `model_runs` uniqueness or store path layout (`s3://weather-data/{model}/{date}/{hh}/cycle.zarr` stays).

Boundary risk flagged: the old cycle keeps ingesting while the new cycle is also ingesting — **[FACT]** the advisory-lock protocol is per-store, so concurrent different-cycle waves do not contend (different lock keys); catalog cost is two runs' worth of rows; availability lists both (already supported, newest-first ordering exists). This is safe to leave as-is until Phase 6 decides supersession.

---

## M. Risk register (severity × likelihood, with mitigations)

| # | Risk | Sev | Lik | Evidence | Mitigation |
|---|---|---|---|---|---|
| 1 | **Wave-vs-horizon coupling**: store axis pre-allocated from first wave's leads → later waves hard-fail; ensemble status never `ready` under per-wave finalize | High | High (certain without the split) | coordinator.py:517; zarr_writer.py:441/791; catalog.py:1004; cli.py:1860 | 5B split + full-horizon pre-allocation (§D); regression tests for repeated disjoint-lead runs (GFS+GEFS) |
| 2 | **Upstream completeness misjudgment** — ingesting a partially published file | High | Low (S3 atomicity verified; NOMADS path less provable) | §E.3 probes; connector fallback path | `data + .idx` predicate; prefer `aws_s3` for realtime; NOMADS fallback stays opportunistic; 5E live validation incl. `.idx` timing; per-file lead validation already fails fast on mislabeled files (`LeadTimeMismatchError`) |
| 3 | **Repeated finalization cost / gate exclusivity**: each wave's `finalize_run` reads O(all committed markers) (GEFS full horizon = 2430 markers) and holds the EXCLUSIVE gate (serving SHARED reads pause) | Med | High | coordinator.py:662–867; benchmark test proves O(regions) design at 1110 regions | Bounded waves (§H) bound wave count; finalizer already bounded and fingerprint-suppressed; monitor finalize duration (existing tracker milestones); revisit only with measured regression (Phase 2 invariant) |
| 4 | **Big-batch and realtime on the same cycle** | Med | Med | Both use admission+EXCLUSIVE gate → serialized, correct (locks.py); pre-update sets `partial` and finalizer reconciles both directions | Documented conflict policy: realtime defers (leader lock + skip while a big-batch holds the store); correctness is already guaranteed — only wasted work is possible |
| 5 | **Two realtime processes** (double start) | Med | Med | Admission turnstile serializes region writes, so no corruption; but duplicated waves/plans waste downloads | PG-advisory "realtime leader" heartbeat key (reuse `domain.locks` derivation); second instance idles/logs |
| 6 | **Predecessor lead gap** (lead % 6 == 0 without committed lead−3) → `MissingPredecessorLeadError` | Med | Med (realtime joining mid-publication) | pipeline.py:529/693 raises on all-NaN predecessor | Planner eligibility rule (§G); test first-wave-at-late-join scenario |
| 7 | **Excessive polling operations** | Low | Low (if LIST-based) | §E.5: ≤7 requests/poll vs ~5000 naïve | Product-scoped prefixes (not `atmos/`); never per-artifact HEAD loops; jitter |
| 8 | **Upstream retention/rollover**: tracked cycle deleted mid-flight after multi-day outage | Low | Low | Open Data buckets retain days, probed | Reconciliation rule: committed ⇒ done; upstream-absent ⇒ never pending; cycle age-out guard |
| 9 | **NOMADS ban / rate limiting** from aggressive polling | Low | Low (default source is aws_s3) | config.py:20 comment; §E.5 costs | Poll only LIST on AWS; downloads remain on-demand (only for wave targets); jitter |
| 10 | **Serving blip during EXCLUSIVE gate** per wave | Low | Med | reader_gate waits (bounded by `API_READER_GATE_TIMEOUT_SECONDS=30`) | Keep waves reasonably large; publish_settled_lead already spreads visibility; monitor gate timeouts |
| 11 | **Config drift** (.env.example missing upstream keys) | Low | — | §J | Fold into 5B config commit |

---

## N. Recommended Phase 5 implementation decomposition

### 5B — Upstream discovery + availability model
- **Scope**: `providers/noaa/discovery.py` (LIST-based cycle inventory, product-scoped prefixes, pagination, snapshot diff/activity), discovery settings, unit tests with recorded XML fixtures + a live-gated optional test; **plus** the 5B structural split from §D (RunSpec/RunCatalogSpec wave-vs-horizon, full-horizon pre-allocation, `_run_wave` extraction) — doing the split here de-risks everything after it and is independently shippable for big-batch.
- **Files**: connector package (new discovery module), cli.py, core/catalog.py (spec dataclasses), core/config.py, .env.example.
- **Invariants**: no behavior change for existing big-batch invocations whose leads == requested set, except stores now pre-allocate the full horizon (documented); discovery performs zero writes; ≤ (1 GFS + 6 GEFS) LISTs per snapshot.
- **Tests**: planner-grade fixture tests for snapshot parsing; repeated disjoint-lead same-cycle ingestion converges to `ready` (GFS deterministic + GEFS ensemble) — the regression test for risk #1; big-batch golden tests unchanged.
- **Acceptance**: two back-to-back CLI runs on one cycle with disjoint lead subsets: second run succeeds, run reaches `ready`, availability shows the union; discovery snapshot of a live cycle matches a hand-counted listing.
- **Non-goals**: no scheduler loop, no polling, no barrier logic.

### 5C — Frontier reconciliation + lead-wave scheduler
- **Scope**: `realtime/planner.py` (lead sequence, barrier, frontier walk, predecessor eligibility, bounded batching), `realtime/scheduler.py` (poll state machine, cycle tracking, wave dispatch via the extracted wave runner, leader lock, shutdown handling), `weather-ingest realtime` subcommand (`--once` for planning dry-run).
- **Invariants**: wave contains only barrier-complete leads; correctness never depends on scheduler state (reconciliation from catalog+upstream each poll); big-batch path untouched; EXCLUSIVE gate held only within existing coordinator phases.
- **Tests**: pure planner tests (prompt §3/§5 scenarios, hourly-cadence confusion cases, late-join predecessor case), scheduler with fake clock/discovery (state transitions §I, restart mid-wave, double-start, wave emission rules), lock-contention test with a concurrent big-batch writer.
- **Acceptance**: in a containerized local MinIO+PG environment, a simulated multi-hour upstream publication (scripted fixture bucket) produces ordered waves, per-lead serving visibility during waves, and `ready` at horizon completion; restart at any point resumes with no duplicate committed work.
- **Non-goals**: no real-upstream dependency in CI; no Phase 6 lifecycle.

### 5D — Failure-mode & concurrency hardening
- **Scope**: the §15 matrix as explicit tests + fixes that fall out: upstream disappearance mid-wave, member delay (GEFS 29/30), catalog failure after writes (finalizer retry semantics), shutdown during polling vs during ingestion, polling failure → backoff, big-batch/realtime overlap.
- **Acceptance**: each §15 row has an automated test or a documented manual procedure + rationale; no unhandled failure class.
- **Non-goals**: GC/cleanup of stale cycles (Phase 6).

### 5E — Real-upstream E2E acceptance
- **Scope**: run the scheduler against live AWS buckets for at least one full GFS+GEFS cycle publication window; validate §E.3 assumptions (`.idx` timing, atomic visibility, listing stability), wave cadence, and serving behavior end-to-end; record measured `wave_max_leads`/`wave_max_wait` guidance and finalize defaults; runbook entry (ops: start/stop/logs/what-to-check).
- **Acceptance**: one realtime cycle ingested end-to-end unattended; availability/serving progression recorded; assumptions list in this report each marked validated or amended.
- **Non-goals**: multi-day soak (can follow), Phase 6.

---

## O. Concrete implementation recommendation

Implement next (5B), in one auditable change-set:

1. **Spec split** — `RunCatalogSpec` gains authoritative cycle-horizon fields (`expected_lead_time_hours` = canonical horizon, `expected_members` = full member set); `RunSpec` keeps wave targets. `_build_spec` sets horizon from the new `REALTIME_LEAD_SEQUENCE_HOURS`-derived constant (default 81-lead contract sequence); `_ingest_one_run`/`_run_wave` pass targets to item generation and horizon to `initialize_run_store`/`finalize_run`. CLI behavior is unchanged for full-horizon batches.
2. **Full-horizon pre-allocation** — `initialize_run_store` always receives the horizon; `prepare_run_store` therefore creates the 81-lead (×30-member) axis on first wave; every later wave lands inside the axis. Fixes the pre-existing disjoint-lead big-batch failure as a side effect.
3. **Discovery module** — `providers/noaa/discovery.py`: anonymous ListObjectsV2 over product-scoped prefixes (GFS: `gfs.{date}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f`, GEFS: `gefs.{date}/{hh}/atmos/pgrb2sp25/gep{NN}.` + `gep{NN}.t{hh}z.pgrb2s.0p25.f` — the member-scoped prefix keeps each GEFS member's listing to a single page, 30 pages total if per-member granularity is preferred over 6 shared pages; either is cheap), returning an immutable `CycleSnapshot` (per (member?, lead) → {data: bool, idx: bool, size, last_modified}).
4. **Runner extraction** — move `_run_wave` (+ helpers) into `ingestion/core/wave_runner.py` with an explicit `WaveTargets`/`RunContext` parameter object; the CLI delegates to it. Purely mechanical.

Then (5C) the scheduler: `realtime/planner.py` (pure: `plan_wave(snapshot_gfs, snapshot_gefs, committed, horizon, wave_cfg) → Wave | None + diagnostics`), `realtime/scheduler.py` (state machine §I, leader lock, dispatch, structured logging), CLI `realtime` subcommand. **No changes** to sharded_v1, markers, locks, catalog schema, or any serving code.

Everything above reuses the Phase 1–4 invariants as-is; the only new correctness logic is the planner's frontier walk and the barrier policy, both pure functions with offline tests.

---

## Appendix: verified-fact index

| Fact | Source |
|---|---|
| Store path template; one store per cycle | cli.py:93 (`STORE_PATH_TEMPLATE`) |
| Wave protocol order; commit order | coordinator.py docstring; cli.py `_run_wave` |
| Lead-axis pre-allocation from run's leads | zarr_writer.py:257 `prepare_run_store`; cli.py:1860 `_build_spec` |
| Commit fails for leads outside axis | coordinator.py:517; zarr_writer.py:791 `_coordinate_index` |
| Ensemble ready requires exact pair-set equality vs invoking spec | catalog.py:1000–1005 |
| Deterministic ready uses subset check | catalog.py:982 |
| `partial`/`processing` are servable; `failed` is not | reader_gate.py:196; availability.py:52 |
| Per-lead servability = ≥85% of 30 members | domain/coverage.py:30,136; availability.py:266–271 |
| Manifest generation flushes serving cache | api/core/zarr.py:453 `open_serving_dataset`; manifest_reader.py |
| Finalizer O(regions), evidence-based, idempotent | coordinator.py:662; test_finalization_latency_optimization.py; test_finalization_benchmark.py |
| Settled-lead publication exists and is idempotent | coordinator.py:872; test_settled_lead_publication.py |
| GEFS = 30 perturbation members only; gec00 out of scope | connector.py:37–40,71–73; domain/coverage.py:30 |
| Predecessor (lead−3) store fallback exists | pipeline.py:366,546 |
| Anonymous ListObjectsV2 works; page counts; lead cadences; .idx timing | live probes 2026-09-02 (§E.1–E.3) |
| Platform contract horizon = 0–240 h @ 3 h (81 leads) | docs/investigations/gfs-gefs-variable-inventory/README.md |
| Upstream GFS publishes hourly→120 h, 3-hourly→384 h (209 files) | live probe 2026-09-02 |
| Upstream GEFS 0.25° pgrb2s = f000–f240 @ 3 h per member | live probe 2026-09-02 |
