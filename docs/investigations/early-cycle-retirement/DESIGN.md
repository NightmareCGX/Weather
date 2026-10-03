# Early Cycle Retirement + Reclamation Ledger ("terminal-state retirement") — Design

**Status:** Implemented (branch `feat/early-retirement-and-reclamation-ledger`)
**Date:** 2026-10-02 (combined plan approved same day)
**Scope:** `services/ingestion` (finalizer, planner, worker, sweeper, config), `services/api` (fence read paths), one Alembic migration.
**Decision record:** Permanent retirement moves from `cycle_time + 240h` to *the moment the cycle is fully and physically reclaimed* (~T0+14h). Upstream late corrections arriving after that moment are refused. Accepted by the owner (2026-10-02), on the following rationale: at a 6 h cycle cadence, a cycle's serving relevance ends when the next cycle is ready (~T0+6h; the interval T−2C fallback window closes ~T0+12h). A correction applied to an older cycle is therefore **never selected by serving** — every valid_time it covers has a newer winner — so the legacy 10-day acceptance window stored data that serving could no longer read. The same holds for manual re-ingest after a hypothetical wrongful deletion: the only units worth restoring are ones still canonically held, and held units keep the cycle non-terminal (no tombstone) regardless of the gate. The window's residual value was a low-probability stacked-bug last resort; its cost (1.9 GB queue, 2g GC, doubled serving latency) was certain and continuous.

**Combined plan (owner decision, 2026-10-02):** implement **both** parts of this design in the same release — Part A (early retirement) and Part B (the aggregated reclamation ledger). They compose on orthogonal axes: Part A shortens the lifetime of reclamation bookkeeping rows (11 days → ~2 days), Part B reduces the number of rows a reclaimed unit occupies (queue row → aggregated bitmap row) and turns `reclamation_queue` into a pure in-flight work list. Together they take the bookkeeping footprint from a projected 1.9 GB steady state to **~10–20 MB**.
**Decision record:** Permanent retirement moves from `cycle_time + 240h` to *the moment the cycle is fully and physically reclaimed* (~T0+14h). Upstream late corrections arriving after that moment are refused. Accepted by the owner (2026-10-02), on the following rationale: at a 6 h cycle cadence, a cycle's serving relevance ends when the next cycle is ready (~T0+6h; the interval T−2C fallback window closes ~T0+12h). A correction applied to an older cycle is therefore **never selected by serving** — every valid_time it covers has a newer winner — so the legacy 10-day acceptance window stored data that serving could no longer read. The same holds for manual re-ingest after a hypothetical wrongful deletion: the only units worth restoring are ones still canonically held, and held units keep the cycle non-terminal (no tombstone) regardless of the gate. The window's residual value was a low-probability stacked-bug last resort; its cost (1.9 GB queue, 2g GC, doubled serving latency) was certain and continuous.

---

## 1. Problem

`reclamation_queue` rows and catalog rows (`model_runs`, `forecast_products`, `ensemble_member_products`) currently live **~11 days**. Decomposition of the timeline for a cycle starting at T0:

| Event | When | Why |
|---|---|---|
| Physical deletion of the cycle's units | **~T0+14h** | Progressive: each unit becomes reclaimable when the newest cycle covers its valid_time (canonical is the sole deletion authority); the interval variables' T−2C fallback window closes when the next cycle is ready (~T0+12h); the worker drains ~35k units in ~70 min at current throughput |
| Claim (`deletion_started_at`) | T0+240h | `claim_fresh_candidate` requires `is_cycle_horizon_expired` — `cycle_time + 240h < serving_start` |
| Tombstone (`deleted_at`) | T0+240h+ε | Claim + all committed units terminal |
| Row & catalog deletion | T0+241h | Metadata sweeper, `deleted_at` + `METADATA_RETENTION_DAYS=1` |

The 240h claim gate delays rows **and** catalog by ~9.5 days during which nothing physical is being waited on: the shards are already gone, serving is already fenced per-unit, and canonical already stopped selecting the cycle. Under the legacy gate the post-2026-10-05 steady state would be **~1.6M queue rows / ~1.9 GB**, an 88-run catalog, and `/v1/forecast/availability` p95 degraded to ~15–20s.

**Fix:** let a cycle be claimed and tombstoned as soon as it is *fully and physically reclaimed*, instead of waiting for horizon expiry.

---

## 2. The change

Add an **early-retirement eligibility path** to the finalizer. A cycle may be claimed (and in the same pass tombstoned) when **all** of the following hold:

- **E1.** `LIFECYCLE_EARLY_RETIREMENT_ENABLED=true` (staging flag, default `false` — mirrors the planner/delete/sweeper flag pattern).
- **E2.** Every `model_runs` row of the `(model_id, cycle_time)` has `status='ready'` — no `partial`/`processing` run. **Mandatory correctness gate, see §2.2.**
- **E3.** `cycle_reclamation_units_terminal()` is true: every committed unit has a `reclamation_queue` row with `status='deleted'`, and no non-terminal (`queued`/`deleting`/`failed`) rows exist for the cycle.

The **legacy 240h path is preserved unchanged** as the fallthrough for cycles that can never satisfy E2 — permanently-`partial` ("zombie") cycles, e.g. an upstream-discontinued cycle held in quarantine. Evidence-damaged cycles do **not** need it: the early path self-heals them (see E-3), faster than the legacy path ever could.

### 2.1 Exact change points

All in `services/ingestion/src/ingestion/gc/finalizer.py` unless stated:

1. `claim_fresh_candidate` (L176): new keyword parameter `allow_terminal: bool = False`. After the existing `FOR UPDATE` + already-claimed recheck (step 2), when `allow_terminal` is set and the horizon check (step 5) fails, evaluate **E2 then E3** *inside the same locked transaction*; if both hold, stamp `deletion_started_at` and return `True` (log reason `units_terminal`). When the horizon check passes, behavior is byte-identical to today. When both eligibility paths fail, return `False` (blocked) as today.
2. `finalize_cycle_bookkeeping` (L462): forward `allow_terminal=settings.LIFECYCLE_EARLY_RETIREMENT_ENABLED` for fresh candidates. Recovery candidates (already claimed) are untouched.
3. `services/ingestion/src/ingestion/core/config.py`: `LIFECYCLE_EARLY_RETIREMENT_ENABLED: bool = False` with docstring.

**Lock-hold note:** E2 is one indexed query (reuse the existing version/run resolution of step 3); E3 at a ~2-day catalog is a few indexed queries (ms-scale). Holding the lifecycle row `FOR UPDATE` for that duration is acceptable; ingestion's `ensure_lifecycle_row` upsert may briefly wait (~ms). This keeps lock-hold time in the same class as today.

**Crash safety:** claim and tombstone are separate transactions today and remain so. A crash after claim leaves a claimed-not-tombstoned cycle; the next pass picks it up as a *recovery candidate* (existing flow), re-verifies terminality (true), and tombstones. No new recovery machinery.

### 2.2 The trivially-terminal trap (why E2 is mandatory)

`cycle_reclamation_units_terminal` returns **True vacuously** when a cycle has zero committed units — which is the normal state of a cycle in its first minutes (run recorded, waves not yet committed). Without E2, the early path would claim-and-tombstone brand-new cycles, and the mutation fence (`catalog.py:1649` raises `CycleTombstonedError` when `deletion_started_at IS NOT NULL`) would **brick their entire ingestion**.

E2 closes this: `status='ready'` is the promotion gate meaning the full horizon was committed, so the committed-unit enumeration is complete and "all units terminal" is a meaningful statement. The two gates compose to exactly the intended set: *old, fully-ingested, fully-reclaimed cycles* — and nothing else. The newest ready cycle never satisfies E3 (its units are held as canonical sources), so live cycles are unreachable.

---

## 3. Edge cases

| # | Scenario | Handling |
|---|---|---|
| E-1 | In-flight ingestion (run `partial`/`processing`) | E2 false → no early tombstone. Its committed-but-unprotected units may be reclaimed, as today. |
| E-2 | Backfill/recovery mid-flight | The recovering cycle's far-future valid_times are still protected → units held → not terminal → no early tombstone while recovery is in flight. Residual regression vs today: a cycle within *hours* of its 240h edge that recovers slower than its remaining window loses that margin (such a cycle serves nothing anyway). |
| E-3 | Missing queue rows (historical purge damage, e.g. cycle `2026-09-26/18`, 2,410-row gap) | **Self-heals through the early path, no special handling.** Row absence reads as "never enqueued" to the planner → the missing units are re-enqueued on the next pass → the worker's `DeleteObjects` on absent keys is an idempotent success → rows marked `deleted` → E3 becomes true → early tombstone → sweep. Converges in hours. The re-enqueue cannot re-form the churn loop of the 2026-09 incident: that loop persisted only because tombstone was unreachable behind the 240h gate, so the cycle never left the planner scope and its rows were re-purged after a day; here the tombstone lands as soon as the gap is filled, the cycle exits the planner scope, and its catalog + queue rows are swept together at +1 day. One repair pass, then gone. (Requires E2: a damaged-but-`ready` cycle heals; if the damage coincides with a never-`ready` run, the legacy 240h path retires it at expiry as today.) |
| E-4 | `failed` quarantine rows | Non-terminal → blocks early path (and legacy). `reclamation requeue` flow unchanged. |
| E-5 | Upstream late correction / republish **after** early tombstone | Refused by the mutation fence (`catalog.py:1649`) and anti-resurrection (`:901`). **Accepted trade-off** (see decision record): the refused corrections were already invisible to serving — a cycle stops being a serving source when the next cycle is ready (~T0+6h) — so the acceptance window's reduction from 10 days to ~14h removes only dead storage. Corrections landing inside the live fill window (~6h) are unaffected. |
| E-6 | Same-cycle re-ingest racing the early claim (ready→partial downgrade just after E2 was evaluated inside the lock) | The wave's next catalog write hits the mutation fence → `CycleTombstonedError` → wave aborts cleanly; any written shards become store orphans → detected by the scheduled orphan inventory (discovery-only; reap stays manual). Requires a re-ingest of a fully-reclaimed old cycle to coincide with the claim moment; the realtime scheduler only ingests current cycles and backlog candidates exclude claimed/tombstoned cycles. Residual risk: negligible, with a standing detector. |
| E-7 | Zombie cycles (permanently `partial`, never ready) | Never early-retire (E2 false); the legacy 240h path tombstones them at expiry, as today. |
| E-8 | Multiple versions per cycle | E2 requires **every** run ready; conservative. |
| E-9 | Interval T−2C lead0 fallback (R3/R5) | Encoded by existing planner holds: N−1's interval units are held until N+1 is ready; E3 ("all units terminal") cannot fire before that. The early path cannot shorten the fallback window. |
| E-10 | `blocked=N` in GC pass summaries | Drops from ~48 (cycles waiting on 240h) to roughly the partial-cycle count. Expected log change, not a regression — called out so ops doesn't misread it. |

---

## 4. What deliberately does **not** change

- The legacy 240h claim path (kept as repair), the sweeper (`tombstone + 1 day`), the planner (including its retired-cycle fast path — vacuous under the early path but harmless), the worker, the per-unit serving fence, the canonical deletion authority, serving semantics (R1–R12), tombstone permanence, `METADATA_RETENTION_DAYS`.
- No Alembic migration. No new table. No API code.

---

## 5. Rollout

1. Merge + deploy the ingestion image with `LIFECYCLE_EARLY_RETIREMENT_ENABLED` unset → behavior identical to #140 (flag-gated, zero-risk staging).
2. Set the env on `weather_ingestion_gc` and recreate the container.
3. Observation window: one full cycle pair (12h) plus 48h for the first sweeps. Signals:
   - `forecast_cycle_lifecycle` tombstone count grows (first non-zero ever);
   - `reclamation_queue` plateaus at ~2 days of rows (~250–300k rows, ~250–300 MB);
   - `/v1/forecast/availability` p95 trends back toward ~4–5s;
   - GC planner stage duration drops; `blocked=N` shrinks;
   - `weather-ingest audit` shows no new invariant violations.

---

## 6. Rollback

- Set `LIFECYCLE_EARLY_RETIREMENT_ENABLED=false` (or unset) and recreate the GC container — **no image rebuild, no schema step**. Behavior is immediately the legacy path.
- **Permanence note:** cycles tombstoned while the flag was on stay tombstoned (permanence is the design). Rollback does not resurrect them; their catalog rows are swept 1 day after their tombstone regardless of the flag (the sweeper reads `deleted_at`, not the flag). State remains consistent in both directions.
- If a full code rollback is preferred: `git revert` of the single commit; nothing else is entangled.

---

## 7. Verification plan

Tests (extend `services/ingestion/tests/test_gc_finalizer.py`):

- **T1** (core): a cycle with all runs `ready` and all units `deleted` is claimed **and** tombstoned in a single bookkeeping pass, *without* horizon expiry.
- **T2**: a `partial` run blocks the early path even when its committed units are all terminal (E2).
- **T3**: a brand-new cycle (run recorded, zero committed products) is **not** claimed — the trivially-terminal trap (§2.2).
- **T4** (self-heal): a cycle with missing queue rows is re-enqueued by the planner, the worker's idempotent no-op delete marks the gap filled, and the cycle tombstones via the **early** path — without horizon expiry. (The historical purge damage would have self-healed under this design.)
- **T4b**: a damaged cycle whose run is never `ready` does not early-tombstone; the legacy 240h path retires it at expiry.
- **T5**: flag off → legacy behavior identical (claim only at horizon expiry).
- **T6**: crash between claim and tombstone → next pass completes the tombstone via the recovery path (existing pattern, extend).
- **T7**: interval sequencing: N ready → N−1 **not** terminal (interval units held by T−2C) → N+1 ready → N−1 tombstones. Proves E-9. (Implemented as: the flag cannot fire while any committed unit is still held — the holds are the planner's existing coverage.)
- **T8**: a claimed (early-retired) cycle refuses `record_run` — Guarantee B applies to early claims equally (existing fence test, parameterized).

Implementation-time assertions: `finalizer_claim_stuck` alert semantics unchanged (claims now clear in minutes); `weather_lifecycle_*` gauges documented.

---

## 8. Docs to update during implementation

- `docs/DATABASE.md` — lifecycle state machine: claim trigger = horizon expiry **or** terminal-state (flag).
- `docs/lifecycle-v3-architecture.md` — §7/§8 retirement trigger; move the item from outstanding list.
- `docs/RUNBOOKS.md` — claim-age alert note (§19.6): claims now resolve in minutes.
- `docs/MONITORING.md` — `weather_lifecycle_*` gauge semantics under early retirement.

---

## 9. Expected steady state (after the 2-day row lifetime is reached)

| Metric | Legacy trajectory (10-05 self-heal) | With early retirement |
|---|---|---|
| `reclamation_queue` | ~1.6M rows / ~1.9 GB | **~250–300k rows / ~250–300 MB** |
| Catalog in-scope window | 11 days (~88 runs) | **~2 days (~12–16 runs)** |
| `/v1/forecast/availability` p95 | degrades to ~15–20s | **returns toward ~4–5s** |
| GC planner working set | ~1.2–1.5 GB (needs 2g limit) | **~1/3 → `mem_limit: 512m` restorable** (observe first) |
| Schema / fence / serving changes | — | **none / none / none** |

Deferred by this design (revisit only if ~300 MB steady state is later judged unacceptable): the per-cycle bitmap ledger ("方案 I-1", ~40 MB steady state, requires a new table + 5 fence read-path rewrites) and the serving candidate prefilter ("I-2", see the accompanying discussion — largely obsoleted here because the catalog itself shrinks to ~2 days).

---

# Part B — the aggregated reclamation ledger

## B1. Rationale

Under Part A alone, terminal rows remain in `reclamation_queue` until the cycle's tombstone + 1 day sweep (~2 days of rows, ~250–300 MB). Part B splits the queue's two roles:

* **`reclamation_queue` becomes a pure in-flight work list** — only `queued` / `deleting` / `failed` rows. A row is removed in the same transaction that records the unit's reclamation.
* **`reclamation_ledger` is the durable "this unit is gone" record** — consumed by (1) planner idempotency, (2) the serving physical fence, (3) finalizer terminality. Rows cascade away with the cycle's catalog at the tombstone+1d sweep.

Aggregation: one ledger row per `(run_id, lead_time_hours, variable_code, target_kind)` with a `deleted_members_mask BIGINT` for the `mem` kind (bit *m* ⇔ member *m* reclaimed; members 1..30). `det`/`mean` rows carry mask 0 — row existence is the record. Expected size under Part A's ~2-day lifetime: ~7k rows, single-digit MB.

## B2. Schema (migration `011_reclamation_ledger`)

```
reclamation_ledger:
  run_id           String      FK model_runs.id ON DELETE CASCADE, PK part
  lead_time_hours  Integer     PK part
  variable_code    String      PK part
  target_kind      String(16)  PK part   -- det | mean | mem
  deleted_members_mask BigInteger NOT NULL DEFAULT 0
  store_path       String      NOT NULL  -- denormalised for the tiles fence
  reclaimed_at     DateTime(tz) NOT NULL
  created_at / updated_at
  PK (run_id, lead_time_hours, variable_code, target_kind)
  INDEX store_path
```

FK cascade mirrors `reclamation_queue`: when the metadata sweeper deletes a cycle's `model_runs`, its ledger rows cascade away automatically; `purge_cycle_metadata` additionally deletes them explicitly (child-first order) for count logging. **No backfill**: pre-existing `deleted` queue rows keep fencing/terminality through the union reads below, and they drain within ~1 day of enablement via Part A's first tombstones.

## B3. Writer — the worker

When the worker finalizes a batch of units as reclaimed, in the SAME transaction it previously used to mark rows `deleted` it now:

1. Upserts ledger rows grouped by `(run_id, lead_time_hours, variable_code, target_kind)`:
   `INSERT … ON CONFLICT (pk) DO UPDATE SET deleted_members_mask = reclamation_ledger.deleted_members_mask | :bits, reclaimed_at = LEAST(existing, :now)` — a 2,500-unit batch spans ~85 groups.
2. Deletes the queue rows for the finalized batch.

Idempotent by construction: a re-enqueued evidence-gap unit re-runs the no-op delete and re-merges the same bits.

## B4. Readers (all union with the queue's in-flight states)

* **Planner** (`plan_reclamation_pass`): a unit is "already handled" iff a queue row exists (any status) **or** the ledger covers it. One bulk ledger query per pass for the in-scope runs (~10–30k aggregated rows at a 2-day catalog, vs ~300k queue rows today); bits are expanded into per-unit tuples by a shared helper.
* **Finalizer** (`cycle_reclamation_units_terminal`): every committed unit is covered by **queue row `deleted` (coexistence with pre-migration rows) or ledger**.
* **Serving fence** — all per-unit fence readers become union queries:
  - `resolver.py` / `availability.py` det/mean fence: queue row (`deleting`/`deleted`/`failed` — kept for coexistence) OR ledger row exists.
  - `availability.py` / `resolver.py` member coverage pre-query: queue row for the member OR ledger `mem` row with bit *m* set (any variable, mirroring the queue's variable-agnostic semantics).
  - `ensemble_data.py` member fence: queue member rows (in-flight) ∪ ledger mask bits, resolved in Python (point query per (run, lead, variable), same cost class as today).
  - `tiles.py` fence: queue row by `(store_path, physical_key)` OR ledger row by `(store_path, variable, kind, lead)` with the member bit (the call site already holds the key components; no string parsing needed).
  - `vector_field.py`: same treatment as its current queue query.
- SQLite guard: the fence sites' `has_table` introspection pattern is mirrored for the ledger so sqlite-backed tests skip the branch cleanly.

## B5. What does not change

Canonical deletion authority, the worker's counterfactual revalidation and lease machinery, `failed` quarantine + requeue semantics (a `failed` row stays in the queue; a requeue promotion to terminal writes the ledger instead of a queue `deleted` row), the 240h legacy claim path (Part A), tombstone permanence, serving semantics.

## B6. Expected combined steady state

| Store | Part A only | Part A + Part B |
|---|---|---|
| `reclamation_queue` | ~250–300k rows / ~250–300 MB | ~2–5k rows (in-flight only) |
| `reclamation_ledger` | — | ~7k rows / <10 MB |
| Serving fence | queue rows (2-day window) | queue ∪ ledger (unchanged semantics) |
