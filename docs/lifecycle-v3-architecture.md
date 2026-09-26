# Data Lifecycle Architecture (Lifecycle V3 · Converged)

> Status: authoritative design document. This edition consolidates the Lifecycle V3 locked contract and the eight convergence corrections from the 2026-09 code audit.
> The previous edition's formulation "Whole-cycle finalizer as a GC layer" has been abolished, unified into the canonical-necessity
> single source of truth model.
> Code map: `packages/domain` (pure logic), `services/api` (serving), `services/ingestion`
> (realtime ingestion and GC).

---

## 0. Core Model After Final Convergence

```text
Realtime ingestion
  → variable / lead / member independently committed and published

Serving
  → variable / valid_time canonical source resolution

GC
  → variable / valid_time necessity determined independently
  → individual physical units reclaimed independently

Cycle lifecycle
  → derived bookkeeping only

Batch ingestion finalize
  → batch-only operational concept
  → not part of realtime serving / GC authority
```

**GC final invariant (locked down):**

```text
There is no cycle-level GC authority.

A physical reclamation unit may be deleted only when:

1. canonical serving no longer selects it for any protected valid_time;
2. it is not held by interval fallback;
3. it is not held by precipitation companion source coupling;
4. it is not held by wind U/V atomicity;
5. it is not held by predecessor/recovery dependencies;
6. ensemble/member integrity does not require it: while a (cycle, lead) remains
   a canonical ensemble source, every committed member of that unit is retained —
   the 85% coverage threshold is a candidate-eligibility bound, never a
   reclamation target;
7. its departure from the protected set is durable: for a valid_time still inside
   the active serving window, the replacement serving state has been durably
   published; for a valid_time that has exited the active serving window, the
   exit is established by the monotonically non-decreasing serving_start
   boundary and requires no replacement publication;
8. deletion-time counterfactual revalidation still proves it unnecessary.

Cycle-level deleted_at is only a derived bookkeeping fact after all
constituent units have independently become terminal.
```

---

## 1. Core Principles

1. **The user faces valid_time**. Frontend and serving semantics revolve around `model + variable + valid_time`;
   `cycle_time / lead_time` remain in the backend, as provenance, cache identity and
   ingestion / lifecycle identity, no longer the user interaction model.
2. **The three concepts are orthogonal and must not be conflated**:
   - **A. UI visibility / expiration**: when a valid_time disappears from the dropdown;
   - **B. Serving eligibility**: which valid_time the API has a serveability guarantee for;
   - **C. Physical deletion**: when a physical unit may be deleted.
3. **Canonical source resolution is GC's only deletion authority**. The deletion test is always:
   > whether a physical reclamation unit is still required by any protected variable/valid_time canonical
   > serving or dependency hold.
   Cutoffs such as `T−C`, `T−2C` are **not** a second deletion
   authority parallel to the canonical resolver, but a **derived corollary / planner fast-path optimization** under the current GFS/GEFS configuration.
4. **GC is fully decoupled to (variable, valid_time) granularity**. Between variables, and between valid_times,
   expiry reclamation is mutually independent; there is no cycle-level GC authority.
5. **Realtime's serving/failure unit is variable-lead (and GEFS member)**, not
   whole-run. `model_runs` status is merely operational/catalog information, not a
   one-size-fits-all authority for serving or GC.
6. **Lifecycle identity is `(model_id, cycle_time)`**. GFS/GEFS advance independently.

---

## 2. Independent Time Metadata (four concepts not bound to each other)

The following values currently happen to match, but architecturally they are **distinct concepts** and must remain independently registered:

| Metadata | Current value | Semantics |
|---|---|---|
| model/product max lead | GFS=240h, GEFS=240h | the product's forecast horizon |
| model cycle cadence | GFS=6h, GEFS=6h | the interval between adjacent cycles |
| variable interval width | precipitation=3h, cloud=3h | the statistical width of an interval quantity |
| variable reset period | precipitation=6h, cloud=6h | the reset period of the upstream accumulation |

Hence, in architectural semantics:

```text
L % 6 == 0            (wrong: bound to a specific number)
L % variable.reset_period_hours == 0   (correct)
```

When products with different horizon or reset semantics are introduced in the future, model cadence will not be
incorrectly coupled with variable accumulation/reset semantics. Code map: `domain/horizon.py` (max lead),
`domain/cadence.py` (cycle cadence), `domain/temporal.py` (interval semantics),
the predecessor determination in `domain/reclamation.py` should be parameterized as `reset_period_hours`.

valid_time grid and anchor: `serving_start = latest model valid_time <= now`
(floored to the 3h grid; 07:00Z → 06Z, 09:00Z → 09Z). anchor advance → frontend forced refresh →
old valid_time exits the active serving window.

---

## 3. Variable Semantic Classification (static contract, never a numeric test)

| Class | Variable | lead 0 semantics |
|---|---|---|
| Instantaneous | temperature_2m, wind_10m (u/v composite), relative_humidity_2m, visibility, snow_depth, cloud_ceiling … | meaningful; T+0 is canonical |
| Interval | precipitation_amount_3h, cloud_cover_3h | NaN in store, must fall back to a positive lead |
| Precip companions | crain, csnow, cfrzr, cicep | not independent resolver variables; strictly follow the actual source of precipitation_amount_3h |
| Wind component pair | wind_u_10m + wind_v_10m | **atomic pair**: enters serve only when both U and V are commit ready; GC moves them in and out as a pair |

Classification comes only from the static set of variable names (`domain/temporal.py`); `if value == 0` or treating
NaN as a serving contract is **absolutely not allowed**—0 mm precipitation is a real forecast.

**Reset lead reconstruction (generalized over W, R)**: let `W = variable.interval_width_hours`,
`R = variable.reset_period_hours` (requires `W < R`; when `W == R` the upstream already provides
a width of W at the reset lead, so no reconstruction is needed). Reconstruction is needed only at a reset lead (`L % R == 0`):

```text
precipitation accumulation:
    A_interval(L) = A_reset_accum(L) − A_reset_accum(L − W)      predecessor = L − W

cloud running average:
    X_W(L) = [ R·C(L) − (R−W)·C(L−W) ] / W
```

The current configuration `W=3, R=6` degenerates to `f006 − f003` and `2·C₆ₕ − C₃ₕ(t−3)` respectively (cloud carries a
physical-range guardrail)—the existing formulas are merely special cases; writing `W = R/2` into the general logic is **forbidden**.
Differencing is done at the **ingestion decode layer**; what serving reads is already an interval value; for GC predecessor holds see §7.1
(a reset lead L holds the same run's L−W).

---

## 4. Serving Model (source selection)

For each `(model, variable, valid_time)`:

- **R1** All `(cycle, lead)` combinations capable of producing that valid_time are candidates.
- **R2** **newest serveable wins**: the newest safely committed and serveable candidate wins.
  A new partial cycle overwrites an old cycle only on the valid_times it actually committed.
- **R3 interval lead0 fallback**: triggered only when the winner's `lead == 0`; the target is the **next-newest serveable positive-lead**
  candidate for the same valid_time; a positive lead never falls back.
- **R4 companion same-source**: crain/csnow/cfrzr/cicep are exactly the same
  (cycle, lead, store) as precipitation_amount_3h; the current cycle's lead0 categorical flags are an ingestion-synthesized representation,
  and mixing sources would misjudge the phase.
- **R5** cloud_cover_3h falls back independently, uncoupled from precipitation.
- **R6 wind atomic pair**: wind_10m requires u and v from the same candidate; serving is possible only when both U and V are commit ready.
- **R7 coverage threshold**: GEFS requires ≥85% members per lead (integer-safe test); fallback
  candidates are likewise bounded—fallback is the "next-newest serveable positive-lead candidate",
  not the "previous cycle".
  **85% is a candidate eligibility threshold, not a GC retention target**: it answers only
  "is this candidate eligible for serving", not "how few members must be retained". As long as a
  (cycle, lead) remains a canonical ensemble source, **all of its committed members are
  retained**—the remaining 4 already-committed members must never be reclaimed just because 26/30 already satisfies 85%.
- **R8 graceful null**: with no fallback, instantaneous still returns values and interval is set to null,
  and the whole request still succeeds.
- **R9 mixed-source is the correct shape**: within a single /v1/points entry different variables may come from different
  sources; valid_time is the unified time axis.
- **R10 availability true semantics**: an interval variable is
  available only if a valid positive-lead source exists; instantaneous is not subject to this constraint.
- **R11 cache bound to the real source**: single-variable endpoint key = actual (cycle, lead) + valid_time +
  the serving generation of that source store. The /v1/points provenance digest generates an entry for **each
  source actually used**
  `(variable, valid_time, run_id, source cycle, source lead, the serving generation of that source store's
  manifest)` (NULL sentinel when there is no fallback source), sorted and then hashed.
  generation must be resolved per each source's **own store**—taking only the primary/anchor
  generation is not allowed, otherwise when the store holding a fallback source republishes (G06 17→18) while the primary
  is unchanged, the cache will hit the old precipitation. companions are separate entries, but by I10 they have the same
  value as the precipitation entry (more conservative: when the coupling is broken the digest changes truthfully).
- **R12 provenance reflects the real origin**: single-variable endpoints report the actual `source_cycle`/`lead`;
  /v1/points keeps primary/anchor at the entry level to maintain API compatibility.

---

## 5. Serving retention Boundary and the `protected_valid_times` primitive

```text
UI visibility, serving eligibility and physical deletion are independent states.

The API guarantees serveability only for valid_times inside the active serving window;
once a valid_time exits the active window it no longer has a serving retention guarantee,
but physical data is likewise not required to be deleted in step.
```

**`protected_valid_times` is a shared domain primitive (the only
authoritative implementation of the active serving window policy)**:

```text
protected_valid_times(model, now) =
    { valid_time : valid_time >= serving_start_valid_time(now) }
    ∩ that model's canonical horizon grid
```

- The only implementation lives in `packages/domain` (a pure function; the existing `serving_start_valid_time` is upgraded to
  be its carrier).
- Three consumers **reference the same implementation and are forbidden from computing the boundary themselves**:
  1. frontend availability—consumed via the serving window metadata exposed by the API; the TS side does not reimplement
     the boundary math;
  2. API serving eligibility—the resolver's window left-boundary test;
  3. GC canonical protection—the planner's `start_valid_time` parameter and the worker's
     deletion-time revalidation.
- The "protected valid_time" in invariants 1/7 is defined by this primitive. The three-way
  boundary drift across UI/API/GC ("UI thinks 03Z expired, API thinks active, GC thinks protected") is
  structurally impossible.

Product behavior chain: anchor advance → frontend forced refresh → old valid_time exits the active serving window →
the user can no longer explicitly request the old valid_time through normal product interaction. GC does **not** need to retain data for "a user
might keep querying a valid_time that has exited the window"; at the same time physical deletion is likewise not required to be in step with window exit.

UI-side mechanism: 3h grace window filtering + 60s heartbeat + scheduled refresh aligned to the cadence boundary.

---

## 6. Realtime ingestion: variable-lead-member commit semantics

### 6.1 commit / publish unit

The unit of serving and failure is **variable-lead (and GEFS member)**, not whole-run:

```text
failed / uncommitted variable-lead unit
  → does not participate in canonical serving, nor produce a serving hold;

other already committed / published variable-lead units in the same cycle
  → unaffected.
```

Write sequence (per wave):

1. **reserve_run** (before physical writes): upsert the lifecycle row + `model_runs('processing')`.
   catalog identity comes first.
2. **region commit**: under the EXCLUSIVE gate, write the variable shard; the region COMPLETE marker
   is written last (the last step on the store side).
3. **lead settlement / promotion**: after all members of a lead have settled, `publish_settled_lead`:
   based on marker evidence, write EnsembleMember / EnsembleMemberProduct / Product rows and bump the
   manifest serving generation. This step does not mark the run ready.
4. **batch finalize (batch-only)**: the `model_runs` ready/partial status is derived by batch ingestion
   finalize, and is an **operational/catalog concept**, not the authority
   for realtime serving or GC.

### 6.2 wave scheduling model

Single leader, waves dispatched serially (GFS → GEFS, the same model never runs concurrently); a new wave starts
only after the previous wave's finalizer completes; graceful shutdown goes through a non-abandoning drain; the scheduler has no state of its own and
rebuilds pending on each poll from the upstream snapshot + durable catalog, so after a crash the next round of
reconciliation only fills in what is missing. **Under the current design a wave will not be permanently abandoned**; the measured orphan stores are
historic leftovers (the pre-catalog-wiring path / database rebuilds), and prevention of recurrence relies on the §9 reconciliation.

---

## 7. Physical GC: canonical necessity as the single source of truth

### 7.1 Deletion authority and the derived fast-path

**The deletion test is always** (items 1–8 of the §0 invariant), executed by exactly the same canonical pure engine
as serving (`select_canonical_sources_bulk`).

Under the current GFS/GEFS configuration, the **derived result** of that test can be written in cutoff form, as the planner's
fast-path / audit criterion (T = the latest ready cycle that has started serving, C = cycle cadence):

- instantaneous variable shard: once `cycle_time ≤ T − C` it is no longer selected by canonical;
- interval variable shard: T−C is still serving as the lead0 fallback, so only after `cycle_time ≤ T − 2C`
  is it no longer selected;
- wind: a u/v atomic pair; once T's wind starts serving ⇒ neither u nor v of T−C is selected any more. Atomicity is
  **semantic atomicity**, not a physical transaction (definition in §7.2).
- predecessor: a reset lead's shard (`L % variable.reset_period_hours == 0`) depends on
  the same run's **L − variable.interval_width_hours** (not L − R/2); before the relevant predecessor is committed
  (recovery scenario) it is held by a dependency hold.

These cutoffs are always a **corollary**, never the source of the test; when configuration changes, canonical resolution prevails.

### 7.2 Execution pipeline (the only one)

```text
Reclamation Planner
  → (canonical resolution + dependency holds) → reclamation_queue
Reclamation Worker
  → lease claim → SHARED store gate → deletion-time counterfactual revalidation
  → DeleteObject (unit level) → unit terminal
(loop until all constituent units are terminal)
```

- **Planner**: canonical resolution for each (model, variable, valid_time); a committed unit that is not the anchor, not a
  variable fallback source, not wind-coupled, not a predecessor, and not required by member integrity
  → is enqueued into `reclamation_queue`.
- **Worker**: lease claim of a batch → SHARED store gate → re-read lifecycle (if the cycle is already
  tombstoned, handle it per bookkeeping semantics) → counterfactual revalidation (treat this batch as
  physically present and the rest of deleting/deleted as fenced, recompute necessity; those still necessary fall back to queued) →
  DeleteObject → mark deleted. **Each physical unit is deleted only after independently passing all 8 conditions.**

  **Wind pair semantic atomicity** (S3 DeleteObject has no cross-object transaction; atomicity is defined
  in layers):
  1. **Eligibility atomic**: only after both u and v have been determined unnecessary may either one
     enter `deleting`—if any component is still necessary, the pair as a whole does not enter the deletion batch;
  2. **Serving atomic**: once the pair is fenced / reclaiming, the resolver must not select that
     pair (the existing `filter_candidates_by_physical_fence` already provides this semantics: a fenced u makes the
     candidate lose wind_u → wind_10m's "same candidate u+v" requirement makes the whole pair unselectable);
  3. **Physical deletion**: u and v may be DeleteObject'd in sequence; simultaneity is not required;
  4. **Crash between deletes**: a transient one-deleted / one-present state is permitted—the pair is already
     fenced as a whole (serving is unaffected), and recovery continues deleting the remaining member still under the deleting lease
     until terminal.
- **There is no whole-cycle GC finalizer**: there is no stage of "whole-cycle GC → decide whether to delete the entire cycle",
  nor any independent authority that deletes an entire prefix directly based on `cycle horizon expired`.
  In the current implementation the horizon-based `run_finalizer_pass` deletion path must be retired (see §11).

### 7.3 Marker / manifest cleanup: likewise dependency-driven

An S3 prefix is not a real directory; there is **no** final stage of "rm the entire cycle prefix". Each metadata
object is reclaimed independently according to its own dependencies:

```text
variable shard
  → deleted after the corresponding canonical/dependency hold disappears (the worker's current path)

region COMPLETE marker
  → deleted after all physical units it attests to are terminal (the current region marker cleanup is exactly this model)

manifest / cycle metadata object (.zattrs/.zgroup/.zmetadata etc.)
  → deleted after all active units still depending on it are terminal
```

GC maintains the same dependency-driven model from start to finish; the "emptying" of a cycle store prefix is the **result** of all
units being deleted independently, not the **action** of any single step.

### 7.4 Failed / uncommitted units and partial protection

- failed / uncommitted variable-lead units do not participate in serving and produce no hold (§6.1);
- orphan/trailing units **older** than the "cycle currently serving": canonical resolution naturally no longer selects them,
  so they can be reclaimed;
- committed units in the newest partial cycle that has no ready cycle after it: they are inside the
  serving window and protected by a canonical hold, and **must not be deleted on the basis of time alone**;
- no arbitrary TTL (48h-timeout-style policies are explicitly rejected).

### 7.5 Deletion hard prerequisites: two classes of GC trigger, with different "durability" evidence

Invariant item 7 splits into two classes by the way a unit leaves the protected set:

**A. Replacement GC (taken over inside the window)**: the valid_time is still inside the active window and is taken over by a newer
cycle unit. Hard prerequisite = the replacing source's manifest serving generation has been bumped—
this is the durable evidence of the serving switch and can be verified by planner/worker. The frontend forced refresh is driven by the API-side
generation change; GC does not directly perceive the UI.

**B. Window-exit GC (window exit)**: the valid_time itself exits the active window
(`VT < serving_start`). **There is no replacement**—serving_start is a deterministic
function of UTC and monotonically non-decreasing; the exit is self-evidencing and requires no publication. Since the smallest
valid_time the next cycle can produce is `cycle_time + cadence`, every cycle's **lead-0 / lead-3 units (generally:
all units whose VT is below the next cycle's first VT) as well as the anchor-region units of interval variables**
are never replaced and go only through this class. If a replacement generation were mandatory, these units would never
be reclaimable.

Safety basis: ① serving_start is monotonically non-decreasing, so a valid_time that has exited can never re-enter the window;
② condition 8's deletion-time counterfactual revalidation recomputes the protected
set at the instant of deletion, so transients such as a clock rollback are caught by the last line of revalidation; ③ an optional belt-and-braces margin (reclaiming only
`VT < serving_start − margin`), not required.

---

## 8. Cycle lifecycle: derived bookkeeping only

`forecast_cycle_lifecycle.deleted_at` continues to be permanently retained, but its semantics are explicitly:

```text
all physical reclamation units of this cycle have independently entered the terminal state
```

and not "deleted_at / some finalizer authorizes deletion of the entire cycle". The direction can only be:

```text
per-variable/per-valid-time GC
  → all constituent units terminal
  → derive cycle-level deleted_at (starts the 14-day metadata retention clock)
```

never the reverse. In the current implementation the component playing that role should be renamed to
**Lifecycle bookkeeping / metadata reconciliation**, and it may only:

- observe whether all units are terminal;
- write `deleted_at`;
- start the 14-day metadata retention clock (the sweeper cleans the detail catalog:
  model_runs / forecast_products / ensemble_members / ensemble_member_products /
  reclamation_queue; tombstones are retained permanently).

It **may not**: DeleteObject, delete an entire prefix, decide whether a given shard is reclaimed, or bypass the canonical
planner.

---

## 9. Store ↔ catalog reconciler: monotonic recoverability frontier

The system is monotonic: **an old cycle will not be reactivated by the realtime scheduler**. Therefore reconciliation needs no
anti-resurrection subsystem, and adopts a simple rule:

```text
store exists + catalog missing detected
        ↓
determine whether the cycle still lies inside the recoverable / active frontier

recoverability frontier already crossed
        ↓
never restore catalog
        ↓
handle as orphan cleanup (units go through the canonical necessity test one by one and are then reclaimed;
in practice at this point they are all unnecessary and can be made terminal quickly)

still inside the legitimate recoverable region
        ↓
check COMPLETE marker / durable evidence
        ↓
restoring catalog is allowed
```

The permanent lifecycle tombstone may serve as an extra sanity guard, but no separate new mechanism is built
around resurrection.

**Scheduling semantics (already implemented):** reconciliation is a built-in low-frequency phase of the GC daemon, no longer a purely manual one-shot command.
By default the daemon, every 24h (`--inventory-interval-hours`, `0` disables, env fallback
`GC_INVENTORY_INTERVAL_HOURS`), runs `run_orphan_inventory(reap=False)` once at the end of some pass, and the
summary is appended to that round's `GC pass:` output; `--once` mode does not trigger it. The scheduling path **only discovers and reports, never
reaps**—physical deletion of orphan prefixes is still possible only through the manual one-shot command
`weather-ingest gc --inventory --inventory-reap` (fail-closed sanity guards unchanged).
Reconciliation results are exposed as in-process metrics via `ingestion/monitoring/gc_metrics.py`
(`weather_gc_inventory_orphans{beyond_frontier=...}` etc., with the `gc --metrics-port` endpoint opened),
and an orphan count > 0 inside the frontier triggers the `gc_orphan_stores_detected` alert.

---

## 10. Invariant Inventory

| # | Invariant |
|---|---|
| I1 | user semantics are only `model + variable + valid_time`; cycle/lead exist only in the backend |
| I2 | anchor = latest model valid_time ≤ now; boundary advance triggers an immediate frontend refresh |
| I3 | UI visibility, serving eligibility and physical deletion are mutually independent; the API guarantees serveability only inside the active window |
| I4 | lifecycle identity is (model_id, cycle_time); zero coupling between models |
| I5 | GC test granularity is (variable, valid_time); **canonical necessity is the only deletion authority** |
| I6 | cutoff (T−C / T−2C) is a derived fast-path, not a deletion authority |
| I7 | **No cycle-level GC authority**; a unit deletion must pass all of invariant conditions 1–8 |
| I8 | wind u/v **semantic atomicity**: eligibility atomic (only after u/v are both determined unnecessary may either enter deleting) + serving atomic (once the pair is fenced/reclaiming, the resolver must not select that pair); physical DeleteObject may be executed in sequence, and the transient one-deleted/one-present produced by a crash is covered by the whole-pair fence, with recovery deleting the remainder |
| I9 | a reset lead depends on the same run's predecessor `L − W` (W=interval width, not R/2); it is held by a dependency hold before the predecessor is consumed; applicability is determined by `L % R == 0` |
| I10 | companions are always from the same source as precipitation_amount_3h |
| I11 | fallback is triggered only by semantic class + lead==0; never based on a numeric value/NaN |
| I12 | a fallback candidate must pass the coverage threshold |
| I13 | the realtime serving/failure unit is variable-lead-member; whole-run status is merely operational/catalog information |
| I14 | a unit leaving the protected set must be durable: replaced inside the window → the replacement generation has been bumped; window exit → self-evidencing via monotonic serving_start, no publication needed |
| I15 | the newest committed partial with no successor must not be deleted on the basis of time alone; older orphans may be reclaimed |
| I16 | `deleted_at` is derived bookkeeping; tombstones are retained permanently; detail metadata for 14 days |
| I17 | cache identity and provenance are bound to the actual source and serving generation |
| I18 | marker / manifest cleanup is dependency-driven just like shard cleanup; there is no whole-prefix deletion stage |
| I19 | max lead / cycle cadence / interval width / reset period are independent metadata; the reconstruction formula is expressed only in terms of (W, R), and the implicit binding `W = R/2` is forbidden |
| I20 | `protected_valid_times(model, now)` is the only authoritative primitive of the active serving window policy; frontend/API/GC reference the same implementation and are forbidden from computing the boundary themselves |

---

## 11. Gap Between the Current Implementation and the Target State

> Rewritten on 2026-09-13 after review: after verifying the original 11 gap items one by one against the code, 10 have landed (§11.1,
> of which item 7's tool has been built while scheduling and historical orphan cleanup remain outstanding), and the remaining work converges to the
> 7 outstanding items in §11.2. This section defers to the current state of the code, and items are annotated with evidence locations.

### 11.1 Landed (original inventory → current state)

| # | Original gap item | Current state |
|---|---|---|
| 1 | Retire the horizon-based whole-cycle deletion path | Done. The finalizer has been demoted to §8 Lifecycle bookkeeping: zero physical storage operations, no DeleteObject, and `deleted_at` is a derived fact (`gc/finalizer.py` module docstring). |
| 2 | Take down the leftover V2 reconciler | Done. No reconciler class/leftover path anywhere in the repo; store↔catalog divergence is handled centrally by the §9 inventory. |
| 3 | Land the planner fast path | Done. canonical resolution is the only authority and cutoff serves merely as a batch-pruning fast-path (`gc/planner.py` module docstring); `batch_size` enqueues in batches (`planner.py:157,595-597`). |
| 4 | Make the worker's revalidation of wind coupling pairwise | Done. u/v are jointly evaluated as a pair by `(run, lead, kind, member)`; if any component is counterfactually determined necessary or the counterpart is not in a terminal state, the whole pair returns to the queue (`gc/worker.py:446-497`). |
| 5 | /v1/points fallback implementation | Done. `/v1/points` converges to a single implementation (`routers/points.py`); fallback is resolved inside the resolver by `(variable, valid_time)` and carries the provenance digest (`api/services/resolver.py:589,709-779`), with no parallel second engine. |
| 6 | ensemble_data post-read fence recheck failing silently | Done. After the read the fence is explicitly re-queried and reported via a named log event (`api/services/ensemble_data.py:376-391,643-658`), no longer silently swallowed by `except: pass`. |
| 8 | Split time metadata (I19) and generalize the reconstruction formula | Done. `is_predecessor_dependent_lead` / `get_predecessor_lead` are parameterized by `(W, R)` (`domain/reclamation.py:229-277`); `reconstruct_running_average_interval` implements `[R·C(L) − (R−W)·C(L−W)] / W` (`domain/models/cloud.py:77-161`); the `(W, R)` metadata registry is in `domain/temporal.py:59-79`, with generalized tests across (R, W) combinations. |
| 9 | availability legacy `initial_times` view | Done. The legacy view explicitly evicts lead 0 for interval variables (`api/services/availability.py:387-406`). |
| 10 | Tests polluting the production catalog | Done. Integration tests require an explicit `TEST_DATABASE_URL`, otherwise they are skipped (`services/*/tests/_integration_db.py`); one-shot cleanup tool `scripts/cleanup_test_pollution.py` (commit `87b8207`). |
| 11 | Extract the `protected_valid_times` primitive (I20) | Mostly done. The primitive has been upgraded to a per-model horizon grid and provides `is_valid_time_protected(model_id=...)` / `protected_valid_times(model, now)` (`domain/temporal.py:154-295`, PR #87); planner and API have both adopted it. The frontend TS still reimplements the boundary computation, see outstanding item 6. |
| 7 | store↔catalog reconciliation (§9) | Partially done. The reconciliation tool has been built and wired to the CLI: `gc/inventory.py` + `--inventory` / `--inventory-reap` (PR #88); scheduling and cleanup of historical orphans (gefs 2026-09-12/00, 06 etc. ≈57GB) remain outstanding, see outstanding items 1/2. |
| — | Switch the GC revalidation boundary to the per-model primitive | Done. Added `domain.temporal.model_serving_start_valid_time(model_id, now)` (derives the boundary from the model's registered horizon cadence, consistent with the boundary component of `is_valid_time_protected(model_id=...)`); planner / worker / finalizer / inventory have all been switched, and inventory keeps a global cadence fallback for unregistered models. |
| — | Mechanized verification of the I14 generation | Done. The direction is set as "only delete if unchanged" (fail-closed): on enqueue the planner snapshots the store's committed-manifest generation (migration 009 `reclamation_queue.store_generation`, bumped on every EXCLUSIVE commit, including same-set same-cycle replacement), and the worker compares inside the store gate—if the generation changed or disappeared it resets the baseline and returns the item to the queue for one round without deleting; pre-existing rows with no baseline are backfilled on first claim. A repeatedly replaced store is never deleted, and a stable store resumes deletion on the next round (no livelock). |

### 11.2 Outstanding items that are still valid (in recommended priority order)

1. **Bring §9 reconciliation into the schedule**: `gc --inventory` is still a one-shot manual command (`cli.py:746-748`
   exits on completion), the daemon loop contains no reconciliation phase, and neither compose nor cron schedules it. It should be a low-frequency
   daemon phase (reap stays manual).
2. **Historical orphan cleanup**: the ≈57GB of orphan stores such as gefs 2026-09-12/00, 06 need operations to
   explicitly clean them with `gc --inventory --inventory-reap` (the tool is ready, the action has not been executed).
3. **In-pipeline stage metrics**: planner/worker/sweeper per-round results are only a stdout summary
   (`cli.py:1378`), and there are no Prometheus count/duration metrics inside `gc/`. State-level coverage already exists
   (`monitoring/lifecycle_collector.py:65-79` + Grafana sections 7/8/9 + webhook alerts
   `alerts.py:456-466`); what is missing is per-pass process metrics and the corresponding panels.
4. **Frontend TS reimplementing the serving window**: `frontend/src/lib/forecast/availability.ts`
   reimplements the floor logic of `serving_start_valid_time` (I20's "single implementation across the stack" is not yet
   fully achieved); the semantics currently agree, but it is a maintenance-drift risk.
5. **Daemon deployment vehicle**: `docker-compose.yml` has no ingestion/GC service, and the repo has no
   systemd unit/crontab; the `--enable-planner` → `--enable-delete` gradual enablement is still
   a manual operational action (`RUNBOOKS.md` GC section). Per the DEPLOYMENT.md convention this is left to Stage 8.
6. **Per-variable provenance exposure to clients**: the provenance digest already enters the cache key
   (`resolver.py:709-779`), but the response schema exposes only the per-series `cycle_time`
   (`api/schemas.py:287-294`), so clients still cannot see which cycle/run each variable comes from.
7. **T−2C claim timing**: a pure pruning-efficiency optimization; correctness is covered by canonical necessity;
   the latest release of tombstone/last-batch units is deferred by 10 days. To be evaluated after GC
   runs stably and there is storage reclamation curve data.
