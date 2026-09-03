# Phase 5C — Implementation Report

**Realtime Frontier Planner + Shared Barrier + Bounded Lead-Wave Scheduler.**
Phase 5B semantics are unchanged; big-batch ingestion behavior and performance
are untouched; Phase 6 (supersession/retention/GC) remains out of scope.

---

## 1. Files changed

### New production files (`services/ingestion/src/ingestion/realtime/`)

| File | Purpose |
|---|---|
| `planner.py` | Pure, offline-testable planner: shared GFS+GEFS barrier, three distinct frontiers, pending walk with no-jump + predecessor eligibility, bounded batching, structured `FrontierPlan` diagnostics (never a bare lead list). |
| `polling.py` | Explicit `ACTIVE / PUBLISHING / BACKOFF` poll state machine with injectable RNG (`jitter_interval`); discovery failures never reach it. |
| `committed.py` | Durable committed-state reader: two indexed catalog queries per model (`model_versions → model_runs → forecast_products / ensemble_member_products`); no Zarr/object-store scans. |
| `leadership.py` | Session-level PostgreSQL advisory-lock leadership (`domain.locks.scheduler_leader_key`), optimization-only, crash-released; `NoopLeadership` for tests. |
| `scheduler.py` | `RealtimeScheduler`: cycle selection (explicit + auto), discovery polling, snapshot diffing, planner invocation, wave timing/dispatch, poll state, graceful shutdown, structured JSON diagnostics. No download/decode/write/finalize logic. |
| `__init__.py` | Package marker. |

### Modified production files

| File | Change |
|---|---|
| `packages/domain/src/domain/locks.py` | New `_NS_SCHEDULER_LEADER` namespace + `scheduler_leader_key()` + `REALTIME_SCHEDULER_LEADER_IDENTITY` (pure, tested; domain coverage gate held at 100%). |
| `services/ingestion/src/ingestion/core/config.py` | `REALTIME_*` settings block (see §6) with validator invariants. The canonical horizon is deliberately **not** config (it stays the `domain.horizon` product contract). |
| `services/ingestion/src/ingestion/core/wave_runner.py` | `_run_wave` gains one **optional** parameter, `cancel_event: threading.Event \| None = None` (default preserves CLI behavior exactly; when provided, the wave uses the caller's event so shutdown can trigger the existing non-abandoning drain). Body otherwise untouched. |
| `services/ingestion/src/ingestion/cli.py` | New `realtime` subcommand (§11). |
| `.env.example` | Documented the new `REALTIME_*` settings. |

### New tests

| File | Count | Coverage |
|---|---|---|
| `tests/test_realtime_planner.py` | 21 | every planner scenario in the task spec |
| `tests/test_realtime_polling.py` | 9 | transitions, backoff growth/reset, jitter bounds, invalid config |
| `tests/test_realtime_scheduler.py` | 19 | discover→plan→dispatch, reconciliation, partial success, restart, big-batch interleave, leadership, shutdown (sleeping + during wave), staggered cycles, failure semantics, jitter bounds |
| `tests/test_realtime_committed.py` | 3 | catalog reader on SQLite: per-model leads/pairs, GFS-ahead-of-GEFS, absent cycle/model, version scoping |
| `tests/test_realtime_cli.py` | 4 | flag pairing validation, master-switch gate, `--once --dry-run` offline end-to-end, argparse validation |

---

## 2. Planner / frontier model

`plan_wave(...)` consumes both `CycleSnapshot`s (same `CycleIdentity` — pairing
across timestamps is structurally impossible), per-model durable
`ModelCommittedState`, the canonical horizon (`domain.horizon`), the required
GEFS member set (`domain.coverage`: gep01..gep30), the `WavePolicy`, and
timing metadata; it returns a `FrontierPlan` with:

- `observed_frontier` — highest lead with ANY upstream artifact across both
  models (any member, data or `.idx`, including upstream-only GFS hourly
  leads). Upstream reality; never wave content.
- `complete_frontier` — the contiguous shared-barrier-complete prefix of the
  canonical horizon.
- `committed_frontier` — the contiguous prefix committed for BOTH models.
- `pending_complete_leads` — shared-complete, uncommitted, contiguous,
  predecessor-eligible leads after the committed frontier (untruncated).
- `next_blocked_lead` + `blocked_reason` (`gfs-incomplete` / `gefs-incomplete`
  / `not-observed` / `predecessor` / `horizon-complete`) with exact
  `missing_gfs_artifacts` (`data`/`idx`) and `missing_gefs_members` (sorted
  member identities) for the blocked lead.
- `wave_due` / `wave_candidate` / `wave_targets_gfs` / `wave_targets_gefs` /
  `oldest_pending_age_seconds`.

The prompt's worked example is pinned by tests: committed f003, f006/f009/f012
complete, f015 partially publishing → pending exactly `[f006, f009, f012]`,
blocked at f015 with the exact missing-member diagnostics; f015 never enters a
wave regardless of how fast it is publishing.

## 3. Shared barrier implementation

A lead is complete iff **GFS data + `.idx`** AND **all 30 GEFS perturbation
members each have data + `.idx`** (`gec00`/`geavg`/`gespr` are never parsed or
required). The barrier is implemented exclusively inside `planner.py` as
scheduler policy over the Phase 5B discovery snapshots — discovery still
exposes per-model reality only, and no storage/marker/catalog/serving code
knows it exists. Decoupling GFS/GEFS later means replacing this policy
function; nothing else changes. The 85% serving threshold (Phase 3) is
untouched and applies to serving, not to the realtime barrier.

## 4. Durable committed-state reconstruction

`realtime/committed.py::read_cycle_committed_state(engine, cycle_time)` reads,
per model, the run row via `model_versions (model_id, version_string) →
model_runs (cycle_time)` and then the committed `forecast_products` leads /
`ensemble_member_products` pairs — the same durable truth the serving tier
uses. Two small indexed queries per model per poll; **zero** physical store
scans. Missing run rows → empty state (nothing committed for that model).

Asymmetric progress is handled by design: each model's state is read
independently (tests pin GFS-ahead-of-GEFS), and the planner excludes each
model's durably committed leads from that model's wave targets, so big-batch
commits between polls are detected on the next reconciliation and never
duplicated. A catalog read failure **skips planning entirely** for that poll
(never plan against an empty committed state — that would duplicate work) and
retries on the failure interval.

## 5. Batching semantics

`wave_max_leads OR wave_max_wait`, whichever first (both `REALTIME_*`
configurable). The wave is the pending prefix truncated to `max_leads` — the
spec's example (pending `[3,6,9,12,15,18]`, max 4 → emit `[3,6,9,12]`) is a
pinned test. `first_seen_complete_at` is scheduler-memory timing only: after a
restart ages reset to 0 (timers-only state; pinned test), and pending work is
reconstructed from upstream + durable state, so correctness is unaffected.

## 6. Polling state machine and defaults

`ACTIVE (600 s) → PUBLISHING (120 s) → BACKOFF (1800 s doubling to 3600 s)`,
all configurable (`REALTIME_ACTIVE_POLL_SECONDS`,
`REALTIME_PUBLICATION_POLL_SECONDS`, `REALTIME_IDLE_BACKOFF_INITIAL_SECONDS`,
`REALTIME_IDLE_BACKOFF_MAX_SECONDS`, `REALTIME_POLL_JITTER_FRACTION`,
`REALTIME_WAVE_MAX_LEADS`, `REALTIME_WAVE_MAX_WAIT_SECONDS`,
`REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS`,
`REALTIME_FIRST_PUBLICATION_DELAY_SECONDS`, `REALTIME_ENABLED`).

Publication **activity** (→ PUBLISHING, resets backoff) = any snapshot change:
new data object, new `.idx`, GEFS member-count growth (8/30 → 22/30 pinned as
a test), new observed lead, or complete-frontier growth — via the Phase 5B
`publication_changed()` fingerprint diff. An unchanged successful poll is
**idle** (→ BACKOFF), never a failure. Jitter (± fraction, seeded-RNG
testable) applies only to poll intervals.

## 7. Discovery failure behavior

Discovery exceptions (`DiscoveryUnavailableError`, invalid response,
pagination failure) are caught in `poll_once` **before** any state transition:
the outcome is `discovery-failed`, the last good snapshots are preserved as
the diff baseline, the poll state machine is untouched (pinned test: state and
snapshot identity survive a failure), and the loop retries on
`REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS` (60 s) instead of backing off.
Network failure can therefore never masquerade as "upstream idle". The same
protection applies to committed-state read failures (`state-read-failed`).

## 8. GFS/GEFS partial-success reconciliation

A shared wave is dispatched as **independent per-model store operations**,
sequentially (GFS wave, then GEFS wave — never concurrently against
themselves), each through the unchanged wave runner with its own store gate.
There are no cross-model transactions. If GFS succeeds and GEFS fails, GFS
stays committed; because targets are candidate-minus-per-model-committed, the
next reconciliation retries **only** the missing GEFS work (pinned test:
second dispatch list is `["gefs"]` only). GEFS committed ahead of GFS is the
mirror case (planner test `wave_targets_gfs == ()`).

## 9. Scheduler leadership

`SchedulerLeadership` holds a session-level `pg_try_advisory_lock` on
`domain.locks.scheduler_leader_key()` (new dedicated namespace, disjoint from
store-gate/admission/region keys) on a dedicated connection for the
scheduler's lifetime. Crash → session dies → leadership released naturally;
no stale leader state. A second instance's `try_lock` fails → clear log line →
exit 0 without doing realtime work (pinned test). Leadership is
optimization-only: it does not replace the per-store coordinator gates, and
big-batch execution acquires nothing new.

## 10. Cycle selection

- **Explicit mode** (`--cycle-date` + `--cycle-hour`): deterministic
  operation/testing; both flags required together (validated).
- **Auto mode** (default): `newest_eligible_cycle(now, delay)` — the newest
  cycle whose `cycle_time + REALTIME_FIRST_PUBLICATION_DELAY_SECONDS` (default
  3 h, matching the probed publication onset) has passed; if its snapshots are
  both empty, the scheduler falls back to the tracked/previous cycle (at most
  two probes per poll). A cycle is adopted as soon as either model shows
  artifacts; the shared barrier simply stays blocked until both publish
  (pinned "staggered appearance" test). Cycle identity always matches between
  models because both snapshots are taken for one `CycleIdentity`. No
  supersession/retention/GC — when a newer cycle is adopted the old one is
  simply no longer tracked (Phase 6 owns lifecycle).

## 11. CLI commands

```
weather-ingest realtime [--cycle-date D --cycle-hour H]
                        [--once] [--dry-run]
                        [--download-dir DIR] [--concurrency N]
```

- default: poll loop until shutdown (SIGINT/SIGTERM handled → prompt stop;
  active wave drains non-abandoningly via the runner's external cancel event).
- `--once`: **exactly one** poll iteration *including* wave dispatch, then
  exit (cron-driven operation / deterministic testing).
- `--dry-run`: plan and log diagnostics **without dispatching** (the two flags
  compose: `--once --dry-run` is the one-iteration diagnostic).
- Requires `REALTIME_ENABLED=true` and PostgreSQL (leadership + catalog);
  big-batch `ingest` is untouched and doesn't read any of this.

## 12. Shutdown behavior

- Polling/waiting: the sleep waits on the stop event → prompt return.
- Active wave: `request_stop()` sets the external cancel event passed to
  `_run_wave`, which triggers the runner's existing non-abandoning drain
  (workers finish, remaining tasks cancel, the finalizer still runs) — commit
  invariants unchanged. Pinned test: a dispatch blocked mid-wave observes the
  cancel request and the loop exits promptly.
- Double signal-safe: `request_stop()` is idempotent; signal handlers are
  installed with main-thread guards.

## 13. Observability

Every poll emits one structured JSON log line (`realtime_poll …`) with: cycle,
poll kind, poll state, next interval, last successful poll, last publication
activity, leader status, discovery error, activity flag, observed/complete/
committed frontiers, pending leads, oldest pending age, blocked lead + reason,
missing GFS artifacts, missing GEFS members, wave due/candidate, per-model
wave targets, and per-model wave results (status, failure count, error) —
reusing the existing `logging` patterns; no new observability platform.

## 14. Performance impact

- **Big-batch performs zero discovery work**: `ingestion/cli.py`,
  `core/wave_runner.py`, `core/coordinator.py`, `core/pipeline.py` import
  nothing from `ingestion.realtime` or `providers.noaa.discovery` (verified by
  source inspection; the realtime package is only reachable via the
  `realtime` subcommand).
- **No per-artifact HEAD polling**: discovery is ListObjectsV2-only
  (≤ 7 requests per GFS+GEFS cycle snapshot).
- **No physical store scan per poll**: the committed reader touches only
  catalog tables.
- **No new lock in big-batch**: the only new advisory key is scheduler
  leadership, acquired solely by the `realtime` command.
- **Wave-runner concurrency unchanged**: the only runner change is the
  optional external `cancel_event` (default = previous behavior byte-for-byte).
- **Finalization frequency unchanged**: once per model wave (the runner's
  coalesced finalizer), not once per lead; `publish_settled_lead` per-lead
  visibility is existing behavior.

## 15. Exact test results

Environment: Windows (3.12.13 venv from `poetry.lock` pins) with the
docker-compose service stack (PostgreSQL/PostGIS, Redis, MinIO) running.

| Gate | Command | Result |
|---|---|---|
| Full ingestion suite incl. all realtime + service-backed tests | `WEATHER_TEST_MINIO=1 python -m pytest -W error::ResourceWarning --junitxml=ingestion-junit.xml` | **524 passed, 0 failed** (468 pre-5C + 56 new realtime tests) |
| Realtime suites alone | `pytest tests/test_realtime_{planner,polling,scheduler,committed,cli}.py` | **56 passed** (21 planner / 9 polling / 19 scheduler / 3 committed / 4 CLI) |
| Domain suite + 100% coverage gate | `packages/domain: pytest` | **449 passed**, coverage **100.00%** (2 new leader-key tests) |
| Ingestion ruff 0.3.4 | `ruff check src tests` | clean |
| Ingestion mypy 1.9.0 strict | `mypy` | clean (31 files) |
| Domain ruff + mypy | `ruff check .` / `mypy` | clean (22 files) |

No NOAA live dependency in any test (snapshots are constructed directly or
faked at the snapshot-function boundary). CI-only remainder: none new — the
service-backed tests ran locally this time; Linux CI re-runs the same matrix.

## 16. Unexpected findings / regressions

1. **Async snapshot functions in production wiring** — mypy caught that
   `_discover_production` initially returned coroutines; fixed with
   `asyncio.run` per snapshot call (one event loop per model per poll;
   discovery is an off-hot-path poll concern).
2. **Idle-poll backoff vs jitter test assumption** — my first scheduler test
   asserted an absolute interval bound while idle polls legitimately back off
   to 1800/3600 s; corrected to assert bounds relative to the machine's
   current base interval (which is the actual invariant).
3. **Planner contiguity vs the predecessor guard**: analysis during test
   design showed the contiguous walk + per-model target filtering satisfy the
   predecessor dependency structurally (a lead is only reachable if every
   earlier horizon lead is either committed-for-both or pending). The guard is
   still implemented and unit-tested (`predecessor_satisfied`) as a defensive
   invariant so future policy changes cannot silently violate the dependency.
4. No production regressions: the full 468-test pre-5C suite passes unchanged.

## 17. What remains before Phase 5 can close

- **5E — real-upstream E2E acceptance**: run `weather-ingest realtime --once`
  (and then the loop) against the live AWS buckets through a full GFS+GEFS
  publication window; validate the discovery predicate assumptions live
  (`.idx` timing, listing stability), confirm wave cadence and per-lead
  serving visibility end-to-end, and tune
  `REALTIME_WAVE_MAX_LEADS`/`REALTIME_WAVE_MAX_WAIT_SECONDS` against measured
  publication windows (GFS ≈ 1.7 h, GEFS ≈ 3.5–5.5 h per cycle).
- Operational defaults review: current polling/wave defaults are the Phase 5A
  proposal values; they should be confirmed against the 5E rehearsal before
  deployment (they are configuration, not code).
- Explicitly deferred (already scoped): cycle handover polish beyond
  track/untrack (Phase 6 owns supersession/retention/GC), deployment
  architecture (process supervision is a deployment concern).
