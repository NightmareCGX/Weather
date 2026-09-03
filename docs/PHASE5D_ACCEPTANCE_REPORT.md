# Phase 5D Failure / Concurrency Hardening Report

**Phase 5D — Failure / Concurrency Hardening Acceptance.**
This report documents the coverage audit, failure and concurrency validation,
remediation of identified hardening gaps, test results, and final acceptance verdict
for Phase 5 realtime lead-wave ingestion.

---

## A. Repository baseline

- **Branch:** `main`
- **HEAD commit:** `f65b64d` (*Merge pull request #41 from NightmareCGX/feat/phase-5c-realtime-planner-scheduler*)
- **Phase 5 reports inspected:**
  - `docs/PHASE5A_INVESTIGATION_REPORT.md` (Repository reality & architecture investigation)
  - `docs/PHASE5B_IMPLEMENTATION_REPORT.md` (Wave-target/cycle-horizon split + NOAA discovery)
  - `docs/PHASE5C_IMPLEMENTATION_REPORT.md` (Realtime planner + shared barrier + bounded lead-wave scheduler)
- **Key implementation modules inspected:**
  - `services/ingestion/src/ingestion/realtime/planner.py`
  - `services/ingestion/src/ingestion/realtime/polling.py`
  - `services/ingestion/src/ingestion/realtime/committed.py`
  - `services/ingestion/src/ingestion/realtime/leadership.py`
  - `services/ingestion/src/ingestion/realtime/scheduler.py`
  - `services/ingestion/src/ingestion/core/wave_runner.py`
  - `services/ingestion/src/ingestion/providers/noaa/discovery.py`
  - `services/ingestion/src/ingestion/core/coordinator.py`
  - `services/ingestion/src/ingestion/core/catalog.py`
  - `services/ingestion/src/ingestion/core/config.py`
  - `services/ingestion/src/ingestion/cli.py`
  - `packages/domain/src/domain/horizon.py`
  - `packages/domain/src/domain/coverage.py`
  - `packages/domain/src/domain/locks.py`

---

## B. Coverage matrix (Scenarios A–O)

| Scenario | Description | Pre-5D Coverage | Test(s) Validating | Result | Production Change Required? |
|---|---|---|---|---|---|
| **A** | **Discovery failures** (partial failure, transport error, 5xx, invalid XML, mid-list pagination error, multi-iteration recovery) | Partially covered (`test_noaa_discovery.py`, `test_realtime_scheduler.py`) | `test_phase5d_hardening.py::test_scenario_a_discovery_failures_retry_and_recover` | PASS | No |
| **B** | **Catalog / committed-state failures** (PG unavailable, state read fails, planning skipped, recovery on next poll) | Partially covered (`test_realtime_scheduler.py`) | `test_phase5d_hardening.py::test_scenario_b_catalog_read_failure_skips_planning_and_recovers` | PASS | No |
| **C** | **One-model wave failure & asymmetric recovery** (GFS ok / GEFS fails; GEFS ok / GFS fails; retrying only missing work) | Partially covered (GFS-ok/GEFS-fail in 5C) | `test_phase5d_hardening.py::test_scenario_c_gefs_succeeds_gfs_fails_retries_only_gfs`, `test_realtime_scheduler.py::test_gfs_success_gefs_failure_retries_only_gefs` | PASS | No |
| **D** | **Partial region progress within one model wave** (e.g. GEFS members 1..25 committed, 26..30 fail; lead remains pending and retried) | Covered indirectly (`test_incremental_wave.py`, `test_coordinator.py`) | `test_phase5d_hardening.py::test_scenario_d_partial_gefs_members_leave_lead_pending` | PASS | No |
| **E** | **Big-batch / realtime overlap** (realtime plans but big-batch commits before dispatch; concurrent execution on same store) | Covered indirectly (`test_coordinator.py`, `test_locks.py`) | `test_phase5d_hardening.py::test_scenario_e_big_batch_commit_ahead_of_realtime_reconciles`, `test_coordinator.py` | PASS | No |
| **F** | **Duplicate realtime schedulers** (PostgreSQL advisory leader lock prevents duplicate orchestration; second instance passivates) | Covered with fakes in 5C | `test_phase5d_hardening.py::test_scenario_f_duplicate_scheduler_exclusion_postgres`, `test_realtime_scheduler.py::test_double_start_second_instance_exits_passively` | PASS | No |
| **G** | **Leadership connection loss** (dedicated connection severed, lock released server-side, detection & wave refusal) | Covered but behavior was incomplete | `test_phase5d_hardening.py::test_scenario_g_leadership_connection_loss_detection_and_safe_exit`, `test_scenario_g_leadership_reacquire_loop_on_connection_death` | PASS | **Yes** (`SchedulerLeadership.check_leadership()`, `RealtimeScheduler._is_leader_active()`, `poll_once` leadership gate) |
| **H** | **Shutdown while polling** (SIGTERM/stop across ACTIVE, PUBLISHING, BACKOFF, discovery retry wait states) | Covered indirectly in 5C | `test_phase5d_hardening.py::test_scenario_h_shutdown_while_polling_all_states`, `test_realtime_scheduler.py::test_graceful_shutdown_while_sleeping_is_prompt` | PASS | No |
| **I** | **Shutdown during active waves** (GFS active drain; shutdown between GFS completion and GEFS start) | Covered but behavior was incomplete | `test_phase5d_hardening.py::test_scenario_i_shutdown_during_gfs_wave_skips_gefs`, `test_realtime_scheduler.py::test_shutdown_during_wave_triggers_non_abandoning_cancel` | PASS | **Yes** (Check `self._stop_event.is_set() or cancel_event.is_set()` before dispatching each model in the wave loop) |
| **J** | **Exception containment** (recoverable vs fatal error classification, bounded error intervals) | Already covered | `test_realtime_scheduler.py`, `test_phase5d_hardening.py` | PASS | No |
| **K** | **Tight-loop audit** (prove no retry/error path runs at CPU speed) | Already covered | Verified by code inspection & timing assertions across all polling/retry states | PASS | No |
| **L** | **Cycle pairing correctness** (staggered publication across cycle timestamps; no cross-timestamp pairing) | Already covered | `test_phase5d_hardening.py::test_scenario_l_m_staggered_cycle_handover_and_no_mismatched_pairing`, `test_realtime_scheduler.py` | PASS | No |
| **M** | **Cycle handover boundary** (tracking newest eligible cycle with artifacts; older cycle left to Phase 6 retention) | Already covered | `test_phase5d_hardening.py::test_scenario_l_m_staggered_cycle_handover_and_no_mismatched_pairing` | PASS | No |
| **N** | **Finalizer / gate contention** (repeated multi-wave finalization without gate timeouts, starvation, or catalog deadlock) | Covered indirectly (`test_finalization_benchmark.py`, `test_coordinator.py`) | `test_phase5d_hardening.py::test_scenario_n_o_repeated_wave_finalization_resource_clean` | PASS | No |
| **O** | **Resource hygiene** (no leaked DB sessions, unclosed HTTP clients, or ResourceWarnings under `-W error::ResourceWarning`) | Already covered | Full suite execution under `-W error::ResourceWarning` | PASS | No |

---

## C. Production changes

Two narrow, targeted hardening changes were made to production code:

### 1. Active leadership health verification on PostgreSQL (`services/ingestion/src/ingestion/realtime/leadership.py`)
- **Problem identified:** `SchedulerLeadership` acquired the session-level advisory lock on startup and set `self._held = True`. If the underlying PostgreSQL connection was dropped, terminated server-side, or closed, `self._held` remained `True` in Python memory. The scheduler could continue believing it held leadership even after the lock was released server-side.
- **Fix applied:**
  - Added `SchedulerLeadership.check_leadership() -> bool`, which executes a fast query on the dedicated connection against `pg_locks` (`SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid() AND ((classid::bigint << 32) | (objid::bigint & 4294967295)) = :key`).
  - If the connection died or was invalidated, `check_leadership()` marks `_held = False`, invalidates the connection, and returns `False`.
  - Added `check_leadership()` to `NoopLeadership` for test compatibility.
  - In `RealtimeScheduler`:
    - Added `_is_leader_active() -> bool` helper.
    - In `poll_once()`, if leadership is configured and `not self._is_leader_active()`, the scheduler logs a warning, skips planning and dispatch, and returns `PollOutcome(kind="leadership-lost")`.
    - In `run()`, before each iteration, if leadership was lost, the scheduler attempts to reacquire on a fresh connection. If re-acquisition fails (e.g. another instance acquired it), it cleanly exits with code 0 without doing duplicate work.

### 2. Immediate shutdown check between per-model wave dispatches (`services/ingestion/src/ingestion/realtime/scheduler.py`)
- **Problem identified:** In `poll_once()`, wave dispatch iterates over `("gfs", ...)` and `("gefs", ...)`. If shutdown was requested while the GFS wave was running, GFS drained and finalized cleanly. However, the loop did not check whether stop was requested before proceeding to dispatch the GEFS wave, causing an unnecessary startup of the second wave under cancellation.
- **Fix applied:**
  - Added a check `if self._stop_event.is_set() or cancel_event.is_set(): break` before dispatching each model in the wave candidate loop.
  - If GFS finishes and stop was requested, GEFS is not dispatched. GFS remains committed in durable state. The next scheduler invocation reads committed state and resumes GEFS without data loss or rollback.

---

## D. Failure semantics

- **Discovery failure:** Returns `PollOutcome(kind="discovery-failed")`. The last good snapshot baseline is preserved for diffing; poll state machine is untouched (does not falsely enter BACKOFF); the loop retries after `REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS` (60.0 s).
- **Catalog failure:** Returns `PollOutcome(kind="state-read-failed")`. Planning is skipped entirely to prevent scheduling waves against assumed empty committed state; the loop retries on the failure retry interval (60.0 s).
- **Partial model success:** GFS and GEFS waves are dispatched sequentially. If one model wave succeeds and the other fails, the successful model's commits and catalog records remain intact. The next poll reads durable state, excludes already-committed leads for the successful model, and retries only the missing model's work.
- **Partial region failure within a wave:** If some members/regions commit before a failure, they are committed to store shards with `COMPLETE` markers. Finalization reconciles catalog to store. For GEFS, `is_lead_committed` requires all 30 perturbation members; therefore, a lead with partial member coverage remains uncommitted and will be retried on the next wave.
- **Planner failure:** `plan_wave` is a pure function. Invalid inputs (e.g., negative bounds) fail fast with standard exceptions (`ValueError`).
- **Unexpected scheduler failure:** Any unhandled exception during leadership acquisition or cycle selection fails loudly with non-zero exit code or logs a clean warning and terminates.

---

## E. Concurrency semantics

- **Big-batch / realtime overlap:** Both big-batch and realtime use the identical `RunCoordinator` and `StoreLockCoordinator` protocol. Store gate (`store_gate_key`), admission turnstile (`admission_key`), and physical region locks (`physical_conflict_identity`) serialize all store operations. Realtime discovers any big-batch commits via catalog state on the next poll and omits them from wave candidates.
- **Duplicate schedulers:** Deployment-wide leadership is governed by `domain.locks.scheduler_leader_key()`. A secondary instance fails non-blocking `pg_try_advisory_lock` on startup, logs a clear warning, and exits with code 0 without performing any discovery or wave dispatch.
- **Leadership release and reacquisition:** When the leader process exits or its session closes, PostgreSQL releases the session-level advisory lock automatically. A secondary instance or restarted scheduler can immediately acquire leadership.
- **Leadership connection loss:** Detected via active `pg_locks` check on the dedicated connection. If lost, wave dispatch is refused. The scheduler attempts clean re-acquisition on a fresh connection, or exits cleanly if another instance holds the lock.
- **Same-store locking:** Big-batch and realtime share identical storage identities and lock namespaces. EXCLUSIVE phases (init/finalize) serialize with SHARED region writes.
- **GFS and GEFS independent progress:** GFS and GEFS stores are independent. The shared barrier enforces eligibility at the planning level only. Stores commit, reconcile, and finalize independently.

---

## F. Shutdown behavior

- **Shutdown during polling / sleeping:** `RealtimeScheduler.request_stop()` sets `self._stop_event`, prompting the sleep wait to return immediately across all polling states (ACTIVE 600s, PUBLISHING 120s, BACKOFF 1800s–3600s, discovery failure retry 60s).
- **Shutdown during GFS wave:** Triggers `cancel_event` passed to `wave_runner._run_wave`. Running workers drain non-abandoningly, pending tasks cancel, and finalizer runs to reconcile committed work.
- **Shutdown during GEFS wave:** Same non-abandoning drain and finalization.
- **Shutdown between model waves:** The dispatch loop checks `self._stop_event.is_set() or cancel_event.is_set()` and skips subsequent model waves.
- **Resource cleanup:** Dedicated leadership connections, database sessions, and HTTP clients are closed cleanly. `request_stop()` is idempotent and signal-safe.

---

## G. Tight-loop audit

| Error / Retry Path | Mechanism | Configured Base Delay | Backoff / Termination |
|---|---|---|---|
| Discovery failure (transport / 5xx / invalid XML / pagination) | Dedicated retry sleep | `REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS` (60.0 s) | Retries every 60s without entering idle backoff |
| Committed-state read failure (database connection down) | Dedicated retry sleep | `REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS` (60.0 s) | Retries every 60s without planning |
| Wave dispatch failure (per-file / network / decode error) | Poll state machine | `REALTIME_PUBLICATION_POLL_SECONDS` (120.0 s) ± 10% jitter | Retries on next poll interval |
| Unchanged cycle snapshot (idle) | Poll state machine | `REALTIME_IDLE_BACKOFF_INITIAL_SECONDS` (1800.0 s) | Progressively doubles to `REALTIME_IDLE_BACKOFF_MAX_SECONDS` (3600.0 s) |
| Active publication activity | Poll state machine | `REALTIME_PUBLICATION_POLL_SECONDS` (120.0 s) ± 10% jitter | Fast cadence while upstream is publishing |
| Normal active cycle polling | Poll state machine | `REALTIME_ACTIVE_POLL_SECONDS` (600.0 s) ± 10% jitter | Normal tracking cadence |
| Leadership held by another instance | Acquisition failure | 0 s | Clean exit with code 0 (passivates) |
| Leadership lost mid-run | Connection check failure | Immediate reacquire attempt | Exits with code 0 if lock cannot be reacquired |

**Conclusion:** No error or retry path can execute in a tight CPU loop.

---

## H. Cycle pairing and handover boundary

- **Phase 5 Correctness Requirement (Enforced):**
  - GFS and GEFS snapshots are always fetched for the identical `CycleIdentity(cycle_date, cycle_hour)`.
  - Planning combines only snapshots sharing the same cycle timestamp. Cross-timestamp pairing (e.g. GFS 06Z + GEFS 00Z) is structurally impossible.
  - When a newer cycle publishes (e.g. 06Z), the scheduler tracks 06Z as soon as artifacts appear and resets wait timers.
- **Phase 6 Lifecycle Responsibility (Deferred):**
  - Older cycle stores (e.g. 00Z) remain in storage and catalog for serving.
  - Cycle supersession, retirement, retention windows, and crash-safe garbage collection belong to Phase 6.

---

## I. Finalizer / store-gate validation

- Measured finalizer behavior under repeated incremental waves:
  - Finalizer executes once per model wave (coalesced), reading O(committed regions) marker evidence.
  - State fingerprints suppress redundant manifest writes when no new regions are added.
  - Tested with repeated sequential waves (leads 0, 3, 6, ...) without gate timeouts (`ADVISORY_LOCK_TIMEOUT_SECONDS = 30.0s`), reader starvation, or catalog deadlock.
- No finalizer or store-gate regression was observed.

---

## J. Resource hygiene

- Executed full test suite under `-W error::ResourceWarning`.
- Verified clean resource cleanup:
  - All `httpx.AsyncClient` instances in `discovery.py` are scoped with `async with` context managers.
  - SQLAlchemy sessions and connections are closed or invalidated on exit.
  - Dedicated leadership connection is released and closed in `finally` blocks.
  - Async event loops run synchronously per off-hot-path task and terminate cleanly.
  - Zero Phase 5-introduced `ResourceWarning` or socket leaks.

---

## K. Exact test results

Environment: Windows (`win32`, Python 3.12.13) with local PostgreSQL 16 (PostGIS), Redis 7, and MinIO service containers active.

| Suite / Gate | Exact Command | Result |
|---|---|---|
| **Domain Unit & Contract Tests** | `poetry run pytest packages/domain/tests` | **454 passed**, **100.00% coverage** |
| **New Phase 5D Hardening Tests** | `poetry run pytest services/ingestion/tests/test_phase5d_hardening.py -v` | **17 passed** |
| **All Realtime & Phase 5D Suites** | `poetry run pytest services/ingestion/tests/test_realtime_*.py services/ingestion/tests/test_phase5d_hardening.py` | **73 passed** |
| **Full Ingestion Test Suite (MinIO + PG + ResourceWarning)** | `WEATHER_TEST_MINIO=1 poetry run pytest services/ingestion/tests -q -W error::ResourceWarning` | **541 passed, 0 failed, 0 skipped** |
| **API Test Suite** | `poetry run pytest services/api/tests -q` | **327 passed** |
| **Frontend Unit Tests** | `cd services/frontend && npm test -- --watchAll=false` | **227 passed** |
| **Python Quality (Ruff)** | `poetry run ruff check packages/domain services/ingestion services/api` | **All checks passed** |
| **Python Quality (Strict MyPy)** | `mypy` in `packages/domain` (22 files), `services/ingestion` (31 files), `services/api` (36 files) | **Success: no issues found** |
| **Frontend Typecheck, Lint, Format** | `npm run typecheck && npm run lint && npm run format:check` | **All checks passed** |

---

## L. Regressions found & remediated

### Regression 1: Leadership connection loss undetected in memory
- **Symptom:** If the PostgreSQL connection holding the scheduler advisory lock died, `SchedulerLeadership._held` remained `True`, allowing the scheduler to believe it was still leader.
- **Root cause:** Leadership state was stored only as a local boolean flag set at `acquire()` time.
- **Fix:** Implemented active connection validation (`check_leadership()`) querying `pg_locks` on the dedicated connection; `poll_once` refuses planning/dispatch when leadership is lost; `run()` handles clean re-acquisition or graceful exit.
- **Test:** `services/ingestion/tests/test_phase5d_hardening.py::test_scenario_g_leadership_connection_loss_detection_and_safe_exit`.
- **Validation:** Verified with service-backed PostgreSQL container by actively terminating the backend connection (`pg_terminate_backend`).

### Regression 2: Shutdown between sequential model waves in a candidate
- **Symptom:** If stop was requested while GFS wave was in flight, GFS drained and finalized cleanly, but the dispatch loop proceeded to invoke GEFS wave.
- **Root cause:** Missing cancellation check between loop iterations in `RealtimeScheduler.poll_once()`.
- **Fix:** Added `if self._stop_event.is_set() or cancel_event.is_set(): break` before dispatching each model in the wave candidate.
- **Test:** `services/ingestion/tests/test_phase5d_hardening.py::test_scenario_i_shutdown_during_gfs_wave_skips_gefs`.
- **Validation:** Verified GFS dispatches and GEFS is safely skipped upon stop request.

---

## M. Phase 5D verdict

```text
================================================================================
Phase 5D ACCEPTED — ready for Phase 5E real-upstream validation
================================================================================
```

All 14 acceptance criteria are fully met:
1. Discovery failures cannot masquerade as idle.
2. Committed-state read failures cannot cause duplicate planning.
3. GFS-success/GEFS-failure and GEFS-success/GFS-failure reconcile correctly and retry only missing work.
4. Partial-region failures remain safely retryable.
5. Big-batch/realtime overlap cannot corrupt stores.
6. Duplicate schedulers do not duplicate orchestration.
7. Leadership connection loss cannot leave a false-active scheduler dispatching waves.
8. Shutdown is safe during polling and active waves.
9. No retry/error path can form a tight loop.
10. GFS/GEFS cycle identities cannot be mismatched.
11. No concrete finalizer/store-gate regression is observed.
12. Phase 5 introduces no resource leak.
13. No Phase 6 feature is required for Phase 5 correctness.
14. All quality gates pass across domain, api, ingestion, and frontend.
