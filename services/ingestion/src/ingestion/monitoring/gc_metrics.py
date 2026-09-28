"""In-process Prometheus metrics for the GC pipeline stages (Lifecycle V3).

These metrics capture "what happened during each GC pass" — stage wall-clock
durations, per-stage success, and the planner/worker/sweeper result counters —
complementing the DB-state probe metrics (``weather_reclamation_queue_count*``
from ``lifecycle_collector``) which describe "what the queue looks like right
now". They are strictly process-local to the GC daemon that executes the
passes; the standalone exporter (9112) never holds them. To make them
scrapable, the gc daemon serves its own live registry via
``weather-ingest gc --metrics-port`` (same pattern as the realtime daemon).

Metric semantics and source fields (all aggregate per pass; planner/worker
results carry no per-model split, so no ``model`` label is exposed):

- ``weather_gc_pass_duration_seconds`` (histogram, stage): wall-clock of each
  stage per pass. Stages: bookkeeping / planner / worker / sweeper / inventory.
- ``weather_gc_pass_success`` (gauge, stage): 1 if the stage completed in the
  latest pass, 0 if it raised (stages are failure-isolated; a stage error never
  kills the pass).
- ``weather_gc_pass_last_success_timestamp`` (gauge, stage): Unix time of the
  stage's last successful execution.
- ``weather_gc_planner_enqueued_total`` (counter): cumulative shard targets
  enqueued (``ReclamationPlanResult.enqueued_count``) — a true per-pass
  increment, so it sums.
- ``weather_gc_planner_reclaimable_shards`` (gauge): the LATEST pass's
  reclaimable shard count (``ReclamationPlanResult.reclaimable_shards``). The
  planner recomputes this absolute set every pass, so it is a state reading and
  is deliberately not a counter: the retired
  ``weather_gc_planner_reclaimable_total`` counter added the whole queue size on
  every pass and was read in Grafana as a rising "reclaim pressure" line that
  measured only the poll rate.
- ``weather_gc_worker_claimed_total`` / ``weather_gc_worker_deleted_total`` /
  ``weather_gc_worker_failed_total`` (counters): worker pass outcomes
  (``ReclamationWorkerResult.claimed_count`` / ``deleted_count`` /
  ``failed_count``).
- ``weather_gc_worker_markers_cleaned_total`` (counter): region commit markers
  cleaned (``ReclamationWorkerResult.markers_cleaned_count``).
- ``weather_gc_sweeper_swept_total`` / ``weather_gc_sweeper_failed_total``
  (counters): sweeper pass outcomes (``SweeperPassResult.swept_cycles`` /
  ``failed_cycles``).
- ``weather_gc_purge_deleted_rows_total`` (counter) and
  ``weather_gc_purge_oldest_remaining_age_seconds`` (gauge): scheduled
  terminal-row purge outcomes (``QueuePurgeResult.deleted_rows`` and the age of
  the oldest surviving terminal row; -1 when none remains).
- ``weather_gc_inventory_orphans`` (gauge, beyond_frontier): orphan store
  count from the latest orphan inventory pass (store <-> catalog
  reconciliation, architecture doc section 9).
- ``weather_gc_inventory_last_success_timestamp`` (gauge): Unix time of the
  last successful inventory pass.
- ``weather_gc_inventory_errors_total`` (counter): inventory pass failures
  (fail-open; the GC loop continues after an inventory error).

Style reference: the exporter's fail-open ``COLLECTOR_SUCCESS`` pattern in
``ingestion/monitoring/exporter.py``.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from ingestion.monitoring.metrics import REGISTRY, Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Per-stage pass observability
# ---------------------------------------------------------------------------

GC_PASS_DURATION: Histogram = REGISTRY.histogram(
    "weather_gc_pass_duration_seconds",
    "Wall-clock duration of each GC pipeline stage per pass",
    labelnames=("stage",),
)

GC_PASS_SUCCESS: Gauge = REGISTRY.gauge(
    "weather_gc_pass_success",
    "1 if the named GC stage completed successfully during the latest pass, "
    "0 if it raised (stage errors are failure-isolated)",
    labelnames=("stage",),
)

GC_PASS_LAST_SUCCESS_TIMESTAMP: Gauge = REGISTRY.gauge(
    "weather_gc_pass_last_success_timestamp",
    "Unix timestamp of the last successful execution of the named GC stage",
    labelnames=("stage",),
)

# ---------------------------------------------------------------------------
# Stage outcome counters
# ---------------------------------------------------------------------------

GC_PLANNER_ENQUEUED_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_planner_enqueued_total",
    "Cumulative reclamation shard targets enqueued by the planner stage "
    "(ReclamationPlanResult.enqueued_count)",
)

GC_PLANNER_RECLAIMABLE_SHARDS: Gauge = REGISTRY.gauge(
    "weather_gc_planner_reclaimable_shards",
    "Reclaimable shards in the latest planner pass "
    "(ReclamationPlanResult.reclaimable_shards). This is an absolute queue "
    "state, not a per-pass increment: the planner recomputes the full "
    "reclaimable set every pass, so it cannot be summed and must not be "
    "exposed as a counter.",
)

GC_WORKER_CLAIMED_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_worker_claimed_total",
    "Cumulative shard targets claimed by the reclamation worker stage "
    "(ReclamationWorkerResult.claimed_count)",
)

GC_WORKER_DELETED_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_worker_deleted_total",
    "Cumulative shard targets physically deleted by the worker stage "
    "(ReclamationWorkerResult.deleted_count)",
)

GC_WORKER_FAILED_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_worker_failed_total",
    "Cumulative shard targets that ended in failed quarantine during worker "
    "passes (ReclamationWorkerResult.failed_count)",
)

GC_WORKER_MARKERS_CLEANED_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_worker_markers_cleaned_total",
    "Cumulative region commit markers cleaned by the worker stage "
    "(ReclamationWorkerResult.markers_cleaned_count)",
)

GC_SWEEPER_SWEPT_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_sweeper_swept_total",
    "Cumulative cycles swept by the metadata retention sweeper "
    "(SweeperPassResult.swept_cycles)",
)

GC_SWEEPER_FAILED_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_sweeper_failed_total",
    "Cumulative cycles that failed the metadata sweeper pass "
    "(SweeperPassResult.failed_cycles)",
)

GC_PURGE_DELETED_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_purge_deleted_rows_total",
    "Cumulative terminal (deleted) reclamation_queue rows removed by the "
    "scheduled queue-purge stage (QueuePurgeResult.deleted_rows)",
)

GC_PURGE_OLDEST_REMAINING_AGE_SECONDS: Gauge = REGISTRY.gauge(
    "weather_gc_purge_oldest_remaining_age_seconds",
    "Age of the oldest terminal reclamation_queue row still present after the "
    "latest queue-purge pass; -1 when no terminal row remains. A rising value "
    "means the purge is not keeping up with terminal-row production.",
)

# ---------------------------------------------------------------------------
# Orphan inventory (store <-> catalog reconciliation, architecture doc §9)
# ---------------------------------------------------------------------------

GC_INVENTORY_ORPHANS: Gauge = REGISTRY.gauge(
    "weather_gc_inventory_orphans",
    "Orphan store count from the latest orphan inventory pass, split by "
    "whether the cycle is beyond the recoverability frontier (never "
    "re-activatable) or still within it",
    labelnames=("beyond_frontier",),
)

GC_INVENTORY_LAST_SUCCESS_TIMESTAMP: Gauge = REGISTRY.gauge(
    "weather_gc_inventory_last_success_timestamp",
    "Unix timestamp of the last successful orphan inventory pass",
)

GC_INVENTORY_ERRORS_TOTAL: Counter = REGISTRY.counter(
    "weather_gc_inventory_errors_total",
    "Cumulative orphan inventory pass failures (fail-open; the GC loop "
    "continues after an inventory error)",
)


@contextmanager
def gc_stage_timer(stage: str) -> Iterator[None]:
    """Time one GC stage and record its success/failure gauges.

    Usage: wrap the stage body; on exception the failure is recorded and the
    exception re-raised so the caller keeps its existing failure-isolation
    (log + summary marker + continue) behavior.
    """
    started = time.perf_counter()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        GC_PASS_DURATION.labels(stage=stage).observe(time.perf_counter() - started)
        GC_PASS_SUCCESS.labels(stage=stage).set(1.0 if ok else 0.0)
        if ok:
            GC_PASS_LAST_SUCCESS_TIMESTAMP.labels(stage=stage).set_to_current_time()


def record_planner_pass(*, enqueued_count: int, reclaimable_shards: int) -> None:
    """Record one planner pass: a cumulative enqueue counter and a state gauge.

    ``enqueued_count`` is a genuine per-pass increment (rows the planner newly
    queued), so it accumulates. ``reclaimable_shards`` is the pass's absolute
    reclaimable set, recomputed from scratch every pass, so it is published as
    a gauge — summing it (the previous ``weather_gc_planner_reclaimable_total``
    counter) made the series grow by the entire queue size every pass and
    measured nothing.
    """
    GC_PLANNER_ENQUEUED_TOTAL.inc(max(0, int(enqueued_count)))
    GC_PLANNER_RECLAIMABLE_SHARDS.set(max(0, int(reclaimable_shards)))


def record_worker_pass(
    *,
    claimed_count: int,
    deleted_count: int,
    failed_count: int,
    markers_cleaned_count: int,
) -> None:
    """Increment worker-stage counters from one ``ReclamationWorkerResult``."""
    GC_WORKER_CLAIMED_TOTAL.inc(max(0, int(claimed_count)))
    GC_WORKER_DELETED_TOTAL.inc(max(0, int(deleted_count)))
    GC_WORKER_FAILED_TOTAL.inc(max(0, int(failed_count)))
    GC_WORKER_MARKERS_CLEANED_TOTAL.inc(max(0, int(markers_cleaned_count)))
    # Alert snapshot tracks the LATEST pass (not the cumulative counter) so a
    # recovered failure lets the alert fire a recovery event.
    _set_alert_state("worker_failed_total", max(0, int(failed_count)))


def record_sweeper_pass(*, swept_cycles: int, failed_cycles: int) -> None:
    """Increment sweeper-stage counters from one ``SweeperPassResult``."""
    GC_SWEEPER_SWEPT_TOTAL.inc(max(0, int(swept_cycles)))
    GC_SWEEPER_FAILED_TOTAL.inc(max(0, int(failed_cycles)))
    _set_alert_state("sweeper_failed_total", max(0, int(failed_cycles)))


def record_purge_pass(
    *,
    deleted_rows: int,
    oldest_remaining: datetime | None,
    now: datetime | None = None,
) -> None:
    """Record one scheduled queue-purge pass.

    ``oldest_remaining`` is aged against the wall clock rather than reported as
    a raw timestamp so the gauge is directly comparable to the configured
    retention window: a value far above ``RECLAMATION_PURGE_RETENTION_DAYS``
    means terminal rows are accumulating faster than the stage removes them.
    """
    GC_PURGE_DELETED_TOTAL.inc(max(0, int(deleted_rows)))
    if oldest_remaining is None:
        GC_PURGE_OLDEST_REMAINING_AGE_SECONDS.set(-1.0)
        return
    reference = now if now is not None else datetime.now(timezone.utc)
    GC_PURGE_OLDEST_REMAINING_AGE_SECONDS.set(
        max(0.0, (reference - oldest_remaining).total_seconds())
    )


def record_inventory_pass(
    *, orphan_stores_beyond_frontier: int, orphan_stores_within_frontier: int, errors: int
) -> None:
    """Publish one successful orphan inventory pass's gauges/counters."""
    GC_INVENTORY_ORPHANS.labels(beyond_frontier="true").set(
        max(0, int(orphan_stores_beyond_frontier))
    )
    GC_INVENTORY_ORPHANS.labels(beyond_frontier="false").set(
        max(0, int(orphan_stores_within_frontier))
    )
    GC_INVENTORY_LAST_SUCCESS_TIMESTAMP.set(time.time())
    if errors > 0:
        GC_INVENTORY_ERRORS_TOTAL.inc(errors)
    _set_alert_state("inventory_orphans_within_frontier", int(orphan_stores_within_frontier))


def record_inventory_failure() -> None:
    """Count one failed (raised) orphan inventory pass."""
    GC_INVENTORY_ERRORS_TOTAL.inc()


# ---------------------------------------------------------------------------
# Alert-engine snapshot
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_state: dict[str, int] = {
    "worker_failed_total": 0,
    "sweeper_failed_total": 0,
    "inventory_orphans_within_frontier": 0,
}


def _set_alert_state(key: str, value: int) -> None:
    with _state_lock:
        _state[key] = value


def snapshot_alert_state() -> dict[str, int]:
    """Snapshot in-process GC pass counters for the alert engine.

    The alert engine's rule evaluation is synchronous and pull-based; the GC
    daemon hands it this snapshot after each pass (see the gc rules in
    ``alerts.py``). The ``*_total`` values here reflect the LATEST pass (not
    the cumulative metric counters) so a recovered failure lets the matching
    alert fire a recovery event.
    """
    with _state_lock:
        return dict(_state)
