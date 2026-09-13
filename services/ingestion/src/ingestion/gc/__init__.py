"""Garbage collection (GC) and physical forecast store reclamation package.

V3 converged model: the reclamation planner + worker are the only physical
deletion authorities at (variable, valid_time) granularity; the finalizer
module is lifecycle bookkeeping only (derived tombstones, zero physical
deletion). The legacy V2 whole-cycle GC engine has been retired.
"""

from ingestion.gc.finalizer import (
    FinalizerCandidate,
    FinalizerPassResult,
    claim_fresh_candidate,
    cycle_reclamation_units_terminal,
    enumerate_cycle_store_paths,
    finalize_cycle_bookkeeping,
    finalize_cycle_physical_and_queue,
    run_lifecycle_bookkeeping_pass,
)
from ingestion.gc.planner import ReclamationPlanResult, plan_reclamation_pass
from ingestion.gc.sweeper import (
    DEFAULT_SWEEPER_BATCH_SIZE,
    CyclePurgeResult,
    SweeperCandidate,
    SweeperPassResult,
    discover_sweeper_candidates,
    purge_cycle_metadata,
    run_metadata_sweeper_pass,
)
from ingestion.gc.worker import (
    ReclamationWorkerResult,
    requeue_failed_reclamation_targets,
    run_reclamation_worker_pass,
)

__all__ = [
    "DEFAULT_SWEEPER_BATCH_SIZE",
    "CyclePurgeResult",
    "FinalizerCandidate",
    "FinalizerPassResult",
    "ReclamationPlanResult",
    "ReclamationWorkerResult",
    "SweeperCandidate",
    "SweeperPassResult",
    "claim_fresh_candidate",
    "cycle_reclamation_units_terminal",
    "discover_sweeper_candidates",
    "enumerate_cycle_store_paths",
    "finalize_cycle_bookkeeping",
    "finalize_cycle_physical_and_queue",
    "plan_reclamation_pass",
    "purge_cycle_metadata",
    "requeue_failed_reclamation_targets",
    "run_lifecycle_bookkeeping_pass",
    "run_metadata_sweeper_pass",
    "run_reclamation_worker_pass",
]
