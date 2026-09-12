"""Garbage collection (GC) and physical forecast store reclamation package."""

from ingestion.gc.finalizer import (
    FinalizerCandidate,
    FinalizerPassResult,
    claim_fresh_candidate,
    enumerate_cycle_store_paths,
    finalize_cycle_eol,
    finalize_cycle_physical_and_queue,
    run_finalizer_pass,
)
from ingestion.gc.planner import ReclamationPlanResult, plan_reclamation_pass
from ingestion.gc.reconciler import GcCandidateInfo, GcPassResult, run_gc_pass
from ingestion.gc.worker import (
    ReclamationWorkerResult,
    requeue_failed_reclamation_targets,
    run_reclamation_worker_pass,
)

__all__ = [
    "FinalizerCandidate",
    "FinalizerPassResult",
    "GcCandidateInfo",
    "GcPassResult",
    "ReclamationPlanResult",
    "ReclamationWorkerResult",
    "claim_fresh_candidate",
    "enumerate_cycle_store_paths",
    "finalize_cycle_eol",
    "finalize_cycle_physical_and_queue",
    "plan_reclamation_pass",
    "requeue_failed_reclamation_targets",
    "run_finalizer_pass",
    "run_gc_pass",
    "run_reclamation_worker_pass",
]

