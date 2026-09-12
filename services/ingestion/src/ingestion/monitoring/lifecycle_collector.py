"""Lifecycle, Finalizer, Sweeper, Reclamation, and Anti-Resurrection health collector.

Observes the authoritative Data Lifecycle V3 contracts:
- Physical cycle lifecycle (active -> deletion_started_at -> deleted_at)
- Finalizer progress and stuck deletion claims
- 14-day detailed metadata sweeper backlog and overdue metadata
- Granular reclamation queue states (queued, deleting, deleted, failed)
- Lifecycle invariants and permanent anti-resurrection tombstone enforcement
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from ingestion.monitoring.metrics import REGISTRY

logger = logging.getLogger(__name__)

# Register Prometheus metrics
LIFECYCLE_ACTIVE_CYCLES = REGISTRY.gauge(
    "weather_lifecycle_active_cycles",
    "Number of active cycles not yet claimed for deletion",
)
LIFECYCLE_CLAIMED_CYCLES = REGISTRY.gauge(
    "weather_lifecycle_claimed_cycles",
    "Number of cycles claimed (deletion_started_at set) but not yet finalized",
)
LIFECYCLE_TOMBSTONE_CYCLES = REGISTRY.gauge(
    "weather_lifecycle_tombstone_cycles",
    "Number of permanent anti-resurrection tombstones (deleted_at set)",
)
LIFECYCLE_OLDEST_CLAIM_AGE_SECONDS = REGISTRY.gauge(
    "weather_lifecycle_oldest_claim_age_seconds",
    "Age in seconds of the oldest active deletion claim",
)
LIFECYCLE_STUCK_CLAIMS_WARNING = REGISTRY.gauge(
    "weather_lifecycle_stuck_claims_warning",
    "Count of deletion claims older than 1 hour",
)
LIFECYCLE_STUCK_CLAIMS_CRITICAL = REGISTRY.gauge(
    "weather_lifecycle_stuck_claims_critical",
    "Count of deletion claims older than 4 hours",
)

METADATA_SWEEPER_ELIGIBLE = REGISTRY.gauge(
    "weather_metadata_sweeper_eligible_tombstones",
    "Number of tombstones older than 14-day retention window",
)
METADATA_SWEEPER_UNPURGED = REGISTRY.gauge(
    "weather_metadata_sweeper_unpurged_metadata_count",
    "Tombstones older than 14 days that still retain detailed model_runs metadata",
)
METADATA_SWEEPER_OLDEST_OVERDUE_SECONDS = REGISTRY.gauge(
    "weather_metadata_sweeper_oldest_overdue_seconds",
    "Age in seconds of the oldest unpurged metadata past the 14-day deadline",
)

RECLAMATION_QUEUE_COUNT = REGISTRY.gauge(
    "weather_reclamation_queue_count",
    "Granular reclamation queue row count by state",
    labelnames=("status",),
)
RECLAMATION_OLDEST_QUEUED_AGE_SECONDS = REGISTRY.gauge(
    "weather_reclamation_oldest_queued_age_seconds",
    "Age in seconds of the oldest queued reclamation target",
)
RECLAMATION_OLDEST_DELETING_AGE_SECONDS = REGISTRY.gauge(
    "weather_reclamation_oldest_deleting_age_seconds",
    "Age in seconds of the oldest actively leased deleting reclamation target",
)
RECLAMATION_OLDEST_FAILED_AGE_SECONDS = REGISTRY.gauge(
    "weather_reclamation_oldest_failed_age_seconds",
    "Age in seconds of the oldest quarantined failed reclamation target",
)

INVARIANT_VIOLATIONS_COUNT = REGISTRY.gauge(
    "weather_lifecycle_invariant_violations_count",
    "Number of detected lifecycle contract invariant violations",
)
ANTI_RESURRECTION_VIOLATIONS_COUNT = REGISTRY.gauge(
    "weather_anti_resurrection_violations_count",
    "Number of detected anti-resurrection violations (recreated runs under tombstones)",
)


@dataclass(frozen=True)
class ReclamationQueueStats:
    """Statistics for granular reclamation queue."""

    queued_count: int = 0
    deleting_count: int = 0
    deleted_count: int = 0
    failed_count: int = 0
    oldest_queued_age_s: float = 0.0
    oldest_deleting_age_s: float = 0.0
    oldest_failed_age_s: float = 0.0


@dataclass(frozen=True)
class InvariantViolation:
    """Details of a lifecycle consistency or anti-resurrection violation."""

    violation_type: str
    model_id: str
    cycle_time: datetime
    description: str


@dataclass
class LifecycleHealthReport:
    """Comprehensive health report for Data Lifecycle V3 components."""

    active_cycles: int = 0
    claimed_cycles: int = 0
    tombstone_cycles: int = 0
    oldest_claim_age_s: float = 0.0
    stuck_claims_warning: int = 0
    stuck_claims_critical: int = 0

    # Sweeper
    sweeper_eligible_count: int = 0
    sweeper_unpurged_count: int = 0
    sweeper_oldest_overdue_s: float = 0.0

    # Reclamation
    reclamation: ReclamationQueueStats = field(default_factory=ReclamationQueueStats)

    # Invariants
    violations: list[InvariantViolation] = field(default_factory=list)
    error: str | None = None


class LifecycleHealthCollector:
    """Collects bounded metrics on cycle lifecycle, finalizer, sweeper, and reclamation."""

    CLAIM_WARNING_THRESHOLD_S = 3600.0  # 1 hour
    CLAIM_CRITICAL_THRESHOLD_S = 14400.0  # 4 hours
    DELETING_LEASE_WARNING_S = 600.0  # 10 minutes

    def __init__(self, engine: Engine | Connection) -> None:
        self.engine = engine

    def collect(self) -> LifecycleHealthReport:
        """Run bounded SQL aggregates across lifecycle tables."""
        try:
            conn_ctx = self.engine.connect() if hasattr(self.engine, "connect") else nullcontext(self.engine)
            with conn_ctx as conn:
                # 1. Whole-cycle lifecycle counts and claim ages
                q_cycle = text(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE deletion_started_at IS NULL) AS active_cnt,
                        COUNT(*) FILTER (WHERE deletion_started_at IS NOT NULL AND deleted_at IS NULL) AS claimed_cnt,
                        COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) AS tombstone_cnt,
                        COALESCE(
                            EXTRACT(EPOCH FROM (NOW() - MIN(deletion_started_at) FILTER (WHERE deletion_started_at IS NOT NULL AND deleted_at IS NULL))),
                            0
                        ) AS oldest_claim_age,
                        COUNT(*) FILTER (
                            WHERE deletion_started_at IS NOT NULL
                              AND deleted_at IS NULL
                              AND deletion_started_at <= NOW() - make_interval(secs => :warn_thresh)
                        ) AS stuck_warn_cnt,
                        COUNT(*) FILTER (
                            WHERE deletion_started_at IS NOT NULL
                              AND deleted_at IS NULL
                              AND deletion_started_at <= NOW() - make_interval(secs => :crit_thresh)
                        ) AS stuck_crit_cnt
                    FROM forecast_cycle_lifecycle
                    """
                )
                row_cycle = conn.execute(
                    q_cycle,
                    {
                        "warn_thresh": self.CLAIM_WARNING_THRESHOLD_S,
                        "crit_thresh": self.CLAIM_CRITICAL_THRESHOLD_S,
                    },
                ).fetchone()

                active_cnt = int(row_cycle.active_cnt or 0) if row_cycle else 0
                claimed_cnt = int(row_cycle.claimed_cnt or 0) if row_cycle else 0
                tombstone_cnt = int(row_cycle.tombstone_cnt or 0) if row_cycle else 0
                oldest_claim_age = float(row_cycle.oldest_claim_age or 0.0) if row_cycle else 0.0
                stuck_warn_cnt = int(row_cycle.stuck_warn_cnt or 0) if row_cycle else 0
                stuck_crit_cnt = int(row_cycle.stuck_crit_cnt or 0) if row_cycle else 0

                # 2. 14-day Metadata sweeper stats
                q_sweeper = text(
                    """
                    SELECT
                        COUNT(*) AS eligible_cnt,
                        COUNT(r.id) AS unpurged_cnt,
                        COALESCE(
                            EXTRACT(EPOCH FROM (NOW() - (MIN(l.deleted_at) + INTERVAL '14 days'))),
                            0
                        ) AS oldest_overdue_s
                    FROM forecast_cycle_lifecycle l
                    LEFT JOIN model_runs r ON r.cycle_time = l.cycle_time
                    WHERE l.deleted_at IS NOT NULL
                      AND l.deleted_at <= NOW() - INTERVAL '14 days'
                    """
                )
                row_sweep = conn.execute(q_sweeper).fetchone()
                sweep_eligible = int(row_sweep.eligible_cnt or 0) if row_sweep else 0
                sweep_unpurged = int(row_sweep.unpurged_cnt or 0) if row_sweep else 0
                sweep_oldest_overdue = (
                    max(0.0, float(row_sweep.oldest_overdue_s or 0.0))
                    if (row_sweep and sweep_unpurged > 0)
                    else 0.0
                )

                # 3. Granular reclamation queue stats
                q_rec = text(
                    """
                    SELECT
                        status,
                        COUNT(*) AS cnt,
                        COALESCE(EXTRACT(EPOCH FROM (NOW() - MIN(created_at))), 0) AS oldest_age
                    FROM reclamation_queue
                    GROUP BY status
                    """
                )
                rec_counts: dict[str, int] = {"queued": 0, "deleting": 0, "deleted": 0, "failed": 0}
                rec_oldest: dict[str, float] = {"queued": 0.0, "deleting": 0.0, "failed": 0.0}
                for r in conn.execute(q_rec):
                    st = str(r.status)
                    rec_counts[st] = int(r.cnt or 0)
                    if st in rec_oldest:
                        rec_oldest[st] = float(r.oldest_age or 0.0)

                rec_stats = ReclamationQueueStats(
                    queued_count=rec_counts["queued"],
                    deleting_count=rec_counts["deleting"],
                    deleted_count=rec_counts["deleted"],
                    failed_count=rec_counts["failed"],
                    oldest_queued_age_s=rec_oldest["queued"],
                    oldest_deleting_age_s=rec_oldest["deleting"],
                    oldest_failed_age_s=rec_oldest["failed"],
                )

                # 4. Invariant Checks (bounded LIMIT 10)
                violations: list[InvariantViolation] = []

                # Invariant 1: deleted_at without deletion_started_at
                q_inv1 = text(
                    """
                    SELECT model_id, cycle_time
                    FROM forecast_cycle_lifecycle
                    WHERE deleted_at IS NOT NULL
                      AND deletion_started_at IS NULL
                    LIMIT 10
                    """
                )
                for r in conn.execute(q_inv1):
                    violations.append(
                        InvariantViolation(
                            violation_type="invalid_lifecycle_transition",
                            model_id=str(r.model_id),
                            cycle_time=r.cycle_time,
                            description="Cycle marked deleted_at without prior deletion_started_at claim",
                        )
                    )

                # Invariant 2: Anti-resurrection audit: run active or created after tombstone
                q_inv2 = text(
                    """
                    SELECT l.model_id, l.cycle_time, r.id, r.status, r.created_at, l.deleted_at
                    FROM forecast_cycle_lifecycle l
                    JOIN model_runs r ON r.cycle_time = l.cycle_time
                    WHERE l.deleted_at IS NOT NULL
                      AND (r.created_at > l.deleted_at OR r.status IN ('processing', 'partial'))
                    LIMIT 10
                    """
                )
                anti_resurrect_count = 0
                for r in conn.execute(q_inv2):
                    anti_resurrect_count += 1
                    violations.append(
                        InvariantViolation(
                            violation_type="anti_resurrection_violation",
                            model_id=str(r.model_id),
                            cycle_time=r.cycle_time,
                            description=(
                                f"ModelRun id={r.id} status={r.status} created_at={r.created_at} "
                                f"violates permanent tombstone deleted_at={r.deleted_at}"
                            ),
                        )
                    )

            # Update Prometheus metrics
            LIFECYCLE_ACTIVE_CYCLES.set(float(active_cnt))
            LIFECYCLE_CLAIMED_CYCLES.set(float(claimed_cnt))
            LIFECYCLE_TOMBSTONE_CYCLES.set(float(tombstone_cnt))
            LIFECYCLE_OLDEST_CLAIM_AGE_SECONDS.set(oldest_claim_age)
            LIFECYCLE_STUCK_CLAIMS_WARNING.set(float(stuck_warn_cnt))
            LIFECYCLE_STUCK_CLAIMS_CRITICAL.set(float(stuck_crit_cnt))

            METADATA_SWEEPER_ELIGIBLE.set(float(sweep_eligible))
            METADATA_SWEEPER_UNPURGED.set(float(sweep_unpurged))
            METADATA_SWEEPER_OLDEST_OVERDUE_SECONDS.set(sweep_oldest_overdue)

            for st, cnt in rec_counts.items():
                RECLAMATION_QUEUE_COUNT.labels(status=st).set(float(cnt))
            RECLAMATION_OLDEST_QUEUED_AGE_SECONDS.set(rec_stats.oldest_queued_age_s)
            RECLAMATION_OLDEST_DELETING_AGE_SECONDS.set(rec_stats.oldest_deleting_age_s)
            RECLAMATION_OLDEST_FAILED_AGE_SECONDS.set(rec_stats.oldest_failed_age_s)

            INVARIANT_VIOLATIONS_COUNT.set(float(len(violations)))
            ANTI_RESURRECTION_VIOLATIONS_COUNT.set(float(anti_resurrect_count))

            return LifecycleHealthReport(
                active_cycles=active_cnt,
                claimed_cycles=claimed_cnt,
                tombstone_cycles=tombstone_cnt,
                oldest_claim_age_s=oldest_claim_age,
                stuck_claims_warning=stuck_warn_cnt,
                stuck_claims_critical=stuck_crit_cnt,
                sweeper_eligible_count=sweep_eligible,
                sweeper_unpurged_count=sweep_unpurged,
                sweeper_oldest_overdue_s=sweep_oldest_overdue,
                reclamation=rec_stats,
                violations=violations,
            )
        except Exception as exc:
            logger.warning("Lifecycle health collection failed: %s", exc)
            return LifecycleHealthReport(error=str(exc))
