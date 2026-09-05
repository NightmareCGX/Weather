"""Pure domain forecast cycle lifecycle and retention planning math (Lifecycle V2).

This module is the single source of truth for the platform's Data Lifecycle V2
policy. It is pure, deterministic, and side-effect-free: it contains no
SQLAlchemy, database connections, or object-store calls.

Lifecycle V2 Policy (Locked Contract):
--------------------------------------
1. Model Independence:
   GFS and GEFS lifecycles advance completely independently. A lagging or
   failed GEFS cycle does not block GFS retention advancement, and vice versa.

2. Authoritative Cadence & Terminal State:
   Each model M has an authoritative cycle cadence C_M (e.g. GFS=6h, GEFS=6h,
   future_model=3h) defined in :mod:`domain.cadence`. The authoritative terminal
   state for generation completion is ``model_runs.status == "ready"``.

3. Generation Cutoff Formula:
   Given:
       T = latest cycle for model M with status == "ready"
       C = cycle cadence for model M
       cutoff = T - C

   Then:
       retain:
           cycles >= cutoff
       deletion eligible:
           cycles < cutoff

   * Cycles < cutoff are eligible for visibility retirement and physical GC.
   * Partial / in-progress newer cycles (cycle_time > T) do NOT advance T or cutoff.
   * Failed or skipped cycles do not alter the formula: cutoff is strictly T - C.
     (We intentionally do NOT implement "keep two successful cycles").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from domain.cadence import canonical_cycle_cadence


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_lifecycle_cutoff(
    ready_cycle_times: Iterable[datetime],
    cadence: timedelta,
) -> tuple[datetime | None, datetime | None]:
    """Compute latest ready cycle T and deletion cutoff (T - C).

    Args:
        ready_cycle_times: Datetimes of all cycles with status == 'ready' for the model.
        cadence: The model's cycle cadence as a timedelta (e.g. 6 hours).

    Returns:
        ``(latest_ready_T, cutoff_T_minus_C)``, or ``(None, None)`` if no cycles are ready.
    """
    ready = sorted({_ensure_utc(t) for t in ready_cycle_times})
    if not ready:
        return None, None
    latest_t = ready[-1]
    cutoff = latest_t - cadence
    return latest_t, cutoff


@dataclass(frozen=True)
class ModelLifecycleSnapshot:
    """Snapshot of a single model cycle's durable lifecycle and run status.

    Attributes:
        model_id: Platform model identifier (e.g. 'gfs', 'gefs').
        cycle_time: The logical UTC cycle datetime.
        status: Model run status if known (e.g. 'ready', 'partial', 'failed', 'processing').
        retired_at: Timestamp when cycle visibility was retired, or None.
        retired_by_cycle_time: The anchor cycle T that triggered retirement, or None.
        deletion_started_at: Timestamp when physical deletion claimed the cycle, or None.
        deleted_at: Timestamp when physical deletion completed (tombstone), or None.
    """

    model_id: str
    cycle_time: datetime
    status: str | None = None
    retired_at: datetime | None = None
    retired_by_cycle_time: datetime | None = None
    deletion_started_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", self.model_id.lower().strip())
        object.__setattr__(self, "cycle_time", _ensure_utc(self.cycle_time))
        if self.retired_at is not None:
            object.__setattr__(self, "retired_at", _ensure_utc(self.retired_at))
        if self.retired_by_cycle_time is not None:
            object.__setattr__(
                self,
                "retired_by_cycle_time",
                _ensure_utc(self.retired_by_cycle_time),
            )
        if self.deletion_started_at is not None:
            object.__setattr__(
                self,
                "deletion_started_at",
                _ensure_utc(self.deletion_started_at),
            )
        if self.deleted_at is not None:
            object.__setattr__(self, "deleted_at", _ensure_utc(self.deleted_at))

    @property
    def is_ready(self) -> bool:
        """Whether this model run is in authoritative terminal status 'ready'."""
        return self.status == "ready"

    @property
    def is_retired(self) -> bool:
        """Whether this cycle is durably retired from public visibility."""
        return self.retired_at is not None

    @property
    def is_deletion_started(self) -> bool:
        """Whether physical deletion has been claimed/started (durable fence)."""
        return self.deletion_started_at is not None

    @property
    def is_deleted(self) -> bool:
        """Whether this cycle has been physically deleted (tombstone)."""
        return self.deleted_at is not None


# Backwards compatibility alias
CycleLifecycleSnapshot = ModelLifecycleSnapshot


@dataclass(frozen=True)
class LifecycleDecision:
    """Evaluation result for a model cycle under Data Lifecycle V2.

    Attributes:
        model_id: Model identifier.
        cycle_time: The evaluated cycle time.
        is_eligible_for_deletion: True if cycle_time < cutoff (T - C).
        should_retire: True if eligible for deletion and not yet marked retired.
        retired_by_cycle_time: The anchor cycle T that triggered retirement/GC, or None.
        cutoff: The cutoff datetime (T - C) used for the decision, or None.
        reason: Structured diagnostic reason for observability and logs.
    """

    model_id: str
    cycle_time: datetime
    is_eligible_for_deletion: bool
    should_retire: bool
    retired_by_cycle_time: datetime | None
    cutoff: datetime | None
    reason: str


# Backwards compatibility alias
RetirementDecision = LifecycleDecision
GcEligibilityDecision = LifecycleDecision


def canonical_cycle_store_path(
    model: str,
    cycle_time: datetime,
    *,
    base_bucket: str = "weather-data",
) -> str:
    """Derive the canonical sharded_v1 Zarr store URL for a model and cycle.

    Args:
        model: Model identifier ('gfs' or 'gefs').
        cycle_time: UTC cycle datetime.
        base_bucket: S3/MinIO bucket name (defaults to 'weather-data').

    Returns:
        Canonical S3 URL, e.g. 's3://weather-data/gfs/2026-09-02/18/cycle.zarr'.
    """
    dt = _ensure_utc(cycle_time)
    date_str = dt.strftime("%Y-%m-%d")
    hour_str = f"{dt.hour:02d}"
    return f"s3://{base_bucket}/{model.lower().strip()}/{date_str}/{hour_str}/cycle.zarr"


def evaluate_cycle_lifecycle(
    snapshot: ModelLifecycleSnapshot,
    latest_ready_cycle: datetime | None,
    cadence: timedelta,
) -> LifecycleDecision:
    """Evaluate lifecycle retention and deletion eligibility for one cycle snapshot.

    Args:
        snapshot: The cycle's durable lifecycle snapshot.
        latest_ready_cycle: The model's newest cycle with status == 'ready' (T).
        cadence: The model's cycle cadence as timedelta (C).

    Returns:
        A LifecycleDecision describing whether the cycle is retained or eligible for GC.
    """
    if snapshot.is_deleted:
        return LifecycleDecision(
            model_id=snapshot.model_id,
            cycle_time=snapshot.cycle_time,
            is_eligible_for_deletion=False,
            should_retire=False,
            retired_by_cycle_time=snapshot.retired_by_cycle_time,
            cutoff=None,
            reason="cycle_already_deleted",
        )

    if latest_ready_cycle is None:
        # No ready cycle exists yet to anchor the lifecycle
        return LifecycleDecision(
            model_id=snapshot.model_id,
            cycle_time=snapshot.cycle_time,
            is_eligible_for_deletion=False,
            should_retire=False,
            retired_by_cycle_time=None,
            cutoff=None,
            reason="no_ready_cycle_anchor",
        )

    t_utc = _ensure_utc(latest_ready_cycle)
    cutoff = t_utc - cadence

    if snapshot.cycle_time >= cutoff:
        # Retained: at or above cutoff (T or T - C)
        return LifecycleDecision(
            model_id=snapshot.model_id,
            cycle_time=snapshot.cycle_time,
            is_eligible_for_deletion=False,
            should_retire=False,
            retired_by_cycle_time=None,
            cutoff=cutoff,
            reason="retained_at_or_above_cutoff",
        )

    # Older than cutoff (< T - C): eligible for retirement and physical GC
    should_retire = not snapshot.is_retired
    reason = f"older_than_cutoff_{cutoff.strftime('%Y%m%d%H%M')}"
    return LifecycleDecision(
        model_id=snapshot.model_id,
        cycle_time=snapshot.cycle_time,
        is_eligible_for_deletion=True,
        should_retire=should_retire,
        retired_by_cycle_time=t_utc,
        cutoff=cutoff,
        reason=reason,
    )


@dataclass(frozen=True)
class ModelLifecyclePlan:
    """Deterministic, batch-evaluated lifecycle transitions for one model.

    Attributes:
        model_id: Model identifier.
        latest_ready_cycle: Anchor cycle T with status == 'ready', or None.
        cutoff: Cutoff datetime T - C, or None.
        decisions: Tuple of all per-cycle decisions, sorted by cycle_time ascending.
        active_visible_cycles: Sorted list of retained, non-deleted cycle times.
        retired_cycles: Sorted list of retired, non-deleted cycle times.
        deleted_cycles: Sorted list of physically deleted tombstone cycle times.
        eligible_for_deletion: Decisions for cycles eligible for physical GC (oldest-first).
    """

    model_id: str
    latest_ready_cycle: datetime | None
    cutoff: datetime | None
    decisions: tuple[LifecycleDecision, ...]
    active_visible_cycles: tuple[datetime, ...]
    retired_cycles: tuple[datetime, ...]
    deleted_cycles: tuple[datetime, ...]
    eligible_for_deletion: tuple[LifecycleDecision, ...]

    @property
    def would_retire(self) -> tuple[LifecycleDecision, ...]:
        """Subset of decisions where should_retire is True."""
        return tuple(d for d in self.decisions if d.should_retire)

    @property
    def would_gc(self) -> tuple[LifecycleDecision, ...]:
        """Subset of decisions eligible for physical GC (alias of eligible_for_deletion)."""
        return self.eligible_for_deletion

    @property
    def blocked(self) -> tuple[LifecycleDecision, ...]:
        """Subset of non-deleted decisions that are retained (not eligible for deletion)."""
        return tuple(d for d in self.decisions if not d.is_eligible_for_deletion)

    # Legacy compatibility aliases
    @property
    def retirements(self) -> tuple[LifecycleDecision, ...]:
        return self.would_retire

    @property
    def gc_eligibilities(self) -> tuple[LifecycleDecision, ...]:
        return self.decisions


# Legacy compatibility alias
LifecyclePlan = ModelLifecyclePlan


def plan_model_lifecycle(
    model_id: str,
    snapshots: Iterable[ModelLifecycleSnapshot],
    ready_cycle_times: Iterable[datetime],
    *,
    cadence: timedelta | None = None,
) -> ModelLifecyclePlan:
    """Evaluate lifecycle transitions deterministically for one model.

    Args:
        model_id: Platform model identifier (e.g. 'gfs', 'gefs').
        snapshots: All known lifecycle snapshots for this model.
        ready_cycle_times: Datetimes of all runs with status == 'ready' for this model.
        cadence: Optional cadence override. Defaults to canonical_cycle_cadence(model_id).

    Returns:
        A ModelLifecyclePlan with sorted decisions and categorized cycle groupings.
    """
    m_id = model_id.lower().strip()
    eff_cadence = cadence if cadence is not None else canonical_cycle_cadence(m_id)
    t_ready, cutoff = compute_lifecycle_cutoff(ready_cycle_times, eff_cadence)

    sorted_snapshots = sorted(
        (s for s in snapshots if s.model_id == m_id),
        key=lambda s: s.cycle_time,
    )

    decisions: list[LifecycleDecision] = []
    active_visible: list[datetime] = []
    retired: list[datetime] = []
    deleted: list[datetime] = []
    eligible_for_gc: list[LifecycleDecision] = []

    for snapshot in sorted_snapshots:
        if snapshot.is_deleted:
            deleted.append(snapshot.cycle_time)
            continue

        decision = evaluate_cycle_lifecycle(snapshot, t_ready, eff_cadence)
        decisions.append(decision)

        if decision.is_eligible_for_deletion:
            eligible_for_gc.append(decision)
            retired.append(snapshot.cycle_time)
        else:
            if snapshot.is_retired:
                retired.append(snapshot.cycle_time)
            else:
                active_visible.append(snapshot.cycle_time)

    return ModelLifecyclePlan(
        model_id=m_id,
        latest_ready_cycle=t_ready,
        cutoff=cutoff,
        decisions=tuple(decisions),
        active_visible_cycles=tuple(active_visible),
        retired_cycles=tuple(retired),
        deleted_cycles=tuple(deleted),
        eligible_for_deletion=tuple(eligible_for_gc),
    )


def plan_lifecycle(
    snapshots: Iterable[ModelLifecycleSnapshot],
    ready_cycle_times: Iterable[datetime],
    *,
    model_id: str = "gfs",
    cadence: timedelta | None = None,
) -> ModelLifecyclePlan:
    """Convenience / backward-compatible wrapper around plan_model_lifecycle."""
    return plan_model_lifecycle(
        model_id=model_id,
        snapshots=snapshots,
        ready_cycle_times=ready_cycle_times,
        cadence=cadence,
    )
