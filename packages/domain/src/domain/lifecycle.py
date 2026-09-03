"""Pure domain forecast cycle lifecycle and retention planning math.

This module is the single source of truth for the platform's Phase 6 lifecycle
policy. It is pure, deterministic, and side-effect-free: it contains no
SQLAlchemy, database connections, or object-store calls.

Lifecycle Policy:
-----------------
The lifecycle unit is a logical forecast cycle:

    cycle_time C = GFS(C) + GEFS(C)

A cycle C is paired-ready iff:

    GFS(C).status == "ready" AND GEFS(C).status == "ready"

Only paired-ready cycles advance the lifecycle. All cycles follow the identical
lifecycle policy, regardless of whether cycle C itself is ready, partial,
failed, or processing.

Stage 1 — Visibility Retirement (R1):
-------------------------------------
For any cycle C, R1 is the earliest paired-ready cycle such that:

    R1 >= C + 24 hours

* If no such R1 exists, C remains VISIBLE / active.
* If R1 exists, C becomes RETIRED (durable retired_by_cycle_time = R1).

Stage 2 — Physical GC Eligibility (R2):
---------------------------------------
After C is retired by R1, R2 is the earliest paired-ready cycle such that:

    R2 >= R1 + 6 hours

* If R2 exists, C becomes GC-ELIGIBLE (derivable from R1 and R2).
* If no such R2 exists, C is NOT GC-eligible.

IMPORTANT: This is NOT a fixed "C + 30h" rule. The 6-hour deletion safety window
begins from the actual successful successor cycle R1 that caused retirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

#: Minimum time delta between cycle C and retirement successor R1.
RETIREMENT_DELTA: timedelta = timedelta(hours=24)

#: Minimum time delta between retirement successor R1 and GC successor R2.
GC_SAFETY_DELTA: timedelta = timedelta(hours=6)


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def find_r1(
    cycle_time: datetime,
    paired_ready_times: Iterable[datetime],
) -> datetime | None:
    """Find the earliest paired-ready cycle R1 such that R1 >= cycle_time + 24h.

    Args:
        cycle_time: The UTC cycle time being evaluated (C).
        paired_ready_times: All known paired-ready cycle timestamps.

    Returns:
        The earliest qualifying paired-ready cycle time, or None if none exists.
    """
    c_utc = _ensure_utc(cycle_time)
    min_r1 = c_utc + RETIREMENT_DELTA
    candidates = [
        _ensure_utc(t) for t in paired_ready_times if _ensure_utc(t) >= min_r1
    ]
    if not candidates:
        return None
    return min(candidates)


def find_r2(
    retired_by_cycle_time: datetime,
    paired_ready_times: Iterable[datetime],
) -> datetime | None:
    """Find the earliest paired-ready cycle R2 such that R2 >= R1 + 6h.

    Args:
        retired_by_cycle_time: The UTC cycle time R1 that retired the cycle.
        paired_ready_times: All known paired-ready cycle timestamps.

    Returns:
        The earliest qualifying paired-ready cycle time, or None if none exists.
    """
    r1_utc = _ensure_utc(retired_by_cycle_time)
    min_r2 = r1_utc + GC_SAFETY_DELTA
    candidates = [
        _ensure_utc(t) for t in paired_ready_times if _ensure_utc(t) >= min_r2
    ]
    if not candidates:
        return None
    return min(candidates)


@dataclass(frozen=True)
class CycleLifecycleSnapshot:
    """Snapshot of a single forecast cycle's durable lifecycle and run status.

    Attributes:
        cycle_time: The logical UTC cycle time (C).
        retired_at: When the cycle was retired, or None if currently visible.
        retired_by_cycle_time: The cycle R1 that triggered retirement, or None.
        deleted_at: When physical GC completed (tombstone), or None.
        gfs_status: Status of the GFS run if known (e.g. 'ready', 'partial').
        gefs_status: Status of the GEFS run if known (e.g. 'ready', 'partial').
    """

    cycle_time: datetime
    retired_at: datetime | None = None
    retired_by_cycle_time: datetime | None = None
    deletion_started_at: datetime | None = None
    deleted_at: datetime | None = None
    gfs_status: str | None = None
    gefs_status: str | None = None

    def __post_init__(self) -> None:
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
    def is_paired_ready(self) -> bool:
        """Whether both GFS and GEFS runs are ready."""
        return self.gfs_status == "ready" and self.gefs_status == "ready"

    @property
    def is_retired(self) -> bool:
        """Whether this cycle is durably retired."""
        return self.retired_at is not None

    @property
    def is_deletion_started(self) -> bool:
        """Whether physical deletion has been claimed/started."""
        return self.deletion_started_at is not None

    @property
    def is_deleted(self) -> bool:
        """Whether this cycle has been physically deleted (tombstone)."""
        return self.deleted_at is not None


@dataclass(frozen=True)
class RetirementDecision:
    """Evaluation result for whether a cycle should transition to RETIRED.

    Attributes:
        cycle_time: The evaluated cycle time.
        should_retire: True if the cycle is not yet retired and a qualifying R1 exists.
        retired_by_cycle_time: The qualifying R1 cycle time, or None.
        reason: Human/log-readable reason for the decision.
    """

    cycle_time: datetime
    should_retire: bool
    retired_by_cycle_time: datetime | None
    reason: str


@dataclass(frozen=True)
class GcEligibilityDecision:
    """Evaluation result for whether a retired cycle is eligible for physical GC.

    Attributes:
        cycle_time: The evaluated cycle time.
        is_gc_eligible: True if the cycle is retired and a qualifying R2 exists.
        retired_by_cycle_time: The R1 cycle time that retired this cycle.
        gc_eligible_by_cycle_time: The R2 cycle time that unlocked GC, or None.
        reason: Human/log-readable reason for the decision.
    """

    cycle_time: datetime
    is_gc_eligible: bool
    retired_by_cycle_time: datetime | None
    gc_eligible_by_cycle_time: datetime | None
    reason: str


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
    return f"s3://{base_bucket}/{model}/{date_str}/{hour_str}/cycle.zarr"


@dataclass(frozen=True)
class LifecyclePlan:
    """Deterministic, batch-evaluated lifecycle transitions and active sets.

    Attributes:
        retirements: New retirement decisions for previously visible cycles.
        gc_eligibilities: GC eligibility decisions for retired, non-deleted cycles.
        active_visible_cycles: Sorted list of currently visible cycle times.
        retired_cycles: Sorted list of currently retired, non-deleted cycle times.
        deleted_cycles: Sorted list of physically deleted tombstone cycle times.
    """

    retirements: tuple[RetirementDecision, ...]
    gc_eligibilities: tuple[GcEligibilityDecision, ...]
    active_visible_cycles: tuple[datetime, ...]
    retired_cycles: tuple[datetime, ...]
    deleted_cycles: tuple[datetime, ...]

    @property
    def would_retire(self) -> tuple[RetirementDecision, ...]:
        """Subset of retirement decisions where should_retire is True."""
        return tuple(r for r in self.retirements if r.should_retire)

    @property
    def would_gc(self) -> tuple[GcEligibilityDecision, ...]:
        """Subset of GC eligibility decisions where is_gc_eligible is True (oldest-first)."""
        return tuple(g for g in self.gc_eligibilities if g.is_gc_eligible)

    @property
    def blocked(self) -> tuple[RetirementDecision | GcEligibilityDecision, ...]:
        """Decisions where a cycle is not yet ready to advance (missing R1 or R2)."""
        blocked_ret: list[RetirementDecision | GcEligibilityDecision] = [
            r for r in self.retirements if not r.should_retire
        ]
        blocked_gc: list[RetirementDecision | GcEligibilityDecision] = [
            g for g in self.gc_eligibilities if not g.is_gc_eligible
        ]
        return tuple(blocked_ret + blocked_gc)


def evaluate_retirement(
    snapshot: CycleLifecycleSnapshot,
    paired_ready_times: Iterable[datetime],
) -> RetirementDecision:
    """Evaluate whether an individual cycle snapshot should become retired.

    Args:
        snapshot: The cycle snapshot.
        paired_ready_times: Known paired-ready cycle timestamps.

    Returns:
        A RetirementDecision.
    """
    if snapshot.is_deleted:
        return RetirementDecision(
            cycle_time=snapshot.cycle_time,
            should_retire=False,
            retired_by_cycle_time=snapshot.retired_by_cycle_time,
            reason="cycle_already_deleted",
        )
    if snapshot.is_retired:
        return RetirementDecision(
            cycle_time=snapshot.cycle_time,
            should_retire=False,
            retired_by_cycle_time=snapshot.retired_by_cycle_time,
            reason="cycle_already_retired",
        )

    r1 = find_r1(snapshot.cycle_time, paired_ready_times)
    if r1 is not None:
        return RetirementDecision(
            cycle_time=snapshot.cycle_time,
            should_retire=True,
            retired_by_cycle_time=r1,
            reason=f"qualifying_r1_found_{r1.strftime('%Y%m%d%H%M')}",
        )
    return RetirementDecision(
        cycle_time=snapshot.cycle_time,
        should_retire=False,
        retired_by_cycle_time=None,
        reason="no_qualifying_r1_paired_ready",
    )


def evaluate_gc_eligibility(
    snapshot: CycleLifecycleSnapshot,
    paired_ready_times: Iterable[datetime],
) -> GcEligibilityDecision:
    """Evaluate whether an individual cycle snapshot is eligible for physical GC.

    Args:
        snapshot: The cycle snapshot.
        paired_ready_times: Known paired-ready cycle timestamps.

    Returns:
        A GcEligibilityDecision.
    """
    if snapshot.is_deleted:
        return GcEligibilityDecision(
            cycle_time=snapshot.cycle_time,
            is_gc_eligible=False,
            retired_by_cycle_time=snapshot.retired_by_cycle_time,
            gc_eligible_by_cycle_time=None,
            reason="cycle_already_deleted",
        )
    if not snapshot.is_retired or snapshot.retired_by_cycle_time is None:
        return GcEligibilityDecision(
            cycle_time=snapshot.cycle_time,
            is_gc_eligible=False,
            retired_by_cycle_time=None,
            gc_eligible_by_cycle_time=None,
            reason="cycle_not_retired",
        )

    r1 = snapshot.retired_by_cycle_time
    r2 = find_r2(r1, paired_ready_times)
    if r2 is not None:
        return GcEligibilityDecision(
            cycle_time=snapshot.cycle_time,
            is_gc_eligible=True,
            retired_by_cycle_time=r1,
            gc_eligible_by_cycle_time=r2,
            reason=f"qualifying_r2_found_{r2.strftime('%Y%m%d%H%M')}",
        )
    return GcEligibilityDecision(
        cycle_time=snapshot.cycle_time,
        is_gc_eligible=False,
        retired_by_cycle_time=r1,
        gc_eligible_by_cycle_time=None,
        reason="no_qualifying_r2_paired_ready",
    )


def plan_lifecycle(
    snapshots: Iterable[CycleLifecycleSnapshot],
    paired_ready_times: Iterable[datetime],
) -> LifecyclePlan:
    """Evaluate lifecycle transitions deterministically across all cycle snapshots.

    Args:
        snapshots: All known cycle lifecycle snapshots.
        paired_ready_times: All known paired-ready cycle timestamps.

    Returns:
        A LifecyclePlan containing sorted decisions and category groupings.
    """
    sorted_snapshots = sorted(snapshots, key=lambda s: s.cycle_time)
    paired_ready_set = tuple(sorted({_ensure_utc(t) for t in paired_ready_times}))

    retirements: list[RetirementDecision] = []
    gc_eligibilities: list[GcEligibilityDecision] = []
    active_visible: list[datetime] = []
    retired: list[datetime] = []
    deleted: list[datetime] = []

    for snapshot in sorted_snapshots:
        if snapshot.is_deleted:
            deleted.append(snapshot.cycle_time)
            continue

        if not snapshot.is_retired:
            ret_decision = evaluate_retirement(snapshot, paired_ready_set)
            if ret_decision.should_retire:
                retirements.append(ret_decision)
                # When planned for retirement, evaluate whether it also meets R2
                # under the newly determined R1
                assert ret_decision.retired_by_cycle_time is not None
                virtual_snapshot = CycleLifecycleSnapshot(
                    cycle_time=snapshot.cycle_time,
                    retired_at=datetime.now(timezone.utc),
                    retired_by_cycle_time=ret_decision.retired_by_cycle_time,
                    gfs_status=snapshot.gfs_status,
                    gefs_status=snapshot.gefs_status,
                )
                gc_decision = evaluate_gc_eligibility(
                    virtual_snapshot, paired_ready_set
                )
                gc_eligibilities.append(gc_decision)
                retired.append(snapshot.cycle_time)
            else:
                active_visible.append(snapshot.cycle_time)
        else:
            retired.append(snapshot.cycle_time)
            gc_decision = evaluate_gc_eligibility(snapshot, paired_ready_set)
            gc_eligibilities.append(gc_decision)

    return LifecyclePlan(
        retirements=tuple(retirements),
        gc_eligibilities=tuple(gc_eligibilities),
        active_visible_cycles=tuple(active_visible),
        retired_cycles=tuple(retired),
        deleted_cycles=tuple(deleted),
    )
