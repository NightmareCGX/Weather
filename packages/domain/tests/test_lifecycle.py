"""Pure unit tests for domain.lifecycle (Data Lifecycle V2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from domain.cadence import register_canonical_cycle_cadence
from domain.lifecycle import (
    CycleLifecycleSnapshot,
    ModelLifecyclePlan,
    ModelLifecycleSnapshot,
    canonical_cycle_store_path,
    compute_lifecycle_cutoff,
    evaluate_cycle_lifecycle,
    plan_model_lifecycle,
)


def _dt(year: int, month: int, day: int, hour: int, tz: bool = True) -> datetime:
    """Helper to build UTC datetime for tests."""
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc if tz else None)


# ---------------------------------------------------------------------------
# Cutoff Computation Tests: cutoff = T - C
# ---------------------------------------------------------------------------


def test_compute_lifecycle_cutoff_empty() -> None:
    """Verify empty ready cycles returns (None, None)."""
    t, cutoff = compute_lifecycle_cutoff([], timedelta(hours=6))
    assert t is None
    assert cutoff is None


def test_compute_lifecycle_cutoff_6h_normal() -> None:
    """Verify 6h cadence cutoff derivation."""
    ready_cycles = [_dt(2026, 9, 1, 18), _dt(2026, 9, 2, 0)]
    t, cutoff = compute_lifecycle_cutoff(ready_cycles, timedelta(hours=6))
    assert t == _dt(2026, 9, 2, 0)
    assert cutoff == _dt(2026, 9, 1, 18)


def test_compute_lifecycle_cutoff_3h() -> None:
    """Verify 3h cadence cutoff derivation."""
    ready_cycles = [_dt(2026, 9, 1, 3), _dt(2026, 9, 1, 6), _dt(2026, 9, 1, 9)]
    t, cutoff = compute_lifecycle_cutoff(ready_cycles, timedelta(hours=3))
    assert t == _dt(2026, 9, 1, 9)
    assert cutoff == _dt(2026, 9, 1, 6)


# ---------------------------------------------------------------------------
# Principle 6: Normal Progression Lifecycle
# ---------------------------------------------------------------------------


def test_lifecycle_normal_progression_6h() -> None:
    """Verify normal 6h cadence lifecycle progression:

    Stage A:
      18Z ready
      00Z ready
      06Z partial (in-progress)
      -> T = 00Z, cutoff = 18Z
      -> 18Z retained (>= 18Z)
      -> 00Z retained (>= 18Z)
      -> 06Z retained (>= 18Z)
      -> older cycles (< 18Z) eligible for deletion

    Stage B:
      06Z becomes ready
      -> T = 06Z, cutoff = 00Z
      -> 18Z becomes eligible for deletion (< 00Z)
      -> 00Z retained (>= 00Z)
      -> 06Z retained (>= 00Z)
    """
    c_12z = _dt(2026, 9, 1, 12)
    c_18z = _dt(2026, 9, 1, 18)
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    snapshots_a = [
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_12z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_18z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_00z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_06z, status="partial"),
    ]

    # Stage A: 00Z is latest ready (06Z is partial)
    ready_a = [c_12z, c_18z, c_00z]
    plan_a = plan_model_lifecycle("gfs", snapshots_a, ready_a)

    assert plan_a.latest_ready_cycle == c_00z
    assert plan_a.cutoff == c_18z
    # 18Z, 00Z, 06Z are all >= cutoff (retained)
    assert plan_a.active_visible_cycles == (c_18z, c_00z, c_06z)
    # 12Z is < cutoff (eligible for deletion)
    assert len(plan_a.eligible_for_deletion) == 1
    assert plan_a.eligible_for_deletion[0].cycle_time == c_12z
    assert plan_a.eligible_for_deletion[0].is_eligible_for_deletion is True

    # Stage B: 06Z becomes ready
    snapshots_b = [
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_12z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_18z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_00z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_06z, status="ready"),
    ]
    ready_b = [c_12z, c_18z, c_00z, c_06z]
    plan_b = plan_model_lifecycle("gfs", snapshots_b, ready_b)

    assert plan_b.latest_ready_cycle == c_06z
    assert plan_b.cutoff == c_00z
    # 00Z and 06Z are >= cutoff (retained)
    assert plan_b.active_visible_cycles == (c_00z, c_06z)
    # 12Z and 18Z are < cutoff (eligible for deletion)
    eligible_b = [d.cycle_time for d in plan_b.eligible_for_deletion]
    assert eligible_b == [c_12z, c_18z]


# ---------------------------------------------------------------------------
# Principle 7: Failed Adjacency (DO NOT keep two successful cycles)
# ---------------------------------------------------------------------------


def test_lifecycle_failed_adjacency_cutoff_is_strict() -> None:
    """Verify that a failed cycle does NOT alter the T - C cutoff.

    Timeline:
      00Z ready
      06Z failed
      12Z ready

    T = 12Z, C = 6h -> cutoff = 06Z.
    00Z < 06Z -> 00Z IS ELIGIBLE FOR DELETION.
    We assert that 00Z is NOT retained merely because 06Z failed!
    """
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)
    c_12z = _dt(2026, 9, 2, 12)

    snapshots = [
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_00z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_06z, status="failed"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_12z, status="ready"),
    ]
    # Only 00Z and 12Z are ready
    ready_cycles = [c_00z, c_12z]

    plan = plan_model_lifecycle("gfs", snapshots, ready_cycles)

    assert plan.latest_ready_cycle == c_12z
    assert plan.cutoff == c_06z

    # 06Z (failed) and 12Z (ready) are >= cutoff (06Z)
    assert plan.active_visible_cycles == (c_06z, c_12z)

    # 00Z is < 06Z -> must be eligible for deletion!
    eligible_cycles = [d.cycle_time for d in plan.eligible_for_deletion]
    assert eligible_cycles == [c_00z]
    assert plan.eligible_for_deletion[0].is_eligible_for_deletion is True


# ---------------------------------------------------------------------------
# Principle 2: Model Independence (GFS vs GEFS Decoupled)
# ---------------------------------------------------------------------------


def test_model_independence_decoupled() -> None:
    """Verify GFS and GEFS lifecycle plans are completely independent.

    GFS:
      00Z ready, 06Z ready -> T = 06Z, cutoff = 00Z -> 18Z eligible, 00Z/06Z retained.

    GEFS:
      00Z ready, 06Z partial -> T = 00Z, cutoff = 18Z -> 18Z retained, 00Z/06Z retained.
    """
    c_18z = _dt(2026, 9, 1, 18)
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    gfs_snapshots = [
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_18z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_00z, status="ready"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_06z, status="ready"),
    ]
    gefs_snapshots = [
        ModelLifecycleSnapshot(model_id="gefs", cycle_time=c_18z, status="ready"),
        ModelLifecycleSnapshot(model_id="gefs", cycle_time=c_00z, status="ready"),
        ModelLifecycleSnapshot(model_id="gefs", cycle_time=c_06z, status="partial"),
    ]

    gfs_plan = plan_model_lifecycle(
        "gfs", gfs_snapshots, ready_cycle_times=[c_18z, c_00z, c_06z]
    )
    gefs_plan = plan_model_lifecycle(
        "gefs", gefs_snapshots, ready_cycle_times=[c_18z, c_00z]
    )

    # GFS advanced to 06Z; 18Z is eligible for deletion
    assert gfs_plan.latest_ready_cycle == c_06z
    assert gfs_plan.cutoff == c_00z
    assert [d.cycle_time for d in gfs_plan.eligible_for_deletion] == [c_18z]

    # GEFS stayed at 00Z; 18Z is RETAINED (not eligible for deletion)
    assert gefs_plan.latest_ready_cycle == c_00z
    assert gefs_plan.cutoff == c_18z
    assert gefs_plan.eligible_for_deletion == ()
    assert c_18z in gefs_plan.active_visible_cycles


# ---------------------------------------------------------------------------
# 3-Hour Cadence Product Test
# ---------------------------------------------------------------------------


def test_3h_cadence_lifecycle() -> None:
    """Verify 3h cadence model retention math without special-casing."""
    register_canonical_cycle_cadence("future_3h", 3)

    c_03z = _dt(2026, 9, 1, 3)
    c_06z = _dt(2026, 9, 1, 6)
    c_09z = _dt(2026, 9, 1, 9)

    snapshots = [
        ModelLifecycleSnapshot(model_id="future_3h", cycle_time=c_03z, status="ready"),
        ModelLifecycleSnapshot(model_id="future_3h", cycle_time=c_06z, status="ready"),
        ModelLifecycleSnapshot(model_id="future_3h", cycle_time=c_09z, status="ready"),
    ]
    ready = [c_03z, c_06z, c_09z]

    plan = plan_model_lifecycle("future_3h", snapshots, ready)

    assert plan.latest_ready_cycle == c_09z
    assert plan.cutoff == c_06z  # 09Z - 3h = 06Z
    # 06Z and 09Z retained
    assert plan.active_visible_cycles == (c_06z, c_09z)
    # 03Z < 06Z -> eligible for deletion
    assert [d.cycle_time for d in plan.eligible_for_deletion] == [c_03z]


# ---------------------------------------------------------------------------
# Old Cycle Status Invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old_status", ["ready", "partial", "failed", "processing"])
def test_old_cycle_status_invariance(old_status: str) -> None:
    """Verify that an old cycle behind cutoff is eligible regardless of its status."""
    c_old = _dt(2026, 9, 1, 12)
    c_t_minus_c = _dt(2026, 9, 1, 18)
    c_t = _dt(2026, 9, 2, 0)

    snapshot = ModelLifecycleSnapshot(
        model_id="gfs",
        cycle_time=c_old,
        status=old_status,
    )
    decision = evaluate_cycle_lifecycle(
        snapshot, latest_ready_cycle=c_t, cadence=timedelta(hours=6)
    )
    assert decision.is_eligible_for_deletion is True
    assert decision.cutoff == c_t_minus_c
    assert decision.retired_by_cycle_time == c_t


# ---------------------------------------------------------------------------
# Already Deleted and Already Retired Handling
# ---------------------------------------------------------------------------


def test_already_deleted_and_already_retired() -> None:
    """Verify deleted cycles are excluded and already-retired cycles do not re-retire."""
    c_old = _dt(2026, 9, 1, 12)
    c_t = _dt(2026, 9, 2, 0)

    # Tombstone: already deleted
    tombstone = ModelLifecycleSnapshot(
        model_id="gfs",
        cycle_time=c_old,
        deleted_at=_dt(2026, 9, 1, 18),
    )
    dec1 = evaluate_cycle_lifecycle(tombstone, c_t, timedelta(hours=6))
    assert dec1.is_eligible_for_deletion is False
    assert dec1.reason == "cycle_already_deleted"

    # Already retired: eligible for deletion, but should_retire is False (idempotent)
    retired_snap = ModelLifecycleSnapshot(
        model_id="gfs",
        cycle_time=c_old,
        retired_at=_dt(2026, 9, 1, 18),
        retired_by_cycle_time=_dt(2026, 9, 2, 0),
    )
    dec2 = evaluate_cycle_lifecycle(retired_snap, c_t, timedelta(hours=6))
    assert dec2.is_eligible_for_deletion is True
    assert dec2.should_retire is False


def test_no_ready_cycles_does_not_advance_lifecycle() -> None:
    """Verify that when no ready cycles exist, nothing is eligible for deletion."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    snapshots = [
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_00z, status="partial"),
        ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_06z, status="processing"),
    ]
    plan = plan_model_lifecycle("gfs", snapshots, ready_cycle_times=[])

    assert plan.latest_ready_cycle is None
    assert plan.cutoff is None
    assert plan.eligible_for_deletion == ()
    assert plan.active_visible_cycles == (c_00z, c_06z)
    assert len(plan.blocked) == 2


def test_plan_lifecycle_with_deleted_and_retired_snapshots() -> None:
    """Verify plan_model_lifecycle with deleted and retained-retired snapshots."""
    c_deleted = _dt(2026, 9, 1, 6)
    c_old_retired = _dt(2026, 9, 1, 12)
    c_retained = _dt(2026, 9, 1, 18)
    c_ready = _dt(2026, 9, 2, 0)

    snapshots = [
        ModelLifecycleSnapshot(
            model_id="gfs",
            cycle_time=c_deleted,
            deleted_at=_dt(2026, 9, 1, 12),
        ),
        ModelLifecycleSnapshot(
            model_id="gfs",
            cycle_time=c_old_retired,
            retired_at=_dt(2026, 9, 1, 18),
            status="ready",
        ),
        ModelLifecycleSnapshot(
            model_id="gfs",
            cycle_time=c_retained,
            retired_at=_dt(2026, 9, 1, 20),
            status="ready",
        ),
        ModelLifecycleSnapshot(
            model_id="gfs",
            cycle_time=c_ready,
            status="ready",
        ),
    ]
    from domain.lifecycle import plan_lifecycle

    plan = plan_lifecycle(snapshots, ready_cycle_times=[c_ready])

    assert isinstance(plan, ModelLifecyclePlan)
    assert CycleLifecycleSnapshot is ModelLifecycleSnapshot
    assert plan.deleted_cycles == (c_deleted,)
    assert c_old_retired in [d.cycle_time for d in plan.eligible_for_deletion]
    assert c_retained in plan.retired_cycles
    assert c_ready in plan.active_visible_cycles

    # Test property aliases
    assert len(plan.would_retire) >= 0
    assert len(plan.would_gc) == len(plan.eligible_for_deletion)
    assert len(plan.retirements) == len(plan.would_retire)
    assert len(plan.gc_eligibilities) == len(plan.decisions)


def test_naive_datetime_handling() -> None:
    """Verify naive datetimes are safely normalized to UTC."""
    c_naive = _dt(2026, 9, 2, 0, tz=False)
    snap = ModelLifecycleSnapshot(model_id="gfs", cycle_time=c_naive)
    assert snap.cycle_time.tzinfo is not None

    t, cutoff = compute_lifecycle_cutoff([c_naive], timedelta(hours=6))
    assert t is not None and t.tzinfo is not None


# ---------------------------------------------------------------------------
# Store Path & Snapshot Properties
# ---------------------------------------------------------------------------


def test_canonical_cycle_store_path() -> None:
    dt = _dt(2026, 9, 2, 18)
    assert (
        canonical_cycle_store_path("gfs", dt)
        == "s3://weather-data/gfs/2026-09-02/18/cycle.zarr"
    )
    assert (
        canonical_cycle_store_path("gefs", dt, base_bucket="custom")
        == "s3://custom/gefs/2026-09-02/18/cycle.zarr"
    )


def test_snapshot_properties() -> None:
    dt = _dt(2026, 9, 2, 0)
    snap = ModelLifecycleSnapshot(model_id="gfs", cycle_time=dt, status="ready")
    assert snap.is_ready is True
    assert snap.is_retired is False
    assert snap.is_deletion_started is False
    assert snap.is_deleted is False

    claimed = ModelLifecycleSnapshot(
        model_id="gfs",
        cycle_time=dt,
        deletion_started_at=dt,
    )
    assert claimed.is_deletion_started is True
