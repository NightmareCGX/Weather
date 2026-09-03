"""Pure unit tests for domain.lifecycle R1/R2 math and retention planning."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from domain.lifecycle import (
    GC_SAFETY_DELTA,
    RETIREMENT_DELTA,
    CycleLifecycleSnapshot,
    evaluate_gc_eligibility,
    evaluate_retirement,
    find_r1,
    find_r2,
    plan_lifecycle,
)


def _dt(
    year: int, month: int, day: int, hour: int, tz: bool = True
) -> datetime:
    """Helper to build UTC datetime for tests."""
    return datetime(
        year, month, day, hour, 0, 0, tzinfo=timezone.utc if tz else None
    )


# ---------------------------------------------------------------------------
# R1 Tests: Earliest paired-ready >= C + 24h
# ---------------------------------------------------------------------------


def test_find_r1_exact_24h_boundary() -> None:
    c = _dt(2026, 9, 1, 6)
    r1_target = _dt(2026, 9, 2, 6)
    assert r1_target == c + RETIREMENT_DELTA

    result = find_r1(c, [r1_target])
    assert result == r1_target


def test_find_r1_same_hour_failed_later_ready_selected() -> None:
    c = _dt(2026, 9, 1, 6)
    # 09-02 06Z is absent/failed; 09-02 12Z is paired-ready
    later_ready = _dt(2026, 9, 2, 12)
    paired_ready = [
        _dt(2026, 9, 1, 12),  # C + 6h: too early
        _dt(2026, 9, 1, 18),  # C + 12h: too early
        _dt(2026, 9, 2, 0),   # C + 18h: too early
        later_ready,          # C + 30h: earliest qualifying
        _dt(2026, 9, 2, 18),  # C + 36h
    ]
    result = find_r1(c, paired_ready)
    assert result == later_ready


def test_find_r1_multiple_later_ready_returns_earliest() -> None:
    c = _dt(2026, 9, 1, 0)
    ready1 = _dt(2026, 9, 2, 0)  # C + 24h
    ready2 = _dt(2026, 9, 2, 6)  # C + 30h
    ready3 = _dt(2026, 9, 2, 12) # C + 36h
    result = find_r1(c, [ready3, ready1, ready2])
    assert result == ready1


def test_find_r1_only_newer_cycles_below_24h_returns_none() -> None:
    c = _dt(2026, 9, 1, 0)
    # All ready cycles are < C + 24h
    paired_ready = [
        _dt(2026, 9, 1, 6),   # +6h
        _dt(2026, 9, 1, 12),  # +12h
        _dt(2026, 9, 1, 18),  # +18h
    ]
    assert find_r1(c, paired_ready) is None


def test_find_r1_empty_or_no_later_returns_none() -> None:
    c = _dt(2026, 9, 1, 6)
    assert find_r1(c, []) is None
    assert find_r1(c, [_dt(2026, 9, 1, 0)]) is None


def test_find_r1_input_order_insensitive() -> None:
    c = _dt(2026, 9, 1, 6)
    t1 = _dt(2026, 9, 2, 6)
    t2 = _dt(2026, 9, 2, 12)
    t3 = _dt(2026, 9, 3, 0)
    assert find_r1(c, [t3, t1, t2]) == find_r1(c, [t1, t2, t3]) == t1


def test_find_r1_handles_naive_and_aware_datetimes() -> None:
    c_naive = _dt(2026, 9, 1, 6, tz=False)
    r1_aware = _dt(2026, 9, 2, 6, tz=True)
    res = find_r1(c_naive, [r1_aware])
    assert res is not None
    assert res == _dt(2026, 9, 2, 6, tz=True)


# ---------------------------------------------------------------------------
# R2 Tests: Earliest paired-ready >= R1 + 6h
# ---------------------------------------------------------------------------


def test_find_r2_exact_6h_boundary() -> None:
    r1 = _dt(2026, 9, 2, 12)
    r2_target = _dt(2026, 9, 2, 18)
    assert r2_target == r1 + GC_SAFETY_DELTA

    result = find_r2(r1, [r2_target])
    assert result == r2_target


def test_find_r2_immediate_next_cycle_unsuccessful_later_qualifies() -> None:
    r1 = _dt(2026, 9, 2, 12)
    # 09-02 18Z is missing/failed; 09-03 00Z is paired-ready
    r2_later = _dt(2026, 9, 3, 0)
    paired_ready = [
        _dt(2026, 9, 2, 12),  # R1 itself (delta = 0h)
        r2_later,             # R1 + 12h
        _dt(2026, 9, 3, 6),   # R1 + 18h
    ]
    result = find_r2(r1, paired_ready)
    assert result == r2_later


def test_find_r2_no_qualifying_returns_none() -> None:
    r1 = _dt(2026, 9, 2, 12)
    # Only R1 itself is known
    assert find_r2(r1, [r1]) is None
    assert find_r2(r1, []) is None


def test_find_r2_multiple_qualifying_returns_earliest() -> None:
    r1 = _dt(2026, 9, 2, 12)
    t1 = _dt(2026, 9, 2, 18)
    t2 = _dt(2026, 9, 3, 0)
    t3 = _dt(2026, 9, 3, 6)
    assert find_r2(r1, [t3, t2, t1]) == t1


# ---------------------------------------------------------------------------
# Regression Test: NO C + 30h shortcut
# ---------------------------------------------------------------------------


def test_no_c_plus_30h_shortcut_regression() -> None:
    """Prove that C is NOT GC-eligible at R1 merely because wall-clock/R1 >= C + 30h.

    Example:
      C = 09-01 06Z
      09-02 06Z failed / missing
      09-02 12Z paired-ready = R1  (Note: 09-02 12Z is C + 30h!)

    At 09-02 12Z:
      - C is RETIRED by R1 (09-02 12Z >= C + 24h).
      - C is NOT GC-eligible! R2 requires a paired-ready cycle >= R1 + 6h (09-02 18Z).
      - Having R1 at C + 30h does NOT satisfy R2.
    """
    c = _dt(2026, 9, 1, 6)
    r1 = _dt(2026, 9, 2, 12)
    assert r1 == c + timedelta(hours=30)

    # Only R1 is paired-ready so far
    paired_ready = [r1]

    # Evaluate retirement
    snapshot = CycleLifecycleSnapshot(cycle_time=c)
    ret_decision = evaluate_retirement(snapshot, paired_ready)
    assert ret_decision.should_retire is True
    assert ret_decision.retired_by_cycle_time == r1

    # Evaluate GC eligibility for newly retired cycle
    retired_snapshot = CycleLifecycleSnapshot(
        cycle_time=c,
        retired_at=_dt(2026, 9, 2, 12, 1),
        retired_by_cycle_time=r1,
    )
    gc_decision = evaluate_gc_eligibility(retired_snapshot, paired_ready)
    # MUST NOT be eligible yet!
    assert gc_decision.is_gc_eligible is False
    assert gc_decision.gc_eligible_by_cycle_time is None
    assert gc_decision.reason == "no_qualifying_r2_paired_ready"

    # Now simulate R2 arriving at 09-02 18Z (R1 + 6h)
    paired_ready_with_r2 = [r1, _dt(2026, 9, 2, 18)]
    gc_decision_with_r2 = evaluate_gc_eligibility(
        retired_snapshot, paired_ready_with_r2
    )
    assert gc_decision_with_r2.is_gc_eligible is True
    assert gc_decision_with_r2.gc_eligible_by_cycle_time == _dt(2026, 9, 2, 18)


# ---------------------------------------------------------------------------
# Old Cycle Status Invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["ready", "partial", "failed", "processing"])
def test_old_cycle_status_invariance(status: str) -> None:
    """Prove that C's own status does not change the lifecycle retirement or GC math."""
    c = _dt(2026, 9, 1, 6)
    r1 = _dt(2026, 9, 2, 6)
    r2 = _dt(2026, 9, 2, 12)
    paired_ready = [r1, r2]

    snapshot = CycleLifecycleSnapshot(
        cycle_time=c,
        gfs_status=status,
        gefs_status=status,
    )
    ret_decision = evaluate_retirement(snapshot, paired_ready)
    assert ret_decision.should_retire is True
    assert ret_decision.retired_by_cycle_time == r1

    retired_snapshot = CycleLifecycleSnapshot(
        cycle_time=c,
        retired_at=_dt(2026, 9, 2, 6, 1),
        retired_by_cycle_time=r1,
        gfs_status=status,
        gefs_status=status,
    )
    gc_decision = evaluate_gc_eligibility(retired_snapshot, paired_ready)
    assert gc_decision.is_gc_eligible is True
    assert gc_decision.gc_eligible_by_cycle_time == r2


# ---------------------------------------------------------------------------
# Already Retired and Deleted Handlings
# ---------------------------------------------------------------------------


def test_evaluate_retirement_already_retired_or_deleted() -> None:
    c = _dt(2026, 9, 1, 6)
    paired_ready = [_dt(2026, 9, 2, 6)]

    retired_snap = CycleLifecycleSnapshot(
        cycle_time=c,
        retired_at=_dt(2026, 9, 2, 6),
        retired_by_cycle_time=_dt(2026, 9, 2, 6),
    )
    res1 = evaluate_retirement(retired_snap, paired_ready)
    assert res1.should_retire is False
    assert res1.reason == "cycle_already_retired"

    deleted_snap = CycleLifecycleSnapshot(
        cycle_time=c,
        retired_at=_dt(2026, 9, 2, 6),
        retired_by_cycle_time=_dt(2026, 9, 2, 6),
        deleted_at=_dt(2026, 9, 2, 12),
    )
    res2 = evaluate_retirement(deleted_snap, paired_ready)
    assert res2.should_retire is False
    assert res2.reason == "cycle_already_deleted"


def test_evaluate_gc_eligibility_not_retired_or_deleted() -> None:
    c = _dt(2026, 9, 1, 6)
    paired_ready = [_dt(2026, 9, 2, 6), _dt(2026, 9, 2, 12)]

    visible_snap = CycleLifecycleSnapshot(cycle_time=c)
    res1 = evaluate_gc_eligibility(visible_snap, paired_ready)
    assert res1.is_gc_eligible is False
    assert res1.reason == "cycle_not_retired"

    deleted_snap = CycleLifecycleSnapshot(
        cycle_time=c,
        retired_at=_dt(2026, 9, 2, 6),
        retired_by_cycle_time=_dt(2026, 9, 2, 6),
        deleted_at=_dt(2026, 9, 2, 12),
    )
    res2 = evaluate_gc_eligibility(deleted_snap, paired_ready)
    assert res2.is_gc_eligible is False
    assert res2.reason == "cycle_already_deleted"


# ---------------------------------------------------------------------------
# Batch Lifecycle Planning (plan_lifecycle)
# ---------------------------------------------------------------------------


def test_plan_lifecycle_mixed_timeline() -> None:
    c0 = _dt(2026, 9, 1, 0)   # Already retired by 09-02 00Z, now GC eligible by 09-02 06Z
    c1 = _dt(2026, 9, 1, 6)   # Planned to retire (R1=09-02 06Z) and GC eligible (R2=09-02 12Z)
    c2 = _dt(2026, 9, 1, 12)  # Planned to retire (R1=09-02 12Z) but not GC eligible yet
    c3 = _dt(2026, 9, 1, 18)  # Active visible: requires >= 09-02 18Z (not in ready set)
    c4 = _dt(2026, 9, 2, 0)   # Active visible
    c5 = _dt(2026, 9, 2, 6)   # Active visible
    c6 = _dt(2026, 9, 2, 12)  # Active visible
    tombstone = _dt(2026, 8, 31, 18)  # Already deleted

    snapshots = [
        CycleLifecycleSnapshot(
            cycle_time=tombstone,
            retired_at=_dt(2026, 9, 1, 18),
            retired_by_cycle_time=_dt(2026, 9, 1, 18),
            deleted_at=_dt(2026, 9, 2, 0),
        ),
        CycleLifecycleSnapshot(
            cycle_time=c0,
            retired_at=_dt(2026, 9, 2, 0),
            retired_by_cycle_time=_dt(2026, 9, 2, 0),
        ),
        CycleLifecycleSnapshot(cycle_time=c1),
        CycleLifecycleSnapshot(cycle_time=c2),
        CycleLifecycleSnapshot(cycle_time=c3),
        CycleLifecycleSnapshot(cycle_time=c4, gfs_status="ready", gefs_status="ready"),
        CycleLifecycleSnapshot(cycle_time=c5, gfs_status="ready", gefs_status="ready"),
        CycleLifecycleSnapshot(cycle_time=c6, gfs_status="ready", gefs_status="ready"),
    ]

    paired_ready = [c4, c5, c6]  # 09-02 00Z, 09-02 06Z, 09-02 12Z

    plan = plan_lifecycle(snapshots, paired_ready)

    # 1. Check retirements (newly planned)
    assert len(plan.retirements) == 2
    r_map = {r.cycle_time: r.retired_by_cycle_time for r in plan.retirements}
    assert r_map[c1] == c5  # 09-02 06Z >= 09-01 06Z + 24h
    assert r_map[c2] == c6  # 09-02 12Z >= 09-01 12Z + 24h

    # 2. Check GC eligibility decisions
    gc_map = {g.cycle_time: g.is_gc_eligible for g in plan.gc_eligibilities}
    assert gc_map[c0] is True   # R1=09-02 00Z, R2=09-02 06Z (>= 00Z + 6h)
    assert gc_map[c1] is True   # R1=09-02 06Z, R2=09-02 12Z (>= 06Z + 6h)
    assert gc_map[c2] is False  # R1=09-02 12Z, R2 needs >= 09-02 18Z (not present)

    # 3. Check categories
    assert plan.deleted_cycles == (tombstone,)
    assert plan.active_visible_cycles == (c3, c4, c5, c6)
    assert plan.retired_cycles == (c0, c1, c2)


def test_snapshot_properties() -> None:
    dt = _dt(2026, 9, 1, 0)
    snap = CycleLifecycleSnapshot(
        cycle_time=dt,
        gfs_status="ready",
        gefs_status="ready",
    )
    assert snap.is_paired_ready is True
    assert snap.is_retired is False
    assert snap.is_deleted is False

    partial_snap = CycleLifecycleSnapshot(
        cycle_time=dt,
        gfs_status="ready",
        gefs_status="partial",
    )
    assert partial_snap.is_paired_ready is False
