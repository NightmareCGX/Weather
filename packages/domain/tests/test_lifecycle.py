"""Pure unit tests for domain.lifecycle (Data Lifecycle V3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from domain.lifecycle import (
    METADATA_RETENTION_DAYS,
    ModelLifecycleSnapshot,
    canonical_cycle_store_path,
    is_cycle_horizon_expired,
    is_metadata_purge_eligible,
)


def _dt(year: int, month: int, day: int, hour: int, tz: bool = True) -> datetime:
    """Helper to build UTC datetime for tests."""
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc if tz else None)


# ---------------------------------------------------------------------------
# Store Path Tests
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


# ---------------------------------------------------------------------------
# ModelLifecycleSnapshot Tests
# ---------------------------------------------------------------------------


def test_snapshot_properties() -> None:
    dt = _dt(2026, 9, 2, 0)
    snap = ModelLifecycleSnapshot(model_id="GFS ", cycle_time=dt, status="ready")
    assert snap.model_id == "gfs"
    assert snap.is_ready is True
    assert snap.is_deletion_started is False
    assert snap.is_deleted is False

    claimed = ModelLifecycleSnapshot(
        model_id="gfs",
        cycle_time=dt,
        deletion_started_at=dt,
    )
    assert claimed.is_deletion_started is True
    assert claimed.is_deleted is False

    deleted = ModelLifecycleSnapshot(
        model_id="gfs",
        cycle_time=dt,
        deleted_at=dt,
    )
    assert deleted.is_deletion_started is False
    assert deleted.is_deleted is True


def test_snapshot_naive_datetime_handling() -> None:
    """Verify naive datetimes are safely normalized to UTC in __post_init__."""
    c_naive = _dt(2026, 9, 2, 0, tz=False)
    claim_naive = _dt(2026, 9, 2, 6, tz=False)
    del_naive = _dt(2026, 9, 2, 7, tz=False)

    snap = ModelLifecycleSnapshot(
        model_id="gfs",
        cycle_time=c_naive,
        deletion_started_at=claim_naive,
        deleted_at=del_naive,
    )
    assert snap.cycle_time.tzinfo is not None
    assert snap.deletion_started_at is not None
    assert snap.deletion_started_at.tzinfo is not None
    assert snap.deleted_at is not None
    assert snap.deleted_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Strict Horizon Expiry Tests
# ---------------------------------------------------------------------------


def test_is_cycle_horizon_expired_strict_boundary() -> None:
    """Verify strict < comparison for cycle horizon expiry.

    cycle_time + max_lead == serving_start -> False (NOT expired)
    cycle_time + max_lead < serving_start  -> True (expired)
    cycle_time + max_lead > serving_start  -> False (NOT expired)
    """
    serving_start = _dt(2026, 9, 10, 6)
    max_lead = 240

    # Exact boundary: cycle_time + 240h == 2026-09-10 06:00:00 -> NOT expired
    exact_cycle = serving_start - timedelta(hours=max_lead)
    assert (
        is_cycle_horizon_expired(
            exact_cycle, max_lead_hours=max_lead, serving_start=serving_start
        )
        is False
    )

    # 1 second past boundary (< serving_start): EXPIRED
    expired_cycle_1s = exact_cycle - timedelta(seconds=1)
    assert (
        is_cycle_horizon_expired(
            expired_cycle_1s, max_lead_hours=max_lead, serving_start=serving_start
        )
        is True
    )

    # 1 hour past boundary: EXPIRED
    expired_cycle_1h = exact_cycle - timedelta(hours=1)
    assert (
        is_cycle_horizon_expired(
            expired_cycle_1h, max_lead_hours=max_lead, serving_start=serving_start
        )
        is True
    )

    # 1 second before boundary (> serving_start): NOT expired
    future_cycle_1s = exact_cycle + timedelta(seconds=1)
    assert (
        is_cycle_horizon_expired(
            future_cycle_1s, max_lead_hours=max_lead, serving_start=serving_start
        )
        is False
    )

    # Naive datetime handling
    naive_cycle = exact_cycle.replace(tzinfo=None)
    naive_serving = serving_start.replace(tzinfo=None)
    assert (
        is_cycle_horizon_expired(
            naive_cycle, max_lead_hours=max_lead, serving_start=naive_serving
        )
        is False
    )


# ---------------------------------------------------------------------------
# Metadata Purge Eligibility Tests
# ---------------------------------------------------------------------------


def test_is_metadata_purge_eligible_inclusive_boundary() -> None:
    """Verify inclusive <= comparison and locked 14-day policy for metadata purge.

    deleted_at is None                 -> False
    deleted_at == now_utc - 14 days    -> True (inclusive <= boundary)
    deleted_at < now_utc - 14 days     -> True
    deleted_at > now_utc - 14 days     -> False (must be retained)
    """
    assert METADATA_RETENTION_DAYS == 14

    now_utc = _dt(2026, 9, 25, 12)
    boundary_14d = now_utc - timedelta(days=14)

    # deleted_at is None
    assert is_metadata_purge_eligible(None, now_utc=now_utc) is False

    # Exact 14-day boundary: eligible (inclusive <=)
    assert is_metadata_purge_eligible(boundary_14d, now_utc=now_utc) is True

    # Older than 14 days (e.g. 15 days, or 14d + 1s ago): eligible
    assert is_metadata_purge_eligible(boundary_14d - timedelta(seconds=1), now_utc=now_utc) is True
    assert is_metadata_purge_eligible(boundary_14d - timedelta(days=1), now_utc=now_utc) is True

    # Younger than 14 days (e.g. 14d - 1s ago): NOT eligible
    assert is_metadata_purge_eligible(boundary_14d + timedelta(seconds=1), now_utc=now_utc) is False
    assert is_metadata_purge_eligible(now_utc - timedelta(days=7), now_utc=now_utc) is False

    # Naive datetime handling
    assert (
        is_metadata_purge_eligible(
            boundary_14d.replace(tzinfo=None), now_utc=now_utc.replace(tzinfo=None)
        )
        is True
    )
