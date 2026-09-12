"""Pure domain forecast cycle lifecycle and retention planning math (Lifecycle V3).

This module is the single source of truth for the platform's Data Lifecycle V3
policy. It is pure, deterministic, and side-effect-free: it contains no
SQLAlchemy, database connections, or object-store calls.

Lifecycle V3 Policy (Locked Contract):
--------------------------------------
1. Physical Deletion Fencing & Tombstones:
   Authoritative lifecycle state is exclusively governed by:
   - ``deletion_started_at``: durable physical deletion claim / serving & mutation fence
   - ``deleted_at``: permanent anti-resurrection tombstone
   - ``reclamation_queue``: granular variable shard reclamation

2. Multi-Version Conservative Horizon Expiry:
   Whole-cycle physical end-of-life eligibility is strictly conservative across
   all versions:
   cycle_time + max_lead_hours < serving_start_valid_time(now)
   Exact equality means NOT expired.

3. Detailed Metadata Retention:
   Detailed catalog metadata (model_runs, forecast_products, etc.) is retained
   for 14 days after physical deletion (deleted_at) before being purged by the
   metadata sweeper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

#: Locked product policy: detailed metadata is retained for 14 days after physical deletion.
METADATA_RETENTION_DAYS: int = 14


def is_cycle_horizon_expired(
    cycle_time: datetime,
    *,
    max_lead_hours: int,
    serving_start: datetime,
) -> bool:
    """Return True strictly if cycle_time + max_lead_hours < serving_start.

    Strict boundary:
    cycle_time + max_lead_hours == serving_start -> False (NOT expired)
    cycle_time + max_lead_hours < serving_start  -> True (expired)
    """
    c_utc = _ensure_utc(cycle_time)
    s_utc = _ensure_utc(serving_start)
    return c_utc + timedelta(hours=max_lead_hours) < s_utc


def is_metadata_purge_eligible(
    deleted_at: datetime | None,
    *,
    now_utc: datetime,
) -> bool:
    """Return True if detailed metadata for a cycle is eligible for retention purge.

    Locked policy contract:
    - deleted_at is None -> False (cycle not yet physically deleted)
    - deleted_at <= now_utc - 14 days -> True (eligible)
    - deleted_at > now_utc - 14 days -> False (must be retained)
    """
    if deleted_at is None:
        return False
    d_utc = _ensure_utc(deleted_at)
    n_utc = _ensure_utc(now_utc)
    return d_utc <= n_utc - timedelta(days=METADATA_RETENTION_DAYS)


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


@dataclass(frozen=True)
class ModelLifecycleSnapshot:
    """Snapshot of a single model cycle's durable physical lifecycle and run status.

    Attributes:
        model_id: Platform model identifier (e.g. 'gfs', 'gefs').
        cycle_time: The logical UTC cycle datetime.
        status: Model run status if known (e.g. 'ready', 'partial', 'failed', 'processing').
        deletion_started_at: Timestamp when physical deletion claimed the cycle, or None.
        deleted_at: Timestamp when physical deletion completed (tombstone), or None.
    """

    model_id: str
    cycle_time: datetime
    status: str | None = None
    deletion_started_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", self.model_id.lower().strip())
        object.__setattr__(self, "cycle_time", _ensure_utc(self.cycle_time))
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
    def is_deletion_started(self) -> bool:
        """Whether physical deletion has been claimed/started (durable fence)."""
        return self.deletion_started_at is not None

    @property
    def is_deleted(self) -> bool:
        """Whether this cycle has been physically deleted (tombstone)."""
        return self.deleted_at is not None
