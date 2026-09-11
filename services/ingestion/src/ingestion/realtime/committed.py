"""Durable committed-state reader for the realtime scheduler (narrow read path).

Reconstructs what is durably committed for one cycle **from the catalog** —
the same PostgreSQL truth the serving tier uses — with two small indexed
queries per model (run row by ``(model_id, cycle_time)``, then the committed
lead / member-pair rows). No physical Zarr/object-store scans happen per poll;
the store-side marker evidence remains the ingestion finalizer's concern and
is reconciled into this catalog by every wave.

The reader is deliberately tolerant of asymmetric progress: GFS may be
committed ahead of GEFS or vice versa (big-batch commits between polls count
too) — each model's state is read independently and the planner combines them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from domain.horizon import CANONICAL_MAX_LEAD_HOURS
from domain.temporal import serving_start_valid_time
from ingestion.core.catalog import (
    EnsembleMemberProductRecord,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
)
from ingestion.realtime.planner import ModelCommittedState

if TYPE_CHECKING:
    from ingestion.realtime.scheduler import CycleIdentity


def _read_model_committed_state(
    db: Session, *, model_id: str, cycle_time: datetime, version_string: str
) -> ModelCommittedState:
    """Read one model's committed leads/pairs for a cycle (empty when absent)."""
    version_id = db.execute(
        select(ModelVersionRecord.id).where(
            (ModelVersionRecord.model_id == model_id)
            & (ModelVersionRecord.version_string == version_string)
        )
    ).scalar_one_or_none()
    if version_id is None:
        return ModelCommittedState()
    run_id = db.execute(
        select(ModelRunRecord.id).where(
            (ModelRunRecord.model_version_id == version_id)
            & (ModelRunRecord.cycle_time == cycle_time)
        )
    ).scalar_one_or_none()
    if run_id is None:
        return ModelCommittedState()

    leads = frozenset(
        int(lead)
        for lead in db.execute(
            select(ProductRecord.lead_time_hours).where(
                ProductRecord.run_id == run_id
            )
        ).scalars()
    )
    pairs = frozenset(
        (int(member), int(lead))
        for member, lead in db.execute(
            select(
                EnsembleMemberProductRecord.member_index,
                EnsembleMemberProductRecord.lead_time_hours,
            ).where(EnsembleMemberProductRecord.run_id == run_id)
        ).all()
    )
    return ModelCommittedState(leads=leads, pairs=pairs)


def read_cycle_committed_state(
    engine: Engine,
    *,
    cycle_time: datetime,
    version_string: str = "v1.0",
) -> tuple[ModelCommittedState, ModelCommittedState]:
    """Read the durable committed state of both models for one cycle.

    Args:
        engine: The catalog engine (PostgreSQL in production).
        cycle_time: The UTC cycle time shared by both models (never paired
            across different cycle timestamps).
        version_string: The model version string the runs were recorded under.

    Returns:
        ``(gfs_state, gefs_state)`` — empty states when a model has no run row
        yet (nothing committed for that model's cycle).
    """
    with Session(engine) as db:
        gfs = _read_model_committed_state(
            db, model_id="gfs", cycle_time=cycle_time, version_string=version_string
        )
        gefs = _read_model_committed_state(
            db, model_id="gefs", cycle_time=cycle_time, version_string=version_string
        )
    return gfs, gefs


def is_cycle_durably_complete(
    engine_or_db: Engine | Session,
    *,
    cycle_time: datetime,
    version_string: str = "v1.0",
) -> bool:
    """Return whether cycle_time has achieved formal ready commit completeness for both models.

    A cycle is durably complete if and only if both GFS and GEFS model runs exist
    under the specified version and both have status == 'ready'.
    """
    if isinstance(engine_or_db, Session):
        return _is_cycle_durably_complete_db(
            engine_or_db, cycle_time=cycle_time, version_string=version_string
        )
    with Session(engine_or_db) as db:
        return _is_cycle_durably_complete_db(
            db, cycle_time=cycle_time, version_string=version_string
        )


def _is_cycle_durably_complete_db(
    db: Session,
    *,
    cycle_time: datetime,
    version_string: str,
) -> bool:
    c_utc = (
        cycle_time.replace(tzinfo=timezone.utc)
        if cycle_time.tzinfo is None
        else cycle_time.astimezone(timezone.utc)
    )
    stmt = (
        select(ModelVersionRecord.model_id, ModelRunRecord.status)
        .join(
            ModelVersionRecord,
            ModelRunRecord.model_version_id == ModelVersionRecord.id,
        )
        .where(
            ModelVersionRecord.version_string == version_string,
            ModelVersionRecord.model_id.in_(["gfs", "gefs"]),
            ModelRunRecord.cycle_time == c_utc,
        )
    )
    raw_rows = db.execute(stmt).all()
    rows: dict[str, str] = {str(m_id): str(status) for m_id, status in raw_rows}
    return rows.get("gfs") == "ready" and rows.get("gefs") == "ready"


def discover_incomplete_historical_cycles(
    engine_or_db: Engine | Session,
    *,
    active_cycle_time: datetime,
    now_utc: datetime,
    version_string: str = "v1.0",
) -> list[CycleIdentity]:
    """Discover incomplete historical cycles eligible for backlog recovery.

    A cycle is an eligible historical backlog candidate if:
    1. cycle_time < active_cycle_time
    2. Durable catalog evidence exists (at least one model has a model_runs row)
    3. Cycle is not durably complete (both GFS and GEFS ready)
    4. Cycle is not tombstoned/fenced for deletion in forecast_cycle_lifecycle
    5. Cycle remains in the useful forecast horizon:
       cycle_time >= serving_start_valid_time(now_utc) - CANONICAL_MAX_LEAD_HOURS

    Returns:
        List of CycleIdentity objects sorted by cycle_time descending (newest first).
    """
    if isinstance(engine_or_db, Session):
        return _discover_incomplete_historical_cycles_db(
            engine_or_db,
            active_cycle_time=active_cycle_time,
            now_utc=now_utc,
            version_string=version_string,
        )
    with Session(engine_or_db) as db:
        return _discover_incomplete_historical_cycles_db(
            db,
            active_cycle_time=active_cycle_time,
            now_utc=now_utc,
            version_string=version_string,
        )


def is_cycle_retired_or_deleted(
    db: Session, cycle_time: datetime, model_id: str | None = None
) -> bool:
    """Return whether cycle_time is logically retired or claimed/tombstoned for deletion.

    A cycle with retired_at IS NOT NULL, deletion_started_at IS NOT NULL, or
    deleted_at IS NOT NULL is excluded from scheduling recovery.
    """
    c_utc = (
        cycle_time.replace(tzinfo=timezone.utc)
        if cycle_time.tzinfo is None
        else cycle_time.astimezone(timezone.utc)
    )
    if model_id is not None:
        m_id = model_id.lower().strip()
        row = db.get(ForecastCycleLifecycleRecord, (m_id, c_utc))
        if row is None:
            return False
        return (
            row.retired_at is not None
            or row.deletion_started_at is not None
            or row.deleted_at is not None
        )

    rows = db.execute(
        select(
            ForecastCycleLifecycleRecord.retired_at,
            ForecastCycleLifecycleRecord.deletion_started_at,
            ForecastCycleLifecycleRecord.deleted_at,
        ).where(ForecastCycleLifecycleRecord.cycle_time == c_utc)
    ).all()
    return any(
        r is not None or s is not None or d is not None for r, s, d in rows
    )


def _discover_incomplete_historical_cycles_db(
    db: Session,
    *,
    active_cycle_time: datetime,
    now_utc: datetime,
    version_string: str,
) -> list[CycleIdentity]:
    from ingestion.realtime.scheduler import CycleIdentity

    act_utc = (
        active_cycle_time.replace(tzinfo=timezone.utc)
        if active_cycle_time.tzinfo is None
        else active_cycle_time.astimezone(timezone.utc)
    )
    now_tz = (
        now_utc.replace(tzinfo=timezone.utc)
        if now_utc.tzinfo is None
        else now_utc.astimezone(timezone.utc)
    )
    serving_start = serving_start_valid_time(now_tz)
    min_cycle_time = serving_start - timedelta(hours=CANONICAL_MAX_LEAD_HOURS)

    stmt = (
        select(
            ModelRunRecord.cycle_time,
            ModelVersionRecord.model_id,
            ModelRunRecord.status,
        )
        .join(
            ModelVersionRecord,
            ModelRunRecord.model_version_id == ModelVersionRecord.id,
        )
        .where(
            ModelVersionRecord.version_string == version_string,
            ModelVersionRecord.model_id.in_(["gfs", "gefs"]),
            ModelRunRecord.cycle_time < act_utc,
            ModelRunRecord.cycle_time >= min_cycle_time,
        )
    )
    rows = db.execute(stmt).all()

    cycle_model_statuses: dict[datetime, dict[str, str]] = {}
    for c_time, m_id, status in rows:
        c_utc = (
            c_time.replace(tzinfo=timezone.utc)
            if c_time.tzinfo is None
            else c_time.astimezone(timezone.utc)
        )
        cycle_model_statuses.setdefault(c_utc, {})[str(m_id)] = str(status)

    candidates: list[CycleIdentity] = []
    for c_utc in sorted(cycle_model_statuses.keys(), reverse=True):
        statuses = cycle_model_statuses[c_utc]

        # Exclude if durably complete
        if statuses.get("gfs") == "ready" and statuses.get("gefs") == "ready":
            continue

        # Exclude if lifecycle retired, deletion started, or deleted
        if is_cycle_retired_or_deleted(db, c_utc):
            continue

        # Exclude if horizon expired
        if c_utc + timedelta(hours=CANONICAL_MAX_LEAD_HOURS) < serving_start:
            continue

        candidates.append(
            CycleIdentity(cycle_date=c_utc.date(), cycle_hour=c_utc.hour)
        )

    return candidates
