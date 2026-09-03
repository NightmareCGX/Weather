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

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ingestion.core.catalog import (
    EnsembleMemberProductRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
)
from ingestion.realtime.planner import ModelCommittedState


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
