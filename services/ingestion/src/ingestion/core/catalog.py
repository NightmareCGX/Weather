"""PostgreSQL forecast catalog writer for the ingestion worker.

After a GRIB2 file has been parsed and written to a Zarr store, the run must
be recorded in the PostgreSQL catalog so the API serving tier can discover and
serve it (``/v1/runs``, ``/v1/points``, ``/v1/ensembles``, ...). This module
owns that write path.

The ORM models here deliberately mirror the subset of the Milestone 3 schema
that the write path needs (``services/api/src/api/models/entities.py``) rather
than importing the API service's models: the ingestion worker must stay
independent of the serving tier, and neither service should import the other
(``ARCHITECTURE.md`` sections 3.1/3.5). The Alembic migration in
``services/api`` remains the single source of truth for the DDL; this module
only inserts/updates rows through SQLAlchemy.

The entry point is :func:`record_run`, which upserts a ``ready`` run plus its
model/version/center/variable/grid/product/ensemble-member catalog rows. A
worker calls it immediately after a successful Zarr write:

    dataset = parse_grib2(path)
    store = write_dataset(dataset, spec.zarr_store_path)
    record_run(session, spec, dataset)

:func:`record_ingested_dataset` wraps that with the configured engine for
worker convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import xarray as xr
from sqlalchemy import (
    Boolean,
    Column,
    ColumnElement,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ingestion.core.db import CatalogBase, SessionLocal

#: Maximum attempts to record a run when a concurrent worker commits the same
#: (model, cycle) run between our SELECT and INSERT. The writes are
#: idempotent, so a bounded retry converges; a genuine data-integrity failure
#: re-raises after the final attempt.
_MAX_CONCURRENT_WRITE_RETRIES = 3


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CommittedState:
    """The actual committed forecast contents of a run's Zarr store.

    This is the source of truth for catalog reconciliation and for the
    store↔catalog READY consistency gate. It is derived from **real data
    regions**, never from the preallocated coordinate axis: a cycle store is
    pre-allocated with the full expected lead (and member) axis NaN-filled, so
    a lead/member is only "committed" when its region actually holds non-NaN
    forecast data. The detection helper lives in the pipeline layer
    (``ingestion.core.pipeline.read_committed_state``), which reads the store
    and builds one of these values.

    Attributes:
        leads: The set of committed lead times (hours). For deterministic
            runs this is the committed lead set.
        members: The set of committed member indices (real upstream member
            identity). ``None`` for deterministic runs.
        pairs: The set of committed ``(member, lead)`` pairs for ensemble
            runs. ``None`` for deterministic runs.
        is_ensemble: Whether the store carries a ``member`` axis.
        variables: The set of data variables actually present in the store
            (the run's real store contents). ``None`` when unknown (legacy
            callers that never read the store); when known it is the source of
            truth for catalog ↔ store variable honesty — products are only
            restored for variables the store actually carries, and products for
            store-absent variables are stale and deleted.
    """

    leads: frozenset[int]
    members: frozenset[int] | None
    pairs: frozenset[tuple[int, int]] | None
    is_ensemble: bool
    variables: frozenset[str] | None = None

    @classmethod
    def deterministic(
        cls, leads: set[int], variables: set[str] | None = None
    ) -> CommittedState:
        """Build the committed state of a deterministic (non-ensemble) store.

        Args:
            leads: The committed lead set.
            variables: The store's data-variable set, when known.
        """
        return cls(
            leads=frozenset(leads),
            members=None,
            pairs=None,
            is_ensemble=False,
            variables=frozenset(variables) if variables is not None else None,
        )

    @classmethod
    def ensemble(
        cls,
        pairs: set[tuple[int, int]],
        members: set[int],
        variables: set[str] | None = None,
    ) -> CommittedState:
        """Build the committed state of an ensemble store.

        Args:
            pairs: The committed ``(member, lead)`` pairs.
            members: The member indices that have at least one committed pair.
            variables: The store's data-variable set, when known.
        """
        return cls(
            leads=frozenset(lead for _, lead in pairs),
            members=frozenset(members),
            pairs=frozenset(pairs),
            is_ensemble=True,
            variables=frozenset(variables) if variables is not None else None,
        )

    def lead_set(self) -> set[int]:
        """Return the committed lead set (shared by both store kinds)."""
        return set(self.leads)

    def member_set(self) -> set[int]:
        """Return the committed member set (empty for deterministic runs)."""
        return set(self.members or ())


def is_live_run_store(db: Session, store_path: str) -> bool:
    """Return whether ``store_path`` is referenced by a live ``model_runs`` row.

    A "live" run is any ``model_runs`` row whose ``zarr_store_path`` equals the
    target. Overwriting such a store with a full ``mode="w"`` rebuild would
    silently replace/shrink the run's contents without catalog reconciliation,
    recreating the stale ``forecast_products`` debt. The orchestration layer
    uses this to guard the full-overwrite helpers at the pipeline boundary.

    Args:
        db: Database session.
        store_path: The store path/URL to check.

    Returns:
        True when at least one ``model_runs`` row references the path.
    """
    row = db.execute(
        select(ModelRunRecord.id).where(ModelRunRecord.zarr_store_path == store_path)
    ).scalars().first()
    return row is not None


def is_ready_run_store(db: Session, store_path: str) -> bool:
    """Return whether ``store_path`` is referenced by a **ready** ``model_runs`` row.

    A ``ready`` run is quiescent and committed: its store carries the content
    the catalog advertises. A ``ready`` row whose store is missing is an
    external shrink / corruption condition that must never be silently repaired
    by a cold-start full overwrite — doing so would replace the run's contents
    without catalog reconciliation.

    Non-ready rows (``processing``/``partial``) are placeholders from an
    in-flight or aborted first ingestion: their store may be genuinely absent
    (the run was recorded before any committed region existed — e.g. a
    region-write failure). Cold-start initializing such a store is the legitimate
    recovery path, so those rows are NOT "live" for the full-overwrite guard.

    Args:
        db: Database session.
        store_path: The store path/URL to check.

    Returns:
        True when at least one **ready** ``model_runs`` row references the path.
    """
    row = db.execute(
        select(ModelRunRecord.id).where(
            (ModelRunRecord.zarr_store_path == store_path)
            & (ModelRunRecord.status == "ready")
        )
    ).scalars().first()
    return row is not None


def is_cycle_fenced_or_deleted(
    db: Session, cycle_time: datetime, model_id: str | None = None
) -> bool:
    """Return whether ``cycle_time`` is claimed for deletion or already tombstoned.

    A cycle with ``deletion_started_at IS NOT NULL`` or ``deleted_at IS NOT NULL``
    is fenced against new ingestion writes to prevent recreating physical stores
    during or after physical GC.
    """
    c_utc = _ensure_utc_datetime(cycle_time)
    if model_id is not None:
        m_id = model_id.lower().strip()
        row = db.get(ForecastCycleLifecycleRecord, (m_id, c_utc))
        if row is None:
            return False
        return row.deletion_started_at is not None or row.deleted_at is not None

    rows = db.execute(
        select(
            ForecastCycleLifecycleRecord.deletion_started_at,
            ForecastCycleLifecycleRecord.deleted_at,
        ).where(ForecastCycleLifecycleRecord.cycle_time == c_utc)
    ).all()
    return any(s is not None or d is not None for s, d in rows)


def is_cycle_tombstoned(
    db: Session, cycle_time: datetime, model_id: str | None = None
) -> bool:
    """Return whether ``cycle_time`` has a deletion fence or deleted_at tombstone."""
    return is_cycle_fenced_or_deleted(db, cycle_time, model_id=model_id)




class CenterRecord(CatalogBase):
    __tablename__ = "forecast_centers"

    id = Column(String, primary_key=True)
    center_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    country = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ModelRecord(CatalogBase):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    model_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    center_id = Column(String, ForeignKey("forecast_centers.center_id"), nullable=False)
    is_ensemble = Column(Boolean, default=False, nullable=False)
    resolution_km = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ModelVersionRecord(CatalogBase):
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.model_id"), nullable=False)
    version_string = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("model_id", "version_string", name="uq_model_version"),
    )


class ModelRunRecord(CatalogBase):
    __tablename__ = "model_runs"

    id = Column(String, primary_key=True)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)
    cycle_time = Column(DateTime(timezone=True), nullable=False)
    status = Column(String, nullable=False, default="processing")
    zarr_store_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("model_version_id", "cycle_time", name="uq_model_run_cycle"),
    )


class EnsembleMemberRecord(CatalogBase):
    __tablename__ = "ensemble_members"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("model_runs.id"), nullable=False)
    member_index = Column(Integer, nullable=False)
    member_name = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "member_index", name="uq_ensemble_member_index"),
    )


class EnsembleMemberProductRecord(CatalogBase):
    """One committed ``(member, lead)`` pair of an ensemble run.

    The catalog's ``forecast_products`` rows record lead completion *without*
    member identity, and ``ensemble_members`` rows record member presence
    *without* lead identity. Neither can answer "has member 3 committed lead 6".
    This table records exactly the committed ``(member_index, lead_time_hours)``
    pairs, one row per per-file ingest, so run-level readiness can enforce the
    Cartesian product of expected members × expected leads.
    """

    __tablename__ = "ensemble_member_products"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("model_runs.id"), nullable=False)
    member_index = Column(Integer, nullable=False)
    lead_time_hours = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "member_index",
            "lead_time_hours",
            name="uq_ensemble_member_product",
        ),
    )


class ForecastCycleLifecycleRecord(CatalogBase):
    __tablename__ = "forecast_cycle_lifecycle"

    model_id = Column(
        String, ForeignKey("models.model_id", ondelete="CASCADE"), primary_key=True
    )
    cycle_time = Column(DateTime(timezone=True), primary_key=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)
    retired_by_cycle_time = Column(DateTime(timezone=True), nullable=True)
    deletion_started_at = Column(DateTime(timezone=True), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)



class VariableRecord(CatalogBase):
    __tablename__ = "forecast_variables"

    id = Column(String, primary_key=True)
    variable_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    unit = Column(String, nullable=False)


class GridRecord(CatalogBase):
    __tablename__ = "forecast_grids"

    id = Column(String, primary_key=True)
    grid_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    resolution_km = Column(Float, nullable=False)


class ProductRecord(CatalogBase):
    __tablename__ = "forecast_products"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("model_runs.id"), nullable=False)
    variable_id = Column(
        String, ForeignKey("forecast_variables.variable_code"), nullable=False
    )
    grid_id = Column(String, ForeignKey("forecast_grids.grid_code"), nullable=False)
    product_type = Column(String, nullable=False)
    lead_time_hours = Column(Integer, nullable=False)
    zarr_chunk_path = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "variable_id",
            "grid_id",
            "product_type",
            "lead_time_hours",
            name="uq_forecast_product_coords",
        ),
    )


@dataclass(frozen=True)
class VariableSpec:
    """Forecast variable metadata recorded into ``forecast_variables``.

    Attributes:
        code: The platform ``variable_code`` (e.g. ``temperature_2m``).
        name: Human-readable variable name.
        unit: SI unit string (e.g. ``°C``).
        source_code: The raw upstream (GRIB2 ``shortName``) variable name this
            platform code is mapped from, used by the ingestion orchestration
            layer to rename a parsed dataset's data variables to the platform
            vocabulary. ``None`` (the default) means the platform code is used
            directly.
    """

    code: str
    name: str
    unit: str
    source_code: str | None = None


@dataclass(frozen=True)
class RunCatalogSpec:
    """The catalog metadata of a single ingested forecast run.

    Attributes:
        center_id: Natural key of the ``forecast_centers`` row.
        center_name: Display name of the center.
        center_country: Country of the center.
        model_id: Natural key of the ``models`` row.
        model_name: Display name of the model.
        is_ensemble: Whether the model is an ensemble product.
        resolution_km: Nominal grid resolution of the model in km.
        version_string: Model version string (e.g. ``v1.0``).
        cycle_time: UTC model run cycle time.
        grid_id: Natural key of the ``forecast_grids`` row.
        grid_name: Display name of the grid.
        grid_resolution_km: Grid resolution in km.
        product_type: Product type (e.g. ``surface``).
        zarr_store_path: Path/URL of the Zarr store holding the run data.
        variables: The forecast variables present in the dataset, recorded
            into ``forecast_variables`` and used to build
            ``forecast_products`` rows.
    """

    center_id: str
    center_name: str
    center_country: str
    model_id: str
    model_name: str
    is_ensemble: bool
    resolution_km: float
    version_string: str
    cycle_time: datetime
    grid_id: str
    grid_name: str
    grid_resolution_km: float
    product_type: str = "surface"
    zarr_store_path: str | None = None
    variables: tuple[VariableSpec, ...] = ()
    #: The complete set of lead times this run is expected to serve. Run-level
    #: readiness compares the committed lead set against this. Empty means the
    #: caller makes no completeness claim (the recorded dataset is the truth).
    expected_lead_time_hours: tuple[int, ...] = ()
    #: The complete set of GEFS member identities (1..30) this run is expected
    #: to serve. Empty for deterministic models. Readiness requires every
    #: expected member to be committed.
    expected_members: tuple[int, ...] = ()


def _get_or_create(
    db: Session,
    model: type[CatalogBase],
    where: ColumnElement[bool],
    create_kwargs: dict[str, object],
) -> CatalogBase:
    """Return the row matching ``where`` or create it with ``create_kwargs``.

    The row is flushed so its primary key and foreign-key targets are
    available to subsequent statements within the same transaction.

    Args:
        db: Database session.
        model: Catalog model class.
        where: SQLAlchemy filter expression selecting the row.
        create_kwargs: Column values used when creating a missing row.

    Returns:
        The existing or newly created row.
    """
    existing = db.execute(select(model).where(where)).scalars().first()
    if existing is not None:
        return existing
    row = model(**create_kwargs)
    db.add(row)
    db.flush()
    return row


def _lead_times(dataset: xr.Dataset) -> list[int]:
    """Return the ``lead_time_hours`` coordinate values of a dataset.

    A dataset without the coordinate is treated as a single 0-hour lead (the
    normalized parser always sets ``lead_time_hours``).

    Args:
        dataset: The normalized ingested dataset.

    Returns:
        The list of lead times in hours.
    """
    if "lead_time_hours" in dataset.coords:
        values = dataset.coords["lead_time_hours"].values
        if np.ndim(values) == 0:
            return [int(values)]
        return [int(value) for value in values]
    return [0]


def _reconcile_catalog_to_store(
    db: Session,
    run: ModelRunRecord,
    committed_state: CommittedState,
    spec: RunCatalogSpec | None = None,
) -> None:
    """Make the catalog products exactly match the actual committed store.

    The actual committed Zarr state (derived exclusively from COMPLETE
    marker/generation evidence) is the source of truth for catalog
    reconciliation. PATCH semantics in BOTH directions:

    * **stale removal** — a lead/member present in the catalog but absent from
      the store is deleted (existing behavior);
    * **missing restoration** — a lead/member physically committed in the store
      but absent from the catalog is inserted/upserted (new: the bug being
      fixed). A coordinator-path multi-region run only reached READY when every
      committed region also had a catalog row; because region workers commit to
      Zarr without ``record_run``, the catalog silently lagged the store and the
      run stayed ``partial``.

    Restoration requires authoritative metadata from the run's catalog spec:
    the variables, grid, product type, store path, and model id used by
    ``record_run``'s deterministic row-ID convention. When ``spec`` is ``None``
    (legacy callers that only delete), restoration is skipped and the function
    behaves exactly as before (delete-only). Row identity is deterministic, so
    re-running is idempotent (no duplicates, no ID churn).

    Deletion order respects foreign keys (``ensemble_member_products`` then
    ``ensemble_members`` are children of ``model_runs``; ``forecast_products``
    is a child of ``model_runs``). Restoration order creates parent member rows
    before their child pair rows, and reuses ``_get_or_create`` so unique
    constraints are respected (no new race: the caller's transaction + advisory
    gate serialize reconciliation in production).

    This runs in the SAME transaction as the status derivation, so a failed
    reconciliation rolls back atomically.

    Args:
        db: Database session.
        run: The run row.
        committed_state: The actual committed Zarr state (post-write).
        spec: The run's catalog metadata (variables, grid, product type, store
            path, model id) used to reconstruct missing rows. ``None`` restricts
            reconciliation to delete-only (legacy behavior).
    """
    committed_leads = committed_state.lead_set()

    if committed_state.is_ensemble:
        # 1. Delete stale member-product pairs first (child table; no FK to
        #    ensemble_members, but deleting rows before parent member rows keeps
        #    the member set derivable for step 2).
        pair_rows = db.execute(
            select(
                EnsembleMemberProductRecord.member_index,
                EnsembleMemberProductRecord.lead_time_hours,
            ).where(EnsembleMemberProductRecord.run_id == run.id)
        ).all()
        existing_pairs = {
            (int(member_num), int(lead_num)) for member_num, lead_num in pair_rows
        }
        committed_pairs = set(committed_state.pairs or ())
        for member_num, lead_num in existing_pairs:
            key = (int(member_num), int(lead_num))
            if key not in committed_pairs:
                db.execute(
                    EnsembleMemberProductRecord.__table__.delete().where(
                        EnsembleMemberProductRecord.run_id == run.id,
                        EnsembleMemberProductRecord.member_index == member_num,
                        EnsembleMemberProductRecord.lead_time_hours == lead_num,
                    )
                )

        # 2. Delete ensemble_members whose member index has no committed pair.
        committed_members = committed_state.member_set()
        db.execute(
            EnsembleMemberRecord.__table__.delete().where(
                EnsembleMemberRecord.run_id == run.id,
                EnsembleMemberRecord.member_index.not_in(committed_members)
                if committed_members
                else EnsembleMemberRecord.member_index.is_not(None),
            )
        )

        # 3. Restore missing ensemble member rows + (member, lead) pairs.
        #    A member is authoritative iff it has at least one committed pair.
        if spec is not None:
            for member_number in sorted(committed_members):
                _get_or_create(
                    db,
                    EnsembleMemberRecord,
                    (EnsembleMemberRecord.run_id == run.id)
                    & (EnsembleMemberRecord.member_index == member_number),
                    {
                        "id": f"member_{member_number}_{run.id}",
                        "run_id": run.id,
                        "member_index": member_number,
                        "member_name": f"{spec.model_id}_member_{member_number}",
                    },
                )
            for member_num, lead_num in sorted(committed_pairs):
                _get_or_create(
                    db,
                    EnsembleMemberProductRecord,
                    (EnsembleMemberProductRecord.run_id == run.id)
                    & (EnsembleMemberProductRecord.member_index == member_num)
                    & (EnsembleMemberProductRecord.lead_time_hours == lead_num),
                    {
                        "id": f"member_product_{member_num}_{lead_num}_{run.id}",
                        "run_id": run.id,
                        "member_index": member_num,
                        "lead_time_hours": lead_num,
                    },
                )

    # 4. Delete forecast_products the store does not actually carry. Two stale
    #    classes are removed: products whose lead is absent from the committed
    #    set, and (when the store's real variable set is known) products whose
    #    variable is absent from the store (e.g. a GEFS store that never holds
    #    ``precipitation_rate`` because GEFS pgrb2s has no instant prate). A
    #    store-absent variable must never remain advertised — that is exactly
    #    the map-422 / ensemble-404 false-availability class.
    db.execute(
        ProductRecord.__table__.delete().where(
            ProductRecord.run_id == run.id,
            ProductRecord.lead_time_hours.not_in(committed_leads)
            if committed_leads
            else ProductRecord.lead_time_hours.is_not(None),
        )
    )
    store_vars = committed_state.variables
    if store_vars is not None:
        db.execute(
            ProductRecord.__table__.delete().where(
                ProductRecord.run_id == run.id,
                ProductRecord.variable_id.not_in(store_vars)
                if store_vars
                else ProductRecord.variable_id.is_not(None),
            )
        )

    # 5. Restore missing forecast_products for committed leads. The store's
    #    committed lead set is authoritative; the per-lead product rows are
    #    reconstructed from the run spec's variables/grid/product metadata. Only
    #    variables the store actually carries are restored (a spec variable that
    #    is absent from the store — GEFS ``precipitation_rate`` — is never
    #    reconstructed, consistent with step 4 and with ``record_run``).
    if spec is not None and spec.variables:
        grid_code = spec.grid_id
        product_type = spec.product_type
        zarr_chunk_path = spec.zarr_store_path or run.zarr_store_path
        existing_products = set(
            int(p) for p in db.execute(
                select(ProductRecord.lead_time_hours).where(
                    ProductRecord.run_id == run.id
                )
            ).scalars()
        )
        missing_leads = committed_leads - existing_products
        restore_variables = [
            v.code
            for v in spec.variables
            if store_vars is None or v.code in store_vars
        ]
        for lead in sorted(missing_leads):
            for variable_code in restore_variables:
                _get_or_create(
                    db,
                    ProductRecord,
                    (ProductRecord.run_id == run.id)
                    & (ProductRecord.variable_id == variable_code)
                    & (ProductRecord.grid_id == grid_code)
                    & (ProductRecord.product_type == product_type)
                    & (ProductRecord.lead_time_hours == lead),
                    {
                        "id": (
                            f"product_{run.id}_{variable_code}_{grid_code}_"
                            f"{product_type}_{lead}"
                        ),
                        "run_id": run.id,
                        "variable_id": variable_code,
                        "grid_id": grid_code,
                        "product_type": product_type,
                        "lead_time_hours": lead,
                        "zarr_chunk_path": zarr_chunk_path,
                    },
                )


def record_run(
    db: Session,
    spec: RunCatalogSpec,
    dataset: xr.Dataset,
    *,
    member: int | None = None,
    committed_state: CommittedState | None = None,
) -> ModelRunRecord:
    """Create or update the PostgreSQL catalog rows for an ingested run.

    Upserts the center, model, model version, model run, grid, variables,
    forecast products (one row per variable x lead time), and ensemble members
    (one per real upstream ``member`` identity when the dataset has a member
    dimension). The run's status is derived from completeness: when the
    committed lead/member sets cover ``spec.expected_lead_time_hours`` /
    ``spec.expected_members`` the run is ``ready``; when some but not all are
    committed it is ``partial``; otherwise ``processing``. An individual file's
    catalog write therefore no longer implies the whole run is ready.

    When ``committed_state`` is provided, the catalog is additionally
    **reconciled to the actual committed Zarr state** (rows whose lead/member
    is absent from the store are deleted — PATCH preserves unrelated valid
    rows) and run status requires the **store↔catalog consistency gate**
    (``committed == actual committed Zarr``) in addition to invocation
    completeness.

    Args:
        db: Database session.
        spec: The catalog metadata of the run (including the expected lead and
            member sets used to decide readiness).
        dataset: The normalized dataset written to the Zarr store; its
            ``lead_time_hours`` coordinate and ``member`` dimension drive the
            product/member rows.
        member: The upstream GEFS member identity (``1..30``) being recorded.
            When provided, exactly one ``ensemble_members`` row for that real
            member number is created (never a positional index).
        committed_state: The actual committed lead/member state of the run's
            store (read post-write). When ``None`` (library callers without
            store access), reconciliation and the store-consistency gate are
            skipped and status is completeness-only.

    Returns:
        The recorded :class:`ModelRunRecord` (in its completeness-derived
        status).
    """
    cycle_time = spec.cycle_time
    if cycle_time.tzinfo is None:
        cycle_time = cycle_time.replace(tzinfo=timezone.utc)

    if is_cycle_tombstoned(db, cycle_time, model_id=spec.model_id):
        from ingestion.core.base import CycleTombstonedError

        raise CycleTombstonedError(
            f"Refusing to ingest forecast data for cycle {cycle_time.isoformat()}: "
            "cycle is claimed for deletion or already tombstoned."
        )

    center = _get_or_create(
        db,
        CenterRecord,
        CenterRecord.center_id == spec.center_id,
        {
            "id": f"center_{spec.center_id}",
            "center_id": spec.center_id,
            "name": spec.center_name,
            "country": spec.center_country,
        },
    )
    model = _get_or_create(
        db,
        ModelRecord,
        ModelRecord.model_id == spec.model_id,
        {
            "id": f"model_{spec.model_id}",
            "model_id": spec.model_id,
            "name": spec.model_name,
            "center_id": center.center_id,
            "is_ensemble": spec.is_ensemble,
            "resolution_km": spec.resolution_km,
        },
    )
    version = _get_or_create(
        db,
        ModelVersionRecord,
        (ModelVersionRecord.model_id == spec.model_id)
        & (ModelVersionRecord.version_string == spec.version_string),
        {
            "id": f"version_{spec.model_id}_{spec.version_string}",
            "model_id": model.model_id,
            "version_string": spec.version_string,
        },
    )
    # ``_get_or_create`` is typed over ``CatalogBase``; the caller knows the
    # concrete ``ModelRunRecord`` type (the ``model`` argument), so ``cast``
    # restores it so ``run.status``/``run.zarr_store_path`` assignments and the
    # ``ModelRunRecord`` return are type-checked. Making ``_get_or_create``
    # generic would require typing the ORM constructor ``model(**kwargs)``
    # return, which is ``Any`` by design.
    run = cast(
        ModelRunRecord,
        _get_or_create(
            db,
            ModelRunRecord,
            (ModelRunRecord.model_version_id == version.id)
            & (ModelRunRecord.cycle_time == cycle_time),
            {
                # The run id is version-scoped so two versions of the same
                # model at the same cycle cannot collide on the primary key
                # (the schema's ``(model_version_id, cycle_time)`` uniqueness
                # allows both rows, so the id must be able to distinguish
                # them). It still captures model + cycle for readability.
                "id": (
                    f"run_{version.id}_{cycle_time.strftime('%Y%m%d%H%M')}_"
                    f"{spec.model_id}"
                ),
                "model_version_id": version.id,
                "cycle_time": cycle_time,
                "status": "processing",
                "zarr_store_path": spec.zarr_store_path,
            },
        ),
    )
    # An existing row is updated to the current store path. Readiness is
    # recomputed after the product/member rows are written (see
    # ``_derive_run_status``). ``setattr`` is used because the ORM
    # ``Column``-typed class attributes are not instrumented for mypy's
    # assignment checking under strict mode.
    if spec.zarr_store_path is not None:
        setattr(run, "zarr_store_path", spec.zarr_store_path)

    grid = _get_or_create(
        db,
        GridRecord,
        GridRecord.grid_code == spec.grid_id,
        {
            "id": f"grid_{spec.grid_id}",
            "grid_code": spec.grid_id,
            "name": spec.grid_name,
            "resolution_km": spec.grid_resolution_km,
        },
    )

    # The catalog must only advertise variables the run's store actually
    # carries. ``spec.variables`` is the platform's documented vocabulary (the
    # same list is shared by GFS and GEFS), but a model product may genuinely
    # omit a variable (e.g. GEFS ``pgrb2s`` files have no instantaneous
    # ``prate`` field, so a GEFS store never holds ``precipitation_rate``).
    # Recording products for a variable absent from the store would make
    # availability advertise data that cannot be served (a map 422 / ensemble
    # 404). ``dataset.data_vars`` is the truth: a variable is recorded only when
    # the mapped dataset actually contains it. The ``forecast_variables``
    # catalog row is still created for every documented variable (the platform
    # vocabulary is model-agnostic); only the per-run ``forecast_products``
    # rows (which drive availability/serving) are filtered to the store's real
    # contents.
    dataset_vars = {str(name) for name in dataset.data_vars}
    variable_codes: list[str] = []
    for variable in spec.variables:
        variable_record = _get_or_create(
            db,
            VariableRecord,
            VariableRecord.variable_code == variable.code,
            {
                "id": f"var_{variable.code}",
                "variable_code": variable.code,
                "name": variable.name,
                "unit": variable.unit,
            },
        )
        if variable_record.variable_code in dataset_vars:
            variable_codes.append(variable_record.variable_code)

    for lead in _lead_times(dataset):
        for variable_code in variable_codes:
            _get_or_create(
                db,
                ProductRecord,
                (ProductRecord.run_id == run.id)
                & (ProductRecord.variable_id == variable_code)
                & (ProductRecord.grid_id == grid.grid_code)
                & (ProductRecord.product_type == spec.product_type)
                & (ProductRecord.lead_time_hours == lead),
                {
                    "id": (
                        f"product_{run.id}_{variable_code}_{grid.grid_code}_"
                        f"{spec.product_type}_{lead}"
                    ),
                    "run_id": run.id,
                    "variable_id": variable_code,
                    "grid_id": grid.grid_code,
                    "product_type": spec.product_type,
                    "lead_time_hours": lead,
                    "zarr_chunk_path": spec.zarr_store_path,
                },
            )

    # Ensemble member rows keyed by the real upstream member number. When a
    # per-member file is recorded (``member`` provided), exactly one row for
    # that number is created. When a combined multi-member dataset is recorded,
    # one row per member coordinate value is created (the coordinate may be a
    # dimension or a scalar coordinate — a single-member file decodes member as
    # a scalar coordinate).
    has_member = "member" in dataset.dims or "member" in dataset.coords
    if has_member:
        if member is not None:
            member_numbers = [int(member)]
        else:
            member_values = dataset.coords["member"].values
            if np.ndim(member_values) == 0:
                member_numbers = [int(member_values)]
            else:
                member_numbers = [int(value) for value in member_values]
        for member_number in member_numbers:
            _get_or_create(
                db,
                EnsembleMemberRecord,
                (EnsembleMemberRecord.run_id == run.id)
                & (EnsembleMemberRecord.member_index == member_number),
                {
                    # The id includes the run id so member rows of different
                    # runs of the same model cannot collide on the PK. The
                    # member_index is the REAL upstream member number, not a
                    # positional completion index.
                    "id": f"member_{member_number}_{run.id}",
                    "run_id": run.id,
                    "member_index": member_number,
                    "member_name": f"{spec.model_id}_member_{member_number}",
                },
            )
            # Record each committed (member, lead) pair so run-level readiness
            # can enforce the Cartesian product of expected members × expected
            # leads. A single-file ingest carries one member and one (or more)
            # leads; a combined multi-member file carries every member at each
            # lead. Duplicate/retry ingestion is idempotent via the unique
            # (run_id, member_index, lead_time_hours) constraint.
            for pair_lead in _lead_times(dataset):
                _get_or_create(
                    db,
                    EnsembleMemberProductRecord,
                    (EnsembleMemberProductRecord.run_id == run.id)
                    & (EnsembleMemberProductRecord.member_index == member_number)
                    & (EnsembleMemberProductRecord.lead_time_hours == pair_lead),
                    {
                        "id": (
                            f"member_product_{member_number}_{pair_lead}_{run.id}"
                        ),
                        "run_id": run.id,
                        "member_index": member_number,
                        "lead_time_hours": pair_lead,
                    },
                )

    # Reconcile the catalog to the actual committed Zarr state (source of
    # truth for PATCH preservation, stale-row elimination, AND missing-row
    # restoration). This runs BEFORE status derivation so status reflects the
    # reconciled catalog.
    if committed_state is not None:
        _reconcile_catalog_to_store(db, run, committed_state, spec)

    # Derive run-level readiness from committed vs expected sets. This is the
    # single decision point: a run becomes READY only when every expected lead
    # (and, for ensembles, every expected member) has a committed product /
    # member row, AND (when the committed state is known) the catalog matches
    # the actual committed Zarr state.
    setattr(run, "status", _derive_run_status(db, run, spec, committed_state))
    db.commit()
    return run


def set_run_partial(db: Session, run_id: str) -> None:
    """Downgrade a run to ``partial`` (the minimal same-cycle pre-update).

    The wave pre-update sets ``model_runs.status = partial`` before mutating
    any region so readers (which select only ``status == 'ready'``) exclude the
    run during the in-progress re-ingest. Product/member rows are **not**
    deleted here; the coalesced finalizer reconciles the catalog to the store
    after the wave.

    Args:
        db: Database session.
        run_id: The ``model_runs.id`` to downgrade.
    """
    run = db.get(ModelRunRecord, run_id)
    if run is None:
        return
    setattr(run, "status", "partial")
    db.flush()


def _derive_run_status(
    db: Session,
    run: ModelRunRecord,
    spec: RunCatalogSpec,
    committed_state: CommittedState | None = None,
) -> str:
    """Return the run's status from committed catalog rows and store state.

    Readiness requires BOTH:

    * **invocation completeness** — every expected lead (and, for ensembles,
      every expected ``(member, lead)`` Cartesian pair) has a committed
      catalog row; AND
    * **store↔catalog consistency** (when ``committed_state`` is provided) —
      the committed catalog rows equal the actual committed Zarr state.

    The invocation-completeness rule is the existing **subset** check
    (``expected ⊆ committed``), which is PATCH-safe: a healthy run whose
    catalog contains MORE leads than the current single-lead invocation's
    expected set stays complete. It is deliberately NOT changed to exact
    equality.

    The store↔catalog gate is independent: when the catalog's committed set
    differs from the actual committed Zarr set (e.g. a store shrunk to ``{6}``
    while the catalog still claims ``{0,6,12,18}``), the run is ``partial``
    even if invocation completeness holds. This prevents a stale catalog from
    ever being reported ``ready``.

    ``partial`` when some but not all expected items are committed, or when the
    catalog does not match the actual committed store.
    ``processing`` when nothing is yet committed or the run declares no
    expectations (the caller's single write is treated as the whole truth).

    Args:
        db: Database session.
        run: The run row.
        spec: The run's catalog spec (expected lead/member sets).
        committed_state: The actual committed Zarr state (post-write). When
            ``None`` (library caller without store access), the store↔catalog
            gate is skipped and status is completeness-only.

    Returns:
        ``ready``, ``partial``, or ``processing``.
    """
    committed_leads = set(
        db.execute(
            select(ProductRecord.lead_time_hours).where(
                ProductRecord.run_id == run.id
            )
        ).scalars()
    )
    expected_leads = set(spec.expected_lead_time_hours)
    if not expected_leads:
        # No declared expectations: the recorded dataset is the whole truth.
        # Still honor the store↔catalog gate when the store is known.
        return "ready" if _store_consistency_holds(db, run, committed_state) else "partial"

    if not expected_leads.issubset(committed_leads):
        # Nothing committed yet.
        if not committed_leads:
            return "processing"
        return "partial"

    if spec.is_ensemble and spec.expected_members:
        # Enforce the Cartesian product: every expected (member, lead) pair must
        # have a committed ensemble_member_products row. Query the committed
        # pairs directly from the pair table.
        rows = db.execute(
            select(
                EnsembleMemberProductRecord.member_index,
                EnsembleMemberProductRecord.lead_time_hours,
            ).where(EnsembleMemberProductRecord.run_id == run.id)
        ).all()
        committed_pairs = {(int(member_num), int(lead_num)) for member_num, lead_num in rows}
        expected_pairs = {
            (int(member), int(lead))
            for member in spec.expected_members
            for lead in spec.expected_lead_time_hours
        }
        if committed_pairs != expected_pairs:
            return "partial"

    if not _store_consistency_holds(db, run, committed_state):
        return "partial"

    return "ready"


def _store_consistency_holds(
    db: Session,
    run: ModelRunRecord,
    committed_state: CommittedState | None,
) -> bool:
    """Return whether the catalog's committed contents equal the store's.

    The store↔catalog consistency gate: a run is only READY-eligible when its
    committed catalog contents match the actual committed Zarr state. When the
    store is unknown (``committed_state is None``) the gate is trivially
    satisfied (completeness-only caller). When the store is known, a mismatch
    (catalog claims leads/members the store does not actually hold) returns
    False so the run cannot be ``ready``.

    Args:
        db: Database session.
        run: The run row.
        committed_state: The actual committed Zarr state, or ``None``.

    Returns:
        True when the store is unknown or the catalog matches the store;
        False when a known store disagrees with the catalog.
    """
    if committed_state is None:
        return True
    committed_leads = set(
        db.execute(
            select(ProductRecord.lead_time_hours).where(
                ProductRecord.run_id == run.id
            )
        ).scalars()
    )
    if committed_leads != committed_state.lead_set():
        return False
    if committed_state.is_ensemble:
        rows = db.execute(
            select(
                EnsembleMemberProductRecord.member_index,
                EnsembleMemberProductRecord.lead_time_hours,
            ).where(EnsembleMemberProductRecord.run_id == run.id)
        ).all()
        committed_pairs = {(int(member_num), int(lead_num)) for member_num, lead_num in rows}
        return committed_pairs == set(committed_state.pairs or ())
    return True


def record_ingested_dataset(
    spec: RunCatalogSpec,
    dataset: xr.Dataset,
    *,
    effective_store_path: str | None = None,
    member: int | None = None,
    committed_state: CommittedState | None = None,
) -> ModelRunRecord:
    """Record an ingested dataset to the configured PostgreSQL catalog.

    Convenience wrapper over :func:`record_run` that opens a session from the
    configured ``DATABASE_URL`` (``IngestionSettings``). A worker calls this
    immediately after a successful Zarr write. Run status is derived from
    completeness inside ``record_run``; an individual file no longer marks the
    whole run ready.

    The write is retried on a concurrent uniqueness collision: two workers
    ingesting the same (model, cycle) run can both miss the SELECT and both
    INSERT, and the second hits ``uq_model_run_cycle``. The catalog writes
    are idempotent, so after the winning worker's commit a bounded retry
    re-reads the existing row instead of surfacing an ``IntegrityError``.

    Args:
        spec: The catalog metadata of the run (including expected lead/member
            sets used to decide readiness).
        dataset: The normalized dataset written to the Zarr store.
        effective_store_path: The actual store path the dataset was written
            to. Defaults to ``spec.zarr_store_path``. The orchestration layer
            passes a staged sibling path when it re-ingests an existing run so
            the catalog records exactly the store that was written.
        member: The upstream GEFS member identity (``1..30``) being recorded.
        committed_state: The actual committed lead/member state of the run's
            Zarr store (read post-write). When provided, it is the source of
            truth for catalog reconciliation and the store↔catalog READY gate;
            when omitted (library callers without store access), reconciliation
            and the store-consistency gate are skipped and the existing
            completeness-only status is used.

    Returns:
        The recorded :class:`ModelRunRecord` (in its completeness-derived
        status).
    """
    if effective_store_path is None:
        effective_store_path = spec.zarr_store_path
    last_exc: IntegrityError | None = None
    for _ in range(_MAX_CONCURRENT_WRITE_RETRIES):
        with SessionLocal() as session:
            try:
                run = record_run(
                    session,
                    spec,
                    dataset,
                    member=member,
                    committed_state=committed_state,
                )
                if effective_store_path is not None:
                    # ``run.zarr_store_path`` is typed as a ``Column`` on the
                    # declarative class; SQLAlchemy instruments instance
                    # attributes at runtime, so assign the value with
                    # ``setattr`` to keep mypy happy.
                    setattr(run, "zarr_store_path", effective_store_path)
                    session.flush()
                    session.commit()
                # The context-manager session closes (and detaches the
                # instance) on exit; refresh the attributes so the returned
                # object is usable after the session is gone.
                session.refresh(run)
                return run
            except IntegrityError as exc:
                # The transaction is now invalid; close it and retry the
                # whole idempotent write from a fresh session.
                session.rollback()
                last_exc = exc
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Phase 6: Lifecycle & Retention Catalog Operations
# ---------------------------------------------------------------------------


def _ensure_utc_datetime(dt: datetime) -> datetime:
    """Normalize datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def list_model_ready_cycle_times(
    db: Session, model_id: str, *, version_string: str = "v1.0"
) -> list[datetime]:
    """Return all sorted UTC cycle times where model_id has status == 'ready'.

    Args:
        db: Database session.
        model_id: Target model identifier ('gfs', 'gefs').
        version_string: Target model version (default 'v1.0').

    Returns:
        Sorted list of unique UTC timezone-aware datetimes.
    """
    m_id = model_id.lower().strip()
    version_id = db.execute(
        select(ModelVersionRecord.id).where(
            (ModelVersionRecord.model_id == m_id)
            & (ModelVersionRecord.version_string == version_string)
        )
    ).scalar_one_or_none()
    if version_id is None:
        return []

    times = db.execute(
        select(ModelRunRecord.cycle_time).where(
            (ModelRunRecord.model_version_id == version_id)
            & (ModelRunRecord.status == "ready")
        )
    ).scalars().all()
    return sorted(_ensure_utc_datetime(dt) for dt in set(times))


def list_paired_ready_cycle_times(
    db: Session, *, version_string: str = "v1.0"
) -> list[datetime]:
    """Legacy helper: return intersection of ready cycle times across gfs and gefs."""
    gfs_ready = set(list_model_ready_cycle_times(db, "gfs", version_string=version_string))
    gefs_ready = set(list_model_ready_cycle_times(db, "gefs", version_string=version_string))
    return sorted(gfs_ready & gefs_ready)


def list_cycle_lifecycle_snapshots(
    db: Session,
    *,
    model_id: str | None = None,
    version_string: str = "v1.0",
) -> list[Any]:
    """Reconstruct ModelLifecycleSnapshot for cycles in the catalog.

    Discovers cycles from both forecast_cycle_lifecycle and model_runs so that
    newly created model_runs without a lifecycle row yet are naturally captured.

    Args:
        db: Database session.
        model_id: Optional model identifier ('gfs', 'gefs'). When omitted, queries all models.
        version_string: Model version string to discover run statuses for.

    Returns:
        List of ModelLifecycleSnapshot objects sorted by cycle_time ascending.
    """
    from domain.lifecycle import ModelLifecycleSnapshot

    stmt_lc = select(ForecastCycleLifecycleRecord)
    if model_id is not None:
        stmt_lc = stmt_lc.where(
            ForecastCycleLifecycleRecord.model_id == model_id.lower().strip()
        )
    lifecycle_rows = {
        (str(row.model_id), _ensure_utc_datetime(cast(datetime, row.cycle_time))): row
        for row in db.execute(stmt_lc).scalars().all()
    }

    stmt_runs = (
        select(
            ModelVersionRecord.model_id,
            ModelRunRecord.cycle_time,
            ModelRunRecord.status,
        )
        .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
        .where(ModelVersionRecord.version_string == version_string)
    )
    if model_id is not None:
        stmt_runs = stmt_runs.where(
            ModelVersionRecord.model_id == model_id.lower().strip()
        )
    else:
        stmt_runs = stmt_runs.where(
            ModelVersionRecord.model_id.in_(["gfs", "gefs"])
        )
    runs = db.execute(stmt_runs).all()

    model_statuses: dict[tuple[str, datetime], str] = {}
    all_keys: set[tuple[str, datetime]] = set(lifecycle_rows.keys())
    for m_id, cycle_time, status in runs:
        m_str = str(m_id)
        c_utc = _ensure_utc_datetime(cast(datetime, cycle_time))
        model_statuses[(m_str, c_utc)] = str(status)
        all_keys.add((m_str, c_utc))

    snapshots: list[ModelLifecycleSnapshot] = []
    for m_str, c_utc in sorted(all_keys, key=lambda k: (k[1], k[0])):
        lc = lifecycle_rows.get((m_str, c_utc))
        snapshots.append(
            ModelLifecycleSnapshot(
                model_id=m_str,
                cycle_time=c_utc,
                status=model_statuses.get((m_str, c_utc)),
                retired_at=cast(datetime | None, lc.retired_at) if lc else None,
                retired_by_cycle_time=cast(datetime | None, lc.retired_by_cycle_time) if lc else None,
                deletion_started_at=cast(datetime | None, lc.deletion_started_at) if lc else None,
                deleted_at=cast(datetime | None, lc.deleted_at) if lc else None,
            )
        )
    return snapshots


def ensure_lifecycle_row(
    db: Session,
    model_id: str,
    cycle_time: datetime,
) -> ForecastCycleLifecycleRecord:
    """Return existing lifecycle record for (model_id, cycle_time) or create initial row."""
    m_id = model_id.lower().strip()
    c_utc = _ensure_utc_datetime(cycle_time)
    row = db.get(ForecastCycleLifecycleRecord, (m_id, c_utc))
    if row is not None:
        return row
    now = _utcnow()
    row = ForecastCycleLifecycleRecord(
        model_id=m_id,
        cycle_time=c_utc,
        retired_at=None,
        retired_by_cycle_time=None,
        deletion_started_at=None,
        deleted_at=None,
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.flush()
    return row


def mark_cycle_retired(
    db: Session,
    model_id: str,
    cycle_time: datetime,
    retired_at: datetime,
    retired_by_cycle_time: datetime,
) -> bool:
    """Mark a forecast cycle as retired by anchor T.

    Idempotent: if already retired with identical retired_by_cycle_time, returns False.
    """
    import logging

    m_id = model_id.lower().strip()
    c_utc = _ensure_utc_datetime(cycle_time)
    r_at_utc = _ensure_utc_datetime(retired_at)
    r_by_utc = _ensure_utc_datetime(retired_by_cycle_time)

    row = ensure_lifecycle_row(db, m_id, c_utc)
    if row.retired_at is not None:
        if row.retired_by_cycle_time is not None:
            existing_r1 = _ensure_utc_datetime(cast(datetime, row.retired_by_cycle_time))
            if existing_r1 != r_by_utc:
                logging.getLogger(__name__).error(
                    "Model %s cycle %s already retired by %s; refusing to overwrite with %s",
                    m_id,
                    c_utc.isoformat(),
                    existing_r1.isoformat(),
                    r_by_utc.isoformat(),
                )
                raise ValueError(
                    f"Model {m_id} cycle {c_utc.isoformat()} already retired by {existing_r1.isoformat()}; "
                    f"cannot overwrite with {r_by_utc.isoformat()}"
                )
        return False

    setattr(row, "retired_at", r_at_utc)
    setattr(row, "retired_by_cycle_time", r_by_utc)
    setattr(row, "updated_at", _utcnow())
    db.flush()
    return True


def reconcile_cycle_lifecycle(
    db: Session,
    *,
    model_id: str | None = None,
    models: tuple[str, ...] = ("gfs", "gefs"),
    now: datetime | None = None,
    version_string: str = "v1.0",
) -> Any:
    """Evaluate lifecycle transitions and persist new retirements to PostgreSQL per model.

    Args:
        db: Database session.
        model_id: Optional model identifier. When provided, reconciles only that model.
        models: Tuple of models to reconcile when model_id is None.
        now: Optional current timestamp override for retired_at (defaults to UTC now).
        version_string: Model version string (defaults to 'v1.0').

    Returns:
        The evaluated ModelLifecyclePlan (if single model) or dict of plans per model.
    """
    import logging
    from domain.lifecycle import plan_model_lifecycle

    log = logging.getLogger(__name__)
    now_utc = _ensure_utc_datetime(now) if now is not None else _utcnow()
    target_models = (model_id.lower().strip(),) if model_id is not None else models

    plans: dict[str, Any] = {}
    for m_id in target_models:
        ready_cycles = list_model_ready_cycle_times(db, m_id, version_string=version_string)
        snapshots = list_cycle_lifecycle_snapshots(db, model_id=m_id, version_string=version_string)
        plan = plan_model_lifecycle(m_id, snapshots, ready_cycles)
        plans[m_id] = plan

        for decision in plan.would_retire:
            if decision.retired_by_cycle_time is not None:
                mark_cycle_retired(
                    db,
                    m_id,
                    decision.cycle_time,
                    retired_at=now_utc,
                    retired_by_cycle_time=decision.retired_by_cycle_time,
                )
                log.info(
                    "cycle_retired: model=%s cycle_time=%s retired_by=%s reason=%s",
                    m_id,
                    decision.cycle_time.isoformat(),
                    decision.retired_by_cycle_time.isoformat(),
                    decision.reason,
                    extra={
                        "event": "cycle_retired",
                        "model": m_id,
                        "cycle_time": decision.cycle_time.isoformat(),
                        "retired_by": decision.retired_by_cycle_time.isoformat(),
                    },
                )
    db.commit()
    if model_id is not None:
        return plans[model_id.lower().strip()]
    return plans

