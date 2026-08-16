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
from typing import cast

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
    """

    leads: frozenset[int]
    members: frozenset[int] | None
    pairs: frozenset[tuple[int, int]] | None
    is_ensemble: bool

    @classmethod
    def deterministic(cls, leads: set[int]) -> CommittedState:
        """Build the committed state of a deterministic (non-ensemble) store."""
        return cls(leads=frozenset(leads), members=None, pairs=None, is_ensemble=False)

    @classmethod
    def ensemble(
        cls, pairs: set[tuple[int, int]], members: set[int]
    ) -> CommittedState:
        """Build the committed state of an ensemble store.

        Args:
            pairs: The committed ``(member, lead)`` pairs.
            members: The member indices that have at least one committed pair.
        """
        return cls(
            leads=frozenset(lead for _, lead in pairs),
            members=frozenset(members),
            pairs=frozenset(pairs),
            is_ensemble=True,
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
) -> None:
    """Delete catalog rows whose lead/member is absent from the committed store.

    The actual committed Zarr state is the source of truth for catalog
    reconciliation. PATCH semantics: a lead/member is preserved when it is
    present in the store (regardless of whether the current invocation touched
    it); a lead/member is deleted only when it is **absent from the store**
    (genuinely stale). Absence from the current incoming patch is NEVER a
    deletion signal.

    Deletion order respects foreign keys (``ensemble_member_products`` and
    ``ensemble_members`` are children of ``model_runs``; ``forecast_products``
    is a child of ``model_runs``):

    1. ``ensemble_member_products`` pairs not in the committed pair set;
    2. ``ensemble_members`` whose member index no longer has any committed
       pair (only for ensemble runs);
    3. ``forecast_products`` rows whose lead is not in the committed lead set.

    This runs in the SAME transaction as the status derivation, so a failed
    reconciliation rolls back atomically.

    Args:
        db: Database session.
        run: The run row.
        committed_state: The actual committed Zarr state (post-write).
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
        committed_pairs = set(committed_state.pairs or ())
        for member_num, lead_num in pair_rows:
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

    # 3. Delete forecast_products whose lead is absent from the committed set.
    db.execute(
        ProductRecord.__table__.delete().where(
            ProductRecord.run_id == run.id,
            ProductRecord.lead_time_hours.not_in(committed_leads)
            if committed_leads
            else ProductRecord.lead_time_hours.is_not(None),
        )
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
    # truth for PATCH preservation and stale-row elimination). This runs
    # BEFORE status derivation so status reflects the reconciled catalog.
    if committed_state is not None:
        _reconcile_catalog_to_store(db, run, committed_state)

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
