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


def record_run(
    db: Session,
    spec: RunCatalogSpec,
    dataset: xr.Dataset,
) -> ModelRunRecord:
    """Create or update the PostgreSQL catalog rows for an ingested run.

    Upserts the center, model, model version, model run, grid, variables,
    forecast products (one row per variable x lead time), and ensemble members
    (one per ``member`` index when the dataset has a member dimension). The
    run is recorded as ``status='ready'`` so the API serving tier's
    ``_resolve_run`` discovers it. All writes happen in a single transaction
    that is committed on success.

    Args:
        db: Database session.
        spec: The catalog metadata of the run.
        dataset: The normalized dataset written to the Zarr store; its
            ``lead_time_hours`` coordinate and ``member`` dimension drive the
            product/member rows.

    Returns:
        The recorded :class:`ModelRunRecord` (ready).
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
                "id": f"run_{cycle_time.strftime('%Y%m%d%H%M')}_{spec.model_id}",
                "model_version_id": version.id,
                "cycle_time": cycle_time,
                "status": "ready",
                "zarr_store_path": spec.zarr_store_path,
            },
        ),
    )
    # An existing row is refreshed to ready and the current store path so a
    # re-ingested run (upsert) is immediately discoverable and serveable.
    # ``setattr`` is used (as in ``record_ingested_dataset``) because the ORM
    # ``Column``-typed class attributes are not instrumented for mypy's
    # assignment checking under strict mode.
    setattr(run, "status", "ready")
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

    if "member" in dataset.dims:
        member_count = int(dataset.sizes["member"])
        for member_index in range(member_count):
            _get_or_create(
                db,
                EnsembleMemberRecord,
                (EnsembleMemberRecord.run_id == run.id)
                & (EnsembleMemberRecord.member_index == member_index),
                {
                    # The id includes the run id so member rows of different
                    # runs of the same model cannot collide on the PK.
                    "id": f"member_{member_index}_{run.id}",
                    "run_id": run.id,
                    "member_index": member_index,
                    "member_name": f"{spec.model_id}_member_{member_index}",
                },
            )

    db.commit()
    return run


def record_ingested_dataset(
    spec: RunCatalogSpec,
    dataset: xr.Dataset,
    *,
    effective_store_path: str | None = None,
) -> ModelRunRecord:
    """Record an ingested dataset to the configured PostgreSQL catalog.

    Convenience wrapper over :func:`record_run` that opens a session from the
    configured ``DATABASE_URL`` (``IngestionSettings``). A worker calls this
    immediately after a successful Zarr write.

    The write is retried on a concurrent uniqueness collision: two workers
    ingesting the same (model, cycle) run can both miss the SELECT and both
    INSERT, and the second hits ``uq_model_run_cycle``. The catalog writes
    are idempotent, so after the winning worker's commit a bounded retry
    re-reads the existing row instead of surfacing an ``IntegrityError``.

    Args:
        spec: The catalog metadata of the run.
        dataset: The normalized dataset written to the Zarr store.
        effective_store_path: The actual store path the dataset was written
            to. Defaults to ``spec.zarr_store_path``. The orchestration layer
            passes a staged sibling path when it re-ingests an existing run so
            the catalog records exactly the store that was written.

    Returns:
        The recorded :class:`ModelRunRecord` (ready).
    """
    if effective_store_path is None:
        effective_store_path = spec.zarr_store_path
    last_exc: IntegrityError | None = None
    for _ in range(_MAX_CONCURRENT_WRITE_RETRIES):
        with SessionLocal() as session:
            try:
                run = record_run(session, spec, dataset)
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
