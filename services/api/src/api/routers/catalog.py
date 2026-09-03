"""Catalog endpoints: forecast centers, models, runs, variables, and grids.

These read-only endpoints query the Milestone 3 PostgreSQL schema and return
the response envelope and resource shapes defined in ``docs/API.md`` Domain 1.
Routers remain thin: they validate query parameters, run catalog queries, and
serialize results. No weather calculations live here (ENGINEERING_CONTRACT
section 2).
"""

from typing import TypeVar

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from api.core.database import get_db
from api.deps import Page, Pagination, PaginationParams, paginate
from api.models.entities import (
    ForecastCenter,
    ForecastGrid,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.lifecycle import filter_visible_runs
from api.schemas import (
    CenterOut,
    GridOut,
    ListEnvelope,
    ModelOut,
    RunOut,
    VariableOut,
)

router = APIRouter()

#: Cache policy for stable catalog resources (API.md: 24 hours).
CACHE_CONTROL_DAILY = "public, max-age=86400"
#: Cache policy for model runs: run status and available runs mutate upon
#: ingestion, so revalidation (no-cache) guarantees status changes are seen.
CACHE_CONTROL_RUNS = "no-cache"

T = TypeVar("T", bound=BaseModel)


def _envelope(
    response: Response,
    data: list[T],
    page: Page,
    cache_control: str,
) -> ListEnvelope[T]:
    """Apply the cache policy and wrap page items in the list envelope."""
    response.headers["Cache-Control"] = cache_control
    return ListEnvelope[T](data=data, has_more=page.has_more, next_cursor=page.next_cursor)


@router.get(
    "/centers",
    response_model=ListEnvelope[CenterOut],
    summary="List forecast centers",
)
def list_centers(
    response: Response,
    pagination: PaginationParams = Pagination,
    db: Session = Depends(get_db),
) -> ListEnvelope[CenterOut]:
    """Retrieve all supported meteorological forecast centers."""
    stmt = select(ForecastCenter)
    page = paginate(db, stmt, ForecastCenter.center_id, pagination)
    data = [
        CenterOut(id=center.center_id, name=center.name, country=center.country)
        for center in page.items
    ]
    return _envelope(response, data, page, CACHE_CONTROL_DAILY)


@router.get(
    "/models",
    response_model=ListEnvelope[ModelOut],
    summary="List models",
)
def list_models(
    response: Response,
    pagination: PaginationParams = Pagination,
    center_id: str | None = None,
    is_ensemble: bool | None = None,
    db: Session = Depends(get_db),
) -> ListEnvelope[ModelOut]:
    """Retrieve supported operational and AI weather models."""
    stmt = select(Model)
    if center_id is not None:
        stmt = stmt.where(Model.center_id == center_id)
    if is_ensemble is not None:
        stmt = stmt.where(Model.is_ensemble == is_ensemble)
    page = paginate(db, stmt, Model.model_id, pagination)
    data = [
        ModelOut(
            id=model.model_id,
            name=model.name,
            center_id=model.center_id,
            is_ensemble=model.is_ensemble,
            resolution_km=model.resolution_km,
        )
        for model in page.items
    ]
    return _envelope(response, data, page, CACHE_CONTROL_DAILY)


@router.get(
    "/runs",
    response_model=ListEnvelope[RunOut],
    summary="List model runs",
)
def list_runs(
    response: Response,
    pagination: PaginationParams = Pagination,
    model_id: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
) -> ListEnvelope[RunOut]:
    """Retrieve ingested model execution cycles and their statuses.

    A run's ``model_id`` is resolved through the schema's existing
    ``model_versions`` relationship (a run is keyed by model version, not by
    model directly).
    """
    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .options(selectinload(ModelRun.model_version).selectinload(ModelVersion.model))
    )
    stmt = filter_visible_runs(stmt)
    if model_id is not None:
        stmt = stmt.where(Model.model_id == model_id)
    if status is not None:
        stmt = stmt.where(ModelRun.status == status)
    page = paginate(db, stmt, ModelRun.id, pagination)
    data = [
        RunOut(
            id=run.id,
            model_id=run.model_version.model.model_id,
            cycle_time=run.cycle_time,
            status=run.status,
        )
        for run in page.items
    ]
    return _envelope(response, data, page, CACHE_CONTROL_RUNS)


@router.get(
    "/variables",
    response_model=ListEnvelope[VariableOut],
    summary="List forecast variables",
)
def list_variables(
    response: Response,
    pagination: PaginationParams = Pagination,
    db: Session = Depends(get_db),
) -> ListEnvelope[VariableOut]:
    """Retrieve standardized physical meteorological variables."""
    stmt = select(ForecastVariable).where(
        ForecastVariable.variable_code.not_in(
            ["wind_u_10m", "wind_v_10m", "crain", "csnow", "cfrzr", "cicep"]
        )
    )
    page = paginate(db, stmt, ForecastVariable.variable_code, pagination)
    data = [
        VariableOut(id=variable.variable_code, name=variable.name, unit=variable.unit)
        for variable in page.items
    ]
    if not any(v.id == "wind_10m" for v in data):
        u_exists = db.execute(
            select(ForecastVariable.variable_code).where(
                ForecastVariable.variable_code == "wind_u_10m"
            )
        ).scalar_one_or_none()
        if u_exists is not None:
            data.append(VariableOut(id="wind_10m", name="10-Meter Wind", unit="km/h"))
            data.sort(key=lambda v: v.id)
    return _envelope(response, data, page, CACHE_CONTROL_DAILY)


@router.get(
    "/grids",
    response_model=ListEnvelope[GridOut],
    summary="List forecast grids",
)
def list_grids(
    response: Response,
    pagination: PaginationParams = Pagination,
    db: Session = Depends(get_db),
) -> ListEnvelope[GridOut]:
    """Retrieve supported spatial grid definitions."""
    stmt = select(ForecastGrid)
    page = paginate(db, stmt, ForecastGrid.grid_code, pagination)
    data = [
        GridOut(id=grid.grid_code, name=grid.name, resolution_km=grid.resolution_km)
        for grid in page.items
    ]
    return _envelope(response, data, page, CACHE_CONTROL_DAILY)
