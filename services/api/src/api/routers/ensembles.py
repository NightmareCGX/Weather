"""Ensemble statistics endpoint (API.md section 5.1).

Returns statistical dispersion (mean, median, spread, P10/P25/P50/P75/P90)
across ensemble perturbation members for a given location and lead time. The
router is thin (ENGINEERING_CONTRACT section 2): it validates parameters,
calls the ensemble-data and cache services, and serializes the documented
``ensemble_statistics`` envelope.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import EnsembleStatisticsEnvelope
from api.services.cache import PointCache, build_ensemble_cache_key
from api.services.ensemble_data import build_ensemble_statistics
from api.services.point_forecast import resolve_latest_run_cycle_time

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for ensemble statistics (API.md 5.1: 30 minutes).
CACHE_CONTROL_ENSEMBLE = "public, max-age=1800"

#: Cache TTL in seconds for ensemble statistics.
CACHE_TTL_SECONDS_ENSEMBLE = 1800

#: In-memory ensemble cache instance for the API process.
_cache = PointCache(ttl_seconds=CACHE_TTL_SECONDS_ENSEMBLE)


@router.get(
    "/ensembles",
    response_model=EnsembleStatisticsEnvelope,
    summary="Get ensemble statistics and spread",
)
def get_ensemble_statistics(
    response: Response,
    lat: Annotated[float, Query(description="Latitude in decimal degrees.")],
    lon: Annotated[float, Query(description="Longitude in decimal degrees.")],
    variable: Annotated[str, Query(description="A forecast variable code.")],
    model: Annotated[
        str, Query(description="A single ensemble model identifier.")
    ] = "gefs",
    # ``lead_time_hours`` defaults to 0 (the model run's cycle time), matching
    # API.md section 5.1 which documents the parameter as optional with
    # ``Default: 0``.
    lead_time_hours: Annotated[
        int, Query(ge=0, description="Forecast offset hours from cycle time.")
    ] = 0,
    # ``include_members`` is an opt-in additive extension (API.md section 5.1):
    # when true, the response additionally carries the raw ensemble-member
    # forecast values for the Ensemble Distribution View. Statistics-only
    # requests (the default) stay lightweight and omit the member array.
    include_members: Annotated[
        bool, Query(description="Return raw ensemble-member forecast values.")
    ] = False,
    # Optional cycle pinning (GAP-2): lets a client request a specific forecast
    # run's ensemble rather than always the newest. Additive and non-breaking.
    initial_time: Annotated[
        str | None,
        Query(
            description=(
                "Optional ISO 8601 UTC cycle time pinning the model run "
                "(e.g. 2026-08-13T00:00:00Z). Defaults to the newest ready run."
            )
        ),
    ] = None,
    db: Session = DB,
) -> EnsembleStatisticsEnvelope:
    """Return ensemble dispersion statistics for a forecast variable.

    The ensemble model (default ``gefs``) must exist and be an ensemble model;
    member values are interpolated at the requested point and lead time. When
    ``include_members=true`` the genuine raw member values are attached. An
    optional ``initial_time`` pins the forecast run.
    """
    cache_key = build_ensemble_cache_key(
        model=model,
        latitude=lat,
        longitude=lon,
        variable=variable,
        lead_time_hours=lead_time_hours,
        include_members=include_members,
        cycle_time=resolve_latest_run_cycle_time(db, model, initial_time),
    )
    query_params = (
        f"lat={lat}&lon={lon}&variable={variable}&model={model}"
        f"&lead_time_hours={lead_time_hours}&include_members={include_members}"
        f"&initial_time={initial_time}"
    )

    envelope = _cache.compute_or_retrieve(
        db,
        cache_key,
        query_params,
        lambda: _compute(
            db,
            lat,
            lon,
            variable,
            model,
            lead_time_hours,
            include_members,
            initial_time,
        ),
        model_type=EnsembleStatisticsEnvelope,
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_ENSEMBLE
    return envelope


def _compute(
    db: Session,
    lat: float,
    lon: float,
    variable: str,
    model: str,
    lead_time_hours: int,
    include_members: bool,
    initial_time: str | None,
) -> EnsembleStatisticsEnvelope:
    data = build_ensemble_statistics(
        db,
        latitude=lat,
        longitude=lon,
        variable=variable,
        model=model,
        lead_time_hours=lead_time_hours,
        include_members=include_members,
        initial_time=initial_time,
    )
    return EnsembleStatisticsEnvelope(data=data)
