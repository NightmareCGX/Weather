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
    # ``lead_time_hours`` defaults to 0 (the model run's cycle time) even
    # though API.md section 5.1 documents it as required with no default. The
    # default is intentionally kept for backward compatibility with the
    # initial /v1/ensembles release; the documented contract treats the
    # parameter as required.
    lead_time_hours: Annotated[
        int, Query(ge=0, description="Forecast offset hours from cycle time.")
    ] = 0,
    db: Session = DB,
) -> EnsembleStatisticsEnvelope:
    """Return ensemble dispersion statistics for a forecast variable.

    The ensemble model (default ``gefs``) must exist and be an ensemble model;
    member values are interpolated at the requested point and lead time.
    """
    cache_key = build_ensemble_cache_key(
        model=model,
        latitude=lat,
        longitude=lon,
        variable=variable,
        lead_time_hours=lead_time_hours,
    )
    query_params = (
        f"lat={lat}&lon={lon}&variable={variable}&model={model}"
        f"&lead_time_hours={lead_time_hours}"
    )

    envelope = _cache.compute_or_retrieve(
        db,
        cache_key,
        query_params,
        lambda: _compute(db, lat, lon, variable, model, lead_time_hours),
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
) -> EnsembleStatisticsEnvelope:
    data = build_ensemble_statistics(
        db,
        latitude=lat,
        longitude=lon,
        variable=variable,
        model=model,
        lead_time_hours=lead_time_hours,
    )
    return EnsembleStatisticsEnvelope(data=data)
