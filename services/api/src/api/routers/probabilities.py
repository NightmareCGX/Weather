"""Exceedance probability endpoint (API.md section 3.1).

Returns the empirical probability that a forecast variable exceeds a
threshold, based on ensemble member spread, along with a deterministic Wilson
95% confidence interval. The router is thin (ENGINEERING_CONTRACT section 2):
it validates parameters, calls the ensemble-data and cache services, and
serializes the documented ``probability_forecast`` envelope.

Specification notes:

* ``operator=between`` requires an upper bound. API.md section 3.1 defines
  only a single ``threshold`` parameter, so an additive ``threshold_max``
  query parameter carries the upper bound (a non-breaking addition per API.md
  section 1.3). ``threshold_max`` is required when ``operator=between`` and is
  rejected when ``operator`` is ``gt`` or ``lt``.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import ProbabilityForecastEnvelope
from api.services.cache import (
    PointCache,
    build_probability_cache_key,
)
from api.services.ensemble_data import build_probability_forecast
from api.services.point_forecast import (
    resolve_latest_run_cycle_time,
    resolve_latest_run_serving_generation,
)

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for probability forecasts: resolves to the newest ready run
#: when initial_time is omitted, so revalidation (no-cache) ensures new cycles
#: or same-cycle updates are seen immediately while Redis caches by generation.
CACHE_CONTROL_PROBABILITY = "no-cache"

#: Cache TTL in seconds for probability forecasts.
CACHE_TTL_SECONDS_PROBABILITY = 3600

#: In-memory probability cache instance for the API process.
_cache = PointCache(ttl_seconds=CACHE_TTL_SECONDS_PROBABILITY)


@router.get(
    "/probabilities",
    response_model=ProbabilityForecastEnvelope,
    summary="Get exceedance probability",
)
def get_probability(
    response: Response,
    lat: Annotated[float, Query(description="Latitude in decimal degrees.")],
    lon: Annotated[float, Query(description="Longitude in decimal degrees.")],
    variable: Annotated[str, Query(description="A forecast variable code.")],
    threshold: Annotated[float, Query(description="The probability threshold.")],
    operator: Annotated[
        Literal["gt", "lt", "between"],
        Query(description="Threshold comparison operator."),
    ],
    lead_time_hours: Annotated[
        int, Query(ge=0, description="Forecast offset hours from cycle time.")
    ],
    model: Annotated[
        str, Query(description="A single ensemble model identifier.")
    ] = "gefs",
    threshold_max: Annotated[
        float | None,
        Query(
            description=(
                "Upper bound of the 'between' operator (required only when "
                "operator=between)."
            )
        ),
    ] = None,
    # Optional cycle pinning (GAP-2): additive and non-breaking.
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
) -> ProbabilityForecastEnvelope:
    """Return the exceedance probability for a forecast variable.

    The ensemble model (default ``gefs``) must exist and be an ensemble model;
    member values are interpolated at the requested point and lead time. An
    optional ``initial_time`` pins the forecast run.
    """
    _validate_threshold_bounds(operator, threshold, threshold_max)

    cache_key = build_probability_cache_key(
        model=model,
        latitude=lat,
        longitude=lon,
        variable=variable,
        threshold=threshold,
        operator=operator,
        lead_time_hours=lead_time_hours,
        threshold_max=threshold_max,
        cycle_time=resolve_latest_run_cycle_time(db, model, initial_time),
        serving_generation=resolve_latest_run_serving_generation(
            db, model, initial_time
        ),
    )
    query_params = (
        f"lat={lat}&lon={lon}&variable={variable}&threshold={threshold}"
        f"&operator={operator}&lead_time_hours={lead_time_hours}"
        f"&model={model}&threshold_max={threshold_max}&initial_time={initial_time}"
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
            threshold,
            operator,
            lead_time_hours,
            model,
            threshold_max,
            initial_time,
        ),
        model_type=ProbabilityForecastEnvelope,
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_PROBABILITY
    return envelope


def _validate_threshold_bounds(
    operator: Literal["gt", "lt", "between"],
    threshold: float,
    threshold_max: float | None,
) -> None:
    """Enforce the ``between``/``threshold_max`` pairing rules.

    ``between`` requires ``threshold_max``; ``gt``/``lt`` reject it. An
    out-of-range ordering (``threshold_max < threshold``) is surfaced later by
    the domain math as a 422.
    """
    if operator == "between" and threshold_max is None:
        raise HTTPException(
            status_code=422,
            detail="threshold_max is required when operator is 'between'.",
        )
    if operator in ("gt", "lt") and threshold_max is not None:
        raise HTTPException(
            status_code=422,
            detail="threshold_max is only valid when operator is 'between'.",
        )


def _compute(
    db: Session,
    lat: float,
    lon: float,
    variable: str,
    threshold: float,
    operator: Literal["gt", "lt", "between"],
    lead_time_hours: int,
    model: str,
    threshold_max: float | None,
    initial_time: str | None,
) -> ProbabilityForecastEnvelope:
    data = build_probability_forecast(
        db,
        latitude=lat,
        longitude=lon,
        variable=variable,
        threshold=threshold,
        operator=operator,
        lead_time_hours=lead_time_hours,
        model=model,
        threshold_max=threshold_max,
        initial_time=initial_time,
    )
    return ProbabilityForecastEnvelope(data=data)
