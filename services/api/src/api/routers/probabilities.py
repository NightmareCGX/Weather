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

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import ProbabilityForecastEnvelope
from api.services.cache import (
    PointCache,
    build_probability_cache_key,
)
from api.services.ensemble_data import build_probability_forecast
from api.services.lifecycle import require_cycle_visible
from api.services.point_forecast import (
    resolve_latest_run_cycle_time,
    resolve_latest_run_store_path_and_retirement,
    resolve_serving_generation_for_store,
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
        Literal["gt", "gte", "lt", "lte", "between"],
        Query(description="Threshold comparison operator."),
    ],
    lead_time_hours: Annotated[
        int | None,
        Query(ge=0, description="Forecast offset hours from cycle time."),
    ] = None,
    valid_time: Annotated[
        str | None,
        Query(
            description=(
                "Optional ISO 8601 UTC valid time (Lifecycle V2). Resolves to the newest "
                "serveable cycle capable of providing this valid time."
            )
        ),
    ] = None,
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
    direction_sector: Annotated[
        str | None,
        Query(
            description=(
                "Optional 8-point cardinal sector for directional wind probabilities "
                "(e.g. 'SW', 'N')."
            )
        ),
    ] = None,
    phase: Annotated[
        str | None,
        Query(
            description=(
                "Optional physical precipitation phase (e.g. 'snow', 'rain', 'freezing_rain', 'ice_pellets') "
                "for joint exceedance probability."
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

    Under Lifecycle V2:
    - If ``valid_time`` is supplied, the newest committed source cycle and lead
      are dynamically resolved via the shared ValidTimeResolver.
    - If ``lead_time_hours`` is supplied without ``valid_time``, legacy cycle/lead
      serving is preserved for backward compatibility.
    """
    from api.services.resolver import resolve_valid_time_source

    _validate_threshold_bounds(operator, threshold, threshold_max)

    if valid_time is not None and initial_time is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide either valid_time or initial_time, not both.",
        )
    if valid_time is None and lead_time_hours is None:
        raise RequestValidationError(
            [{"loc": ("query", "lead_time_hours"), "msg": "Field required", "type": "missing"}]
        )

    resolved_valid_time: datetime | None = None
    resolved_source_cycle: datetime | None = None
    target_initial: str | None = None
    cycle_time: str | None = None
    store_path: str | None = None
    serving_generation: str | None = None
    resolved_lead: int = 0

    if valid_time is not None:
        source = resolve_valid_time_source(db, model, valid_time, variable=variable)
        resolved_lead = source.lead_time_hours
        cycle_time = source.cycle_time.isoformat().replace("+00:00", "Z")
        store_path = source.store_path
        serving_generation = source.serving_generation
        target_initial = cycle_time
        resolved_valid_time = source.valid_time
        resolved_source_cycle = source.cycle_time
        db.close()
    else:
        assert lead_time_hours is not None
        resolved_lead = lead_time_hours
        target_initial = initial_time
        if target_initial is not None:
            require_cycle_visible(db, target_initial, model_id=model)

        cycle_time = resolve_latest_run_cycle_time(db, model, target_initial)
        store_path, latest_retired_iso = resolve_latest_run_store_path_and_retirement(
            db, model, target_initial
        )
        db.close()

        serving_generation = resolve_serving_generation_for_store(
            store_path, latest_retired_iso
        )

    cache_key = build_probability_cache_key(
        model=model,
        latitude=lat,
        longitude=lon,
        variable=variable,
        threshold=threshold,
        operator=operator,
        lead_time_hours=resolved_lead,
        threshold_max=threshold_max,
        direction_sector=direction_sector,
        phase=phase,
        cycle_time=cycle_time,
        serving_generation=serving_generation,
        valid_time=valid_time,
    )
    query_params = (
        f"lat={lat}&lon={lon}&variable={variable}&threshold={threshold}"
        f"&operator={operator}&lead_time_hours={resolved_lead}"
        f"&model={model}&threshold_max={threshold_max}&direction_sector={direction_sector}"
        f"&phase={phase}&initial_time={target_initial}&valid_time={valid_time}"
    )

    envelope = _cache.compute_or_retrieve(
        None,
        cache_key,
        query_params,
        lambda: _compute(
            lat,
            lon,
            variable,
            threshold,
            operator,
            resolved_lead,
            model,
            threshold_max,
            direction_sector,
            phase,
            target_initial,
            resolved_valid_time,
            resolved_source_cycle,
        ),
        model_type=ProbabilityForecastEnvelope,
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_PROBABILITY
    return envelope


def _validate_threshold_bounds(
    operator: Literal["gt", "gte", "lt", "lte", "between"],
    threshold: float,
    threshold_max: float | None,
) -> None:
    """Enforce the ``between``/``threshold_max`` pairing rules.

    ``between`` requires ``threshold_max``; ``gt``/``gte``/``lt``/``lte`` reject it. An
    out-of-range ordering (``threshold_max < threshold``) is surfaced later by
    the domain math as a 422.
    """
    if operator == "between" and threshold_max is None:
        raise HTTPException(
            status_code=422,
            detail="threshold_max is required when operator is 'between'.",
        )
    if operator in ("gt", "gte", "lt", "lte") and threshold_max is not None:
        raise HTTPException(
            status_code=422,
            detail="threshold_max is only valid when operator is 'between'.",
        )


def _compute(
    lat: float,
    lon: float,
    variable: str,
    threshold: float,
    operator: Literal["gt", "gte", "lt", "lte", "between"],
    lead_time_hours: int,
    model: str,
    threshold_max: float | None,
    direction_sector: str | None,
    phase: str | None,
    initial_time: str | None,
    valid_time_dt: datetime | None = None,
    source_cycle_dt: datetime | None = None,
) -> ProbabilityForecastEnvelope:
    from api.core.database import SessionLocal

    with SessionLocal() as db:
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
            direction_sector=direction_sector,
            phase=phase,
            initial_time=initial_time,
        )
        if valid_time_dt is not None:
            data.valid_time = valid_time_dt
        if source_cycle_dt is not None:
            data.source_cycle = source_cycle_dt
    return ProbabilityForecastEnvelope(data=data)
