"""Point forecast endpoint (API.md section 2.1).

Returns hourly forecasts indexed by ``lead_time_hours`` for a specific,
already-resolved geographic location. The router is thin (ENGINEERING_CONTRACT
section 2): it validates parameters, calls the point forecast and cache
services, and serializes the documented ``point_forecast`` envelope.

Specification gaps (not implemented in Milestone 9):

* **Address geocoding.** API.md section 2.1 lists ``address`` as a spatial
  specifier, but the Milestone 3 schema has no address/geocoding table and
  the design documents define no geocoding service. This endpoint does not
  geocode addresses; clients resolve a location through ``/v1/search`` first
  and then query ``/v1/points`` with ``lat``/``lon``, ``city_id``, or
  ``resort_id``.

* **Multi-model responses.** API.md documents ``models`` as a single model
  identifier (default ``gfs``) and defines no response format for multiple
  models. This endpoint exposes an unambiguous single-model contract:
  ``models`` accepts exactly one identifier (default ``gfs``). Requests that
  pass more than one identifier are rejected with HTTP 422 and a clear
  message; multi-model responses are recorded as a specification gap to
  resolve in a future contract update.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import PointForecastEnvelope
from api.services.cache import PointCache, build_point_cache_key
from api.services.point_forecast import (
    ResolvedLocation,
    build_point_forecast,
    resolve_latest_run_cycle_time,
    resolve_latest_run_serving_generation,
    resolve_location,
)

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for point forecasts (API.md 2.1: 30 minutes).
CACHE_CONTROL_POINT = "public, max-age=1800"

#: In-memory point cache instance for the API process.
_cache = PointCache()


@router.get(
    "/points",
    response_model=PointForecastEnvelope,
    summary="Get point forecast",
)
def get_point_forecast(
    response: Response,
    lat: Annotated[float | None, Query()] = None,
    lon: Annotated[float | None, Query()] = None,
    city_id: Annotated[str | None, Query()] = None,
    resort_id: Annotated[str | None, Query()] = None,
    models: Annotated[str, Query(description="A single model identifier.")] = "gfs",
    variables: Annotated[str | None, Query()] = None,
    units: Annotated[Literal["metric", "imperial"], Query()] = "metric",
    start_lead_time_hours: Annotated[int | None, Query(ge=0)] = None,
    end_lead_time_hours: Annotated[int | None, Query(ge=0)] = None,
    db: Session = DB,
) -> PointForecastEnvelope:
    """Return hourly forecasts for a resolved geographic location.

    Exactly one spatial specifier is required: a ``lat``/``lon`` pair, a
    ``city_id``, or a ``resort_id``. ``models`` accepts exactly one model
    identifier; multi-model requests are rejected with 422 (see the module
    docstring).
    """
    model_ids = _parse_models(models)

    location = resolve_location(
        db, lat=lat, lon=lon, city_id=city_id, resort_id=resort_id
    )
    var_codes = _parse_variables(variables)

    cache_key = build_point_cache_key(
        model=model_ids[0],
        latitude=location.latitude,
        longitude=location.longitude,
        resolved_via=location.resolved_via,
        location_id=location.id,
        # The newest READY cycle is the cache discriminator: when a new cycle
        # becomes READY it may change the minimum-lead selection, so the cached
        # cross-cycle series must be invalidated. The series itself is built
        # from all READY cycles.
        cycle_time=resolve_latest_run_cycle_time(db, model_ids[0]),
        serving_generation=resolve_latest_run_serving_generation(db, model_ids[0]),
        variables=tuple(var_codes) if var_codes else None,
        units=units,
        start_lead_time_hours=start_lead_time_hours,
        end_lead_time_hours=end_lead_time_hours,
        cross_cycle=True,
    )
    query_params = (
        f"lat={lat}&lon={lon}&city_id={city_id}&resort_id={resort_id}"
        f"&models={models}&variables={variables}&units={units}"
        f"&start_lead_time_hours={start_lead_time_hours}"
        f"&end_lead_time_hours={end_lead_time_hours}"
    )

    envelope = _cache.compute_or_retrieve(
        db,
        cache_key,
        query_params,
        lambda: _compute(
            db,
            location,
            model_ids[0],
            var_codes,
            units,
            start_lead_time_hours,
            end_lead_time_hours,
        ),
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_POINT
    return envelope


def _parse_models(models: str) -> list[str]:
    """Validate the ``models`` parameter as a single model identifier.

    Rejects multi-model requests with a 422 explaining that the multi-model
    response contract is not defined by the approved specification, rather
    than silently discarding requested models.
    """
    identifiers = [part.strip() for part in models.split(",") if part.strip()]
    if not identifiers:
        raise HTTPException(
            status_code=422,
            detail="The models parameter must contain at least one model identifier.",
        )
    if len(identifiers) > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Multi-model point forecasts are not yet supported: the "
                "response contract for multiple models is not defined by the "
                "approved specification. Request a single model (e.g. "
                "models=gfs)."
            ),
        )
    return identifiers


def _parse_variables(variables: str | None) -> list[str] | None:
    if variables is None or variables.strip() == "":
        return None
    return [part.strip() for part in variables.split(",") if part.strip()]


def _compute(
    db: Session,
    location: ResolvedLocation,
    model: str,
    var_codes: list[str] | None,
    units: str,
    start_lead_time_hours: int | None,
    end_lead_time_hours: int | None,
) -> PointForecastEnvelope:
    data = build_point_forecast(
        db,
        location=location,
        model=model,
        variables=var_codes,
        units=units,
        start_lead_time_hours=start_lead_time_hours,
        end_lead_time_hours=end_lead_time_hours,
    )
    return PointForecastEnvelope(data=data)
