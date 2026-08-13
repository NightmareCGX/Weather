"""Forecast availability endpoint: what forecast data actually exists.

``GET /v1/forecast/availability`` returns the real set of available forecast
combinations — model, variable, initial time (cycle time), and lead times —
derived entirely from the PostgreSQL catalog (``model_runs`` +
``forecast_products`` + ``forecast_variables`` + ``models``). The frontend
uses this single response to build its Model / Variable / Initial Time / Lead
Time selectors, so every option shown is traceable to a real ``ready`` run.

This is a read-only discovery endpoint that is a non-breaking addition to the
v1 surface (API.md section 1.3: additive endpoints are allowed). The router
is thin (ENGINEERING_CONTRACT section 2): it calls the availability service
and serializes the envelope.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import ForecastAvailabilityEnvelope
from api.services.availability import build_forecast_availability

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy: availability changes only when ingestion writes new runs, so
#: a short TTL keeps the selectors fresh after an ingest without hammering the
#: database on every interaction.
CACHE_CONTROL_AVAILABILITY = "public, max-age=60"


@router.get(
    "/forecast/availability",
    response_model=ForecastAvailabilityEnvelope,
    summary="Get available forecast combinations",
)
def get_forecast_availability(
    response: Response,
    db: Session = DB,
) -> ForecastAvailabilityEnvelope:
    """Return the available model/variable/initial-time/lead-time structure.

    The response is generated from the database on every request, so newly
    ingested runs (new models, variables, initial times, or lead times)
    become visible automatically without any code or configuration change.
    """
    data = build_forecast_availability(db)
    response.headers["Cache-Control"] = CACHE_CONTROL_AVAILABILITY
    return ForecastAvailabilityEnvelope(data=data)
