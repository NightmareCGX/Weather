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

The response is served with ``Cache-Control: no-cache`` plus a strong ``ETag``,
so a client polling it (the frontend does so every 60s) revalidates rather
than re-downloads: the payload is ~444KB, and its content is invariant for the
whole 3-hour serving window.
"""

import hashlib
import json
from datetime import datetime

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.time import get_current_time
from api.schemas import ForecastAvailabilityData, ForecastAvailabilityEnvelope
from api.services.availability import build_forecast_availability

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)
#: Current UTC time dependency for serving window left boundary.
CURRENT_TIME = Depends(get_current_time)

#: Cache policy: availability is derived dynamically from PostgreSQL, so
#: revalidation (no-cache) guarantees newly ingested runs are visible
#: immediately on browser refresh without waiting for a stale TTL.
CACHE_CONTROL_AVAILABILITY = "no-cache"


def _etag_for(data: ForecastAvailabilityData) -> str:
    """Build a strong ETag over the payload's serving-relevant content.

    ``generated_at`` is deliberately EXCLUDED: the service stamps it with the
    per-request wall clock, so hashing the whole payload would mint a distinct
    tag on every request and no revalidation would ever hit.

    The remaining content is a pure function of database state and the floored
    3-hour serving boundary: every consumer of the request's ``now`` (the
    builder and the canonical resolver alike) derives its window from
    ``serving_start_valid_time(now)``, never from the raw clock. The tag is
    therefore stable across requests within a serving window and across
    processes and restarts, which is what lets a shared/revalidating client
    cache collapse repeated polls to a 304.
    """
    canonical = json.dumps(
        {
            key: value
            # ``model_dump(mode="json")`` runs the schema's field serializers,
            # so the tag is keyed on exactly the bytes the client would have
            # received rather than on a Python-repr of the same values.
            for key, value in data.model_dump(mode="json").items()
            if key != "generated_at"
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return '"' + hashlib.sha1(canonical.encode("utf-8")).hexdigest() + '"'


@router.get(
    "/forecast/availability",
    response_model=ForecastAvailabilityEnvelope,
    summary="Get available forecast combinations",
)
def get_forecast_availability(
    request: Request,
    response: Response,
    db: Session = DB,
    now: datetime = CURRENT_TIME,
) -> ForecastAvailabilityEnvelope | Response:
    """Return the available model/variable/initial-time/lead-time structure.

    The response is generated from the database on every request, so newly
    ingested runs (new models, variables, initial times, or lead times)
    become visible automatically without any code or configuration change.

    When the caller presents an ``If-None-Match`` that matches the current
    payload, a bodyless ``304`` is returned instead of the full payload.
    """
    data = build_forecast_availability(db, now=now)
    etag = _etag_for(data)
    headers = {"Cache-Control": CACHE_CONTROL_AVAILABILITY, "ETag": etag}

    if_none_match = request.headers.get("if-none-match")
    if if_none_match is not None and etag in if_none_match:
        return Response(status_code=304, headers=headers)

    response.headers.update(headers)
    return ForecastAvailabilityEnvelope(data=data)
