"""Ensemble statistics endpoint (API.md section 5.1).

Returns statistical dispersion (mean, median, spread, P10/P25/P50/P75/P90)
across ensemble perturbation members for a given location and lead time. The
router is thin (ENGINEERING_CONTRACT section 2): it validates parameters,
calls the ensemble-data and cache services, and serializes the documented
``ensemble_statistics`` envelope.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import EnsembleStatisticsEnvelope
from api.services.cache import PointCache, build_ensemble_cache_key
from api.services.ensemble_data import build_ensemble_statistics
from api.services.lifecycle import require_cycle_visible
from api.services.point_forecast import (
    resolve_latest_run_cycle_time,
    resolve_latest_run_store_path_and_retirement,
    resolve_serving_generation_for_store,
)

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for ensemble statistics: resolves to the newest ready run
#: when initial_time is omitted, so revalidation (no-cache) ensures new cycles
#: or same-cycle updates are seen immediately while Redis caches by generation.
CACHE_CONTROL_ENSEMBLE = "no-cache"

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
        int | None, Query(ge=0, description="Forecast offset hours from cycle time.")
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

    Under Lifecycle V2:
    - If ``valid_time`` is supplied, the newest committed source cycle and lead
      are dynamically resolved via the shared ValidTimeResolver.
    - If ``lead_time_hours`` is supplied without ``valid_time``, legacy cycle/lead
      serving is preserved for backward compatibility.
    """
    from api.services.resolver import resolve_valid_time_source

    if valid_time is not None and initial_time is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide either valid_time or initial_time, not both.",
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
        resolved_lead = lead_time_hours if lead_time_hours is not None else 0
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

    cache_key = build_ensemble_cache_key(
        model=model,
        latitude=lat,
        longitude=lon,
        variable=variable,
        lead_time_hours=resolved_lead,
        include_members=include_members,
        cycle_time=cycle_time,
        serving_generation=serving_generation,
        valid_time=valid_time,
    )
    query_params = (
        f"lat={lat}&lon={lon}&variable={variable}&model={model}"
        f"&lead_time_hours={resolved_lead}&include_members={include_members}"
        f"&initial_time={target_initial}&valid_time={valid_time}"
    )

    envelope = _cache.compute_or_retrieve(
        None,
        cache_key,
        query_params,
        lambda: _compute(
            lat,
            lon,
            variable,
            model,
            resolved_lead,
            include_members,
            target_initial,
            resolved_valid_time,
            resolved_source_cycle,
        ),
        model_type=EnsembleStatisticsEnvelope,
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_ENSEMBLE
    return envelope


def _compute(
    lat: float,
    lon: float,
    variable: str,
    model: str,
    lead_time_hours: int,
    include_members: bool,
    initial_time: str | None,
    valid_time_dt: datetime | None = None,
    source_cycle_dt: datetime | None = None,
) -> EnsembleStatisticsEnvelope:
    from api.core.database import SessionLocal

    with SessionLocal() as db:
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
        if valid_time_dt is not None:
            data.valid_time = valid_time_dt
        if source_cycle_dt is not None:
            data.source_cycle = source_cycle_dt
    return EnsembleStatisticsEnvelope(data=data)
