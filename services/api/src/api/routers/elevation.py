"""Elevation endpoint: terrain elevation resolution for geographic coordinates.

This endpoint resolves terrain elevation in meters for dynamic coordinates
(arbitrary map clicks, custom GPS locations, place autocomplete results).
Elevation is UI display metadata only and does not participate in forecast
serving or meteorological calculations.
"""

from typing import Annotated

from fastapi import APIRouter, Query, Response

from api.schemas import ElevationOut
from api.services.elevation import get_elevation_provider

router = APIRouter()

#: Cache policy for successful terrain elevation (static per coordinate: 24 hours).
CACHE_CONTROL_ELEVATION_SUCCESS = "public, max-age=86400"
#: Cache policy for missing/unavailable elevation or transient provider failures.
#: Never cache transient failures or unavailable states for 24 hours; no-store
#: guarantees immediate retry on subsequent requests once the provider recovers.
CACHE_CONTROL_ELEVATION_UNAVAILABLE = "no-store"


@router.get(
    "/elevation",
    response_model=ElevationOut,
    summary="Get terrain elevation for geographic coordinates",
)
def get_elevation(
    response: Response,
    lat: Annotated[
        float,
        Query(
            ge=-90.0,
            le=90.0,
            description="WGS 84 latitude in decimal degrees [-90.0, 90.0].",
        ),
    ],
    lon: Annotated[
        float,
        Query(
            ge=-180.0,
            le=180.0,
            description="WGS 84 longitude in decimal degrees [-180.0, 180.0].",
        ),
    ],
) -> ElevationOut:
    """Return terrain elevation in meters for a coordinate, or null if unavailable.

    Delegates to the configured elevation provider (Open-Meteo or null) wrapped
    in a popularity-aware, decaying coordinate cache. Provider timeouts,
    outages, or ocean/void terrain values return HTTP 200 with ``elevation_m: null``
    so the client can render 'unavailable' gracefully without error states.

    HTTP Cache semantics:
    * Non-null elevation (successful lookup) -> ``public, max-age=86400``
    * Null elevation (failure/timeout/unavailable) -> ``no-store`` (prevents
      transient provider outages from being negatively cached for 24 hours).
    """
    elevation_m = get_elevation_provider().get_elevation(lat, lon)
    if elevation_m is not None:
        response.headers["Cache-Control"] = CACHE_CONTROL_ELEVATION_SUCCESS
    else:
        response.headers["Cache-Control"] = CACHE_CONTROL_ELEVATION_UNAVAILABLE
    return ElevationOut(
        latitude=lat,
        longitude=lon,
        elevation_m=elevation_m,
    )
