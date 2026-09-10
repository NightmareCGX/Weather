"""Locate endpoint: coarse IP-based geolocation fallback via infrastructure headers.

When enabled and behind trusted Cloudflare infrastructure, extracts visitor
coordinates from Cloudflare Managed Transforms headers (cf-iplatitude,
cf-iplongitude, cf-ipcity, cf-region, cf-ipcountry).
"""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, Request, Response

from api.core.config import settings
from api.schemas import LocateOut

router = APIRouter()

#: Cache policy for IP location: private and no-cache so shared proxies never
#: serve one user's location to another user.
CACHE_CONTROL_LOCATE = "private, no-cache"


@router.get(
    "/locate",
    response_model=LocateOut,
    summary="Get coarse approximate location from infrastructure IP headers",
)
def locate(request: Request, response: Response) -> LocateOut:
    """Resolve coarse approximate location from trusted Cloudflare headers.

    Safety:
    - Disabled by default (TRUST_CLOUDFLARE_LOCATION_HEADERS=False). In this
      mode, externally supplied headers are never trusted and 404 is returned.
    - When enabled, headers are parsed, validated for finite range (lat in [-90, 90],
      lon in [-180, 180]), and normalized.
    - If headers are missing, malformed, or out of range, returns HTTP 404
      (location unavailable).
    """
    response.headers["Cache-Control"] = CACHE_CONTROL_LOCATE

    if not settings.TRUST_CLOUDFLARE_LOCATION_HEADERS:
        raise HTTPException(status_code=404, detail="Location unavailable")

    lat_str = request.headers.get("cf-iplatitude")
    lon_str = request.headers.get("cf-iplongitude")

    if not lat_str or not lon_str:
        raise HTTPException(status_code=404, detail="Location unavailable")

    try:
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=404, detail="Location unavailable") from exc

    if not math.isfinite(lat) or not math.isfinite(lon):
        raise HTTPException(status_code=404, detail="Location unavailable")

    if lat < -90.0 or lat > 90.0 or lon < -180.0 or lon > 180.0:
        raise HTTPException(status_code=404, detail="Location unavailable")

    # Canonicalize longitude (-180 becomes +180 per project convention)
    if lon == -180.0:
        lon = 180.0

    city = request.headers.get("cf-ipcity")
    region = request.headers.get("cf-region") or request.headers.get("cf-region-code")
    country = request.headers.get("cf-ipcountry") or request.headers.get("cf-country")

    return LocateOut(
        latitude=lat,
        longitude=lon,
        city=city if city else None,
        region=region if region else None,
        country=country if country else None,
        approximate=True,
    )
