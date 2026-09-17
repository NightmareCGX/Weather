"""Locate endpoint: coarse IP-based geolocation for the startup map viewport.

Two sources are consulted in order:

1. Cloudflare Managed Transforms headers (``cf-iplatitude`` / ``cf-iplongitude``),
   trusted only under ``TRUST_CLOUDFLARE_LOCATION_HEADERS`` on an origin that is
   firewalled to Cloudflare. These cost nothing to read, so they win when present.
2. A local GeoLite2 database (``LOCATE_PROVIDER=maxmind``) resolving the visitor
   address from the request itself, for self-hosted origins with no CDN in front
   of them, where the ``cf-*`` headers can never arrive.

Every failure — disabled, unusable address, address not in the database — answers
the same HTTP 404, so the endpoint never tells a caller which branch it hit and
the frontend keeps degrading silently to the default CONUS viewport.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, Request, Response

from api.core.config import settings
from api.core.geoip import (
    PROVIDER_MAXMIND,
    GeoLocation,
    client_ip,
    is_public_ip,
    lookup_location,
)
from api.schemas import LocateOut

router = APIRouter()

#: Cache policy for IP location: private and no-cache so shared proxies never
#: serve one user's location to another user.
CACHE_CONTROL_LOCATE = "private, no-cache"


def _unavailable() -> HTTPException:
    """The single failure response shared by every unavailable-location path."""
    return HTTPException(status_code=404, detail="Location unavailable")


def _canonical_longitude(longitude: float) -> float:
    """Canonicalize longitude (-180 becomes +180 per project convention)."""
    return 180.0 if longitude == -180.0 else longitude


def _from_cloudflare_headers(request: Request) -> LocateOut | None:
    """Resolve the visitor location from trusted Cloudflare infrastructure headers.

    Returns None when the trust flag is off, a header is missing, or the payload
    is not a finite in-range coordinate pair, so the caller can fall through to
    the next source instead of failing the request.
    """
    if not settings.TRUST_CLOUDFLARE_LOCATION_HEADERS:
        return None

    lat_str = request.headers.get("cf-iplatitude")
    lon_str = request.headers.get("cf-iplongitude")
    if not lat_str or not lon_str:
        return None

    try:
        latitude = float(lat_str.strip())
        longitude = float(lon_str.strip())
    except (ValueError, TypeError):
        return None

    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None

    if latitude < -90.0 or latitude > 90.0 or longitude < -180.0 or longitude > 180.0:
        return None

    city = request.headers.get("cf-ipcity")
    region = request.headers.get("cf-region") or request.headers.get("cf-region-code")
    country = request.headers.get("cf-ipcountry") or request.headers.get("cf-country")

    return LocateOut(
        latitude=latitude,
        longitude=_canonical_longitude(longitude),
        city=city or None,
        region=region or None,
        country=country or None,
        approximate=True,
    )


def _from_local_database(request: Request) -> LocateOut | None:
    """Resolve the visitor location from the local GeoLite2 database."""
    if str(settings.LOCATE_PROVIDER).strip().lower() != PROVIDER_MAXMIND:
        return None

    address = client_ip(request)
    if address is None or not is_public_ip(address):
        return None

    resolved: GeoLocation | None = lookup_location(address)
    if resolved is None:
        return None

    return LocateOut(
        latitude=resolved.latitude,
        longitude=_canonical_longitude(resolved.longitude),
        city=resolved.city,
        region=resolved.region,
        country=resolved.country,
        approximate=True,
    )


@router.get(
    "/locate",
    response_model=LocateOut,
    summary="Get coarse approximate location from the visitor's IP address",
)
def locate(request: Request, response: Response) -> LocateOut:
    """Resolve a coarse approximate location for the requesting visitor.

    Safety:
    - Both sources are disabled by default (``TRUST_CLOUDFLARE_LOCATION_HEADERS``
      False, ``LOCATE_PROVIDER`` ``none``): externally supplied headers are never
      trusted and no lookup is performed unless explicitly enabled.
    - Cloudflare values are validated for finite range (lat in [-90, 90], lon in
      [-180, 180]) and normalized.
    - The database source only accepts routable addresses, so loopback, private
      and same-host requests are treated as unavailable rather than resolving to
      whatever the database records for that range.
    """
    response.headers["Cache-Control"] = CACHE_CONTROL_LOCATE

    location = _from_cloudflare_headers(request)
    if location is None:
        location = _from_local_database(request)
    if location is None:
        raise _unavailable()
    return location
