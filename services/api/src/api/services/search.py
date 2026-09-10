"""Location search across platform records and external place providers.

Search surfaces provided (Phase 1 Location Discovery):

* **Local Weather Entities**: Station fast-path (exact match on
  ``stations.station_code``), plus PostGIS records (``cities``,
  ``ski_resorts``, ``stations``) via ``ILIKE`` substring match.
* **External Place Autocomplete**: Backed by the configured primary provider
  (Geoapify Address Autocomplete by default in V1) with automatic circuit
  breaker and failover to a secondary provider (LocationIQ Autocomplete),
  returning direct WGS84 coordinates.

Clients resolve a location through ``/v1/search`` first, then query
``/v1/points`` with the resolved coordinates or a platform id.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.core.config import settings
from api.models.entities import City, SkiResort, Station
from api.schemas import SearchBias, SearchResultOut
from api.services.circuit_breaker import circuit_breaker, search_cache
from api.services.places import (
    PlaceAutocompleteError,
    PlaceRateLimitError,
    PlaceSuggestion,
    PlaceTimeoutError,
    get_fallback_provider,
    get_provider,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchDiagnostics:
    """Safe diagnostic metadata for search execution telemetry and tests."""

    cache_hit: bool = False
    primary_provider: str = "geoapify"
    primary_attempted: bool = False
    primary_latency_ms: float | None = None
    primary_result: str | None = None
    primary_timeout: bool = False
    fallback_provider: str = "locationiq"
    fallback_attempted: bool = False
    fallback_latency_ms: float | None = None
    fallback_result: str | None = None

#: Supported location types for the ``type`` query parameter (API.md 6.1).
VALID_LOCATION_TYPES = frozenset({"city", "resort", "station", "all", "place"})
#: Default location type when ``type`` is omitted (API.md lists ``all`` but
#: does not state a default; ``all`` is assumed).
DEFAULT_LOCATION_TYPE: Literal["all"] = "all"
#: Default result limit when ``limit`` is omitted.
DEFAULT_LIMIT = 20


def match_station_fast_path(db: Session, query: str) -> SearchResultOut | None:
    """Exact match lookup against stations.station_code (case-insensitive).

    Only single-token queries up to 20 characters are checked against the
    indexed ``station_code`` column. Ordinary words like 'VAIL' or 'ROME' that
    do not exist in the stations table safely return None.
    """
    cleaned = query.strip()
    if not cleaned or len(cleaned) > 20 or " " in cleaned:
        return None

    stmt = (
        select(Station, func.ST_X(Station.geom), func.ST_Y(Station.geom))
        .where(func.upper(Station.station_code) == func.upper(cleaned))
        .limit(1)
    )
    row = db.execute(stmt).first()
    if row is None:
        return None

    station = row[0]
    return SearchResultOut(
        id=station.id,
        object="station",
        name=station.name,
        elevation_m=station.elevation_m,
        latitude=float(row[2]),
        longitude=float(row[1]),
        place_id=station.station_code,
    )


def search_locations(
    db: Session | None,
    query: str,
    location_type: str = DEFAULT_LOCATION_TYPE,
    limit: int = DEFAULT_LIMIT,
    session_token: str | None = None,
    bias: SearchBias | None = None,
    diagnostics: SearchDiagnostics | None = None,
) -> list[SearchResultOut]:
    """Search cities, ski resorts, stations, and external places.

    When ``location_type`` is ``city``, ``resort``, or ``station``, only that
    specific platform table is queried.

    When ``location_type`` is ``all`` or ``place``, the gateway combines local
    weather entities (station fast-path, ski resorts, cities) with external
    place autocomplete (Geoapify primary with LocationIQ fallback).
    """
    if location_type == "station":
        if db is None:
            return []
        pattern = f"%{query}%"
        exact = match_station_fast_path(db, query)
        stations = _search_stations(db, pattern)
        results: list[SearchResultOut] = []
        if exact is not None:
            results.append(exact)
        for st in stations:
            if exact is None or st.id != exact.id:
                results.append(st)
        return results[:limit]

    if location_type == "city":
        if db is None:
            return []
        pattern = f"%{query}%"
        return _search_cities(db, pattern)[:limit]

    if location_type == "resort":
        if db is None:
            return []
        pattern = f"%{query}%"
        return _search_ski_resorts(db, pattern)[:limit]

    if location_type == "place":
        try:
            return _search_places_with_failover(
                query,
                limit,
                session_token=session_token,
                bias=bias,
                diagnostics=diagnostics,
            )[:limit]
        except PlaceAutocompleteError:
            if db is None:
                raise
            # If db is available, fall back to local cities/resorts if any
            fallback_local: list[SearchResultOut] = []
            pattern = f"%{query}%"
            fallback_local.extend(_search_cities(db, pattern))
            fallback_local.extend(_search_ski_resorts(db, pattern))
            if fallback_local:
                return fallback_local[:limit]
            raise

    # --- location_type == "all" (the default gateway search) ---
    local_results: list[SearchResultOut] = []
    station_match: SearchResultOut | None = None
    pattern = f"%{query}%"

    if db is not None:
        try:
            station_match = match_station_fast_path(db, query)
            local_results.extend(_search_ski_resorts(db, pattern))
            local_results.extend(_search_cities(db, pattern))
            stations = _search_stations(db, pattern)
            if station_match is not None:
                stations = [st for st in stations if st.id != station_match.id]
            local_results.extend(stations)
        except Exception as exc:
            logger.warning("Local database search unavailable: %s", exc)

    # Query external place autocomplete (Geoapify -> LocationIQ -> local fallback)
    external_results: list[SearchResultOut] = []
    try:
        external_results = _search_places_with_failover(
            query,
            limit,
            session_token=session_token,
            bias=bias,
            diagnostics=diagnostics,
        )
    except PlaceAutocompleteError:
        logger.warning("External place providers failed; continuing with local results.")

    # Deduplicate external places matching local entities by (name, region)
    seen_local = {
        (item.name.lower(), (item.region or "").lower())
        for item in local_results
    }
    filtered_external = [
        item
        for item in external_results
        if (item.name.lower(), (item.region or "").lower()) not in seen_local
    ]

    # Assemble final merged result list:
    # 1. Exact station match (if any)
    # 2. Local entities sorted deterministically by name, object, id
    # 3. External place results
    final_results: list[SearchResultOut] = []
    if station_match is not None:
        final_results.append(station_match)

    local_results.sort(key=lambda item: (item.name, item.object, item.id))
    final_results.extend(local_results)
    final_results.extend(filtered_external)

    return final_results[:limit]


def _search_places_with_failover(
    query: str,
    limit: int,
    session_token: str | None = None,
    bias: SearchBias | None = None,
    diagnostics: SearchDiagnostics | None = None,
) -> list[SearchResultOut]:
    """Execute external place autocomplete with circuit breaking and fallback."""
    primary = get_provider()
    primary_name = getattr(primary, "provider_name", "mock")
    if diagnostics is not None:
        diagnostics.primary_provider = primary_name

    # 1. Check response cache
    cached = search_cache.get(primary_name, query, bias)
    if cached is not None:
        logger.info("Search cache hit: provider=%s query=%s", primary_name, query)
        if diagnostics is not None:
            diagnostics.cache_hit = True
            diagnostics.primary_result = "cache_hit"
        return cached[:limit]

    primary_error: PlaceAutocompleteError | None = None
    fallback_error: PlaceAutocompleteError | None = None

    # 2. Try primary provider if healthy
    if circuit_breaker.is_available(primary_name):
        if diagnostics is not None:
            diagnostics.primary_attempted = True
        t0 = time.perf_counter()
        try:
            logger.info("Primary search attempt: provider=%s query=%s", primary_name, query)
            try:
                suggestions = primary.suggest(
                    query, session_token=session_token, limit=limit, bias=bias
                )
            except TypeError:
                suggestions = primary.suggest(
                    query, session_token=session_token, limit=limit
                )
            circuit_breaker.record_success(primary_name)
            if diagnostics is not None:
                diagnostics.primary_latency_ms = (time.perf_counter() - t0) * 1000
                diagnostics.primary_result = "success"
            logger.info(
                "Primary search success: provider=%s query=%s count=%d",
                primary_name,
                query,
                len(suggestions),
            )
            results = [_suggestion_to_result(s) for s in suggestions]
            search_cache.set(
                primary_name,
                query,
                results,
                bias,
                ttl_seconds=settings.SEARCH_CACHE_TTL_SECONDS,
            )
            return results
        except PlaceAutocompleteError as exc:
            primary_error = exc
            is_429 = isinstance(exc, PlaceRateLimitError)
            is_timeout = isinstance(exc, PlaceTimeoutError)
            if diagnostics is not None:
                diagnostics.primary_latency_ms = (time.perf_counter() - t0) * 1000
                diagnostics.primary_timeout = is_timeout
                diagnostics.primary_result = (
                    "timeout" if is_timeout else ("rate_limit" if is_429 else "error")
                )
            circuit_breaker.record_failure(primary_name, is_429=is_429)
            logger.warning(
                "Primary search provider '%s' failed (429=%s): %s",
                primary_name,
                is_429,
                exc,
            )
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.primary_latency_ms = (time.perf_counter() - t0) * 1000
                diagnostics.primary_result = "error"
            circuit_breaker.record_failure(primary_name, is_429=False)
            logger.warning(
                "Unexpected error in primary search provider '%s': %s",
                primary_name,
                exc,
            )
    else:
        logger.info(
            "Primary search provider '%s' circuit is open; routing to fallback",
            primary_name,
        )
        if diagnostics is not None:
            diagnostics.primary_result = "circuit_open"

    # 3. Try fallback provider
    fallback = get_fallback_provider()
    if fallback is not None:
        fallback_name = (
            "locationiq" if settings.SEARCH_PROVIDER == "geoapify" else "geoapify"
        )
        if diagnostics is not None:
            diagnostics.fallback_provider = fallback_name

        cached_fb = search_cache.get(fallback_name, query, bias)
        if cached_fb is not None:
            logger.info("Fallback search cache hit: provider=%s query=%s", fallback_name, query)
            if diagnostics is not None:
                diagnostics.cache_hit = True
                diagnostics.fallback_result = "cache_hit"
            return cached_fb[:limit]

        if circuit_breaker.is_available(fallback_name):
            if diagnostics is not None:
                diagnostics.fallback_attempted = True
            t0_fb = time.perf_counter()
            try:
                logger.info(
                    "Fallback search attempt: provider=%s query=%s", fallback_name, query
                )
                try:
                    suggestions = fallback.suggest(
                        query, session_token=session_token, limit=limit, bias=bias
                    )
                except TypeError:
                    suggestions = fallback.suggest(
                        query, session_token=session_token, limit=limit
                    )
                circuit_breaker.record_success(fallback_name)
                if diagnostics is not None:
                    diagnostics.fallback_latency_ms = (time.perf_counter() - t0_fb) * 1000
                    diagnostics.fallback_result = "success"
                logger.info(
                    "Fallback search success: provider=%s query=%s count=%d",
                    fallback_name,
                    query,
                    len(suggestions),
                )
                results = [_suggestion_to_result(s) for s in suggestions]
                search_cache.set(
                    fallback_name,
                    query,
                    results,
                    bias,
                    ttl_seconds=settings.SEARCH_CACHE_TTL_SECONDS,
                )
                return results
            except PlaceAutocompleteError as exc:
                fallback_error = exc
                is_429 = isinstance(exc, PlaceRateLimitError)
                if diagnostics is not None:
                    diagnostics.fallback_latency_ms = (time.perf_counter() - t0_fb) * 1000
                    diagnostics.fallback_result = (
                        "rate_limit" if is_429 else "error"
                    )
                circuit_breaker.record_failure(fallback_name, is_429=is_429)
                logger.warning(
                    "Fallback search provider '%s' failed (429=%s): %s",
                    fallback_name,
                    is_429,
                    exc,
                )
            except Exception as exc:
                if diagnostics is not None:
                    diagnostics.fallback_latency_ms = (time.perf_counter() - t0_fb) * 1000
                    diagnostics.fallback_result = "error"
                circuit_breaker.record_failure(fallback_name, is_429=False)
                logger.warning(
                    "Unexpected error in fallback search provider '%s': %s",
                    fallback_name,
                    exc,
                )
        else:
            logger.info(
                "Fallback search provider '%s' circuit is open; skipping",
                fallback_name,
            )
            if diagnostics is not None:
                diagnostics.fallback_result = "circuit_open"

    if primary_error is not None:
        raise primary_error
    if fallback_error is not None:
        raise fallback_error

    raise PlaceAutocompleteError("All search providers unavailable (circuits open)")


def _suggestion_to_result(suggestion: PlaceSuggestion) -> SearchResultOut:
    """Map a provider suggestion to the shared search-result shape."""
    return SearchResultOut(
        id=f"place_{suggestion.place_id}",
        object="place",
        name=suggestion.main_text,
        region=suggestion.secondary_text,
        country=suggestion.country,
        latitude=suggestion.latitude,
        longitude=suggestion.longitude,
        elevation_m=suggestion.elevation_m,
        place_id=suggestion.place_id,
    )


def resolve_place(
    place_id: str,
    session_token: str | None = None,
) -> SearchResultOut:
    """Resolve a place suggestion's canonical location (backwards compatibility)."""
    place = get_provider().resolve(place_id, session_token=session_token)
    return SearchResultOut(
        id=f"place_{place.place_id}",
        object="place",
        name=place.display_name,
        region=place.region,
        country=place.country,
        latitude=place.latitude,
        longitude=place.longitude,
        place_id=place.place_id,
    )


def _search_cities(db: Session, pattern: str) -> list[SearchResultOut]:
    stmt = (
        select(City, func.ST_X(City.geom), func.ST_Y(City.geom))
        .where(City.city_name.ilike(pattern))
        .order_by(City.city_name.asc())
    )
    return [
        SearchResultOut(
            id=row[0].id,
            object="city",
            name=row[0].city_name,
            region=row[0].region,
            country=row[0].country,
            elevation_m=row[0].elevation_m,
            latitude=float(row[2]),
            longitude=float(row[1]),
        )
        for row in db.execute(stmt).all()
    ]


def _search_ski_resorts(db: Session, pattern: str) -> list[SearchResultOut]:
    stmt = (
        select(SkiResort, func.ST_X(SkiResort.geom), func.ST_Y(SkiResort.geom))
        .where(SkiResort.resort_name.ilike(pattern))
        .order_by(SkiResort.resort_name.asc())
    )
    return [
        SearchResultOut(
            id=row[0].id,
            object="ski_resort",
            name=row[0].resort_name,
            region=row[0].region,
            country=row[0].country,
            elevation_m=row[0].summit_elevation_m,
            latitude=float(row[2]),
            longitude=float(row[1]),
        )
        for row in db.execute(stmt).all()
    ]


def _search_stations(db: Session, pattern: str) -> list[SearchResultOut]:
    stmt = (
        select(Station, func.ST_X(Station.geom), func.ST_Y(Station.geom))
        .where(Station.name.ilike(pattern))
        .order_by(Station.name.asc())
    )
    return [
        SearchResultOut(
            id=row[0].id,
            object="station",
            name=row[0].name,
            elevation_m=row[0].elevation_m,
            latitude=float(row[2]),
            longitude=float(row[1]),
            place_id=row[0].station_code,
        )
        for row in db.execute(stmt).all()
    ]
