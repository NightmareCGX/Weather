"""Location search across platform records and external place providers.

Two search surfaces are provided (ACCEPTANCE_REMEDIATION_PLAN §13):

* the platform's own PostGIS records (``cities``, ``ski_resorts``,
  ``stations``) via ``ILIKE`` substring match — the original ``/v1/search``
  behavior (API.md section 6.1);
* **place autocomplete** via ``type=place``, backed by the
  :class:`~api.services.places.PlaceAutocompleteProvider` (Google Places API
  (New) by default), returning ranked global place suggestions and their
  canonical resolution.

Clients resolve a location through ``/v1/search`` first, then query
``/v1/points`` with the resolved coordinates or a platform id.
"""

from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.models.entities import City, SkiResort, Station
from api.schemas import SearchResultOut
from api.services.places import (
    PlaceSuggestion,
    get_provider,
)

#: Supported location types for the ``type`` query parameter (API.md 6.1).
VALID_LOCATION_TYPES = frozenset({"city", "resort", "station", "all", "place"})
#: Default location type when ``type`` is omitted (API.md lists ``all`` but
#: does not state a default; ``all`` is assumed).
DEFAULT_LOCATION_TYPE: Literal["all"] = "all"
#: Default result limit when ``limit`` is omitted.
DEFAULT_LIMIT = 20


def search_locations(
    db: Session,
    query: str,
    location_type: str = DEFAULT_LOCATION_TYPE,
    limit: int = DEFAULT_LIMIT,
    session_token: str | None = None,
) -> list[SearchResultOut]:
    """Search locations by name (or autocomplete places).

    Matching is a case-insensitive substring match on the name field of each
    platform location type (API.md does not define matching semantics; ILIKE
    substring is assumed). ``location_type="all"`` (the default) merges results
    across all three source tables. ``limit`` is a true global limit: every
    matching row is collected from each selected table (no per-table
    pre-limit), the merged result is ordered deterministically by name, object
    type, then id, and only then truncated to ``limit``. This guarantees the
    top-``limit`` matches across all tables are returned regardless of how
    matches are distributed across tables.

    When ``location_type="place"`` the search delegates to the configured
    place-autocomplete provider (Google Places API (New) by default), returning
    ranked global place suggestions.

    Args:
        db: Database session.
        query: The ``q`` search query string.
        location_type: One of ``city``, ``resort``, ``station``, ``all``, or
            ``place``.
        limit: Maximum number of results to return.

    Returns:
        The search result items in the API.md section 6.1 shape.
    """
    if location_type == "place":
        # Place autocomplete via the external provider (Google Places API
        # (New) by default). ``session_token`` is optional; when provided it
        # is forwarded so the same search session is billed once.
        return _search_places(query, limit, session_token)
    pattern = f"%{query}%"
    results: list[SearchResultOut] = []
    if location_type in ("city", "all"):
        results.extend(_search_cities(db, pattern))
    if location_type in ("resort", "all"):
        results.extend(_search_ski_resorts(db, pattern))
    if location_type in ("station", "all"):
        results.extend(_search_stations(db, pattern))
    results.sort(key=lambda item: (item.name, item.object, item.id))
    return results[:limit]


def _search_places(
    query: str,
    limit: int,
    session_token: str | None = None,
) -> list[SearchResultOut]:
    """Autocomplete places via the configured provider.

    Raises:
        PlaceAutocompleteError: If the provider call fails (mapped to 502 by
            the router so the combobox shows its error state gracefully).
    """
    suggestions = get_provider().suggest(query, session_token=session_token, limit=limit)
    return [_suggestion_to_result(item) for item in suggestions]


def _suggestion_to_result(suggestion: PlaceSuggestion) -> SearchResultOut:
    """Map a provider suggestion to the shared search-result shape.

    A suggestion is a *candidate*: it carries the provider place id but no
    resolved coordinates yet. Coordinates are populated only after the user
    selects the suggestion and the canonical place is resolved (via
    ``resolve_place``), so the combobox can update the map/forecast with the
    final lat/lon.
    """
    return SearchResultOut(
        id=f"place_{suggestion.place_id}",
        object="place",
        name=suggestion.main_text,
        region=suggestion.secondary_text,
        country=None,
        latitude=0.0,
        longitude=0.0,
        place_id=suggestion.place_id,
    )


def resolve_place(
    place_id: str,
    session_token: str | None = None,
) -> SearchResultOut:
    """Resolve a place suggestion's canonical location.

    Returns a ``SearchResultOut`` populated with the canonical display name,
    coordinates, country, and region. Used when the user selects an
    autocomplete suggestion, so the map recenters and the point forecast
    updates with real coordinates.

    Raises:
        PlaceAutocompleteError: If the provider cannot resolve the place.
    """
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
        )
        for row in db.execute(stmt).all()
    ]
