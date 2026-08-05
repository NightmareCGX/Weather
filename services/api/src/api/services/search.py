"""PostGIS location search across cities, ski resorts, and stations.

This service performs location search/resolution over the platform's known
location records (the ``cities``, ``ski_resorts``, and ``stations`` tables).
It is not an address geocoding service: ``/v1/search`` matches existing
location records only, per API.md section 6.1. Clients resolve an address
through this search first, then query ``/v1/points`` with the resolved
coordinates or a platform id.
"""

from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.models.entities import City, SkiResort, Station
from api.schemas import SearchResultOut

#: Supported location types for the ``type`` query parameter (API.md 6.1).
VALID_LOCATION_TYPES = frozenset({"city", "resort", "station", "all"})
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
) -> list[SearchResultOut]:
    """Search cities, ski resorts, and stations by name.

    Matching is a case-insensitive substring match on the name field of each
    location type (API.md does not define matching semantics; ILIKE substring
    is assumed). ``location_type="all"`` (the default) merges results across
    all three source tables. ``limit`` is a true global limit: every matching
    row is collected from each selected table (no per-table pre-limit), the
    merged result is ordered deterministically by name, object type, then id,
    and only then truncated to ``limit``. This guarantees the top-``limit``
    matches across all tables are returned regardless of how matches are
    distributed across tables.

    Args:
        db: Database session.
        query: The ``q`` search query string.
        location_type: One of ``city``, ``resort``, ``station``, or ``all``.
        limit: Maximum number of results to return.

    Returns:
        The search result items in the API.md section 6.1 shape.
    """
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
