"""Search endpoint: location search and resolution (API.md section 6.1).

This endpoint performs location search/resolution over the platform's known
location records (cities, ski resorts, and observation stations). It is not
an address geocoding service: clients resolve a location here first, then
query ``/v1/points`` with the resolved coordinates or a platform id. The
router is thin (ENGINEERING_CONTRACT section 2): it validates parameters and
serializes results from :mod:`api.services.search`.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import ListEnvelope, SearchResultOut
from api.services.search import DEFAULT_LIMIT, DEFAULT_LOCATION_TYPE, search_locations

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for stable search results (API.md 6.1: 24 hours).
CACHE_CONTROL_DAILY = "public, max-age=86400"


@router.get(
    "/search",
    response_model=ListEnvelope[SearchResultOut],
    summary="Search locations",
)
def search(
    response: Response,
    q: Annotated[str, Query(min_length=1, description="Search query string.")],
    type: Annotated[
        Literal["city", "resort", "station", "all"],
        Query(description="Location type to search."),
    ] = DEFAULT_LOCATION_TYPE,
    limit: Annotated[int, Query(ge=1, le=100)] = DEFAULT_LIMIT,
    db: Session = DB,
) -> ListEnvelope[SearchResultOut]:
    """Search cities, ski resorts, and observation stations.

    The ``q`` parameter is required. ``type`` restricts the search to one
    location table (``city``, ``resort``, or ``station``) or searches all of
    them (``all``, the default). Results are returned in the universal list
    envelope.
    """
    results = search_locations(db, q, type, limit)
    response.headers["Cache-Control"] = CACHE_CONTROL_DAILY
    return ListEnvelope[SearchResultOut](data=results)
