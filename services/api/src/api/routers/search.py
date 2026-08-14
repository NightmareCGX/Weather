"""Search endpoint: location search and resolution (API.md section 6.1).

This endpoint performs location search/resolution over the platform's known
location records (cities, ski resorts, and observation stations) plus, via
``type=place``, external place autocomplete (Google Places API (New) by
default). It is not an unrestricted geocoder: clients resolve a location here
first, then query ``/v1/points`` with the resolved coordinates or a platform
id. The router is thin (ENGINEERING_CONTRACT section 2): it validates
parameters and serializes results from :mod:`api.services.search`.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import ListEnvelope, SearchResultOut
from api.services.places import PlaceAutocompleteError
from api.services.search import (
    DEFAULT_LIMIT,
    DEFAULT_LOCATION_TYPE,
    resolve_place,
    search_locations,
)

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for stable search results (API.md 6.1: 24 hours).
CACHE_CONTROL_DAILY = "public, max-age=86400"
#: Cache policy for place-autocomplete results (quota-conscious: short TTL so
#: repeated queries can be served from the browser cache, but provider results
#: are not considered immutable).
CACHE_CONTROL_PLACES = "public, max-age=300"


@router.get(
    "/search",
    response_model=ListEnvelope[SearchResultOut],
    summary="Search locations",
)
def search(
    response: Response,
    q: Annotated[str, Query(min_length=1, description="Search query string.")],
    type: Annotated[
        Literal["city", "resort", "station", "all", "place"],
        Query(description="Location type to search."),
    ] = DEFAULT_LOCATION_TYPE,
    limit: Annotated[int, Query(ge=1, le=100)] = DEFAULT_LIMIT,
    session_token: Annotated[
        str | None,
        Query(
            description=(
                "Places search-session token (Google billing semantics): one "
                "token spans an autocomplete session so the provider bills one "
                "Autocomplete request instead of one per keystroke."
            )
        ),
    ] = None,
    db: Session = DB,
) -> ListEnvelope[SearchResultOut]:
    """Search cities, ski resorts, stations, or places.

    The ``q`` parameter is required. ``type`` restricts the search to one
    platform location table (``city``, ``resort``, or ``station``), searches
    all of them (``all``, the default), or delegates to the place-autocomplete
    provider (``place``). Results are returned in the universal list envelope.
    """
    try:
        results = search_locations(db, q, type, limit, session_token=session_token)
    except PlaceAutocompleteError as exc:
        # The provider failed: surface a graceful 502 so the combobox shows
        # its error state rather than crashing or returning a partial result.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    response.headers["Cache-Control"] = (
        CACHE_CONTROL_PLACES if type == "place" else CACHE_CONTROL_DAILY
    )
    return ListEnvelope[SearchResultOut](data=results)


@router.get(
    "/search/places/{place_id}",
    response_model=SearchResultOut,
    summary="Resolve a place suggestion",
)
def get_place(
    place_id: str,
    session_token: Annotated[
        str | None,
        Query(
            description=(
                "Places search-session token; reuse the same token that was "
                "used for the autocomplete call so the session is billed once."
            )
        ),
    ] = None,
) -> SearchResultOut:
    """Resolve a selected place suggestion to its canonical location.

    The user selects an autocomplete suggestion (which carries a ``place_id``
    but no coordinates yet); this endpoint resolves the canonical display
    name, latitude/longitude, country, and region so the map recenters and the
    point forecast updates.
    """
    try:
        return resolve_place(place_id, session_token=session_token)
    except PlaceAutocompleteError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
