"""Location place-autocomplete provider for the serving tier.

The frontend combobox already provides debounce/abort/stale-guard/keyboard
navigation; what it lacked is a real place-autocomplete data source. This
module is the backend provider abstraction that backs ``/v1/search?type=place``
(ACCEPTANCE_REMEDIATION_PLAN §13):

* :class:`PlaceAutocompleteProvider` is the application-level interface;
* :class:`GooglePlacesAutocompleteProvider` calls the **Places API (New)**
  Autocomplete + Place Details endpoints server-side, so the API key never
  reaches the browser;
* :class:`MapboxGeocodingProvider` is the drop-in alternative behind the same
  interface.

Session-token semantics: a fresh UUIDv4 is generated per search session, reused
for every autocomplete keystroke and the subsequent place-details resolution, so
Google bills one session per completed selection rather than per keystroke.

The provider is network-free by construction for tests: the HTTP transport is a
small injectable callable (``httpx``-style), so tests supply a fake transport and
never touch live Google services.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from api.core.config import settings

#: Default max suggestions returned by autocomplete.
DEFAULT_SUGGESTION_LIMIT = 8
#: Place types included in autocomplete so results are actual places (localities,
#: addresses), not text queries (which are billed at the higher Text Search rate).
DEFAULT_INCLUDED_PRIMARY_TYPES = ["locality", "address", "airport", "establishment"]


@dataclass(frozen=True)
class PlaceSuggestion:
    """A ranked place suggestion returned by autocomplete.

    Attributes:
        place_id: The provider's stable place identifier (fed to ``resolve``).
        main_text: The primary display name (e.g. "Denver").
        secondary_text: The secondary line (e.g. "CO, USA").
        full_text: The full formatted suggestion text.
    """

    place_id: str
    main_text: str
    secondary_text: str | None = None
    full_text: str | None = None


@dataclass(frozen=True)
class ResolvedPlace:
    """The canonical place resolved from a suggestion.

    Attributes:
        place_id: The provider's stable place identifier.
        display_name: Canonical display name.
        latitude: WGS 84 latitude.
        longitude: WGS 84 longitude.
        country: ISO country code/name when available.
        region: Administrative region (state/province) when available.
        formatted_address: Full formatted address when available.
    """

    place_id: str
    display_name: str
    latitude: float
    longitude: float
    country: str | None = None
    region: str | None = None
    formatted_address: str | None = None


#: HTTP transport: a callable ``(method, url, headers, body) -> (status, json)``.
#: Tests inject a fake; production uses :func:`_http_request`.
HttpTransport = Callable[[str, str, Mapping[str, str], str | None], tuple[int, Any]]


def _http_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: str | None,
) -> tuple[int, Any]:
    """Perform an HTTP request and return ``(status, json_or_error)``.

    Uses only the standard library so no runtime HTTP dependency is required.
    A non-2xx response is returned as ``(status, parsed_error)``; callers map
    it to a domain error.
    """
    data = body.encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(req, timeout=settings.GOOGLE_PLACES_TIMEOUT) as resp:
            payload = resp.read().decode("utf-8")
            return resp.status, json.loads(payload)
    except urllib.error.HTTPError as exc:
        try:
            err = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - non-JSON error body
            err = {"error": {"message": str(exc)}}
        return exc.code, err
    except urllib.error.URLError as exc:
        return 0, {"error": {"message": f"network error: {exc.reason}"}}


class PlaceAutocompleteError(Exception):
    """Base error for place-autocomplete provider failures."""


class PlaceAutocompleteProvider(ABC):
    """Application-level interface for place-autocomplete providers.

    Implementations call an external place service (Google Places API (New),
    Mapbox Geocoding, ...) server-side. The interface is provider-agnostic so
    the product is not permanently coupled to one vendor.
    """

    @abstractmethod
    def suggest(
        self,
        text: str,
        session_token: str | None = None,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
    ) -> list[PlaceSuggestion]:
        """Return ranked place suggestions for a partial query.

        Args:
            text: The user's partial input (e.g. "den").
            session_token: The search-session token (Google billing semantics).
            limit: Maximum number of suggestions.

        Returns:
            Ranked place suggestions.

        Raises:
            PlaceAutocompleteError: If the provider call fails.
        """

    @abstractmethod
    def resolve(
        self,
        place_id: str,
        session_token: str | None = None,
    ) -> ResolvedPlace:
        """Resolve a suggestion's canonical place (name + lat/lon + region).

        Args:
            place_id: The provider's place identifier.
            session_token: The search-session token (Google billing semantics).

        Returns:
            The canonical place.

        Raises:
            PlaceAutocompleteError: If the provider call fails or the place is
                unknown.
        """


class GooglePlacesAutocompleteProvider(PlaceAutocompleteProvider):
    """Google Places API (New) Autocomplete + Place Details provider.

    Calls ``POST https://places.googleapis.com/v1/places:autocomplete`` for
    suggestions and ``GET https://places.googleapis.com/v1/places/{placeId}``
    for canonical resolution, using a server-side API key that never reaches
    the browser. ``includedPrimaryTypes`` is set so results are actual places
    (not text queries billed at the higher Text Search rate).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        transport: HttpTransport = _http_request,
    ) -> None:
        """Create the provider.

        Args:
            api_key: Google Places API key (defaults to settings).
            api_base: Base URL (defaults to the official endpoints).
            transport: Injectable HTTP transport for tests.
        """
        self._api_key = api_key or settings.GOOGLE_PLACES_API_KEY
        self._autocomplete_url = (
            api_base or "https://places.googleapis.com/v1"
        ) + "/places:autocomplete"
        self._places_url = (api_base or "https://places.googleapis.com/v1") + "/places"
        self._transport = transport

    def suggest(
        self,
        text: str,
        session_token: str | None = None,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
    ) -> list[PlaceSuggestion]:
        body: dict[str, Any] = {
            "input": text,
            "includedPrimaryTypes": DEFAULT_INCLUDED_PRIMARY_TYPES,
        }
        if session_token is not None:
            body["sessionToken"] = session_token
        if settings.GOOGLE_PLACES_REGION is not None:
            body["regionCode"] = settings.GOOGLE_PLACES_REGION
        status, payload = self._transport(
            "POST",
            self._autocomplete_url,
            self._headers(),
            json.dumps(body),
        )
        if status != 200:
            raise PlaceAutocompleteError(
                f"Places autocomplete failed (HTTP {status}): "
                f"{_error_message(payload)}"
            )
        suggestions = payload.get("suggestions", [])
        results: list[PlaceSuggestion] = []
        for item in suggestions[:limit]:
            prediction = item.get("placePrediction")
            if prediction is None:
                # A ``queryPrediction`` (text-only) carries no placeId and is
                # billed as Text Search; skip it rather than expose it.
                continue
            text_ = prediction.get("text", {}).get("text", "")
            structured = prediction.get("structuredFormat", {})
            main_text = structured.get("mainText", {}).get("text") or text_
            secondary_text = structured.get("secondaryText", {}).get("text")
            results.append(
                PlaceSuggestion(
                    place_id=prediction["placeId"],
                    main_text=main_text,
                    secondary_text=secondary_text,
                    full_text=text_,
                )
            )
        return results

    def resolve(
        self,
        place_id: str,
        session_token: str | None = None,
    ) -> ResolvedPlace:
        url = f"{self._places_url}/{place_id}"
        fields = (
            "places.id,places.displayName,places.location,"
            "places.formattedAddress,places.addressComponents"
        )
        url += f"?fields={fields}"
        body = "{}"
        if session_token is not None:
            body = json.dumps({"sessionToken": session_token})
        status, payload = self._transport(
            "GET",
            url,
            self._headers(),
            body,
        )
        if status != 200:
            raise PlaceAutocompleteError(
                f"Places details failed (HTTP {status}): {_error_message(payload)}"
            )
        location = payload.get("location", {})
        components = payload.get("addressComponents", [])
        country = _component(components, "country")
        region = _component(components, "administrativeAreaLevel1")
        return ResolvedPlace(
            place_id=payload.get("id") or place_id,
            display_name=payload.get("displayName", {}).get("text", place_id),
            latitude=float(location.get("latitude", 0.0)),
            longitude=float(location.get("longitude", 0.0)),
            country=country,
            region=region,
            formatted_address=payload.get("formattedAddress"),
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
        }


class MapboxGeocodingProvider(PlaceAutocompleteProvider):
    """Mapbox Geocoding provider (drop-in alternative to Google).

    Uses the Mapbox Geocoding API's ``autocomplete=true`` forward geocoding.
    This is an alternative backend behind the same interface; it is not the
    default. The token lives server-side.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        transport: HttpTransport = _http_request,
    ) -> None:
        self._token = token or settings.MAPBOX_TOKEN
        self._transport = transport

    def suggest(
        self,
        text: str,
        session_token: str | None = None,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
    ) -> list[PlaceSuggestion]:
        import urllib.parse

        url = (
            "https://api.mapbox.com/geocoding/v5/mapbox.places/"
            f"{urllib.parse.quote(text)}.json"
            f"?access_token={self._token}&autocomplete=true&limit={limit}"
        )
        status, payload = self._transport("GET", url, {}, None)
        if status != 200:
            raise PlaceAutocompleteError(
                f"Mapbox geocoding failed (HTTP {status}): {_error_message(payload)}"
            )
        results: list[PlaceSuggestion] = []
        for feature in payload.get("features", [])[:limit]:
            place_id = feature.get("id", "")
            results.append(
                PlaceSuggestion(
                    place_id=place_id,
                    main_text=feature.get("text", ""),
                    secondary_text=feature.get("place_name"),
                    full_text=feature.get("place_name"),
                )
            )
        return results

    def resolve(
        self,
        place_id: str,
        session_token: str | None = None,
    ) -> ResolvedPlace:
        url = (
            "https://api.mapbox.com/geocoding/v5/mapbox.places/"
            f"{place_id}.json?access_token={self._token}"
        )
        status, payload = self._transport("GET", url, {}, None)
        if status != 200:
            raise PlaceAutocompleteError(
                f"Mapbox place resolution failed (HTTP {status}): {_error_message(payload)}"
            )
        feature = payload.get("features", [{}])[0]
        center = feature.get("center", [0.0, 0.0])
        context = feature.get("context", [])
        country = _mapbox_context(context, "country")
        region = _mapbox_context(context, "region")
        return ResolvedPlace(
            place_id=place_id,
            display_name=feature.get("place_name", place_id),
            latitude=float(center[1]) if len(center) > 1 else 0.0,
            longitude=float(center[0]) if center else 0.0,
            country=country,
            region=region,
            formatted_address=feature.get("place_name"),
        )


def _component(components: list[dict[str, Any]], wanted: str) -> str | None:
    """Return the long text of an address component by type, or ``None``."""
    for comp in components:
        if wanted in comp.get("types", []):
            return comp.get("longText") or comp.get("shortText")
    return None


def _mapbox_context(context: list[dict[str, Any]], wanted: str) -> str | None:
    for entry in context:
        if wanted in entry.get("id", ""):
            return entry.get("text")
    return None


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(payload)


def new_session_token() -> str:
    """Generate a fresh Places search-session token (UUIDv4).

    One token spans an entire autocomplete session: generated when the user
    focuses the search box, reused for every keystroke and the subsequent
    place resolution, then discarded. This keeps Google billing to one
    Autocomplete + one Place Details per completed selection.
    """
    return str(uuid.uuid4())


def get_provider() -> PlaceAutocompleteProvider:
    """Return the configured place-autocomplete provider.

    ``SEARCH_PROVIDER`` selects the backend (``google`` or ``mapbox``). The
    provider is constructed per request (it is cheap and stateless).
    """
    if settings.SEARCH_PROVIDER == "mapbox":
        return MapboxGeocodingProvider()
    return GooglePlacesAutocompleteProvider()
