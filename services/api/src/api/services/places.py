"""Location place-autocomplete provider for the serving tier.

The frontend combobox already provides debounce/abort/stale-guard/keyboard
navigation; what it lacked is a real place-autocomplete data source. This
module is the backend provider abstraction that backs ``/v1/search?type=place``:

* :class:`PlaceAutocompleteProvider` is the application-level interface;
* :class:`GeoapifyAutocompleteProvider` calls Geoapify Address Autocomplete
  server-side (V1 Primary) and extracts native WGS84 coordinates directly;
* :class:`LocationIQAutocompleteProvider` calls LocationIQ Autocomplete
  server-side (V1 Fallback);
* :class:`GooglePlacesAutocompleteProvider` calls Google Places API (New)
  (retained for reference / backwards compatibility);
* :class:`MapboxGeocodingProvider` calls Mapbox Geocoding
  (retained for reference / backwards compatibility).

The provider is network-free by construction for tests: the HTTP transport is a
small injectable callable (``httpx``-style), so tests supply a fake transport and
never touch live external services.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from api.core.config import settings
from api.schemas import SearchBias

#: Default max suggestions returned by autocomplete.
DEFAULT_SUGGESTION_LIMIT = 8
#: Place types included in Google autocomplete so results are actual places.
DEFAULT_INCLUDED_PRIMARY_TYPES = ["locality", "address", "airport", "establishment"]


@dataclass(frozen=True)
class PlaceSuggestion:
    """A ranked place suggestion returned by autocomplete.

    Attributes:
        place_id: The provider's stable place identifier (fed to ``resolve``).
        main_text: The primary display name (e.g. "Denver").
        secondary_text: The secondary line (e.g. "CO, USA").
        full_text: The full formatted suggestion text.
        latitude: WGS 84 latitude (when returned directly by provider).
        longitude: WGS 84 longitude (when returned directly by provider).
        country: ISO country code/name when available.
        region: Administrative region (state/province) when available.
        elevation_m: Elevation in meters when available.
    """

    place_id: str
    main_text: str
    secondary_text: str | None = None
    full_text: str | None = None
    latitude: float = 0.0
    longitude: float = 0.0
    country: str | None = None
    region: str | None = None
    elevation_m: float | None = None


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
HttpTransport = Callable[..., tuple[int, Any]]


def _http_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: str | None,
    timeout: float | None = None,
) -> tuple[int, Any]:
    """Perform an HTTP request and return ``(status, json_or_error)``.

    Uses only the standard library so no runtime HTTP dependency is required.
    A non-2xx response is returned as ``(status, parsed_error)``; callers map
    it to a domain error.
    """
    timeout_s = timeout if timeout is not None else settings.GOOGLE_PLACES_TIMEOUT
    data = body.encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
            return resp.status, json.loads(payload)
    except urllib.error.HTTPError as exc:
        try:
            err = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - non-JSON error body
            err = {"error": {"message": str(exc)}}
        return exc.code, err
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, {"error": {"message": f"network error: {exc}"}}


class PlaceAutocompleteError(Exception):
    """Base error for place-autocomplete provider failures."""


class PlaceRateLimitError(PlaceAutocompleteError):
    """Raised when an external place provider returns HTTP 429 (rate limit)."""


class PlaceTimeoutError(PlaceAutocompleteError):
    """Raised when an external place provider request times out."""


class PlaceAutocompleteProvider(ABC):
    """Application-level interface for place-autocomplete providers.

    Implementations call an external place service (Geoapify, LocationIQ,
    Google, ...) server-side. The interface is provider-agnostic so the product
    is not permanently coupled to one vendor.
    """

    @abstractmethod
    def suggest(
        self,
        text: str,
        session_token: str | None = None,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
        bias: SearchBias | None = None,
    ) -> list[PlaceSuggestion]:
        """Return ranked place suggestions for a partial query.

        Args:
            text: The user's partial input (e.g. "den").
            session_token: Optional search-session token.
            limit: Maximum number of suggestions.
            bias: Optional soft proximity bias.

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
            session_token: Optional search-session token.

        Returns:
            The canonical place.

        Raises:
            PlaceAutocompleteError: If the provider call fails or the place is
                unknown.
        """


class GeoapifyAutocompleteProvider(PlaceAutocompleteProvider):
    """Geoapify Address Autocomplete provider (V1 Primary).

    Calls ``GET https://api.geoapify.com/v1/geocode/autocomplete`` with
    ``lang=en`` and extracts WGS84 GeoJSON Point coordinates directly.
    """

    provider_name: str = "geoapify"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_ms: int | None = None,
        transport: HttpTransport = _http_request,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.GEOAPIFY_API_KEY
        self._api_base = (
            api_base or settings.GEOAPIFY_API_BASE or "https://api.geoapify.com/v1"
        ).rstrip("/")
        timeout = timeout_ms if timeout_ms is not None else settings.SEARCH_PRIMARY_TIMEOUT_MS
        self._timeout_s = timeout / 1000.0
        self._transport = transport

    def _call_transport(
        self, method: str, url: str, headers: Mapping[str, str], body: str | None
    ) -> tuple[int, Any]:
        try:
            return self._transport(method, url, headers, body, timeout=self._timeout_s)
        except TypeError:
            return self._transport(method, url, headers, body)

    def suggest(
        self,
        text: str,
        session_token: str | None = None,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
        bias: SearchBias | None = None,
    ) -> list[PlaceSuggestion]:
        if not self._api_key:
            raise PlaceAutocompleteError("Geoapify API key not configured")

        url = (
            f"{self._api_base}/geocode/autocomplete"
            f"?text={urllib.parse.quote(text)}"
            f"&lang=en"
            f"&limit={limit}"
            f"&apiKey={self._api_key}"
        )
        if bias is not None:
            url += f"&bias=proximity:{bias.longitude},{bias.latitude}"

        status, payload = self._call_transport(
            "GET", url, {"Accept": "application/json"}, None
        )
        if status == 429:
            raise PlaceRateLimitError("Geoapify rate limit reached (HTTP 429)")
        if status == 0:
            raise PlaceTimeoutError(
                f"Geoapify request timed out or network error: {_error_message(payload)}"
            )
        if status != 200:
            raise PlaceAutocompleteError(
                f"Geoapify autocomplete failed (HTTP {status}): {_error_message(payload)}"
            )

        if not isinstance(payload, dict):
            raise PlaceAutocompleteError(
                f"Malformed Geoapify response: expected dict, got {type(payload).__name__}"
            )
        features = payload.get("features")
        if not isinstance(features, list):
            raise PlaceAutocompleteError(
                "Malformed Geoapify response: missing features list"
            )

        results: list[PlaceSuggestion] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            try:
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                coords = geom.get("coordinates") if isinstance(geom, dict) else None
                lon = (
                    float(coords[0])
                    if coords and len(coords) >= 2
                    else float(props.get("lon", 0.0))
                )
                lat = (
                    float(coords[1])
                    if coords and len(coords) >= 2
                    else float(props.get("lat", 0.0))
                )

                place_id = str(props.get("place_id") or f"{lon:.4f},{lat:.4f}")
                name = (
                    props.get("name")
                    or props.get("city")
                    or props.get("address_line1")
                    or props.get("formatted")
                    or text
                )
                region = props.get("state") or props.get("county") or props.get("region")
                country = props.get("country")
                full_text = (
                    props.get("formatted")
                    or f"{name}, {region or ''}, {country or ''}".strip(", ")
                )
                secondary = (
                    props.get("address_line2")
                    or ", ".join(filter(None, [region, country]))
                    or None
                )

                results.append(
                    PlaceSuggestion(
                        place_id=place_id,
                        main_text=name,
                        secondary_text=secondary,
                        full_text=full_text,
                        latitude=lat,
                        longitude=lon,
                        country=country,
                        region=region,
                    )
                )
            except (ValueError, TypeError, KeyError) as err:
                raise PlaceAutocompleteError(f"Malformed Geoapify feature: {err}") from err
        return results

    def resolve(
        self,
        place_id: str,
        session_token: str | None = None,
    ) -> ResolvedPlace:
        if "," in place_id:
            try:
                parts = place_id.split(",")
                lon = float(parts[0])
                lat = float(parts[1])
                return ResolvedPlace(
                    place_id=place_id,
                    display_name=place_id,
                    latitude=lat,
                    longitude=lon,
                )
            except ValueError:
                pass
        return ResolvedPlace(
            place_id=place_id,
            display_name=place_id,
            latitude=0.0,
            longitude=0.0,
        )


class LocationIQAutocompleteProvider(PlaceAutocompleteProvider):
    """LocationIQ Autocomplete provider (V1 Secondary Failover).

    Calls ``GET https://api.locationiq.com/v1/autocomplete`` with
    ``accept-language=en`` and extracts WGS84 coordinates directly.
    """

    provider_name: str = "locationiq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_ms: int | None = None,
        transport: HttpTransport = _http_request,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else settings.LOCATIONIQ_API_KEY
        )
        self._api_base = (
            api_base or settings.LOCATIONIQ_API_BASE or "https://api.locationiq.com/v1"
        ).rstrip("/")
        timeout = (
            timeout_ms if timeout_ms is not None else settings.SEARCH_FALLBACK_TIMEOUT_MS
        )
        self._timeout_s = timeout / 1000.0
        self._transport = transport

    def _call_transport(
        self, method: str, url: str, headers: Mapping[str, str], body: str | None
    ) -> tuple[int, Any]:
        try:
            return self._transport(method, url, headers, body, timeout=self._timeout_s)
        except TypeError:
            return self._transport(method, url, headers, body)

    def suggest(
        self,
        text: str,
        session_token: str | None = None,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
        bias: SearchBias | None = None,
    ) -> list[PlaceSuggestion]:
        if not self._api_key:
            raise PlaceAutocompleteError("LocationIQ API key not configured")

        url = (
            f"{self._api_base}/autocomplete"
            f"?q={urllib.parse.quote(text)}"
            f"&key={self._api_key}"
            f"&limit={limit}"
            f"&format=json"
            f"&accept-language=en"
        )
        if bias is not None:
            lon = bias.longitude
            lat = bias.latitude
            viewbox = f"{lon - 2.0:.4f},{lat + 2.0:.4f},{lon + 2.0:.4f},{lat - 2.0:.4f}"
            url += f"&viewbox={viewbox}&bounded=0"

        status, payload = self._call_transport(
            "GET", url, {"Accept": "application/json"}, None
        )
        if status == 429:
            raise PlaceRateLimitError("LocationIQ rate limit reached (HTTP 429)")
        if status == 0:
            raise PlaceTimeoutError(
                f"LocationIQ request timed out or network error: {_error_message(payload)}"
            )
        if status != 200:
            raise PlaceAutocompleteError(
                f"LocationIQ autocomplete failed (HTTP {status}): {_error_message(payload)}"
            )

        if not isinstance(payload, list):
            raise PlaceAutocompleteError(
                f"Malformed LocationIQ response: expected list, got {type(payload).__name__}"
            )

        results: list[PlaceSuggestion] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                place_id = str(item.get("place_id") or item.get("osm_id") or "")
                lat = float(item.get("lat", 0.0))
                lon = float(item.get("lon", 0.0))
                address = (
                    item.get("address", {})
                    if isinstance(item.get("address"), dict)
                    else {}
                )
                display_name = str(item.get("display_name") or "")

                main_text = (
                    item.get("display_place")
                    or address.get("city")
                    or address.get("town")
                    or address.get("village")
                    or (display_name.split(",")[0].strip() if display_name else text)
                )
                region = address.get("state") or address.get("county")
                country = address.get("country")
                secondary = (
                    item.get("display_address")
                    or ", ".join(filter(None, [region, country]))
                    or None
                )

                results.append(
                    PlaceSuggestion(
                        place_id=place_id,
                        main_text=main_text,
                        secondary_text=secondary,
                        full_text=display_name,
                        latitude=lat,
                        longitude=lon,
                        country=country,
                        region=region,
                    )
                )
            except (ValueError, TypeError, KeyError) as err:
                raise PlaceAutocompleteError(f"Malformed LocationIQ item: {err}") from err
        return results

    def resolve(
        self,
        place_id: str,
        session_token: str | None = None,
    ) -> ResolvedPlace:
        return ResolvedPlace(
            place_id=place_id,
            display_name=place_id,
            latitude=0.0,
            longitude=0.0,
        )


class GooglePlacesAutocompleteProvider(PlaceAutocompleteProvider):
    """Google Places API (New) Autocomplete + Place Details provider.

    Calls ``POST https://places.googleapis.com/v1/places:autocomplete`` for
    suggestions and ``GET https://places.googleapis.com/v1/places/{placeId}``
    for canonical resolution, using a server-side API key that never reaches
    the browser.
    """

    provider_name: str = "google"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        transport: HttpTransport = _http_request,
    ) -> None:
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
        bias: SearchBias | None = None,
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
    """Mapbox Geocoding provider (drop-in alternative).

    Uses the Mapbox Geocoding API's ``autocomplete=true`` forward geocoding.
    """

    provider_name: str = "mapbox"

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
        bias: SearchBias | None = None,
    ) -> list[PlaceSuggestion]:
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
            center = feature.get("center", [0.0, 0.0])
            results.append(
                PlaceSuggestion(
                    place_id=place_id,
                    main_text=feature.get("text", ""),
                    secondary_text=feature.get("place_name"),
                    full_text=feature.get("place_name"),
                    latitude=float(center[1]) if len(center) > 1 else 0.0,
                    longitude=float(center[0]) if center else 0.0,
                )
            )
        return results

    def resolve(
        self,
        place_id: str,
        session_token: str | None = None,
    ) -> ResolvedPlace:
        url = (
            f"https://api.mapbox.com/geocoding/v5/mapbox.places/"
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
    """Generate a fresh search-session token (UUIDv4)."""
    return str(uuid.uuid4())


def get_provider() -> PlaceAutocompleteProvider:
    """Return the configured primary place-autocomplete provider.

    Defaults to ``GeoapifyAutocompleteProvider`` (V1 Primary).
    """
    if settings.SEARCH_PROVIDER == "locationiq":
        return LocationIQAutocompleteProvider()
    if settings.SEARCH_PROVIDER == "google":
        return GooglePlacesAutocompleteProvider()
    if settings.SEARCH_PROVIDER == "mapbox":
        return MapboxGeocodingProvider()
    return GeoapifyAutocompleteProvider()


def get_fallback_provider() -> PlaceAutocompleteProvider | None:
    """Return the configured secondary fallback provider, if available."""
    if settings.SEARCH_PROVIDER == "geoapify":
        # When Geoapify is primary, LocationIQ is fallback
        return LocationIQAutocompleteProvider()
    if settings.SEARCH_PROVIDER == "locationiq":
        # When LocationIQ is primary, Geoapify is fallback
        return GeoapifyAutocompleteProvider()
    return None
