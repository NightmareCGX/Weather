"""Unit tests for the place-autocomplete provider (no live Google services).

The provider calls an external place service (Google Places API (New) by
default), so every test injects a fake HTTP transport. No test depends on live
Google credentials or network access (ENGINEERING_CONTRACT §8, docs/TESTING.md).
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from api.services.places import (
    GooglePlacesAutocompleteProvider,
    MapboxGeocodingProvider,
    PlaceAutocompleteError,
    PlaceSuggestion,
    ResolvedPlace,
    new_session_token,
)
from api.services.search import _suggestion_to_result, resolve_place


def _fake_transport(responses: list[tuple[int, Any]]):
    """Build a transport that returns the given responses in order."""

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: str | None,
    ) -> tuple[int, Any]:
        return responses.pop(0)

    return transport


def _google_provider(responses: list[tuple[int, Any]]) -> GooglePlacesAutocompleteProvider:
    return GooglePlacesAutocompleteProvider(
        api_key="test-key",
        transport=_fake_transport(responses),
    )


def test_suggest_returns_ranked_place_suggestions() -> None:
    """Partial input (e.g. 'den') yields ranked place suggestions via the mock."""
    provider = _google_provider(
        [
            (
                200,
                {
                    "suggestions": [
                        {
                            "placePrediction": {
                                "placeId": "ChIJden",
                                "text": {
                                    "text": "Denver, CO, USA",
                                    "matchedSubstrings": [{"length": 3, "offset": 0}],
                                },
                                "structuredFormat": {
                                    "mainText": {"text": "Denver"},
                                    "secondaryText": {"text": "CO, USA"},
                                },
                                "types": ["locality"],
                            }
                        },
                        {
                            "placePrediction": {
                                "placeId": "ChIJdenairport",
                                "text": {"text": "Denver International Airport, CO, USA"},
                                "structuredFormat": {
                                    "mainText": {"text": "Denver International Airport"},
                                    "secondaryText": {"text": "CO, USA"},
                                },
                                "types": ["airport"],
                            }
                        },
                    ]
                },
            )
        ]
    )
    suggestions = provider.suggest("den", session_token="tok")
    assert len(suggestions) == 2
    assert suggestions[0].main_text == "Denver"
    assert suggestions[0].place_id == "ChIJden"
    assert suggestions[1].main_text == "Denver International Airport"


def test_suggest_skips_query_prediction() -> None:
    """Text-only query predictions (no placeId) are skipped, not exposed."""
    provider = _google_provider(
        [
            (
                200,
                {
                    "suggestions": [
                        {
                            "queryPrediction": {
                                "text": {"text": "denver weather"},
                                "structuredFormat": {},
                            }
                        }
                    ]
                },
            )
        ]
    )
    suggestions = provider.suggest("den")
    assert suggestions == []


def test_resolve_returns_canonical_place() -> None:
    """Resolving a place_id returns canonical name + coordinates + region."""
    provider = _google_provider(
        [
            (
                200,
                {
                    "id": "ChIJden",
                    "displayName": {"text": "Denver", "languageCode": "en"},
                    "location": {"latitude": 39.7392, "longitude": -104.9903},
                    "formattedAddress": "Denver, CO, USA",
                    "addressComponents": [
                        {"longText": "United States", "shortText": "US", "types": ["country"]},
                        {
                            "longText": "Colorado",
                            "shortText": "CO",
                            "types": ["administrativeAreaLevel1"],
                        },
                    ],
                },
            )
        ]
    )
    place = provider.resolve("ChIJden", session_token="tok")
    assert place.display_name == "Denver"
    assert place.latitude == pytest.approx(39.7392)
    assert place.longitude == pytest.approx(-104.9903)
    assert place.country == "United States"
    assert place.region == "Colorado"


def test_provider_error_raises_domain_error() -> None:
    """A provider HTTP failure surfaces as PlaceAutocompleteError (graceful)."""
    provider = _google_provider([(500, {"error": {"message": "boom"}})])
    with pytest.raises(PlaceAutocompleteError, match="boom"):
        provider.suggest("den")


def test_suggestion_maps_to_search_result() -> None:
    """A suggestion maps to the shared SearchResultOut with a place_id."""
    suggestion = PlaceSuggestion(
        place_id="ChIJden",
        main_text="Denver",
        secondary_text="CO, USA",
        full_text="Denver, CO, USA",
    )
    result = _suggestion_to_result(suggestion)
    assert result.object == "place"
    assert result.name == "Denver"
    assert result.place_id == "ChIJden"
    # A suggestion carries no resolved coordinates yet.
    assert result.latitude == 0.0 and result.longitude == 0.0


def test_resolve_place_service_populates_coordinates() -> None:
    """resolve_place (via the provider abstraction) returns real coordinates."""
    import api.services.search as search_mod

    class _Stub:
        def resolve(self, place_id, session_token=None) -> ResolvedPlace:
            assert place_id == "ChIJden"
            return ResolvedPlace(
                place_id="ChIJden",
                display_name="Denver",
                latitude=39.7392,
                longitude=-104.9903,
                country="United States",
                region="Colorado",
            )

    original = search_mod.get_provider
    search_mod.get_provider = lambda: _Stub()  # type: ignore[assignment]
    try:
        result = resolve_place("ChIJden", session_token="tok")
    finally:
        search_mod.get_provider = original
    assert result.latitude == pytest.approx(39.7392)
    assert result.longitude == pytest.approx(-104.9903)
    assert result.country == "United States"
    assert result.region == "Colorado"


def test_new_session_token_is_unique() -> None:
    """Each search session gets a fresh, distinct UUIDv4 token."""
    assert new_session_token() != new_session_token()
    import uuid

    uuid.UUID(new_session_token())  # valid UUID


def test_mapbox_provider_alternative() -> None:
    """The Mapbox alternative implements the same interface."""
    def _mapbox_features():
        return {
            "features": [
                {
                    "id": "poi.1",
                    "text": "Denver",
                    "place_name": "Denver, Colorado, United States",
                    "center": [-104.9903, 39.7392],
                    "context": [{"id": "region.1", "text": "Colorado"}],
                }
            ]
        }

    provider = MapboxGeocodingProvider(
        token="test-token",
        transport=_fake_transport([(200, _mapbox_features()), (200, _mapbox_features())]),
    )
    suggestions = provider.suggest("den")
    assert suggestions[0].main_text == "Denver"
    assert suggestions[0].place_id == "poi.1"
    place = provider.resolve("poi.1")
    assert place.latitude == pytest.approx(39.7392)


# --- Geoapify Provider Tests (V1 Primary) ---


def test_geoapify_suggest_parses_geojson_with_direct_coordinates() -> None:
    from api.schemas import SearchBias
    from api.services.places import GeoapifyAutocompleteProvider

    captured_url: list[str] = []

    def transport(method: str, url: str, headers: Mapping[str, str], body: str | None, **kw) -> tuple[int, Any]:
        captured_url.append(url)
        return (
            200,
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "name": "Denver",
                            "city": "Denver",
                            "state": "Colorado",
                            "country": "United States",
                            "formatted": "Denver, CO, United States",
                            "address_line2": "Colorado, United States",
                            "place_id": "51a3denver",
                            "lon": -104.9903,
                            "lat": 39.7392,
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [-104.9903, 39.7392],
                        },
                    }
                ],
            },
        )

    provider = GeoapifyAutocompleteProvider(
        api_key="geo-key-123",
        transport=transport,
    )
    bias = SearchBias(latitude=39.74, longitude=-104.99)
    suggestions = provider.suggest("denver", bias=bias, limit=5)
    assert len(suggestions) == 1
    s = suggestions[0]
    assert s.main_text == "Denver"
    assert s.place_id == "51a3denver"
    assert s.latitude == pytest.approx(39.7392)
    assert s.longitude == pytest.approx(-104.9903)
    assert s.region == "Colorado"
    assert s.country == "United States"

    # Verify English language and bias parameter presence
    assert len(captured_url) == 1
    assert "lang=en" in captured_url[0]
    assert "apiKey=geo-key-123" in captured_url[0]
    assert "bias=proximity%3A-104.99%2C39.74" in captured_url[0] or "bias=proximity:-104.99,39.74" in captured_url[0]


def test_geoapify_malformed_response_raises() -> None:
    from api.services.places import GeoapifyAutocompleteProvider

    # Non-dict payload
    def transport_non_dict(method, url, headers, body, **kw):
        return 200, ["not", "a", "dict"]

    provider = GeoapifyAutocompleteProvider(api_key="k", transport=transport_non_dict)
    with pytest.raises(PlaceAutocompleteError, match="expected dict"):
        provider.suggest("denver")

    # Missing features list
    def transport_missing_features(method, url, headers, body, **kw):
        return 200, {"features": "not_a_list"}

    provider = GeoapifyAutocompleteProvider(api_key="k", transport=transport_missing_features)
    with pytest.raises(PlaceAutocompleteError, match="missing features list"):
        provider.suggest("denver")


def test_geoapify_missing_key_raises() -> None:
    from api.services.places import GeoapifyAutocompleteProvider

    provider = GeoapifyAutocompleteProvider(api_key="")
    with pytest.raises(PlaceAutocompleteError, match="Geoapify API key not configured"):
        provider.suggest("denver")


def test_geoapify_rate_limit_raises_429() -> None:
    from api.services.places import GeoapifyAutocompleteProvider, PlaceRateLimitError

    def transport(method: str, url: str, headers: Mapping[str, str], body: str | None, **kw) -> tuple[int, Any]:
        return 429, {"message": "Daily limit reached"}

    provider = GeoapifyAutocompleteProvider(api_key="k", transport=transport)
    with pytest.raises(PlaceRateLimitError, match="rate limit"):
        provider.suggest("denver")


def test_geoapify_timeout_raises_timeout_error() -> None:
    from api.services.places import GeoapifyAutocompleteProvider, PlaceTimeoutError

    def transport(method: str, url: str, headers: Mapping[str, str], body: str | None, **kw) -> tuple[int, Any]:
        return 0, {"error": {"message": "timed out"}}

    provider = GeoapifyAutocompleteProvider(api_key="k", transport=transport)
    with pytest.raises(PlaceTimeoutError, match="timed out"):
        provider.suggest("denver")


# --- LocationIQ Provider Tests (V1 Fallback) ---


def test_locationiq_suggest_parses_json_with_direct_coordinates() -> None:
    from api.schemas import SearchBias
    from api.services.places import LocationIQAutocompleteProvider

    captured_url: list[str] = []

    def transport(method: str, url: str, headers: Mapping[str, str], body: str | None, **kw) -> tuple[int, Any]:
        captured_url.append(url)
        return (
            200,
            [
                {
                    "place_id": "12345",
                    "lat": "39.7392",
                    "lon": "-104.9903",
                    "display_name": "Denver, Colorado, United States",
                    "display_place": "Denver",
                    "display_address": "Colorado, United States",
                    "address": {
                        "city": "Denver",
                        "state": "Colorado",
                        "country": "United States",
                    },
                }
            ],
        )

    provider = LocationIQAutocompleteProvider(
        api_key="loc-key-456",
        transport=transport,
    )
    bias = SearchBias(latitude=39.74, longitude=-104.99)
    suggestions = provider.suggest("denver", bias=bias, limit=5)
    assert len(suggestions) == 1
    s = suggestions[0]
    assert s.main_text == "Denver"
    assert s.place_id == "12345"
    assert s.latitude == pytest.approx(39.7392)
    assert s.longitude == pytest.approx(-104.9903)
    assert s.region == "Colorado"
    assert s.country == "United States"

    # Verify English parameter and viewbox presence
    assert len(captured_url) == 1
    assert "accept-language=en" in captured_url[0]
    assert "key=loc-key-456" in captured_url[0]
    assert "viewbox=" in captured_url[0]


def test_locationiq_malformed_response_raises() -> None:
    from api.services.places import LocationIQAutocompleteProvider

    # Non-list payload (e.g. dict or string)
    def transport_non_list(method, url, headers, body, **kw):
        return 200, {"not": "a list"}

    provider = LocationIQAutocompleteProvider(api_key="k", transport=transport_non_list)
    with pytest.raises(PlaceAutocompleteError, match="expected list"):
        provider.suggest("denver")


def test_locationiq_missing_key_raises() -> None:
    from api.services.places import LocationIQAutocompleteProvider

    provider = LocationIQAutocompleteProvider(api_key="")
    with pytest.raises(PlaceAutocompleteError, match="LocationIQ API key not configured"):
        provider.suggest("denver")


def test_locationiq_rate_limit_raises_429() -> None:
    from api.services.places import LocationIQAutocompleteProvider, PlaceRateLimitError

    def transport(method: str, url: str, headers: Mapping[str, str], body: str | None, **kw) -> tuple[int, Any]:
        return 429, {"error": "Rate limit exceeded"}

    provider = LocationIQAutocompleteProvider(api_key="k", transport=transport)
    with pytest.raises(PlaceRateLimitError, match="rate limit"):
        provider.suggest("denver")


def test_locationiq_timeout_raises_timeout_error() -> None:
    from api.services.places import LocationIQAutocompleteProvider, PlaceTimeoutError

    def transport(method: str, url: str, headers: Mapping[str, str], body: str | None, **kw) -> tuple[int, Any]:
        return 0, {"error": {"message": "network error: timed out"}}

    provider = LocationIQAutocompleteProvider(api_key="k", transport=transport)
    with pytest.raises(PlaceTimeoutError, match="timed out"):
        provider.suggest("denver")

