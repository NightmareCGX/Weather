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
