"""Contract and integration tests for the Milestone 9 /v1/search endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the response envelope, item shape, filters, and cache headers defined in
``docs/API.md`` section 6.1. When PostgreSQL is unreachable they skip,
following the existing ``test_catalog.py`` convention.

Place-autocomplete tests (``type=place``) are pure unit tests with a mocked
provider and never require live Google services.
"""

import pytest


def test_search_contract_and_all_types(client):
    resp = client.get("/v1/search?q=Aspen")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    # "Aspen" matches the Aspen city, the Aspen Mountain resort, and the
    # Aspen Station (type=all merges all three source tables).
    names = {item["name"] for item in body["data"]}
    assert {"Aspen", "Aspen Mountain", "Aspen Station"} <= names
    objects = {item["object"] for item in body["data"]}
    assert {"city", "ski_resort", "station"} <= objects

    aspen_resort = next(item for item in body["data"] if item["name"] == "Aspen Mountain")
    assert aspen_resort["object"] == "ski_resort"
    assert aspen_resort["region"] == "Colorado"
    assert aspen_resort["country"] == "USA"
    assert aspen_resort["elevation_m"] == 3417.0
    assert abs(aspen_resort["latitude"] - 38.19) < 1e-6
    assert abs(aspen_resort["longitude"] - -106.82) < 1e-6


def test_search_type_filter(client):
    resp = client.get("/v1/search?q=Aspen&type=resort")
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["object"] == "ski_resort" for item in body["data"])
    names = {item["name"] for item in body["data"]}
    assert "Aspen Mountain" in names
    assert "Aspen" not in names  # city excluded by type=resort

    resp = client.get("/v1/search?q=Aspen&type=city")
    assert resp.status_code == 200
    assert all(item["object"] == "city" for item in resp.json()["data"])

    resp = client.get("/v1/search?q=Aspen&type=station")
    assert resp.status_code == 200
    assert all(item["object"] == "station" for item in resp.json()["data"])


def test_search_case_insensitive(client):
    resp = client.get("/v1/search?q=aspen")
    assert resp.status_code == 200
    names = {item["name"] for item in resp.json()["data"]}
    assert "Aspen" in names


def test_search_empty_result(client):
    resp = client.get("/v1/search?q=zzzznomatch")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == []
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_search_limit(client):
    resp = client.get("/v1/search?q=Aspen&limit=2")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2


def test_search_limit_is_global_across_types(client):
    # "Aspen" matches the Aspen city, the Aspen Mountain resort, and the
    # Aspen Station (one per table). A limit of 1 must return exactly the
    # top-1 match across ALL tables (not one per table). Sorted by name
    # ascending, "Aspen" < "Aspen Mountain" < "Aspen Station", so the city
    # is the single global result.
    resp = client.get("/v1/search?q=Aspen&limit=1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["name"] == "Aspen"
    assert data[0]["object"] == "city"


def test_search_cache_control_header(client):
    resp = client.get("/v1/search?q=Aspen")
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "public, max-age=86400"


def test_search_requires_q(client):
    resp = client.get("/v1/search")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["type"] == "validation_error"


def test_search_rejects_invalid_type(client):
    resp = client.get("/v1/search?q=Aspen&type=mountain")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["type"] == "validation_error"


# --- Place autocomplete (ACCEPTANCE_REMEDIATION_PLAN §13); provider mocked ---


def test_place_search_returns_suggestions(monkeypatch) -> None:
    """type=place delegates to the (mocked) provider and maps suggestions."""
    from api.services import search as search_mod
    from api.services.places import PlaceSuggestion

    class _Stub:
        def suggest(self, text, session_token=None, limit=8):
            return [
                PlaceSuggestion("ChIJden", "Denver", "CO, USA", "Denver, CO, USA"),
                PlaceSuggestion(
                    "ChIJdenairport", "Denver International Airport", "CO, USA"
                ),
            ]

    monkeypatch.setattr(search_mod, "get_provider", lambda: _Stub())
    results = search_mod.search_locations(None, "den", "place", 20)  # type: ignore[arg-type]
    assert len(results) == 2
    assert results[0].object == "place"
    assert results[0].name == "Denver"
    assert results[0].place_id == "ChIJden"
    assert results[1].name == "Denver International Airport"


def test_place_search_provider_error_degrades(monkeypatch) -> None:
    """A provider failure surfaces as a graceful error, never a crash."""
    from api.services import search as search_mod
    from api.services.places import PlaceAutocompleteError

    class _Stub:
        def suggest(self, text, session_token=None, limit=8):
            raise PlaceAutocompleteError("Places autocomplete failed (HTTP 500): boom")

    monkeypatch.setattr(search_mod, "get_provider", lambda: _Stub())
    with pytest.raises(PlaceAutocompleteError, match="boom"):
        search_mod.search_locations(None, "den", "place", 20)  # type: ignore[arg-type]


def test_place_resolve_updates_coordinates(monkeypatch) -> None:
    """Resolving a selected place yields canonical coordinates."""
    from api.services import search as search_mod
    from api.services.places import ResolvedPlace

    class _Stub:
        def resolve(self, place_id, session_token=None):
            assert place_id == "ChIJden"
            return ResolvedPlace(
                "ChIJden",
                "Denver",
                39.7392,
                -104.9903,
                country="United States",
                region="Colorado",
            )

    monkeypatch.setattr(search_mod, "get_provider", lambda: _Stub())
    result = search_mod.resolve_place("ChIJden", session_token="tok")
    assert result.latitude == pytest.approx(39.7392)
    assert result.longitude == pytest.approx(-104.9903)
    assert result.country == "United States"
    assert result.region == "Colorado"
