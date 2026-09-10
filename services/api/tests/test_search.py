"""Contract and integration tests for the Milestone 9 /v1/search endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the response envelope, item shape, filters, and cache headers defined in
``docs/API.md`` section 6.1. When PostgreSQL is unreachable they skip,
following the existing ``test_catalog.py`` convention.

Place-autocomplete tests (``type=place``) are pure unit tests with a mocked
provider and never require live Google services.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_search_state():
    from api.services.circuit_breaker import circuit_breaker, search_cache

    circuit_breaker.reset()
    search_cache.clear()
    yield
    circuit_breaker.reset()
    search_cache.clear()


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
    monkeypatch.setattr(search_mod, "get_fallback_provider", lambda: None)
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


# --- Phase 1 Gateway, Station Fast Path, and Failover Tests ---


def test_station_fast_path_exact_match(client) -> None:
    """Exact station code match returns station as top-1 result."""
    resp = client.get("/v1/search?q=KASE")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) >= 1
    top = data[0]
    assert top["object"] == "station"
    assert top["name"] == "Aspen Station"
    assert top["place_id"] == "KASE"


def test_station_fast_path_case_insensitive(client) -> None:
    """Station fast-path is case-insensitive (e.g. kase -> KASE)."""
    resp = client.get("/v1/search?q=kase")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) >= 1
    assert data[0]["object"] == "station"
    assert data[0]["name"] == "Aspen Station"


def test_station_fast_path_non_station_word_continues_search(client) -> None:
    """A four-letter word that is not a station code continues normal search."""
    resp = client.get("/v1/search?q=aspen")
    assert resp.status_code == 200
    names = {item["name"] for item in resp.json()["data"]}
    assert "Aspen" in names


def test_gateway_geoapify_direct_coordinates(client, monkeypatch) -> None:
    """Geoapify primary provider returns direct WGS84 coordinates in SearchResultOut."""
    from api.services import search as search_mod
    from api.services.places import PlaceSuggestion

    class _GeoapifyStub:
        provider_name = "geoapify"

        def suggest(self, text, session_token=None, limit=8, bias=None):
            return [
                PlaceSuggestion(
                    place_id="geo_123",
                    main_text="Denver",
                    secondary_text="Colorado, United States",
                    full_text="Denver, CO, United States",
                    latitude=39.7392,
                    longitude=-104.9903,
                    country="United States",
                    region="Colorado",
                )
            ]

    monkeypatch.setattr(search_mod, "get_provider", lambda: _GeoapifyStub())
    resp = client.get("/v1/search?q=denver&type=place")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["object"] == "place"
    assert data[0]["name"] == "Denver"
    assert data[0]["latitude"] == pytest.approx(39.7392)
    assert data[0]["longitude"] == pytest.approx(-104.9903)
    assert data[0]["country"] == "United States"


def test_gateway_failover_to_locationiq(client, monkeypatch) -> None:
    """When primary provider fails with 429, gateway fails over to LocationIQ."""
    from api.services import search as search_mod
    from api.services.places import PlaceRateLimitError, PlaceSuggestion

    class _FailingPrimary:
        provider_name = "geoapify"

        def suggest(self, text, session_token=None, limit=8, bias=None):
            raise PlaceRateLimitError("Geoapify rate limit reached (HTTP 429)")

    class _WorkingFallback:
        provider_name = "locationiq"

        def suggest(self, text, session_token=None, limit=8, bias=None):
            return [
                PlaceSuggestion(
                    place_id="loc_456",
                    main_text="Denver",
                    secondary_text="Colorado, USA",
                    full_text="Denver, Colorado, USA",
                    latitude=39.7392,
                    longitude=-104.9903,
                    country="United States",
                    region="Colorado",
                )
            ]

    monkeypatch.setattr(search_mod, "get_provider", lambda: _FailingPrimary())
    monkeypatch.setattr(search_mod, "get_fallback_provider", lambda: _WorkingFallback())

    resp = client.get("/v1/search?q=denver&type=place")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["place_id"] == "loc_456"
    assert data[0]["latitude"] == pytest.approx(39.7392)


def test_gateway_both_providers_fail_preserves_local_results(client, monkeypatch) -> None:
    """When all external providers fail, search degrades to local DB without crashing."""
    from api.services import search as search_mod
    from api.services.places import PlaceAutocompleteError

    class _FailingPrimary:
        provider_name = "geoapify"

        def suggest(self, text, session_token=None, limit=8, bias=None):
            raise PlaceAutocompleteError("Primary down")

    class _FailingFallback:
        provider_name = "locationiq"

        def suggest(self, text, session_token=None, limit=8, bias=None):
            raise PlaceAutocompleteError("Fallback down")

    monkeypatch.setattr(search_mod, "get_provider", lambda: _FailingPrimary())
    monkeypatch.setattr(search_mod, "get_fallback_provider", lambda: _FailingFallback())

    # Searching "Aspen" has local city and resort rows in PostgreSQL
    resp = client.get("/v1/search?q=Aspen&type=all")
    assert resp.status_code == 200
    data = resp.json()["data"]
    names = {item["name"] for item in data}
    assert "Aspen" in names


def test_search_bias_parameter_forwarding(client, monkeypatch) -> None:
    """The /v1/search endpoint passes typed SearchBias parameter to the provider."""
    from api.schemas import SearchBias
    from api.services import search as search_mod
    from api.services.places import PlaceSuggestion

    captured_bias: list[SearchBias | None] = []

    class _Stub:
        provider_name = "geoapify"

        def suggest(self, text, session_token=None, limit=8, bias=None):
            captured_bias.append(bias)
            return [PlaceSuggestion("p1", "Test Place")]

    monkeypatch.setattr(search_mod, "get_provider", lambda: _Stub())
    resp = client.get("/v1/search?q=test&bias_lat=39.74&bias_lon=-104.99&type=place")
    assert resp.status_code == 200
    assert len(captured_bias) == 1
    assert captured_bias[0] == SearchBias(latitude=39.74, longitude=-104.99)


def test_search_bias_incomplete_pair_rejected(client) -> None:
    """Providing only bias_lat or only bias_lon is rejected with HTTP 422."""
    resp1 = client.get("/v1/search?q=test&bias_lat=39.74")
    assert resp1.status_code == 422
    assert "bias_lat and bias_lon must both be provided together" in resp1.json()["error"]["message"]

    resp2 = client.get("/v1/search?q=test&bias_lon=-104.99")
    assert resp2.status_code == 422
    assert "bias_lat and bias_lon must both be provided together" in resp2.json()["error"]["message"]


def test_search_bias_range_validation(client) -> None:
    """Out-of-range bias coordinates are rejected with HTTP 422."""
    resp1 = client.get("/v1/search?q=test&bias_lat=95.0&bias_lon=-104.99")
    assert resp1.status_code == 422

    resp2 = client.get("/v1/search?q=test&bias_lat=39.74&bias_lon=190.0")
    assert resp2.status_code == 422


def test_gateway_failover_on_5xx_and_malformed(client, monkeypatch) -> None:
    """When primary provider fails with 5xx or malformed data, gateway fails over to LocationIQ."""
    from api.services import search as search_mod
    from api.services.places import PlaceAutocompleteError, PlaceSuggestion

    class _5xxPrimary:
        provider_name = "geoapify"

        def suggest(self, text, session_token=None, limit=8, bias=None):
            raise PlaceAutocompleteError("Geoapify failed (HTTP 500): internal error")

    class _WorkingFallback:
        provider_name = "locationiq"

        def suggest(self, text, session_token=None, limit=8, bias=None):
            return [PlaceSuggestion("loc_789", "Fallback Result")]

    monkeypatch.setattr(search_mod, "get_provider", lambda: _5xxPrimary())
    monkeypatch.setattr(search_mod, "get_fallback_provider", lambda: _WorkingFallback())

    resp = client.get("/v1/search?q=denver&type=place")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["place_id"] == "loc_789"


def test_gateway_circuits_open_routes_to_local_only(client, monkeypatch) -> None:
    """When both primary and fallback circuits are open, search routes to local DB."""
    from api.services import search as search_mod
    from api.services.circuit_breaker import circuit_breaker

    circuit_breaker.record_failure("geoapify", is_429=True)
    circuit_breaker.record_failure("locationiq", is_429=True)
    assert circuit_breaker.is_available("geoapify") is False
    assert circuit_breaker.is_available("locationiq") is False

    called_providers: list[str] = []

    class _MockProv:
        def __init__(self, name):
            self.provider_name = name

        def suggest(self, text, **kw):
            called_providers.append(self.provider_name)
            return []

    monkeypatch.setattr(search_mod, "get_provider", lambda: _MockProv("geoapify"))
    monkeypatch.setattr(search_mod, "get_fallback_provider", lambda: _MockProv("locationiq"))

    resp = client.get("/v1/search?q=Aspen&type=all")
    assert resp.status_code == 200
    # Neither provider was called because both circuits were open
    assert called_providers == []
    # Local Aspen entities still returned
    names = {item["name"] for item in resp.json()["data"]}
    assert "Aspen" in names


def test_cache_hit_while_circuit_open(client, monkeypatch) -> None:
    """A valid cached search result is served even if the provider circuit is open."""
    from api.services.circuit_breaker import circuit_breaker, search_cache
    from api.schemas import SearchResultOut

    # Seed cache
    cached_items = [
        SearchResultOut(
            id="p_cached",
            object="place",
            name="Cached Place",
            latitude=40.0,
            longitude=-105.0,
        )
    ]
    search_cache.set("geoapify", "cachedq", cached_items, bias=None, ttl_seconds=60)

    # Trip breaker
    circuit_breaker.record_failure("geoapify", is_429=True)
    assert circuit_breaker.is_available("geoapify") is False

    # Should return cached result directly without error
    resp = client.get("/v1/search?q=cachedq&type=place")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["name"] == "Cached Place"


def test_valid_zero_results_does_not_trip_circuit(client, monkeypatch) -> None:
    """A valid zero search result response is not considered a provider failure."""
    from api.services import search as search_mod
    from api.services.circuit_breaker import circuit_breaker

    class _EmptyProvider:
        provider_name = "geoapify"

        def suggest(self, text, **kw):
            return []

    monkeypatch.setattr(search_mod, "get_provider", lambda: _EmptyProvider())
    resp = client.get("/v1/search?q=nonexistent_query_xyz&type=place")
    assert resp.status_code == 200
    assert resp.json()["data"] == []
    assert circuit_breaker.is_available("geoapify") is True


def test_circuit_breaker_and_cache_unit() -> None:
    """Unit tests for SearchCircuitBreaker and SearchCache."""
    from api.services.circuit_breaker import SearchCircuitBreaker, SearchCache, quantize_bias
    from api.schemas import SearchBias, SearchResultOut

    # Bias quantization
    assert quantize_bias(SearchBias(latitude=39.7392, longitude=-104.9903)) == "-105.0_39.7"
    assert quantize_bias(None) == "none"

    # Circuit breaker in-memory fallback
    cb = SearchCircuitBreaker(redis_url="redis://localhost:9999/0", failure_threshold=2, cooldown_seconds=10)
    assert cb.is_available("test_prov") is True

    # Record 1 failure (below threshold 2)
    cb.record_failure("test_prov", is_429=False)
    assert cb.is_available("test_prov") is True

    # Record 2nd failure (trips breaker)
    cb.record_failure("test_prov", is_429=False)
    assert cb.is_available("test_prov") is False

    # HTTP 429 trips immediately
    cb.record_success("test_prov_429")
    assert cb.is_available("test_prov_429") is True
    cb.record_failure("test_prov_429", is_429=True)
    assert cb.is_available("test_prov_429") is False

    # Success resets breaker
    cb.record_success("test_prov")
    assert cb.is_available("test_prov") is True

    # Cache in-memory
    cache = SearchCache(redis_url="redis://localhost:9999/0")
    items = [
        SearchResultOut(
            id="p1",
            object="place",
            name="Denver",
            latitude=39.7392,
            longitude=-104.9903,
        )
    ]
    bias = SearchBias(latitude=39.7, longitude=-105.0)
    cache.set("geoapify", "denver", items, bias=bias, ttl_seconds=10)
    hit = cache.get("geoapify", "denver", bias=bias)
    assert hit is not None
    assert len(hit) == 1
    assert hit[0].name == "Denver"

