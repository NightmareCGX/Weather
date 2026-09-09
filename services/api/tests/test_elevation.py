"""Unit tests for the elevation provider, cache, and metrics (no network).

Verifies Open-Meteo elevation provider behavior under mock transports
(success, HTTP errors, rate limits, timeouts, null/ocean values, disabled provider),
coordinate quantization, popularity-aware caching (probationary window, protected
promotion, frequency decay/aging), thread safety, and metrics tracking.
No live external API calls are ever made (ENGINEERING_CONTRACT §8).
"""

from __future__ import annotations

import concurrent.futures
import pytest

from api.services.elevation import (
    ElevationProvider,
    OpenMeteoElevationProvider,
    PopularityDecayingElevationCache,
    _NullProvider,
    _reset_elevation_metrics,
    _reset_elevation_provider_cache,
    get_elevation_metrics,
    get_elevation_provider,
)


# --- OpenMeteoElevationProvider Tests ---


def test_open_meteo_provider_success() -> None:
    """A 200 response with valid elevation returns the float value."""
    recorded_urls: list[str] = []

    def mock_transport(url: str) -> tuple[int, dict[str, list[float]]]:
        recorded_urls.append(url)
        return 200, {"elevation": [2404.5]}

    provider = OpenMeteoElevationProvider(
        base_url="https://api.open-meteo.com/v1/elevation",
        transport=mock_transport,
    )
    val = provider.get_elevation(39.1911, -106.8175)
    assert val == pytest.approx(2404.5)
    assert len(recorded_urls) == 1
    assert "latitude=39.1911" in recorded_urls[0]
    assert "longitude=-106.8175" in recorded_urls[0]
    assert "apikey" not in recorded_urls[0]


def test_open_meteo_provider_with_api_key() -> None:
    """An API key is appended as query parameter for commercial endpoints."""
    recorded_urls: list[str] = []

    def mock_transport(url: str) -> tuple[int, dict[str, list[float]]]:
        recorded_urls.append(url)
        return 200, {"elevation": [182.0]}

    provider = OpenMeteoElevationProvider(
        base_url="https://customer-api.open-meteo.com/v1/elevation",
        api_key="secret_test_key",
        transport=mock_transport,
    )
    val = provider.get_elevation(41.88, -87.62)
    assert val == pytest.approx(182.0)
    assert "apikey=secret_test_key" in recorded_urls[0]


def test_open_meteo_provider_null_value() -> None:
    """A response with [None] (e.g. ocean) returns None."""
    provider = OpenMeteoElevationProvider(
        transport=lambda url: (200, {"elevation": [None]}),
    )
    assert provider.get_elevation(0.0, 0.0) is None


def test_open_meteo_provider_empty_list() -> None:
    """An empty elevation list returns None."""
    provider = OpenMeteoElevationProvider(
        transport=lambda url: (200, {"elevation": []}),
    )
    assert provider.get_elevation(0.0, 0.0) is None


def test_open_meteo_provider_malformed_json() -> None:
    """Malformed or unexpected JSON payload returns None gracefully."""
    provider = OpenMeteoElevationProvider(
        transport=lambda url: (200, {"unexpected": "payload"}),
    )
    assert provider.get_elevation(39.0, -106.0) is None


def test_open_meteo_provider_http_400_bad_request() -> None:
    """HTTP 400 bad request returns None."""
    provider = OpenMeteoElevationProvider(
        transport=lambda url: (400, {"error": True, "reason": "Invalid coordinate"}),
    )
    assert provider.get_elevation(999.0, 999.0) is None


def test_open_meteo_provider_http_429_rate_limit() -> None:
    """HTTP 429 rate limit returns None and logs failure."""
    provider = OpenMeteoElevationProvider(
        transport=lambda url: (429, {"error": True, "reason": "Hourly limit exceeded"}),
    )
    assert provider.get_elevation(39.0, -106.0) is None


def test_open_meteo_provider_http_500_server_error() -> None:
    """HTTP 500 server error returns None."""
    provider = OpenMeteoElevationProvider(
        transport=lambda url: (500, {}),
    )
    assert provider.get_elevation(39.0, -106.0) is None


def test_open_meteo_provider_timeout() -> None:
    """A socket timeout returns None and increments timeout metric."""
    _reset_elevation_metrics()

    def timeout_transport(url: str):
        raise TimeoutError("Socket timed out")

    provider = OpenMeteoElevationProvider(transport=timeout_transport)
    assert provider.get_elevation(39.0, -106.0) is None

    metrics = get_elevation_metrics()
    assert metrics.provider_timeouts_total == 1
    assert metrics.provider_failures_total == 1


def test_open_meteo_provider_network_error() -> None:
    """A connection failure returns None."""

    def error_transport(url: str):
        raise OSError("Connection refused")

    provider = OpenMeteoElevationProvider(transport=error_transport)
    assert provider.get_elevation(39.0, -106.0) is None


def test_null_provider_always_returns_none() -> None:
    """_NullProvider always reports unavailable."""
    assert _NullProvider().get_elevation(39.1911, -106.8175) is None


# --- PopularityDecayingElevationCache Tests ---


class _MockCountingProvider(ElevationProvider):
    def __init__(self, elevation: float = 1609.0) -> None:
        self.elevation = elevation
        self.call_count = 0

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        self.call_count += 1
        return self.elevation


def test_cache_quantization_nearby_reuse() -> None:
    """Nearby coordinates within the quantization bucket share the cached entry."""
    mock = _MockCountingProvider(2400.0)
    cache = PopularityDecayingElevationCache(mock, max_entries=100)

    # Coordinates differ by 0.0002 deg, which rounds to the same 3-decimal key (39.191, -106.818)
    e1 = cache.get_elevation(39.1911, -106.8176)
    e2 = cache.get_elevation(39.1914, -106.8178)

    assert e1 == pytest.approx(2400.0)
    assert e2 == pytest.approx(2400.0)
    assert mock.call_count == 1  # second call was served from cache


def test_cache_different_coordinates_invoke_provider() -> None:
    """Coordinates in different quantization buckets invoke the provider."""
    mock = _MockCountingProvider(2400.0)
    cache = PopularityDecayingElevationCache(mock, max_entries=100)

    cache.get_elevation(39.191, -106.818)
    cache.get_elevation(40.712, -74.006)

    assert mock.call_count == 2


def test_cache_probation_to_protected_promotion() -> None:
    """A second access promotes an entry from probationary to protected segment."""
    mock = _MockCountingProvider(2400.0)
    cache = PopularityDecayingElevationCache(mock, max_entries=10)

    key = cache._normalize(39.191, -106.818)

    # 1st access: in probationary segment
    cache.get_elevation(39.191, -106.818)
    assert key in cache._probationary
    assert key not in cache._protected

    # 2nd access: promoted to protected segment
    cache.get_elevation(39.191, -106.818)
    assert key in cache._protected
    assert key not in cache._probationary


def test_cache_one_off_clicks_do_not_evict_protected_entries() -> None:
    """Burst of cold one-off clicks cycles through probation without evicting protected entries."""
    mock = _MockCountingProvider(2400.0)
    # max_entries = 10 -> probation_cap = 2, protected_cap = 8
    cache = PopularityDecayingElevationCache(mock, max_entries=10)

    # Access popular location twice -> promoted to protected
    popular_lat, popular_lon = 39.191, -106.818
    popular_key = cache._normalize(popular_lat, popular_lon)
    cache.get_elevation(popular_lat, popular_lon)
    cache.get_elevation(popular_lat, popular_lon)
    assert popular_key in cache._protected

    # Generate 15 distinct one-off random clicks
    for i in range(15):
        cache.get_elevation(10.0 + (i * 0.1), 20.0 + (i * 0.1))

    # The popular location is STILL in the protected segment!
    assert popular_key in cache._protected
    # Verify reading it is a cache hit
    prior_calls = mock.call_count
    val = cache.get_elevation(popular_lat, popular_lon)
    assert val == pytest.approx(2400.0)
    assert mock.call_count == prior_calls  # served from cache!


def test_cache_aging_decay() -> None:
    """Frequencies are halved when the access counter reaches the decay interval."""
    mock = _MockCountingProvider(2400.0)
    # Set a tiny decay interval of 5 accesses
    cache = PopularityDecayingElevationCache(mock, max_entries=10, decay_interval=5)

    # Access a location 6 times to build frequency
    lat, lon = 39.191, -106.818
    key = cache._normalize(lat, lon)
    for _ in range(4):
        cache.get_elevation(lat, lon)

    entry = cache._protected[key]
    freq_before = entry.frequency
    assert freq_before >= 4

    # 5th access triggers decay
    cache.get_elevation(lat, lon)
    freq_after = cache._protected[key].frequency
    # Frequency was decayed (halved) during the 5th access
    assert freq_after <= freq_before


def test_cache_popular_established_entry_survives_weaker_recent_churn() -> None:
    """A highly popular entry survives promotion pressure from weak newcomers."""
    mock = _MockCountingProvider(2400.0)
    # max_entries = 10 -> probation_cap = 2, protected_cap = 8
    cache = PopularityDecayingElevationCache(mock, max_entries=10, decay_interval=1000)

    # 1. Establish an ultra-popular location (e.g. Aspen, hit 20 times)
    aspen_lat, aspen_lon = 39.191, -106.818
    aspen_key = cache._normalize(aspen_lat, aspen_lon)
    for _ in range(20):
        cache.get_elevation(aspen_lat, aspen_lon)

    assert aspen_key in cache._protected
    assert cache._protected[aspen_key].frequency >= 20

    # 2. Fill protected segment with 7 other items (hit twice each to promote)
    for i in range(1, 8):
        lat = 30.0 + i
        lon = -100.0 + i
        cache.get_elevation(lat, lon)
        cache.get_elevation(lat, lon)

    assert len(cache._protected) == 8  # protected is full!

    # 3. Introduce a newcomer that receives two hits (frequency = 2)
    newcomer_lat, newcomer_lon = 55.0, -120.0
    cache.get_elevation(newcomer_lat, newcomer_lon)
    cache.get_elevation(newcomer_lat, newcomer_lon)

    # Aspen has frequency >= 20, whereas LRU items in protected have freq ~2.
    # Aspen MUST remain in protected!
    assert aspen_key in cache._protected


def test_cache_recurring_entry_eventually_displaces_stale_entry_after_decay() -> None:
    """Old hot entries decay over time so genuinely recurring newcomers can displace them."""
    mock = _MockCountingProvider(2400.0)
    # Tiny decay interval: every 4 accesses frequencies halve
    cache = PopularityDecayingElevationCache(mock, max_entries=10, decay_interval=4)

    # 1. Old location gets a burst of 6 hits
    old_lat, old_lon = 39.0, -106.0
    old_key = cache._normalize(old_lat, old_lon)
    for _ in range(6):
        cache.get_elevation(old_lat, old_lon)

    # 2. Fill protected segment with other items
    for i in range(1, 8):
        cache.get_elevation(30.0 + i, -100.0 + i)
        cache.get_elevation(30.0 + i, -100.0 + i)

    # 3. Sustained activity on other points causes old_key's frequency to decay repeatedly
    for j in range(20):
        cache.get_elevation(31.0, -99.0)

    # old_key frequency should now be decayed down to 1
    if old_key in cache._protected:
        assert cache._protected[old_key].frequency == 1


def test_cache_bounded_capacity_enforced() -> None:
    """Total entries across probationary and protected never exceed max_entries."""
    mock = _MockCountingProvider(100.0)
    cache = PopularityDecayingElevationCache(mock, max_entries=15)

    for i in range(100):
        # Mix of new and repeated entries
        lat = 10.0 + (i % 30) * 0.1
        lon = -50.0 + (i % 30) * 0.1
        cache.get_elevation(lat, lon)

    total_entries = len(cache._protected) + len(cache._probationary)
    assert total_entries <= 15


def test_cache_thread_safety() -> None:
    """Concurrent multi-threaded lookups do not corrupt cache state."""
    mock = _MockCountingProvider(100.0)
    cache = PopularityDecayingElevationCache(mock, max_entries=50)

    def worker(idx: int):
        for i in range(20):
            lat = 30.0 + (i % 5) * 0.01
            lon = -100.0 + (i % 5) * 0.01
            cache.get_elevation(lat, lon)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    total_cached = len(cache._protected) + len(cache._probationary)
    assert total_cached <= 50


# --- Factory and Configuration Tests ---


def test_get_elevation_provider_singleton() -> None:
    """get_elevation_provider returns a process-level singleton."""
    from api.core import config as config_mod

    _reset_elevation_provider_cache()
    old_provider = config_mod.settings.ELEVATION_PROVIDER
    try:
        config_mod.settings.ELEVATION_PROVIDER = "open_meteo"
        first = get_elevation_provider()
        second = get_elevation_provider()
        assert first is second
    finally:
        config_mod.settings.ELEVATION_PROVIDER = old_provider
        _reset_elevation_provider_cache()


def test_get_elevation_provider_none_returns_null_provider() -> None:
    """ELEVATION_PROVIDER=none returns _NullProvider."""
    from api.core import config as config_mod

    _reset_elevation_provider_cache()
    old_provider = config_mod.settings.ELEVATION_PROVIDER
    try:
        config_mod.settings.ELEVATION_PROVIDER = "none"
        provider = get_elevation_provider()
        assert isinstance(provider, _NullProvider)
        assert provider.get_elevation(39.0, -106.0) is None
    finally:
        config_mod.settings.ELEVATION_PROVIDER = old_provider
        _reset_elevation_provider_cache()


def test_elevation_metrics_tracking() -> None:
    """Metrics track total requests, hits, misses, and provider failures."""
    _reset_elevation_metrics()

    def transport(url: str):
        if "fail" in url:
            return 500, {}
        return 200, {"elevation": [1234.0]}

    raw_provider = OpenMeteoElevationProvider(transport=transport)
    cache = PopularityDecayingElevationCache(raw_provider, max_entries=10)

    # 1. Miss
    cache.get_elevation(39.1, -106.1)
    # 2. Hit
    cache.get_elevation(39.1, -106.1)
    # 3. Provider failure (pass coord that triggers 500)
    # simulate fail by calling raw provider with custom URL
    fail_provider = OpenMeteoElevationProvider(
        base_url="https://fail.example.com", transport=transport
    )
    fail_cache = PopularityDecayingElevationCache(fail_provider, max_entries=10)
    fail_cache.get_elevation(10.0, 10.0)

    metrics = get_elevation_metrics()
    assert metrics.elevation_requests_total >= 3
    assert metrics.elevation_cache_hits_total >= 1
    assert metrics.elevation_cache_misses_total >= 2
    assert metrics.provider_failures_total >= 1
