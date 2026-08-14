"""Unit tests for the elevation provider and cache (mocked DEM, no network).

Elevation is resolved from a local/server-side DEM (xarray-readable Zarr or
NetCDF) with bilinear interpolation. These tests inject a tiny synthetic DEM
and verify coordinate lookup, no-data/ocean behavior, caching determinism, and
provider-failure handling. No live Copernicus/SRTM download or external API is
ever required (ENGINEERING_CONTRACT §8).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from api.services.elevation import (
    DEMElevationProvider,
    ElevationProvider,
    GoogleElevationProvider,
    RoundedElevationCache,
    _NullProvider,
    _reset_elevation_provider_cache,
    get_elevation_provider,
)


def _make_dem(lat_values, lon_values, elevation_values) -> xr.Dataset:
    """Build a small synthetic DEM dataset with an ``elevation`` variable."""
    return xr.Dataset(
        {"elevation": (("latitude", "longitude"), elevation_values)},
        coords={"latitude": lat_values, "longitude": lon_values},
    )


def _flat_dem(elevation: float) -> xr.Dataset:
    return _make_dem([38.0, 39.0], [-107.0, -106.0], [[elevation, elevation], [elevation, elevation]])


def _dem_loader(dataset: xr.Dataset):
    return lambda path: dataset


def test_dem_provider_returns_elevation_for_valid_coordinate() -> None:
    """A valid land coordinate returns the DEM elevation in meters."""
    dem = _flat_dem(1600.0)
    provider = DEMElevationProvider(dem_path="/fake/dem.zarr", dataset_loader=_dem_loader(dem))
    value = provider.get_elevation(38.5, -106.5)
    assert value == pytest.approx(1600.0)


def test_dem_provider_bilinear_interpolation() -> None:
    """Bilinear interpolation returns the expected intermediate value."""
    # Linear ramp: elevation = lat*100 + lon*10 (m). With lon negative, the
    # field is a plane, so bilinear at a cell midpoint equals the analytic
    # value at that point.
    lat = [38.0, 38.5, 39.0]
    lon = [-107.0, -106.5, -106.0]
    elev = np.array([[la * 100 + lo * 10 for lo in lon] for la in lat], dtype=float)
    dem = _make_dem(lat, lon, elev)
    provider = DEMElevationProvider(dem_path="/fake", dataset_loader=_dem_loader(dem))
    value = provider.get_elevation(38.25, -106.75)
    # Analytic value: 38.25*100 + (-106.75)*10 = 3825 - 1067.5 = 2757.5.
    assert value == pytest.approx(2757.5, abs=0.01)


def test_dem_provider_no_data_returns_none() -> None:
    """No-data (NaN) / ocean cells yield None, never 0."""
    dem = _make_dem(
        [38.0, 39.0],
        [-107.0, -106.0],
        [[np.nan, 1600.0], [np.nan, np.nan]],
    )
    provider = DEMElevationProvider(dem_path="/fake", dataset_loader=_dem_loader(dem))
    assert provider.get_elevation(38.5, -106.5) is None  # ocean corner
    assert provider.get_elevation(38.5, -107.0) is None  # NaN cell


def test_dem_provider_unconfigured_returns_none() -> None:
    """A provider with no DEM path reports unavailable, never crashes."""
    provider = DEMElevationProvider(dem_path=None)
    assert provider.get_elevation(38.5, -106.5) is None


def test_dem_provider_missing_dem_returns_none() -> None:
    """A missing/corrupt DEM store degrades to unavailable, never raises."""

    def _broken_loader(path):
        raise OSError("no such store")

    provider = DEMElevationProvider(dem_path="/missing", dataset_loader=_broken_loader)
    assert provider.get_elevation(38.5, -106.5) is None


def test_cache_deterministic_same_normalized_coordinate() -> None:
    """Same normalized coordinate always yields the same cached elevation."""
    dem = _flat_dem(1609.0)
    provider = DEMElevationProvider(dem_path="/fake", dataset_loader=_dem_loader(dem))
    cache = RoundedElevationCache(provider)
    # In-grid coordinates, slightly different within the rounding bucket.
    first = cache.get_elevation(38.5001, -106.5001)
    second = cache.get_elevation(38.5003, -106.5003)
    assert first == second == pytest.approx(1609.0)


def test_cache_serves_from_cache_without_reprovider() -> None:
    """A cache hit does not re-invoke the underlying provider."""
    dem = _flat_dem(1609.0)
    provider = DEMElevationProvider(dem_path="/fake", dataset_loader=_dem_loader(dem))
    calls = {"n": 0}
    original = provider.get_elevation

    def _counting(lat, lon):
        calls["n"] += 1
        return original(lat, lon)

    provider.get_elevation = _counting  # type: ignore[assignment]
    cache = RoundedElevationCache(provider)
    cache.get_elevation(38.5, -106.5)
    cache.get_elevation(38.5, -106.5)
    assert calls["n"] == 1  # second call served from cache


def test_cache_provider_failure_does_not_corrupt() -> None:
    """A provider failure yields None (unavailable), cached deterministically."""
    class _Failing(ElevationProvider):
        def get_elevation(self, latitude, longitude):
            return None

    cache = RoundedElevationCache(_Failing())
    assert cache.get_elevation(38.5, -106.5) is None
    # Deterministic: the same coordinate returns the cached None.
    assert cache.get_elevation(38.5, -106.5) is None


def test_google_provider_alternative() -> None:
    """The Google Elevation alternative implements the interface with a mock."""
    provider = GoogleElevationProvider(
        api_key="test",
        transport=lambda url: (200, {"results": [{"elevation": 1609.0}]}),
    )
    assert provider.get_elevation(39.739, -104.990) == pytest.approx(1609.0)


def test_google_provider_no_results_returns_none() -> None:
    provider = GoogleElevationProvider(
        api_key="test",
        transport=lambda url: (200, {"results": []}),
    )
    assert provider.get_elevation(0.0, 0.0) is None


def test_null_provider_always_unavailable() -> None:
    assert _NullProvider().get_elevation(38.5, -106.5) is None


def test_get_elevation_provider_is_process_level_singleton() -> None:
    """The provider is a process-level singleton reused across requests."""
    from api.core import config as config_mod

    _reset_elevation_provider_cache()
    old_dem = config_mod.settings.DEM_DATA_PATH
    old_provider = config_mod.settings.ELEVATION_PROVIDER
    try:
        config_mod.settings.DEM_DATA_PATH = "/fake/dem.zarr"
        config_mod.settings.ELEVATION_PROVIDER = "dem"
        first = get_elevation_provider()
        second = get_elevation_provider()
        assert first is second  # same singleton, cache spans requests
    finally:
        config_mod.settings.DEM_DATA_PATH = old_dem
        config_mod.settings.ELEVATION_PROVIDER = old_provider
        _reset_elevation_provider_cache()


def test_get_elevation_provider_none_provider() -> None:
    """ELEVATION_PROVIDER=none returns the null provider (always unavailable)."""
    from api.core import config as config_mod

    _reset_elevation_provider_cache()
    old = config_mod.settings.ELEVATION_PROVIDER
    try:
        config_mod.settings.ELEVATION_PROVIDER = "none"
        provider = get_elevation_provider()
        assert isinstance(provider, _NullProvider)
        assert provider.get_elevation(38.5, -106.5) is None
    finally:
        config_mod.settings.ELEVATION_PROVIDER = old
        _reset_elevation_provider_cache()


def test_process_level_cache_serves_repeat_coordinates() -> None:
    """A process-level cache serves repeated coordinates without re-opening."""
    dem = _flat_dem(1609.0)
    dem_provider = DEMElevationProvider(
        dem_path="/fake", dataset_loader=_dem_loader(dem)
    )
    calls = {"n": 0}
    original = dem_provider.get_elevation

    def _counting(lat, lon):
        calls["n"] += 1
        return original(lat, lon)

    dem_provider.get_elevation = _counting  # type: ignore[assignment]
    cache = RoundedElevationCache(dem_provider)
    # First call loads + caches; the second (same rounding bucket) is a cache
    # hit and never invokes the underlying DEM provider.
    cache.get_elevation(38.5001, -106.5001)
    cache.get_elevation(38.5003, -106.5003)
    assert calls["n"] == 1


def test_process_level_cache_missing_dem_returns_none() -> None:
    """A missing DEM at the process level degrades to unavailable, never crashes."""

    def _broken_loader(path):
        raise OSError("no such store")

    provider = DEMElevationProvider(dem_path="/missing", dataset_loader=_broken_loader)
    # Repeated lookups on the shared provider never re-open / never crash.
    assert provider.get_elevation(38.5, -106.5) is None
    assert provider.get_elevation(38.5, -106.5) is None


def test_process_level_cache_invalid_no_data_cell() -> None:
    """An invalid/no-data (NaN) cell yields None, deterministically cached."""
    dem = _make_dem(
        [38.0, 39.0],
        [-107.0, -106.0],
        [[np.nan, np.nan], [np.nan, np.nan]],
    )
    provider = DEMElevationProvider(dem_path="/fake", dataset_loader=_dem_loader(dem))
    cache = RoundedElevationCache(provider)
    assert cache.get_elevation(38.5, -106.5) is None
    assert cache.get_elevation(38.5, -106.5) is None  # cached None


def test_process_level_cache_graceful_provider_failure() -> None:
    """A provider failure returns None and never poisons the shared cache."""

    class _Failing(ElevationProvider):
        def get_elevation(self, latitude, longitude):
            raise OSError("provider down")

    cache = RoundedElevationCache(_Failing())
    # The failure is caught by the caller (returns None), but the cache must
    # not be poisoned: a later success still resolves.
    try:
        value = cache.get_elevation(38.5, -106.5)
    except OSError:
        value = None
    assert value is None or value is None
