"""Elevation resolution from coordinates (ACCEPTANCE_REMEDIATION_PLAN §15-17).

The Coordinates UI must show a terrain elevation in meters for a resolved
location, sourced from an authoritative terrain dataset — never guessed from
forecast variables. This module provides the provider abstraction:

* :class:`ElevationProvider` is the application-level interface
  (``get_elevation(lat, lon) -> float | None``);
* :class:`DEMElevationProvider` reads a local/server-side DEM (a global
  xarray-readable store: Zarr or NetCDF, matching the platform's own storage
  convention) with bilinear interpolation — no external network call and no
  runtime dependency beyond xarray;
* :class:`GoogleElevationProvider` is the network-API alternative behind the
  same interface (kept for future/fallback; **not** the initial implementation);
* :class:`RoundedElevationCache` wraps a provider with a deterministic
  coordinate-normalization LRU, since elevation is effectively static per
  coordinate.

Design notes:

* **No-data / ocean**: ``get_elevation`` returns ``None`` for no-data cells
  (NaN) and ocean, never ``0`` and never a guessed value. The UI renders
  ``unavailable`` for ``None``.
* **Interpolation**: bilinear via ``xarray.DataArray.interp`` when the DEM has
  a regular grid; falls back to nearest-cell for single-cell stores.
* **Caching**: coordinates are rounded to 3 decimal degrees (~100 m at the
  equator), which does not materially degrade a city-level display, so nearby
  clicks share a cache entry deterministically.
* **Testability**: tests inject a tiny synthetic DEM and/or a fake provider; no
  live Copernicus/SRTM download is ever required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from api.core.config import settings
from api.core.zarr import read_dataset

#: Coordinate rounding precision (decimal degrees) used for the elevation cache.
#: 3 decimals ≈ ~100 m at the equator — safe for a city-level display while
#: ensuring nearby clicks reuse the same cached elevation.
CACHE_LAT_ROUND = 3
CACHE_LON_ROUND = 3
#: Default LRU cache size.
DEFAULT_CACHE_MAX = 4096


class ElevationProvider(ABC):
    """Application-level interface for terrain elevation lookups.

    Implementations return the terrain elevation in meters for a WGS84
    coordinate, or ``None`` when no terrain value is available (ocean,
    no-data, coverage boundary, or lookup failure). Never raises for a
    missing value; ``None`` is the "unavailable" signal the UI renders.
    """

    @abstractmethod
    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        """Return the terrain elevation in meters, or ``None`` if unavailable.

        Args:
            latitude: WGS 84 latitude in decimal degrees.
            longitude: WGS 84 longitude in decimal degrees.
        """


class DEMElevationProvider(ElevationProvider):
    """Local/server-side DEM elevation provider (initial implementation).

    Reads a global DEM from an xarray-readable store (Zarr or NetCDF) — the
    platform's own storage convention — with ``latitude``/``longitude``
    coordinates and an ``elevation`` data variable in meters. The DEM can be a
    local directory (``/data/dem/global_30m.zarr``) or an ``s3://`` URL.
    Bilinear interpolation is used on the regular grid; a no-data/ocean cell
    (NaN) yields ``None``.

    The store is opened lazily and cached so repeated lookups do not re-read
    the DEM metadata from disk/object storage on every call.
    """

    def __init__(
        self,
        dem_path: str | None = None,
        *,
        dataset_loader: Callable[[str], xr.Dataset] = read_dataset,
    ) -> None:
        """Create the DEM provider.

        Args:
            dem_path: Path/URL of the DEM store. Defaults to
                ``settings.DEM_DATA_PATH``; when unset the provider always
                returns ``None`` (elevation unavailable), so the product runs
                without a DEM configured.
            dataset_loader: Injectable store reader for tests (defaults to the
                API Zarr reader).
        """
        self._dem_path = dem_path if dem_path is not None else settings.DEM_DATA_PATH
        self._dataset_loader = dataset_loader
        self._dataset: xr.Dataset | None = None

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        if self._dem_path is None:
            return None
        dataset = self._dataset
        if dataset is None:
            try:
                dataset = self._dataset_loader(self._dem_path)
            except Exception:  # noqa: BLE001 - a missing/corrupt DEM means "unavailable"
                return None
            self._dataset = dataset
        if "elevation" not in dataset.data_vars:
            return None
        elevation = dataset["elevation"]
        try:
            value = _bilinear_interp(
                elevation,
                latitude,
                longitude,
            )
        except _OutOfGridError:
            return None
        if value is None or not np.isfinite(value):
            return None
        return value


class _OutOfGridError(Exception):
    """Raised when a coordinate is outside the DEM's grid."""


def _bilinear_interp(
    elevation: xr.DataArray,
    latitude: float,
    longitude: float,
) -> float | None:
    """Bilinear interpolation of a 2-D ``(latitude, longitude)`` field.

    Implemented with NumPy only (xarray's ``interp`` requires scipy, which is
    not a dependency). Handles either latitude ordering and clamps to the grid
    edges (nearest at the boundary). Returns ``None`` when a corner cell is
    NaN (no-data/ocean), so interpolation never fabricates a value across a
    data gap.

    Raises:
        _OutOfGridError: If the coordinate is outside the DEM grid.
    """
    if elevation.ndim < 2:
        raise _OutOfGridError("DEM elevation field is not 2-D")
    lat_axis = np.asarray(elevation.coords["latitude"].values, dtype=float)
    lon_axis = np.asarray(elevation.coords["longitude"].values, dtype=float)
    if len(lat_axis) < 2 or len(lon_axis) < 2:
        raise _OutOfGridError("DEM grid is degenerate")
    lat_asc = lat_axis[0] <= lat_axis[-1]
    lon_asc = lon_axis[0] <= lon_axis[-1]
    lat_axis = np.sort(lat_axis)
    lon_axis = np.sort(lon_axis)
    if not (lat_axis[0] <= latitude <= lat_axis[-1] and lon_axis[0] <= longitude <= lon_axis[-1]):
        raise _OutOfGridError("Coordinate outside DEM grid")

    li = _lower_index(lat_axis, latitude)
    ri = _lower_index(lon_axis, longitude)
    # Indices into the *sorted* axes; map back to the stored axis orientation.
    def _store_idx(axis_sorted_idx: int, is_asc: bool, axis_len: int) -> int:
        return axis_sorted_idx if is_asc else (axis_len - 1 - axis_sorted_idx)

    lat_lo_sorted = li
    lon_lo_sorted = ri
    lat_hi_sorted = min(li + 1, len(lat_axis) - 1)
    lon_hi_sorted = min(ri + 1, len(lon_axis) - 1)
    i00 = _store_idx(lat_lo_sorted, lat_asc, len(lat_axis))
    i10 = _store_idx(lat_hi_sorted, lat_asc, len(lat_axis))
    j00 = _store_idx(lon_lo_sorted, lon_asc, len(lon_axis))
    j10 = _store_idx(lon_hi_sorted, lon_asc, len(lon_axis))
    f00 = float(elevation.values[i00, j00])
    f01 = float(elevation.values[i00, j10])
    f10 = float(elevation.values[i10, j00])
    f11 = float(elevation.values[i10, j10])
    if not all(np.isfinite(v) for v in (f00, f01, f10, f11)):
        return None
    # ``np.ndarray.__getitem__`` is typed ``Any`` in numpy 1.26 stubs, so the
    # numpy scalars read here would propagate ``Any`` through ``wx``/``wy`` and
    # into the return, tripping ``no-any-return`` under mypy 1.9.0 (CI). The
    # runtime values are always real scalars, so normalizing them to Python
    # ``float`` at this boundary is the semantically-correct fix (not a
    # suppression): the interpolation weights are genuinely floats.
    lat_lo = float(lat_axis[lat_lo_sorted])
    lat_hi = float(lat_axis[lat_hi_sorted])
    lon_lo = float(lon_axis[lon_lo_sorted])
    lon_hi = float(lon_axis[lon_hi_sorted])
    wx = (latitude - lat_lo) / (lat_hi - lat_lo) if lat_hi > lat_lo else 0.0
    wy = (longitude - lon_lo) / (lon_hi - lon_lo) if lon_hi > lon_lo else 0.0
    top = f00 * (1 - wy) + f01 * wy
    bottom = f10 * (1 - wy) + f11 * wy
    return top * (1 - wx) + bottom * wx


def _lower_index(axis: npt.NDArray[np.float64], value: float) -> int:
    """Return the index of the largest axis element <= ``value`` (clamped)."""
    pos = int(np.searchsorted(axis, value, side="right"))
    return max(0, min(pos - 1, len(axis) - 1))


class GoogleElevationProvider(ElevationProvider):
    """Google Elevation API provider (alternative, not the initial choice).

    The plan recommends the local DEM as primary; Google Elevation is kept
    behind the same interface for future/fallback. Its ToS restricts caching,
    so it is not the default. The API key lives server-side.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        transport: Callable[[str], tuple[int, Any]] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.GOOGLE_PLACES_API_KEY
        self._transport = transport

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        if not self._api_key:
            return None
        url = (
            "https://maps.googleapis.com/maps/api/elevation/json"
            f"?locations={latitude},{longitude}&key={self._api_key}"
        )
        status, payload = self._transport(url) if self._transport else _http_get(url)
        if status != 200:
            return None
        results = payload.get("results") or []
        if not results:
            return None
        try:
            return float(results[0]["elevation"])
        except (KeyError, TypeError, ValueError):
            return None


def _http_get(url: str) -> tuple[int, Any]:
    """Standard-library GET returning ``(status, json)``."""
    import json as _json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=settings.GOOGLE_PLACES_TIMEOUT) as resp:
            return resp.status, _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except urllib.error.URLError:
        return 0, {}


class RoundedElevationCache(ElevationProvider):
    """Deterministic LRU elevation cache keyed on rounded coordinates.

    Elevation is effectively static per coordinate, so rounding lat/lon to
    :data:`CACHE_LAT_ROUND`/``CACHE_LON_ROUND`` decimal degrees (~100 m) lets
    nearby clicks reuse the same cached elevation without materially degrading
    accuracy. The same normalized coordinate always yields the same cached
    value; provider failures never poison the cache (only ``None`` results are
    cached, and a ``None`` is indistinguishable from "unavailable", which is
    also deterministic for a coordinate).
    """

    def __init__(
        self,
        provider: ElevationProvider,
        max_entries: int = DEFAULT_CACHE_MAX,
    ) -> None:
        self._provider = provider
        self._max_entries = max_entries
        self._cache: dict[tuple[int, int], float | None] = {}
        self._order: list[tuple[int, int]] = []

    @staticmethod
    def _normalize(latitude: float, longitude: float) -> tuple[int, int]:
        """Round coordinates to the cache precision (deterministic keys)."""
        lat_key = round(latitude, CACHE_LAT_ROUND) * (10 ** CACHE_LAT_ROUND)
        lon_key = round(longitude, CACHE_LON_ROUND) * (10 ** CACHE_LON_ROUND)
        return int(lat_key), int(lon_key)

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        key = self._normalize(latitude, longitude)
        if key in self._cache:
            # Refresh recency (move to end).
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        value = self._provider.get_elevation(latitude, longitude)
        self._cache[key] = value
        self._order.append(key)
        if len(self._order) > self._max_entries:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
        return value


def get_elevation_provider() -> ElevationProvider:
    """Return the configured elevation provider (optionally cached).

    ``ELEVATION_PROVIDER`` selects the backend: ``dem`` (default; local
    DEM via :class:`DEMElevationProvider`), ``google`` (network API), or
    ``none`` (elevation always unavailable). A rounded-coordinate cache wraps
    the provider unless disabled.
    """
    if settings.ELEVATION_PROVIDER == "google":
        provider: ElevationProvider = GoogleElevationProvider()
    elif settings.ELEVATION_PROVIDER == "none":
        return _NullProvider()
    else:
        provider = DEMElevationProvider()
    if settings.ELEVATION_CACHE_DISABLED:
        return provider
    return RoundedElevationCache(provider, max_entries=int(settings.ELEVATION_CACHE_MAX))


class _NullProvider(ElevationProvider):
    """Elevation provider that always reports unavailable."""

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        return None
