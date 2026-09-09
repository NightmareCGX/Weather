"""Elevation resolution from coordinates (Tier 2 Dynamic Coordinates Architecture).

The platform displays terrain elevation in meters for resolved locations as UI
metadata only. Elevation never participates in forecast interpolation, lapse-rate
correction, ingestion, or verification.

This module provides the provider abstraction and popularity-aware caching:

* :class:`ElevationProvider` is the application-level interface
  (``get_elevation(lat, lon) -> float | None``);
* :class:`OpenMeteoElevationProvider` queries the Open-Meteo Elevation API
  (backed by Copernicus GLO-90 90m DEM) with bounded timeouts and graceful
  failure fallback (returns ``None`` on failure, never raises);
* :class:`_NullProvider` always returns ``None`` (safe offline default);
* :class:`PopularityDecayingElevationCache` wraps a provider with a strictly
  bounded, popularity-aware segmented cache (probationary window + protected
  frequent segment with periodic frequency decay).

Design notes:

* **UI-only & Non-blocking**: Elevation failure produces ``None`` (rendered as
  ``unavailable`` by the frontend). It never blocks or fails forecast rendering.
* **Coordinate Quantization**: Coordinates are quantized to 3 decimal degrees
  (:data:`CACHE_LAT_ROUND`, :data:`CACHE_LON_ROUND`), an application-level
  spatial bucketing (~100 m scale at the equator) for UI metadata reuse. It does
  not represent native DEM raster grid cells.
* **Cache Semantics**:
  * popular + recent -> retained in protected segment;
  * popular + quiet -> protected until frequency decays;
  * rare + recent -> admitted to probationary window;
  * rare + old -> evicted first from probationary window;
  * random one-off clicks cannot trivially flush popular terrain.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from api.core.config import settings

logger = logging.getLogger(__name__)

#: Application-level coordinate quantization (decimal degrees) for the cache key.
#: ~0.001 deg is ~111 m lat and ~70-111 m lon depending on latitude. This is an
#: application-level spatial quantization for UI metadata reuse, NOT a native
#: raster cell identity.
CACHE_LAT_ROUND = 3
CACHE_LON_ROUND = 3

#: Default maximum entries in the dynamic coordinate cache.
#: At ~250 bytes per entry, 10,000 entries consumes ~2.5 MB (well under 100 MB).
DEFAULT_CACHE_MAX = 10000

#: Fraction of cache reserved for the probationary admission window.
PROBATION_RATIO = 0.20

#: Number of cache lookups between periodic frequency decay cycles.
DEFAULT_DECAY_INTERVAL = 1000


class ElevationProvider(ABC):
    """Application-level interface for terrain elevation lookups.

    Implementations return the terrain elevation in meters for a WGS84
    coordinate, or ``None`` when no terrain value is available (ocean,
    no-data, coverage boundary, provider failure, or disabled). Never raises
    exceptions to callers; ``None`` is the standard 'unavailable' signal.
    """

    @abstractmethod
    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        """Return the terrain elevation in meters, or ``None`` if unavailable.

        Args:
            latitude: WGS 84 latitude in decimal degrees [-90.0, 90.0].
            longitude: WGS 84 longitude in decimal degrees [-180.0, 180.0].
        """


class OpenMeteoElevationProvider(ElevationProvider):
    """Open-Meteo Elevation API provider (Copernicus DEM 2021 GLO-90, 90m).

    Queries the Open-Meteo elevation endpoint via HTTP GET with bounded timeouts.
    If an API key is configured (commercial tier), it is appended as query param.
    All network, HTTP, timeout, or parsing failures are caught and return ``None``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        transport: Callable[[str], tuple[int, Any]] | None = None,
    ) -> None:
        """Initialize the Open-Meteo elevation provider.

        Args:
            base_url: Base endpoint URL (defaults to ``settings.ELEVATION_BASE_URL``).
            api_key: Optional API key for commercial plans.
            timeout: Socket timeout in seconds (defaults to ``settings.ELEVATION_TIMEOUT_SECONDS``).
            transport: Injectable transport for unit testing (returns ``(status_code, json_dict)``).
        """
        self._base_url = (
            base_url if base_url is not None else str(settings.ELEVATION_BASE_URL)
        )
        self._api_key = (
            api_key if api_key is not None else str(settings.ELEVATION_API_KEY)
        )
        self._timeout = (
            float(timeout)
            if timeout is not None
            else float(settings.ELEVATION_TIMEOUT_SECONDS)
        )
        self._transport = transport

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        """Resolve terrain elevation via Open-Meteo Elevation API."""
        global _METRICS
        _METRICS.provider_requests_total += 1

        url = f"{self._base_url}?latitude={latitude}&longitude={longitude}"
        if self._api_key:
            url += f"&apikey={self._api_key}"

        try:
            if self._transport is not None:
                status, payload = self._transport(url)
            else:
                status, payload = _http_get_json(url, timeout=self._timeout)
        except TimeoutError:
            _METRICS.provider_timeouts_total += 1
            _METRICS.provider_failures_total += 1
            logger.warning(
                "Open-Meteo elevation request timed out for (%f, %f)",
                latitude,
                longitude,
            )
            return None
        except Exception as exc:
            _METRICS.provider_failures_total += 1
            logger.warning(
                "Open-Meteo elevation request failed for (%f, %f): %s",
                latitude,
                longitude,
                exc,
            )
            return None

        if status != 200:
            _METRICS.provider_failures_total += 1
            logger.warning(
                "Open-Meteo elevation returned HTTP %d for (%f, %f)",
                status,
                latitude,
                longitude,
            )
            return None

        if not isinstance(payload, dict):
            _METRICS.provider_failures_total += 1
            return None

        elevations = payload.get("elevation")
        if not isinstance(elevations, list) or len(elevations) == 0:
            _METRICS.provider_failures_total += 1
            return None

        raw_val = elevations[0]
        if raw_val is None:
            return None

        try:
            val_float = float(raw_val)
            if math.isnan(val_float) or math.isinf(val_float):
                return None
            return val_float
        except (TypeError, ValueError):
            _METRICS.provider_failures_total += 1
            return None


def _http_get_json(url: str, timeout: float) -> tuple[int, Any]:
    """Perform standard-library HTTP GET returning (status_code, parsed_json)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "WeatherPlatform-Elevation/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError from exc
        raise


class _NullProvider(ElevationProvider):
    """Elevation provider that always returns unavailable (safe default)."""

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        return None


@dataclass
class _CacheEntry:
    """Internal cache entry with elevation value and popularity/recency metadata."""

    elevation: float | None
    frequency: int
    last_access: int


class PopularityDecayingElevationCache(ElevationProvider):
    """Bounded, popularity-aware segmented cache with periodic frequency decay.

    Architecture:
    * **Probationary Segment (20%)**: New entries enter here upon first access.
      One-off exploratory clicks cycle through probation and are evicted first
      without disturbing the protected popular entries.
    * **Protected Segment (80%)**: Entries hit more than once are promoted to
      the protected segment.
    * **Aging / Decay**: Every :data:`decay_interval` accesses, all stored
      frequency counters are halved (``freq = max(1, freq // 2)``). This ensures
      stale historical popularity decays so old hot entries do not persist indefinitely.
    * **Coordinate Quantization**: Keys are normalized to :data:`CACHE_LAT_ROUND`
      and :data:`CACHE_LON_ROUND` decimal degrees (~100 m scale).
    * **Memory Bound**: Strictly bounded to ``max_entries`` (<= 100 MB guaranteed).
    """

    def __init__(
        self,
        provider: ElevationProvider,
        max_entries: int = DEFAULT_CACHE_MAX,
        decay_interval: int = DEFAULT_DECAY_INTERVAL,
    ) -> None:
        if max_entries < 10:
            max_entries = 10
        self._provider = provider
        self._max_entries = max_entries
        self._probation_cap = max(2, int(max_entries * PROBATION_RATIO))
        self._protected_cap = max(2, max_entries - self._probation_cap)
        self._decay_interval = decay_interval

        # Ordered mappings for LRU order within segments
        self._probationary: OrderedDict[tuple[int, int], _CacheEntry] = OrderedDict()
        self._protected: OrderedDict[tuple[int, int], _CacheEntry] = OrderedDict()

        self._access_counter = 0
        self._lock = threading.Lock()

    @staticmethod
    def _normalize(latitude: float, longitude: float) -> tuple[int, int]:
        """Quantize coordinates to integer keys (~100 m application bucketing)."""
        lat_key = int(round(latitude, CACHE_LAT_ROUND) * (10**CACHE_LAT_ROUND))
        lon_key = int(round(longitude, CACHE_LON_ROUND) * (10**CACHE_LON_ROUND))
        return lat_key, lon_key

    def _decay_frequencies(self) -> None:
        """Halve frequency counters across all cached entries (aging)."""
        for entry in self._probationary.values():
            entry.frequency = max(1, entry.frequency // 2)
        for entry in self._protected.values():
            entry.frequency = max(1, entry.frequency // 2)

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        global _METRICS
        key = self._normalize(latitude, longitude)

        with self._lock:
            self._access_counter += 1
            _METRICS.elevation_requests_total += 1

            if self._access_counter % self._decay_interval == 0:
                self._decay_frequencies()

            # 1. Check Protected Segment
            if key in self._protected:
                _METRICS.elevation_cache_hits_total += 1
                entry = self._protected[key]
                entry.frequency += 1
                entry.last_access = self._access_counter
                self._protected.move_to_end(key)
                return entry.elevation

            # 2. Check Probationary Segment
            if key in self._probationary:
                _METRICS.elevation_cache_hits_total += 1
                entry = self._probationary[key]
                entry.frequency += 1
                entry.last_access = self._access_counter

                # Admission / Promotion check into Protected segment:
                # If protected is not full, promote directly.
                # If protected is full, compare newcomer frequency against the least-recently
                # used victim in protected. A weak newcomer cannot evict a heavily used entry!
                if len(self._protected) < self._protected_cap:
                    self._probationary.pop(key)
                    self._protected[key] = entry
                else:
                    # Peek at the LRU candidate in protected
                    lru_protected_key = next(iter(self._protected))
                    lru_protected_entry = self._protected[lru_protected_key]

                    if entry.frequency > lru_protected_entry.frequency:
                        # Promotion justified: demote LRU protected to probationary
                        self._probationary.pop(key)
                        demoted_key, demoted_entry = self._protected.popitem(last=False)
                        self._protected[key] = entry
                        self._probationary[demoted_key] = demoted_entry
                        if len(self._probationary) > self._probation_cap:
                            self._probationary.popitem(last=False)
                    else:
                        # Keep in probationary window; refresh its recency in probation
                        self._probationary.move_to_end(key)
                return entry.elevation

            # Cache Miss
            _METRICS.elevation_cache_misses_total += 1

        # Resolve via underlying provider outside the lock
        val = self._provider.get_elevation(latitude, longitude)

        with self._lock:
            # Re-check in case another thread populated it
            if key in self._protected:
                return self._protected[key].elevation
            if key in self._probationary:
                return self._probationary[key].elevation

            new_entry = _CacheEntry(
                elevation=val,
                frequency=1,
                last_access=self._access_counter,
            )
            # Add to probationary window
            self._probationary[key] = new_entry
            if len(self._probationary) > self._probation_cap:
                # Evict oldest unpromoted item from probationary window
                self._probationary.popitem(last=False)

            return val


# Alias for backward compatibility with existing tests
RoundedElevationCache = PopularityDecayingElevationCache


@dataclass
class ElevationMetrics:
    """Lightweight in-memory observability metrics for elevation resolution."""

    elevation_requests_total: int = 0
    elevation_cache_hits_total: int = 0
    elevation_cache_misses_total: int = 0
    provider_requests_total: int = 0
    provider_failures_total: int = 0
    provider_timeouts_total: int = 0


_METRICS = ElevationMetrics()


def get_elevation_metrics() -> ElevationMetrics:
    """Return current elevation service operational metrics."""
    return _METRICS


def _reset_elevation_metrics() -> None:
    """Reset operational metrics (test hook only)."""
    global _METRICS
    _METRICS = ElevationMetrics()


#: Module-level singleton provider.
_PROVIDER: ElevationProvider | None = None
_PROVIDER_LOCK = threading.Lock()


def get_elevation_provider() -> ElevationProvider:
    """Return the configured elevation provider (process-level singleton).

    Configured via ``settings.ELEVATION_PROVIDER``:
    * ``open_meteo`` -> :class:`OpenMeteoElevationProvider`
    * ``none``       -> :class:`_NullProvider` (always returns ``None``)

    Wrapped by :class:`PopularityDecayingElevationCache` unless disabled via
    ``settings.ELEVATION_CACHE_DISABLED``.
    """
    global _PROVIDER
    if _PROVIDER is None:
        with _PROVIDER_LOCK:
            if _PROVIDER is None:
                provider_type = str(settings.ELEVATION_PROVIDER).lower()
                if provider_type == "open_meteo":
                    provider: ElevationProvider = OpenMeteoElevationProvider()
                elif provider_type == "none":
                    _PROVIDER = _NullProvider()
                    return _PROVIDER
                else:
                    # Unknown/unconfigured provider degrades safely to null
                    logger.warning(
                        "Unknown ELEVATION_PROVIDER '%s'; using null provider",
                        provider_type,
                    )
                    _PROVIDER = _NullProvider()
                    return _PROVIDER

                if settings.ELEVATION_CACHE_DISABLED:
                    _PROVIDER = provider
                else:
                    _PROVIDER = PopularityDecayingElevationCache(
                        provider, max_entries=int(settings.ELEVATION_CACHE_MAX)
                    )
    return _PROVIDER


def _reset_elevation_provider_cache() -> None:
    """Reset the module-level provider singleton (test hook only)."""
    global _PROVIDER
    with _PROVIDER_LOCK:
        _PROVIDER = None
    _reset_elevation_metrics()
