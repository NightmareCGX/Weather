"""Circuit breaker and response cache for search providers.

Supports multi-worker Redis synchronization with an automatic in-memory
fallback when Redis is unreachable or unconfigured.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, cast

import redis as redis_lib
from redis import Redis

from api.core.config import settings
from api.schemas import SearchBias, SearchResultOut

logger = logging.getLogger(__name__)

REDIS_CONNECT_TIMEOUT_SECONDS = 1.0
REDIS_TIMEOUT_SECONDS = 1.0


class SearchCircuitBreaker:
    """Lightweight circuit breaker for external search providers.

    Transitions:
    - HEALTHY: normal routing.
    - DEGRADED: tripped after N consecutive failures or an HTTP 429. Traffic
      routes to fallback provider for ``cooldown_seconds``. After cooldown, a
      canary request is permitted to probe recovery.
    """

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        failure_threshold: int | None = None,
        cooldown_seconds: int | None = None,
    ) -> None:
        self._redis_url = redis_url or settings.REDIS_URL
        self._failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else settings.SEARCH_CIRCUIT_BREAKER_FAILURES
        )
        self._cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else settings.SEARCH_CIRCUIT_BREAKER_COOLDOWN_S
        )
        self._in_memory_failures: dict[str, int] = {}
        self._in_memory_cooldown_until: dict[str, float] = {}

    def _get_redis(self) -> Redis | None:
        try:
            client: Redis = redis_lib.from_url(  # type: ignore[no-untyped-call]
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS,
                socket_timeout=REDIS_TIMEOUT_SECONDS,
            )
            client.ping()
            return client
        except Exception:
            return None

    def is_available(self, provider: str) -> bool:
        """Return True if the provider is healthy or ready for a canary probe."""
        if not provider or provider in ("mock", "stub") or self._failure_threshold <= 0:
            return True

        r = self._get_redis()
        if r is not None:
            try:
                breaker_key = f"weather:search:breaker:{provider}"
                is_open = r.get(breaker_key)
                if is_open:
                    return False
                return True
            except Exception as exc:
                logger.debug("Redis circuit breaker check failed, using in-memory: %s", exc)

        # In-memory fallback
        cooldown_until = self._in_memory_cooldown_until.get(provider, 0.0)
        return time.time() >= cooldown_until

    def record_success(self, provider: str) -> None:
        """Record a successful request; resets failure counters and circuit state."""
        if not provider or provider in ("mock", "stub"):
            return

        r = self._get_redis()
        if r is not None:
            try:
                r.delete(
                    f"weather:search:failures:{provider}",
                    f"weather:search:breaker:{provider}",
                )
            except Exception as exc:
                logger.debug("Redis circuit breaker success record failed: %s", exc)

        self._in_memory_failures.pop(provider, None)
        self._in_memory_cooldown_until.pop(provider, None)

    def record_failure(self, provider: str, *, is_429: bool = False) -> None:
        """Record a provider failure or rate-limit throttle.

        HTTP 429 trips the breaker immediately. General timeouts/5xx trip after
        ``failure_threshold`` occurrences.
        """
        if not provider or provider in ("mock", "stub"):
            return
        r = self._get_redis()
        if r is not None:
            try:
                failures_key = f"weather:search:failures:{provider}"
                breaker_key = f"weather:search:breaker:{provider}"
                if is_429:
                    r.set(breaker_key, "degraded", ex=self._cooldown_seconds)
                    r.delete(failures_key)
                    return

                count = cast(int, r.incr(failures_key))
                r.expire(failures_key, self._cooldown_seconds * 2)
                if count >= self._failure_threshold:
                    r.set(breaker_key, "degraded", ex=self._cooldown_seconds)
                    r.delete(failures_key)
                return
            except Exception as exc:
                logger.debug("Redis circuit breaker failure record failed: %s", exc)

        # In-memory fallback
        if is_429:
            self._in_memory_cooldown_until[provider] = time.time() + self._cooldown_seconds
            self._in_memory_failures.pop(provider, None)
            return

        current = self._in_memory_failures.get(provider, 0) + 1
        self._in_memory_failures[provider] = current
        if current >= self._failure_threshold:
            self._in_memory_cooldown_until[provider] = time.time() + self._cooldown_seconds
            self._in_memory_failures.pop(provider, None)

    def reset(self, provider: str | None = None) -> None:
        """Reset breaker state and failure counters."""
        r = self._get_redis()
        if r is not None:
            try:
                if provider:
                    r.delete(
                        f"weather:search:failures:{provider}",
                        f"weather:search:breaker:{provider}",
                    )
                else:
                    failure_keys = cast(list[str], r.keys("weather:search:failures:*"))
                    for key in failure_keys:
                        r.delete(key)
                    breaker_keys = cast(list[str], r.keys("weather:search:breaker:*"))
                    for key in breaker_keys:
                        r.delete(key)
            except Exception as exc:
                logger.debug("Redis circuit breaker reset failed: %s", exc)

        if provider:
            self._in_memory_failures.pop(provider, None)
            self._in_memory_cooldown_until.pop(provider, None)
        else:
            self._in_memory_failures.clear()
            self._in_memory_cooldown_until.clear()


def quantize_bias(bias: SearchBias | None) -> str:
    """Quantize proximity bias coordinate to a ~11km grid for safe cache sharing."""
    if bias is None:
        return "none"
    lon = round(bias.longitude, 1)
    lat = round(bias.latitude, 1)
    return f"{lon:.1f}_{lat:.1f}"


class SearchCache:
    """Short-lived cache for autocomplete suggestions (max 300s TTL)."""

    def __init__(self, *, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or settings.REDIS_URL
        self._in_memory: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def _get_redis(self) -> Redis | None:
        try:
            client: Redis = redis_lib.from_url(  # type: ignore[no-untyped-call]
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS,
                socket_timeout=REDIS_TIMEOUT_SECONDS,
            )
            client.ping()
            return client
        except Exception:
            return None

    def make_key(self, provider: str, query: str, bias: SearchBias | None) -> str:
        q_norm = query.strip().lower()
        bias_bucket = quantize_bias(bias)
        return f"weather:search:cache:{provider}:en:{bias_bucket}:{q_norm}"

    def get(
        self, provider: str, query: str, bias: SearchBias | None = None
    ) -> list[SearchResultOut] | None:
        if settings.SEARCH_CACHE_DISABLED or provider in ("mock", "stub"):
            return None

        key = self.make_key(provider, query, bias)
        r = self._get_redis()
        if r is not None:
            try:
                raw_data = r.get(key)
                if raw_data is not None:
                    data_str = cast(str, raw_data)
                    items = json.loads(data_str)
                    return [SearchResultOut(**item) for item in items]
            except Exception as exc:
                logger.debug("Redis search cache get failed: %s", exc)

        # In-memory fallback
        entry = self._in_memory.get(key)
        if entry is not None:
            expires_at, items = entry
            if time.time() < expires_at:
                return [SearchResultOut(**item) for item in items]
            self._in_memory.pop(key, None)

        return None

    def set(
        self,
        provider: str,
        query: str,
        results: list[SearchResultOut],
        bias: SearchBias | None = None,
        ttl_seconds: int = 300,
    ) -> None:
        if settings.SEARCH_CACHE_DISABLED or not results or provider in ("mock", "stub"):
            return

        key = self.make_key(provider, query, bias)
        payload = [item.model_dump() for item in results]
        r = self._get_redis()
        if r is not None:
            try:
                r.set(key, json.dumps(payload), ex=ttl_seconds)
                return
            except Exception as exc:
                logger.debug("Redis search cache set failed: %s", exc)

        # In-memory fallback
        self._in_memory[key] = (time.time() + ttl_seconds, payload)

    def clear(self) -> None:
        """Clear all search cache entries."""
        r = self._get_redis()
        if r is not None:
            try:
                cache_keys = cast(list[str], r.keys("weather:search:cache:*"))
                for key in cache_keys:
                    r.delete(key)
            except Exception as exc:
                logger.debug("Redis search cache clear failed: %s", exc)
        self._in_memory.clear()


# Singleton instances for runtime use
circuit_breaker = SearchCircuitBreaker()
search_cache = SearchCache()
