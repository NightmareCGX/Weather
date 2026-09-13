"""Unit tests for the vector-field cache layering (process-local L1 + Redis L2).

The shared Redis L2 is what makes the background prewarm fleet-wide: a
payload computed (or prewarmed) by one worker must satisfy another worker's
request. Redis is a best-effort accelerator — outages degrade to per-process
recomputation, never failing a request — and can be disabled entirely.
"""

import redis as redis_lib
import pytest

from api.core.config import settings
from api.services import vector_field
from api.services.vector_field import (
    _vector_cache,
    _vector_cache_get,
    _vector_cache_key,
    _vector_cache_set,
)

KEY_A = _vector_cache_key("gfs", "wind_10m", 6, None, "gen1", 2)
KEY_B = _vector_cache_key("gfs", "wind_10m", 12, None, "gen1", 2)


class FakeRedis:
    """Minimal sync-client stub (bytes values, RedisError injection)."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False

    def get(self, key: str) -> bytes | None:
        if self.fail:
            raise redis_lib.RedisError("connection down")
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: bytes) -> None:
        if self.fail:
            raise redis_lib.RedisError("connection down")
        self.store[key] = value
        self.ttls[key] = ttl


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    """Fresh L1 and a fake Redis client for every test."""
    _vector_cache.clear()
    fake = FakeRedis()
    monkeypatch.setattr(vector_field, "_redis_client", fake)
    monkeypatch.setattr(vector_field, "_redis_degraded", False)
    yield fake
    _vector_cache.clear()
    vector_field._redis_degraded = False


def test_set_writes_through_to_shared_redis(_isolated_cache):
    _vector_cache_set(KEY_A, b"PAYLOAD-A")
    redis_key = vector_field._vector_redis_key(KEY_A)
    assert _isolated_cache.store[redis_key] == b"PAYLOAD-A"
    assert _isolated_cache.ttls[redis_key] == vector_field._VECTOR_CACHE_TTL_SECONDS


def test_l2_hit_from_another_worker_populates_l1(_isolated_cache):
    """A payload written by a *different* worker is served via the shared L2."""
    redis_key = vector_field._vector_redis_key(KEY_A)
    _isolated_cache.store[redis_key] = b"PAYLOAD-FROM-OTHER-WORKER"
    # L1 is empty here, so this read must go through Redis.
    assert _vector_cache_get(KEY_A) == b"PAYLOAD-FROM-OTHER-WORKER"
    # And the entry is now cached locally: even with Redis down it is served.
    _isolated_cache.fail = True
    assert _vector_cache_get(KEY_A) == b"PAYLOAD-FROM-OTHER-WORKER"


def test_redis_outage_degrades_without_failing(_isolated_cache):
    """Read misses stay misses and writes are dropped while Redis is down."""
    _isolated_cache.fail = True

    assert _vector_cache_get(KEY_A) is None
    _vector_cache_set(KEY_A, b"PAYLOAD-A")  # must not raise

    # The L1 write still landed, so the computing process serves its own entry.
    assert _vector_cache_get(KEY_A) == b"PAYLOAD-A"
    assert _isolated_cache.store == {}


def test_redis_disabled_skips_l2_entirely(_isolated_cache, monkeypatch):
    monkeypatch.setattr(settings, "API_VECTOR_CACHE_REDIS_ENABLED", False)
    _vector_cache_set(KEY_B, b"PAYLOAD-B")

    assert _vector_cache_get(KEY_B) == b"PAYLOAD-B"  # served from L1
    assert _isolated_cache.store == {}  # Redis untouched


def test_l1_budget_is_bounded(_isolated_cache):
    """The process-local layer evicts oldest entries beyond its budget."""
    budget = vector_field._VECTOR_CACHE_MAX_ENTRIES
    for i in range(budget + 4):
        key = _vector_cache_key("gfs", "wind_10m", i, None, "gen1", 2)
        _vector_cache_set(key, b"x" * 8)
    assert len(_vector_cache) == budget
