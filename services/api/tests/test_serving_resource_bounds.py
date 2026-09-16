"""Bounds on process-wide serving resources (memory and concurrency).

These cover limits that exist to keep the API within the deployed host's budget:

1. ``api.core.zarr`` caches one ``ShardedV1Reader`` per store path, and each reader
   owns its own s3fs client plus ~90 MB of bounded index/chunk caches. The cache
   must therefore be a bounded LRU, not an unbounded map that grows by one entry
   per ingested cycle and is never reclaimed.
2. ``API_MAX_CONCURRENT_GATED_READS`` is documented (and set in production) as the
   bound on concurrent gated Zarr reads. It must actually be enforced, and
   enforced *before* any reader-lock connection is taken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import api.core.zarr as zarr_mod
from api.core import reader_gate as rg
from api.core.config import settings
from api.core.zarr import MAX_READERS, clear_sharded_readers, get_sharded_reader


# ---------------------------------------------------------------------------
# 1. Sharded reader cache is a bounded LRU
# ---------------------------------------------------------------------------


def test_sharded_reader_cache_is_bounded(tmp_path: Path) -> None:
    """Caching must not grow without bound as new store paths are served."""
    clear_sharded_readers()
    try:
        paths = [str(tmp_path / f"store{i}.zarr") for i in range(MAX_READERS + 4)]
        readers = [get_sharded_reader(p) for p in paths]

        assert len(zarr_mod._readers) == MAX_READERS
        # Oldest entries were evicted, newest retained.
        assert paths[0] not in zarr_mod._readers
        assert paths[-1] in zarr_mod._readers
        # A retained store is still served from cache (same instance, not a rebuild).
        assert get_sharded_reader(paths[-1]) is readers[-1]
    finally:
        clear_sharded_readers()


def test_sharded_reader_cache_hit_does_not_evict(tmp_path: Path) -> None:
    """A cache hit refreshes recency, so a hot store survives a cold sweep."""
    clear_sharded_readers()
    try:
        hot = str(tmp_path / "hot.zarr")
        hot_reader = get_sharded_reader(hot)

        # Fill the cache to capacity, then touch the hot entry to make it recent.
        for i in range(MAX_READERS - 1):
            get_sharded_reader(str(tmp_path / f"cold{i}.zarr"))
        assert get_sharded_reader(hot) is hot_reader

        # One more new store must evict a cold entry, not the hot one.
        get_sharded_reader(str(tmp_path / "newcomer.zarr"))
        assert hot in zarr_mod._readers
        assert get_sharded_reader(hot) is hot_reader
    finally:
        clear_sharded_readers()


def test_reader_s3_pool_size_comes_from_settings(tmp_path: Path) -> None:
    """The per-reader s3fs pool is configurable, not a hardcoded 64."""
    assert settings.API_S3_MAX_POOL_CONNECTIONS == 16

    reader = get_sharded_reader("s3://bucket/does-not-need-to-exist.zarr")
    fs = reader._resolve_fs_and_root()[0]
    assert fs is not None
    assert fs.config_kwargs.get("max_pool_connections") == 16
    clear_sharded_readers()


# ---------------------------------------------------------------------------
# 2. Gated-read admission is enforced
# ---------------------------------------------------------------------------


def test_admission_semaphore_is_sized_from_settings(monkeypatch) -> None:
    """The limiter admits exactly ``API_MAX_CONCURRENT_GATED_READS`` readers."""
    monkeypatch.setattr(settings, "API_MAX_CONCURRENT_GATED_READS", 3)
    rg.reset_admission_semaphore()
    try:
        sem = rg._get_admission_semaphore()
        assert sem.acquire(timeout=0.5)
        assert sem.acquire(timeout=0.5)
        assert sem.acquire(timeout=0.5)
        # The bound is exactly the configured value: one more must not be admitted.
        assert not sem.acquire(timeout=0.05)
    finally:
        for _ in range(3):
            sem.release()
        rg.reset_admission_semaphore()


def test_gated_read_rejects_when_admission_is_saturated(monkeypatch) -> None:
    """A saturated admission limit fails fast, before touching gate resources.

    This is the regression guard for ``API_MAX_CONCURRENT_GATED_READS``: it was
    declared, documented and set in production while nothing read it, so the only
    limit was the reader-lock pool — which surfaces exhaustion as a pool timeout
    after consuming a connection, not as queueing.
    """
    monkeypatch.setattr(settings, "API_MAX_CONCURRENT_GATED_READS", 1)
    rg.reset_admission_semaphore()
    sem = rg._get_admission_semaphore()
    assert sem.acquire(timeout=1.0)
    try:

        class _UntouchedPool:
            def connect(self):  # pragma: no cover - must never run
                raise AssertionError("admission must be checked before the pool")

        class _UntouchedLifecycle:
            def enter(self):  # pragma: no cover - must never run
                raise AssertionError("admission must be checked before the lifecycle")

        with pytest.raises(rg.ReaderGateTimeout, match="admission limit"):
            rg.gated_read(
                _UntouchedPool(),  # type: ignore[arg-type]
                _UntouchedLifecycle(),  # type: ignore[arg-type]
                store_path="s3://bucket/x.zarr",
                revalidate_db_url="postgresql://unused",
                materialize=lambda: "never",
                timeout_seconds=0.2,
            )
    finally:
        sem.release()
        rg.reset_admission_semaphore()


def test_admission_slot_is_released_on_failure(monkeypatch) -> None:
    """A failed gated read must give its admission slot back."""
    monkeypatch.setattr(settings, "API_MAX_CONCURRENT_GATED_READS", 1)
    rg.reset_admission_semaphore()
    try:

        class _FailingPool:
            def connect(self):
                raise RuntimeError("simulated pool failure")

        with pytest.raises(RuntimeError, match="simulated pool failure"):
            rg.gated_read(
                _FailingPool(),  # type: ignore[arg-type]
                rg.ReaderGateLifecycle(),
                store_path="s3://bucket/x.zarr",
                revalidate_db_url="postgresql://unused",
                materialize=lambda: "never",
                timeout_seconds=1.0,
            )

        # The slot came back, so a fresh admission succeeds immediately.
        assert rg._get_admission_semaphore().acquire(timeout=0.5)
    finally:
        rg.reset_admission_semaphore()
