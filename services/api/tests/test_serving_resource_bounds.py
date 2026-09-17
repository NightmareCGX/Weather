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


def test_ingestion_only_s3_pool_setting_warns_instead_of_being_silently_ignored(
    monkeypatch, caplog
) -> None:
    """Naming the ingestion pool in the API environment must not fail silently.

    Both tiers share one ``.env``, so the ingestion ``S3_MAX_POOL_CONNECTIONS`` is
    routinely present in the API process. ``extra="ignore"`` drops it, which left a
    deployment tuning the API's pool by that name with no effect and no explanation.
    """
    import logging

    from api.core.config import warn_about_ingestion_only_s3_pool

    monkeypatch.delenv("API_S3_MAX_POOL_CONNECTIONS", raising=False)

    # Ingestion-only name present, no API-specific override -> warned.
    monkeypatch.setenv("S3_MAX_POOL_CONNECTIONS", "50")
    with caplog.at_level(logging.WARNING):
        assert warn_about_ingestion_only_s3_pool() is True
    assert "ingestion-only" in caplog.text

    # An explicit API override means the operator is aware; no warning.
    caplog.clear()
    monkeypatch.setenv("API_S3_MAX_POOL_CONNECTIONS", "32")
    with caplog.at_level(logging.WARNING):
        assert warn_about_ingestion_only_s3_pool() is False
    assert caplog.text == ""

    # Neither name set is the normal case; no warning.
    caplog.clear()
    monkeypatch.delenv("S3_MAX_POOL_CONNECTIONS", raising=False)
    monkeypatch.delenv("API_S3_MAX_POOL_CONNECTIONS", raising=False)
    with caplog.at_level(logging.WARNING):
        assert warn_about_ingestion_only_s3_pool() is False
    assert caplog.text == ""


def test_chunk_cache_ceiling_comes_from_settings() -> None:
    """The chunk cache is right-sized for memory, and overridable.

    Simulating a realistic serving session touches 630 distinct chunks and the LRU
    hit rate plateaus at 1024 entries, so the old 2048 default (~82 MB per reader)
    bought nothing while dominating the API container's memory. 512 costs one
    percentage point of hit rate for a 4x reduction.
    """
    from api.core.zarr import ShardedV1Reader

    assert settings.API_READER_MAX_CACHED_CHUNKS == 512
    assert ShardedV1Reader("s3://bucket/x.zarr").max_cached_chunks == 512
    assert ShardedV1Reader("s3://bucket/x.zarr", max_cached_chunks=7).max_cached_chunks == 7


def test_chunk_cache_evicts_at_its_ceiling(tmp_path: Path) -> None:
    """The ceiling is actually enforced, not merely stored."""
    from api.core.zarr import ShardedV1Reader
    from tests.test_sharded_reader import _build_test_shard

    store = tmp_path / "store.zarr"
    shard = store / "temperature_2m" / "shard.det_L0000.shard"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(_build_test_shard(val_offset=1.0))

    reader = ShardedV1Reader(str(store), max_cached_chunks=10)
    # Read all 8x15 = 120 chunks, far more than the ceiling.
    for row in range(8):
        for col in range(15):
            reader.read_chunk(
                "temperature_2m",
                member=None,
                lead_time_hours=0,
                chunk_row=row,
                chunk_col=col,
            )

    assert len(reader._chunk_cache) <= 10


def test_index_cache_ceiling_comes_from_settings() -> None:
    """The shard-index cache is configurable and no longer a hardcoded 4096.

    A full 80-lead ensemble series resolves ~4.8k shard indexes for wind and ~14.4k
    for 3-hour precipitation, so the old hardcoded ceiling evicted a single series
    and every repeat request re-fetched every index trailer.
    """
    from api.core.zarr import ShardedV1Reader

    assert settings.API_READER_MAX_CACHED_INDICES == 16384
    assert ShardedV1Reader("s3://bucket/x.zarr").max_cached_indices == 16384
    assert ShardedV1Reader("s3://bucket/x.zarr", max_cached_indices=7).max_cached_indices == 7


def test_index_entries_are_compact_not_python_tuples(tmp_path: Path) -> None:
    """Index entries are a uint64 array, which is what makes a larger ceiling cheap.

    The same index costs ~2 KB as an ndarray versus ~14 KB as a list of 120 Python
    int tuples, so 16384 entries here use less memory than the previous 4096 did.
    """
    import numpy as np

    from api.core.zarr import INDEX_ENTRY_SIZE, ShardedV1Reader
    from tests.test_sharded_reader import _build_test_shard

    store = tmp_path / "store.zarr"
    shard = store / "temperature_2m" / "shard.det_L0000.shard"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(_build_test_shard(val_offset=1.0))

    reader = ShardedV1Reader(str(store))
    entries = reader.get_shard_index("temperature_2m/shard.det_L0000.shard")

    assert isinstance(entries, np.ndarray)
    assert entries.shape == (120, 2)
    assert entries.dtype == np.dtype("<u8")
    assert entries.nbytes == 120 * INDEX_ENTRY_SIZE


def test_index_cache_evicts_at_its_ceiling(tmp_path: Path) -> None:
    """The index ceiling is enforced, not merely stored."""
    from api.core.zarr import ShardedV1Reader
    from tests.test_sharded_reader import _build_test_shard

    store = tmp_path / "store.zarr"
    shard_data = _build_test_shard(val_offset=1.0)
    for lead in range(6):
        shard = store / "temperature_2m" / f"shard.det_L{lead:04d}.shard"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(shard_data)

    reader = ShardedV1Reader(str(store), max_cached_indices=3)
    for lead in range(6):
        reader.get_shard_index(f"temperature_2m/shard.det_L{lead:04d}.shard")

    assert len(reader._index_cache) <= 3


def test_point_cache_ceiling_comes_from_settings() -> None:
    """The interpolated-corner cache is configurable."""
    from api.core.zarr import ShardedV1Reader

    assert settings.API_READER_MAX_CACHED_POINTS == 32768
    assert ShardedV1Reader("s3://bucket/x.zarr").max_cached_points == 32768
    assert ShardedV1Reader("s3://bucket/x.zarr", max_cached_points=7).max_cached_points == 7


@pytest.mark.parametrize("use_chunk_hit_path", [True, False])
def test_point_cache_serves_repeat_reads_without_touching_storage(
    tmp_path: Path, use_chunk_hit_path: bool
) -> None:
    """A repeated 2x2 window must not re-read storage, and must return the same value.

    Covers both reader paths: corners inside one 100x100 chunk, and corners straddling
    a chunk boundary (which falls back to four point reads).
    """
    import numpy as np

    from api.core.zarr import ShardedV1Reader
    from tests.test_sharded_reader import _build_test_shard

    store = tmp_path / "store.zarr"
    shard_data = _build_test_shard(val_offset=1.0)
    for lead in range(2):
        shard = store / "temperature_2m" / f"shard.det_L{lead:04d}.shard"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(shard_data)

    reader = ShardedV1Reader(str(store))
    # Window 1: all four corners inside one chunk. Window 2: straddles a chunk boundary.
    if use_chunk_hit_path:
        lat_idx, lon_idx = [10, 11], [20, 21]
    else:
        lat_idx, lon_idx = [99, 100], [20, 21]

    reads: list[str] = []
    original_chunk = reader.read_chunk
    original_point = reader.read_point_value

    def counting_chunk(*args, **kwargs):
        reads.append("chunk")
        return original_chunk(*args, **kwargs)

    def counting_point(*args, **kwargs):
        reads.append("point")
        return original_point(*args, **kwargs)

    reader.read_chunk = counting_chunk  # type: ignore[method-assign]
    reader.read_point_value = counting_point  # type: ignore[method-assign]

    kwargs = dict(
        variable="temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_idx=lat_idx,
        lon_idx=lon_idx,
        t_row=0.25,
        t_col=0.75,
    )
    first = reader.interpolate_point(**kwargs)
    assert len(reads) > 0
    first_reads = len(reads)

    # Repeats must hit the cache and add no storage calls at all (the point cache
    # short-circuits before either reader helper), and must reproduce the cold-reader
    # result for whatever weights the caller asks for. The cache stores the four corner
    # values, not the blended result, so the weights must still take effect: in the
    # straddling window the two corner rows differ, which makes a "frozen blend" bug
    # visible.
    cold = ShardedV1Reader(str(store))
    for weights in ({"t_row": 0.6, "t_col": 0.1}, {"t_row": 0.9, "t_col": 0.9}):
        assert reader.interpolate_point(**{**kwargs, **weights}) == cold.interpolate_point(
            **{**kwargs, **weights}
        )
        assert len(reads) == first_reads

    assert np.isfinite(first)
    if not use_chunk_hit_path:
        # Guard the guard: the straddling fixture really does vary with t_row, so a
        # cache that froze the first blend would be caught above.
        assert reader.interpolate_point(**{**kwargs, "t_row": 0.1}) != reader.interpolate_point(
            **{**kwargs, "t_row": 0.9}
        )


def test_point_cache_serves_a_series_predecessor_lead(tmp_path: Path) -> None:
    """A lead-3 predecessor read is cache-served instead of re-fetched.

    ``build_ensemble_statistics_series`` derives a lead divisible by 6 from the phase
    state at lead-3 and re-reads exactly the quantities (amount, four phase flags, 2m
    temperature) it already read three leads earlier in the same ascending loop. Because
    the point cache key contains the lead, those re-reads land on entries the earlier
    lead populated -- including across request batches, since one reader instance serves
    them all. This is the mechanism that makes a separate predecessor memo unnecessary;
    it fails if the cache key ever stops distinguishing leads.
    """
    from api.core.zarr import ShardedV1Reader
    from tests.test_sharded_reader import _build_test_shard

    import numpy as np

    store = tmp_path / "store.zarr"
    shard_data = _build_test_shard(val_offset=1.0)
    for lead in (3, 6):
        shard = store / "precipitation_amount_3h" / f"shard.mem001_L{lead:04d}.shard"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(shard_data)

    reader = ShardedV1Reader(str(store))

    def window(lead: int) -> float:
        return reader.interpolate_point(
            variable="precipitation_amount_3h",
            member=1,
            lead_time_hours=lead,
            lat_idx=[10, 11],
            lon_idx=[20, 21],
            t_row=0.5,
            t_col=0.5,
        )

    reads: list[str] = []
    original_chunk = reader.read_chunk

    def counting_chunk(*args, **kwargs):
        reads.append("chunk")
        return original_chunk(*args, **kwargs)

    reader.read_chunk = counting_chunk  # type: ignore[method-assign]

    # The series visits lead 3 (populating the window) and then lead 6, whose
    # predecessor read is lead 3 again.
    window(3)
    window(6)
    reads_after_series = len(reads)

    predecessor = window(3)

    assert len(reads) == reads_after_series, "the predecessor read must not touch storage"
    assert np.isfinite(predecessor)


def test_point_cache_is_scoped_to_window_and_generation(tmp_path: Path) -> None:
    """Distinct windows and distinct store generations must not share an entry."""
    from api.core.zarr import ShardedV1Reader
    from tests.test_sharded_reader import _build_test_shard

    store = tmp_path / "store.zarr"
    shard_data = _build_test_shard(val_offset=1.0)
    for lead in range(2):
        shard = store / "temperature_2m" / f"shard.det_L{lead:04d}.shard"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(shard_data)

    reader = ShardedV1Reader(str(store))
    base = dict(
        variable="temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_idx=[10, 11],
        lon_idx=[20, 21],
        t_row=0.5,
        t_col=0.5,
    )
    reader.interpolate_point(**base)
    assert len(reader._point_cache) == 1

    # A different 2x2 window is a different entry.
    reader.interpolate_point(**{**base, "lat_idx": [12, 13]})
    assert len(reader._point_cache) == 2

    # A different generation is a different entry, so a rewritten store cannot be
    # served from a stale window.
    reader.interpolate_point(**{**base, "generation": "gen-2"})
    assert len(reader._point_cache) == 3

    # A different lead is a different entry.
    reader.interpolate_point(**{**base, "lead_time_hours": 1})
    assert len(reader._point_cache) == 4


def test_point_cache_evicts_at_its_ceiling(tmp_path: Path) -> None:
    """The point cache ceiling is enforced."""
    from api.core.zarr import ShardedV1Reader
    from tests.test_sharded_reader import _build_test_shard

    store = tmp_path / "store.zarr"
    shard = store / "temperature_2m" / "shard.det_L0000.shard"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(_build_test_shard(val_offset=1.0))

    reader = ShardedV1Reader(str(store), max_cached_points=3)
    for row in range(6):
        reader.interpolate_point(
            variable="temperature_2m",
            member=None,
            lead_time_hours=0,
            lat_idx=[row, row + 1],
            lon_idx=[20, 21],
            t_row=0.5,
            t_col=0.5,
        )

    assert len(reader._point_cache) <= 3


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
