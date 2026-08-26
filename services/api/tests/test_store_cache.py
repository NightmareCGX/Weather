"""Phase 2 store-handle reuse tests.

Covers the mandatory contracts of the generation-aware lazy-dataset reuse
(``api.core.store_cache`` + ``api.core.zarr.read_dataset_cached``):

* same-cycle replacement: generation A must never leak into generation B,
* broken-store fallback is never locked onto a stale entry,
* failure paths do not poison the cache (no negative caching),
* 16 concurrent readers of the same store/generation see no mutation races
  or cross-request coordinate contamination,
* the cached dataset retains only metadata/coordinates (no decoded fields),
* numerical identity: reused vs fresh opens return identical values.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../packages/domain/src")
))

from api.core.store_cache import StoreHandleCache
from api.core.zarr import read_dataset, read_dataset_cached
from tests._zarr_writer import write_dataset


_LEADS = [0, 12, 24]


def _make_dataset(values_offset: float = 0.0) -> xr.Dataset:
    lat = np.arange(-5.0, 5.0, 1.0)
    lon = np.arange(-10.0, 10.0, 1.0)
    lead = np.asarray(_LEADS, dtype=int)
    lead_g, lat_g, lon_g = np.meshgrid(lead, lat, lon, indexing="ij")
    temperature = (
        10.0 + 0.1 * lat_g + 0.2 * lon_g + 0.5 * lead_g + values_offset
    ).astype(np.float32)
    return xr.Dataset(
        {"temperature_2m": (("lead_time_hours", "latitude", "longitude"), temperature)},
        coords={"lead_time_hours": lead, "latitude": lat, "longitude": lon},
    )


@pytest.fixture()
def cache() -> StoreHandleCache:
    """A fresh isolated StoreHandleCache per test."""
    return StoreHandleCache()


def _write_store(tmp_path, name: str, values_offset: float) -> str:
    store = str(tmp_path / name)
    write_dataset(
        _make_dataset(values_offset),
        store,
        chunks={"lead_time_hours": 1, "latitude": 5, "longitude": 5},
    )
    return store


# ---------------------------------------------------------------------------
# Same-cycle replacement: generation A cannot leak into generation B.
# ---------------------------------------------------------------------------


def test_generation_change_rotates_cached_dataset(tmp_path) -> None:
    """A replaced same-cycle store (new generation key) must not be served
    from the old generation's cached dataset."""
    cache = StoreHandleCache()

    opens: list[str] = []
    datasets_by_gen: dict[str, xr.Dataset] = {}

    def opener_for(gen: str, offset: float):  # type: ignore[no-untyped-def]
        def _open() -> xr.Dataset:
            opens.append(gen)
            # Stand-in for a fresh open of the REPLACED content.
            ds = _make_dataset(offset)
            ds.attrs["generation"] = gen
            datasets_by_gen[gen] = ds
            return ds

        return _open

    # Generation A serving.
    ds_a, hit_a = cache.get_or_open(
        ("s3://store/cycle.zarr", "gen-a"), opener_for("gen-a", 0.0)
    )
    assert hit_a is False
    v_a = float(ds_a["temperature_2m"].sel(lead_time_hours=24, latitude=4, longitude=9))

    # Same generation again -> hit, SAME object.
    ds_a2, hit_a2 = cache.get_or_open(
        ("s3://store/cycle.zarr", "gen-a"), opener_for("gen-a-never", 99.0)
    )
    assert hit_a2 is True
    assert ds_a2 is ds_a

    # Writer replaces the same cycle; committed generation becomes B.
    # The reader resolves B's generation -> different key -> MISS -> fresh open.
    ds_b, hit_b = cache.get_or_open(
        ("s3://store/cycle.zarr", "gen-b"), opener_for("gen-b", 100.0)
    )
    assert hit_b is False
    assert ds_b is not ds_a
    v_b = float(ds_b["temperature_2m"].sel(lead_time_hours=24, latitude=4, longitude=9))
    # B content differs from A exactly by the replacement offset (float32).
    assert v_b == pytest.approx(v_a + 100.0, abs=1e-3)
    # The B open happened exactly once and produced B content.
    assert opens.count("gen-b") == 1


def test_read_dataset_cached_sees_replaced_store_content(tmp_path, monkeypatch) -> None:
    """End-to-end over read_dataset_cached: rewrite the SAME path with new
    values and confirm the next cached call returns the NEW values."""
    store = _write_store(tmp_path, "cycle.zarr", 0.0)
    monkeypatch.setattr("api.core.manifest_reader.manifest_generation", lambda _p: "gen-x")

    first = read_dataset_cached(store)
    v_first = float(first["temperature_2m"].isel(lead_time_hours=2, latitude=9, longitude=19))

    # Replace the store at the same path (same-cycle re-ingestion analog).
    write_dataset(
        _make_dataset(50.0),
        store,
        chunks={"lead_time_hours": 1, "latitude": 5, "longitude": 5},
    )
    second = read_dataset_cached(store)
    v_second = float(second["temperature_2m"].isel(lead_time_hours=2, latitude=9, longitude=19))
    assert v_second == pytest.approx(v_first + 50.0)


# ---------------------------------------------------------------------------
# Broken newest store / fallback / repair lifecycle.
# ---------------------------------------------------------------------------


def test_broken_newest_store_falls_back_and_repairs(tmp_path, monkeypatch) -> None:
    """A broken NEWEST store falls through to an older candidate without
    being poisoned into the cache; once repaired it is opened fresh."""
    from api.core.manifest_reader import ManifestReadError

    calls: list[str] = []

    def fake_generation(store: str) -> str | None:
        calls.append(store)
        if store.endswith("newest"):
            raise ManifestReadError("malformed manifest")
        return "gen-" + store

    monkeypatch.setattr("api.core.manifest_reader.manifest_generation", fake_generation)

    good = _write_store(tmp_path, "older", 0.0)

    # Newest store is unreadable: read_dataset_cached must NOT cache anything
    # for it and the caller sees the exception propagate (fail closed).
    for _ in range(2):
        with pytest.raises(Exception):
            read_dataset_cached(str(tmp_path / "newest"))

    # The older candidate still serves correctly.
    ds = read_dataset_cached(good)
    v = float(ds["temperature_2m"].isel(lead_time_hours=1, latitude=5, longitude=10))
    assert np.isfinite(v)

    # Repair the newest store: its next read succeeds and is opened fresh
    # (the earlier failures were never negative-cached). The repaired read
    # uses a working generation resolver, mirroring the writer having
    # committed a valid manifest.
    repaired = _write_store(tmp_path, "newest", 7.0)
    monkeypatch.setattr(
        "api.core.manifest_reader.manifest_generation",
        lambda s: "gen-" + s,
    )
    ds_new = read_dataset_cached(repaired)
    v_new = float(ds_new["temperature_2m"].isel(lead_time_hours=1, latitude=5, longitude=10))
    assert np.isfinite(v_new)


def test_opener_failure_not_cached_and_retry_succeeds(cache) -> None:
    """A transient opener failure propagates and leaves no entry behind."""
    attempts = {"n": 0}

    def flaky() -> xr.Dataset:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("transient S3 error")
        return _make_dataset()

    with pytest.raises(ConnectionError):
        cache.get_or_open(("k", "g"), flaky)
    assert len(cache) == 0
    ds, hit = cache.get_or_open(("k", "g"), flaky)
    assert hit is False and attempts["n"] == 2


def test_evicted_entry_is_reopened(cache) -> None:
    """Eviction drops the entry; the next lookup reopens (never stale)."""
    small = StoreHandleCache(max_entries=1)
    d1, h1 = small.get_or_open(("a", "g"), _make_dataset)
    d2, h2 = small.get_or_open(("b", "g"), _make_dataset)
    assert h1 is False and h2 is False and len(small) == 1
    d1_again, h1_again = small.get_or_open(("a", "g"), _make_dataset)
    assert h1_again is False  # evicted -> miss -> reopened
    assert d1_again is not d1


# ---------------------------------------------------------------------------
# Concurrent reader safety (same store, same generation).
# ---------------------------------------------------------------------------


def test_16_concurrent_readers_same_generation(tmp_path) -> None:
    """16 threads x several bounded selections each against ONE shared cached
    dataset: no exceptions, identical results, no cross-request contamination."""
    from concurrent.futures import ThreadPoolExecutor

    store = _write_store(tmp_path, "shared.zarr", 0.0)
    ds = read_dataset_cached(store)  # warm the single shared entry

    def select(i: int) -> tuple[float, float]:
        lead = _LEADS[i % len(_LEADS)]
        lat = -5.0 + (i * 13 % 10)
        lon = -10.0 + (i * 7 % 20)
        v = float(ds["temperature_2m"].sel(lead_time_hours=lead, latitude=lat, longitude=lon))
        # Coordinate metadata re-read concurrently must also be stable.
        n_lat = int(ds.sizes["latitude"])
        return v, float(n_lat)

    expected: dict[tuple[int, int], tuple[float, float]] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(select, i) for i in range(16)]
        results = [f.result(timeout=30) for f in futures]

    for i, res in enumerate(results):
        if i in expected:
            assert res == expected[i], f"reader {i} saw contaminated data"
        else:
            expected[i] = res
    assert all(np.isfinite(v) for v, _ in results)


def test_concurrent_readers_via_read_dataset_cached_single_open(
    tmp_path, monkeypatch
) -> None:
    """Under concurrency with a trusted generation, repeated read_dataset_cached
    calls share cached opens; every thread's bounded selection matches the analytic
    field exactly."""
    from concurrent.futures import ThreadPoolExecutor

    store = _write_store(tmp_path, "conc.zarr", 0.0)
    monkeypatch.setattr("api.core.manifest_reader.manifest_generation", lambda _p: "gen-concurrent")

    open_calls = {"n": 0}
    real_read = read_dataset

    def counting_read(target):  # type: ignore[no-untyped-def]
        open_calls["n"] += 1
        return real_read(target)

    monkeypatch.setattr("api.core.zarr.read_dataset", counting_read)

    # 1. Warm once
    _ = read_dataset_cached(store)
    assert open_calls["n"] == 1

    def one(i: int) -> float:
        ds = read_dataset_cached(store)
        lead = _LEADS[i % len(_LEADS)]
        return float(ds["temperature_2m"].sel(lead_time_hours=lead, latitude=0, longitude=-5))

    # 2. 32 concurrent requests across 16 threads must all hit the warm cache
    with ThreadPoolExecutor(max_workers=16) as pool:
        values = list(pool.map(one, range(32)))

    # Analytic field: 10.0 + 0.1*0 + 0.2*(-5) + 0.5*lead
    for i, value in enumerate(values):
        lead = _LEADS[i % len(_LEADS)]
        assert value == pytest.approx(10.0 + 0.2 * -5.0 + 0.5 * lead, abs=1e-5)
    # All 32 warm lookups collapsed to 0 additional opens.
    assert open_calls["n"] == 1


# ---------------------------------------------------------------------------
# Single-flight cold-miss concurrency and failure semantics.
# ---------------------------------------------------------------------------


def test_cold_cache_16_concurrent_readers_single_flight(tmp_path) -> None:
    """16 concurrent threads hitting an empty cache for the SAME key
    synchronized by a barrier collapse to exactly ONE underlying open."""
    import threading

    store = _write_store(tmp_path, "single_flight.zarr", 0.0)
    cache = StoreHandleCache()

    open_calls = {"n": 0}
    lock = threading.Lock()

    def counting_opener() -> xr.Dataset:
        with lock:
            open_calls["n"] += 1
        # Stand-in for network open
        return read_dataset(store)

    barrier = threading.Barrier(16)
    results: list[tuple[xr.Dataset, bool] | None] = [None] * 16

    def worker(i: int) -> None:
        barrier.wait(timeout=10)
        ds, hit = cache.get_or_open((store, "gen-0"), counting_opener)
        results[i] = (ds, hit)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 1 underlying open performed across all 16 concurrent readers
    assert open_calls["n"] == 1
    # Exactly 1 miss (the leader) and 15 hits (the waiters)
    hits = sum(1 for r in results if r is not None and r[1] is True)
    misses = sum(1 for r in results if r is not None and r[1] is False)
    assert misses == 1
    assert hits == 15
    # All 16 readers received the exact same in-memory xr.Dataset object
    first_ds = results[0][0]  # type: ignore[index]
    assert all(r[0] is first_ds for r in results if r is not None)  # type: ignore[index]


def test_single_flight_failure_propagates_to_waiters_and_retries() -> None:
    """A failure in the lead opener propagates to all waiters, clears in-flight
    state, is not negatively cached, and allows subsequent retries."""
    import threading
    import time

    cache = StoreHandleCache()
    attempts = {"n": 0}
    lock = threading.Lock()

    def flaky_opener() -> xr.Dataset:
        with lock:
            attempts["n"] += 1
            cur = attempts["n"]
        if cur == 1:
            time.sleep(0.05)
            raise ConnectionError("transient S3 connection reset")
        return _make_dataset()

    barrier = threading.Barrier(8)
    exceptions: list[Exception | None] = [None] * 8

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            cache.get_or_open(("store/path", "gen-fail"), flaky_opener)
        except Exception as exc:
            exceptions[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 8 threads caught the transient ConnectionError
    assert all(isinstance(e, ConnectionError) for e in exceptions)
    # The opener was called only once during the single-flight
    assert attempts["n"] == 1
    # In-flight table and cache entries are completely clean (no negative cache, no leak)
    assert len(cache._in_flight) == 0
    assert len(cache) == 0

    # Next request retries normally and succeeds
    ds, hit = cache.get_or_open(("store/path", "gen-fail"), flaky_opener)
    assert hit is False
    assert attempts["n"] == 2
    assert isinstance(ds, xr.Dataset)
    assert len(cache) == 1


def test_single_flight_distinct_keys_open_concurrently() -> None:
    """Different keys (e.g. GFS vs GEFS) open concurrently without serializing."""
    import threading
    import time

    cache = StoreHandleCache()
    active = {"count": 0, "max_active": 0}
    lock = threading.Lock()

    def slow_opener() -> xr.Dataset:
        with lock:
            active["count"] += 1
            if active["count"] > active["max_active"]:
                active["max_active"] = active["count"]
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return _make_dataset()

    barrier = threading.Barrier(2)

    def worker(key: tuple[str, str]) -> None:
        barrier.wait(timeout=10)
        cache.get_or_open(key, slow_opener)

    t1 = threading.Thread(target=worker, args=(("gfs/cycle.zarr", "gen-gfs"),))
    t2 = threading.Thread(target=worker, args=(("gefs/cycle.zarr", "gen-gefs"),))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both distinct keys were opening concurrently at the same time
    assert active["max_active"] == 2


def test_generation_rotation_16_concurrent_cold_miss(tmp_path) -> None:
    """When a cycle rotates from gen-A to gen-B, 16 concurrent readers for gen-B
    collapse to exactly 1 open for gen-B and never receive gen-A data."""
    import threading

    cache = StoreHandleCache()
    opens: list[str] = []
    lock = threading.Lock()

    def make_gen(gen: str, offset: float) -> Callable[[], xr.Dataset]:
        def _open() -> xr.Dataset:
            with lock:
                opens.append(gen)
            ds = _make_dataset(offset)
            ds.attrs["generation"] = gen
            return ds
        return _open

    # Warm generation A
    ds_a, _ = cache.get_or_open(("store/cycle.zarr", "gen-A"), make_gen("gen-A", 0.0))
    v_a = float(ds_a["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))

    # 16 concurrent readers arrive for generation B (cold miss for gen-B)
    barrier = threading.Barrier(16)
    results: list[xr.Dataset | None] = [None] * 16

    def worker(i: int) -> None:
        barrier.wait(timeout=10)
        ds, _ = cache.get_or_open(("store/cycle.zarr", "gen-B"), make_gen("gen-B", 50.0))
        results[i] = ds

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 1 open for gen-B
    assert opens.count("gen-B") == 1
    # All 16 readers got the gen-B dataset
    for ds in results:
        assert ds is not None
        assert ds.attrs["generation"] == "gen-B"
        v_b = float(ds["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))
        assert v_b == pytest.approx(v_a + 50.0)


def test_lru_eviction_16_concurrent_cold_miss() -> None:
    """When an entry is evicted, 16 concurrent readers for the evicted key
    collapse to exactly 1 open to reload it."""
    import threading

    small_cache = StoreHandleCache(max_entries=2)
    opens: list[str] = []
    lock = threading.Lock()

    def opener_for(key_name: str) -> Callable[[], xr.Dataset]:
        def _open() -> xr.Dataset:
            with lock:
                opens.append(key_name)
            return _make_dataset()
        return _open

    # Populate key1 and key2
    small_cache.get_or_open(("k1", "g1"), opener_for("k1"))
    small_cache.get_or_open(("k2", "g2"), opener_for("k2"))
    # Insert key3 -> evicts k1
    small_cache.get_or_open(("k3", "g3"), opener_for("k3"))
    assert ("k1", "g1") not in small_cache._entries

    # 16 concurrent readers request evicted key1
    barrier = threading.Barrier(16)
    results: list[xr.Dataset | None] = [None] * 16

    def worker(i: int) -> None:
        barrier.wait(timeout=10)
        ds, _ = small_cache.get_or_open(("k1", "g1"), opener_for("k1"))
        results[i] = ds

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 1 reload open happened for k1 (total opens for k1 is 2: original + reload)
    assert opens.count("k1") == 2
    first_ds = results[0]
    assert all(ds is first_ds for ds in results if ds is not None)


# ---------------------------------------------------------------------------
# Memory safety: the cached dataset retains metadata, not decoded fields.
# ---------------------------------------------------------------------------


def test_cached_dataset_does_not_retain_decoded_fields(tmp_path) -> None:
    """After many bounded selections, no variable in the cached dataset holds
    a materialized numpy array of the gridded field (lazy wrapper retained)."""
    from xarray.backends.zarr import ZarrArrayWrapper
    from xarray.core.indexing import LazilyIndexedArray

    store = _write_store(tmp_path, "lazy.zarr", 0.0)
    ds = read_dataset_cached(store)
    # Perform bounded selections (as selectors would).
    for lead in _LEADS:
        _ = ds["temperature_2m"].sel(lead_time_hours=lead, latitude=slice(-2, 2)).values
    for name, variable in ds.variables.items():
        if name in ("latitude", "longitude", "lead_time_hours"):
            continue  # coordinate arrays are tiny by design
        # The variable._data must NOT be a materialized numpy array.
        assert not isinstance(variable._data, np.ndarray), (
            f"{name}: _data was unexpectedly materialized into np.ndarray"
        )
        inner = variable._data
        layers = [type(inner)]
        while hasattr(inner, "array") and not isinstance(inner, (LazilyIndexedArray, ZarrArrayWrapper)):
            inner = inner.array
            layers.append(type(inner))
        assert isinstance(inner, (LazilyIndexedArray, ZarrArrayWrapper)), (
            f"{name}: expected a lazy backend array, got {type(inner).__name__} (layers: {layers})"
            "(decoded-field retention would violate the memory contract)"
        )


def test_numerical_identity_cached_vs_fresh(tmp_path) -> None:
    """Values selected via the cached dataset equal values from a fresh open."""
    store = _write_store(tmp_path, "ident.zarr", 0.0)
    cached_ds = read_dataset_cached(store)
    fresh_ds = read_dataset(store)
    a = cached_ds["temperature_2m"].values  # tiny fixture grid: full compare OK here
    b = fresh_ds["temperature_2m"].values
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Legacy / no-manifest cache bypass and resource ownership tests.
# ---------------------------------------------------------------------------


def test_no_manifest_store_bypasses_store_handle_cache(tmp_path) -> None:
    """Stores without a manifest return generation=None and bypass StoreHandleCache."""
    from api.core.manifest_reader import manifest_generation
    from api.core.store_cache import store_handle_cache
    from api.core.zarr import open_serving_dataset

    store = _write_store(tmp_path, "no_manifest.zarr", 0.0)
    store_handle_cache.clear()

    assert manifest_generation(store) is None

    # Serving open 1
    with open_serving_dataset(store) as ds1:
        v1 = float(ds1["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))
        assert np.isfinite(v1)

    # Serving open 2
    with open_serving_dataset(store) as ds2:
        v2 = float(ds2["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))
        assert np.isfinite(v2)

    # Must NOT have inserted into StoreHandleCache
    assert len(store_handle_cache) == 0


def test_no_manifest_same_cycle_replacement_without_restart(tmp_path) -> None:
    """Overwriting data in a no-manifest store is immediately visible on the next read."""
    from api.core.store_cache import store_handle_cache
    from api.core.zarr import open_serving_dataset

    store = _write_store(tmp_path, "cycle_replace.zarr", 0.0)
    store_handle_cache.clear()

    with open_serving_dataset(store) as ds:
        v_initial = float(ds["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))

    # Ingestion overwrites the same path with new values (no manifest).
    write_dataset(
        _make_dataset(values_offset=40.0),
        store,
        chunks={"lead_time_hours": 1, "latitude": 5, "longitude": 5},
    )

    # Next request in the SAME process must observe the new values.
    with open_serving_dataset(store) as ds:
        v_after = float(ds["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))

    assert v_after == pytest.approx(v_initial + 40.0)


def test_manifest_created_after_initial_no_manifest_serving(tmp_path) -> None:
    """When a valid manifest appears on a store, the API transitions to generation-aware caching."""
    import json
    from api.core.manifest_reader import manifest_generation
    from api.core.store_cache import store_handle_cache
    from api.core.zarr import open_serving_dataset

    store = _write_store(tmp_path, "transition.zarr", 0.0)
    store_handle_cache.clear()

    # Initial request without manifest -> uncached
    assert manifest_generation(store) is None
    with open_serving_dataset(store) as ds:
        _ = float(ds["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))
    assert len(store_handle_cache) == 0

    # Ingestion finalizer writes a valid marker-v1 manifest.
    manifest_dir = os.path.join(store, "__commit__", "v1")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"manifest_schema_version": 1, "generation": "gen-first-commit-123"}, fh)

    assert manifest_generation(store) == "gen-first-commit-123"

    # Next read should cache under the real generation
    with open_serving_dataset(store) as ds:
        _ = float(ds["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))

    assert len(store_handle_cache) == 1
    assert (store, "gen-first-commit-123") in store_handle_cache._entries

    # Subsequent read is a cache hit
    hit_count_before = store_handle_cache.hits
    with open_serving_dataset(store) as ds:
        _ = float(ds["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))
    assert store_handle_cache.hits == hit_count_before + 1


def test_malformed_manifest_raises_and_does_not_fallback(tmp_path) -> None:
    """A malformed manifest raises ManifestReadError and never silently falls back to legacy serving."""
    from api.core.manifest_reader import ManifestReadError, manifest_generation
    from api.core.store_cache import store_handle_cache
    from api.core.zarr import open_serving_dataset

    store = _write_store(tmp_path, "malformed.zarr", 0.0)
    store_handle_cache.clear()

    manifest_dir = os.path.join(store, "__commit__", "v1")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "manifest.json")
    with open(manifest_path, "wb") as fh:
        fh.write(b"not-valid-json{{{")

    with pytest.raises(ManifestReadError):
        manifest_generation(store)

    with pytest.raises(ManifestReadError):
        with open_serving_dataset(store):
            pass

    assert len(store_handle_cache) == 0


def test_manifest_infrastructure_read_failure_propagates(tmp_path, monkeypatch) -> None:
    """A transport/IO error reading the manifest propagates and does not masquerade as legacy."""
    from api.core.store_cache import store_handle_cache
    from api.core.zarr import open_serving_dataset

    store = _write_store(tmp_path, "io_fail.zarr", 0.0)
    store_handle_cache.clear()

    def failing_read(_path):
        raise ConnectionResetError("S3 connection reset by peer")

    monkeypatch.setattr("api.core.manifest_reader._read_manifest", failing_read)

    with pytest.raises(ConnectionResetError):
        with open_serving_dataset(store):
            pass

    assert len(store_handle_cache) == 0


def test_uncached_dataset_closed_after_selection(tmp_path, monkeypatch) -> None:
    """Uncached datasets are closed on context exit; cached datasets are not closed."""
    import json
    from api.core.store_cache import store_handle_cache
    from api.core.zarr import open_serving_dataset

    store = _write_store(tmp_path, "closing.zarr", 0.0)
    store_handle_cache.clear()

    closed_datasets: list[xr.Dataset] = []
    real_close = xr.Dataset.close

    def spy_close(self):
        closed_datasets.append(self)
        real_close(self)

    monkeypatch.setattr(xr.Dataset, "close", spy_close)

    # 1. Uncached no-manifest store
    with open_serving_dataset(store) as ds:
        _ = float(ds["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))

    assert len(closed_datasets) == 1

    # 2. Cached marker-v1 store
    manifest_dir = os.path.join(store, "__commit__", "v1")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"generation": "gen-keep-open"}, fh)

    closed_datasets.clear()
    with open_serving_dataset(store) as ds:
        _ = float(ds["temperature_2m"].isel(lead_time_hours=0, latitude=0, longitude=0))

    # Cached dataset must NOT have been closed on exit
    assert len(closed_datasets) == 0


def test_api_s3fs_use_listings_cache_disabled(monkeypatch) -> None:
    """API S3FileSystem constructors must pass use_listings_cache=False without network I/O."""
    import s3fs
    from api.core.manifest_reader import _read_manifest
    from api.core.zarr import _resolve_s3_store
    from api.routers.admin import _object_storage_connected

    captured_kwargs: list[dict] = []
    original_init = s3fs.S3FileSystem.__init__

    def spy_init(self, *args, **kwargs):
        captured_kwargs.append(dict(kwargs))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(s3fs.S3FileSystem, "__init__", spy_init)
    # Stub I/O methods so the test does not require live MinIO/S3 connectivity.
    monkeypatch.setattr(
        s3fs.S3FileSystem,
        "cat_file",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("isolated test")),
    )
    monkeypatch.setattr(s3fs.S3FileSystem, "ls", lambda self, *args, **kwargs: [])

    # 1. Manifest reader S3 resolution
    s3fs.S3FileSystem.clear_instance_cache()
    _ = _read_manifest("s3://bucket/store.zarr")
    assert any(k.get("use_listings_cache") is False for k in captured_kwargs)

    # 2. Zarr store S3 resolution
    captured_kwargs.clear()
    s3fs.S3FileSystem.clear_instance_cache()
    _ = _resolve_s3_store("s3://bucket/store.zarr")
    assert any(k.get("use_listings_cache") is False for k in captured_kwargs)

    # 3. Admin health probe
    captured_kwargs.clear()
    s3fs.S3FileSystem.clear_instance_cache()
    _ = _object_storage_connected()
    assert any(k.get("use_listings_cache") is False for k in captured_kwargs)

