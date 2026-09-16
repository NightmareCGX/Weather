"""Tests for P0 First Cold-Load Latency Optimizations.

Covers:
1. Manifest Reader in-memory TTL micro-cache (concurrent collapse, hit ratio, TTL expiration, _clear_manifest_cache).
2. Resolver in-memory short TTL micro-cache (concurrent DB query collapse, hit ratio, commit invalidation, _clear_resolver_cache).
3. PNG encoding level 1 vs level 6 speed and byte equivalence.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.core.manifest_reader import (
    _clear_manifest_cache,
    manifest_generation,
    manifest_storage_format,
)
from api.core.png import encode_rgba_png
from api.models.entities import (
    Base,
    ForecastCenter,
    ForecastCycleLifecycle,
    ForecastGrid,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
    ReclamationQueue,
)
from api.services.resolver import (
    _clear_resolver_cache,
    resolve_canonical_source,
    resolve_valid_time_source,
    resolve_variable_source,
)


# ---------------------------------------------------------------------------
# 1. Manifest Reader Micro-Cache Tests
# ---------------------------------------------------------------------------


def test_manifest_cache_concurrency_and_hit_ratio(tmp_path, monkeypatch):
    """Simulate 20 concurrent tile requests reading the manifest simultaneously.

    Asserts that the underlying storage is read exactly 1 time, yielding a
    95% (19/20) hit ratio.
    """
    _clear_manifest_cache()

    manifest_dir = tmp_path / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps({"manifest_schema_version": 1, "generation": "gen-concurrent-test-1", "storage_format_version": "sharded_v1"}),
        encoding="utf-8",
    )

    import api.core.manifest_reader as mr

    real_read = mr._read_manifest_uncached
    read_count = 0

    def spy_read(store_path: str):
        nonlocal read_count
        # Add slight delay to widen concurrency window
        time.sleep(0.01)
        read_count += 1
        return real_read(store_path)

    monkeypatch.setattr(mr, "_read_manifest_uncached", spy_read)

    store_path = str(tmp_path)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(manifest_generation, store_path) for _ in range(20)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # All 20 threads received the valid generation
    assert len(results) == 20
    assert all(r == "gen-concurrent-test-1" for r in results)
    # Single-flight / Double-checked locking ensures exactly 1 storage read!
    assert read_count == 1

    # manifest_storage_format for the same store immediately hits the warm cache (0 additional reads)
    fmt = manifest_storage_format(store_path)
    assert fmt == "sharded_v1"
    assert read_count == 1


def test_manifest_cache_ttl_expiration(tmp_path, monkeypatch):
    """Test that after TTL expires, a new manifest read is triggered."""
    _clear_manifest_cache()

    manifest_dir = tmp_path / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps({"manifest_schema_version": 1, "generation": "gen-v1"}),
        encoding="utf-8",
    )

    store_path = str(tmp_path)
    import api.core.manifest_reader as mr

    # Initial read
    assert manifest_generation(store_path) == "gen-v1"

    # Simulate manifest update on disk
    manifest_file.write_text(
        json.dumps({"manifest_schema_version": 1, "generation": "gen-v2"}),
        encoding="utf-8",
    )

    # Within TTL (clock not advanced), cache returns old generation
    assert manifest_generation(store_path) == "gen-v1"

    # Fast-forward monotonic clock beyond 30 seconds
    orig_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: orig_monotonic() + 31.0)

    # Now expired, must re-read and discover gen-v2
    assert manifest_generation(store_path) == "gen-v2"


def test_manifest_clear_cache(tmp_path):
    """_clear_manifest_cache() immediately clears cache without waiting for TTL."""
    _clear_manifest_cache()

    manifest_dir = tmp_path / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps({"manifest_schema_version": 1, "generation": "gen-alpha"}),
        encoding="utf-8",
    )
    store_path = str(tmp_path)

    assert manifest_generation(store_path) == "gen-alpha"

    # Update manifest on disk
    manifest_file.write_text(
        json.dumps({"manifest_schema_version": 1, "generation": "gen-beta"}),
        encoding="utf-8",
    )

    # Force clear cache
    _clear_manifest_cache()

    # Immediate perception of new generation
    assert manifest_generation(store_path) == "gen-beta"


# ---------------------------------------------------------------------------
# 2. Resolver Micro-Cache Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def test_db_engine(tmp_path):
    """Create isolated SQLite engine for cold-load resolver tests."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/cold_load_test.db",
        connect_args={"check_same_thread": False},
    )
    tables = [
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ForecastGrid.__table__,
        ForecastVariable.__table__,
        ModelRun.__table__,
        ForecastProduct.__table__,
        ForecastCycleLifecycle.__table__,
        ReclamationQueue.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)

    with Session(engine) as session:
        center = ForecastCenter(id="noaa", center_id="noaa", name="NOAA", country="US")
        gfs = Model(id="gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0)
        v_gfs = ModelVersion(id="v_gfs", model_id="gfs", version_string="v1.0")
        grid = ForecastGrid(id="grid_global", grid_code="global_025deg", name="Global 0.25", resolution_km=25.0)
        var = ForecastVariable(id="var_t2m", variable_code="temperature_2m", name="temperature_2m", unit="°C")
        c_00z = datetime(2026, 9, 15, 0, 0, tzinfo=timezone.utc)
        run_00z = ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z")
        prod = ForecastProduct(id="p_00_12", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=12)
        session.add_all([center, gfs, v_gfs, grid, var, run_00z, prod])
        session.commit()

    yield engine
    engine.dispose()


def test_resolver_cache_concurrency_and_hit_ratio(test_db_engine, monkeypatch):
    """Simulate 20 concurrent tile requests hitting resolve_valid_time_source.

    Verifies that _discover_candidates_bulk is called exactly 1 time across
    all 20 threads, reusing ResolvedForecastSource with 0ms overhead for the other 19.
    """
    _clear_resolver_cache()

    import api.services.resolver as res_mod

    orig_discover = res_mod._discover_candidates_bulk
    discover_calls = 0

    def spy_discover(*args, **kwargs):
        nonlocal discover_calls
        time.sleep(0.01)  # Simulate DB query latency
        discover_calls += 1
        return orig_discover(*args, **kwargs)

    monkeypatch.setattr(res_mod, "_discover_candidates_bulk", spy_discover)

    target_v = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    fixed_now = datetime(2026, 9, 15, 2, 0, tzinfo=timezone.utc)

    def worker():
        with Session(test_db_engine) as session:
            return resolve_valid_time_source(
                session, "gfs", target_v, variable="temperature_2m", now=fixed_now
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(worker) for _ in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 20
    assert all(r.run_id == "r_00z" for r in results)
    assert all(r.lead_time_hours == 12 for r in results)

    # 20 concurrent tile queries triggered only 1 DB discovery query!
    assert discover_calls == 1


def test_resolver_cache_ttl_expiration(test_db_engine, monkeypatch):
    """Verify that after the 8s resolver TTL expires, a fresh DB resolution is executed."""
    _clear_resolver_cache()

    import api.services.resolver as res_mod

    orig_discover = res_mod._discover_candidates_bulk
    discover_calls = 0

    def spy_discover(*args, **kwargs):
        nonlocal discover_calls
        discover_calls += 1
        return orig_discover(*args, **kwargs)

    monkeypatch.setattr(res_mod, "_discover_candidates_bulk", spy_discover)

    target_v = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    fixed_now = datetime(2026, 9, 15, 2, 0, tzinfo=timezone.utc)

    with Session(test_db_engine) as session:
        src1 = resolve_variable_source(session, "gfs", "temperature_2m", target_v, now=fixed_now)
        assert src1 is not None
        assert discover_calls == 1

        # Second call within TTL hits cache
        src2 = resolve_variable_source(session, "gfs", "temperature_2m", target_v, now=fixed_now)
        assert src2 is not None
        assert discover_calls == 1

        # Advance monotonic time by 9 seconds (> 8s TTL)
        orig_monotonic = time.monotonic
        monkeypatch.setattr(time, "monotonic", lambda: orig_monotonic() + 9.0)

        # Third call re-executes DB discovery
        src3 = resolve_variable_source(session, "gfs", "temperature_2m", target_v, now=fixed_now)
        assert src3 is not None
        assert discover_calls == 2


def test_resolver_cache_commit_invalidation(test_db_engine):
    """Verify that a database commit automatically invalidates the resolver micro-cache."""
    _clear_resolver_cache()

    target_v = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    fixed_now = datetime(2026, 9, 15, 2, 0, tzinfo=timezone.utc)

    with Session(test_db_engine) as session:
        src = resolve_canonical_source(session, "gfs", target_v, now=fixed_now)
        assert src.run_id == "r_00z"
        assert src.lead_time_hours == 12

        # A newer cycle commits (06Z with lead 6h)
        c_06z = datetime(2026, 9, 15, 6, 0, tzinfo=timezone.utc)
        run_06z = ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="ready", zarr_store_path="/store/06z")
        prod_06z = ForecastProduct(id="p_06_06", run_id="r_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6)
        session.add_all([run_06z, prod_06z])
        session.commit()

        # Due to after_commit hook, the resolver micro-cache is immediately invalidated
        src_new = resolve_canonical_source(session, "gfs", target_v, now=fixed_now)
        assert src_new.run_id == "r_06z"
        assert src_new.lead_time_hours == 6


# ---------------------------------------------------------------------------
# 3. PNG Encoding Level 1 Optimization Tests
# ---------------------------------------------------------------------------


def test_png_level_1_encode_performance_and_fidelity():
    """Verify level 1 encoding produces correct output and is significantly faster than level 6."""
    # Standard 256x256 RGBA tile (256KB buffer)
    width, height = 256, 256
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels += bytes(((x + y) % 256, (x * 2) % 256, (y * 2) % 256, 255))
    raw_tile = bytes(pixels)

    # Encode with level 1 (default) vs level 6
    iterations = 20

    t0 = time.perf_counter()
    for _ in range(iterations):
        png_lvl1 = encode_rgba_png(raw_tile, width, height, compress_level=1)
    t_lvl1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        png_lvl6 = encode_rgba_png(raw_tile, width, height, compress_level=6)
    t_lvl6 = time.perf_counter() - t0

    # Level 1 must be strictly faster (typically 3-5x faster)
    assert t_lvl1 < t_lvl6

    # Verify byte size difference is minimal (< 15% increase)
    size_lvl1 = len(png_lvl1)
    size_lvl6 = len(png_lvl6)
    ratio = size_lvl1 / size_lvl6
    assert ratio < 1.15, f"Level 1 size {size_lvl1} vs Level 6 size {size_lvl6} exceeds 15% threshold"

    # Verify decompressed pixels from level 1 match original raw_tile
    import zlib
    idat_data = b""
    pos = 8
    import struct
    while pos < len(png_lvl1):
        length = struct.unpack(">I", png_lvl1[pos : pos + 4])[0]
        chunk_type = png_lvl1[pos + 4 : pos + 8]
        if chunk_type == b"IDAT":
            idat_data += png_lvl1[pos + 8 : pos + 8 + length]
        pos += 12 + length

    decompressed = zlib.decompress(idat_data)
    stride = width * 4
    recovered = bytearray()
    for row in range(height):
        offset = row * (stride + 1)
        assert decompressed[offset] == 0
        recovered += decompressed[offset + 1 : offset + 1 + stride]

    assert bytes(recovered) == raw_tile
