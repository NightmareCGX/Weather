"""Unit and integration tests for ShardedV1Reader and dual-reader dispatch in API service."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numcodecs import Zstd

from api.core.zarr import (
    INDEX_ENTRY_SIZE,
    SHARD_MAGIC,
    ShardedV1Reader,
)


@pytest.fixture(autouse=True)
def _no_reader_pool(monkeypatch):
    """Force the direct bounded read path for standalone store tests without DB fixtures."""
    try:
        import api.main as main

        if hasattr(main, "reader_pool"):
            monkeypatch.setattr(main, "reader_pool", None)
        if hasattr(main, "reader_lifecycle"):
            monkeypatch.setattr(main, "reader_lifecycle", None)
    except ImportError:
        pass
    yield


def _build_test_shard(val_offset: float = 0.0) -> bytes:
    compressor = Zstd(level=5)
    raw_chunks = []
    for i in range(120):
        arr = np.full((1, 1, 100, 100), float(i) + val_offset, dtype=np.float32)
        raw_chunks.append(compressor.encode(arr.tobytes()))

    buf = io.BytesIO()
    index_entries = []
    curr = 0
    for c in raw_chunks:
        index_entries.append((curr, len(c)))
        buf.write(c)
        curr += len(c)

    for off, length in index_entries:
        buf.write(struct.pack("<QQ", off, length))

    num_chunks = len(raw_chunks)
    index_size = len(index_entries) * INDEX_ENTRY_SIZE
    buf.write(struct.pack("<III", num_chunks, index_size, SHARD_MAGIC))
    return buf.getvalue()


def test_sharded_v1_reader_read_point_value(tmp_path: Path) -> None:
    """Test reading a single cell via ShardedV1Reader."""
    shard_data = _build_test_shard(val_offset=10.0)
    shard_path = tmp_path / "temperature_2m" / "shard.mem001_L0006.shard"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(shard_data)

    reader = ShardedV1Reader(str(tmp_path))

    # Test coordinate: lat_idx=201 (row 2), lon_idx=1020 (col 10) -> chunk_idx = 2 * 15 + 10 = 40
    # Value in chunk 40 should be 40.0 + 10.0 = 50.0
    val = reader.read_point_value(
        "temperature_2m",
        member=1,
        lead_time_hours=6,
        lat_idx=201,
        lon_idx=1020,
    )
    assert val == 50.0


def test_sharded_v1_reader_lru_cache(tmp_path: Path) -> None:
    """Test that index tables are cached in LRU cache upon first read."""
    shard_data = _build_test_shard(val_offset=0.0)
    shard_path = tmp_path / "temperature_2m" / "shard.det_L0012.shard"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(shard_data)

    reader = ShardedV1Reader(str(tmp_path), max_cached_indices=10)
    shard_key = "temperature_2m/shard.det_L0012.shard"

    # First fetch: cache miss
    entries1 = reader.get_shard_index(shard_key)
    assert len(entries1) == 120
    cache_key = f"{tmp_path}::live::{shard_key}"
    assert cache_key in reader._index_cache

    # Second fetch: cache hit
    entries2 = reader.get_shard_index(shard_key)
    assert entries2 == entries1


def test_sharded_v1_reader_interpolate_point(tmp_path: Path) -> None:
    """Test bilinear interpolation of 2x2 grid neighborhood."""
    shard_data = _build_test_shard(val_offset=0.0)
    shard_path = tmp_path / "temperature_2m" / "shard.det_L0000.shard"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(shard_data)

    reader = ShardedV1Reader(str(tmp_path))

    # In chunk 0 (val = 0.0), lat 10..11, lon 20..21
    # All 4 corners have value 0.0 -> interpolated = 0.0
    val = reader.interpolate_point(
        "temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_idx=[10, 11],
        lon_idx=[20, 21],
        t_row=0.5,
        t_col=0.5,
    )
    assert val == 0.0


def test_sharded_v1_reader_read_window(tmp_path: Path) -> None:
    """Test window reading across chunk boundaries."""
    shard_data = _build_test_shard(val_offset=10.0)
    shard_path = tmp_path / "temperature_2m" / "shard.det_L0000.shard"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(shard_data)

    reader = ShardedV1Reader(str(tmp_path))

    # Window from lat 95..105 (crossing chunk row 0 and 1), lon 95..105 (crossing chunk col 0 and 1)
    window = reader.read_window(
        "temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_min=95,
        lat_max=105,
        lon_min=95,
        lon_max=105,
    )
    assert window.shape == (11, 11)
    # chunk 0 (row 0, col 0) value is 0 + 10 = 10
    assert window[0, 0] == 10.0
    # chunk 1 (row 0, col 1) value is 1 + 10 = 11
    assert window[0, 10] == 11.0
    # chunk 15 (row 1, col 0) value is 15 + 10 = 25
    assert window[10, 0] == 25.0
    # chunk 16 (row 1, col 1) value is 16 + 10 = 26
    assert window[10, 10] == 26.0


def test_manifest_storage_format_dispatch(tmp_path: Path) -> None:
    """Test storage format version resolution from committed manifest."""
    import json
    from api.core.manifest_reader import manifest_storage_format

    store_path = str(tmp_path)
    # No manifest -> legacy v2_unsharded
    assert manifest_storage_format(store_path) == "v2_unsharded"

    # Write sharded_v1 manifest
    manifest_dir = tmp_path / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"manifest_schema_version": 1, "generation": "abc", "storage_format_version": "sharded_v1"})
    )
    assert manifest_storage_format(store_path) == "sharded_v1"


def test_gated_point_interpolations_gefs_sharded_v1(tmp_path: Path) -> None:
    """Test that gated_point_interpolations correctly serves GEFS ensemble means on sharded_v1."""
    import json
    import xarray as xr
    from api.services.point_forecast import gated_point_interpolations

    # Set up sharded_v1 store structure for GEFS with 3 members
    store_dir = tmp_path / "gefs_test.zarr"
    store_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = store_dir / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({
            "manifest_schema_version": 1,
            "generation": "gen_gefs_test",
            "storage_format_version": "sharded_v1",
        })
    )

    # Write members 1, 2, 3 for temperature_2m, wind_u_10m, wind_v_10m, precipitation_amount_3h
    for m, val_offset in [(1, 10.0), (2, 20.0), (3, 30.0)]:
        # temperature_2m
        t_shard = _build_test_shard(val_offset=val_offset)
        p = store_dir / "temperature_2m" / f"shard.mem{m:03d}_L0006.shard"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(t_shard)

        # wind_u_10m (m=1 -> 3.0, m=2 -> 4.0, m=3 -> 5.0 in m/s -> mean = 4.0 m/s)
        u_shard = _build_test_shard(val_offset=val_offset / 10.0 + 2.0)
        p = store_dir / "wind_u_10m" / f"shard.mem{m:03d}_L0006.shard"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(u_shard)

        # wind_v_10m (m=1 -> 3.0, m=2 -> 4.0, m=3 -> 5.0 in m/s -> mean = 4.0 m/s)
        v_shard = _build_test_shard(val_offset=val_offset / 10.0 + 2.0)
        p = store_dir / "wind_v_10m" / f"shard.mem{m:03d}_L0006.shard"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(v_shard)

        # precipitation_amount_3h
        pr_shard = _build_test_shard(val_offset=val_offset / 10.0)
        p = store_dir / "precipitation_amount_3h" / f"shard.mem{m:03d}_L0006.shard"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pr_shard)

    # Write official precomputed mean shards (mean offset: temp=20.0, u=4.0, v=4.0, precip=2.0)
    # Expected temperature mean: 18 (chunk base) + 20 = 38.0
    p_t_mean = store_dir / "temperature_2m" / "shard.mean_L0006.shard"
    p_t_mean.write_bytes(_build_test_shard(val_offset=20.0))

    p_u_mean = store_dir / "wind_u_10m" / "shard.mean_L0006.shard"
    p_u_mean.write_bytes(_build_test_shard(val_offset=4.0))

    p_v_mean = store_dir / "wind_v_10m" / "shard.mean_L0006.shard"
    p_v_mean.write_bytes(_build_test_shard(val_offset=4.0))

    p_pr_mean = store_dir / "precipitation_amount_3h" / "shard.mean_L0006.shard"
    p_pr_mean.write_bytes(_build_test_shard(val_offset=2.0))

    # Write .zmetadata / coords for xarray metadata open
    latitudes = [90.0 - i * 0.25 for i in range(721)]
    longitudes = [0.0 + j * 0.25 for j in range(1440)]
    ds_meta = xr.Dataset(
        data_vars={
            "temperature_2m": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((3, 1, 721, 1440), dtype=np.float32)),
            "wind_u_10m": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((3, 1, 721, 1440), dtype=np.float32)),
            "wind_v_10m": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((3, 1, 721, 1440), dtype=np.float32)),
            "precipitation_amount_3h": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((3, 1, 721, 1440), dtype=np.float32)),
        },
        coords={
            "member": [1, 2, 3],
            "lead_time_hours": [6],
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    ds_meta.to_zarr(str(store_dir), mode="a", consolidated=True, zarr_format=2)

    # Interpolate at lat=45.0, lon=90.0 (inside chunk 0: lat idx 180, lon idx 360 -> chunk row 1, col 3 -> chunk_idx = 18)
    # Chunk 18 base value = 18.0
    # Expected temperature mean: mean([18+10, 18+20, 18+30]) = 38.0
    res = gated_point_interpolations(
        str(store_dir),
        var_codes=("temperature_2m", "wind_10m", "precipitation_amount_3h"),
        lead=6,
        latitude=45.0,
        longitude=90.0,
    )
    assert res is not None
    assert np.isclose(res["temperature_2m"], 38.0)
    assert not np.isnan(res["wind_10m"])
    assert res["wind_10m"] > 0.0
    assert not np.isnan(res["precipitation_amount_3h"])


def test_gated_point_interpolations_gefs_missing_mean_raises_no_fallback(tmp_path: Path) -> None:
    """Prove that if official mean shard is missing on sharded_v1, it returns None (FileNotFoundError)
    and NEVER silently falls back to 30-member runtime reduction."""
    import json
    import xarray as xr
    from api.services.point_forecast import gated_point_interpolations

    store_dir = tmp_path / "gefs_missing_mean.zarr"
    store_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = store_dir / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({
            "manifest_schema_version": 1,
            "generation": "gen_gefs_missing",
            "storage_format_version": "sharded_v1",
        })
    )

    # Write member shards ONLY (no mean shard)
    for m in range(1, 4):
        p = store_dir / "temperature_2m" / f"shard.mem{m:03d}_L0006.shard"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_build_test_shard(val_offset=float(m * 10)))

    latitudes = [90.0 - i * 0.25 for i in range(721)]
    longitudes = [0.0 + j * 0.25 for j in range(1440)]
    ds_meta = xr.Dataset(
        data_vars={
            "temperature_2m": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((3, 1, 721, 1440), dtype=np.float32)),
        },
        coords={
            "member": [1, 2, 3],
            "lead_time_hours": [6],
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    ds_meta.to_zarr(str(store_dir), mode="a", consolidated=True, zarr_format=2)

    # Missing mean shard must return None (via FileNotFoundError) rather than falling back to members!
    res = gated_point_interpolations(
        str(store_dir),
        var_codes=("temperature_2m",),
        lead=6,
        latitude=45.0,
        longitude=90.0,
    )
    assert res is None, "Expected None (store unreadable/missing mean), but got fallback result!"


def test_interpolate_members_numerical_equivalence(tmp_path: Path) -> None:
    """Prove that bounded concurrent member interpolation is numerically identical to serial."""
    store_dir = tmp_path / "numerical_test.zarr"
    store_dir.mkdir(parents=True, exist_ok=True)

    # 6 members: finite, NaN, mixed values
    for m in range(1, 7):
        if m == 4:
            # Member 4 produces NaN chunk
            val_offset = float("nan")
        else:
            val_offset = float(m * 5.0)

        t_shard = _build_test_shard(val_offset=val_offset)
        p = store_dir / "temperature_2m" / f"shard.mem{m:03d}_L0006.shard"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(t_shard)

    reader = ShardedV1Reader(str(store_dir))
    members = list(range(1, 7))
    lat_idx = [10, 11]
    lon_idx = [20, 21]
    t_row = 0.25
    t_col = 0.75

    # 1. Serial execution
    serial_vals = [
        reader.interpolate_point(
            "temperature_2m",
            member=m,
            lead_time_hours=6,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            t_row=t_row,
            t_col=t_col,
        )
        for m in members
    ]

    # 2. Concurrent execution with max_workers=4
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        concurrent_vals = reader.interpolate_members(
            "temperature_2m",
            members=members,
            lead_time_hours=6,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            t_row=t_row,
            t_col=t_col,
            executor=ex,
        )

    assert len(serial_vals) == len(concurrent_vals) == 6
    for s, c in zip(serial_vals, concurrent_vals, strict=True):
        if np.isnan(s):
            assert np.isnan(c)
        else:
            assert np.isclose(s, c, rtol=1e-9, atol=1e-9)

    # Ensemble mean reduction equivalence
    assert np.isclose(
        float(np.nanmean(serial_vals)),
        float(np.nanmean(concurrent_vals)),
        rtol=1e-9,
        atol=1e-9,
    )


def test_interpolate_members_concurrency_overlap_and_bounded(tmp_path: Path) -> None:
    """Instrumented test proving member reads overlap (>1) and respect limit (<=4)."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    reader = ShardedV1Reader(str(tmp_path))

    lock = threading.Lock()
    active_calls = 0
    max_concurrent_calls = 0

    def mock_interpolate_point(*args: Any, **kwargs: Any) -> float:
        nonlocal active_calls, max_concurrent_calls
        with lock:
            active_calls += 1
            if active_calls > max_concurrent_calls:
                max_concurrent_calls = active_calls
        try:
            time.sleep(0.01)  # 10ms simulated I/O
            return 42.0
        finally:
            with lock:
                active_calls -= 1

    reader.interpolate_point = mock_interpolate_point  # type: ignore[method-assign]

    limit = 4
    with ThreadPoolExecutor(max_workers=limit) as ex:
        res = reader.interpolate_members(
            "temperature_2m",
            members=list(range(1, 13)),  # 12 members
            lead_time_hours=6,
            lat_idx=[0, 1],
            lon_idx=[0, 1],
            t_row=0.5,
            t_col=0.5,
            executor=ex,
        )

    assert len(res) == 12
    assert all(v == 42.0 for v in res)
    # Proves structural concurrency: calls did overlap in parallel
    assert max_concurrent_calls > 1, f"Expected concurrency > 1, got {max_concurrent_calls}"
    # Proves bounded concurrency: strictly bounded by configured limit
    assert max_concurrent_calls <= limit, (
        f"Expected max concurrency <= {limit}, got {max_concurrent_calls}"
    )


def test_interpolate_members_exception_propagation(tmp_path: Path) -> None:
    """Prove that an exception in any member read propagates out of interpolate_members."""
    from concurrent.futures import ThreadPoolExecutor
    import pytest

    reader = ShardedV1Reader(str(tmp_path))

    def failing_interpolate_point(variable: str, *, member: int | None, **kwargs: Any) -> float:
        if member == 3:
            raise OSError("Simulated S3 Range GET network failure on member 3")
        return 20.0

    reader.interpolate_point = failing_interpolate_point  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=4) as ex:
        with pytest.raises(OSError, match="Simulated S3 Range GET network failure on member 3"):
            reader.interpolate_members(
                "temperature_2m",
                members=[1, 2, 3, 4],
                lead_time_hours=6,
                lat_idx=[0, 1],
                lon_idx=[0, 1],
                t_row=0.5,
                t_col=0.5,
                executor=ex,
            )


def test_interpolate_members_edge_cases(tmp_path: Path) -> None:
    """Test fast-path behavior for empty and single-member lists."""
    reader = ShardedV1Reader(str(tmp_path))

    # Empty members
    assert reader.interpolate_members(
        "temperature_2m",
        members=[],
        lead_time_hours=0,
        lat_idx=[0, 1],
        lon_idx=[0, 1],
        t_row=0.5,
        t_col=0.5,
    ) == []

    # Single member fast-path (does not touch executor)
    called = False

    def mock_point(*args: Any, **kwargs: Any) -> float:
        nonlocal called
        called = True
        return 123.45

    reader.interpolate_point = mock_point  # type: ignore[method-assign]
    res = reader.interpolate_members(
        "temperature_2m",
        members=[7],
        lead_time_hours=0,
        lat_idx=[0, 1],
        lon_idx=[0, 1],
        t_row=0.5,
        t_col=0.5,
    )
    assert called is True
    assert res == [123.45]


def test_sharded_v1_reader_mean_shard(tmp_path: Path) -> None:
    """Test reading from an official precomputed mean shard container (shard.mean_Lxxxx.shard)."""
    # Write shard with offset 50.0
    mean_shard_data = _build_test_shard(val_offset=50.0)
    p_mean = tmp_path / "temperature_2m" / "shard.mean_L0006.shard"
    p_mean.parent.mkdir(parents=True, exist_ok=True)
    p_mean.write_bytes(mean_shard_data)

    # Write distinct member shard with offset 10.0 (proving no collision)
    mem_shard_data = _build_test_shard(val_offset=10.0)
    p_mem = tmp_path / "temperature_2m" / "shard.mem001_L0006.shard"
    p_mem.write_bytes(mem_shard_data)

    reader = ShardedV1Reader(str(tmp_path))

    # 1. has_mean_shard
    assert reader.has_mean_shard("temperature_2m", 6) is True
    assert reader.has_mean_shard("temperature_2m", 12) is False

    # 2. interpolate_point with is_mean=True reads mean shard (offset 50)
    # in chunk 0 (base 0), value should be 50.0
    val_mean = reader.interpolate_point(
        "temperature_2m",
        member=None,
        lead_time_hours=6,
        lat_idx=[10, 11],
        lon_idx=[20, 21],
        t_row=0.5,
        t_col=0.5,
        is_mean=True,
    )
    assert val_mean == 50.0

    # 3. interpolate_point with member=1 reads member shard (offset 10)
    val_mem = reader.interpolate_point(
        "temperature_2m",
        member=1,
        lead_time_hours=6,
        lat_idx=[10, 11],
        lon_idx=[20, 21],
        t_row=0.5,
        t_col=0.5,
        is_mean=False,
    )
    assert val_mem == 10.0

    # 4. read_window with is_mean=True reads mean shard
    win_mean = reader.read_window(
        "temperature_2m",
        member=None,
        lead_time_hours=6,
        lat_min=0,
        lat_max=2,
        lon_min=0,
        lon_max=2,
        is_mean=True,
    )
    assert win_mean.shape == (3, 3)
    assert win_mean[0, 0] == 50.0
