"""Unit and integration tests for ShardedV1Reader and dual-reader dispatch in API service."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from numcodecs import Zstd

from api.core.zarr import (
    INDEX_ENTRY_SIZE,
    SHARD_MAGIC,
    ShardedV1Reader,
)


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
