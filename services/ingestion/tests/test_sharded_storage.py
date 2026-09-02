"""Unit and integration tests for Weather Platform Sharded v1 (sharded_v1) storage format."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Zstd

from ingestion.core.inventory import (
    region_expected_object_keys,
    verify_shard_integrity,
)
from ingestion.core.zarr_writer import (
    INDEX_ENTRY_SIZE,
    SHARD_MAGIC,
    TRAILER_SIZE,
    build_sharded_v1_container,
    encode_region_sharded_v1,
    parse_sharded_v1_index,
)


def _synthetic_region_dataset(lead: int = 6, member: int | None = 1) -> xr.Dataset:
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    dims = ("latitude", "longitude")
    shape = (721, 1440)

    data_vars = {
        "temperature_2m": (dims, np.full(shape, 290.0, dtype=np.float32)),
        "precipitation_rate": (dims, np.full(shape, 2.5, dtype=np.float32)),
    }
    coords = {
        "lead_time_hours": [lead],
        "latitude": lat,
        "longitude": lon,
        "time": np.datetime64("2026-07-21T00:00:00"),
    }
    if member is not None:
        coords["member"] = [member]
    return xr.Dataset(data_vars=data_vars, coords=coords)


def test_build_and_parse_sharded_v1_container() -> None:
    """Test packing and parsing of sharded_v1 container format."""
    compressor = Zstd(level=5)
    raw_chunks = [
        compressor.encode(np.full((1, 1, 100, 100), float(i), dtype=np.float32).tobytes())
        for i in range(120)
    ]

    container_bytes = build_sharded_v1_container(raw_chunks)
    assert len(container_bytes) > 120 * INDEX_ENTRY_SIZE + TRAILER_SIZE

    # Inspect trailer
    trailer = container_bytes[-TRAILER_SIZE:]
    num_chunks, index_size, magic = struct.unpack("<III", trailer)
    assert magic == SHARD_MAGIC
    assert num_chunks == 120
    assert index_size == 120 * INDEX_ENTRY_SIZE

    # Inspect index
    index_bytes = container_bytes[-(TRAILER_SIZE + index_size) : -TRAILER_SIZE]
    entries = parse_sharded_v1_index(index_bytes, num_chunks)
    assert len(entries) == 120

    # Verify each chunk roundtrip
    for idx, (off, length) in enumerate(entries):
        assert off >= 0
        assert length == len(raw_chunks[idx])
        chunk_data = container_bytes[off : off + length]
        assert chunk_data == raw_chunks[idx]
        decoded = compressor.decode(chunk_data)
        arr = np.frombuffer(decoded, dtype=np.float32).reshape(1, 1, 100, 100)
        assert arr[0, 0, 0, 0] == float(idx)


def test_encode_region_sharded_v1() -> None:
    """Test encoding single-region dataset into 1 shard object per variable."""
    ds = _synthetic_region_dataset(lead=6, member=1)
    shards = encode_region_sharded_v1(ds, member=1, lead_time_hours=6)

    # 2 variables in synthetic dataset -> exactly 2 shard files
    assert len(shards) == 2
    keys = [k for k, _ in shards]
    assert "temperature_2m/shard.mem001_L0006.shard" in keys
    assert "precipitation_rate/shard.mem001_L0006.shard" in keys

    # Deterministic model test (member=None)
    ds_det = _synthetic_region_dataset(lead=12, member=None)
    shards_det = encode_region_sharded_v1(ds_det, member=None, lead_time_hours=12)
    assert len(shards_det) == 2
    keys_det = [k for k, _ in shards_det]
    assert "temperature_2m/shard.det_L0012.shard" in keys_det
    assert "precipitation_rate/shard.det_L0012.shard" in keys_det


def test_region_expected_object_keys_sharded() -> None:
    """Test that region_expected_object_keys derives 14 shard objects under sharded_v1."""
    vars_list = [
        "temperature_2m", "precipitation_rate", "precipitation_amount_3h",
        "crain", "csnow", "cfrzr", "cicep", "relative_humidity_2m",
        "wind_gust", "visibility", "snow_depth", "wind_u_10m", "wind_v_10m",
        "cloud_cover_3h",
    ]
    keys = region_expected_object_keys(
        "s3://weather-data/test/cycle.zarr",
        member=17,
        lead_index=2,
        lead_time_hours=6,
        data_var_paths=vars_list,
        format_version="sharded_v1",
    )
    assert len(keys) == 14
    assert "cfrzr/shard.mem017_L0006.shard" in keys
    assert "cicep/shard.mem017_L0006.shard" in keys
    assert "temperature_2m/shard.mem017_L0006.shard" in keys


def test_verify_shard_integrity(tmp_path: Path) -> None:
    """Test structural integrity validation for sharded_v1 objects."""
    compressor = Zstd(level=5)
    raw_chunks = [compressor.encode(np.zeros((1, 1, 100, 100), dtype=np.float32).tobytes()) for _ in range(120)]
    valid_shard = build_sharded_v1_container(raw_chunks)

    shard_file = tmp_path / "temperature_2m" / "shard.mem001_L0006.shard"
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    shard_file.write_bytes(valid_shard)

    store_path = str(tmp_path)
    shard_key = "temperature_2m/shard.mem001_L0006.shard"

    # Valid shard
    assert verify_shard_integrity(store_path, shard_key, expected_num_chunks=120) is True

    # Corrupted trailer magic
    corrupted = bytearray(valid_shard)
    corrupted[-4:] = b"BAD!"
    shard_file.write_bytes(bytes(corrupted))
    assert verify_shard_integrity(store_path, shard_key, expected_num_chunks=120) is False

    # Short/truncated file
    shard_file.write_bytes(valid_shard[:1000])
    assert verify_shard_integrity(store_path, shard_key, expected_num_chunks=120) is False
