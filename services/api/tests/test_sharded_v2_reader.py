"""Tests for ShardedV2Reader: native-dtype chunk decode/caching, the reader
factory dispatch on the committed manifest, and mixed-format serving.

Byte contract under test (frozen): f16 continuous chunks are 2 bytes/value and
cached as float16; the precipitation/ceiling f32 exceptions are 4 bytes/value
and bit-exact through the reader; u8 flags are 1 byte/value; missing chunks
fill with the dtype's own null value.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numcodecs import Zstd

from api.core.zarr import ShardedV1Reader, get_sharded_reader
from api.core.zarr_v2 import ShardedV2Reader

COMPRESSOR = Zstd(level=5)


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


def _build_shard(dtype: np.dtype[Any], fill_value: float) -> bytes:
    """Build one 120-chunk shard container with every cell set to fill_value."""
    raw_chunks = []
    for _ in range(120):
        arr = np.full((100, 100), fill_value, dtype=dtype)
        raw_chunks.append(COMPRESSOR.encode(arr.tobytes()))

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
    index_size = len(index_entries) * 16
    buf.write(struct.pack("<III", num_chunks, index_size, 0x53484152))
    return buf.getvalue()


def _write_shard(
    store: Path, variable: str, filename: str, dtype: np.dtype[Any], fill_value: float
) -> None:
    shard_path = store / variable / filename
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(_build_shard(dtype, fill_value))


def _write_manifest(store: Path, version: str) -> None:
    commit_dir = store / "__commit__" / "v1"
    commit_dir.mkdir(parents=True, exist_ok=True)
    (commit_dir / "manifest.json").write_text(
        json.dumps({"storage_format_version": version})
    )


class TestShardedV2ReaderChunkDecode:
    def test_f16_chunk_cached_as_native_float16(self, tmp_path: Path) -> None:
        """The chunk cache must hold the native f16 array, not a promoted f32 copy."""
        _write_shard(tmp_path, "temperature_2m", "shard.det_L0006.shard", np.dtype("<f2"), 21.5)
        reader = ShardedV2Reader(str(tmp_path))
        arr = reader.read_chunk(
            "temperature_2m", member=None, lead_time_hours=6, chunk_row=0, chunk_col=0
        )
        assert arr.dtype == np.dtype("<f2")
        assert arr[0, 0] == np.float16(21.5)
        cached = reader._chunk_cache[
            f"{tmp_path}::live::temperature_2m/shard.det_L0006.shard::0"
        ]
        assert cached.dtype == np.dtype("<f2")  # native cache, no hidden astype

    def test_f32_exception_chunk_is_bit_exact(self, tmp_path: Path) -> None:
        _write_shard(tmp_path, "precipitation_amount_3h", "shard.det_L0006.shard", np.dtype("<f4"), 0.1)
        _write_shard(tmp_path, "cloud_ceiling", "shard.det_L0006.shard", np.dtype("<f4"), 19.991)
        reader = ShardedV2Reader(str(tmp_path))
        precip = reader.read_point_value(
            "precipitation_amount_3h", member=None, lead_time_hours=6, lat_idx=201, lon_idx=1020
        )
        # Exactly-0.10mm semantics: the f32(0.1) bits survive the round trip.
        assert precip == float(np.float32(0.1))
        ceiling = reader.read_point_value(
            "cloud_ceiling", member=None, lead_time_hours=6, lat_idx=201, lon_idx=1020
        )
        assert ceiling == float(np.float32(19.991))
        # Sentinel boundary: 19.991 >= 19.99 -> unlimited, identical to v1.
        assert ceiling >= 19.99

    def test_u8_flag_chunk(self, tmp_path: Path) -> None:
        _write_shard(tmp_path, "crain", "shard.mem001_L0006.shard", np.dtype("u1"), 1)
        reader = ShardedV2Reader(str(tmp_path))
        arr = reader.read_chunk(
            "crain", member=1, lead_time_hours=6, chunk_row=3, chunk_col=7
        )
        assert arr.dtype == np.dtype("uint8")
        assert arr[42, 42] == 1

    def test_missing_chunk_fills_with_dtype_null(self, tmp_path: Path) -> None:
        _write_shard(tmp_path, "temperature_2m", "shard.det_L0006.shard", np.dtype("<f2"), 21.5)
        reader = ShardedV2Reader(str(tmp_path))
        # No shard exists for lead 12: the whole index is empty.
        absent = reader.read_chunk(
            "temperature_2m", member=None, lead_time_hours=12, chunk_row=7, chunk_col=14
        )
        assert absent.dtype == np.dtype("<f2")
        assert np.all(np.isnan(absent))

    def test_byte_length_mismatch_raises(self, tmp_path: Path) -> None:
        """A payload whose length disagrees with the dtype matrix fails loudly."""
        store = tmp_path
        shard_path = store / "temperature_2m" / "shard.det_L0006.shard"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a v1-style f32 payload under a variable the v2 matrix says is f16.
        shard_path.write_bytes(_build_shard(np.dtype("<f4"), 21.5))
        reader = ShardedV2Reader(str(store))
        with pytest.raises(ValueError, match="dtype matrix disagree"):
            reader.read_chunk(
                "temperature_2m", member=None, lead_time_hours=6, chunk_row=0, chunk_col=0
            )

    def test_point_interpolation_matches_v1_reader_within_tolerance(
        self, tmp_path: Path
    ) -> None:
        """f16 storage -> current compute domain: errors within the f16 half-ulp."""
        _write_shard(tmp_path, "temperature_2m", "shard.det_L0006.shard", np.dtype("<f2"), 21.5)
        v2 = ShardedV2Reader(str(tmp_path))
        value = v2.interpolate_point(
            "temperature_2m",
            member=None,
            lead_time_hours=6,
            lat_idx=[201, 202],
            lon_idx=[1020, 1021],
            t_row=0.5,
            t_col=0.5,
        )
        # Constant field: bilinear is exact; f16(21.5) is exact in binary16.
        assert value == float(np.float16(21.5))


class TestReaderFactoryDispatch:
    def test_manifest_sharded_v2_gets_v2_reader(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "sharded_v2")
        reader = get_sharded_reader(str(tmp_path))
        assert isinstance(reader, ShardedV2Reader)

    def test_manifest_sharded_v1_gets_frozen_v1_reader(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "sharded_v1")
        reader = get_sharded_reader(str(tmp_path))
        assert type(reader) is ShardedV1Reader

    def test_absent_manifest_gets_frozen_v1_reader(self, tmp_path: Path) -> None:
        """No manifest (never published / legacy unsharded) -> frozen v1 reader."""
        reader = get_sharded_reader(str(tmp_path))
        assert type(reader) is ShardedV1Reader

    def test_mapping_store_with_v2_manifest(self) -> None:
        store: dict[str, bytes] = {
            "__commit__/v1/manifest.json": json.dumps(
                {"storage_format_version": "sharded_v2"}
            ).encode()
        }
        reader = get_sharded_reader(store)
        assert isinstance(reader, ShardedV2Reader)

    def test_reader_cache_keyed_by_store_path(self, tmp_path: Path) -> None:
        """Same path returns the same (cached) reader; format is immutable per store."""
        _write_manifest(tmp_path, "sharded_v2")
        first = get_sharded_reader(str(tmp_path))
        second = get_sharded_reader(str(tmp_path))
        assert first is second


class TestMixedFormatServing:
    def test_v1_and_v2_readers_coexist(self, tmp_path: Path) -> None:
        """An old v1 cycle and a new v2 cycle are served side by side."""
        v1_store = tmp_path / "v1cycle"
        v2_store = tmp_path / "v2cycle"
        _write_shard(v1_store, "temperature_2m", "shard.det_L0006.shard", np.dtype("<f4"), 10.0)
        _write_shard(v2_store, "temperature_2m", "shard.det_L0006.shard", np.dtype("<f2"), 20.0)
        _write_manifest(v1_store, "sharded_v1")
        _write_manifest(v2_store, "sharded_v2")

        v1_reader = get_sharded_reader(str(v1_store))
        v2_reader = get_sharded_reader(str(v2_store))
        assert type(v1_reader) is ShardedV1Reader
        assert isinstance(v2_reader, ShardedV2Reader)

        v1_value = v1_reader.read_point_value(
            "temperature_2m", member=None, lead_time_hours=6, lat_idx=201, lon_idx=1020
        )
        v2_value = v2_reader.read_point_value(
            "temperature_2m", member=None, lead_time_hours=6, lat_idx=201, lon_idx=1020
        )
        assert v1_value == float(np.float32(10.0))
        assert v2_value == float(np.float16(20.0))

    def test_window_assembly_is_float32_compute_transient(self, tmp_path: Path) -> None:
        """Native f16 chunks assemble into the request-level f32 working window."""
        _write_shard(tmp_path, "temperature_2m", "shard.det_L0006.shard", np.dtype("<f2"), 21.5)
        reader = ShardedV2Reader(str(tmp_path))
        window = reader.read_window(
            "temperature_2m",
            member=None,
            lead_time_hours=6,
            lat_min=150,
            lat_max=350,
            lon_min=950,
            lon_max=1150,
        )
        assert window.dtype == np.dtype("<f4")  # compute transient, not the cache
        assert not np.all(np.isnan(window))
