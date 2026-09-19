"""Tests for aggregate shard containers (ingestion/core/aggregate_writer.py).

The container is the boundary between the writer and every reader, so these tests pin the
addressing arithmetic, the round trip at chunk granularity, and the failures that must be
loud -- a reader that recovers the wrong layout would decode a statistic plane as a
different field and look plausible.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from domain.aggregate import AggregateSpec, compute_aggregate
from domain.shard_format import SHARD_V1_MAGIC, SHARD_V2_MAGIC
from ingestion.core.aggregate_writer import (
    AGGREGATE_SHARD_SUFFIX,
    DEFAULT_CHUNK_LAT,
    DEFAULT_CHUNK_LON,
    AggregateShardLayout,
    AggregateWriterError,
    aggregate_store_relative_key,
    chunk_ordinals_for_field,
    decode_aggregate_chunk,
    encode_aggregate_shard,
    field_metadata,
    layout_for_spec,
    layout_from_descriptor,
)

GRID_LAT, GRID_LON = 721, 1440


def _identical_allow_nan(left: np.ndarray, right: np.ndarray) -> bool:
    """Exact equality treating NaN as equal to NaN.

    ``np.array_equal`` reports False for two arrays with NaN in the same positions, which
    would make every padded chunk look like a mismatch.
    """
    return bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(left[~np.isnan(left)], right[~np.isnan(right)])
    )


def _layout(n_fields: int = 3, **overrides: int) -> AggregateShardLayout:
    kwargs = {
        "n_fields": n_fields,
        "grid_lat": GRID_LAT,
        "grid_lon": GRID_LON,
        "chunk_lat": DEFAULT_CHUNK_LAT,
        "chunk_lon": DEFAULT_CHUNK_LON,
    }
    kwargs.update(overrides)
    return AggregateShardLayout(**kwargs)  # type: ignore[arg-type]


def _planes(n_fields: int = 3, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.normal(280.0, 8.0, (GRID_LAT, GRID_LON)).astype(np.float32)
        for _ in range(n_fields)
    ]


# ---------------------------------------------------------------------------
# Layout arithmetic
# ---------------------------------------------------------------------------


def test_layout_chunk_geometry_matches_the_member_shard_convention() -> None:
    """The aggregate layout reuses the member-shard chunk shape and grid extent.

    Sharing them is what lets the reader keep one geometry path, and it keeps random access
    at exactly one chunk for a point query.
    """
    layout = _layout(n_fields=2)
    assert layout.lat_chunks == 8
    assert layout.lon_chunks == 15
    assert layout.chunks_per_field == 120
    assert layout.num_chunks == 240
    assert layout.to_descriptor().num_chunks == 240


def test_layout_round_trips_through_its_descriptor() -> None:
    """A reader must recover the exact layout from the container alone."""
    for n_fields in (1, 2, 34):
        layout = _layout(n_fields=n_fields)
        assert layout_from_descriptor(layout.to_descriptor()) == layout


def test_layout_handles_a_non_standard_grid() -> None:
    layout = _layout(n_fields=1, grid_lat=180, grid_lon=360, chunk_lat=100, chunk_lon=100)
    assert layout.lat_chunks == 2
    assert layout.lon_chunks == 4
    assert layout.num_chunks == 8
    assert layout_from_descriptor(layout.to_descriptor()) == layout


def test_chunk_ordinal_is_spatial_major_then_field() -> None:
    """One location's fields are contiguous; that is what the point path needs whole.

    The ordering is not cosmetic: a point query reads every stored field at one latitude and
    longitude, so placing those fields together turns its window into four contiguous ranges
    instead of 2 x n_fields of them.
    """
    layout = _layout(n_fields=3)
    assert layout.chunk_ordinal(0, 0, 0) == 0
    assert layout.chunk_ordinal(1, 0, 0) == 1
    assert layout.chunk_ordinal(2, 0, 0) == 2
    # the next location starts after this location's fields
    assert layout.chunk_ordinal(0, 0, 1) == 3
    assert layout.chunk_ordinal(0, 1, 0) == 3 * layout.lon_chunks
    assert layout.chunk_ordinal(2, 7, 14) == layout.num_chunks - 1


def test_chunk_ordinal_rejects_off_grid_positions() -> None:
    layout = _layout(n_fields=2)
    for field, row, col in ((2, 0, 0), (-1, 0, 0), (0, 8, 0), (0, 0, 15), (0, -1, 0), (0, 0, -1)):
        assert layout.chunk_ordinal(field, row, col) == -1


def test_locate_inverts_chunk_ordinal_for_every_chunk() -> None:
    layout = _layout(n_fields=2)
    for field in range(layout.n_fields):
        for row in range(layout.lat_chunks):
            for col in range(layout.lon_chunks):
                ordinal = layout.chunk_ordinal(field, row, col)
                assert layout.locate(ordinal) == (field, row, col)


def test_locate_rejects_out_of_range_ordinals() -> None:
    layout = _layout(n_fields=1)
    for bad in (-1, layout.num_chunks, layout.num_chunks + 5):
        with pytest.raises(AggregateWriterError, match="outside"):
            layout.locate(bad)


def test_layout_rejects_invalid_geometry() -> None:
    with pytest.raises(AggregateWriterError, match="n_fields must be positive"):
        _layout(n_fields=0)
    for name, value in (
        ("grid_lat", 0),
        ("grid_lon", 0),
        ("chunk_lat", 0),
        ("chunk_lon", 0),
    ):
        with pytest.raises(AggregateWriterError, match=f"{name} must be positive"):
            _layout(**{name: value})


def test_layout_from_descriptor_rejects_a_chunk_count_that_is_not_field_aligned() -> None:
    """A num_chunks that does not divide evenly means the field count is unknowable."""
    descriptor = _layout(n_fields=2).to_descriptor()
    broken = descriptor.__class__(**{**descriptor.__dict__, "num_chunks": 241,
                                     "index_byte_size": 241 * 16})
    with pytest.raises(AggregateWriterError, match="not a multiple of"):
        layout_from_descriptor(broken)


def test_layout_for_spec_matches_the_spec_field_count() -> None:
    spec = AggregateSpec(n_bins=8)
    layout = layout_for_spec(spec, grid_lat=GRID_LAT, grid_lon=GRID_LON)
    assert layout.n_fields == spec.n_fields == 10


def test_spatial_group_covers_every_field_at_one_location() -> None:
    """The group a point query consumes whole, in field order."""
    layout = _layout(n_fields=3)
    for row in (0, layout.lat_chunks - 1):
        for col in (0, layout.lon_chunks - 1):
            group = list(layout.spatial_group(row, col))
            assert len(group) == layout.n_fields
            assert group == sorted(group)
            assert [layout.locate(k) for k in group] == [
                (field, row, col) for field in range(layout.n_fields)
            ]
    with pytest.raises(AggregateWriterError, match="outside the"):
        layout.spatial_group(layout.lat_chunks, 0)
    with pytest.raises(AggregateWriterError, match="outside the"):
        layout.spatial_group(0, layout.lon_chunks)


def test_chunk_ordinals_for_field_spans_exactly_one_plane() -> None:
    """Scattered under the spatial-major layout, but still exactly one plane's worth."""
    layout = _layout(n_fields=3)
    for field in range(3):
        ordinals = chunk_ordinals_for_field(layout, field)
        assert len(ordinals) == layout.chunks_per_field
        assert ordinals == sorted(ordinals)
        assert all(layout.locate(k)[0] == field for k in ordinals)
        # the field-to-field stride is the field count, which is what makes it scattered
        if layout.chunks_per_field > 1:
            assert ordinals[1] - ordinals[0] == layout.n_fields
    with pytest.raises(AggregateWriterError, match="outside"):
        chunk_ordinals_for_field(layout, 3)


# ---------------------------------------------------------------------------
# Encoding round trip
# ---------------------------------------------------------------------------


def test_encode_then_decode_every_chunk_bit_for_bit() -> None:
    """Every (field, row, col) chunk must survive encoding unchanged.

    A layout error here would decode a statistic plane as a different field and still look
    like a plausible field of numbers, so the check is per chunk and exact.
    """
    planes = _planes(n_fields=3, seed=1)
    layout = _layout(n_fields=3)
    container = encode_aggregate_shard(planes, layout)

    for ordinal in range(layout.num_chunks):
        field, row, col = layout.locate(ordinal)
        decoded = decode_aggregate_chunk(container, ordinal)
        expected = np.full((layout.chunk_lat, layout.chunk_lon), np.nan, dtype=np.float32)
        r0, c0 = row * layout.chunk_lat, col * layout.chunk_lon
        r1 = min(r0 + layout.chunk_lat, layout.grid_lat)
        c1 = min(c0 + layout.chunk_lon, layout.grid_lon)
        expected[: r1 - r0, : c1 - c0] = planes[field][r0:r1, c0:c1]
        assert _identical_allow_nan(decoded, expected), ordinal


def test_container_is_a_sharded_v2_object_with_the_declared_trailer() -> None:
    layout = _layout(n_fields=2)
    container = encode_aggregate_shard(_planes(n_fields=2, seed=2), layout)
    num_chunks, index_size, magic = struct.unpack("<III", container[-12:])
    assert magic == SHARD_V2_MAGIC
    assert num_chunks == layout.num_chunks
    assert index_size == layout.num_chunks * 16


def test_edge_chunks_are_padded_to_the_full_chunk_extent() -> None:
    """Every chunk carries a fixed-size buffer, so a reader needs no per-chunk shape.

    721 = 7*100 + 21 and 1440 = 14*100 + 40, so the last row and column of chunks are
    partial; the padding is what keeps their decoded shape identical to the interior.
    """
    planes = _planes(n_fields=1, seed=3)
    layout = _layout(n_fields=1)
    container = encode_aggregate_shard(planes, layout)
    corner = layout.chunk_ordinal(0, layout.lat_chunks - 1, layout.lon_chunks - 1)
    decoded = decode_aggregate_chunk(container, corner)
    assert decoded.shape == (layout.chunk_lat, layout.chunk_lon)
    assert np.isnan(decoded[21:, :]).all()
    assert np.isnan(decoded[:, 40:]).all()
    assert np.isfinite(decoded[:21, :40]).all()


def test_nan_cells_survive_the_round_trip() -> None:
    layout = _layout(n_fields=1)
    plane = _planes(n_fields=1, seed=4)[0]
    plane[0:60, 0:60] = np.nan
    container = encode_aggregate_shard([plane], layout)
    decoded = decode_aggregate_chunk(container, 0)
    assert np.isnan(decoded[:60, :60]).all()
    assert np.isfinite(decoded[60:, 60:]).all()


def test_encode_rejects_a_plane_count_that_disagrees_with_the_layout() -> None:
    with pytest.raises(AggregateWriterError, match="planes but layout declares"):
        encode_aggregate_shard(_planes(n_fields=2), _layout(n_fields=3))


def test_encode_rejects_a_plane_with_the_wrong_shape() -> None:
    layout = _layout(n_fields=1)
    wrong = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(AggregateWriterError, match="expected"):
        encode_aggregate_shard([wrong], layout)


def test_decode_rejects_a_non_v2_container() -> None:
    """A v1 object must not be decoded as an aggregate, whatever its bytes look like."""
    layout = _layout(n_fields=1)
    container = bytearray(encode_aggregate_shard(_planes(n_fields=1, seed=5), layout))
    struct.pack_into("<I", container, len(container) - 4, SHARD_V1_MAGIC)
    with pytest.raises(AggregateWriterError, match="not a sharded_v2 container"):
        decode_aggregate_chunk(bytes(container), 0)


def test_decode_rejects_a_container_too_short_to_hold_a_descriptor() -> None:
    with pytest.raises(AggregateWriterError, match="too short"):
        decode_aggregate_chunk(b"\x00" * 20, 0)


def test_decode_rejects_a_declared_index_that_exceeds_the_container() -> None:
    """A corrupted index size must fail rather than slice from a negative offset."""
    layout = _layout(n_fields=1)
    container = bytearray(encode_aggregate_shard(_planes(n_fields=1, seed=6), layout))
    struct.pack_into("<I", container, len(container) - 8, 1 << 30)
    with pytest.raises(AggregateWriterError, match="exceeds the container length"):
        decode_aggregate_chunk(bytes(container), 0)


def test_decode_rejects_an_out_of_range_ordinal() -> None:
    layout = _layout(n_fields=1)
    container = encode_aggregate_shard(_planes(n_fields=1, seed=7), layout)
    with pytest.raises(AggregateWriterError, match="outside"):
        decode_aggregate_chunk(container, layout.num_chunks)


# ---------------------------------------------------------------------------
# Key and metadata
# ---------------------------------------------------------------------------


def test_aggregate_key_follows_the_shard_key_grammar() -> None:
    key = aggregate_store_relative_key("temperature_2m", 6)
    assert key == "temperature_2m/shard.agg_L0006.shard"
    assert AGGREGATE_SHARD_SUFFIX in key


def test_field_metadata_records_only_the_parameters_that_apply() -> None:
    """A bin parameter on a quantile shard would be a field a reader could mistrust."""
    from domain.aggregate import KIND_QUANTILE_FUNCTION

    bins_spec = AggregateSpec(n_bins=8)
    bins_layout = layout_for_spec(bins_spec, grid_lat=GRID_LAT, grid_lon=GRID_LON)
    bins_meta = field_metadata(bins_spec, bins_layout)
    assert bins_meta["n_bins"] == 8
    assert bins_meta["sigma_range"] == bins_spec.sigma_range
    assert "levels" not in bins_meta

    quantile_spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    quantile_layout = layout_for_spec(
        quantile_spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    quantile_meta = field_metadata(quantile_spec, quantile_layout)
    assert quantile_meta["levels"] == list(quantile_spec.levels)
    assert "n_bins" not in quantile_meta
    assert "sigma_range" not in quantile_meta


def test_field_metadata_describes_the_spec_and_layout_completely() -> None:
    """A reader must be able to interpret the container from the recorded metadata."""
    spec = AggregateSpec(n_bins=8)
    layout = layout_for_spec(spec, grid_lat=GRID_LAT, grid_lon=GRID_LON)
    meta = field_metadata(spec, layout)

    assert meta["kind"] == spec.kind
    assert meta["n_fields"] == spec.n_fields
    assert meta["n_bins"] == 8
    assert meta["field_names"] == list(spec.field_names)
    assert meta["field_scales"] == list(spec.field_scales)
    assert meta["chunks_per_field"] == layout.chunks_per_field
    assert meta["container_format"] == "sharded_v2"
    assert meta["grid_lat"] == GRID_LAT
    assert meta["grid_lon"] == GRID_LON


def test_decode_does_not_depend_on_the_write_level() -> None:
    """A zstd frame is self-describing, so no level has to be negotiated with the reader."""
    planes = _planes(n_fields=1, seed=9)
    layout = _layout(n_fields=1)
    for level in (1, 5, 19):
        container = encode_aggregate_shard(planes, layout, level=level)
        decoded = decode_aggregate_chunk(container, 0)
        expected = np.full((layout.chunk_lat, layout.chunk_lon), np.nan, dtype=np.float32)
        expected[:100, :100] = planes[0][:100, :100]
        assert _identical_allow_nan(decoded, expected), level


def test_container_from_a_realistic_spec_round_trips() -> None:
    """End to end through the actual encode path with the default field set."""
    spec = AggregateSpec()
    stack = np.random.default_rng(8).normal(280.0, 8.0, (30, 64, 64)).astype(np.float32)
    fields = compute_aggregate(stack, spec)
    layout = AggregateShardLayout(
        n_fields=spec.n_fields, grid_lat=64, grid_lon=64, chunk_lat=64, chunk_lon=64
    )
    container = encode_aggregate_shard(list(fields), layout)
    decoded = decode_aggregate_chunk(container, 0)
    assert _identical_allow_nan(decoded, fields[0])
