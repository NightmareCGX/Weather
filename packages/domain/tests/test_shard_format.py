"""Unit tests for the shard container format (packages/domain/src/domain/shard_format.py).

The format is the one piece of the storage stack that cannot be fixed after the fact: a
wrong container is unreadable data. These tests therefore pin the byte layout, the
validation that makes a wrong container fail loudly, and the round trip -- including the
adversarial cases where a plausible-looking but inconsistent container must be rejected.
"""

from __future__ import annotations

import struct

import pytest
from domain.shard_format import (
    DESCRIPTOR_SIZE,
    ENCODING_F32,
    INDEX_ENTRY_SIZE,
    SHARD_V1_MAGIC,
    SHARD_V2_MAGIC,
    TAIL_PROBE_SIZE,
    TRAILER_SIZE,
    ShardDescriptor,
    ShardFormatError,
    build_container_v2,
    build_descriptor,
    build_trailer,
    container_tail_size,
    extract_chunk,
    parse_descriptor,
    parse_index,
    parse_trailer,
    split_v2_tail,
    v1_container_tail_size,
)

GRID_LAT, GRID_LON = 721, 1440
CHUNK = 100
LAT_CHUNKS = 8
LON_CHUNKS = 15
NUM_CHUNKS = LAT_CHUNKS * LON_CHUNKS


def _descriptor(**overrides: object) -> ShardDescriptor:
    base: dict[str, object] = {
        "encoding_id": ENCODING_F32,
        "scale": 1.0,
        "chunk_lat": CHUNK,
        "chunk_lon": CHUNK,
        "grid_lat": GRID_LAT,
        "grid_lon": GRID_LON,
        "num_chunks": NUM_CHUNKS,
        "index_byte_size": NUM_CHUNKS * INDEX_ENTRY_SIZE,
    }
    base.update(overrides)
    return ShardDescriptor(**base)  # type: ignore[arg-type]


def _chunks(seed: bytes = b"c") -> list[bytes]:
    return [seed * (i + 1) for i in range(NUM_CHUNKS)]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_descriptor_and_trailer_sizes_are_exact() -> None:
    """The tail probe the reader performs depends on these being the declared widths."""
    assert DESCRIPTOR_SIZE == 40
    assert TRAILER_SIZE == 12
    assert TAIL_PROBE_SIZE == DESCRIPTOR_SIZE + TRAILER_SIZE == 52
    assert len(build_descriptor(_descriptor())) == DESCRIPTOR_SIZE
    assert len(build_trailer(NUM_CHUNKS, NUM_CHUNKS * INDEX_ENTRY_SIZE)) == TRAILER_SIZE


def test_container_tail_sizes_include_the_descriptor_only_for_v2() -> None:
    assert container_tail_size(NUM_CHUNKS) == (
        NUM_CHUNKS * INDEX_ENTRY_SIZE + DESCRIPTOR_SIZE + TRAILER_SIZE
    )
    assert v1_container_tail_size(NUM_CHUNKS) == (
        NUM_CHUNKS * INDEX_ENTRY_SIZE + TRAILER_SIZE
    )
    assert container_tail_size(NUM_CHUNKS) - v1_container_tail_size(NUM_CHUNKS) == DESCRIPTOR_SIZE


def test_container_layout_is_payload_index_descriptor_trailer() -> None:
    """Object order is load-bearing: the reader derives offsets from the tail inward."""
    chunks = _chunks()
    container = build_container_v2(chunks, descriptor=_descriptor())
    payload_len = sum(len(c) for c in chunks)

    assert struct.unpack("<III", container[-TRAILER_SIZE:]) == (
        NUM_CHUNKS,
        NUM_CHUNKS * INDEX_ENTRY_SIZE,
        SHARD_V2_MAGIC,
    )
    descriptor = parse_descriptor(container[-(TRAILER_SIZE + DESCRIPTOR_SIZE) : -TRAILER_SIZE])
    assert descriptor.num_chunks == NUM_CHUNKS

    index_start = payload_len
    index_bytes = container[index_start : index_start + NUM_CHUNKS * INDEX_ENTRY_SIZE]
    entries = parse_index(index_bytes, NUM_CHUNKS)
    assert entries[0] == (0, len(chunks[0]))
    offset = 0
    for entry, chunk in zip(entries, chunks, strict=True):
        assert entry == (offset, len(chunk))
        assert extract_chunk(container, *entry) == chunk
        offset += len(chunk)


def test_zero_length_entries_are_recorded_for_empty_chunks() -> None:
    """An all-fill chunk must cost its index entry but no payload bytes."""
    chunks = _chunks()
    chunks[7] = b""
    chunks[42] = b""
    container = build_container_v2(chunks, descriptor=_descriptor())
    entries = parse_index(
        container[
            sum(len(c) for c in chunks) : sum(len(c) for c in chunks)
            + NUM_CHUNKS * INDEX_ENTRY_SIZE
        ],
        NUM_CHUNKS,
    )
    assert entries[7] == (0, 0)
    assert entries[42] == (0, 0)
    assert extract_chunk(container, *entries[7]) == b""
    # every other chunk still round-trips
    for i, chunk in enumerate(chunks):
        if chunk:
            assert extract_chunk(container, *entries[i]) == chunk


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_descriptor_round_trips_exactly() -> None:
    for descriptor in (
        _descriptor(),
        _descriptor(chunk_lat=64, chunk_lon=64),
        _descriptor(grid_lat=180, grid_lon=360, num_chunks=12,
                    index_byte_size=12 * INDEX_ENTRY_SIZE),
    ):
        assert parse_descriptor(build_descriptor(descriptor)) == descriptor


def test_expected_shape_is_derived_from_geometry_not_assumed() -> None:
    """A non-721x1440 grid must report its own chunk counts.

    Hardcoding 8x15 in the readers is what made a differently-shaped store mis-slice
    silently; the descriptor is the authority.
    """
    assert _descriptor().expected_shape() == (LAT_CHUNKS, LON_CHUNKS)
    assert _descriptor(
        grid_lat=180,
        grid_lon=400,
        num_chunks=8,
        index_byte_size=8 * INDEX_ENTRY_SIZE,
    ).expected_shape() == (2, 4)
    # one cell past a boundary still needs a partial chunk
    assert _descriptor(
        grid_lat=201,
        grid_lon=100,
        num_chunks=3,
        index_byte_size=3 * INDEX_ENTRY_SIZE,
    ).expected_shape() == (3, 1)
    assert _descriptor(
        grid_lat=200,
        grid_lon=100,
        num_chunks=2,
        index_byte_size=2 * INDEX_ENTRY_SIZE,
    ).expected_shape() == (2, 1)


def test_chunk_index_is_row_major_and_rejects_off_grid() -> None:
    descriptor = _descriptor()
    assert descriptor.chunk_index(0, 0) == 0
    assert descriptor.chunk_index(0, 1) == 1
    assert descriptor.chunk_index(1, 0) == LON_CHUNKS
    assert descriptor.chunk_index(LAT_CHUNKS - 1, LON_CHUNKS - 1) == NUM_CHUNKS - 1
    for bad in ((LAT_CHUNKS, 0), (0, LON_CHUNKS), (-1, 0), (0, -1)):
        assert descriptor.chunk_index(*bad) == -1


def test_split_v2_tail_returns_index_and_descriptor() -> None:
    chunks = _chunks()
    container = build_container_v2(chunks, descriptor=_descriptor())
    tail = container[-container_tail_size(NUM_CHUNKS) :]
    index_bytes, descriptor = split_v2_tail(tail, NUM_CHUNKS)
    assert descriptor.num_chunks == NUM_CHUNKS
    payload_offset = sum(len(c) for c in chunks[:3])
    assert parse_index(index_bytes, NUM_CHUNKS)[3] == (payload_offset, len(chunks[3]))


# ---------------------------------------------------------------------------
# Validation: every failure below used to be silent
# ---------------------------------------------------------------------------


def test_parse_trailer_accepts_both_generations_and_labels_them() -> None:
    v2 = parse_trailer(build_trailer(NUM_CHUNKS, NUM_CHUNKS * INDEX_ENTRY_SIZE))
    assert v2.is_v2 and not v2.is_v1
    v1 = parse_trailer(
        struct.pack("<III", NUM_CHUNKS, NUM_CHUNKS * INDEX_ENTRY_SIZE, SHARD_V1_MAGIC)
    )
    assert v1.is_v1 and not v1.is_v2


def test_parse_trailer_rejects_unknown_magic() -> None:
    """The v1 reader treated the magic as decorative; that is what made drift silent."""
    with pytest.raises(ShardFormatError, match="unrecognized container magic"):
        parse_trailer(struct.pack("<III", NUM_CHUNKS, NUM_CHUNKS * INDEX_ENTRY_SIZE, 0xDEADBEEF))


def test_parse_trailer_rejects_inconsistent_index_size() -> None:
    with pytest.raises(ShardFormatError, match="index_byte_size"):
        parse_trailer(struct.pack("<III", NUM_CHUNKS, 1234, SHARD_V2_MAGIC))


def test_parse_trailer_rejects_truncated_input() -> None:
    with pytest.raises(ShardFormatError, match="trailer truncated"):
        parse_trailer(b"\x00" * (TRAILER_SIZE - 1))


def test_parse_descriptor_rejects_wrong_magic_and_version() -> None:
    raw = bytearray(build_descriptor(_descriptor()))
    with pytest.raises(ShardFormatError, match="bad descriptor magic"):
        struct.pack_into("<I", raw, 0, 0x11111111)
        parse_descriptor(bytes(raw))

    raw = bytearray(build_descriptor(_descriptor()))
    struct.pack_into("<B", raw, 4, 99)
    with pytest.raises(ShardFormatError, match="unsupported container format version"):
        parse_descriptor(bytes(raw))


def test_parse_descriptor_rejects_unknown_encoding() -> None:
    with pytest.raises(ShardFormatError, match="unknown encoding id"):
        parse_descriptor(build_descriptor(_descriptor(encoding_id=200)))


def test_parse_descriptor_rejects_nonzero_reserved_and_flags() -> None:
    raw = bytearray(build_descriptor(_descriptor()))
    struct.pack_into("<H", raw, 6, 1)
    with pytest.raises(ShardFormatError, match="reserved flags"):
        parse_descriptor(bytes(raw))

    raw = bytearray(build_descriptor(_descriptor()))
    struct.pack_into("<8s", raw, 32, b"\x01" * 8)
    with pytest.raises(ShardFormatError, match="reserved descriptor bytes"):
        parse_descriptor(bytes(raw))


def test_parse_descriptor_rejects_impossible_geometry() -> None:
    with pytest.raises(ShardFormatError, match="chunk extents must be positive"):
        parse_descriptor(build_descriptor(_descriptor(chunk_lat=0)))
    with pytest.raises(ShardFormatError, match="grid extents must be positive"):
        parse_descriptor(build_descriptor(_descriptor(grid_lat=0)))
    # num_chunks is unsigned on the wire, so a negative count cannot be serialized at
    # all -- the boundary is the serializer, not the parser.
    with pytest.raises(struct.error):
        build_descriptor(_descriptor(num_chunks=-1, index_byte_size=-INDEX_ENTRY_SIZE))


def test_parse_descriptor_rejects_inconsistent_num_chunks_and_index_size() -> None:
    with pytest.raises(ShardFormatError, match="index_byte_size"):
        parse_descriptor(build_descriptor(_descriptor(index_byte_size=16)))


def test_parse_descriptor_rejects_a_scale_that_would_rescale_values() -> None:
    """A container declaring a scale while carrying floats silently rescales on read."""
    with pytest.raises(ShardFormatError, match="must declare scale 1.0"):
        parse_descriptor(build_descriptor(_descriptor(scale=0.01)))


def test_parse_descriptor_rejects_truncated_input() -> None:
    with pytest.raises(ShardFormatError, match="descriptor truncated"):
        parse_descriptor(b"\x00" * (DESCRIPTOR_SIZE - 1))


def test_split_v2_tail_rejects_truncation_and_chunk_count_disagreement() -> None:
    container = build_container_v2(_chunks(), descriptor=_descriptor())
    tail = container[-container_tail_size(NUM_CHUNKS) :]
    with pytest.raises(ShardFormatError, match="v2 tail truncated"):
        split_v2_tail(tail[:-1], NUM_CHUNKS)
    with pytest.raises(ShardFormatError, match="disagrees with trailer"):
        split_v2_tail(tail, NUM_CHUNKS - 1)


def test_build_container_rejects_chunk_count_disagreement() -> None:
    with pytest.raises(ShardFormatError, match="chunk payloads but descriptor declares"):
        build_container_v2(_chunks()[:-1], descriptor=_descriptor())


def test_build_container_rejects_inconsistent_descriptor_index_size() -> None:
    with pytest.raises(ShardFormatError, match="index_byte_size"):
        build_container_v2(_chunks(), descriptor=_descriptor(index_byte_size=8))


def test_parse_index_rejects_truncation() -> None:
    with pytest.raises(ShardFormatError, match="index truncated"):
        parse_index(b"\x00" * (INDEX_ENTRY_SIZE - 1), 1)


def test_extract_chunk_rejects_entry_past_the_payload() -> None:
    """An index that points beyond the payload must fail, not return a short buffer."""
    container = build_container_v2(_chunks(), descriptor=_descriptor())
    with pytest.raises(ShardFormatError, match="runs past the payload region"):
        extract_chunk(container, 0, len(container))


def test_zero_length_entry_needs_no_bounds_check() -> None:
    container = build_container_v2(_chunks(), descriptor=_descriptor())
    assert extract_chunk(container, 0, 0) == b""


def test_encoding_name_is_reported_for_known_and_unknown_ids() -> None:
    assert _descriptor().encoding_name == "f32"
    # an unknown id cannot reach a parsed descriptor, but the property must not lie
    assert ShardDescriptor(
        encoding_id=200,
        scale=1.0,
        chunk_lat=1,
        chunk_lon=1,
        grid_lat=1,
        grid_lon=1,
        num_chunks=0,
        index_byte_size=0,
    ).encoding_name == "unknown(200)"


def test_format_version_constant_matches_the_descriptor_byte() -> None:
    """The manifest's format string and the descriptor's byte must not diverge."""
    from domain.shard_format import FORMAT_VERSION_BYTE, FORMAT_VERSION_V2

    assert f"sharded_v{FORMAT_VERSION_BYTE}" == FORMAT_VERSION_V2
    descriptor_bytes = build_descriptor(_descriptor())
    assert struct.unpack_from("<B", descriptor_bytes, 4)[0] == FORMAT_VERSION_BYTE
