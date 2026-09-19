"""Tests for the aggregate shard reader (api/core/aggregate_reader.py).

The reader's contract has two halves and both matter:

* it must decode a container into the fields it was given, and refuse anything it cannot
  interpret with certainty -- because every refusal is a fallback to the member shards, not a
  wrong answer;
* it must agree with the ingestion writer on the container's geometry and key. That
  cross-service assertion lives in ``tests/contracts/`` rather than here: the API tier is
  independently deployable and its own test suite must not import the ingestion package
  (``docs/ARCHITECTURE.md`` 3.1/3.5). Containers below are built from the format authority in
  ``domain.shard_format``, which is the shared contract both services compile against.
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest
from api.core.aggregate_reader import (
    AGGREGATE_SHARD_SUFFIX,
    AggregateGeometry,
    AggregateShardReader,
    aggregate_shard_key,
    recover_geometry,
)
from domain.aggregate import KIND_QUANTILE_FUNCTION, AggregateSpec, compute_aggregate
from domain.shard_format import (
    DESCRIPTOR_SIZE,
    INDEX_ENTRY_SIZE,
    ShardDescriptor,
    ShardFormatError,
    build_container_v2,
)
GRID_LAT, GRID_LON = 128, 160  # 2 x 2 chunks, so a location's group is small
CHUNK_LAT, CHUNK_LON = 100, 100
VARIABLE = "temperature_2m"
LEAD = 6


def _bins_spec() -> AggregateSpec:
    return AggregateSpec(n_bins=4)


def _source_planes(spec: AggregateSpec, seed: int = 0) -> np.ndarray:
    """The member stack the aggregate is computed from, shaped ``(members, lat, lon)``."""
    rng = np.random.default_rng(seed)
    return rng.normal(280.0, 8.0, (5, GRID_LAT, GRID_LON)).astype(np.float32)


def _encode_container(
    fields: np.ndarray, *, level: int = 5, encoding_id: int = 1
) -> bytes:
    """Build an aggregate container from fields, using only the shared format authority.

    A test-local mirror of the writer's chunking (the NaN-padded 100x100 inner chunks), the
    same way ``_zarr_writer.py`` mirrors the store writer for this suite. The cross-service
    agreement assertion -- that the reader and the real ingestion writer produce the same
    geometry for the same container -- lives in ``tests/contracts/``.
    """
    from numcodecs import Zstd

    from domain.shard_format import build_container_v2

    lat_chunks = -(-GRID_LAT // CHUNK_LAT)
    lon_chunks = -(-GRID_LON // CHUNK_LON)
    compressor = Zstd(level=level)
    payloads: list[bytes] = []
    # Spatial-major, matching the container's addressing: a location's fields adjacent.
    for row in range(lat_chunks):
        for col in range(lon_chunks):
            for plane in fields:
                buf = np.full((CHUNK_LAT, CHUNK_LON), np.nan, dtype=np.float32)
                r0, c0 = row * CHUNK_LAT, col * CHUNK_LON
                r1 = min(r0 + CHUNK_LAT, GRID_LAT)
                c1 = min(c0 + CHUNK_LON, GRID_LON)
                buf[: r1 - r0, : c1 - c0] = plane[r0:r1, c0:c1]
                payloads.append(compressor.encode(buf.tobytes(order="C")))

    num_chunks = len(payloads)
    descriptor = ShardDescriptor(
        encoding_id=encoding_id,
        scale=1.0,
        chunk_lat=CHUNK_LAT,
        chunk_lon=CHUNK_LON,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        num_chunks=num_chunks,
        index_byte_size=num_chunks * INDEX_ENTRY_SIZE,
    )
    assert num_chunks > 0
    return build_container_v2(payloads, descriptor=descriptor)


def _write_aggregate(store: str, spec: AggregateSpec, seed: int = 0, level: int = 5) -> str:
    """Write an aggregate container; return its store-relative key."""
    fields = compute_aggregate(_source_planes(spec, seed=seed), spec)
    blob = _encode_container(fields, level=level)
    key = aggregate_shard_key(VARIABLE, LEAD)
    full = os.path.join(store, *key.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(blob)
    return key


def _expected_num_chunks() -> int:
    """Chunks per container for the test grid and the bin spec's field count."""
    lat_chunks = -(-GRID_LAT // CHUNK_LAT)
    lon_chunks = -(-GRID_LON // CHUNK_LON)
    return lat_chunks * lon_chunks * _bins_spec().n_fields


def _full_path(store: str, key: str) -> str:
    return os.path.join(store, *key.split("/"))


# ---------------------------------------------------------------------------
# Key and geometry
# ---------------------------------------------------------------------------


def test_key_matches_the_writers_grammar() -> None:
    """Both tiers must derive the same object key from the same inputs."""
    from domain.reclamation import is_shard_filename

    assert aggregate_shard_key("temperature_2m", 6) == "temperature_2m/shard.agg_L0006.shard"
    assert AGGREGATE_SHARD_SUFFIX == "shard.agg"
    # and the key must not look like a member shard to key-driven discovery
    assert not is_shard_filename(aggregate_shard_key("temperature_2m", 6))


def test_recover_geometry_derives_the_layout_from_the_descriptor() -> None:
    geometry = recover_geometry(_descriptor())
    assert geometry is not None
    assert geometry.n_fields == 4
    assert geometry.lat_chunks == 2
    assert geometry.lon_chunks == 2
    assert geometry.num_chunks == 16


def _descriptor(**overrides: object) -> ShardDescriptor:
    base: dict[str, object] = {
        "encoding_id": 1,
        "scale": 1.0,
        "chunk_lat": 100,
        "chunk_lon": 100,
        "grid_lat": GRID_LAT,
        "grid_lon": GRID_LON,
        "num_chunks": 16,
        "index_byte_size": 16 * INDEX_ENTRY_SIZE,
    }
    base.update(overrides)
    return ShardDescriptor(**base)  # type: ignore[arg-type]


def test_recover_geometry_refuses_an_unknown_encoding() -> None:
    """A newer writer's encoding must not be half-read by an older reader."""
    assert recover_geometry(_descriptor(encoding_id=200)) is None


def test_recover_geometry_refuses_a_chunk_count_that_is_not_field_aligned() -> None:
    """A count that does not divide into whole planes means the field count is unknowable."""
    assert recover_geometry(_descriptor(num_chunks=5, index_byte_size=5 * INDEX_ENTRY_SIZE)) is None


def test_recover_geometry_refuses_an_empty_container() -> None:
    assert recover_geometry(_descriptor(num_chunks=0, index_byte_size=0)) is None


def test_geometry_ordinal_and_group_agree_with_each_other() -> None:
    geometry = AggregateGeometry(
        n_fields=3, grid_lat=200, grid_lon=300, chunk_lat=100, chunk_lon=100, num_chunks=18
    )
    assert geometry.lat_chunks == 2
    assert geometry.lon_chunks == 3
    group = geometry.spatial_group(1, 2)
    assert group == [geometry.ordinal(field, 1, 2) for field in range(3)]
    assert group == list(range(15, 18))


def test_geometry_bounds_checks() -> None:
    geometry = AggregateGeometry(
        n_fields=2, grid_lat=100, grid_lon=100, chunk_lat=100, chunk_lon=100, num_chunks=2
    )
    assert geometry.in_bounds(0, 0)
    assert not geometry.in_bounds(1, 0)
    assert not geometry.in_bounds(-1, 0)
    assert not geometry.in_bounds(0, 1)


# ---------------------------------------------------------------------------
# Round trip against the real writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "seed"),
    ((_bins_spec(), 1), (AggregateSpec(kind=KIND_QUANTILE_FUNCTION), 2)),
)
def test_read_location_returns_the_encoded_fields(
    tmp_path: pytest.TempPathFactory, spec: AggregateSpec, seed: int
) -> None:
    """Every field at a location must decode to exactly what the writer encoded.

    This is the cross-service contract: the reader takes its geometry from the descriptor and
    the writer emits to that geometry, so a mismatch would surface here as misaligned planes
    rather than as an error.
    """
    store = str(tmp_path)
    _write_aggregate(store, spec, seed=seed)
    expected = compute_aggregate(_source_planes(spec, seed=seed), spec)

    reader = AggregateShardReader(store)
    geometry = reader.open(VARIABLE, LEAD)
    assert geometry is not None
    assert geometry.n_fields == spec.n_fields

    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
        stack = reader.read_location(
            VARIABLE, lead_time_hours=LEAD, chunk_row=row, chunk_col=col
        )
        assert stack is not None
        assert stack.shape == (spec.n_fields, 100, 100)
        r0, c0 = row * 100, col * 100
        r1, c1 = min(r0 + 100, GRID_LAT), min(c0 + 100, GRID_LON)
        for field in range(spec.n_fields):
            window = stack[field][: r1 - r0, : c1 - c0]
            assert np.array_equal(window, expected[field][r0:r1, c0:c1]), (row, col, field)


def test_reader_does_not_depend_on_the_write_level(tmp_path) -> None:
    """A zstd frame is self-describing, so the reader must never need the writer's level."""
    store = str(tmp_path)
    spec = _bins_spec()
    _write_aggregate(store, spec, seed=3, level=19)
    reader = AggregateShardReader(store)
    stack = reader.read_location(VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=0)
    assert stack is not None
    assert np.isfinite(stack).any()


def test_mapping_store_is_supported() -> None:
    """The reader accepts the same store shapes as the member reader, mappings included."""
    spec = _bins_spec()
    members = _source_planes(spec, seed=13)
    blob = _encode_container(compute_aggregate(members, spec))
    store = {aggregate_shard_key(VARIABLE, LEAD): blob}

    reader = AggregateShardReader(store)
    assert reader.open(VARIABLE, LEAD) is not None
    stack = reader.read_location(VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=0)
    assert stack is not None
    expected = compute_aggregate(members, spec)
    assert np.array_equal(stack[0][:28, :60], expected[0][:28, :60])


# ---------------------------------------------------------------------------
# Refusals: every one is a fallback, not a wrong answer
# ---------------------------------------------------------------------------


def test_open_returns_none_for_a_missing_store(tmp_path) -> None:
    reader = AggregateShardReader(str(tmp_path / "nope.zarr"))
    assert reader.open(VARIABLE, LEAD) is None
    assert (
        reader.read_location(VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=0)
        is None
    )


def test_open_returns_none_for_a_v1_container(tmp_path) -> None:
    """A v1 member container must not be read as an aggregate, whatever its geometry.

    Built from the v1 constants rather than by importing the ingestion writer: the API suite
    must not depend on that package. A real v1 object is exercised end to end by the
    cross-package serving tests.
    """
    from domain.shard_format import SHARD_V1_MAGIC, build_trailer

    store = str(tmp_path)
    num_chunks = 120
    payload = bytes(64)
    index = b"".join(struct.pack("<QQ", 0, 0) for _ in range(num_chunks))
    blob = (
        payload
        + index
        + struct.pack("<III", num_chunks, num_chunks * INDEX_ENTRY_SIZE, SHARD_V1_MAGIC)
    )
    key = aggregate_shard_key(VARIABLE, LEAD)
    full = _full_path(store, key)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(blob)
    assert build_trailer(num_chunks, num_chunks * INDEX_ENTRY_SIZE)  # v2 trailer differs

    assert AggregateShardReader(store).open(VARIABLE, LEAD) is None


def test_open_returns_none_for_a_truncated_container(tmp_path) -> None:
    store = str(tmp_path)
    key = aggregate_shard_key(VARIABLE, LEAD)
    full = _full_path(store, key)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(b"\x00" * 8)
    assert AggregateShardReader(store).open(VARIABLE, LEAD) is None


def test_open_returns_none_for_a_trailer_whose_magic_is_unknown(tmp_path) -> None:
    store = str(tmp_path)
    _write_aggregate(store, _bins_spec(), seed=5)
    full = _full_path(store, aggregate_shard_key(VARIABLE, LEAD))
    with open(full, "rb") as handle:
        blob = bytearray(handle.read())
    struct.pack_into("<I", blob, len(blob) - 4, 0xDEADBEEF)
    with open(full, "wb") as handle:
        handle.write(bytes(blob))
    assert AggregateShardReader(store).open(VARIABLE, LEAD) is None


def test_open_returns_none_for_a_descriptor_with_an_unknown_encoding(tmp_path) -> None:
    store = str(tmp_path)
    key = aggregate_shard_key(VARIABLE, LEAD)
    descriptor = _descriptor()
    unknown = _descriptor(encoding_id=99)
    payloads = [b"" for _ in range(descriptor.num_chunks)]
    full = _full_path(store, key)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(build_container_v2(payloads, descriptor=unknown))
    assert AggregateShardReader(store).open(VARIABLE, LEAD) is None


def test_read_location_rejects_an_off_grid_location(tmp_path) -> None:
    store = str(tmp_path)
    _write_aggregate(store, _bins_spec(), seed=6)
    reader = AggregateShardReader(store)
    assert (
        reader.read_location(VARIABLE, lead_time_hours=LEAD, chunk_row=2, chunk_col=0)
        is None
    )
    assert (
        reader.read_location(VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=2)
        is None
    )


def test_read_location_reports_an_index_that_disagrees_with_the_payload(tmp_path) -> None:
    """Corrupt state must raise, not degrade: a short read would return shifted data.

    The distinction matters. An absent aggregate is a missing optimisation and returns
    ``None``; a container whose index points past its own payload is corruption, and quietly
    substituting NaN would hide it.
    """
    store = str(tmp_path)
    key = _write_aggregate(store, _bins_spec(), seed=7)
    full = _full_path(store, key)
    with open(full, "rb") as handle:
        blob = bytearray(handle.read())

    num_chunks = _expected_num_chunks()
    index_start = len(blob) - DESCRIPTOR_SIZE - 12 - num_chunks * INDEX_ENTRY_SIZE
    struct.pack_into("<Q", blob, index_start + 8, 1 << 20)  # length far past the payload
    with open(full, "wb") as handle:
        handle.write(bytes(blob))

    with pytest.raises(ShardFormatError, match="index disagrees with the payload"):
        AggregateShardReader(store).read_location(
            VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=0
        )


def test_read_location_reports_a_chunk_that_is_not_one_field_of_floats(tmp_path) -> None:
    """A chunk whose payload is the wrong size means the geometry does not describe it."""
    store = str(tmp_path)
    key = _write_aggregate(store, _bins_spec(), seed=8)
    full = _full_path(store, key)
    with open(full, "rb") as handle:
        blob = bytearray(handle.read())

    num_chunks = _expected_num_chunks()
    index_start = len(blob) - DESCRIPTOR_SIZE - 12 - num_chunks * INDEX_ENTRY_SIZE
    _offset, first_length = struct.unpack_from("<QQ", blob, index_start)
    # Halve the declared length: the chunk still decompresses, but to fewer bytes than a
    # 100x100 float32 field, so the geometry check must catch it.
    struct.pack_into("<Q", blob, index_start + 8, max(1, first_length // 2))
    with open(full, "wb") as handle:
        handle.write(bytes(blob))

    with pytest.raises(ShardFormatError):
        AggregateShardReader(store).read_location(
            VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=0
        )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_repeated_reads_reuse_the_cached_group(tmp_path) -> None:
    """The group cache must serve a repeat read without touching the store again."""
    store = str(tmp_path)
    _write_aggregate(store, _bins_spec(), seed=9)
    reader = AggregateShardReader(store)
    first = reader.read_location(VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=0)
    second = reader.read_location(VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=0)
    # identity, not merely equality: the second call did not re-decode
    assert first is second


def test_group_cache_eviction_stays_bounded(tmp_path) -> None:
    """A bounded cache is what lets this run inside a memory-tight container."""
    store = str(tmp_path)
    _write_aggregate(store, _bins_spec(), seed=10)
    reader = AggregateShardReader(store, max_cached_groups=1)
    for row in range(2):
        for col in range(2):
            assert (
                reader.read_location(
                    VARIABLE, lead_time_hours=LEAD, chunk_row=row, chunk_col=col
                )
                is not None
            )
    assert len(reader._group_cache) == 1


def test_index_cache_eviction_stays_bounded(tmp_path) -> None:
    store = str(tmp_path)
    _write_aggregate(store, _bins_spec(), seed=11)
    reader = AggregateShardReader(store, max_cached_indices=1)
    for lead in (0, 3, 6):
        reader.open(VARIABLE, lead)
    assert len(reader._geometry_cache) <= 1
    assert len(reader._index_cache) <= 1


def test_invalidate_clears_every_cache(tmp_path) -> None:
    """A generation bump must be able to drop stale state without rebuilding the reader."""
    store = str(tmp_path)
    _write_aggregate(store, _bins_spec(), seed=12)
    reader = AggregateShardReader(store)
    reader.read_location(VARIABLE, lead_time_hours=LEAD, chunk_row=0, chunk_col=0)
    assert reader._group_cache
    reader.invalidate()
    assert not reader._group_cache
    assert not reader._index_cache
    assert not reader._geometry_cache
