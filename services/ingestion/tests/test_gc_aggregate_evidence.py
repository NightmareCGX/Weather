"""Tests for the aggregate-evidence probe (ingestion/gc/aggregate_evidence.py).

The probe is what stands between "a container exists" and "a member shard may be deleted", so
every way a container can fail to be evidence is pinned here -- each one is a case where the
alternative is a store holding neither a usable aggregate nor the members it replaced.
"""

from __future__ import annotations

import numpy as np
import pytest
from domain.field_layout import aggregate_fields_for
from domain.shard_format import (
    ENCODING_F32,
    PER_FIELD_SCALE,
    ShardDescriptor,
    build_container_v2,
    build_trailer,
)
from ingestion.core.aggregate_writer import (
    AggregateShardLayout,
    aggregate_store_relative_key,
    encode_aggregate_shard,
)
from ingestion.gc.aggregate_evidence import (
    aggregate_supersedes_members,
    probe_aggregate,
)

GRID_LAT, GRID_LON = 20, 20
CHUNK = 10
LEAD = 6


def _layout(n_fields: int) -> AggregateShardLayout:
    return AggregateShardLayout(
        n_fields=n_fields,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        chunk_lat=CHUNK,
        chunk_lon=CHUNK,
    )


def _planes(n_fields: int) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [rng.normal(0.5, 0.1, (GRID_LAT, GRID_LON)).astype(np.float32) for _ in range(n_fields)]


def _store_with(variable: str, *, scales: tuple[float, ...] | None = None) -> dict[str, bytes]:
    """A one-object store holding a real container written by the production encoder."""
    layout = aggregate_fields_for(variable)
    container = encode_aggregate_shard(
        _planes(layout.n_fields),
        _layout(layout.n_fields),
        member_count=30,
        field_scales=layout.field_scales if scales is None else scales,
    )
    return {aggregate_store_relative_key(variable, LEAD): container}


def test_a_container_written_by_the_writer_is_evidence() -> None:
    """The round trip is the point: the probe reads what the encoder wrote."""
    store = _store_with("temperature_2m")
    evidence = probe_aggregate(store, "temperature_2m", LEAD)
    assert evidence.present is True
    assert evidence.readable is True
    assert evidence.ok is True
    assert evidence.member_count == 30
    assert evidence.detail == ""
    assert aggregate_supersedes_members(store, "temperature_2m", LEAD) is True


def test_the_field_vector_variables_are_all_evidence_too() -> None:
    """Including the group-only ones, whose containers have no distribution at all."""
    for variable in ("crain", "wind_10m", "cloud_ceiling", "precipitation_amount_3h"):
        store = _store_with(variable)
        assert aggregate_supersedes_members(store, variable, LEAD) is True, variable


def test_a_missing_object_is_not_evidence() -> None:
    evidence = probe_aggregate({}, "temperature_2m", LEAD)
    assert evidence.present is False
    assert evidence.readable is False
    assert evidence.ok is False
    assert "cannot read" in evidence.detail
    assert aggregate_supersedes_members({}, "temperature_2m", LEAD) is False


def test_a_wrong_lead_is_a_different_object_and_not_evidence() -> None:
    store = _store_with("temperature_2m")
    assert probe_aggregate(store, "temperature_2m", LEAD + 3).ok is False


def test_a_v1_container_is_not_evidence() -> None:
    """A member shard left at the aggregate key: same key grammar, older container format."""
    payload = b"\x00" * 64
    v1 = payload + build_trailer(2, 2 * 16).replace(b"2VHS", b"RAHS")
    store = {aggregate_store_relative_key("temperature_2m", LEAD): v1}
    evidence = probe_aggregate(store, "temperature_2m", LEAD)
    assert evidence.present is True
    assert evidence.readable is False
    assert "not a sharded_v2 container" in evidence.detail


def test_a_truncated_object_is_not_evidence() -> None:
    store = {aggregate_store_relative_key("temperature_2m", LEAD): b"\x00" * 8}
    evidence = probe_aggregate(store, "temperature_2m", LEAD)
    assert evidence.ok is False
    assert "shorter than a container tail" in evidence.detail


def test_a_container_shorter_than_its_own_index_is_not_evidence() -> None:
    """A trailer claiming more chunks than the object holds: the index is not all there.

    Rewriting only the trailer of a real container is the cheapest way to produce the state a
    half-written or truncated object would have, and the probe has to refuse it: the missing
    bytes may be the descriptor, which is where the encoding and the field count live.
    """
    store = _store_with("temperature_2m")
    key = aggregate_store_relative_key("temperature_2m", LEAD)
    lying = store[key][:-12] + build_trailer(9999, 9999 * 16)
    evidence = probe_aggregate({key: lying}, "temperature_2m", LEAD)
    assert evidence.present is True
    assert evidence.ok is False
    assert "shorter than its own index" in evidence.detail


def test_an_unknown_encoding_is_not_evidence() -> None:
    descriptor = ShardDescriptor(
        encoding_id=99,
        scale=1.0,
        chunk_lat=CHUNK,
        chunk_lon=CHUNK,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        num_chunks=4,
        index_byte_size=4 * 16,
        member_count=30,
    )
    store = {aggregate_store_relative_key("temperature_2m", LEAD): build_container_v2([b"", b"", b"", b""], descriptor=descriptor)}
    evidence = probe_aggregate(store, "temperature_2m", LEAD)
    assert evidence.ok is False
    assert "unknown encoding id 99" in evidence.detail


def test_a_fixed_point_container_without_per_field_scales_is_not_evidence() -> None:
    """A single container-wide step cannot describe fields of different magnitudes."""
    descriptor = ShardDescriptor(
        encoding_id=2,
        scale=0.01,
        chunk_lat=CHUNK,
        chunk_lon=CHUNK,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        num_chunks=4,
        index_byte_size=4 * 16,
        member_count=30,
    )
    store = {aggregate_store_relative_key("temperature_2m", LEAD): build_container_v2([b"", b"", b"", b""], descriptor=descriptor)}
    evidence = probe_aggregate(store, "temperature_2m", LEAD)
    assert evidence.ok is False
    assert "per-field scale marker" in evidence.detail

    # The same container with the marker is fine, which is what pins the marker as the reason.
    marked = build_container_v2(
        [b"", b"", b"", b""],
        descriptor=ShardDescriptor(
            encoding_id=2,
            scale=PER_FIELD_SCALE,
            chunk_lat=CHUNK,
            chunk_lon=CHUNK,
            grid_lat=GRID_LAT,
            grid_lon=GRID_LON,
            num_chunks=4,
            index_byte_size=4 * 16,
            member_count=30,
        ),
    )
    # Four chunks over a 2x2 chunk grid is one field plane, and temperature_2m declares 35.
    evidence = probe_aggregate(
        {aggregate_store_relative_key("temperature_2m", LEAD): marked}, "temperature_2m", LEAD
    )
    assert evidence.ok is False
    assert "field count 1 does not match 35" in evidence.detail


def test_a_chunk_count_that_is_not_whole_field_planes_is_not_evidence() -> None:
    """The geometry check the API's reader also applies, for the same reason."""
    descriptor = ShardDescriptor(
        encoding_id=ENCODING_F32,
        scale=1.0,
        chunk_lat=CHUNK,
        chunk_lon=CHUNK,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        num_chunks=5,  # 2x2 = 4 chunks per plane, so 5 is one plane plus a fragment
        index_byte_size=5 * 16,
        member_count=30,
    )
    store = {aggregate_store_relative_key("temperature_2m", LEAD): build_container_v2([b""] * 5, descriptor=descriptor)}
    evidence = probe_aggregate(store, "temperature_2m", LEAD)
    assert evidence.ok is False
    assert "does not divide into whole field planes" in evidence.detail


def test_a_field_count_from_another_variable_is_not_evidence() -> None:
    """The silent mis-decode ``domain.field_layout`` exists to prevent, refused here too."""
    store = _store_with("temperature_2m")
    key = aggregate_store_relative_key("temperature_2m", LEAD)
    # The same bytes, asked about as another variable with a different field count.
    moved = {aggregate_store_relative_key("wind_gust", LEAD): store[key]}
    evidence = probe_aggregate(moved, "wind_gust", LEAD)
    assert evidence.ok is False
    assert "does not match" in evidence.detail


def test_an_unclassified_variable_is_not_evidence_even_with_a_container() -> None:
    store = _store_with("temperature_2m")
    key = aggregate_store_relative_key("temperature_2m", LEAD)
    moved = {aggregate_store_relative_key("not_a_variable", LEAD): store[key]}
    evidence = probe_aggregate(moved, "not_a_variable", LEAD)
    assert evidence.ok is False
    assert "no declared field layout" in evidence.detail


def test_a_container_with_no_member_count_still_reads() -> None:
    """``member_count`` is reported, not required: an older container may record none."""
    layout = aggregate_fields_for("temperature_2m")
    container = encode_aggregate_shard(
        _planes(layout.n_fields),
        _layout(layout.n_fields),
        member_count=0,
        field_scales=layout.field_scales,
    )
    store = {aggregate_store_relative_key("temperature_2m", LEAD): container}
    evidence = probe_aggregate(store, "temperature_2m", LEAD)
    assert evidence.ok is True
    assert evidence.member_count is None


def test_a_descriptor_that_disagrees_with_the_trailer_is_not_evidence() -> None:
    """``split_v2_tail`` validates the two against each other; the probe propagates the refusal."""
    store = _store_with("temperature_2m")
    key = aggregate_store_relative_key("temperature_2m", LEAD)
    honest = store[key]
    # Replace the real trailer's chunk count: the descriptor still says what it always said, so
    # the two now disagree. This is the state a partially overwritten tail would be in.
    lying = honest[:-12] + build_trailer(9, 9 * 16) + honest[-4:]
    evidence = probe_aggregate({key: lying}, "temperature_2m", LEAD)
    assert evidence.ok is False
    assert "does not parse" in evidence.detail or "shorter than its own index" in evidence.detail


@pytest.mark.parametrize("variable", ["temperature_2m", "wind_10m"])
def test_the_probe_reads_only_the_container_tail(variable: str) -> None:
    """No listing, no payload: one object, and a byte range at the end of it.

    The read count is two rather than one because the trailer has to be read before the index
    length is known, and an object too short to hold a trailer stops after the first. What must
    never happen is a read of the *object*: on S3 that is the difference between a few hundred
    bytes and tens of megabytes per ``(variable, lead)``.
    """
    reads: list[str] = []

    class _CountingStore(dict):  # type: ignore[type-arg]
        def __getitem__(self, key: str) -> bytes:
            reads.append(key)
            return dict.__getitem__(self, key)

        def get(self, key: str, default: object = None) -> object:
            reads.append(key)
            return dict.get(self, key, default)

    key = aggregate_store_relative_key(variable, LEAD)
    store = _CountingStore(_store_with(variable))
    assert probe_aggregate(store, variable, LEAD).ok is True
    assert set(reads) == {key}
    assert len(reads) == 2

    # An object that cannot even hold a trailer is answered from the first read alone.
    reads.clear()
    short = _CountingStore({key: b"\x00" * 8})
    assert probe_aggregate(short, variable, LEAD).ok is False
    assert reads == [key]


def test_the_tail_read_never_materialises_the_payload(tmp_path: object) -> None:
    """``read_range`` is what makes the probe cheap, so its range arithmetic is pinned here.

    ``cat_file``'s convention is the one all three backends share: a negative ``start`` with no
    ``end`` is a *suffix* request, and a range past the end is clamped rather than refused.
    """
    import os
    from pathlib import Path

    from ingestion.core.store_io import StoreIO, _range_to_offset_length

    directory = Path(str(tmp_path))
    blob = bytes(range(256)) * 8  # 2048 bytes, every offset distinguishable
    (directory / "obj.bin").write_bytes(blob)
    io = StoreIO(str(directory))

    assert io.read_range("obj.bin", start=-64) == blob[-64:]
    assert io.read_range("obj.bin", start=0, end=16) == blob[:16]
    assert io.read_range("obj.bin", start=100, end=116) == blob[100:116]
    assert io.read_range("obj.bin", start=-4096) == blob
    assert io.read_range("obj.bin", start=2040, end=9999) == blob[2040:]
    assert io.read_range("obj.bin") == blob
    assert os.path.getsize(directory / "obj.bin") == len(blob)

    # The offsets the resolution produces, separately from the joins above: the suffix case is
    # the whole reason the method exists.
    assert _range_to_offset_length(2048, -64, None) == (1984, 64)
    assert _range_to_offset_length(2048, -4096, None) == (0, 2048)
    assert _range_to_offset_length(2048, 0, None) == (0, 2048)
    assert _range_to_offset_length(2048, None, None) == (0, 2048)
    assert _range_to_offset_length(2048, 100, 116) == (100, 16)
    assert _range_to_offset_length(2048, 2040, 9999) == (2040, 8)
    assert _range_to_offset_length(2048, 3000, 4000) == (2048, 0)
    assert _range_to_offset_length(2048, -8, -4) == (2040, 4)
