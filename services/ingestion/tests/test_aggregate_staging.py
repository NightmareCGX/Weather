"""Tests for the staging area and aggregate pass (ingestion/core/aggregate_staging.py).

The property that matters here is equivalence: the aggregate published from staged member
shards must equal one computed directly from the member planes that were stored. That chain
passes through the v1 container encoder and the reassembly, so a mismatch could come from
either and would still produce a plausible-looking statistic field.
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest
import xarray as xr
from domain.aggregate import AggregateSpec, compute_aggregate
from domain.shard_format import SHARD_V1_MAGIC, SHARD_V2_MAGIC
from ingestion.core import aggregate_staging as staging
from ingestion.core.aggregate_writer import decode_aggregate_chunk, layout_for_spec
from ingestion.core.zarr_writer import encode_region_sharded_v1

GRID_LAT, GRID_LON = 128, 160  # two chunks each way: 4 chunks per field
CHUNK = 100
VARIABLE = "temperature_2m"
LEAD = 6


def _spec() -> AggregateSpec:
    return AggregateSpec(n_bins=4)


def _planes(n_members: int = 5, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.normal(280.0, 8.0, (GRID_LAT, GRID_LON)).astype(np.float32)
        for _ in range(n_members)
    ]


def _stage(store: str, planes: list[np.ndarray], *, lead: int = LEAD, is_mean: bool = False) -> None:
    """Write member planes into the staging area using the EXISTING v1 write path."""
    for index, plane in enumerate(planes, start=1):
        dataset = xr.Dataset(
            {
                VARIABLE: (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    plane[None, None],
                )
            }
        )
        encoded = encode_region_sharded_v1(
            dataset, member=None if is_mean else index, lead_time_hours=lead, is_mean=is_mean
        )
        relative = staging.staging_relative_key(VARIABLE, index, lead)
        full = os.path.join(store, *relative.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(encoded[0][1])


def _identical_allow_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(left[~np.isnan(left)], right[~np.isnan(right)])
    )


# ---------------------------------------------------------------------------
# Namespace and keys
# ---------------------------------------------------------------------------


def test_staging_namespace_is_a_sibling_of_the_commit_namespace() -> None:
    """Staging must not live under __commit__, or it would look like committed data."""
    assert staging.STAGING_ROOT == "__staging__"
    assert not staging.STAGING_ROOT.startswith("__commit__")
    assert staging.STAGING_VERSION == "v1"


def test_staging_key_shape_and_prefix_agree() -> None:
    key = staging.staging_relative_key(VARIABLE, 7, 6)
    assert key == "__staging__/v1/temperature_2m/mem007_L0006.shard"
    assert key.startswith(staging.staging_prefix(VARIABLE))


def test_parse_staging_name_inverts_the_key_builder() -> None:
    for member, lead in ((1, 0), (30, 240), (7, 6)):
        key = staging.staging_relative_key(VARIABLE, member, lead)
        assert staging.parse_staging_name(key) == (member, lead)


def test_parse_staging_name_rejects_anything_else() -> None:
    """The staging area is enumerated, so an unexpected object must not be skipped."""
    for bad in (
        "temperature_2m/shard.det_L0006.shard",
        "mem1_L6.shard",
        "mem001_L0006.json",
        "__commit__/v1/regions/det_L0006.json",
    ):
        with pytest.raises(staging.StagingError, match="not a staging object name"):
            staging.parse_staging_name(bad)


# ---------------------------------------------------------------------------
# Reassembly
# ---------------------------------------------------------------------------


def test_reassembled_plane_matches_the_plane_that_was_stored(tmp_path) -> None:
    """The reassembly must reproduce the stored plane, or equivalence below is circular."""
    plane = _planes(1, seed=1)[0]
    _stage(str(tmp_path), [plane])
    identity = staging.staging_relative_key(VARIABLE, 1, LEAD)
    with open(os.path.join(str(tmp_path), *identity.split("/")), "rb") as handle:
        blob = handle.read()

    recovered = staging.member_plane_from_shard(
        blob, grid_lat=GRID_LAT, grid_lon=GRID_LON, chunk_lat=CHUNK, chunk_lon=CHUNK
    )
    assert _identical_allow_nan(recovered, plane)


def test_reassembly_rejects_a_container_too_short() -> None:
    with pytest.raises(staging.StagingError, match="too short"):
        staging.member_plane_from_shard(
            b"\x00" * 8, grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


def test_reassembly_rejects_a_v2_container() -> None:
    """A v2 object in the staging area means the wrong writer put it there."""
    blob = bytearray(2000)
    struct.pack_into(
        "<III", blob, len(blob) - 12, 4, 4 * 16, SHARD_V2_MAGIC
    )
    with pytest.raises(staging.StagingError, match="sharded_v1 member containers"):
        staging.member_plane_from_shard(
            bytes(blob), grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


def test_reassembly_rejects_a_chunk_count_that_disagrees_with_the_grid(tmp_path) -> None:
    """A container written for a different grid must fail rather than reshape."""
    plane = _planes(1, seed=2)[0]
    _stage(str(tmp_path), [plane])
    identity = staging.staging_relative_key(VARIABLE, 1, LEAD)
    with open(os.path.join(str(tmp_path), *identity.split("/")), "rb") as handle:
        blob = handle.read()

    with pytest.raises(staging.StagingError, match="but the grid implies"):
        staging.member_plane_from_shard(
            blob, grid_lat=64, grid_lon=64, chunk_lat=CHUNK, chunk_lon=CHUNK
        )


def test_reassembly_rejects_an_unknown_trailer_magic(tmp_path) -> None:
    plane = _planes(1, seed=3)[0]
    _stage(str(tmp_path), [plane])
    identity = staging.staging_relative_key(VARIABLE, 1, LEAD)
    with open(os.path.join(str(tmp_path), *identity.split("/")), "rb") as handle:
        blob = bytearray(handle.read())
    struct.pack_into("<I", blob, len(blob) - 4, 0xDEADBEEF)
    with pytest.raises(staging.StagingError, match="unrecognized container magic"):
        staging.member_plane_from_shard(bytes(blob), grid_lat=GRID_LAT, grid_lon=GRID_LON)


# ---------------------------------------------------------------------------
# Staging enumeration
# ---------------------------------------------------------------------------


def test_staged_members_are_reported_sorted_per_lead(tmp_path) -> None:
    store = str(tmp_path)
    _stage(store, _planes(4, seed=4), lead=6)
    _stage(store, _planes(2, seed=5), lead=12)
    assert staging.staged_members_for_lead(store, VARIABLE, 6) == [1, 2, 3, 4]
    assert staging.staged_members_for_lead(store, VARIABLE, 12) == [1, 2]
    assert staging.staged_members_for_lead(store, VARIABLE, 99) == []


def test_staged_objects_maps_member_and_lead_to_key(tmp_path) -> None:
    store = str(tmp_path)
    _stage(store, _planes(2, seed=6), lead=6)
    _stage(store, _planes(1, seed=7), lead=12)
    found = staging.staged_objects(store, VARIABLE)
    assert set(found) == {(1, 6), (2, 6), (1, 12)}
    assert found[(2, 6)].endswith("mem002_L0006.shard")


def test_staged_lead_keys_filters_by_lead(tmp_path) -> None:
    store = str(tmp_path)
    _stage(store, _planes(2, seed=8), lead=6)
    _stage(store, _planes(1, seed=9), lead=12)
    keys = staging.staged_lead_keys(store, VARIABLE, [6])
    assert len(keys) == 2
    assert all("_L0006.shard" in key for key in keys)


def test_staging_area_with_an_unparseable_object_is_reported(tmp_path) -> None:
    """A stray object in the staging area must raise, not be silently ignored."""
    store = str(tmp_path)
    _stage(store, _planes(1, seed=10))
    stray = os.path.join(store, "__staging__", "v1", VARIABLE, "bogus.shard")
    with open(stray, "wb") as handle:
        handle.write(b"x")
    with pytest.raises(staging.StagingError, match="not a staging object name"):
        staging.staged_objects(store, VARIABLE)


def test_staging_object_membership_reports_missing_and_present(tmp_path) -> None:
    store = str(tmp_path)
    _stage(store, _planes(1, seed=11))
    assert staging.staging_object_is_v1_container(store, VARIABLE, LEAD, 1)
    assert not staging.staging_object_is_v1_container(store, VARIABLE, LEAD, 2)


def test_staging_object_membership_rejects_a_truncated_object(tmp_path) -> None:
    store = str(tmp_path)
    relative = staging.staging_relative_key(VARIABLE, 1, LEAD)
    full = os.path.join(store, *relative.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(b"\x00" * 4)
    assert not staging.staging_object_is_v1_container(store, VARIABLE, LEAD, 1)


def test_staging_object_membership_rejects_a_v2_object(tmp_path) -> None:
    store = str(tmp_path)
    relative = staging.staging_relative_key(VARIABLE, 1, LEAD)
    full = os.path.join(store, *relative.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    blob = bytearray(2000)
    struct.pack_into("<III", blob, len(blob) - 12, 4, 4 * 16, SHARD_V2_MAGIC)
    with open(full, "wb") as handle:
        handle.write(bytes(blob))
    assert not staging.staging_object_is_v1_container(store, VARIABLE, LEAD, 1)


def test_collect_rejects_a_lead_with_nothing_staged(tmp_path) -> None:
    with pytest.raises(staging.StagingError, match="no staged members"):
        staging.collect_member_planes(
            str(tmp_path), VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


# ---------------------------------------------------------------------------
# Equivalence: the property the whole design rests on
# ---------------------------------------------------------------------------


def test_aggregate_from_staging_equals_a_direct_computation(tmp_path) -> None:
    """The published aggregate must equal one computed from the stored member planes.

    Every layer in between -- the v1 container encoder, the chunk reassembly, the layout --
    could be wrong in a way that still yields a plausible statistic field, so the comparison
    is per chunk and exact against a direct computation.
    """
    store = str(tmp_path)
    spec = _spec()
    planes = _planes(5, seed=12)
    _stage(store, planes)

    key, member_count = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, spec=spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert member_count == len(planes)
    assert key == f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"

    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()

    expected_fields = compute_aggregate(np.stack(planes), spec)
    layout = layout_for_spec(spec, grid_lat=GRID_LAT, grid_lon=GRID_LON)
    assert layout.num_chunks == spec.n_fields * 4

    for ordinal in range(layout.num_chunks):
        field, row, col = layout.locate(ordinal)
        r0, c0 = row * CHUNK, col * CHUNK
        r1, c1 = min(r0 + CHUNK, GRID_LAT), min(c0 + CHUNK, GRID_LON)
        expected = np.full((CHUNK, CHUNK), np.nan, dtype=np.float32)
        expected[: r1 - r0, : c1 - c0] = expected_fields[field][r0:r1, c0:c1]
        assert _identical_allow_nan(decode_aggregate_chunk(container, ordinal), expected), ordinal


def test_aggregate_is_independent_of_member_staging_order(tmp_path) -> None:
    """Members are aggregated after a sort, so listing order cannot leak into the result."""
    spec = _spec()
    planes = _planes(4, seed=13)
    first = str(tmp_path / "a")
    second = str(tmp_path / "b")
    os.makedirs(first)
    os.makedirs(second)
    _stage(first, planes)
    # stage the same planes in a different write order
    order = [3, 1, 4, 2]
    for index in order:
        sub = xr.Dataset(
            {
                VARIABLE: (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    planes[index - 1][None, None],
                )
            }
        )
        blob = encode_region_sharded_v1(sub, member=index, lead_time_hours=LEAD)[0][1]
        relative = staging.staging_relative_key(VARIABLE, index, LEAD)
        full = os.path.join(second, *relative.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(blob)

    key_a, _ = staging.aggregate_staged_lead(
        first, VARIABLE, LEAD, spec=spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    key_b, _ = staging.aggregate_staged_lead(
        second, VARIABLE, LEAD, spec=spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    with open(os.path.join(first, *key_a.split("/")), "rb") as handle:
        container_a = handle.read()
    with open(os.path.join(second, *key_b.split("/")), "rb") as handle:
        container_b = handle.read()
    assert container_a == container_b


def test_staging_is_dropped_after_the_aggregate_is_written(tmp_path) -> None:
    """Staging bytes must not survive a successful aggregation."""
    store = str(tmp_path)
    _stage(store, _planes(3, seed=14), lead=LEAD)
    _stage(store, _planes(2, seed=15), lead=12)
    staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, spec=_spec(), grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert staging.staged_members_for_lead(store, VARIABLE, LEAD) == []
    # another lead's staging is untouched
    assert staging.staged_members_for_lead(store, VARIABLE, 12) == [1, 2]


def test_drop_staging_can_be_disabled(tmp_path) -> None:
    """Reprocessing a lead without re-staging is a supported operation."""
    store = str(tmp_path)
    _stage(store, _planes(2, seed=16), lead=LEAD)
    staging.aggregate_staged_lead(
        store,
        VARIABLE,
        LEAD,
        spec=_spec(),
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        drop_staging=False,
    )
    assert staging.staged_members_for_lead(store, VARIABLE, LEAD) == [1, 2]


def test_expected_members_refuses_a_partial_aggregate(tmp_path) -> None:
    """Publication policy may require the complete set; the pass must honour that."""
    store = str(tmp_path)
    _stage(store, _planes(3, seed=17), lead=LEAD)
    with pytest.raises(staging.StagingError, match="3 staged members, expected 30"):
        staging.aggregate_staged_lead(
            store,
            VARIABLE,
            LEAD,
            spec=_spec(),
            grid_lat=GRID_LAT,
            grid_lon=GRID_LON,
            expected_members=30,
        )
    # nothing was published or dropped
    assert staging.staged_members_for_lead(store, VARIABLE, LEAD) == [1, 2, 3]


def test_aggregate_from_planes_rejects_shape_and_empty_input() -> None:
    with pytest.raises(staging.StagingError, match="at least one member plane"):
        staging.aggregate_from_planes([], spec=_spec(), grid_lat=GRID_LAT, grid_lon=GRID_LON)
    wrong = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(staging.StagingError, match="expected"):
        staging.aggregate_from_planes(
            [wrong], spec=_spec(), grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


def test_aggregate_layout_helper_matches_the_written_container(tmp_path) -> None:
    store = str(tmp_path)
    spec = _spec()
    _stage(store, _planes(2, seed=18), lead=LEAD)
    key, _ = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, spec=spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    layout = staging.aggregate_layout_for(
        spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()
    num_chunks, _index_size, magic = struct.unpack("<III", container[-12:])
    assert magic == SHARD_V2_MAGIC
    assert num_chunks == layout.num_chunks


def test_aggregate_is_bit_identical_across_a_recompute(tmp_path) -> None:
    """Recomputing from the same staging must produce the same bytes, not merely close ones."""
    store = str(tmp_path)
    spec = _spec()
    _stage(store, _planes(3, seed=19), lead=LEAD)

    first_key, _ = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, spec=spec, grid_lat=GRID_LAT, grid_lon=GRID_LON,
        drop_staging=False,
    )
    with open(os.path.join(store, *first_key.split("/")), "rb") as handle:
        first = handle.read()
    second_key, _ = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, spec=spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    with open(os.path.join(store, *second_key.split("/")), "rb") as handle:
        second = handle.read()
    assert first == second


def test_v1_magic_is_the_expected_container_generation() -> None:
    """Guards against a staging object being written by the v2 path by mistake."""
    assert SHARD_V1_MAGIC == 0x53484152


def _quantile_spec() -> AggregateSpec:
    from domain.aggregate import KIND_QUANTILE_FUNCTION

    return AggregateSpec(kind=KIND_QUANTILE_FUNCTION)


def test_quantile_aggregate_from_staging_equals_a_direct_computation(tmp_path) -> None:
    """The B/C-class encoding must hold up through the same chain as the A-class one.

    The two encodings differ in field count and in what each plane means, so equivalence has
    to be re-established for this kind rather than inferred from the other.
    """
    store = str(tmp_path)
    spec = _quantile_spec()
    planes = _planes(4, seed=20)
    _stage(store, planes)

    key, member_count = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, spec=spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert member_count == len(planes)
    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()

    expected_fields = compute_aggregate(np.stack(planes), spec)
    layout = layout_for_spec(spec, grid_lat=GRID_LAT, grid_lon=GRID_LON)
    assert layout.n_fields == len(spec.levels) == 19

    for ordinal in range(layout.num_chunks):
        field, row, col = layout.locate(ordinal)
        r0, c0 = row * CHUNK, col * CHUNK
        r1, c1 = min(r0 + CHUNK, GRID_LAT), min(c0 + CHUNK, GRID_LON)
        expected = np.full((CHUNK, CHUNK), np.nan, dtype=np.float32)
        expected[: r1 - r0, : c1 - c0] = expected_fields[field][r0:r1, c0:c1]
        assert _identical_allow_nan(decode_aggregate_chunk(container, ordinal), expected), ordinal


def test_quantile_aggregate_exceedance_survives_the_container_round_trip(tmp_path) -> None:
    """The published aggregate must answer a threshold as well as the in-memory fields do."""
    from domain.aggregate import exceedance_from_quantiles

    store = str(tmp_path)
    spec = _quantile_spec()
    planes = _planes(6, seed=21)
    _stage(store, planes)
    key, _ = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, spec=spec, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()

    layout = layout_for_spec(spec, grid_lat=GRID_LAT, grid_lon=GRID_LON)
    # Reassemble the levels from the published container, chunk by chunk.
    levels = np.full((layout.n_fields, GRID_LAT, GRID_LON), np.nan, dtype=np.float32)
    for ordinal in range(layout.num_chunks):
        field, row, col = layout.locate(ordinal)
        chunk = decode_aggregate_chunk(container, ordinal)
        r0, c0 = row * CHUNK, col * CHUNK
        r1, c1 = min(r0 + CHUNK, GRID_LAT), min(c0 + CHUNK, GRID_LON)
        levels[field, r0:r1, c0:c1] = chunk[: r1 - r0, : c1 - c0]

    threshold = float(np.percentile(np.stack(planes), 90.0))
    from_container = exceedance_from_quantiles(levels, spec.levels, threshold)
    from_memory = exceedance_from_quantiles(
        compute_aggregate(np.stack(planes), spec), spec.levels, threshold
    )
    truth = np.mean(np.stack(planes) > threshold, axis=0)
    assert abs(float(np.mean(from_container)) - float(np.mean(truth))) < 0.05
    # quantisation is the only difference, and it is one step of the scale
    assert np.allclose(from_container, from_memory, atol=0.02)
