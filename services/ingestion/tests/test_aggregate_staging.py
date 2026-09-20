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
from domain.aggregate import (
    AggregateSpec,
    compute_aggregate,
)
from domain.shard_format import SHARD_V1_MAGIC, SHARD_V2_MAGIC
from ingestion.core import aggregate_staging as staging
from ingestion.core.aggregate_writer import (
    decode_aggregate_chunk,
    encode_aggregate_shard,
    layout_for_spec,
)
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


def _stage(
    store: str,
    planes: list[np.ndarray],
    *,
    lead: int = LEAD,
    is_mean: bool = False,
    variable: str = VARIABLE,
) -> None:
    """Write member planes into the staging area using the EXISTING v1 write path."""
    for index, plane in enumerate(planes, start=1):
        dataset = xr.Dataset(
            {
                variable: (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    plane[None, None],
                )
            }
        )
        encoded = encode_region_sharded_v1(
            dataset, member=None if is_mean else index, lead_time_hours=lead, is_mean=is_mean
        )
        relative = staging.staging_relative_key(variable, index, lead)
        full = os.path.join(store, *relative.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(encoded[0][1])


def _stage_variable(
    store: str, variable: str, planes: list[np.ndarray], *, lead: int = LEAD
) -> None:
    """The same writer, for a variable other than the module-level default."""
    _stage(store, planes, lead=lead, variable=variable)


def _identical_allow_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(left[~np.isnan(left)], right[~np.isnan(right)])
    )


def _within_quantisation(
    left: np.ndarray, right: np.ndarray, tolerance: float
) -> bool:
    """Equal in NaN pattern, and within ``tolerance`` where both are finite.

    Stored fields are int16 at a per-field step, so an exact comparison is only meaningful for
    an unquantised container; for a stored one the bound is half a step plus the float32
    rounding of the scale multiplication, which is what the quantiser guarantees.
    """
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return False
    finite = ~np.isnan(left)
    # The slack is the scale's own float32 representation error on values of this magnitude:
    # a mean near 280 stored at 0.01 rounds the product 280.49 * 0.01 to within ~1e-5 of the
    # exact value, which is three orders below the half-step bound and not a defect.
    slack = tolerance * 1e-3 + 1e-4
    return bool(np.all(np.abs(left[finite] - right[finite]) <= tolerance + slack))


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
        staging.aggregate_staged_lead(
            str(tmp_path), VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


# ---------------------------------------------------------------------------
# Equivalence: the property the whole design rests on
# ---------------------------------------------------------------------------


def _container_layout(n_fields: int):
    """The container geometry the writer uses for a field vector of this width."""
    from ingestion.core.aggregate_writer import AggregateShardLayout

    return AggregateShardLayout(
        n_fields=n_fields, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )


def _expected_fields(planes: list[np.ndarray], variable: str = VARIABLE):
    """The container's field vector for the variable's registered encoding, from the planes.

    The authority is ``domain.field_layout``, which is also where the writer takes the field order
    and the per-field steps. Recomputing them here independently would make this a test of two
    transcriptions rather than of the container.
    """
    from domain.aggregate import finite_member_count
    from domain.field_layout import aggregate_fields_for
    from domain.variable_class import spec_for

    layout = aggregate_fields_for(variable)
    stack = np.stack(planes)
    fields = [finite_member_count(stack)]
    spec = spec_for(variable) if layout.distribution_slice.stop > 1 else None
    if spec is not None:
        fields.extend(compute_aggregate(stack, spec, expected_members=len(planes)))
    for group in layout.groups:
        # The fixture stages none of the group inputs, so this is only reached for a variable with
        # no groups. Asserted rather than silently skipped.
        raise AssertionError(f"this helper does not build the {group!r} group")
    return np.concatenate([field[None] for field in fields]), layout


def test_aggregate_from_staging_equals_a_direct_computation(tmp_path) -> None:
    """The published container must equal one computed from the stored member planes.

    Every layer in between -- the v1 container encoder, the chunk reassembly, the layout, the
    fixed-point round trip -- could be wrong in a way that still yields a plausible statistic
    field, so the comparison is per chunk against a direct computation, bounded by half a
    quantisation step of the field's own scale.

    The variable is ``temperature_2m``, whose container is its distribution alone, so the field
    vector is the count followed by the spec's own fields. Field 0 is the per-cell member count,
    and its step is one member, so it must come back exactly.
    """
    from domain.field_layout import aggregate_fields_for

    store = str(tmp_path)
    planes = _planes(5, seed=12)
    _stage(store, planes)

    key, member_count = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert member_count == len(planes)
    assert key == f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"

    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()

    layout = aggregate_fields_for(VARIABLE)
    assert layout.groups == (), "the fixture stages no group inputs"
    expected_fields, _declared = _expected_fields(planes)
    assert expected_fields.shape[0] == layout.n_fields
    shard_layout = _container_layout(layout.n_fields)
    assert shard_layout.num_chunks == layout.n_fields * 4

    for ordinal in range(shard_layout.num_chunks):
        field, row, col = shard_layout.locate(ordinal)
        r0, c0 = row * CHUNK, col * CHUNK
        r1, c1 = min(r0 + CHUNK, GRID_LAT), min(c0 + CHUNK, GRID_LON)
        expected = np.full((CHUNK, CHUNK), np.nan, dtype=np.float32)
        expected[: r1 - r0, : c1 - c0] = expected_fields[field][r0:r1, c0:c1]
        decoded = decode_aggregate_chunk(
            container, ordinal, field_scales=layout.field_scales
        )
        assert _within_quantisation(decoded, expected, layout.field_scales[field] / 2), ordinal


def test_aggregate_is_independent_of_member_staging_order(tmp_path) -> None:
    """Members are aggregated after a sort, so listing order cannot leak into the result."""
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
        first, VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    key_b, _ = staging.aggregate_staged_lead(
        second, VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
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
        store, VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
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
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        drop_staging=False,
    )
    assert staging.staged_members_for_lead(store, VARIABLE, LEAD) == [1, 2]


def test_expected_members_sets_the_coverage_floor_rather_than_a_completeness_gate(
    tmp_path,
) -> None:
    """A partial set is published; what the contract's count decides is the coverage floor.

    A lead publishes at 85% coverage after a quiet window as well as at completeness
    (``ingestion.core.settlement``), so refusing a partial set here would refuse the patches the
    publication policy asks for. What the contract's count must not do is *vary*: measured against
    the staged count instead, 3 of 30 members would clear a floor of 85% of 3 and the container
    would report a handful of members as a complete cell -- at coverage the serving tier refuses.
    """
    store = str(tmp_path)
    _stage(store, _planes(3, seed=17), lead=LEAD)

    key, count = staging.aggregate_staged_lead(
        store,
        VARIABLE,
        LEAD,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        expected_members=30,
    )
    assert count == 3
    assert staging.staged_members_for_lead(store, VARIABLE, LEAD) == []
    # 3 of 30 is below the floor, so the container carries no distribution value at all --
    # the count field records how many members there were, and nothing claims more.
    from domain.field_layout import aggregate_fields_for

    layout = aggregate_fields_for(VARIABLE)
    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()
    shard_layout = _container_layout(layout.n_fields)
    mean_chunk = decode_aggregate_chunk(
        container,
        shard_layout.chunk_ordinal(layout.index_of_role("mean"), 0, 0),
        field_scales=layout.field_scales,
    )
    assert np.isnan(mean_chunk[0, 0])


def test_more_members_than_the_contract_declares_is_a_bookkeeping_error(tmp_path) -> None:
    """Thirty-one staged members for a 30-member contract is a fault, not a partial set."""
    store = str(tmp_path)
    _stage(store, _planes(3, seed=18), lead=LEAD)
    with pytest.raises(staging.StagingError, match="more than the 2 the contract declares"):
        staging.aggregate_staged_lead(
            store,
            VARIABLE,
            LEAD,
            grid_lat=GRID_LAT,
            grid_lon=GRID_LON,
            expected_members=2,
        )


def test_aggregate_is_bit_identical_across_a_recompute(tmp_path) -> None:
    """Recomputing from the same staging must produce the same bytes, not merely close ones."""
    store = str(tmp_path)
    _stage(store, _planes(3, seed=19), lead=LEAD)

    first_key, _ = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON, drop_staging=False
    )
    with open(os.path.join(store, *first_key.split("/")), "rb") as handle:
        first = handle.read()
    second_key, _ = staging.aggregate_staged_lead(
        store, VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    with open(os.path.join(store, *second_key.split("/")), "rb") as handle:
        second = handle.read()
    assert first == second


def test_v1_magic_is_the_expected_container_generation() -> None:
    """Guards against a staging object being written by the v2 path by mistake."""
    assert SHARD_V1_MAGIC == 0x53484152


def test_quantile_aggregate_from_staging_equals_a_direct_computation(tmp_path) -> None:
    """The B/C-class encoding must hold up through the same chain as the A-class one.

    The two encodings differ in field count and in what each plane means, so equivalence has
    to be re-established for this kind rather than inferred from the other.
    """
    from domain.aggregate import finite_member_count
    from domain.field_layout import aggregate_fields_for
    from domain.variable_class import spec_for

    store = str(tmp_path)
    variable = "wind_gust"
    planes = _planes(4, seed=20)
    _stage_variable(store, variable, planes)

    key, member_count = staging.aggregate_staged_lead(
        store, variable, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert member_count == len(planes)
    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()

    spec = spec_for(variable)
    layout = aggregate_fields_for(variable)
    assert layout.groups == ()
    stack = np.stack(planes)
    expected_fields = np.concatenate(
        [finite_member_count(stack)[None], compute_aggregate(stack, spec)]
    )
    assert layout.n_fields == len(spec.levels) + 1 == 20
    shard_layout = _container_layout(layout.n_fields)

    for ordinal in range(shard_layout.num_chunks):
        field, row, col = shard_layout.locate(ordinal)
        r0, c0 = row * CHUNK, col * CHUNK
        r1, c1 = min(r0 + CHUNK, GRID_LAT), min(c0 + CHUNK, GRID_LON)
        expected = np.full((CHUNK, CHUNK), np.nan, dtype=np.float32)
        expected[: r1 - r0, : c1 - c0] = expected_fields[field][r0:r1, c0:c1]
        decoded = decode_aggregate_chunk(
            container, ordinal, field_scales=layout.field_scales
        )
        assert _within_quantisation(decoded, expected, layout.field_scales[field] / 2), ordinal


def test_quantile_aggregate_exceedance_survives_the_container_round_trip(tmp_path) -> None:
    """The published aggregate must answer a threshold as well as the in-memory fields do."""
    from domain.aggregate import exceedance_from_quantiles
    from domain.field_layout import aggregate_fields_for
    from domain.variable_class import spec_for

    store = str(tmp_path)
    variable = "wind_gust"
    planes = _planes(6, seed=21)
    _stage_variable(store, variable, planes)
    key, _ = staging.aggregate_staged_lead(
        store, variable, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()

    spec = spec_for(variable)
    layout = aggregate_fields_for(variable)
    shard_layout = _container_layout(layout.n_fields)
    # Reassemble the distribution from the published container, chunk by chunk. Field 0 is the
    # member count, so the levels start at 1.
    distribution = layout.distribution_slice
    levels = np.full((spec.n_fields, GRID_LAT, GRID_LON), np.nan, dtype=np.float32)
    for ordinal in range(shard_layout.num_chunks):
        field, row, col = shard_layout.locate(ordinal)
        if field not in range(distribution.start, distribution.stop):
            continue
        chunk = decode_aggregate_chunk(
            container, ordinal, field_scales=layout.field_scales
        )
        r0, c0 = row * CHUNK, col * CHUNK
        r1, c1 = min(r0 + CHUNK, GRID_LAT), min(c0 + CHUNK, GRID_LON)
        levels[field - distribution.start, r0:r1, c0:c1] = chunk[: r1 - r0, : c1 - c0]

    stack = np.stack(planes)
    threshold = float(np.percentile(stack, 90.0))
    from_container = exceedance_from_quantiles(levels, spec.levels, threshold)
    from_memory = exceedance_from_quantiles(
        compute_aggregate(stack, spec), spec.levels, threshold
    )
    truth = np.mean(stack > threshold, axis=0)
    assert abs(float(np.mean(from_container)) - float(np.mean(truth))) < 0.05
    # quantisation is the only difference, and it is one step of the scale
    assert np.allclose(from_container, from_memory, atol=0.02)


# ---------------------------------------------------------------------------
# Staging writes and the per-lead pass
# ---------------------------------------------------------------------------


def test_stage_region_writes_one_object_per_variable_and_matches_serving_bytes(tmp_path) -> None:
    """A staged object must be byte-identical to the serving shard for the same data.

    That equality is what makes the aggregate pass's equivalence property meaningful: it
    computes from the same bytes a reader would see, not from a parallel encoder.
    """
    store = str(tmp_path)
    plane = _planes(1, seed=30)[0]
    dataset = xr.Dataset(
        {
            VARIABLE: (
                ("lead_time_hours", "latitude", "longitude"),
                plane[None],
            )
        }
    )
    written = staging.stage_region(dataset, store, member=1, lead_time_hours=LEAD)
    assert written == [staging.staging_relative_key(VARIABLE, 1, LEAD)]

    with open(os.path.join(store, *written[0].split("/")), "rb") as handle:
        staged_bytes = handle.read()
    serving_bytes = encode_region_sharded_v1(
        dataset, member=1, lead_time_hours=LEAD
    )[0][1]
    assert staged_bytes == serving_bytes


def test_stage_region_rejects_a_dataset_with_no_encodable_variable(tmp_path) -> None:
    dataset = xr.Dataset({"scalar_only": (("lead_time_hours",), np.zeros(1, dtype=np.float32))})
    with pytest.raises(staging.StagingError, match="nothing encoded"):
        staging.stage_region(dataset, str(tmp_path), member=1, lead_time_hours=LEAD)


def test_staged_objects_by_variable_groups_the_whole_root(tmp_path) -> None:
    store = str(tmp_path)
    _stage(store, _planes(2, seed=31), lead=LEAD)
    other = str(tmp_path / "other.zarr")
    os.makedirs(other)
    for index, plane in enumerate(_planes(1, seed=32), start=1):
        dataset = xr.Dataset(
            {
                "wind_gust": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    plane[None, None],
                )
            }
        )
        blob = encode_region_sharded_v1(dataset, member=index, lead_time_hours=LEAD)[0][1]
        relative = staging.staging_relative_key("wind_gust", index, LEAD)
        full = os.path.join(other, *relative.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(blob)

    grouped = staging.staged_objects_by_variable(other)
    assert set(grouped) == {"wind_gust"}
    assert grouped["wind_gust"][(1, LEAD)].endswith("mem001_L0006.shard")


def test_staged_objects_by_variable_rejects_an_object_without_a_variable() -> None:
    """The staging namespace is owned entirely by the platform, so a stray key is an error."""
    mapping: dict[str, bytes] = {
        f"{staging.STAGING_ROOT}/{staging.STAGING_VERSION}/mem001_L0006.shard": b"x",
        # a well-formed object so the enumeration reaches the malformed one
        f"{staging.STAGING_ROOT}/{staging.STAGING_VERSION}/{VARIABLE}/"
        "mem001_L0006.shard": b"x",
    }
    with pytest.raises(staging.StagingError, match="not under a variable segment"):
        staging.staged_objects_by_variable(mapping)


def test_aggregate_lead_all_variables_covers_every_staged_variable(tmp_path) -> None:
    """One pass must aggregate both classes present at a lead, each with its own encoding."""
    store = str(tmp_path)
    _stage(store, _planes(3, seed=33), lead=LEAD)
    gust_planes = _planes(3, seed=34)
    for index, plane in enumerate(gust_planes, start=1):
        dataset = xr.Dataset(
            {
                "wind_gust": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    plane[None, None],
                )
            }
        )
        blob = encode_region_sharded_v1(dataset, member=index, lead_time_hours=LEAD)[0][1]
        relative = staging.staging_relative_key("wind_gust", index, LEAD)
        full = os.path.join(store, *relative.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(blob)

    results = staging.aggregate_lead_all_variables(
        store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    aggregated = {variable: key for variable, key, _count in results}
    assert set(aggregated) == {VARIABLE, "wind_gust"}
    assert aggregated[VARIABLE] == f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"
    assert aggregated["wind_gust"] == f"wind_gust/shard.agg_L{LEAD:04d}.shard"
    assert all(count == 3 for _v, _k, count in results)
    # both leads' staging is gone and both aggregates exist
    assert staging.staged_objects_by_variable(store) == {}
    for key in aggregated.values():
        assert os.path.isfile(os.path.join(store, *key.split("/")))


def test_aggregate_lead_all_variables_uses_the_per_variable_encoding(tmp_path) -> None:
    """The pass must not apply one class's field set to another class's variable."""
    from domain.aggregate import KIND_QUANTILE_FUNCTION, KIND_MEAN_STD_BINS
    from domain.variable_class import spec_for

    store = str(tmp_path)
    _stage(store, _planes(2, seed=35), lead=LEAD)  # temperature_2m -> A
    staging.aggregate_lead_all_variables(
        store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )

    # The written container's field count is recoverable from its descriptor, which is what
    # tells a reader which encoding it holds without consulting the writer.
    from domain.shard_format import parse_trailer, split_v2_tail
    from ingestion.core.aggregate_writer import layout_from_descriptor

    key = f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"
    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        container = handle.read()
    trailer = parse_trailer(container[-12:])
    _index, descriptor = split_v2_tail(
        container[-(trailer.index_byte_size + 40 + 12) :], trailer.num_chunks
    )
    layout = layout_from_descriptor(descriptor)
    assert spec_for(VARIABLE).kind == KIND_MEAN_STD_BINS
    # One more field than the spec declares: every stored container leads with the per-cell
    # member count, so the encoding's own field count is still recoverable and is still what
    # distinguishes the two classes.
    assert layout.n_fields == spec_for(VARIABLE).n_fields + 1
    assert spec_for("wind_gust").kind == KIND_QUANTILE_FUNCTION


def test_aggregate_lead_all_variables_skips_an_unclassified_variable(tmp_path, caplog) -> None:
    """A variable the platform does not classify must be skipped, not guessed at."""
    store = str(tmp_path)
    _stage(store, _planes(2, seed=36), lead=LEAD)
    dataset = xr.Dataset(
        {
            "mystery_variable": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                _planes(1, seed=37)[0][None, None],
            )
        }
    )
    blob = encode_region_sharded_v1(dataset, member=1, lead_time_hours=LEAD)[0][1]
    relative = staging.staging_relative_key("mystery_variable", 1, LEAD)
    full = os.path.join(store, *relative.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(blob)

    results = staging.aggregate_lead_all_variables(
        store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert {variable for variable, _k, _c in results} == {VARIABLE}
    # the unclassified variable's staging is left alone for a human to resolve
    assert "mystery_variable" in staging.staged_objects_by_variable(store)


def test_aggregate_lead_all_variables_rejects_a_lead_with_nothing_staged(tmp_path) -> None:
    with pytest.raises(staging.StagingError, match="nothing staged for lead"):
        staging.aggregate_lead_all_variables(
            str(tmp_path), 99, grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


# ---------------------------------------------------------------------------
# The pipeline hook (aggregate_phase)
# ---------------------------------------------------------------------------


def _enable_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    from ingestion.core import config as ingestion_config

    monkeypatch.setattr(
        ingestion_config.settings, "ENSEMBLE_STAGING_ENABLED", True, raising=False
    )
    # aggregate_phase reads the module-level settings object lazily, so patching the shared
    # settings instance is enough.
    monkeypatch.setattr(
        "ingestion.core.aggregate_phase.staging_enabled", lambda: True
    )


def test_phase_is_inert_when_disabled(tmp_path) -> None:
    """With the switch off nothing is written, which is what makes the phase safe to land."""
    from ingestion.core import aggregate_phase

    dataset = xr.Dataset(
        {
            VARIABLE: (
                ("member", "lead_time_hours", "latitude", "longitude"),
                _planes(1, seed=40)[0][None, None],
            )
        }
    )
    assert aggregate_phase.staging_enabled() is False
    assert aggregate_phase.stage_member_region(
        dataset, str(tmp_path), member=1, lead_time_hours=LEAD
    ) == ()
    result = aggregate_phase.aggregate_lead(str(tmp_path), LEAD)
    assert result.aggregates == ()
    assert staging.staged_objects_by_variable(str(tmp_path)) == {}


def test_phase_stages_perturbed_members_only(tmp_path, monkeypatch) -> None:
    """The mean and deterministic products have no member axis to aggregate over."""
    from ingestion.core import aggregate_phase

    _enable_phase(monkeypatch)
    dataset = xr.Dataset(
        {
            VARIABLE: (
                ("lead_time_hours", "latitude", "longitude"),
                _planes(1, seed=41)[0][None],
            )
        }
    )
    assert aggregate_phase.stage_member_region(
        dataset, str(tmp_path), member=None, lead_time_hours=LEAD
    ) == ()
    assert aggregate_phase.stage_member_region(
        dataset, str(tmp_path), member=None, lead_time_hours=LEAD, is_mean=True
    ) == ()
    assert staging.staged_objects_by_variable(str(tmp_path)) == {}


def test_phase_stages_a_member_and_reports_the_keys(tmp_path, monkeypatch) -> None:
    from ingestion.core import aggregate_phase

    _enable_phase(monkeypatch)
    dataset = xr.Dataset(
        {
            VARIABLE: (
                ("lead_time_hours", "latitude", "longitude"),
                _planes(1, seed=42)[0][None],
            )
        }
    )
    keys = aggregate_phase.stage_member_region(
        dataset, str(tmp_path), member=7, lead_time_hours=LEAD
    )
    assert keys == (staging.staging_relative_key(VARIABLE, 7, LEAD),)
    assert staging.staged_members_for_lead(str(tmp_path), VARIABLE, LEAD) == [7]


def test_phase_reports_an_unstageable_member_loudly(tmp_path, monkeypatch) -> None:
    """A member committed without a staged copy would be aggregated as a partial set."""
    from ingestion.core import aggregate_phase

    _enable_phase(monkeypatch)
    dataset = xr.Dataset(
        {"scalar_only": (("lead_time_hours",), np.zeros(1, dtype=np.float32))}
    )
    with pytest.raises(aggregate_phase.AggregatePhaseError, match="cannot stage member"):
        aggregate_phase.stage_member_region(
            dataset, str(tmp_path), member=1, lead_time_hours=LEAD
        )


def test_phase_aggregates_a_fully_staged_lead(tmp_path, monkeypatch) -> None:
    from ingestion.core import aggregate_phase

    _enable_phase(monkeypatch)
    store = str(tmp_path)
    _stage(store, _planes(4, seed=43), lead=LEAD)

    result = aggregate_phase.aggregate_lead(
        store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert result.aggregates == (
        (VARIABLE, f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard", 4),
    )
    assert staging.staged_objects_by_variable(store) == {}


def test_the_pass_reads_an_inputs_staging_before_releasing_it(tmp_path) -> None:
    """A container is a function of *other* variables' staging, so the release is deferred.

    The wind container is built from ``wind_u_10m`` and ``wind_v_10m``, which are themselves
    variables with containers of their own. Releasing a variable's staging as soon as its own
    container was written would destroy the wind pair before the wind container was built, and
    every refusal in ``aggregate_fields`` exists precisely to catch a publication that ran with a
    subset of its inputs -- so with the release inline this pass would fail rather than publish
    something wrong, which is the good outcome but still a pass that cannot run.
    """
    store = str(tmp_path)
    # The components are m/s, so the fixture's planes have to be speeds rather than temperatures:
    # a consensus speed stored at a 0.01 step overflows past +-327 m/s and is refused.
    components = {
        name: [
            np.random.default_rng(seed)
            .normal(3.0, 4.0, (GRID_LAT, GRID_LON))
            .astype(np.float32)
            for seed in (71, 72, 73)
        ]
        for name in ("wind_u_10m", "wind_v_10m")
    }
    for name, planes in components.items():
        _stage_variable(store, name, planes)

    results = staging.aggregate_lead_all_variables(
        store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    written = {variable for variable, _key, _count in results}
    # The two components and the synthesised wind variable all have containers.
    assert written == {"wind_u_10m", "wind_v_10m", "wind_10m"}, written
    assert staging.staged_objects_by_variable(store) == {}
    for variable, key, count in results:
        assert count == 3, variable
        assert os.path.isfile(os.path.join(store, *key.split("/"))), key


def test_a_variable_with_no_approved_encoding_keeps_its_staging(tmp_path, caplog) -> None:
    """Nothing can rebuild a skipped variable's members, so its staging is left alone.

    The pass writes at most one container per variable and releases only those, so a name the
    platform does not classify stays inspectable instead of being dropped.
    """
    store = str(tmp_path)
    _stage(store, _planes(2, seed=72))
    _stage_variable(store, "mystery_variable", _planes(2, seed=73))

    results = staging.aggregate_lead_all_variables(
        store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert {variable for variable, _k, _c in results} == {VARIABLE}
    remaining = staging.staged_objects_by_variable(store)
    assert set(remaining) == {"mystery_variable"}


def test_a_missing_input_the_pass_could_build_is_reported_not_skipped(tmp_path) -> None:
    """A known variable whose container cannot be built is a failure, not a variable to skip.

    Distinguishing the two is what keeps a missing input from vanishing into a warning: the
    precipitation container reads the four flags, so with the amount staged and the flags absent
    the pass must raise rather than quietly leaving the variable for a later pass to find.
    """
    store = str(tmp_path)
    _stage_variable(store, "precipitation_amount_3h", _planes(2, seed=74))

    with pytest.raises(staging.StagingError, match="not staged at lead"):
        staging.aggregate_lead_all_variables(
            store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


def test_a_later_variable_can_read_an_earlier_ones_staged_members(tmp_path) -> None:
    """The release is per pass, not per variable, and the pass order is what makes that safe.

    ``precipitation_amount_3h`` reads the four flags, which are variables aggregated in the same
    pass (and sorted before it). Both have containers afterwards and neither keeps staging.
    """
    store = str(tmp_path)
    _stage_variable(store, "precipitation_amount_3h", _planes(3, seed=75))
    for flag in ("crain", "csnow", "cfrzr", "cicep"):
        # Staged as 0/1 planes, which is what the provider emits for a categorical flag.
        rng = np.random.default_rng(hash(flag) % 1000)
        _stage_variable(
            store,
            flag,
            [(rng.random((GRID_LAT, GRID_LON)) < 0.5).astype(np.float32) for _ in range(3)],
        )

    results = staging.aggregate_lead_all_variables(
        store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    written = {variable: count for variable, _key, count in results}
    assert written == {
        "precipitation_amount_3h": 3,
        "crain": 3,
        "csnow": 3,
        "cfrzr": 3,
        "cicep": 3,
    }
    assert staging.staged_objects_by_variable(store) == {}


def test_a_predecessor_the_wave_is_filling_is_refused_by_the_pass(tmp_path) -> None:
    """The pass forwards its wave targets, so a late predecessor is a refusal rather than a gap.

    Without this the container for a reset lead would encode the *absent* predecessor reading
    (``persistent_rain``) for a predecessor that is merely late, and the next patch would silently
    correct it. Here the amount at lead 6 is staged and its predecessor lead 3 is not, while the
    wave declares that it is filling lead 3 -- so the pass refuses the container instead of
    encoding it from half an interval.
    """
    store = str(tmp_path)
    for variable in ("precipitation_amount_3h", "crain", "csnow", "cfrzr", "cicep"):
        _stage_variable(store, variable, _planes(3, seed=76))

    with pytest.raises(staging.StagingError, match="cannot be classified yet"):
        staging.aggregate_lead_all_variables(
            store,
            LEAD,
            grid_lat=GRID_LAT,
            grid_lon=GRID_LON,
            wave_leads=(LEAD, LEAD - 3),
            variables=("precipitation_amount_3h",),
        )
    # Its staging is untouched, so the next publication can build the container.
    assert staging.staged_members_for_lead(
        store, "precipitation_amount_3h", LEAD
    ) == [1, 2, 3]


def test_a_predecessor_outside_the_wave_is_accepted_by_the_pass(tmp_path) -> None:
    """A lead whose predecessor this wave is not filling has none, and that is a definite answer."""
    store = str(tmp_path)
    for variable in ("precipitation_amount_3h", "crain", "csnow", "cfrzr", "cicep"):
        _stage_variable(store, variable, _planes(3, seed=79))

    results = staging.aggregate_lead_all_variables(
        store,
        LEAD,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        wave_leads=(LEAD,),
        variables=("precipitation_amount_3h",),
    )
    assert [variable for variable, _k, _c in results] == ["precipitation_amount_3h"]


def test_phase_aggregate_is_a_no_op_for_a_lead_with_nothing_staged(tmp_path, monkeypatch) -> None:
    """A lead with no members is a normal state (a wave may not have reached it yet)."""
    from ingestion.core import aggregate_phase

    _enable_phase(monkeypatch)
    result = aggregate_phase.aggregate_lead(
        str(tmp_path), LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert result.aggregates == ()


def test_explicit_variable_aggregation_ignores_the_switch(tmp_path) -> None:
    """The explicit form is for a caller that has already decided to aggregate."""
    from ingestion.core import aggregate_phase

    store = str(tmp_path)
    _stage(store, _planes(3, seed=44), lead=LEAD)
    key, count = aggregate_phase.aggregate_variable_lead(
        store, VARIABLE, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert count == 3
    assert key == f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"


def test_explicit_variable_aggregation_rejects_an_unclassified_variable(tmp_path) -> None:
    from ingestion.core import aggregate_phase

    with pytest.raises(aggregate_phase.AggregatePhaseError, match="no approved aggregate"):
        aggregate_phase.aggregate_variable_lead(
            str(tmp_path), "mystery", LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


def test_explicit_variable_aggregation_reports_a_staging_failure(tmp_path) -> None:
    from ingestion.core import aggregate_phase

    with pytest.raises(aggregate_phase.AggregatePhaseError, match="no staged members"):
        aggregate_phase.aggregate_variable_lead(
            str(tmp_path), VARIABLE, 99, grid_lat=GRID_LAT, grid_lon=GRID_LON
        )


def test_point_query_reads_four_contiguous_ranges(tmp_path) -> None:
    """The layout's purpose: a point query's window is four contiguous byte ranges.

    The point path needs every stored field at one location, so under the spatial-major layout
    all ``n_fields`` chunks of a corner are adjacent in the payload and the four corners form
    four ranges. Field-major would have made it 2 x n_fields ranges. Asserted on the actual
    index rather than on the ordinal arithmetic, because the index is what determines the
    fetches a reader issues.
    """
    from domain.aggregate import KIND_QUANTILE_FUNCTION
    from domain.shard_format import (
        DESCRIPTOR_SIZE,
        TRAILER_SIZE,
        parse_index,
        parse_trailer,
        split_v2_tail,
    )

    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    planes = [_planes(1, seed=61 + index)[0] for index in range(spec.n_fields)]
    layout = layout_for_spec(spec, grid_lat=GRID_LAT, grid_lon=GRID_LON)
    container = encode_aggregate_shard(planes, layout)

    trailer = parse_trailer(container[-TRAILER_SIZE:])
    tail = container[-(trailer.index_byte_size + DESCRIPTOR_SIZE + TRAILER_SIZE) :]
    index_bytes, _descriptor = split_v2_tail(tail, trailer.num_chunks)
    entries = parse_index(index_bytes, trailer.num_chunks)

    # The location must have all four corners on-grid: GRID_LAT/LON give 2 x 2 chunks.
    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
        group = list(layout.spatial_group(row, col))
        assert len(group) == spec.n_fields
        offsets = [entries[ordinal][0] for ordinal in group]
        lengths = [entries[ordinal][1] for ordinal in group]
        assert all(
            offsets[index + 1] == offsets[index] + lengths[index]
            for index in range(len(group) - 1)
        ), (row, col, offsets)
