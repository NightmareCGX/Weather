"""Tests for the aggregate serving layer (api/services/aggregate_serving.py).

Two properties are the point of this layer, and both are asserted here:

* the statistics it reports carry the accuracy the encodings were chosen for -- measured
  against the ensemble's own sampling noise, not an invented tolerance;
* every failure falls through to the member path rather than returning a partial answer, so a
  caller cannot mistake a degraded result for a complete one.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from api.core.aggregate_reader import aggregate_shard_key
from api.services.aggregate_serving import (
    STATISTIC_NAMES,
    AggregatePointStatistics,
    exceedance_probability,
    statistics_from_aggregate,
    statistics_from_aggregate_at_cell,
)
from domain.aggregate import (
    KIND_MEAN_STD_BINS,
    KIND_QUANTILE_FUNCTION,
    MEMBER_COUNT_SCALE,
    NAN_SENTINEL,
    AggregateSpec,
    compute_aggregate,
    finite_member_count,
    quantise_field,
)
from domain.field_layout import aggregate_fields_for
from domain.models.cloud import cloud_ceiling_ensemble_summary
from domain.models.precipitation import (
    aggregate_ensemble_phase_support,
    classify_precipitation_phase,
)
from domain.models.wind import compute_consensus_vector, compute_wind_rose
from domain.shard_format import (
    ENCODING_I16,
    PER_FIELD_SCALE,
    ShardDescriptor,
    build_container_v2,
)
from domain.variable_class import spec_for

GRID_LAT, GRID_LON = 128, 160
CHUNK_LAT, CHUNK_LON = 100, 100
LEAD = 6


def _bin_spec() -> AggregateSpec:
    return AggregateSpec(n_bins=16)


def _field_scales(variable: str) -> tuple[float, ...]:
    """The stored-field steps for a variable, exactly as the reader derives them."""
    return aggregate_fields_for(variable).field_scales


def _encode(
    fields: np.ndarray,
    *,
    field_scales: tuple[float, ...],
    level: int = 5,
    member_count: int = 0,
) -> bytes:
    """Mirror the writer's chunking and fixed-point encoding from the shared format authority.

    Local to this suite because the API tier must not import the ingestion package; the
    cross-service agreement assertion lives in ``tests/contracts/``.

    The container is written the way production writes it -- int16 at the variable's per-field
    steps, declaring the per-field scale marker -- so that this suite exercises the reader's
    dequantisation rather than a path no store will ever produce.
    """
    from numcodecs import Zstd

    lat_chunks = -(-GRID_LAT // CHUNK_LAT)
    lon_chunks = -(-GRID_LON // CHUNK_LON)
    compressor = Zstd(level=level)
    codes = [
        quantise_field(plane, scale)
        for plane, scale in zip(fields, field_scales, strict=True)
    ]
    payloads: list[bytes] = []
    for row in range(lat_chunks):
        for col in range(lon_chunks):
            for plane in codes:
                buf = np.full((CHUNK_LAT, CHUNK_LON), NAN_SENTINEL, dtype=np.int16)
                r0, c0 = row * CHUNK_LAT, col * CHUNK_LON
                r1 = min(r0 + CHUNK_LAT, GRID_LAT)
                c1 = min(c0 + CHUNK_LON, GRID_LON)
                buf[: r1 - r0, : c1 - c0] = plane[r0:r1, c0:c1]
                payloads.append(compressor.encode(buf.tobytes(order="C")))
    num_chunks = len(payloads)
    descriptor = ShardDescriptor(
        encoding_id=ENCODING_I16,
        scale=PER_FIELD_SCALE,
        chunk_lat=CHUNK_LAT,
        chunk_lon=CHUNK_LON,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        num_chunks=num_chunks,
        index_byte_size=num_chunks * 16,
        member_count=member_count,
    )
    return build_container_v2(payloads, descriptor=descriptor)


def _store_with(
    tmp_path,
    variable: str,
    members: np.ndarray,
    *,
    member_count: int | None = None,
    group_members: dict[str, np.ndarray] | None = None,
) -> str:
    """Write an aggregate over ``members``, laid out the way the ingestion writer lays it out.

    The field vector is the per-cell member count, then the encoding's own fields, then the
    variable's supplementary groups. Which fields those are comes from
    ``domain.field_layout`` -- the same authority the writer uses -- so the container this
    produces is the one production produces.

    Args:
        tmp_path: Where to write the store.
        variable: Variable code.
        members: The variable's own ``(n_members, lat, lon)`` stack.
        member_count: The count recorded in the descriptor; defaults to the stack's.
        group_members: The *other* variables a group reads, keyed by variable code (the wind
            components, or the four flags). Absent means the groups are written as placeholders,
            which is what a distribution-only test wants.
    """
    from domain.field_layout import group_inputs
    from domain.product_fields import variable_group_fields

    layout = aggregate_fields_for(variable)
    fields: list[np.ndarray] = [finite_member_count(members)]
    if layout.distribution_slice.stop > layout.distribution_slice.start:
        fields.extend(compute_aggregate(members, spec_for(variable)))
    if layout.groups:
        supplied = group_members or {}
        # A group whose inputs are all in hand is computed for real: the censoring and fraction
        # groups read only the variable's own members, so a caller writing a cloud variable gets
        # its real counts without asking for anything extra. A group that reads *another*
        # variable and was not given it is written as a placeholder, which is what a
        # distribution-only test over a wind variable wants.
        if all(
            name in supplied or name == variable
            for group in layout.groups
            for name in group_inputs(group)[0]
        ):
            fields.append(
                variable_group_fields(variable, members=members, extra_members=supplied)
            )
        else:
            fields.append(
                np.zeros(
                    (layout.n_fields - len(fields), GRID_LAT, GRID_LON), dtype=np.float32
                )
            )
    stacked = np.concatenate(
        [field[None] if field.ndim == 2 else field for field in fields]
    )
    assert stacked.shape[0] == layout.n_fields, stacked.shape
    counted = int(members.shape[0]) if member_count is None else member_count
    blob = _encode(stacked, field_scales=layout.field_scales, member_count=counted)
    key = aggregate_shard_key(variable, LEAD)
    store = str(tmp_path)
    full = os.path.join(store, *key.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(blob)
    return store


def _fetch(variable: str, store: str, lat_idx: int, lon_idx: int):
    return statistics_from_aggregate_at_cell(
        variable, store_path=store, lead_time_hours=LEAD, lat_idx=lat_idx, lon_idx=lon_idx
    )


# ---------------------------------------------------------------------------
# Exactness follows the encoding
# ---------------------------------------------------------------------------


def test_bin_encoding_reports_mean_and_spread_as_exact(tmp_path) -> None:
    """The bin encoding stores MEAN and STD as planes, so those need no reconstruction."""
    rng = np.random.default_rng(1)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    store = _store_with(tmp_path, "temperature_2m", members)

    result = _fetch("temperature_2m", store, 5, 7)
    assert result is not None
    assert result.spec.kind == KIND_MEAN_STD_BINS
    assert result.exact == frozenset({"mean", "spread"})
    # "Exact" means stored as its own plane rather than reconstructed from the shape, so the
    # only difference from the member-derived value is the field's fixed-point step.
    column = members[:, 5, 7]
    mean_step = aggregate_fields_for("temperature_2m").field_scales[1]
    assert result.values["mean"] == pytest.approx(float(column.mean()), abs=mean_step)
    assert result.values["spread"] == pytest.approx(float(column.std()), abs=mean_step)


def test_quantile_encoding_reports_the_stored_levels_as_exact(tmp_path) -> None:
    """P10, P50, the median and P90 are stored levels; the rest are reconstructed."""
    rng = np.random.default_rng(2)
    members = np.where(
        rng.random((30, GRID_LAT, GRID_LON)) < 0.5,
        0.0,
        rng.gamma(0.35, 1.5, (30, GRID_LAT, GRID_LON)),
    ).astype(np.float32)
    store = _store_with(tmp_path, "precipitation_amount_3h", members)

    result = _fetch("precipitation_amount_3h", store, 5, 7)
    assert result is not None
    assert result.spec.kind == KIND_QUANTILE_FUNCTION
    assert result.exact == frozenset({"median", "p10", "p50", "p90"})

    column = members[:, 5, 7]
    level_step = aggregate_fields_for("precipitation_amount_3h").field_scales[1]
    for name, probability in (("p10", 10), ("p50", 50), ("p90", 90)):
        assert result.values[name] == pytest.approx(
            float(np.percentile(column, probability)), abs=level_step
        )


@pytest.mark.parametrize(
    ("variable", "seed", "maker"),
    (
        ("temperature_2m", 3, lambda r: r.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON))),
        (
            "precipitation_amount_3h",
            4,
            lambda r: np.where(
                r.random((30, GRID_LAT, GRID_LON)) < 0.5,
                0.0,
                r.gamma(0.35, 1.5, (30, GRID_LAT, GRID_LON)),
            ),
        ),
        (
            "wind_gust",
            5,
            lambda r: np.abs(r.standard_t(2.5, (30, GRID_LAT, GRID_LON))) * 20.0,
        ),
        (
            "cloud_cover_3h",
            6,
            lambda r: r.uniform(0.0, 100.0, (30, GRID_LAT, GRID_LON)),
        ),
    ),
)
def test_served_statistics_are_within_the_ensembles_own_noise(
    tmp_path, variable: str, seed: int, maker
) -> None:
    """The acceptance yardstick is the ensemble's sampling noise at 30 members.

    A point-value tolerance would be the wrong measure: these are statistics estimated from a
    finite ensemble, so the honest question is whether the aggregate answer is distinguishable
    from the answer a different 30 members would have given.
    """
    rng = np.random.default_rng(seed)
    members = np.asarray(maker(rng), dtype=np.float32)
    store = _store_with(tmp_path, variable, members)

    result = _fetch(variable, store, 5, 7)
    assert result is not None
    column = members[:, 5, 7]
    half = 15
    mean_noise = abs(float(column[:half].mean()) - float(column[half:].mean()))
    spread_noise = abs(float(column[:half].std()) - float(column[half:].std()))

    mean_err = abs(result.values["mean"] - float(column.mean()))
    spread_err = abs(result.values["spread"] - float(column.std()))
    assert mean_err <= max(mean_noise, 1e-6), (variable, mean_err, mean_noise)
    assert spread_err <= max(spread_noise, 1e-6), (variable, spread_err, spread_noise)


def test_statistics_kwargs_shape_matches_the_platform_contract(tmp_path) -> None:
    """Every served statistic has a slot, and unexpressible ones are absent rather than zero."""
    rng = np.random.default_rng(7)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    store = _store_with(tmp_path, "temperature_2m", members)
    result = _fetch("temperature_2m", store, 5, 7)
    assert result is not None
    kwargs = result.as_ensemble_statistics_kwargs()
    assert set(kwargs) == set(STATISTIC_NAMES)
    assert kwargs["mean"] is not None
    # the bin encoding reconstructs percentiles, so they are present but not exact
    assert kwargs["p50"] is not None




# ---------------------------------------------------------------------------
# Thresholds: the reason the quantile encoding exists
# ---------------------------------------------------------------------------


def test_exceedance_answers_a_threshold_the_store_never_saw(tmp_path) -> None:
    """The store was never told the threshold; the encoding answers it anyway."""
    rng = np.random.default_rng(8)
    members = np.where(
        rng.random((30, GRID_LAT, GRID_LON)) < 0.5,
        0.0,
        rng.gamma(0.35, 1.5, (30, GRID_LAT, GRID_LON)),
    ).astype(np.float32)
    store = _store_with(tmp_path, "precipitation_amount_3h", members)
    column = members[:, 5, 7]

    for threshold in (0.1, 1.0, 5.0):
        probability = exceedance_probability(
            "precipitation_amount_3h",
            store_path=store,
            lead_time_hours=LEAD,
            threshold=threshold,
            operator="gt",
            chunk_row=0,
            chunk_col=0,
            row_in_chunk=5,
            col_in_chunk=7,
        )
        assert probability is not None
        truth = float((column > threshold).mean())
        assert abs(probability.probability - truth) <= 0.05, (
            threshold,
            probability,
            truth,
        )
        # the interval the API reports needs a sample size, and the container carries it
        assert probability.member_count == 30


def test_exceedance_operator_complement_is_consistent(tmp_path) -> None:
    rng = np.random.default_rng(9)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    store = _store_with(tmp_path, "temperature_2m", members)
    common = dict(
        store_path=store,
        lead_time_hours=LEAD,
        threshold=281.0,
        chunk_row=0,
        chunk_col=0,
        row_in_chunk=5,
        col_in_chunk=7,
    )
    above = exceedance_probability("temperature_2m", operator="gt", **common)
    below = exceedance_probability("temperature_2m", operator="lt", **common)
    assert above is not None
    assert below is not None
    assert above.probability + below.probability == pytest.approx(1.0, abs=1e-6)


def test_exceedance_refuses_the_inclusive_operators(tmp_path) -> None:
    """``gte``/``lte`` need the point mass at the threshold, which a quantile function cannot hold.

    Measured on a zero-inflated field at its natural threshold of 0: the member path reports
    P(x >= 0) = 1.0 against P(x > 0) = 0.47, and that 0.53 gap *is* the atom. The aggregate's
    continuous interpolation gives 0.50, which is neither. Refusing is the only honest answer,
    so the caller serves the members, where the distinction is well defined.
    """
    rng = np.random.default_rng(10)
    members = np.where(
        rng.random((30, GRID_LAT, GRID_LON)) < 0.5,
        0.0,
        rng.gamma(0.35, 1.5, (30, GRID_LAT, GRID_LON)),
    ).astype(np.float32)
    store = _store_with(tmp_path, "precipitation_amount_3h", members)
    common = dict(
        store_path=store,
        lead_time_hours=LEAD,
        threshold=0.0,
        chunk_row=0,
        chunk_col=0,
        row_in_chunk=5,
        col_in_chunk=7,
    )
    assert exceedance_probability("precipitation_amount_3h", operator="gte", **common) is None
    assert exceedance_probability("precipitation_amount_3h", operator="lte", **common) is None
    assert exceedance_probability("precipitation_amount_3h", operator="between", **common) is None
    # The strict operator is served, but only to the resolution the stored levels allow: the
    # atom at zero makes the CDF jump between the 0.30 and 0.40 levels, and linear interpolation
    # inside that gap cannot place the jump more finely than the gap itself. That is the bound
    # asserted here -- not a tolerance chosen to make the test pass, but the stated limit of a
    # 19-level sampled inverse CDF.
    strict = exceedance_probability("precipitation_amount_3h", operator="gt", **common)
    assert strict is not None
    levels = spec_for("precipitation_amount_3h").quantile_levels()
    widest_gap = max(b - a for a, b in zip(levels, levels[1:], strict=False))
    column = members[:, 5, 7]
    assert strict.probability == pytest.approx(
        float((column > 0.0).mean()), abs=widest_gap
    )
    assert strict.member_count == 30


# ---------------------------------------------------------------------------
# Fallback: absent and unusable both mean "use the members"
# ---------------------------------------------------------------------------


def test_unclassified_variable_falls_through(tmp_path) -> None:
    assert _fetch("mystery_variable", str(tmp_path), 0, 0) is None


def test_missing_aggregate_falls_through(tmp_path) -> None:
    assert _fetch("temperature_2m", str(tmp_path), 0, 0) is None


def test_field_count_mismatch_falls_through(tmp_path) -> None:
    """A container written with a different spec must not be read as this variable's fields.

    The field order is a property of the spec, so a mismatched count means the store disagrees
    with the approved encoding. Guessing an order would return plausible numbers under the
    wrong names.
    """
    rng = np.random.default_rng(11)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    spec = _bin_spec()  # 18 statistic fields; temperature_2m's stored vector holds 35
    fields = np.concatenate(
        [finite_member_count(members)[None], compute_aggregate(members, spec)]
    )
    key = aggregate_shard_key("temperature_2m", LEAD)
    store = str(tmp_path)
    full = os.path.join(store, *key.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    # Encoded at the spec's own scales, so the only thing wrong with it is that it is not this
    # variable's layout -- which is exactly the case the field-count check exists to catch.
    with open(full, "wb") as handle:
        handle.write(
            _encode(fields, field_scales=(MEMBER_COUNT_SCALE, *spec.field_scales))
        )

    assert _fetch("temperature_2m", store, 5, 7) is None


def test_all_nan_aggregate_falls_through(tmp_path) -> None:
    """A fill-only location carries no information, and the members can do better."""
    layout = aggregate_fields_for("temperature_2m")
    # Every field is the NaN sentinel, including the member count: a fill-only location.
    fields = np.full((layout.n_fields, GRID_LAT, GRID_LON), np.nan, dtype=np.float32)
    key = aggregate_shard_key("temperature_2m", LEAD)
    store = str(tmp_path)
    full = os.path.join(store, *key.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(_encode(fields, field_scales=layout.field_scales))

    assert _fetch("temperature_2m", store, 5, 7) is None


def test_off_grid_cell_falls_through(tmp_path) -> None:
    rng = np.random.default_rng(12)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    store = _store_with(tmp_path, "temperature_2m", members)
    assert _fetch("temperature_2m", store, GRID_LAT, 0) is None
    assert _fetch("temperature_2m", store, 0, GRID_LON) is None


def test_an_unparseable_container_falls_through(tmp_path) -> None:
    """A truncated object is unreadable, which is the same outcome as absent.

    The two failure kinds are deliberately distinct, and this is the benign one: the tail is
    gone, so nothing can be parsed and the answer is "no aggregate here". The members serve.
    """
    rng = np.random.default_rng(13)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    store = _store_with(tmp_path, "temperature_2m", members)
    key = aggregate_shard_key("temperature_2m", LEAD)
    full = os.path.join(store, *key.split("/"))
    with open(full, "rb") as handle:
        blob = handle.read()
    with open(full, "wb") as handle:
        handle.write(blob[: len(blob) // 2])

    assert _fetch("temperature_2m", store, 5, 7) is None


def test_a_container_whose_index_disagrees_with_its_payload_surfaces(tmp_path) -> None:
    """A parseable but self-inconsistent container must reach an operator.

    The difference from the case above matters: here the tail parses and the geometry is
    coherent, so the only thing wrong is that an index entry points past the payload. Falling
    through would hide genuine corruption behind a working-looking member answer, so this
    layer must let the reader's error out.
    """
    import struct

    from domain.shard_format import DESCRIPTOR_SIZE, ShardFormatError
    from domain.variable_class import spec_for

    rng = np.random.default_rng(15)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    store = _store_with(tmp_path, "temperature_2m", members)
    key = aggregate_shard_key("temperature_2m", LEAD)
    full = os.path.join(store, *key.split("/"))
    with open(full, "rb") as handle:
        blob = bytearray(handle.read())

    lat_chunks = -(-GRID_LAT // CHUNK_LAT)
    lon_chunks = -(-GRID_LON // CHUNK_LON)
    num_chunks = lat_chunks * lon_chunks * spec_for("temperature_2m").n_fields
    index_start = len(blob) - DESCRIPTOR_SIZE - 12 - num_chunks * 16
    struct.pack_into("<Q", blob, index_start + 8, 1 << 20)
    with open(full, "wb") as handle:
        handle.write(bytes(blob))

    with pytest.raises(ShardFormatError):
        _fetch("temperature_2m", store, 5, 7)


def test_explicit_entry_point_accepts_a_reader(tmp_path) -> None:
    """A caller holding a reader can reuse it, so a series request pays one client setup."""
    from api.core.aggregate_reader import AggregateShardReader

    rng = np.random.default_rng(14)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    store = _store_with(tmp_path, "temperature_2m", members)
    result = statistics_from_aggregate(
        "temperature_2m",
        store_path=store,
        lead_time_hours=LEAD,
        chunk_row=0,
        chunk_col=0,
        row_in_chunk=5,
        col_in_chunk=7,
        reader=AggregateShardReader(store),
    )
    assert isinstance(result, AggregatePointStatistics)


# ---------------------------------------------------------------------------
# The per-point member count, which an aggregate cannot derive from its statistics
# ---------------------------------------------------------------------------


def test_observed_point_reports_the_stored_member_count(tmp_path) -> None:
    """The container's count is the API's per-point count wherever the point is observed."""
    rng = np.random.default_rng(21)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    store = _store_with(tmp_path, "temperature_2m", members)

    result = _fetch("temperature_2m", store, 5, 7)
    assert result is not None
    assert result.member_count == 30


def test_a_cell_one_member_short_reports_29_and_still_serves(tmp_path) -> None:
    """Missing members are skipped, and the count says how many were used.

    This is what the serving paths do for every variable: filter to finite members, then apply
    the per-cell coverage floor. The count travels as field 0 of the container precisely so the
    aggregate can report "29 of 30 here" -- a cell the member path would happily describe, and
    so one the aggregate must not refuse.

    The three counts below are independent: the recorded count of the missing member's cell, the
    recorded count of its neighbour, and what a caller reads back.
    """
    rng = np.random.default_rng(22)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    members[7, 5, 7] = np.nan
    store = _store_with(tmp_path, "temperature_2m", members)

    short = _fetch("temperature_2m", store, 5, 7)
    assert short is not None
    assert short.member_count == 29
    # The mean is that of the 29, not of a 30-member sample with a hole in it.
    present = members[np.isfinite(members[:, 5, 7]), 5, 7]
    step = aggregate_fields_for("temperature_2m").field_scales[1]
    assert short.values["mean"] == pytest.approx(float(present.mean()), abs=step)

    # A neighbouring, fully-observed cell reports the full count.
    full = _fetch("temperature_2m", store, 5, 8)
    assert full is not None
    assert full.member_count == 30


def test_a_cell_below_the_floor_has_no_count_and_no_statistics(tmp_path) -> None:
    """Below the coverage floor there is no aggregate, and the count field says so.

    The distinction matters: "no members participated" is a claim about the data, while "this
    cell was refused" is a claim about the aggregate. Only the second is true here, so the
    reader reports no count rather than zero.
    """
    rng = np.random.default_rng(23)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    members[24:, 5, 7] = np.nan  # 24 of 30, below 85%
    store = _store_with(tmp_path, "temperature_2m", members)

    assert _fetch("temperature_2m", store, 5, 7) is None


def test_member_count_is_read_from_the_field_not_inferred(tmp_path) -> None:
    """The count is a stored per-cell field, so a caller reads what the writer recorded.

    A container-wide count in the descriptor could only say the total; the count the API reports
    varies by cell, which is why it is field 0 of the field vector and why this reads the same
    byte range as the statistics rather than a second object.
    """
    from api.core.aggregate_reader import AggregateShardReader

    rng = np.random.default_rng(24)
    members = rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    members[9, 5, 7] = np.nan
    store = _store_with(tmp_path, "temperature_2m", members)
    reader = AggregateShardReader(store)
    at = lambda row, col: {  # noqa: E731 - a local shorthand for six call sites
        "chunk_row": 0,
        "chunk_col": 0,
        "row_in_chunk": row,
        "col_in_chunk": col,
    }
    assert reader.member_count("temperature_2m", LEAD, **at(5, 7)) == 29
    assert reader.member_count("temperature_2m", LEAD, **at(5, 8)) == 30
    # An absent aggregate is "unknown", not zero.
    assert reader.member_count("wind_gust", LEAD, **at(5, 7)) is None


# ---------------------------------------------------------------------------
# The capability check the endpoints consult before they try an aggregate
# ---------------------------------------------------------------------------


def test_the_stored_products_answer_every_field_that_used_to_need_members(tmp_path) -> None:
    """The supplementary groups are why no variable is special any more.

    Each of these products is a function of the *per-member* values -- a (u, v) pair, a 0/1
    phase flag, a censoring rule applied before summarising -- so none is recoverable from a
    distribution. They are stored as fields for exactly that reason, and this reads them back.
    """
    from api.services.aggregate_serving import (
        SPECIAL_PER_MEMBER_VARIABLES,
        cloud_censoring_at_cell,
        precipitation_phase_from_aggregate,
        wind_products_from_aggregate,
    )

    assert SPECIAL_PER_MEMBER_VARIABLES == frozenset(), (
        "every special variable has stored fields now; a name here would mean a response field "
        "no container can answer"
    )

    rng = np.random.default_rng(31)
    u = rng.normal(3.0, 4.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    v = rng.normal(2.0, 4.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    wind_store = _store_with(
        tmp_path / "wind",
        "wind_10m",
        u,
        group_members={"wind_u_10m": u, "wind_v_10m": v},
    )
    products = wind_products_from_aggregate(
        store_path=wind_store,
        lead_time_hours=LEAD,
        chunk_row=0,
        chunk_col=0,
        row_in_chunk=5,
        col_in_chunk=7,
    )
    assert products is not None
    reference = compute_consensus_vector(u[:, 5, 7], v[:, 5, 7])
    assert products["consensus"]["speed_mps"] == pytest.approx(reference.speed_mps, abs=0.02)
    assert products["consensus"]["cardinal"] == reference.cardinal
    # The stored direction is a sin/cos pair at a 0.001 step, so atan2's recovery error is
    # bounded by the pair's quantisation: measured over a full sweep, 0.04 degrees worst case.
    if reference.direction_deg is not None:
        assert products["consensus"]["direction_deg"] == pytest.approx(
            reference.direction_deg, abs=0.05
        )
    rose_reference = compute_wind_rose(u[:, 5, 7], v[:, 5, 7])
    assert products["rose"]["calm_count"] == rose_reference.calm_count
    by_sector = {sector["sector"]: sector for sector in products["rose"]["sectors"]}
    for sector in rose_reference.sectors:
        assert by_sector[sector.sector]["count"] == sector.count, sector.sector

    amounts = np.where(
        rng.random((30, GRID_LAT, GRID_LON)) < 0.5,
        0.0,
        rng.gamma(0.4, 1.5, (30, GRID_LAT, GRID_LON)),
    ).astype(np.float32)
    flags = {
        name: (rng.random((30, GRID_LAT, GRID_LON)) < 0.4).astype(np.float32)
        for name in ("crain", "csnow", "cfrzr", "cicep")
    }
    precip_store = _store_with(
        tmp_path / "precip",
        "precipitation_amount_3h",
        amounts,
        group_members=flags,
    )
    phase = precipitation_phase_from_aggregate(
        store_path=precip_store,
        lead_time_hours=LEAD,
        chunk_row=0,
        chunk_col=0,
        row_in_chunk=5,
        col_in_chunk=7,
    )
    assert phase is not None
    states = [
        classify_precipitation_phase(
            float(amounts[member, 5, 7]),
            {name: int(plane[member, 5, 7] >= 0.5) for name, plane in flags.items()},
        )
        for member in range(30)
    ]
    expected_support = aggregate_ensemble_phase_support(states)
    for name, value in expected_support.items():
        if name.value in phase["phase_support"]:
            assert phase["phase_support"][name.value] == pytest.approx(value, abs=0.002)

    ceiling = np.where(
        rng.random((30, GRID_LAT, GRID_LON)) < 0.4,
        np.float32(20.0),
        rng.uniform(0.2, 12.0, (30, GRID_LAT, GRID_LON)),
    ).astype(np.float32)
    cloud_store = _store_with(tmp_path / "cloud", "cloud_ceiling", ceiling)
    censoring = cloud_censoring_at_cell(
        "cloud_ceiling", store_path=cloud_store, lead_time_hours=LEAD, lat_idx=5, lon_idx=7
    )
    assert censoring is not None
    summary = cloud_ceiling_ensemble_summary(ceiling[:, 5, 7])
    assert summary is not None
    # The counts are integers stored at a step of one member, so they come back exactly.
    assert censoring["valid_member_count"] == summary.valid_member_count
    assert censoring["finite_member_count"] == summary.finite_member_count
    assert censoring["unlimited_member_count"] == summary.unlimited_member_count
    assert censoring["unlimited_probability"] == pytest.approx(
        summary.unlimited_probability, abs=1e-4
    )
    # And the conditional statistics are the writer's, computed over the finite members.
    assert censoring["conditional"]["p50"] == pytest.approx(
        summary.conditional_percentiles["p50"], abs=0.02
    )


def test_a_cell_with_no_rose_reports_an_all_calm_rose_rather_than_nothing(tmp_path) -> None:
    """An all-calm cell has a rose: every member is in it, and none has a direction.

    The reference's own convention: a calm member is counted as calm and contributes to no
    sector, so the rose's cells sum to zero and the calm count is the whole member set. The
    consensus direction is absent -- there is no direction to report -- but the counts are a
    definite answer and a caller reading them should get it.
    """
    from api.services.aggregate_serving import wind_products_from_aggregate

    u = np.full((30, GRID_LAT, GRID_LON), 0.1, dtype=np.float32)
    v = np.full((30, GRID_LAT, GRID_LON), 0.1, dtype=np.float32)
    store = _store_with(
        tmp_path, "wind_10m", u, group_members={"wind_u_10m": u, "wind_v_10m": v}
    )
    products = wind_products_from_aggregate(
        store_path=store,
        lead_time_hours=LEAD,
        chunk_row=0,
        chunk_col=0,
        row_in_chunk=5,
        col_in_chunk=7,
    )
    assert products is not None
    assert products["consensus"]["direction_deg"] is None
    assert products["consensus"]["cardinal"] == "CALM"
    assert products["rose"]["calm_count"] == 30
    assert products["rose"]["calm_probability"] == pytest.approx(1.0)
    assert sum(sector["probability"] for sector in products["rose"]["sectors"]) == 0.0


def test_special_variables_are_never_served_from_an_aggregate() -> None:
    """The list of variables with no stored answer is empty, and stays checked.

    Every response field used to need the members: ``wind_10m``'s consensus vector and rose,
    ``precipitation_amount_3h``'s phase support, and the two cloud variables' censoring. They
    are stored fields now (``domain.field_layout``'s supplementary groups), read back by the
    functions asserted above. The set is kept so a future exception has a home -- and so this
    fails loudly if one is added without the fields that would answer it.
    """
    from api.services.aggregate_serving import (
        SPECIAL_PER_MEMBER_VARIABLES,
        aggregate_can_answer,
    )

    assert SPECIAL_PER_MEMBER_VARIABLES == frozenset()
    # a variable that is aggregated in the ordinary way still passes
    assert aggregate_can_answer("temperature_2m") is True


def test_requesting_members_disqualifies_the_aggregate_path() -> None:
    """``include_members=true`` asks for the raw sample, which the aggregate has collapsed."""
    from api.services.aggregate_serving import aggregate_can_answer

    assert aggregate_can_answer("temperature_2m", needs_members=True) is False
    # Stored products do not change this: the products and the members are different things, and
    # a response that asks for both has to be answered from the members.
    assert aggregate_can_answer("wind_10m", needs_members=True) is False


def test_only_the_strict_operators_can_be_answered() -> None:
    from api.services.aggregate_serving import aggregate_can_answer

    for operator in ("gt", "lt"):
        assert aggregate_can_answer("temperature_2m", operator=operator) is True
    for operator in ("gte", "lte", "between"):
        assert aggregate_can_answer("temperature_2m", operator=operator) is False


def test_an_unclassified_variable_cannot_be_answered() -> None:
    from api.services.aggregate_serving import aggregate_can_answer

    assert aggregate_can_answer("mystery_variable") is False


def test_an_unreadable_store_is_not_an_error_for_the_aggregate_path() -> None:
    """The gate raises when a run is not readable; the aggregate path must not propagate it.

    The aggregate is an optimisation over the member shards, so "this run is not currently
    readable" has to reach the member path's own error handling rather than becoming an error
    raised from a probe the caller only ran as a shortcut.
    """
    from api.services.aggregate_serving import try_read_aggregate

    def _refuses() -> int | None:
        raise FileNotFoundError("run 's3://x/y' is not a ready, readable run")

    assert try_read_aggregate(_refuses) is None
    assert try_read_aggregate(lambda: 7) == 7
    assert try_read_aggregate(lambda: None) is None

    # A damaged container is a real fault and must stay visible.
    from domain.shard_format import ShardFormatError

    def _damaged() -> int | None:
        raise ShardFormatError("chunk 0 failed to decode")

    with pytest.raises(ShardFormatError):
        try_read_aggregate(_damaged)
