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
    tmp_path, variable: str, members: np.ndarray, *, member_count: int | None = None
) -> str:
    """Write an aggregate over ``members``, laid out the way the ingestion writer lays it out.

    The field vector is the per-cell member count, then the encoding's own fields, then one plane
    per supplementary group field -- placeholder zeros here, because this suite exercises the
    distribution path and only the field *count* has to match the layout the reader will check
    against. Every field is stored at its own fixed-point step.
    """
    spec = spec_for(variable)
    layout = aggregate_fields_for(variable)
    distribution = np.concatenate(
        [finite_member_count(members)[None], compute_aggregate(members, spec)]
    )
    placeholders = np.zeros(
        (layout.n_fields - distribution.shape[0], GRID_LAT, GRID_LON), dtype=np.float32
    )
    fields = np.concatenate([distribution, placeholders])
    counted = int(members.shape[0]) if member_count is None else member_count
    blob = _encode(fields, field_scales=layout.field_scales, member_count=counted)
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


def test_special_variables_are_never_served_from_an_aggregate() -> None:
    """Four variables carry per-member answers a collapsed distribution has already lost.

    ``wind_10m`` computes a consensus vector and a wind rose from each member's (u, v) pair;
    ``precipitation_amount_3h`` a phase-support map from each member's 0/1 flag;
    ``cloud_ceiling`` and ``cloud_cover_3h`` censor members individually before summarising.
    None of those is a function of the distribution, so no aggregate can produce them.
    """
    from api.services.aggregate_serving import SPECIAL_PER_MEMBER_VARIABLES, aggregate_can_answer

    for variable in sorted(SPECIAL_PER_MEMBER_VARIABLES):
        assert aggregate_can_answer(variable) is False, variable
    # a variable that is aggregated in the ordinary way still passes
    assert aggregate_can_answer("temperature_2m") is True


def test_requesting_members_disqualifies_the_aggregate_path() -> None:
    """``include_members=true`` asks for the raw sample, which the aggregate has collapsed."""
    from api.services.aggregate_serving import aggregate_can_answer

    assert aggregate_can_answer("temperature_2m", needs_members=True) is False


def test_only_the_strict_operators_can_be_answered() -> None:
    from api.services.aggregate_serving import aggregate_can_answer

    for operator in ("gt", "lt"):
        assert aggregate_can_answer("temperature_2m", operator=operator) is True
    for operator in ("gte", "lte", "between"):
        assert aggregate_can_answer("temperature_2m", operator=operator) is False


def test_an_unclassified_variable_cannot_be_answered() -> None:
    from api.services.aggregate_serving import aggregate_can_answer

    assert aggregate_can_answer("wind_10m") is False  # special, and therefore refused first
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
