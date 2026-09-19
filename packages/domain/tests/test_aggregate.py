"""Unit tests for ensemble aggregate encodings (packages/domain/src/domain/aggregate.py).

The encoding is the contract a writer and every reader must reproduce bit for bit, so the
tests here pin the reference arithmetic rather than only the shape of the output: the exact
bin convention, the exact normalisation floor, and the fact that the float ops are the ones
that were measured to agree with the stored data.
"""

from __future__ import annotations

import numpy as np
import pytest
from domain.aggregate import (
    BIN_SCALE,
    DEFAULT_N_BINS,
    DEFAULT_QUANTILE_LEVELS,
    KIND_MEAN_STD_BINS,
    KIND_QUANTILE_FUNCTION,
    MEAN_SCALE,
    NAN_SENTINEL,
    QUANTILE_SCALE,
    STD_SCALE,
    AggregateError,
    AggregateSpec,
    check_quantisation_range,
    compute_aggregate,
    decode_aggregate,
    dequantise_field,
    encode_aggregate,
    exceedance_from_bins,
    exceedance_from_quantiles,
    quantile_at,
    quantile_function_moments,
    quantise_field,
)

N_MEMBERS = 30
LAT, LON = 8, 6


def _spec(**overrides) -> AggregateSpec:
    base = {"kind": "mean_std_bins", "n_bins": 8, "sigma_range": 4.0}
    base.update(overrides)
    return AggregateSpec(**base)  # type: ignore[arg-type]


def _members(seed: int = 0, n: int = N_MEMBERS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(280.0, 8.0, (LAT, LON)).astype(np.float32)
    return np.stack([base + rng.normal(0, 1.5, (LAT, LON)) for _ in range(n)]).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def test_spec_field_names_and_scales_are_positional_contracts() -> None:
    """Field order is the storage order; a reader depends on both lists agreeing."""
    spec = _spec(n_bins=4)
    assert spec.field_names == ("MEAN", "STD", "BIN00", "BIN01", "BIN02", "BIN03")
    assert spec.field_scales == (MEAN_SCALE, STD_SCALE, BIN_SCALE, BIN_SCALE, BIN_SCALE, BIN_SCALE)
    assert spec.n_fields == len(spec.field_names) == len(spec.field_scales) == 6


def test_default_spec_is_the_measured_choice() -> None:
    """32 bins is the measured sweet spot; a silent default change would move every store."""
    assert AggregateSpec().n_bins == DEFAULT_N_BINS == 32
    assert AggregateSpec().sigma_range == 4.0


def test_spec_rejects_unknown_kind_and_invalid_parameters() -> None:
    with pytest.raises(AggregateError, match="unknown aggregate kind"):
        _spec(kind="qfunc19")
    with pytest.raises(AggregateError, match="n_bins must be positive"):
        _spec(n_bins=0)
    for bad in (0.0, -1.0):
        with pytest.raises(AggregateError, match="sigma_range must be positive"):
            _spec(sigma_range=bad)


def test_bin_edges_are_float32_and_span_the_sigma_range() -> None:
    """The dtype is load-bearing: the comparison against these edges is float32."""
    spec = _spec(n_bins=8)
    edges = spec.bin_edges()
    assert edges.dtype == np.float32
    assert len(edges) == spec.n_bins + 1
    assert edges[0] == np.float32(-4.0)
    assert edges[-1] == np.float32(4.0)
    # evenly spaced, so cumulative CDF reads can assume a constant width
    assert np.allclose(np.diff(edges), np.diff(edges)[0], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Reference arithmetic
# ---------------------------------------------------------------------------


def test_compute_aggregate_matches_the_reference_bit_for_bit() -> None:
    """Pin the arithmetic against an independent transcription of the measured reference.

    The two departures that were tried and rejected -- an arithmetic bin index, and a
    reciprocal multiply for the normalisation -- both disagreed with this on real data, so
    the reference is restated here rather than trusted to hold by construction.
    """
    stack = _members(seed=1)
    spec = _spec(n_bins=8)

    edges = np.linspace(-4.0, 4.0, 9).astype(np.float32)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    z = (stack - mean[None]) / np.maximum(std[None], np.float32(1e-9))
    expected = np.stack(
        [
            mean.astype(np.float32),
            std.astype(np.float32),
            *[
                (np.sum((z >= edges[i]) & (z < edges[i + 1]), axis=0) / N_MEMBERS).astype(
                    np.float32
                )
                for i in range(8)
            ],
        ]
    )

    assert np.array_equal(compute_aggregate(stack, spec), expected)


def test_compute_aggregate_mean_and_std_are_plain_moments() -> None:
    """``mean``/``std``, not the NaN-aware forms: the model fields have no missing members."""
    stack = _members(seed=2)
    got = compute_aggregate(stack, _spec())
    assert np.array_equal(got[0], np.mean(stack, axis=0).astype(np.float32))
    assert np.array_equal(got[1], np.std(stack, axis=0).astype(np.float32))
    assert np.array_equal(got[0], np.nanmean(stack, axis=0).astype(np.float32))
    assert np.array_equal(got[1], np.nanstd(stack, axis=0).astype(np.float32))


def test_bins_sum_to_one_when_all_members_are_inside_the_support() -> None:
    """A fully covered cell's bins are a complete probability mass."""
    stack = _members(seed=3)
    spec = _spec(n_bins=32)
    got = compute_aggregate(stack, spec)
    total = got[2:].sum(axis=0)
    assert np.allclose(total, 1.0, atol=1e-6)


def test_bins_under_count_members_outside_the_sigma_support() -> None:
    """Members beyond +-sigma_range fall outside every bin.

    This is a property of the bounded support, not an implementation defect, and it is also
    why the stored bins cannot answer very rare exceedance thresholds. Pinned so the
    limitation stays visible rather than being mistaken for a rounding bug.
    """
    stack = np.full((N_MEMBERS, LAT, LON), 280.0, dtype=np.float32)
    # one member far outside any plausible +-4 sigma window relative to the others
    stack[0] = 400.0
    got = compute_aggregate(stack, _spec(n_bins=8))
    total = got[2:].sum(axis=0)
    assert np.all(total < 1.0)
    assert np.allclose(total, (N_MEMBERS - 1) / N_MEMBERS, atol=1e-6)


def test_compute_aggregate_propagates_nan_through_the_normalisation() -> None:
    """A partially observed cell has no well-defined ensemble statistic."""
    stack = _members(seed=4)
    stack[0, 0, 0] = np.nan
    got = compute_aggregate(stack, _spec(n_bins=4))
    assert np.all(np.isnan(got[:, 0, 0]))
    assert not np.isnan(got[:, 1, 1]).any()


def test_zero_spread_cell_stays_finite() -> None:
    """All members identical gives zero std; the floor must keep the bins finite."""
    stack = np.full((N_MEMBERS, LAT, LON), 285.0, dtype=np.float32)
    got = compute_aggregate(stack, _spec(n_bins=4))
    assert np.isfinite(got).all()
    assert np.allclose(got[1], 0.0)
    # z is 0 for every member, which lands in the bin containing 0
    edges = _spec(n_bins=4).bin_edges()
    zero_bin = int(np.searchsorted(edges, 0.0, side="right") - 1)
    assert got[2 + zero_bin, 0, 0] == 1.0
    assert got[2:].sum(axis=0)[0, 0] == 1.0


def test_compute_aggregate_rejects_wrong_shape() -> None:
    with pytest.raises(AggregateError, match="must be"):
        compute_aggregate(np.zeros((LAT, LON), dtype=np.float32), _spec())
    with pytest.raises(AggregateError, match="at least one member"):
        compute_aggregate(np.zeros((0, LAT, LON), dtype=np.float32), _spec())


def test_compute_aggregate_accepts_an_integer_member_stack() -> None:
    """Members arrive as float32 in production, but the encode must not depend on that."""
    stack = np.zeros((3, 2, 2), dtype=np.int32)
    got = compute_aggregate(stack, _spec(n_bins=2))
    assert got.dtype == np.float32


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------


def test_quantise_and_dequantise_round_trip_within_half_a_step() -> None:
    values = np.array([0.0, 1.0, -1.0, 327.67, -327.67], dtype=np.float32)
    codes = quantise_field(values, 0.01)
    back = dequantise_field(codes, 0.01)
    assert codes.dtype == np.int16
    assert np.allclose(back, values, atol=0.005)


def test_quantise_preserves_nan_as_the_sentinel_and_back() -> None:
    values = np.array([1.0, np.nan, -2.0], dtype=np.float32)
    codes = quantise_field(values, 0.01)
    assert codes[1] == NAN_SENTINEL
    assert np.isnan(dequantise_field(codes, 0.01)[1])
    assert np.array_equal(dequantise_field(codes, 0.01)[[0, 2]], values[[0, 2]])


def test_quantise_rejects_a_non_positive_scale() -> None:
    for bad in (0.0, -0.01):
        with pytest.raises(AggregateError, match="scale must be positive"):
            quantise_field(np.zeros(2, dtype=np.float32), bad)


def test_check_quantisation_range_names_the_offending_field() -> None:
    fields = np.zeros((2, 2, 2), dtype=np.float32)
    fields[1] = 0.0
    fields[0, 0, 0] = 400.0  # 400 / 0.01 = 40000 > int16 max
    assert check_quantisation_range(fields, (0.01, 0.01)) == ("field[0]",)

    fields[0, 0, 0] = 0.0
    assert check_quantisation_range(fields, (0.01, 0.01)) == ()


def test_check_quantisation_range_ignores_all_nan_fields() -> None:
    fields = np.full((1, 2, 2), np.nan, dtype=np.float32)
    assert check_quantisation_range(fields, (0.01,)) == ()


def test_check_quantisation_range_requires_matching_counts() -> None:
    with pytest.raises(AggregateError, match="fields but"):
        check_quantisation_range(np.zeros((2, 2, 2), dtype=np.float32), (0.01,))


def test_encode_aggregate_refuses_to_clip() -> None:
    """A clipped field is silently wrong *and* compresses better, so it must be refused.

    The near-Gaussian encoding scales MEAN at 0.01, which only spans +-327.67. A variable
    whose values exceed that would quantise to a constant rather than fail, and the
    resulting "improved compression ratio" would look like a win.
    """
    spec = _spec(n_bins=4)
    fields = compute_aggregate(_members(seed=5), spec)
    fields = fields.copy()
    fields[0, 0, 0] = 400.0
    with pytest.raises(AggregateError, match="would clip"):
        encode_aggregate(fields, spec)


def test_encode_then_decode_aggregate_round_trips_all_fields() -> None:
    spec = _spec(n_bins=4)
    fields = compute_aggregate(_members(seed=6), spec)
    planes = encode_aggregate(fields, spec)
    assert len(planes) == spec.n_fields
    assert all(p.dtype == np.int16 for p in planes)

    back = decode_aggregate(planes, spec)
    assert np.allclose(back[0], fields[0], atol=MEAN_SCALE / 2)
    assert np.allclose(back[1], fields[1], atol=STD_SCALE / 2)
    for index in range(2, spec.n_fields):
        assert np.allclose(back[index], fields[index], atol=BIN_SCALE / 2)


def test_decode_aggregate_requires_the_declared_plane_count() -> None:
    spec = _spec(n_bins=4)
    planes = list(encode_aggregate(compute_aggregate(_members(seed=7), spec), spec))
    with pytest.raises(AggregateError, match="expected 6 planes"):
        decode_aggregate(planes[:-1], spec)


# ---------------------------------------------------------------------------
# Threshold reconstruction
# ---------------------------------------------------------------------------


def _fields_with_shapes() -> tuple[AggregateSpec, np.ndarray, np.ndarray]:
    spec = AggregateSpec(n_bins=DEFAULT_N_BINS)
    stack = _members(seed=8)
    fields = compute_aggregate(stack, spec)
    return spec, stack, fields


def test_exceedance_from_bins_answers_a_threshold_the_store_never_saw() -> None:
    """Reading P(x > t) off the stored shape is what the bins exist for."""
    spec, stack, fields = _fields_with_shapes()
    mean, std, bins = fields[0], fields[1], fields[2:]

    threshold = float(np.nanmean(mean))
    got = exceedance_from_bins(mean, std, bins, threshold, spec)
    truth = np.mean(stack > threshold, axis=0)

    # a threshold at the mean should sit near 0.5 and track the ensemble's own answer
    assert np.all(got >= 0.0) and np.all(got <= 1.0)
    assert abs(float(np.mean(got)) - float(np.mean(truth))) < 0.1


def test_exceedance_from_bins_is_monotone_in_the_threshold() -> None:
    """A higher threshold cannot carry a higher exceedance probability."""
    spec, _stack, fields = _fields_with_shapes()
    mean, std, bins = fields[0], fields[1], fields[2:]
    base = float(np.nanmean(mean))
    spread = float(np.nanmean(std))
    lows = exceedance_from_bins(mean, std, bins, base - spread, spec)
    highs = exceedance_from_bins(mean, std, bins, base + spread, spec)
    assert np.all(lows >= highs - 1e-6)


def test_exceedance_from_bins_saturates_outside_the_support() -> None:
    """A threshold below the support is certain; one above it is impossible."""
    spec, _stack, fields = _fields_with_shapes()
    mean, std, bins = fields[0], fields[1], fields[2:]
    below = exceedance_from_bins(mean, std, bins, float(np.nanmin(mean)) - 1e6, spec)
    above = exceedance_from_bins(mean, std, bins, float(np.nanmax(mean)) + 1e6, spec)
    assert np.allclose(below, 1.0)
    assert np.allclose(above, 0.0)


def test_exceedance_from_bins_rejects_a_wrong_bin_count() -> None:
    spec, _stack, fields = _fields_with_shapes()
    with pytest.raises(AggregateError, match="expected 32 bins"):
        exceedance_from_bins(fields[0], fields[1], fields[2:-1], 0.0, spec)


# ---------------------------------------------------------------------------
# Quantile-function encoding (the B/C classes)
# ---------------------------------------------------------------------------


def _quantile_spec(n_levels: int | None = None) -> AggregateSpec:
    if n_levels is None:
        return AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    return AggregateSpec(
        kind=KIND_QUANTILE_FUNCTION, levels=DEFAULT_QUANTILE_LEVELS[:n_levels]
    )


def _skewed_members(seed: int = 0, n: int = N_MEMBERS) -> np.ndarray:
    """Members drawn from a zero-inflated heavy-tailed field, i.e. a B/C-class variable."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 2.0, (LAT, LON)).astype(np.float32)
    return np.stack([base + rng.gamma(0.4, 2.0, (LAT, LON)) for _ in range(n)]).astype(
        np.float32
    )


def test_approved_quantile_level_count_is_nineteen() -> None:
    """19 body-dense levels is the measured optimum; a silent change moves every store.

    17 levels left a reconstruction error of 0.200 at the finest precipitation threshold,
    and 26 levels were *worse* than 19 because a 30-member sample quantile function cannot
    resolve a denser grid.
    """
    assert len(DEFAULT_QUANTILE_LEVELS) == 19
    spec = _quantile_spec()
    assert spec.n_fields == 19
    assert spec.field_scales == (QUANTILE_SCALE,) * 19


def test_quantile_spec_levels_are_strictly_increasing_within_the_open_unit_interval() -> None:
    levels = _quantile_spec().quantile_levels()
    assert all(0.0 < level < 1.0 for level in levels)
    assert list(levels) == sorted(levels)
    assert len(set(levels)) == len(levels)


def test_quantile_spec_field_names_carry_the_level() -> None:
    """A reader must be able to interpret a plane without re-deriving the spec."""
    names = _quantile_spec(n_levels=3).field_names
    assert names == ("Q001", "Q005", "Q020")


def test_spec_rejects_mismatched_parameters_between_kinds() -> None:
    """A bin parameter on a quantile spec means the class was mis-specified."""
    with pytest.raises(AggregateError, match="unknown aggregate kind"):
        AggregateSpec(kind="not_a_kind")
    with pytest.raises(AggregateError, match="levels apply to the quantile_function kind only"):
        AggregateSpec(kind=KIND_MEAN_STD_BINS, levels=(0.5,))
    with pytest.raises(AggregateError, match="requires at least one level"):
        AggregateSpec(kind=KIND_QUANTILE_FUNCTION, levels=())
    with pytest.raises(AggregateError, match="strictly between 0 and 1"):
        AggregateSpec(kind=KIND_QUANTILE_FUNCTION, levels=(0.0, 0.5))
    with pytest.raises(AggregateError, match="strictly increasing"):
        AggregateSpec(kind=KIND_QUANTILE_FUNCTION, levels=(0.5, 0.5))
    with pytest.raises(AggregateError, match="has no bin edges"):
        _quantile_spec().bin_edges()
    with pytest.raises(AggregateError, match="has no quantile levels"):
        _spec().quantile_levels()


def test_quantile_encoding_matches_the_reference_bit_for_bit() -> None:
    """Pin the sample-quantile convention: one sort, interpolate at ``p * (n - 1)``.

    A reader that interpolates between two stored levels with a different convention from
    the writer's sampling would disagree at the scale of a single member step.
    """
    stack = _skewed_members(seed=1)
    spec = _quantile_spec()

    ordered = np.sort(np.where(np.isnan(stack), np.inf, stack), axis=0)
    last = N_MEMBERS - 1
    expected = np.stack(
        [
            (
                ordered[int(np.floor(level * last))]
                * (1 - (level * last - int(np.floor(level * last))))
                + ordered[min(int(np.floor(level * last)) + 1, last)]
                * (level * last - int(np.floor(level * last)))
            ).astype(np.float32)
            for level in DEFAULT_QUANTILE_LEVELS
        ]
    )
    assert np.array_equal(compute_aggregate(stack, spec), expected)


def test_quantile_planes_are_monotone_per_cell() -> None:
    """An inverse CDF cannot decrease as the level rises."""
    fields = compute_aggregate(_skewed_members(seed=2), _quantile_spec())
    assert (np.diff(fields, axis=0) >= -1e-6).all()


def test_quantile_encoding_handles_a_constant_cell() -> None:
    """Every member identical collapses the interpolation span; it must not divide by zero."""
    stack = np.full((N_MEMBERS, LAT, LON), 3.5, dtype=np.float32)
    fields = compute_aggregate(stack, _quantile_spec())
    assert np.isfinite(fields).all()
    assert np.allclose(fields, 3.5)


def test_quantile_encoding_propagates_nan_for_an_incomplete_cell() -> None:
    """A partially observed cell has no quantile function, so every level must be NaN.

    Without the explicit guard the sorted axis would carry ``inf`` for the missing members
    and the upper levels would come back finite -- a plausible-looking but fabricated tail.
    """
    stack = _skewed_members(seed=3)
    stack[0, 0, 0] = np.nan
    fields = compute_aggregate(stack, _quantile_spec())
    assert np.all(np.isnan(fields[:, 0, 0]))
    assert np.isfinite(fields[:, 1, 1]).all()


def test_quantile_encoding_round_trips_through_quantisation() -> None:
    spec = _quantile_spec()
    fields = compute_aggregate(_skewed_members(seed=4), spec)
    planes = encode_aggregate(fields, spec)
    assert len(planes) == spec.n_fields
    back = decode_aggregate(planes, spec)
    assert np.allclose(back, fields, atol=QUANTILE_SCALE / 2)


def test_quantile_encoding_refuses_a_clipping_scale() -> None:
    """A wind gust in km/h approaches the +-327.67 the scale covers, and clipping is silent.

    The clipped field would quantise to fewer distinct values and therefore compress
    *better*, so a range overflow must be a hard failure rather than a warning.
    """
    spec = _quantile_spec()
    fields = compute_aggregate(_skewed_members(seed=5), spec).copy()
    fields[0, 0, 0] = 400.0
    with pytest.raises(AggregateError, match="would clip"):
        encode_aggregate(fields, spec)


def test_exceedance_from_quantiles_tracks_the_ensemble() -> None:
    """The quantile encoding must answer a threshold the store was never told about."""
    stack = _skewed_members(seed=6)
    spec = _quantile_spec()
    values = compute_aggregate(stack, spec)

    for percentile in (50.0, 90.0, 99.0):
        threshold = float(np.percentile(stack, percentile))
        estimated = exceedance_from_quantiles(values, spec.quantile_levels(), threshold)
        truth = np.mean(stack > threshold, axis=0)
        assert abs(float(np.mean(estimated)) - float(np.mean(truth))) < 0.05


def test_exceedance_from_quantiles_is_monotone_in_the_threshold() -> None:
    spec = _quantile_spec()
    values = compute_aggregate(_skewed_members(seed=7), spec)
    levels = spec.quantile_levels()
    low = exceedance_from_quantiles(values, levels, 0.5)
    mid = exceedance_from_quantiles(values, levels, 2.0)
    high = exceedance_from_quantiles(values, levels, 8.0)
    assert (low >= mid - 1e-6).all()
    assert (mid >= high - 1e-6).all()


def test_exceedance_from_quantiles_saturates_at_the_stored_extremes() -> None:
    """The encoding cannot express probabilities beyond its outermost levels.

    Below the lowest level the answer is ``1 - min(level)`` and above the highest it is
    ``1 - max(level)`` -- not 0 and 1. That is a property of storing a finite level set, and
    it is harmless here because the extremes (0.001/0.999) are already far finer than a
    30-member ensemble can resolve (1/30), which is the real floor.
    """
    spec = _quantile_spec()
    values = compute_aggregate(_skewed_members(seed=8), spec)
    levels = spec.quantile_levels()

    below = exceedance_from_quantiles(values, levels, -1e6)
    above = exceedance_from_quantiles(values, levels, 1e6)
    assert np.allclose(below, 1.0 - levels[0])
    assert np.allclose(above, 1.0 - levels[-1])
    # the encoder cannot express anything rarer than a 30-member sample can resolve
    assert levels[0] < 1.0 / N_MEMBERS
    assert (1.0 - levels[-1]) < 1.0 / N_MEMBERS


def test_exceedance_from_quantiles_is_exact_at_a_stored_level() -> None:
    """At a stored level the CDF is known exactly, so the answer is ``1 - level``."""
    spec = _quantile_spec()
    values = compute_aggregate(_skewed_members(seed=9), spec)
    levels = spec.quantile_levels()

    for index in (0, len(levels) // 2, len(levels) - 1):
        threshold = float(values[index][0, 0])
        estimated = exceedance_from_quantiles(values, levels, threshold)
        assert abs(float(estimated[0, 0]) - (1.0 - levels[index])) < 1e-5


def test_exceedance_from_quantiles_handles_a_degenerate_segment() -> None:
    """Two levels with the same value carry no position information; must not divide by zero."""
    spec = _quantile_spec(n_levels=3)
    values = np.zeros((3, LAT, LON), dtype=np.float32)
    result = exceedance_from_quantiles(values, spec.quantile_levels(), 0.0)
    assert np.isfinite(result).all()


def test_exceedance_from_quantiles_rejects_bad_input() -> None:
    spec = _quantile_spec()
    values = compute_aggregate(_skewed_members(seed=10), spec)
    with pytest.raises(AggregateError, match="at least two planes"):
        exceedance_from_quantiles(values[0], (0.5,), 1.0)
    with pytest.raises(AggregateError, match="planes but"):
        exceedance_from_quantiles(values[:3], spec.quantile_levels(), 1.0)
    with pytest.raises(AggregateError, match="strictly increasing"):
        exceedance_from_quantiles(values, (0.5, 0.5, *spec.quantile_levels()[2:]), 1.0)


def test_quantile_encoding_handles_a_single_member() -> None:
    """With one member every level is that member's value.

    The interpolation span collapses (there is no upper neighbour), which is a real case:
    an aggregate may be computed from a degenerate set when a wave is repaired.
    """
    stack = np.full((1, LAT, LON), 7.25, dtype=np.float32)
    fields = compute_aggregate(stack, _quantile_spec())
    assert np.allclose(fields, 7.25)


# ---------------------------------------------------------------------------
# Serving the API's statistics from a stored quantile function
# ---------------------------------------------------------------------------


def test_statistic_level_reports_which_contract_percentiles_are_stored() -> None:
    """P10/P50/P90 are stored levels; P25/P75 are not. A reader needs to know which."""
    spec = _quantile_spec()
    assert spec.statistic_level("p10") == 0.10
    assert spec.statistic_level("P50") == 0.50
    assert spec.statistic_level("median") == 0.50
    assert spec.statistic_level("p90") == 0.90
    assert spec.statistic_level("p25") is None
    assert spec.statistic_level("p75") is None
    # a bin spec has no levels at all
    assert _spec().statistic_level("p50") is None
    # unrecognised names are not guessed at
    assert spec.statistic_level("mean") is None
    assert spec.statistic_level("pXX") is None


def test_quantile_function_moments_recover_mean_and_spread_within_sampling_noise() -> None:
    """The API serves mean and spread for every variable, but the encoding stores neither.

    Measured on the classes this encoding serves, both come back well inside the ensemble's
    own sampling noise, so storing two extra planes would buy precision the display cannot
    show. The assertion is against that noise floor rather than an absolute tolerance,
    because absolute tolerances differ by orders of magnitude between variables.
    """
    rng = np.random.default_rng(23)
    for label, members in (
        (
            "zero-inflated precipitation",
            np.where(
                rng.random((N_MEMBERS, 4, 64)) < 0.5,
                0.0,
                rng.gamma(0.35, 1.5, (N_MEMBERS, 4, 64)),
            ),
        ),
        ("heavy-tailed gust", np.abs(rng.standard_t(2.5, (N_MEMBERS, 4, 64))) * 20.0),
        ("bounded humidity", np.clip(rng.normal(70, 20, (N_MEMBERS, 4, 64)), 0, 100)),
    ):
        stack = np.asarray(members, dtype=np.float32)
        spec = _quantile_spec()
        values = compute_aggregate(stack, spec)
        mean, spread = quantile_function_moments(values, spec.quantile_levels())

        half = N_MEMBERS // 2
        mean_noise = float(
            np.max(np.abs(stack[:half].mean(axis=0) - stack[half:].mean(axis=0)))
        )
        spread_noise = float(
            np.max(np.abs(stack[:half].std(axis=0) - stack[half:].std(axis=0)))
        )
        mean_err = float(np.max(np.abs(mean - stack.mean(axis=0))))
        spread_err = float(np.max(np.abs(spread - stack.std(axis=0))))
        assert mean_err / mean_noise < 1.0, (label, mean_err, mean_noise)
        assert spread_err / spread_noise < 1.0, (label, spread_err, spread_noise)


def test_quantile_at_is_exact_at_a_stored_level() -> None:
    """A contract percentile that coincides with a stored level must be reproduced exactly."""
    stack = _skewed_members(seed=50)
    spec = _quantile_spec()
    values = compute_aggregate(stack, spec)
    levels = spec.quantile_levels()

    for probability in (0.10, 0.50, 0.90):
        index = list(levels).index(probability)
        read = quantile_at(values, levels, probability)
        assert np.array_equal(read, values[index].astype(np.float32))


def test_quantile_at_interpolates_between_stored_levels() -> None:
    """P25 and P75 are not stored levels, so they are read by interpolation."""
    stack = _skewed_members(seed=51)
    spec = _quantile_spec()
    values = compute_aggregate(stack, spec)
    levels = spec.quantile_levels()

    read = quantile_at(values, levels, 0.25)
    lower = values[list(levels).index(0.20)]
    upper = values[list(levels).index(0.30)]
    assert np.all(read >= np.minimum(lower, upper) - 1e-5)
    assert np.all(read <= np.maximum(lower, upper) + 1e-5)
    # and within the ensemble's own noise of the true sample percentile
    truth = np.percentile(stack, 25, axis=0)
    noise = float(
        np.max(
            np.abs(
                np.percentile(stack[:15], 25, axis=0)
                - np.percentile(stack[15:], 25, axis=0)
            )
        )
    )
    assert float(np.max(np.abs(read - truth))) / noise < 1.0


def test_quantile_at_clamps_outside_the_stored_range() -> None:
    stack = _skewed_members(seed=52)
    spec = _quantile_spec()
    values = compute_aggregate(stack, spec)
    levels = spec.quantile_levels()
    assert np.array_equal(quantile_at(values, levels, 0.0), values[0])
    assert np.array_equal(quantile_at(values, levels, 1.0), values[-1])


def test_quantile_at_rejects_bad_input() -> None:
    spec = _quantile_spec()
    values = compute_aggregate(_skewed_members(seed=53), spec)
    with pytest.raises(AggregateError, match="planes but"):
        quantile_at(values[:3], spec.quantile_levels(), 0.5)
    with pytest.raises(AggregateError, match="strictly increasing"):
        quantile_at(values, (0.5, 0.5, *spec.quantile_levels()[2:]), 0.5)
    with pytest.raises(AggregateError, match=r"probability must be in \[0, 1\]"):
        quantile_at(values, spec.quantile_levels(), 1.5)


def test_quantile_function_moments_reject_bad_input() -> None:
    spec = _quantile_spec()
    values = compute_aggregate(_skewed_members(seed=54), spec)
    with pytest.raises(AggregateError, match="planes but"):
        quantile_function_moments(values[:3], spec.quantile_levels())
    with pytest.raises(AggregateError, match="positive probability range"):
        quantile_function_moments(values, (0.5,) * len(spec.levels))
