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
    MEAN_SCALE,
    NAN_SENTINEL,
    STD_SCALE,
    AggregateError,
    AggregateSpec,
    check_quantisation_range,
    compute_aggregate,
    decode_aggregate,
    dequantise_field,
    encode_aggregate,
    exceedance_from_bins,
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
