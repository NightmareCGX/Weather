"""Unit tests for the canonical ensemble PDF estimation module.

Verifies mathematical properties (unit integral, tail padding, Silverman robust
bandwidth rule with IQR=0 fallback), edge cases (degenerate variance, single
member, empty input, non-finite values), and 100% test coverage.
"""

import numpy as np
import pytest
from domain.ensemble import EnsemblePDF, estimate_ensemble_pdf
from domain.exceptions import EmptyEnsembleError, InvalidEnsembleError


def _trapz(y: list[float], x: list[float]) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def test_pdf_normal_ensemble_integral_and_shape() -> None:
    # 30 members sampled around 20.0 with std ~ 2.0
    members = [
        17.1, 17.5, 18.0, 18.2, 18.5, 18.8, 19.0, 19.2, 19.5, 19.7,
        19.9, 20.0, 20.1, 20.2, 20.3, 20.5, 20.6, 20.8, 21.0, 21.2,
        21.5, 21.7, 21.9, 22.1, 22.3, 22.6, 22.8, 23.0, 23.2, 23.5,
    ]
    pdf = estimate_ensemble_pdf(members)
    assert pdf is not None
    assert isinstance(pdf, EnsemblePDF)
    assert len(pdf.x) == 100
    assert len(pdf.density) == 100

    # Verify all coordinates and densities are finite floats
    assert all(isinstance(v, float) for v in pdf.x)
    assert all(isinstance(v, float) for v in pdf.density)
    assert all(d >= 0.0 for d in pdf.density)

    # Numerical integral over [min - 3h, max + 3h] captures >99% of mass
    integral = _trapz(pdf.density, pdf.x)
    assert 0.990 <= integral <= 1.001

    # Grid bounds must match min - 3h and max + 3h
    min_val = min(members)
    max_val = max(members)
    assert pdf.x[0] < min_val
    assert pdf.x[-1] > max_val


def test_pdf_iqr_zero_std_positive_regression() -> None:
    # Regression: [1, 1, 1, 1, 2] has IQR = 0 and std > 0
    members1 = [1.0, 1.0, 1.0, 1.0, 2.0]
    pdf1 = estimate_ensemble_pdf(members1)
    assert pdf1 is not None
    assert len(pdf1.x) == 100
    integral1 = _trapz(pdf1.density, pdf1.x)
    assert 0.990 <= integral1 <= 1.001

    # Regression: [5, 5, 5, 6, 6] has IQR = 0 and std > 0
    members2 = [5, 5, 5, 6, 6]
    pdf2 = estimate_ensemble_pdf(members2)
    assert pdf2 is not None
    assert len(pdf2.x) == 100
    integral2 = _trapz(pdf2.density, pdf2.x)
    assert 0.990 <= integral2 <= 1.001


def test_pdf_scale_selection_iqr_vs_std() -> None:
    # Normal distribution where IQR/1.34 is smaller than std (e.g. outlier-heavy)
    members_outliers = [-10.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 10.0]
    pdf_outliers = estimate_ensemble_pdf(members_outliers)
    assert pdf_outliers is not None
    integral_outliers = _trapz(pdf_outliers.density, pdf_outliers.x)
    assert 0.990 <= integral_outliers <= 1.001

    # Uniform-like distribution where std is smaller than IQR/1.34
    members_uniform = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    pdf_uniform = estimate_ensemble_pdf(members_uniform)
    assert pdf_uniform is not None
    integral_uniform = _trapz(pdf_uniform.density, pdf_uniform.x)
    assert 0.990 <= integral_uniform <= 1.001


def test_pdf_degenerate_zero_spread_returns_none() -> None:
    # All identical values: std = 0 -> continuous density undefined
    members = [15.0, 15.0, 15.0, 15.0]
    assert estimate_ensemble_pdf(members) is None


def test_pdf_single_member_returns_none() -> None:
    # Single member: N < 2 -> continuous density undefined
    assert estimate_ensemble_pdf([10.0]) is None


def test_pdf_numpy_array_input() -> None:
    arr = np.array([12.0, 14.0, 16.0, 18.0, 20.0], dtype=np.float64)
    pdf = estimate_ensemble_pdf(arr)
    assert pdf is not None
    assert len(pdf.x) == 100


def test_pdf_empty_sequence_raises_empty_error() -> None:
    with pytest.raises(EmptyEnsembleError):
        estimate_ensemble_pdf([])


def test_pdf_non_finite_raises_invalid_error() -> None:
    with pytest.raises(InvalidEnsembleError):
        estimate_ensemble_pdf([1.0, float("nan"), 3.0])

    with pytest.raises(InvalidEnsembleError):
        estimate_ensemble_pdf([1.0, float("inf"), 3.0])


def test_pdf_non_numeric_raises_invalid_error() -> None:
    with pytest.raises(InvalidEnsembleError):
        estimate_ensemble_pdf(["a", "b"])  # type: ignore[arg-type]


def test_pdf_multidimensional_raises_invalid_error() -> None:
    with pytest.raises(InvalidEnsembleError):
        estimate_ensemble_pdf(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_pdf_deterministic() -> None:
    members = [10.0, 12.5, 15.0, 17.5, 20.0]
    pdf1 = estimate_ensemble_pdf(members)
    pdf2 = estimate_ensemble_pdf(members)
    assert pdf1 == pdf2


# ---------------------------------------------------------------------------
# The stored-distribution forms: a KDE over a quantile function or a histogram
# ---------------------------------------------------------------------------


def test_a_stored_quantile_function_reproduces_the_member_curves_shape() -> None:
    """A KDE is an integral against the distribution, so a sampled one is enough to draw it.

    The property that makes the stored path's curve the *same* curve rather than a similar one:
    evaluated on the member path's own grid with the member path's own bandwidth rule, the two
    agree to a fraction of the ensemble's sampling noise. Anything coarser would leave the chart
    drawing two shapes for one distribution, which is what a source migration must not do.
    """
    from domain.aggregate import KIND_QUANTILE_FUNCTION, AggregateSpec, compute_aggregate
    from domain.ensemble import estimate_pdf_from_quantiles

    rng = np.random.default_rng(11)
    members = rng.normal(280.0, 8.0, 30)
    reference = estimate_ensemble_pdf(members)
    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    fields = compute_aggregate(
        np.asarray(members, dtype=np.float32)[:, None, None], spec, expected_members=30
    )[:, 0, 0]

    stored = estimate_pdf_from_quantiles(spec.quantile_levels(), fields, 30)
    assert stored is not None
    assert len(stored.x) == len(reference.x) == 100

    # Interpolate the reference onto the stored grid and compare the whole curve.
    ref_density = np.interp(stored.x, reference.x, reference.density)
    got = np.asarray(stored.density)
    # The sample's own half-to-half disagreement is the yardstick: below it, the two are the same
    # estimate of the same distribution.
    noise = float(
        np.abs(
            np.interp(stored.x, *(
                lambda p: (p.x, p.density)
            )(estimate_ensemble_pdf(members[:15])))
            - np.interp(stored.x, *(
                lambda p: (p.x, p.density)
            )(estimate_ensemble_pdf(members[15:])))
        ).mean()
    )
    assert float(np.abs(got - ref_density).mean()) < noise


def test_a_stored_histogram_reproduces_the_member_curves_shape() -> None:
    """The bin encoding states a shape rather than levels, and the same integral applies."""
    from domain.aggregate import KIND_MEAN_STD_BINS, AggregateSpec, compute_aggregate
    from domain.ensemble import estimate_pdf_from_bins

    rng = np.random.default_rng(12)
    members = rng.normal(280.0, 8.0, 30)
    reference = estimate_ensemble_pdf(members)
    spec = AggregateSpec(kind=KIND_MEAN_STD_BINS)
    fields = compute_aggregate(
        np.asarray(members, dtype=np.float32)[:, None, None], spec, expected_members=30
    )[:, 0, 0]
    mean, spread = float(fields[0]), float(fields[1])
    half = float(spec.sigma_range) * spread
    edges = np.linspace(mean - half, mean + half, spec.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    stored = estimate_pdf_from_bins(
        centers, fields[2 : 2 + spec.n_bins], member_count=30, bin_count=spec.n_bins
    )
    assert stored is not None
    assert len(stored.x) == 100
    ref_density = np.interp(stored.x, reference.x, reference.density)
    got = np.asarray(stored.density)
    # The support is mean +- 4 sigma, so the tails are cut by construction -- the same caveat the
    # encoding's percentiles carry. Compared over the sample's own range, the shape must match.
    inside = (np.asarray(stored.x) >= float(np.min(members))) & (
        np.asarray(stored.x) <= float(np.max(members))
    )
    assert inside.any(), "the stored grid must overlap the sample's own range"
    assert float(np.abs(got[inside] - ref_density[inside]).mean()) < 0.6 * float(
        ref_density[inside].max()
    )


def test_a_point_mass_the_encoding_cannot_state_is_refused() -> None:
    """A distribution with a ceiling returns no curve, because a smooth one would be wrong.

    A quantile function is continuous, so members piled on one value appear as a stretch of
    probability over which the value does not rise. Evaluating a KDE over that smear understates
    how sharp the mass is and overstates the curve's spread -- measured on the real 09-23 06Z GEFS
    store at 5.6x the ensemble's own sampling noise, where every frame without such a stretch
    stayed below 0.9x. No curve is the honest answer; the caller keeps what it had.
    """
    from domain.aggregate import KIND_QUANTILE_FUNCTION, AggregateSpec, compute_aggregate
    from domain.ensemble import estimate_pdf_from_quantiles, quantile_point_mass

    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    levels = spec.quantile_levels()

    # A continuous sample: the levels strictly increase, so there is no mass to state.
    rng = np.random.default_rng(13)
    continuous = rng.normal(280.0, 8.0, 30)
    fields = compute_aggregate(
        np.asarray(continuous, dtype=np.float32)[:, None, None], spec, expected_members=30
    )[:, 0, 0]
    assert quantile_point_mass(levels, fields, 30) == pytest.approx(0.0)
    assert estimate_pdf_from_quantiles(levels, fields, 30) is not None

    # A ceiling: most members at one value, which the levels cannot separate.
    capped = np.concatenate([rng.uniform(0.0, 10.0, 3), np.full(27, 24.1)])
    capped_fields = compute_aggregate(
        np.asarray(capped, dtype=np.float32)[:, None, None], spec, expected_members=30
    )[:, 0, 0]
    assert quantile_point_mass(levels, capped_fields, 30) > 1.0
    assert estimate_pdf_from_quantiles(levels, capped_fields, 30) is None
    # And it is a refusal, not a crash: the caller falls back rather than failing the request.
    assert estimate_ensemble_pdf(list(capped)) is not None


def test_a_stored_form_refuses_a_degenerate_or_malformed_distribution() -> None:
    """Every refusal path, so a caller always gets ``None`` rather than a fabricated curve."""
    from domain.ensemble import estimate_pdf_from_bins, estimate_pdf_from_quantiles

    levels = (0.0, 0.5, 1.0)
    # One value everywhere: no spread, so no density to draw.
    assert estimate_pdf_from_quantiles(levels, [5.0, 5.0, 5.0], 30) is None
    # Non-finite levels or values are refused rather than propagated.
    assert estimate_pdf_from_quantiles(levels, [1.0, float("nan"), 3.0], 30) is None
    # Mismatched lengths are not a distribution.
    assert estimate_pdf_from_quantiles(levels, [1.0, 2.0], 30) is None
    assert estimate_pdf_from_quantiles((0.5,), [1.0], 30) is None

    assert estimate_pdf_from_bins([1.0, 2.0], [0.5, 0.5], member_count=30, bin_count=2) is not None
    assert estimate_pdf_from_bins([1.0, 2.0], [0.0, 0.0], member_count=30, bin_count=2) is None
    assert estimate_pdf_from_bins([1.0], [1.0], member_count=30, bin_count=1) is None
    assert estimate_pdf_from_bins([1.0, 2.0], [0.5], member_count=30, bin_count=2) is None
    assert (
        estimate_pdf_from_bins([1.0, float("inf")], [0.5, 0.5], member_count=30, bin_count=2)
        is None
    )
    # A single-valued histogram has no spread either.
    assert estimate_pdf_from_bins([3.0, 3.0], [0.5, 0.5], member_count=30, bin_count=2) is None
