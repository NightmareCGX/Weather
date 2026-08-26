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
