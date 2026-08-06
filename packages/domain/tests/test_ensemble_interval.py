"""Unit tests for domain.ensemble confidence intervals."""

import math

import numpy as np
import numpy.typing as npt
import pytest
from domain.ensemble import probability_confidence_interval
from domain.exceptions import InvalidEnsembleError, InvalidThresholdError


class TestProbabilityConfidenceInterval:
    def test_wilson_known_value(self) -> None:
        # Hand-computed Wilson score interval for (p=0.6, n=5):
        #   z=1.959963984540054, denom = 1 + z^2/5
        #   lower ≈ 0.2307, upper ≈ 0.8824
        lower, upper = probability_confidence_interval(0.6, 5)
        assert lower == pytest.approx(0.2307, abs=1e-4)
        assert upper == pytest.approx(0.8824, abs=1e-4)

    def test_interval_contains_proportion(self) -> None:
        lower, upper = probability_confidence_interval(0.6, 5)
        assert 0.0 <= lower <= 0.6 <= upper <= 1.0

    def test_p_zero_lower_bound_is_zero(self) -> None:
        # Wilson does not collapse to a zero-width interval at p=0, but the
        # lower bound is clamped to 0.
        lower, upper = probability_confidence_interval(0.0, 5)
        assert lower == 0.0
        assert upper > 0.0

    def test_p_one_upper_bound_is_one(self) -> None:
        lower, upper = probability_confidence_interval(1.0, 5)
        assert upper == 1.0
        assert lower < 1.0

    def test_larger_sample_narrows_interval(self) -> None:
        # A larger sample at the same proportion must produce a narrower
        # interval.
        wide_lower, wide_upper = probability_confidence_interval(0.6, 5)
        narrow_lower, narrow_upper = probability_confidence_interval(0.6, 100)
        assert (narrow_upper - narrow_lower) < (wide_upper - wide_lower)
        assert narrow_lower > wide_lower
        assert narrow_upper < wide_upper

    def test_single_element_sample(self) -> None:
        lower, upper = probability_confidence_interval(1.0, 1)
        assert 0.0 <= lower <= upper <= 1.0

    def test_deterministic(self) -> None:
        assert probability_confidence_interval(0.6, 5) == probability_confidence_interval(
            0.6, 5
        )

    def test_returns_floats(self) -> None:
        lower, upper = probability_confidence_interval(0.6, 5)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_accepts_numpy_scalars(self) -> None:
        # Consistent with the sibling domain helpers, numpy scalar numerics are
        # accepted for every numeric parameter.
        probability: npt.NDArray[np.float64] = np.float64(0.6)
        size: npt.NDArray[np.int64] = np.int64(5)
        confidence: npt.NDArray[np.float64] = np.float64(0.95)
        lower, upper = probability_confidence_interval(
            probability, size, confidence=confidence
        )
        expected_lower, expected_upper = probability_confidence_interval(0.6, 5)
        assert lower == pytest.approx(expected_lower)
        assert upper == pytest.approx(expected_upper)

    @pytest.mark.parametrize(
        "bad_probability",
        [
            -0.1,
            1.1,
            math.nan,
            math.inf,
            -math.inf,
            "0.6",
            None,
            True,
        ],
    )
    def test_invalid_probability_rejected(self, bad_probability: object) -> None:
        with pytest.raises(InvalidThresholdError):
            probability_confidence_interval(bad_probability, 5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_size", [0, -1, 1.5, "5", None, True])
    def test_invalid_sample_size_rejected(self, bad_size: object) -> None:
        with pytest.raises(InvalidEnsembleError):
            probability_confidence_interval(0.6, bad_size)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_confidence", [0.90, 0.99, 1.0, "0.95", None, True])
    def test_unsupported_confidence_rejected(self, bad_confidence: object) -> None:
        with pytest.raises(InvalidThresholdError):
            probability_confidence_interval(
                0.6, 5, confidence=bad_confidence  # type: ignore[arg-type]
            )
