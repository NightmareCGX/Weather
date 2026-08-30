"""Unit tests for domain.ensemble probability functions."""

import math

import numpy as np
import numpy.typing as npt
import pytest
from domain.ensemble import (
    probability_above_threshold,
    probability_at_or_above_threshold,
    probability_at_or_below_threshold,
    probability_below_threshold,
    probability_between_thresholds,
)
from domain.exceptions import (
    EmptyEnsembleError,
    InvalidEnsembleError,
    InvalidThresholdError,
)

#: Hand-computed ensemble member values used across the probability tests.
MEMBERS: list[float] = [0.0, 10.0, 20.0, 30.0, 40.0]


class TestProbabilityAboveThreshold:
    def test_partial_above(self) -> None:
        # Members above 25: {30, 40}.
        assert probability_above_threshold(MEMBERS, 25.0) == pytest.approx(0.4)

    def test_strict_gt_excludes_equal_member(self) -> None:
        # Member exactly 10 must not count.
        assert probability_above_threshold(MEMBERS, 10.0) == pytest.approx(0.6)

    def test_none_above(self) -> None:
        assert probability_above_threshold(MEMBERS, 40.0) == pytest.approx(0.0)

    def test_all_above(self) -> None:
        assert probability_above_threshold(MEMBERS, -1.0) == pytest.approx(1.0)

    def test_single_element(self) -> None:
        assert probability_above_threshold([5.0], 3.0) == pytest.approx(1.0)
        assert probability_above_threshold([5.0], 7.0) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        assert isinstance(probability_above_threshold(MEMBERS, 25.0), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert probability_above_threshold(array, 25.0) == pytest.approx(0.4)


class TestProbabilityAtOrAboveThreshold:
    def test_partial_above(self) -> None:
        # Members at or above 25: {30, 40}.
        assert probability_at_or_above_threshold(MEMBERS, 25.0) == pytest.approx(0.4)

    def test_inclusive_gte_includes_equal_member(self) -> None:
        # Member exactly 10 MUST count: {10, 20, 30, 40} -> 4/5 = 0.8
        assert probability_at_or_above_threshold(MEMBERS, 10.0) == pytest.approx(0.8)

    def test_none_above(self) -> None:
        assert probability_at_or_above_threshold(MEMBERS, 45.0) == pytest.approx(0.0)

    def test_all_above(self) -> None:
        assert probability_at_or_above_threshold(MEMBERS, 0.0) == pytest.approx(1.0)

    def test_single_element(self) -> None:
        assert probability_at_or_above_threshold([5.0], 5.0) == pytest.approx(1.0)
        assert probability_at_or_above_threshold([5.0], 7.0) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        assert isinstance(probability_at_or_above_threshold(MEMBERS, 25.0), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert probability_at_or_above_threshold(array, 20.0) == pytest.approx(0.6)


class TestProbabilityBelowThreshold:
    def test_partial_below(self) -> None:
        # Members below 25: {0, 10, 20}.
        assert probability_below_threshold(MEMBERS, 25.0) == pytest.approx(0.6)

    def test_strict_lt_excludes_equal_member(self) -> None:
        # Member exactly 10 must not count.
        assert probability_below_threshold(MEMBERS, 10.0) == pytest.approx(0.2)

    def test_none_below(self) -> None:
        assert probability_below_threshold(MEMBERS, 0.0) == pytest.approx(0.0)

    def test_all_below(self) -> None:
        assert probability_below_threshold(MEMBERS, 100.0) == pytest.approx(1.0)

    def test_single_element(self) -> None:
        assert probability_below_threshold([5.0], 7.0) == pytest.approx(1.0)
        assert probability_below_threshold([5.0], 3.0) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        assert isinstance(probability_below_threshold(MEMBERS, 25.0), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert probability_below_threshold(array, 25.0) == pytest.approx(0.6)


class TestProbabilityAtOrBelowThreshold:
    def test_partial_below(self) -> None:
        # Members at or below 25: {0, 10, 20}.
        assert probability_at_or_below_threshold(MEMBERS, 25.0) == pytest.approx(0.6)

    def test_inclusive_lte_includes_equal_member(self) -> None:
        # Member exactly 10 MUST count: {0, 10} -> 2/5 = 0.4
        assert probability_at_or_below_threshold(MEMBERS, 10.0) == pytest.approx(0.4)

    def test_none_below(self) -> None:
        assert probability_at_or_below_threshold(MEMBERS, -5.0) == pytest.approx(0.0)

    def test_all_below(self) -> None:
        assert probability_at_or_below_threshold(MEMBERS, 40.0) == pytest.approx(1.0)

    def test_single_element(self) -> None:
        assert probability_at_or_below_threshold([5.0], 5.0) == pytest.approx(1.0)
        assert probability_at_or_below_threshold([5.0], 3.0) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        assert isinstance(probability_at_or_below_threshold(MEMBERS, 25.0), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert probability_at_or_below_threshold(array, 20.0) == pytest.approx(0.6)


class TestProbabilityBetweenThresholds:
    def test_inclusive_edges(self) -> None:
        # Members in [10, 30]: {10, 20, 30}.
        assert probability_between_thresholds(MEMBERS, 10.0, 30.0) == pytest.approx(0.6)

    def test_narrow_range(self) -> None:
        assert probability_between_thresholds(MEMBERS, 5.0, 15.0) == pytest.approx(0.2)

    def test_equal_bounds(self) -> None:
        # Single-value inclusive range catches the matching member.
        assert probability_between_thresholds(MEMBERS, 10.0, 10.0) == pytest.approx(0.2)

    def test_covers_all_members(self) -> None:
        assert probability_between_thresholds(MEMBERS, 0.0, 40.0) == pytest.approx(1.0)

    def test_lower_boundary_is_inclusive(self) -> None:
        assert probability_between_thresholds(MEMBERS, 0.0, 10.0) == pytest.approx(0.4)

    def test_upper_boundary_is_inclusive(self) -> None:
        assert probability_between_thresholds(MEMBERS, 30.0, 40.0) == pytest.approx(0.4)

    def test_no_members_between(self) -> None:
        assert probability_between_thresholds(MEMBERS, 41.0, 50.0) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        assert isinstance(probability_between_thresholds(MEMBERS, 10.0, 30.0), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert probability_between_thresholds(array, 10.0, 30.0) == pytest.approx(0.6)

    def test_upper_below_lower_rejected(self) -> None:
        with pytest.raises(InvalidThresholdError):
            probability_between_thresholds(MEMBERS, 30.0, 10.0)


class TestSharedValidation:
    @pytest.mark.parametrize("func", [probability_above_threshold, probability_below_threshold])
    def test_empty_members_rejected(self, func: object) -> None:
        with pytest.raises(EmptyEnsembleError):
            func([], 1.0)  # type: ignore[operator]

    def test_empty_members_between_rejected(self) -> None:
        with pytest.raises(EmptyEnsembleError):
            probability_between_thresholds([], 0.0, 1.0)

    @pytest.mark.parametrize(
        "bad_members",
        [
            [1, "two", 3],
            [1.0, math.nan, 3.0],
            [1.0, math.inf, 3.0],
        ],
    )
    def test_invalid_members_rejected(self, bad_members: object) -> None:
        with pytest.raises(InvalidEnsembleError):
            probability_above_threshold(bad_members, 1.0)  # type: ignore[arg-type]

    def test_two_dimensional_array_rejected(self) -> None:
        matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with pytest.raises(InvalidEnsembleError):
            probability_above_threshold(matrix, 1.0)

    def test_nested_list_members_rejected(self) -> None:
        with pytest.raises(InvalidEnsembleError):
            probability_above_threshold([[1.0, 2.0], [3.0, 4.0]], 1.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_threshold",
        [
            math.nan,
            math.inf,
            -math.inf,
            "1.0",
            None,
            True,
        ],
    )
    def test_non_finite_threshold_rejected(self, bad_threshold: object) -> None:
        with pytest.raises(InvalidThresholdError):
            probability_above_threshold(MEMBERS, bad_threshold)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_threshold",
        [
            math.nan,
            math.inf,
            -math.inf,
            "1.0",
            None,
            True,
        ],
    )
    def test_between_non_finite_threshold_rejected(self, bad_threshold: object) -> None:
        with pytest.raises(InvalidThresholdError):
            probability_between_thresholds(MEMBERS, 0.0, bad_threshold)  # type: ignore[arg-type]
        with pytest.raises(InvalidThresholdError):
            probability_between_thresholds(MEMBERS, bad_threshold, 10.0)  # type: ignore[arg-type]
