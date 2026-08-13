"""Unit tests for domain.ensemble statistics functions."""

import math

import numpy as np
import numpy.typing as npt
import pytest
from domain.ensemble import (
    ensemble_mean,
    ensemble_median,
    ensemble_percentile,
    ensemble_spread,
)
from domain.exceptions import (
    EmptyEnsembleError,
    InvalidEnsembleError,
    InvalidPercentileError,
)

#: Hand-computed ensemble member values used across the statistics tests.
MEMBERS: list[float] = [0.0, 10.0, 20.0, 30.0, 40.0]

#: Mean of :data:`MEMBERS`.
EXPECTED_MEAN = 20.0

#: Median of :data:`MEMBERS`.
EXPECTED_MEDIAN = 20.0

#: Population standard deviation of :data:`MEMBERS`.
EXPECTED_SPREAD = math.sqrt(200.0)


class TestEnsembleMean:
    def test_known_values(self) -> None:
        assert ensemble_mean(MEMBERS) == pytest.approx(EXPECTED_MEAN)

    def test_single_element(self) -> None:
        assert ensemble_mean([7.0]) == pytest.approx(7.0)

    def test_returns_float(self) -> None:
        assert isinstance(ensemble_mean(MEMBERS), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert ensemble_mean(array) == pytest.approx(EXPECTED_MEAN)

    def test_accepts_tuple(self) -> None:
        assert ensemble_mean(tuple(MEMBERS)) == pytest.approx(EXPECTED_MEAN)

    def test_integer_members(self) -> None:
        assert ensemble_mean([1, 2, 3, 4]) == pytest.approx(2.5)


class TestEnsembleMedian:
    def test_known_values(self) -> None:
        assert ensemble_median(MEMBERS) == pytest.approx(EXPECTED_MEDIAN)

    def test_single_element(self) -> None:
        assert ensemble_median([42.0]) == pytest.approx(42.0)

    def test_even_member_count_averages_middle(self) -> None:
        assert ensemble_median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_odd_member_count_returns_middle(self) -> None:
        assert ensemble_median([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_returns_float(self) -> None:
        assert isinstance(ensemble_median(MEMBERS), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert ensemble_median(array) == pytest.approx(EXPECTED_MEDIAN)

    def test_unsorted_input_matches_sorted(self) -> None:
        assert ensemble_median([40.0, 10.0, 30.0, 0.0, 20.0]) == pytest.approx(EXPECTED_MEDIAN)


class TestEnsembleSpread:
    def test_known_values(self) -> None:
        assert ensemble_spread(MEMBERS) == pytest.approx(EXPECTED_SPREAD)

    def test_single_element_is_zero(self) -> None:
        assert ensemble_spread([5.0]) == pytest.approx(0.0)

    def test_identical_members_is_zero(self) -> None:
        assert ensemble_spread([3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_two_element_spread(self) -> None:
        # Population standard deviation of {0, 2} is 1.0.
        assert ensemble_spread([0.0, 2.0]) == pytest.approx(1.0)

    def test_returns_float(self) -> None:
        assert isinstance(ensemble_spread(MEMBERS), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert ensemble_spread(array) == pytest.approx(EXPECTED_SPREAD)


class TestEnsemblePercentile:
    def test_known_percentiles(self) -> None:
        # NumPy linear interpolation of {0, 10, 20, 30, 40}.
        assert ensemble_percentile(MEMBERS, 0) == pytest.approx(0.0)
        assert ensemble_percentile(MEMBERS, 10) == pytest.approx(4.0)
        assert ensemble_percentile(MEMBERS, 25) == pytest.approx(10.0)
        assert ensemble_percentile(MEMBERS, 50) == pytest.approx(20.0)
        assert ensemble_percentile(MEMBERS, 75) == pytest.approx(30.0)
        assert ensemble_percentile(MEMBERS, 90) == pytest.approx(36.0)
        assert ensemble_percentile(MEMBERS, 100) == pytest.approx(40.0)

    def test_interpolates_between_ranks(self) -> None:
        assert ensemble_percentile([0.0, 100.0], 50) == pytest.approx(50.0)

    def test_returns_float(self) -> None:
        assert isinstance(ensemble_percentile(MEMBERS, 50), float)

    def test_accepts_numpy_array(self) -> None:
        array: npt.NDArray[np.float64] = np.asarray(MEMBERS, dtype=np.float64)
        assert ensemble_percentile(array, 50) == pytest.approx(20.0)

    @pytest.mark.parametrize(
        "invalid_q",
        [
            -0.1,
            100.1,
            math.nan,
            math.inf,
            -math.inf,
            "50",
            None,
            True,
        ],
    )
    def test_invalid_percentile_rejected(self, invalid_q: object) -> None:
        with pytest.raises(InvalidPercentileError):
            ensemble_percentile(MEMBERS, invalid_q)  # type: ignore[arg-type]


class TestSharedValidation:
    @pytest.mark.parametrize(
        "bad_members",
        [
            [],
            (),
            np.asarray([], dtype=np.float64),
        ],
    )
    def test_empty_members_rejected(self, bad_members: object) -> None:
        with pytest.raises(EmptyEnsembleError):
            ensemble_mean(bad_members)  # type: ignore[arg-type]

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(InvalidEnsembleError):
            ensemble_mean("")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_members",
        [
            "12,3",
            [1, "two", 3],
            [1, None, 3],
            [1, [2], 3],
            [1.0, {"x": 2}],
        ],
    )
    def test_non_numeric_members_rejected(self, bad_members: object) -> None:
        with pytest.raises(InvalidEnsembleError):
            ensemble_mean(bad_members)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_members",
        [
            # Numeric strings that ``np.float64`` would otherwise coerce.
            ["1.5", "2.5"],
            ["1", 2, 3],
            # Booleans are not member values even though bool is an int.
            [True, False, True],
            [False, 1.0],
            # Numpy boolean / string / object arrays.
            np.asarray([True, False]),
            np.asarray(["1.5", "2.5"]),
            np.asarray([1.0, "x", 3.0], dtype=object),
            # Bytes are treated like strings (silently convertible).
            [b"1.5", b"2.5"],
        ],
    )
    def test_silent_coercion_rejected(self, bad_members: object) -> None:
        with pytest.raises(InvalidEnsembleError):
            ensemble_mean(bad_members)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_members",
        [
            [1.0, math.nan, 3.0],
            [1.0, math.inf, 3.0],
            [1.0, -math.inf, 3.0],
        ],
    )
    def test_non_finite_members_rejected(self, bad_members: object) -> None:
        with pytest.raises(InvalidEnsembleError):
            ensemble_mean(bad_members)  # type: ignore[arg-type]

    def test_two_dimensional_numpy_array_rejected(self) -> None:
        matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with pytest.raises(InvalidEnsembleError):
            ensemble_mean(matrix)

    def test_nested_list_members_rejected(self) -> None:
        with pytest.raises(InvalidEnsembleError):
            ensemble_mean([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_members", [3, 3.5, None, True])
    def test_scalar_members_rejected(self, bad_members: object) -> None:
        with pytest.raises(InvalidEnsembleError):
            ensemble_mean(bad_members)  # type: ignore[arg-type]

    def test_numpy_scalar_members_accepted(self) -> None:
        # Int/float numpy scalars remain valid member values.
        members = [np.float64(1.5), np.int32(2), 3.0]
        assert ensemble_mean(members) == pytest.approx((1.5 + 2.0 + 3.0) / 3.0)

    def test_numeric_object_dtype_array_accepted(self) -> None:
        # An object-dtype array holding numeric scalars is not silently
        # coerced and is accepted.
        array = np.asarray([np.float64(1.0), np.int32(2.0)], dtype=object)
        assert ensemble_mean(array) == pytest.approx(1.5)



def test_complex_members_rejected() -> None:
    """Complex numbers are not ensemble members: converting them to float64
    would silently drop the imaginary part (review finding MINOR-complex).
    """
    with pytest.raises(InvalidEnsembleError):
        ensemble_mean(np.array([1.0 + 2.0j, 3.0 + 4.0j]))
    with pytest.raises(InvalidEnsembleError):
        ensemble_mean([1.0 + 2.0j, 3.0 + 4.0j])
