"""Unit tests for domain.verification metrics functions."""

import math

import numpy as np
import numpy.typing as npt
import pytest
from domain.verification import (
    bias,
    mean_absolute_error,
    root_mean_squared_error,
)
from domain.verification.metrics import VerificationError

#: Hand-computed paired observed/forecast values used across the tests.
OBSERVED = [0.0, 2.0, 4.0, 6.0]
FORECAST = [1.0, 1.0, 5.0, 6.0]

#: Per-pair errors (forecast - observed): [1.0, -1.0, 1.0, 0.0].
#: bias   = mean(errors)            = 0.25
#: MAE    = mean(|errors|)          = 0.75
#: RMSE   = sqrt(mean(errors**2))   = sqrt(0.75)
EXPECTED_BIAS = 0.25
EXPECTED_MAE = 0.75
EXPECTED_RMSE = math.sqrt(0.75)


class TestRootMeanSquaredError:
    def test_known_values(self) -> None:
        assert root_mean_squared_error(OBSERVED, FORECAST) == pytest.approx(EXPECTED_RMSE)

    def test_returns_float(self) -> None:
        assert isinstance(root_mean_squared_error(OBSERVED, FORECAST), float)

    def test_accepts_numpy_arrays(self) -> None:
        obs: npt.NDArray[np.float64] = np.asarray(OBSERVED, dtype=np.float64)
        fcst: npt.NDArray[np.float64] = np.asarray(FORECAST, dtype=np.float64)
        assert root_mean_squared_error(obs, fcst) == pytest.approx(EXPECTED_RMSE)

    def test_accepts_tuples(self) -> None:
        assert root_mean_squared_error(tuple(OBSERVED), tuple(FORECAST)) == pytest.approx(
            EXPECTED_RMSE
        )

    def test_single_pair(self) -> None:
        assert root_mean_squared_error([10.0], [7.0]) == pytest.approx(3.0)

    def test_perfect_forecast_is_zero(self) -> None:
        assert root_mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


class TestMeanAbsoluteError:
    def test_known_values(self) -> None:
        assert mean_absolute_error(OBSERVED, FORECAST) == pytest.approx(EXPECTED_MAE)

    def test_returns_float(self) -> None:
        assert isinstance(mean_absolute_error(OBSERVED, FORECAST), float)

    def test_accepts_numpy_arrays(self) -> None:
        obs: npt.NDArray[np.float64] = np.asarray(OBSERVED, dtype=np.float64)
        fcst: npt.NDArray[np.float64] = np.asarray(FORECAST, dtype=np.float64)
        assert mean_absolute_error(obs, fcst) == pytest.approx(EXPECTED_MAE)

    def test_single_pair(self) -> None:
        assert mean_absolute_error([10.0], [7.0]) == pytest.approx(3.0)

    def test_perfect_forecast_is_zero(self) -> None:
        assert mean_absolute_error([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


class TestBias:
    def test_known_values(self) -> None:
        assert bias(OBSERVED, FORECAST) == pytest.approx(EXPECTED_BIAS)

    def test_returns_float(self) -> None:
        assert isinstance(bias(OBSERVED, FORECAST), float)

    def test_accepts_numpy_arrays(self) -> None:
        obs: npt.NDArray[np.float64] = np.asarray(OBSERVED, dtype=np.float64)
        fcst: npt.NDArray[np.float64] = np.asarray(FORECAST, dtype=np.float64)
        assert bias(obs, fcst) == pytest.approx(EXPECTED_BIAS)

    def test_over_forecast_is_positive(self) -> None:
        assert bias([1.0, 2.0], [3.0, 4.0]) == pytest.approx(2.0)

    def test_under_forecast_is_negative(self) -> None:
        assert bias([5.0, 5.0], [3.0, 3.0]) == pytest.approx(-2.0)

    def test_single_pair(self) -> None:
        assert bias([10.0], [7.0]) == pytest.approx(-3.0)


class TestSharedValidation:
    @pytest.mark.parametrize(
        "bad_values",
        [[], (), np.asarray([], dtype=np.float64)],
    )
    def test_empty_observed_rejected(self, bad_values: object) -> None:
        with pytest.raises(VerificationError):
            root_mean_squared_error(bad_values, FORECAST)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_values",
        [[], (), np.asarray([], dtype=np.float64)],
    )
    def test_empty_forecast_rejected(self, bad_values: object) -> None:
        with pytest.raises(VerificationError):
            root_mean_squared_error(OBSERVED, bad_values)  # type: ignore[arg-type]

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(VerificationError):
            mean_absolute_error([1.0, 2.0], [1.0])

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(VerificationError):
            bias("", FORECAST)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_values",
        [
            "1,2,3",
            [1.0, "two", 3.0],
            [1.0, None, 3.0],
            [1.0, [2.0], 3.0],
        ],
    )
    def test_non_numeric_values_rejected(self, bad_values: object) -> None:
        with pytest.raises(VerificationError):
            root_mean_squared_error(bad_values, FORECAST)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_values",
        [
            # Numeric strings that ``np.float64`` would otherwise coerce.
            ["1.5", "2.5"],
            ["1", 2.0],
            # Booleans are not observed/forecast values.
            [True, False],
            [False, 1.0],
            # Numpy boolean / string / object arrays.
            np.asarray([True, False]),
            np.asarray(["1.5"]),
            np.asarray([1.0, "x"], dtype=object),
        ],
    )
    def test_silent_coercion_rejected(self, bad_values: object) -> None:
        with pytest.raises(VerificationError):
            root_mean_squared_error(bad_values, [1.0, 2.0])  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_values",
        [
            [1.0, math.nan, 3.0],
            [1.0, math.inf, 3.0],
            [1.0, -math.inf, 3.0],
        ],
    )
    def test_non_finite_values_rejected(self, bad_values: object) -> None:
        with pytest.raises(VerificationError):
            root_mean_squared_error(bad_values, FORECAST)  # type: ignore[arg-type]

    def test_two_dimensional_numpy_array_rejected(self) -> None:
        matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with pytest.raises(VerificationError):
            root_mean_squared_error(matrix, FORECAST)

    @pytest.mark.parametrize("bad_values", [3, 3.5, None, True])
    def test_scalar_values_rejected(self, bad_values: object) -> None:
        with pytest.raises(VerificationError):
            root_mean_squared_error(bad_values, FORECAST)  # type: ignore[arg-type]

    def test_all_functions_share_validation(self) -> None:
        with pytest.raises(VerificationError):
            bias([1.0], [1.0, 2.0])
        with pytest.raises(VerificationError):
            mean_absolute_error([1.0], [])



def test_complex_observation_pairs_rejected() -> None:
    """Complex observations are not verification data: float64 conversion
    would silently drop the imaginary part (review finding MINOR-complex).
    """
    with pytest.raises(VerificationError):
        root_mean_squared_error(np.array([1.0 + 2.0j, 2.0 + 1.0j]), np.array([1.0, 2.0]))
    with pytest.raises(VerificationError):
        root_mean_squared_error([1.0 + 2.0j, 2.0 + 1.0j], [1.0, 2.0])
