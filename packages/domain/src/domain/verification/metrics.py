"""Forecast verification metrics (root mean squared error, bias, MAE).

Each function accepts paired observed and forecast sequences and returns a
plain ``float``. All calculations are deterministic: identical inputs always
produce identical outputs (see ``ENGINEERING_CONTRACT.md`` section 5).
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from domain.exceptions import DomainError


class VerificationError(DomainError, ValueError):
    """Raised when paired verification inputs are invalid."""


def root_mean_squared_error(
    observed: Sequence[float | int] | npt.NDArray[np.float64],
    forecast: Sequence[float | int] | npt.NDArray[np.float64],
) -> float:
    """Return the root mean squared error between observed and forecast values.

    Args:
        observed: Observed values.
        forecast: Forecast values (same length as ``observed``).

    Returns:
        The root mean squared error as a ``float``.

    Raises:
        VerificationError: If either sequence is empty, not a one-dimensional
            numeric sequence, contains non-finite values, or the two sequences
            differ in length.
    """
    obs, fcst = _coerce_pairs(observed, forecast)
    return float(np.sqrt(np.mean((fcst - obs) ** 2.0)))


def mean_absolute_error(
    observed: Sequence[float | int] | npt.NDArray[np.float64],
    forecast: Sequence[float | int] | npt.NDArray[np.float64],
) -> float:
    """Return the mean absolute error between observed and forecast values.

    Args:
        observed: Observed values.
        forecast: Forecast values (same length as ``observed``).

    Returns:
        The mean absolute error as a ``float``.

    Raises:
        VerificationError: If either sequence is empty, not a one-dimensional
            numeric sequence, contains non-finite values, or the two sequences
            differ in length.
    """
    obs, fcst = _coerce_pairs(observed, forecast)
    return float(np.mean(np.abs(fcst - obs)))


def bias(
    observed: Sequence[float | int] | npt.NDArray[np.float64],
    forecast: Sequence[float | int] | npt.NDArray[np.float64],
) -> float:
    """Return the mean forecast bias (forecast minus observed).

    A positive value indicates the model over-forecasts; a negative value
    indicates under-forecasting.

    Args:
        observed: Observed values.
        forecast: Forecast values (same length as ``observed``).

    Returns:
        The mean bias as a ``float``.

    Raises:
        VerificationError: If either sequence is empty, not a one-dimensional
            numeric sequence, contains non-finite values, or the two sequences
            differ in length.
    """
    obs, fcst = _coerce_pairs(observed, forecast)
    return float(np.mean(fcst - obs))


def _is_numeric_scalar(value: object) -> bool:
    """Return whether *value* is a numeric scalar (not a bool or string).

    Booleans are excluded even though Python treats ``bool`` as a subclass of
    ``int``; ``True``/``False`` are not valid forecast/observed values.
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _has_only_numeric_elements(
    values: Sequence[object] | npt.NDArray[np.float64],
) -> bool:
    """Return whether every element of ``values`` is a numeric scalar.

    String, byte, boolean, and arbitrary-object elements are rejected so the
    element-type check runs *before* the ``float64`` conversion, preventing
    silent coercion of values such as ``"1.5"`` or ``True``.
    """
    if isinstance(values, np.ndarray):
        if values.dtype == np.bool_:
            return False
        if np.issubdtype(values.dtype, np.number):
            return True
        # Non-numeric dtype (e.g. object/string): validate each element.
        return all(_is_numeric_scalar(value) for value in values.flat)
    return all(_is_numeric_scalar(value) for value in values)


def _coerce_pair_sequence(
    values: Sequence[float | int] | npt.NDArray[np.float64],
    name: str,
) -> npt.NDArray[np.float64]:
    """Validate and normalize one side of a paired verification sequence.

    Args:
        values: The observed or forecast sequence.
        name: ``"observed"`` or ``"forecast"``, used in error messages.

    Returns:
        A one-dimensional ``numpy.float64`` array of the values.

    Raises:
        VerificationError: If the input is not a one-dimensional numeric
            sequence, is empty, contains non-numeric (e.g. boolean or string)
            elements, or contains non-finite values.
    """
    if not isinstance(values, np.ndarray) and (
        not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray))
    ):
        raise VerificationError(
            f"{name} must be a sequence of numeric values, got {type(values).__name__}"
        )

    if not _has_only_numeric_elements(values):
        raise VerificationError(
            f"{name} must be a sequence of numeric values, got elements of a "
            "non-numeric type (e.g. bool or string)"
        )

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise VerificationError(
            f"{name} must be one-dimensional, got {array.ndim} dimensions"
        )
    if array.size == 0:
        raise VerificationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise VerificationError(
            f"{name} must contain only finite numeric values"
        )
    return array


def _coerce_pairs(
    observed: Sequence[float | int] | npt.NDArray[np.float64],
    forecast: Sequence[float | int] | npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and normalize paired observed/forecast sequences.

    Args:
        observed: Observed values.
        forecast: Forecast values.

    Returns:
        The validated ``(observed, forecast)`` arrays.

    Raises:
        VerificationError: If either sequence is invalid or the two sequences
            differ in length.
    """
    obs = _coerce_pair_sequence(observed, "observed")
    fcst = _coerce_pair_sequence(forecast, "forecast")
    if obs.size != fcst.size:
        raise VerificationError(
            "observed and forecast must contain the same number of values, "
            f"got {obs.size} and {fcst.size}"
        )
    return obs, fcst
