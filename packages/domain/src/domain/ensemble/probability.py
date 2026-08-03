"""Threshold exceedance probability functions for ensemble members.

Each function returns the empirical fraction of ensemble members satisfying a
threshold comparison, expressed as a plain ``float`` in ``[0, 1]``. All
calculations are deterministic (see ``ENGINEERING_CONTRACT.md`` section 5).
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from domain.ensemble._validation import _coerce_members
from domain.exceptions import InvalidThresholdError


def probability_above_threshold(
    members: Sequence[float | int] | npt.NDArray[np.float64], threshold: float
) -> float:
    """Return the fraction of members strictly greater than a threshold.

    A member equal to the threshold does not count (strict ``>``).

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.
        threshold: Threshold value.

    Returns:
        The fraction of members above the threshold as a ``float`` in
        ``[0, 1]``.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, or contains non-finite values.
        InvalidThresholdError: If the threshold is not a finite number.
    """
    validated_threshold = _validate_threshold(threshold)
    return _empirical_fraction(members, lambda values: values > validated_threshold)


def probability_below_threshold(
    members: Sequence[float | int] | npt.NDArray[np.float64], threshold: float
) -> float:
    """Return the fraction of members strictly less than a threshold.

    A member equal to the threshold does not count (strict ``<``).

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.
        threshold: Threshold value.

    Returns:
        The fraction of members below the threshold as a ``float`` in
        ``[0, 1]``.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, or contains non-finite values.
        InvalidThresholdError: If the threshold is not a finite number.
    """
    validated_threshold = _validate_threshold(threshold)
    return _empirical_fraction(members, lambda values: values < validated_threshold)


def probability_between_thresholds(
    members: Sequence[float | int] | npt.NDArray[np.float64],
    lower: float,
    upper: float,
) -> float:
    """Return the fraction of members within an inclusive threshold range.

    A member equal to either boundary counts (inclusive ``[lower, upper]``).

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.
        lower: Inclusive lower bound.
        upper: Inclusive upper bound, must be at least ``lower``.

    Returns:
        The fraction of members between the thresholds as a ``float`` in
        ``[0, 1]``.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, or contains non-finite values.
        InvalidThresholdError: If a threshold is not a finite number or
            ``upper`` is below ``lower``.
    """
    validated_lower = _validate_threshold(lower)
    validated_upper = _validate_threshold(upper)
    if validated_upper < validated_lower:
        raise InvalidThresholdError(
            f"upper threshold {validated_upper} must be at least "
            f"lower threshold {validated_lower}"
        )
    return _empirical_fraction(
        members, lambda values: (values >= validated_lower) & (values <= validated_upper)
    )


def _empirical_fraction(
    members: Sequence[float | int] | npt.NDArray[np.float64],
    predicate: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]],
) -> float:
    """Return the fraction of members satisfying a predicate.

    Args:
        members: Ensemble member values to evaluate.
        predicate: Boolean predicate applied element-wise to the member array.

    Returns:
        The fraction of members for which the predicate is True as a ``float``
        in ``[0, 1]``.
    """
    array = _coerce_members(members)
    return float(np.mean(predicate(array)))


def _validate_threshold(threshold: float) -> float:
    """Validate and normalize a probability threshold.

    Args:
        threshold: Threshold value.

    Returns:
        The threshold as a ``float``.

    Raises:
        InvalidThresholdError: If the threshold is not a finite number.
    """
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise InvalidThresholdError(
            f"threshold must be a numeric value, got {type(threshold).__name__}"
        )
    if not math.isfinite(float(threshold)):
        raise InvalidThresholdError(f"threshold must be finite, got {threshold}")
    return float(threshold)
