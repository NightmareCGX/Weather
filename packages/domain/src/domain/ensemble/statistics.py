"""Ensemble statistical functions for the weather forecasting platform.

Each function accepts a flat sequence of ensemble member values and returns a
plain ``float``. All calculations are deterministic: identical inputs always
produce identical outputs (see ``ENGINEERING_CONTRACT.md`` section 5).
"""

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from domain.ensemble._validation import _coerce_members
from domain.exceptions import InvalidPercentileError


def ensemble_mean(members: Sequence[float | int] | npt.NDArray[np.float64]) -> float:
    """Return the arithmetic mean of the ensemble members.

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.

    Returns:
        The arithmetic mean as a ``float``.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, or contains non-finite values.
    """
    array = _coerce_members(members)
    return float(np.mean(array))


def ensemble_median(members: Sequence[float | int] | npt.NDArray[np.float64]) -> float:
    """Return the median of the ensemble members.

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.

    Returns:
        The median as a ``float``.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, or contains non-finite values.
    """
    array = _coerce_members(members)
    return float(np.median(array))


def ensemble_spread(members: Sequence[float | int] | npt.NDArray[np.float64]) -> float:
    """Return the ensemble spread as the population standard deviation.

    The spread uses the population standard deviation (``ddof=0``), the
    standard NWP convention for ensemble dispersion.

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.

    Returns:
        The population standard deviation as a ``float``.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, or contains non-finite values.
    """
    array = _coerce_members(members)
    return float(np.std(array, ddof=0))


def ensemble_percentile(
    members: Sequence[float | int] | npt.NDArray[np.float64], q: float
) -> float:
    """Return the ``q``-th percentile of the ensemble members.

    Percentiles use NumPy's default ``linear`` interpolation between closest
    ranks, matching the convention used across the platform.

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.
        q: Percentile in ``[0, 100]``.

    Returns:
        The requested percentile as a ``float``.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, or contains non-finite values.
        InvalidPercentileError: If ``q`` is not a finite number in ``[0, 100]``.
    """
    array = _coerce_members(members)
    if isinstance(q, bool) or not isinstance(q, (int, float, np.integer, np.floating)):
        raise InvalidPercentileError(
            f"percentile must be a numeric value, got {type(q).__name__}"
        )
    if not math.isfinite(float(q)) or not (0.0 <= float(q) <= 100.0):
        raise InvalidPercentileError(
            f"percentile must be between 0 and 100, got {q}"
        )
    return float(np.percentile(array, float(q), method="linear"))
