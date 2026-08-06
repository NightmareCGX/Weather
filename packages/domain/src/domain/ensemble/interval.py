"""Confidence interval calculation for ensemble exceedance probabilities.

The Wilson score interval is a deterministic, closed-form confidence interval
for a binomial proportion. Unlike the Wald normal approximation, it does not
collapse to a zero-width interval when the observed proportion is 0 or 1, and
it behaves well for the small ensemble sizes typical of NWP ensembles.

The 95% confidence interval returned here backs the ``confidence_interval_95``
field of the ``/v1/probabilities`` response (API.md section 3.1). The
``confidence`` parameter exists for forward compatibility (e.g. 90% intervals)
but only the documented 95% level is supported for now.
"""

import math

import numpy as np

from domain.exceptions import InvalidEnsembleError, InvalidThresholdError

#: Standard normal critical value for a two-sided 95% interval (z_{0.975}).
#: Hardcoded so the interval is fully deterministic and independent of any
#: external statistics library.
_Z_95 = 1.959963984540054


def probability_confidence_interval(
    probability: float,
    sample_size: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a sample proportion.

    The interval is computed from the observed exceedance probability and the
    number of ensemble members alone (no resampling), so identical inputs
    always produce identical outputs (``ENGINEERING_CONTRACT.md`` section 5).

    Args:
        probability: The observed proportion in ``[0, 1]``.
        sample_size: The number of observations (ensemble members), at least 1.
        confidence: The confidence level. Only ``0.95`` is currently
            supported; any other value raises ``InvalidThresholdError``.

    Returns:
        A ``(lower, upper)`` tuple with the interval bounds clamped to
        ``[0, 1]``.

    Raises:
        InvalidThresholdError: If ``probability`` is not a finite number in
            ``[0, 1]``, or ``confidence`` is not ``0.95``.
        InvalidEnsembleError: If ``sample_size`` is not a positive integer.
    """
    if isinstance(probability, bool) or not isinstance(
        probability, (int, float, np.integer, np.floating)
    ):
        raise InvalidThresholdError(
            "probability must be a numeric value, "
            f"got {type(probability).__name__}"
        )
    if not math.isfinite(float(probability)) or not (
        0.0 <= float(probability) <= 1.0
    ):
        raise InvalidThresholdError(
            f"probability must be between 0 and 1, got {probability}"
        )
    if isinstance(sample_size, bool) or not isinstance(
        sample_size, (int, np.integer)
    ):
        raise InvalidEnsembleError(
            "sample_size must be a positive integer, "
            f"got {type(sample_size).__name__}"
        )
    if sample_size < 1:
        raise InvalidEnsembleError(
            f"sample_size must be at least 1, got {sample_size}"
        )
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float, np.integer, np.floating)
    ):
        raise InvalidThresholdError(
            f"confidence must be a numeric value, got {type(confidence).__name__}"
        )
    if float(confidence) != 0.95:
        raise InvalidThresholdError(
            f"unsupported confidence level {confidence}; only 0.95 is supported"
        )

    p = float(probability)
    n = float(sample_size)
    z = _Z_95
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (p + z_squared / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / n + z_squared / (4.0 * n * n))
        / denominator
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper
