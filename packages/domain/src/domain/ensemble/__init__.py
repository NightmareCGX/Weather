"""Ensemble statistics, exceedance probability, and confidence interval modules.

Public API (all functions return primitive ``float`` values except
``probability_confidence_interval`` which returns a ``(lower, upper)`` tuple):

- ``ensemble_mean``
- ``ensemble_median``
- ``ensemble_spread``
- ``ensemble_percentile``
- ``probability_above_threshold``
- ``probability_below_threshold``
- ``probability_between_thresholds``
- ``probability_confidence_interval``

Shared input validation and internal helpers are private and not exported.
"""

from domain.ensemble.interval import probability_confidence_interval
from domain.ensemble.probability import (
    probability_above_threshold,
    probability_below_threshold,
    probability_between_thresholds,
)
from domain.ensemble.statistics import (
    ensemble_mean,
    ensemble_median,
    ensemble_percentile,
    ensemble_spread,
)

__all__ = [
    "ensemble_mean",
    "ensemble_median",
    "ensemble_spread",
    "ensemble_percentile",
    "probability_above_threshold",
    "probability_below_threshold",
    "probability_between_thresholds",
    "probability_confidence_interval",
]
