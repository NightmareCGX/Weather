"""Ensemble statistics, exceedance probability, confidence interval, and PDF modules.

Public API (all functions return primitive ``float`` values except
``probability_confidence_interval`` which returns a ``(lower, upper)`` tuple,
and ``estimate_ensemble_pdf`` which returns an ``EnsemblePDF | None``):

- ``ensemble_mean``
- ``ensemble_median``
- ``ensemble_spread``
- ``ensemble_percentile``
- ``estimate_ensemble_pdf``
- ``EnsemblePDF``
- ``probability_above_threshold``
- ``probability_below_threshold``
- ``probability_between_thresholds``
- ``probability_confidence_interval``

Shared input validation and internal helpers are private and not exported.
"""

from domain.ensemble.interval import probability_confidence_interval
from domain.ensemble.pdf import EnsemblePDF, estimate_ensemble_pdf
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
    "EnsemblePDF",
    "ensemble_mean",
    "ensemble_median",
    "ensemble_percentile",
    "ensemble_spread",
    "estimate_ensemble_pdf",
    "probability_above_threshold",
    "probability_below_threshold",
    "probability_between_thresholds",
    "probability_confidence_interval",
]
