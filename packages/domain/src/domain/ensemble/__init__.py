"""Ensemble statistics, exceedance probability, confidence interval, and PDF modules.

Public API (all functions return primitive ``float`` values except
``probability_confidence_interval`` which returns a ``(lower, upper)`` tuple,
and ``estimate_ensemble_pdf`` which returns an ``EnsemblePDF | None``):

- ``ensemble_mean``
- ``ensemble_median``
- ``ensemble_spread``
- ``ensemble_percentile``
- ``estimate_ensemble_pdf``
- ``estimate_pdf_from_quantiles``
- ``estimate_pdf_from_bins``
- ``quantile_point_mass``
- ``EnsemblePDF``
- ``probability_above_threshold``
- ``probability_below_threshold``
- ``probability_between_thresholds``
- ``probability_confidence_interval``

Shared input validation and internal helpers are private and not exported.
"""

from domain.ensemble.interval import probability_confidence_interval
from domain.ensemble.pdf import (
    EnsemblePDF,
    estimate_ensemble_pdf,
    estimate_pdf_from_bins,
    estimate_pdf_from_quantiles,
    quantile_point_mass,
)
from domain.ensemble.probability import (
    probability_above_threshold,
    probability_at_or_above_threshold,
    probability_at_or_below_threshold,
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
    "estimate_pdf_from_bins",
    "estimate_pdf_from_quantiles",
    "probability_above_threshold",
    "probability_at_or_above_threshold",
    "probability_at_or_below_threshold",
    "probability_below_threshold",
    "probability_between_thresholds",
    "probability_confidence_interval",
    "quantile_point_mass",
]
