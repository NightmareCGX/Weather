"""Canonical ensemble Probability Density Function (PDF) estimation module.

Computes a 1-D Gaussian Kernel Density Estimate (KDE) over a finite ensemble
member sample using Silverman's robust bandwidth rule with an IQR=0 fallback.
The evaluation grid spans ``[min(members) - 3h, max(members) + 3h]`` on 100
linearly spaced points to preserve continuous tail behavior without truncation.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from domain.ensemble._validation import _coerce_members
from domain.ensemble.statistics import ensemble_percentile, ensemble_spread

#: Number of points in the canonical evaluation grid.
_KDE_POINTS: int = 100

#: Tail padding factor in units of bandwidth h for grid support.
_KDE_PADDING_SIGMA: float = 3.0

#: Standard normal normalization constant sqrt(2 * pi).
_SQRT_2PI: float = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class EnsemblePDF:
    """Canonical ensemble probability density function evaluation.

    Attributes:
        x: Linearly spaced evaluation coordinates spanning [min - 3h, max + 3h].
        density: Mathematically normalized Gaussian KDE density values at x.
    """

    x: list[float]
    density: list[float]


def _compute_bandwidth(array: npt.NDArray[np.float64]) -> float | None:
    """Compute Silverman robust bandwidth with IQR=0 fallback.

    Args:
        array: Coerced 1-D numpy float64 array of ensemble member values.

    Returns:
        Bandwidth float h, or None if the sample has size < 2 or zero spread.
    """
    n = array.size
    if n < 2:
        return None

    std = ensemble_spread(array)
    if std <= 0.0:
        return None

    p75 = ensemble_percentile(array, 75.0)
    p25 = ensemble_percentile(array, 25.0)
    iqr_scale = (p75 - p25) / 1.34

    scale = min(std, iqr_scale) if iqr_scale > 0.0 else std

    return float(0.9 * scale * (n**-0.2))


def _evaluate_gaussian_kde(
    array: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    h: float,
) -> npt.NDArray[np.float64]:
    """Evaluate standard Gaussian KDE on a 1-D grid.

    Args:
        array: 1-D numpy float64 array of ensemble member values.
        grid: 1-D numpy float64 array of evaluation points.
        h: Bandwidth float > 0.

    Returns:
        1-D numpy float64 array of density values at each grid point.
    """
    diff = (grid[:, np.newaxis] - array[np.newaxis, :]) / h
    kernels = np.exp(-0.5 * (diff**2)) / _SQRT_2PI
    return np.mean(kernels, axis=1) / h


def estimate_ensemble_pdf(
    members: Sequence[float | int] | npt.NDArray[np.float64],
) -> EnsemblePDF | None:
    """Estimate the canonical 1-D Gaussian PDF for an ensemble member sample.

    Calculates a 1-D Gaussian Kernel Density Estimate with Silverman's robust
    bandwidth rule and IQR=0 fallback on a canonical 100-point grid spanning
    ``[min(members) - 3h, max(members) + 3h]``.

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.

    Returns:
        An EnsemblePDF with x and density coordinates, or None if the ensemble
        has fewer than 2 members or zero spread.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, or contains non-finite values.
    """
    array = _coerce_members(members)
    h = _compute_bandwidth(array)
    if h is None:
        return None

    min_val = float(np.min(array))
    max_val = float(np.max(array))
    x_min = min_val - _KDE_PADDING_SIGMA * h
    x_max = max_val + _KDE_PADDING_SIGMA * h

    grid = np.linspace(x_min, x_max, _KDE_POINTS, dtype=np.float64)
    densities = _evaluate_gaussian_kde(array, grid, h)

    return EnsemblePDF(
        x=[float(val) for val in grid],
        density=[float(val) for val in densities],
    )
