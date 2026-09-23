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

#: Most probability, in members, a stored quantile function may hold at one constant value before
#: it is read as discontinuous rather than as a distribution.
#:
#: A continuous reconstruction spreads a point mass into a bell, which overstates the curve's
#: spread by roughly the ratio of the two widths. One member is the bound because a *sample* of n
#: members cannot put two distinct observations at one value without them being the same
#: observation: a stretch wider than that is a mass the encoding is failing to state, and the
#: remaining levels around it describe a distribution whose shape is not this one. The measured
#: separation is wide -- every sample under this bound stayed below 0.9x the ensemble's own
#: sampling noise, and every sample above it reached 1.6x or more -- so the exact bound is not
#: delicate, and one member is the smallest value that needs no justification.
_MAX_STATED_POINT_MASS: float = 1.0


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


def quantile_point_mass(
    coverage_levels: Sequence[float] | npt.NDArray[np.floating],
    values: Sequence[float] | npt.NDArray[np.floating],
    member_count: int,
) -> float:
    """Members' worth of probability for which a quantile function holds one constant value.

    A quantile function is continuous by construction, so a *point mass* -- several members at
    the same value -- appears in it as a stretch of probability over which the value does not
    rise. This measures the widest such stretch, in members.

    It exists because a KDE is dominated by that mass and the encoding cannot state it: measured
    on the real 09-23 06Z GEFS store, a frame whose 27 of 30 members sat at a ceiling reconstructed
    with a KDE **5.6x** the ensemble's own sampling noise, while every frame with no collapse wider
    than a member stayed under 0.9. So a caller deciding whether a stored distribution can answer a
    KDE asks this first.

    Args:
        coverage_levels: The stored probability levels, strictly increasing.
        values: The stored value at each level, in the same order.
        member_count: Members the container was computed from, for the conversion.

    Returns:
        The widest constant stretch in members; ``0.0`` when the values strictly increase, which
        is what a strictly monotone encoding reports for a continuous distribution.
    """
    widest = 0.0
    for start in range(len(coverage_levels) - 1):
        value = values[start]
        for end in range(start + 1, len(coverage_levels)):
            if values[end] > value:
                break
            widest = max(
                widest,
                (float(coverage_levels[end]) - float(coverage_levels[start])) * member_count,
            )
    return widest


def estimate_pdf_from_quantiles(
    coverage_levels: Sequence[float] | npt.NDArray[np.floating],
    values: Sequence[float] | npt.NDArray[np.floating],
    member_count: int,
) -> EnsemblePDF | None:
    """The canonical KDE of a distribution stated as a quantile function.

    A KDE is an integral against the distribution -- ``(1/n) sum_i K_h(x - X_i) = integral
    K_h(x - t) dF(t)`` -- so a sampled quantile function is enough to evaluate it, and the sample
    it was built from does not have to exist. The store's containers carry exactly that: 19 levels
    per cell, each in the variable's own units.

    The bandwidth is the member path's own rule (Silverman's, on the reconstructed sample), and the
    grid is the member path's own rule too, so the two curves lie on one coordinate system rather
    than merely resembling each other.

    **A distribution with a point mass wider than one member returns ``None``.** The reconstruction
    would smear that mass into a bell and overstate the curve's spread by several times the
    ensemble's own sampling noise (see :func:`quantile_point_mass`); the honest answer is no curve
    than a wrong one, and the caller keeps whatever it had.

    Args:
        coverage_levels: The stored probability levels, strictly increasing.
        values: The stored value at each level, in the same order.
        member_count: Members the container was computed from.

    Returns:
        An EnsemblePDF on the canonical grid, or ``None`` when the distribution is degenerate
        (no spread) or carries a point mass the encoding cannot state.
    """
    levels = np.asarray([float(level) for level in coverage_levels], dtype=np.float64)
    stored = np.asarray([float(value) for value in values], dtype=np.float64)
    if levels.size < 2 or stored.size != levels.size:
        return None
    if not np.isfinite(stored).all() or not np.isfinite(levels).all():
        return None
    if quantile_point_mass(coverage_levels, values, member_count) > _MAX_STATED_POINT_MASS:
        return None

    # The measure implied by the levels, as a density over the reconstructed values: between two
    # levels the quantile function is linear, so its probability mass is the level gap.
    dense_levels = np.linspace(float(levels[0]), float(levels[-1]), _KDE_POINTS * 40)
    dense_values = np.interp(dense_levels, levels, stored)
    weights = np.gradient(dense_levels)
    return _pdf_from_weighted_sample(dense_values, weights, member_count)


def estimate_pdf_from_bins(
    centers: Sequence[float] | npt.NDArray[np.floating],
    weights: Sequence[float] | npt.NDArray[np.floating],
    *,
    member_count: int,
    bin_count: int,
) -> EnsemblePDF | None:
    """The canonical KDE of a distribution stated as a normalised histogram.

    The bin encoding stores a shape over ``mean +- sigma_range * std`` rather than a quantile
    function, so the reconstruction is the histogram itself: each bin's mass at its centre. A
    histogram is already a binned distribution, so it carries no point mass the encoding cannot
    state, and no collapse check applies.

    Args:
        centers: Bin centres, in the variable's own units.
        weights: Probability in each bin, in the same order.
        member_count: Members the container was computed from.
        bin_count: Bins the encoding declares, for the bandwidth's sample size.

    Returns:
        An EnsemblePDF on the canonical grid, or ``None`` for a degenerate shape.
    """
    values = np.asarray([float(value) for value in centers], dtype=np.float64)
    mass = np.asarray([float(value) for value in weights], dtype=np.float64)
    if values.size != mass.size or values.size < 2:
        return None
    if not np.isfinite(values).all() or not np.isfinite(mass).all():
        return None
    # A zero total is left for :func:`_pdf_from_weighted_sample`, which has to check it anyway.
    return _pdf_from_weighted_sample(values, np.maximum(mass, 0.0), max(2, bin_count))


def _pdf_from_weighted_sample(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    sample_size: int,
) -> EnsemblePDF | None:
    """Evaluate the canonical KDE over a weighted sample.

    The shared half of the two stored forms, and of the member path's own evaluation: one
    bandwidth rule, one grid rule, one kernel. ``sample_size`` is what Silverman's rule scales by,
    not the number of weights -- a reconstructed sample's size is the member count the container
    records, because that is how many observations the distribution was estimated from.
    """
    total = float(weights.sum())
    if total <= 0.0:
        return None
    normalized = weights / total
    mean = float(np.sum(normalized * values))
    std = float(np.sqrt(np.sum(normalized * (values - mean) ** 2)))
    if std <= 0.0:
        return None

    order = np.argsort(values)
    cumulative = np.cumsum(normalized[order])
    p25 = float(np.interp(0.25, cumulative, values[order]))
    p75 = float(np.interp(0.75, cumulative, values[order]))
    iqr_scale = (p75 - p25) / 1.34
    # Positive by the spread check above: either branch of the choice is a positive number.
    scale = min(std, iqr_scale) if iqr_scale > 0.0 else std
    h = float(0.9 * scale * (max(2, sample_size) ** -0.2))

    lowest = float(values.min())
    highest = float(values.max())
    grid = np.linspace(
        lowest - _KDE_PADDING_SIGMA * h, highest + _KDE_PADDING_SIGMA * h, _KDE_POINTS,
        dtype=np.float64,
    )
    diff = (grid[:, np.newaxis] - values[np.newaxis, :]) / h
    kernels = np.exp(-0.5 * (diff**2)) / _SQRT_2PI
    densities = (normalized[np.newaxis, :] * kernels).sum(axis=1) / h

    return EnsemblePDF(
        x=[float(value) for value in grid],
        density=[float(value) for value in densities],
    )

