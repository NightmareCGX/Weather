r"""Pure meteorological domain functions and types for cloud products.

Provides:
* Reset-window interval-average reconstruction for 3-hour cloud cover (`cloud_cover_3h`).
* Numerical guardrail clipping and invalidity handling for cloud cover (tolerance: ±5%).
* Cloud ceiling classification into finite heights and the unlimited ceiling sentinel.
* GEFS ensemble cloud-cover statistics with dynamic valid-member denominators ($N_{valid} \ge 21$).
* GEFS ensemble cloud-ceiling summaries separating discrete unlimited probability from
  conditional finite ceiling distributions ($N_{finite} \ge 10$).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Engineering guardrail tolerance for reset-window reconstruction (percentage points).
#: Accounts for upstream GRIB2 packing quantization and radiation timestep averaging.
CLOUD_COVER_RECONSTRUCTION_TOLERANCE_PERCENT: float = 5.0

#: Upstream geopotential height sentinel threshold for unlimited ceiling (km).
#: In NCEP GRIB2 encoding, clear sky / no qualifying ceiling is encoded near 20,000 gpm (20 km).
CLOUD_CEILING_UNLIMITED_THRESHOLD_KM: float = 19.99
CLOUD_CEILING_UNLIMITED_SENTINEL_KM: float = 20.0
# Backward-compatibility aliases
CLOUD_CEILING_UNLIMITED_THRESHOLD_M: float = CLOUD_CEILING_UNLIMITED_THRESHOLD_KM
CLOUD_CEILING_UNLIMITED_SENTINEL_M: float = CLOUD_CEILING_UNLIMITED_SENTINEL_KM

#: Minimum valid ensemble members required out of 30 for ensemble statistics.
#: If invalid_count >= 10 (N_valid <= 20), the ensemble result is marked invalid.
CLOUD_COVER_MIN_VALID_MEMBERS: int = 21

#: Minimum finite ceiling members required to compute robust conditional finite percentiles.
#: If N_finite < 10, conditional percentiles are suppressed (None) while P(Unlimited) is retained.
CLOUD_CEILING_MIN_FINITE_MEMBERS: int = 10

#: Standard conversion factor from meters to feet.
METERS_TO_FEET: float = 3.28084
#: Standard conversion factor from kilometers to feet.
KM_TO_FEET: float = 3280.84


@dataclass(frozen=True, slots=True)
class CloudCeilingClassification:
    """Classification outcome for a single cloud ceiling measurement."""

    is_unlimited: bool
    height_km: float | None

    @property
    def height_m(self) -> float | None:
        """Backward compatibility property returning height."""
        return self.height_km


@dataclass(frozen=True, slots=True)
class CloudCoverEnsembleSummary:
    """Ensemble statistical summary for cloud cover computed strictly over valid members."""

    valid_member_count: int
    invalid_member_count: int
    mean: float
    median: float
    spread: float
    percentiles: dict[str, float]


@dataclass(frozen=True, slots=True)
class CloudCeilingEnsembleSummary:
    """Ensemble summary separating discrete unlimited ceiling probability from finite stats."""

    valid_member_count: int
    finite_member_count: int
    unlimited_member_count: int
    unlimited_probability: float
    conditional_mean: float | None
    conditional_median: float | None
    conditional_spread: float | None
    conditional_percentiles: dict[str, float] | None

    @property
    def conditional_mean_m(self) -> float | None:
        return self.conditional_mean

    @property
    def conditional_median_m(self) -> float | None:
        return self.conditional_median

    @property
    def conditional_spread_m(self) -> float | None:
        return self.conditional_spread

    @property
    def conditional_percentiles_m(self) -> dict[str, float] | None:
        return self.conditional_percentiles


def reconstruct_running_average_interval(
    current_reset_avg: float | NDArray[np.floating[Any]],
    predecessor_avg: float | NDArray[np.floating[Any]],
    *,
    reset_period_hours: int,
    interval_width_hours: int,
    tolerance: float = CLOUD_COVER_RECONSTRUCTION_TOLERANCE_PERCENT,
) -> float | NDArray[np.floating[Any]]:
    r"""Reconstruct a W-width interval average at a reset lead from running averages.

    General form for a variable whose upstream fields are running-since-reset
    averages over ``R = reset_period_hours`` (architecture doc I19; W =
    ``interval_width_hours``, requires ``0 < W <= R``):

    $$X_W = \frac{R \cdot C_{R}(L) - (R - W) \cdot C(L - W)}{W}$$

    where ``C(L)`` is the running-since-reset average at the reset lead L and
    ``C(L - W)`` the running average at the predecessor lead. The historically
    hardcoded cloud formula ``x = 2 * C6 - C3`` is exactly the (R=6, W=3) special
    case: ``(6*C6 - 3*C3)/3``. The pairing ``W == R / 2`` is a configuration
    coincidence and must never be assumed by generic logic.

    Guardrail application:
    * $[0.0, 100.0]$: Valid physical range, retained unchanged.
    * $[-tolerance, 0.0)$: Minor numerical undershoot, clipped to $0.0$.
    * $(100.0, 100.0 + tolerance]$: Minor numerical overshoot, clipped to $100.0$.
    * $< -tolerance$ or $> 100.0 + tolerance$: Invalid reconstruction, set to NaN.

    Args:
        current_reset_avg: Running-since-reset average at the reset lead L.
        predecessor_avg: Running average at the predecessor lead L - W.
        reset_period_hours: R, the upstream reset period in hours.
        interval_width_hours: W, the target interval width in hours.
        tolerance: Reconstruction guardrail tolerance (percentage points, default 5.0).

    Returns:
        Reconstructed W-width interval average (scalar float or NumPy array).
    """
    r_hours = int(reset_period_hours)
    w_hours = int(interval_width_hours)
    if r_hours <= 0 or w_hours <= 0:
        raise ValueError(
            f"reset_period_hours and interval_width_hours must be positive, got "
            f"W={w_hours}, R={r_hours}"
        )
    if w_hours > r_hours:
        raise ValueError(
            f"interval width W={w_hours} must not exceed reset period R={r_hours}"
        )

    if w_hours == r_hours:
        # Upstream already provides the W-width quantity at the reset lead.
        return _apply_reconstruction_guardrails(current_reset_avg, current_reset_avg, tolerance)

    if isinstance(current_reset_avg, np.ndarray) or isinstance(
        predecessor_avg, np.ndarray
    ):
        cur = np.asarray(current_reset_avg, dtype=np.float64)
        pred = np.asarray(predecessor_avg, dtype=np.float64)
        x = (r_hours * cur - (r_hours - w_hours) * pred) / float(w_hours)
        return _apply_reconstruction_guardrails(x, current_reset_avg, tolerance)

    if math.isnan(current_reset_avg) or math.isnan(predecessor_avg):
        return float("nan")

    x_val = (
        r_hours * float(current_reset_avg) - (r_hours - w_hours) * float(predecessor_avg)
    ) / float(w_hours)
    return _apply_reconstruction_guardrails_scalar(x_val, tolerance)


def _apply_reconstruction_guardrails(
    x: float | NDArray[np.floating[Any]],
    template: float | NDArray[np.floating[Any]],
    tolerance: float,
) -> float | NDArray[np.floating[Any]]:
    """Apply the reconstruction guardrails to a pre-computed reconstruction."""
    if isinstance(x, np.ndarray):
        result = np.full_like(x, np.nan, dtype=np.float64)
        valid_mask = (x >= 0.0) & (x <= 100.0)
        result[valid_mask] = x[valid_mask]
        undershoot_mask = (x >= -tolerance) & (x < 0.0)
        result[undershoot_mask] = 0.0
        overshoot_mask = (x > 100.0) & (x <= 100.0 + tolerance)
        result[overshoot_mask] = 100.0
        if isinstance(template, np.ndarray) and template.dtype == np.float32:
            return result.astype(np.float32)
        return result
    return _apply_reconstruction_guardrails_scalar(float(x), tolerance)


def _apply_reconstruction_guardrails_scalar(x_val: float, tolerance: float) -> float:
    if math.isnan(x_val):
        return float("nan")
    if 0.0 <= x_val <= 100.0:
        return x_val
    if -tolerance <= x_val < 0.0:
        return 0.0
    if 100.0 < x_val <= (100.0 + tolerance):
        return 100.0
    return float("nan")


def reconstruct_cloud_cover_3h(
    current_6h_avg: float | NDArray[np.floating[Any]],
    prev_3h_avg: float | NDArray[np.floating[Any]],
    tolerance: float = CLOUD_COVER_RECONSTRUCTION_TOLERANCE_PERCENT,
) -> float | NDArray[np.floating[Any]]:
    r"""Reconstruct 3-hour interval-averaged cloud cover at 6-hour reset leads.

    Legacy (R=6, W=3) special case of :func:`reconstruct_running_average_interval`:
    at reset leads ($f006, f012, \dots$), upstream GRIB provides the 6-hour average
    $\bar{C}_{t-6, t}$ and the second 3-hour mean $\bar{C}_{t-3, t}$ is reconstructed as
    $$x = 2 \cdot C_{6\text{h}} - C_{3\text{h}}(t-3)$$

    New call sites must use :func:`reconstruct_running_average_interval` with
    explicit (R, W) from :data:`domain.temporal.VARIABLE_TEMPORAL_METADATA`.

    Args:
        current_6h_avg: 6-hour interval-average cloud cover at current reset lead.
        prev_3h_avg: Preceding 3-hour interval-average cloud cover.
        tolerance: Reconstruction guardrail tolerance in percentage points (default 5.0).

    Returns:
        Reconstructed 3-hour cloud cover (scalar float or NumPy array).
    """
    return reconstruct_running_average_interval(
        current_6h_avg,
        prev_3h_avg,
        reset_period_hours=6,
        interval_width_hours=3,
        tolerance=tolerance,
    )


def classify_cloud_ceiling(
    value_km: float | None,
    threshold: float = CLOUD_CEILING_UNLIMITED_THRESHOLD_KM,
) -> CloudCeilingClassification:
    """Classify a cloud ceiling value into finite height or unlimited sky.

    Args:
        value_km: Cloud ceiling height in kilometers (or None).
        threshold: Height threshold in kilometers above which ceiling is unlimited.

    Returns:
        A :class:`CloudCeilingClassification` instance.
    """
    if value_km is None or math.isnan(value_km):
        return CloudCeilingClassification(is_unlimited=False, height_km=None)
    if value_km >= threshold:
        return CloudCeilingClassification(is_unlimited=True, height_km=None)
    return CloudCeilingClassification(is_unlimited=False, height_km=float(value_km))


def cloud_cover_ensemble_summary(
    members: Sequence[float | int | None] | NDArray[np.floating[Any]],
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    min_valid: int = CLOUD_COVER_MIN_VALID_MEMBERS,
) -> CloudCoverEnsembleSummary | None:
    r"""Compute ensemble statistics for cloud cover over valid members.

    Enforces the one-third quality rule: at least `min_valid` (21) members must be finite
    and within $[0.0, 100.0]$.

    Args:
        members: Sequence of ensemble member cloud cover values (0-100%).
        percentiles: Percentiles to compute (e.g. 10, 25, 50, 75, 90).
        min_valid: Minimum required valid members (default 21).

    Returns:
        A :class:`CloudCoverEnsembleSummary` or `None` if $N_{valid} < min\_valid$.
    """
    valid_vals: list[float] = []
    invalid_count = 0

    for m in members:
        if m is not None and not math.isnan(m) and 0.0 <= float(m) <= 100.0:
            valid_vals.append(float(m))
        else:
            invalid_count += 1

    valid_count = len(valid_vals)
    if valid_count < min_valid:
        return None

    arr = np.array(valid_vals, dtype=np.float64)
    pct_dict = {
        f"p{int(p) if float(p).is_integer() else p}": float(
            np.percentile(arr, p, method="linear")
        )
        for p in percentiles
    }

    return CloudCoverEnsembleSummary(
        valid_member_count=valid_count,
        invalid_member_count=invalid_count,
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        spread=float(np.std(arr, ddof=0)),
        percentiles=pct_dict,
    )


def cloud_ceiling_ensemble_summary(
    members: Sequence[float | int | None] | NDArray[np.floating[Any]],
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    min_finite: int = CLOUD_CEILING_MIN_FINITE_MEMBERS,
    min_valid: int = CLOUD_COVER_MIN_VALID_MEMBERS,
    threshold: float = CLOUD_CEILING_UNLIMITED_THRESHOLD_KM,
) -> CloudCeilingEnsembleSummary | None:
    r"""Compute ensemble cloud ceiling summary separating unlimited probability from finite stats.

    Semantics:
    * $N_{valid} = N_{finite} + N_{unlimited}$.
    * Overall validity requires $N_{valid} \ge min\_valid$ (21). If $N_{valid} < 21$, returns None.
    * $P(\text{Unlimited}) = N_{unlimited} / N_{valid}$.
    * If $N_{finite} \ge min\_finite$ (10), computes conditional percentiles over finite members.
    * If $N_{finite} < 10$, conditional percentiles and conditional mean/median/spread are None.

    Args:
        members: Sequence of ensemble member ceiling heights in kilometers.
        percentiles: Percentiles to compute for conditional finite distribution.
        min_finite: Minimum finite members required for finite distribution (default 10).
        min_valid: Minimum valid members required for ensemble validity (default 21).
        threshold: Threshold above which ceiling is classified as unlimited.

    Returns:
        A :class:`CloudCeilingEnsembleSummary` or `None` if $N_{valid} < min\_valid$.
    """
    finite_vals: list[float] = []
    unlimited_count = 0

    for m in members:
        if m is None or math.isnan(m):
            continue
        val = float(m)
        if val >= threshold:
            unlimited_count += 1
        elif val >= 0.0:
            finite_vals.append(val)

    finite_count = len(finite_vals)
    valid_count = finite_count + unlimited_count

    if valid_count < min_valid:
        return None

    unlimited_prob = unlimited_count / valid_count

    if finite_count >= min_finite:
        arr = np.array(finite_vals, dtype=np.float64)
        pct_dict = {
            f"p{int(p) if float(p).is_integer() else p}": float(
                np.percentile(arr, p, method="linear")
            )
            for p in percentiles
        }
        mean_val: float | None = float(np.mean(arr))
        median_val: float | None = float(np.median(arr))
        spread_val: float | None = float(np.std(arr, ddof=0))
        pcts: dict[str, float] | None = pct_dict
    else:
        mean_val = None
        median_val = None
        spread_val = None
        pcts = None

    return CloudCeilingEnsembleSummary(
        valid_member_count=valid_count,
        finite_member_count=finite_count,
        unlimited_member_count=unlimited_count,
        unlimited_probability=unlimited_prob,
        conditional_mean=mean_val,
        conditional_median=median_val,
        conditional_spread=spread_val,
        conditional_percentiles=pcts,
    )


def compute_low_ceiling_probability(
    members: Sequence[float | int | None] | NDArray[np.floating[Any]],
    threshold_km: float,
    min_valid: int = CLOUD_COVER_MIN_VALID_MEMBERS,
    sentinel_threshold: float = CLOUD_CEILING_UNLIMITED_THRESHOLD_KM,
) -> float | None:
    r"""Compute probability of cloud ceiling being less than or equal to a height threshold.

    Unlimited ceiling members evaluate as False (ceiling > threshold_km) and contribute to the
    valid denominator $N_{valid}$.

    $$P(C \le h) = \frac{\#\{\text{finite members with } C \le h\}}{N_{valid}}$$

    Args:
        members: Sequence of ensemble member ceiling heights in kilometers.
        threshold_km: Ceiling height threshold in kilometers.
        min_valid: Minimum required valid members (default 21).
        sentinel_threshold: Threshold above which ceiling is unlimited.

    Returns:
        Probability in $[0.0, 1.0]$ or `None` if $N_{valid} < min\_valid$.
    """
    finite_matching = 0
    valid_count = 0

    for m in members:
        if m is None or math.isnan(m):
            continue
        val = float(m)
        if val >= sentinel_threshold:
            valid_count += 1
        elif val >= 0.0:
            valid_count += 1
            if val <= threshold_km:
                finite_matching += 1

    if valid_count < min_valid:
        return None

    return finite_matching / valid_count
