"""Pure meteorological domain functions and conventions for 10 m wind fields.

Provides:
* Wind speed derivation from zonal (u) and meridional (v) velocity components.
* Meteorological wind direction derivation (direction FROM which wind blows,
  measured clockwise from true north in degrees [0, 360)).
* Cardinal compass point mapping (8-point and 16-point).
* Calm wind evaluation with explicit policy threshold (default 0.5 m/s).
* Ensemble vectorized speed and consensus vector flow calculations.
* Ensemble Wind Rose binning and sector probability aggregation.
* Directional and joint speed-direction probability calculations.

All calculations operate in physical units (m/s for velocity components and
speed, degrees for direction). Conversions to display units (km/h, mph) are
performed at the presentation/API boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from domain.ensemble.interval import probability_confidence_interval
from domain.exceptions import EmptyEnsembleError, InvalidEnsembleError, InvalidThresholdError

#: Policy threshold below which wind speed is classified as calm (m/s).
#: In calm conditions, wind direction is physically undefined.
CALM_WIND_THRESHOLD_MPS: float = 0.5

#: 8-point cardinal compass direction labels.
CARDINAL_DIRECTIONS_8: tuple[str, ...] = (
    "N",
    "NE",
    "E",
    "SE",
    "S",
    "SW",
    "W",
    "NW",
)

#: 16-point cardinal compass direction labels.
CARDINAL_DIRECTIONS_16: tuple[str, ...] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)

#: Physical speed bins for 8-sector Wind Rose aggregation (in m/s).
#: Each bin is [lower, upper) in m/s.
WIND_ROSE_SPEED_BINS_MPS: dict[str, tuple[float, float]] = {
    "light": (0.5, 5.5),
    "moderate": (5.5, 11.0),
    "strong": (11.0, 17.0),
    "gale": (17.0, float("inf")),
}


@dataclass(frozen=True)
class ConsensusVectorData:
    """Ensemble consensus vector flow metrics.

    Attributes:
        speed_mps: Magnitude of the mean vector (hypot(mean_u, mean_v)) in m/s.
        direction_deg: Direction of the mean vector in degrees [0, 360), or None if calm.
        cardinal: Cardinal label of the mean vector direction, or 'CALM'.
        coherence: Directional agreement index R in [0, 1].
    """

    speed_mps: float
    direction_deg: float | None
    cardinal: str
    coherence: float


@dataclass(frozen=True)
class WindRoseSector:
    """Aggregation for one directional sector in the Wind Rose.

    Attributes:
        sector: Cardinal sector code (e.g. 'N', 'SW').
        count: Number of members falling into this sector.
        probability: Fraction of total members in this sector (count / N).
        bins: Map of speed bin name to probability fraction.
        bin_counts: Map of speed bin name to integer member count.
    """

    sector: str
    count: int
    probability: float
    bins: dict[str, float]
    bin_counts: dict[str, int]


@dataclass(frozen=True)
class WindRoseData:
    """Complete 8-sector Wind Rose for an ensemble forecast.

    Attributes:
        member_count: Total ensemble members evaluated (e.g. 30).
        calm_count: Number of calm members (speed < calm_threshold).
        calm_probability: Fraction of calm members (calm_count / member_count).
        sectors: List of 8 WindRoseSector instances ordered N..NW.
    """

    member_count: int
    calm_count: int
    calm_probability: float
    sectors: list[WindRoseSector]


@overload
def derive_wind_speed(u: float, v: float) -> float: ...


@overload
def derive_wind_speed(
    u: npt.NDArray[np.floating[Any]], v: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]: ...


def derive_wind_speed(
    u: float | npt.NDArray[np.floating[Any]],
    v: float | npt.NDArray[np.floating[Any]],
) -> float | npt.NDArray[np.floating[Any]]:
    """Compute scalar wind speed from u (zonal) and v (meridional) components.

    Args:
        u: Zonal wind velocity component (m/s).
        v: Meridional wind velocity component (m/s).

    Returns:
        Scalar wind speed in m/s (sqrt(u^2 + v^2)). Returns NaN if either
        component is NaN.
    """
    if isinstance(u, (int, float)) and isinstance(v, (int, float)):
        if math.isnan(u) or math.isnan(v):
            return float("nan")
        return float(math.hypot(u, v))
    return np.hypot(u, v)


def derive_meteorological_direction(
    u: float,
    v: float,
    *,
    calm_threshold: float = CALM_WIND_THRESHOLD_MPS,
) -> float | None:
    """Compute meteorological wind direction in degrees [0, 360).

    Meteorological wind direction is defined as the azimuth FROM which the wind
    is blowing, measured clockwise from True North:
    * 0.0° / 360.0° = North (wind blowing toward South)
    * 90.0° = East (wind blowing toward West)
    * 180.0° = South (wind blowing toward North)
    * 270.0° = West (wind blowing toward East)

    Args:
        u: Zonal wind component in m/s (positive eastward flow).
        v: Meridional wind component in m/s (positive northward flow).
        calm_threshold: Speed threshold in m/s below which direction is
            undefined (calm). Defaults to ``CALM_WIND_THRESHOLD_MPS`` (0.5 m/s).

    Returns:
        Direction in degrees in [0, 360), or ``None`` if speed < calm_threshold
        or if either component is NaN / non-finite.
    """
    if math.isnan(u) or math.isnan(v) or math.isinf(u) or math.isinf(v):
        return None

    speed = math.hypot(u, v)
    if speed < calm_threshold:
        return None

    direction = math.degrees(math.atan2(-u, -v)) % 360.0
    return float(direction)


def derive_meteorological_direction_array(
    u: npt.NDArray[np.floating[Any]],
    v: npt.NDArray[np.floating[Any]],
    *,
    calm_threshold: float = CALM_WIND_THRESHOLD_MPS,
) -> npt.NDArray[np.floating[Any]]:
    """Compute meteorological wind direction array in degrees [0, 360).

    Vectorized counterpart to :func:`derive_meteorological_direction`.

    Args:
        u: Array of zonal wind components in m/s.
        v: Array of meridional wind components in m/s.
        calm_threshold: Speed threshold in m/s below which direction is NaN.

    Returns:
        Array of directions in degrees in [0, 360), with NaN for calm, missing,
        or non-finite values.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    speed = np.hypot(u_arr, v_arr)

    rad = np.arctan2(-u_arr, -v_arr)
    deg = np.degrees(rad) % 360.0

    is_calm_or_invalid = (speed < calm_threshold) | ~np.isfinite(u_arr) | ~np.isfinite(v_arr)
    return np.where(is_calm_or_invalid, np.nan, deg)


def get_cardinal_direction(
    direction_deg: float | None,
    *,
    sectors: int = 8,
) -> str | None:
    """Map a meteorological wind direction in degrees to a cardinal label.

    Args:
        direction_deg: Wind direction in degrees [0, 360), or ``None`` if calm.
        sectors: Number of compass sectors (8 or 16). Defaults to 8.

    Returns:
        Cardinal string (e.g. ``"N"``, ``"SW"``, ``"NNW"``), or ``None`` if
        direction is ``None``, NaN, or non-finite.

    Raises:
        ValueError: If ``sectors`` is not 8 or 16.
    """
    if direction_deg is None or math.isnan(direction_deg) or math.isinf(direction_deg):
        return None

    norm_deg = direction_deg % 360.0

    if sectors == 8:
        idx = int(math.floor((norm_deg + 22.5) / 45.0)) % 8
        return CARDINAL_DIRECTIONS_8[idx]

    if sectors == 16:
        idx = int(math.floor((norm_deg + 11.25) / 22.5)) % 16
        return CARDINAL_DIRECTIONS_16[idx]

    raise ValueError(f"Unsupported sector count: {sectors}. Must be 8 or 16.")


def derive_ensemble_member_speeds(
    u_members: npt.NDArray[np.floating[Any]],
    v_members: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Compute member-wise scalar wind speeds across ensemble members.

    Args:
        u_members: Array of u components across ensemble members.
        v_members: Array of v components across ensemble members.

    Returns:
        Array of scalar speeds matching the input shape.
    """
    return np.hypot(u_members, v_members)


def derive_ensemble_mean_vector(
    u_members: npt.NDArray[np.floating[Any]],
    v_members: npt.NDArray[np.floating[Any]],
    *,
    axis: int = 0,
) -> tuple[npt.NDArray[np.floating[Any]] | float, npt.NDArray[np.floating[Any]] | float]:
    """Compute the ensemble mean vector components along the member axis.

    Args:
        u_members: Array of member u components.
        v_members: Array of member v components.
        axis: Dimension along which ensemble members lie (default 0).

    Returns:
        Tuple of (mean_u, mean_v).
    """
    mean_u = np.mean(u_members, axis=axis)
    mean_v = np.mean(v_members, axis=axis)
    if np.ndim(mean_u) == 0:
        return float(mean_u), float(mean_v)
    return mean_u, mean_v


def derive_ensemble_mean_scalar_speed(
    u_members: npt.NDArray[np.floating[Any]],
    v_members: npt.NDArray[np.floating[Any]],
    *,
    axis: int = 0,
) -> npt.NDArray[np.floating[Any]] | float:
    """Compute the ensemble mean of member scalar wind speeds (mean(hypot(u_i, v_i))).

    Note that mean scalar wind speed is strictly greater than or equal to the
    magnitude of the ensemble mean vector by Jensen's inequality.

    Args:
        u_members: Array of member u components.
        v_members: Array of member v components.
        axis: Dimension along which ensemble members lie (default 0).

    Returns:
        Mean scalar wind speed.
    """
    member_speeds = np.hypot(u_members, v_members)
    mean_speed = np.mean(member_speeds, axis=axis)
    if np.ndim(mean_speed) == 0:
        return float(mean_speed)
    return mean_speed  # type: ignore[no-any-return]


def compute_consensus_vector(
    u_members: npt.NDArray[np.floating[Any]] | list[float],
    v_members: npt.NDArray[np.floating[Any]] | list[float],
    *,
    calm_threshold: float = CALM_WIND_THRESHOLD_MPS,
) -> ConsensusVectorData:
    """Compute consensus flow vector metrics across 1-D ensemble members.

    Args:
        u_members: 1-D sequence of u velocity components across members (m/s).
        v_members: 1-D sequence of v velocity components across members (m/s).
        calm_threshold: Threshold in m/s below which consensus vector is calm.

    Returns:
        A :class:`ConsensusVectorData` instance.

    Raises:
        EmptyEnsembleError: If member arrays are empty.
        InvalidEnsembleError: If member shapes mismatch or contain non-finite values.
    """
    u_arr = np.asarray(u_members, dtype=np.float64)
    v_arr = np.asarray(v_members, dtype=np.float64)

    if u_arr.size == 0 or v_arr.size == 0:
        raise EmptyEnsembleError("Ensemble member arrays cannot be empty")
    if u_arr.shape != v_arr.shape or u_arr.ndim != 1:
        raise InvalidEnsembleError("Member arrays must be 1-D with matching shapes")
    if not np.all(np.isfinite(u_arr)) or not np.all(np.isfinite(v_arr)):
        raise InvalidEnsembleError("Member arrays must contain finite numeric values")

    mean_u = float(np.mean(u_arr))
    mean_v = float(np.mean(v_arr))
    speed_mps = float(math.hypot(mean_u, mean_v))
    direction_deg = derive_meteorological_direction(
        mean_u, mean_v, calm_threshold=calm_threshold
    )
    cardinal_val = get_cardinal_direction(direction_deg)
    cardinal_str = cardinal_val if cardinal_val is not None else "CALM"

    member_speeds = np.hypot(u_arr, v_arr)
    sum_speeds = float(np.sum(member_speeds))
    coherence = speed_mps / (sum_speeds / len(u_arr)) if sum_speeds > 0 else 0.0
    coherence = min(1.0, max(0.0, coherence))

    return ConsensusVectorData(
        speed_mps=speed_mps,
        direction_deg=direction_deg,
        cardinal=cardinal_str,
        coherence=coherence,
    )


def compute_wind_rose(
    u_members: npt.NDArray[np.floating[Any]] | list[float],
    v_members: npt.NDArray[np.floating[Any]] | list[float],
    *,
    calm_threshold: float = CALM_WIND_THRESHOLD_MPS,
) -> WindRoseData:
    """Compute 8-sector Wind Rose distributions across 1-D ensemble members.

    Args:
        u_members: 1-D sequence of u velocity components across members (m/s).
        v_members: 1-D sequence of v velocity components across members (m/s).
        calm_threshold: Threshold in m/s below which a member is classified as calm.

    Returns:
        A :class:`WindRoseData` instance.

    Raises:
        EmptyEnsembleError: If member arrays are empty.
        InvalidEnsembleError: If member shapes mismatch or contain non-finite values.
    """
    u_arr = np.asarray(u_members, dtype=np.float64)
    v_arr = np.asarray(v_members, dtype=np.float64)

    if u_arr.size == 0 or v_arr.size == 0:
        raise EmptyEnsembleError("Ensemble member arrays cannot be empty")
    if u_arr.shape != v_arr.shape or u_arr.ndim != 1:
        raise InvalidEnsembleError("Member arrays must be 1-D with matching shapes")
    if not np.all(np.isfinite(u_arr)) or not np.all(np.isfinite(v_arr)):
        raise InvalidEnsembleError("Member arrays must contain finite numeric values")

    n = len(u_arr)
    speeds = np.hypot(u_arr, v_arr)

    sector_counts = {sec: 0 for sec in CARDINAL_DIRECTIONS_8}
    sector_bin_counts = {
        sec: {bin_name: 0 for bin_name in WIND_ROSE_SPEED_BINS_MPS}
        for sec in CARDINAL_DIRECTIONS_8
    }

    calm_count = 0
    for u_val, v_val, spd in zip(u_arr, v_arr, speeds, strict=True):
        if spd < calm_threshold:
            calm_count += 1
            continue

        deg = (math.degrees(math.atan2(-u_val, -v_val)) % 360.0)
        idx = int(math.floor((deg + 22.5) / 45.0)) % 8
        sector = CARDINAL_DIRECTIONS_8[idx]
        sector_counts[sector] += 1

        for bin_name, (low, high) in WIND_ROSE_SPEED_BINS_MPS.items():
            if low <= spd < high:
                sector_bin_counts[sector][bin_name] += 1
                break

    calm_prob = calm_count / n
    sectors_out: list[WindRoseSector] = []
    for sec in CARDINAL_DIRECTIONS_8:
        count = sector_counts[sec]
        prob = count / n
        bin_counts = sector_bin_counts[sec]
        bins_prob = {bin_name: c / n for bin_name, c in bin_counts.items()}
        sectors_out.append(
            WindRoseSector(
                sector=sec,
                count=count,
                probability=prob,
                bins=bins_prob,
                bin_counts=bin_counts,
            )
        )

    return WindRoseData(
        member_count=n,
        calm_count=calm_count,
        calm_probability=calm_prob,
        sectors=sectors_out,
    )


def compute_directional_probability(
    u_members: npt.NDArray[np.floating[Any]] | list[float],
    v_members: npt.NDArray[np.floating[Any]] | list[float],
    *,
    sector: str,
    speed_threshold: float | None = None,
    operator: str = "gte",
    calm_threshold: float = CALM_WIND_THRESHOLD_MPS,
) -> tuple[float, tuple[float, float]]:
    """Compute directional and joint speed-direction probability with 95% CI.

    Evaluates:
    * Direction-only: P(wind from `sector` AND speed >= calm_threshold)
    * Joint speed x direction: P(speed `operator` `speed_threshold` AND wind from `sector`)

    Args:
        u_members: 1-D sequence of u velocity components across members (m/s).
        v_members: 1-D sequence of v velocity components across members (m/s).
        sector: Target 8-point cardinal sector string ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW').
        speed_threshold: Optional speed threshold in m/s.
        operator: Comparison operator ('gte', 'gt', 'lte', 'lt'). Defaults to 'gte'.
        calm_threshold: Calm wind threshold in m/s.

    Returns:
        Tuple of (probability, (ci_lower, ci_upper)).

    Raises:
        InvalidThresholdError: If `sector` is invalid or `speed_threshold` is negative.
        EmptyEnsembleError: If member arrays are empty.
        InvalidEnsembleError: If member shapes mismatch or contain non-finite values.
    """
    sec_upper = sector.strip().upper()
    if sec_upper not in CARDINAL_DIRECTIONS_8:
        raise InvalidThresholdError(
            f"Invalid directional sector: {sector!r}. Expected one of {CARDINAL_DIRECTIONS_8}"
        )

    if speed_threshold is not None and (math.isnan(speed_threshold) or speed_threshold < 0):
        raise InvalidThresholdError(
            f"speed_threshold must be a non-negative number, got {speed_threshold}"
        )

    u_arr = np.asarray(u_members, dtype=np.float64)
    v_arr = np.asarray(v_members, dtype=np.float64)

    if u_arr.size == 0 or v_arr.size == 0:
        raise EmptyEnsembleError("Ensemble member arrays cannot be empty")
    if u_arr.shape != v_arr.shape or u_arr.ndim != 1:
        raise InvalidEnsembleError("Member arrays must be 1-D with matching shapes")
    if not np.all(np.isfinite(u_arr)) or not np.all(np.isfinite(v_arr)):
        raise InvalidEnsembleError("Member arrays must contain finite numeric values")

    n = len(u_arr)
    speeds = np.hypot(u_arr, v_arr)

    target_idx = CARDINAL_DIRECTIONS_8.index(sec_upper)
    matched = 0

    for u_val, v_val, spd in zip(u_arr, v_arr, speeds, strict=True):
        if spd < calm_threshold:
            continue

        deg = (math.degrees(math.atan2(-u_val, -v_val)) % 360.0)
        idx = int(math.floor((deg + 22.5) / 45.0)) % 8
        if idx != target_idx:
            continue

        if speed_threshold is None:
            matched += 1
            continue

        if operator in ("gte", "ge"):
            if spd >= speed_threshold:
                matched += 1
        elif operator in ("gt", ">"):
            if spd > speed_threshold:
                matched += 1
        elif operator in ("lte", "le"):
            if spd <= speed_threshold:
                matched += 1
        elif operator in ("lt", "<"):
            if spd < speed_threshold:
                matched += 1
        else:
            raise InvalidThresholdError(
                f"Unsupported operator: {operator!r}. Must be 'gte', 'gt', 'lte', or 'lt'."
            )

    prob = matched / n
    ci = probability_confidence_interval(prob, n)
    return prob, ci
