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
import struct
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from domain.ensemble.interval import probability_confidence_interval
from domain.exceptions import (
    EmptyEnsembleError,
    InvalidEnsembleError,
    InvalidGridError,
    InvalidThresholdError,
)

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


VECTOR_FIELD_MAGIC = b"WNDQ"
VECTOR_FIELD_VERSION = 1
VECTOR_FIELD_FLAG_INT16 = 1
VECTOR_FIELD_HEADER_FMT = "<4sBBHf ffIffI"
VECTOR_FIELD_HEADER_SIZE = struct.calcsize(VECTOR_FIELD_HEADER_FMT)
EARTH_RADIUS_METERS = 6371000.0


@dataclass(frozen=True)
class VectorGridMetadata:
    """Metadata describing a quantized 2-D vector wind grid.

    Attributes:
        lat_start: Latitude coordinate of row index 0.
        lat_step: Uniform latitude step between adjacent rows.
        lat_count: Number of latitude rows.
        lon_start: Longitude coordinate of column index 0.
        lon_step: Uniform longitude step between adjacent columns.
        lon_count: Number of longitude columns.
        scale: Physical scale factor (e.g. 0.01 for 0.01 m/s precision).
    """

    lat_start: float
    lat_step: float
    lat_count: int
    lon_start: float
    lon_step: float
    lon_count: int
    scale: float = 0.01

    def __post_init__(self) -> None:
        if self.lat_count < 2 or self.lon_count < 2:
            raise InvalidGridError(
                "Vector grid must have at least 2 latitude rows and 2 longitude columns."
            )
        if self.lat_step == 0.0 or self.lon_step == 0.0:
            raise InvalidGridError("Vector grid step sizes must be non-zero.")
        if self.scale <= 0.0 or not math.isfinite(self.scale):
            raise InvalidThresholdError(
                f"Scale factor must be positive and finite, got {self.scale}"
            )
        if not (math.isfinite(self.lat_start) and math.isfinite(self.lon_start)):
            raise InvalidGridError("Vector grid coordinates must be finite.")


def encode_vector_field_int16(
    u: npt.NDArray[np.floating[Any]],
    v: npt.NDArray[np.floating[Any]],
    *,
    lat_start: float,
    lat_step: float,
    lon_start: float,
    lon_step: float,
    scale: float = 0.01,
) -> bytes:
    """Encode a 2-D (lat, lon) wind vector field to quantized Int16 binary.

    Args:
        u: 2-D array of zonal wind velocity (m/s).
        v: 2-D array of meridional wind velocity (m/s).
        lat_start: Latitude of row 0.
        lat_step: Uniform latitude spacing.
        lon_start: Longitude of column 0.
        lon_step: Uniform longitude spacing.
        scale: Precision scale factor (m/s per discrete int16 quantum).

    Returns:
        Packed bytes with 36-byte header followed by contiguous u then v int16 buffers.

    Raises:
        InvalidGridError: If shapes mismatch, ndim != 2, or axes are degenerate.
        InvalidThresholdError: If scale <= 0.
    """
    u_arr = np.asarray(u, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)

    if u_arr.ndim != 2 or v_arr.ndim != 2:
        raise InvalidGridError("u and v fields must be 2-dimensional (lat, lon) arrays.")
    if u_arr.shape != v_arr.shape:
        raise InvalidGridError(f"Shape mismatch: u {u_arr.shape} vs v {v_arr.shape}.")
    if not np.all(np.isfinite(u_arr)) or not np.all(np.isfinite(v_arr)):
        raise InvalidGridError("Vector fields must contain finite numeric values.")

    meta = VectorGridMetadata(
        lat_start=lat_start,
        lat_step=lat_step,
        lat_count=u_arr.shape[0],
        lon_start=lon_start,
        lon_step=lon_step,
        lon_count=u_arr.shape[1],
        scale=scale,
    )

    header = struct.pack(
        VECTOR_FIELD_HEADER_FMT,
        VECTOR_FIELD_MAGIC,
        VECTOR_FIELD_VERSION,
        VECTOR_FIELD_FLAG_INT16,
        0,  # reserved
        float(meta.scale),
        float(meta.lat_start),
        float(meta.lat_step),
        meta.lat_count,
        float(meta.lon_start),
        float(meta.lon_step),
        meta.lon_count,
    )

    u_i16 = np.clip(np.round(u_arr / meta.scale), -32768, 32767).astype("<i2")
    v_i16 = np.clip(np.round(v_arr / meta.scale), -32768, 32767).astype("<i2")

    return bytes(header + u_i16.tobytes() + v_i16.tobytes())


def decode_vector_field_int16(
    raw_bytes: bytes,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], VectorGridMetadata]:
    """Decode a quantized Int16 binary wind vector field.

    Args:
        raw_bytes: Binary payload starting with 36-byte WNDQ header.

    Returns:
        Tuple of (u_field_mps, v_field_mps, metadata).

    Raises:
        ValueError: If magic, version, flags, or payload size is invalid.
    """
    if len(raw_bytes) < VECTOR_FIELD_HEADER_SIZE:
        raise ValueError(
            f"Payload too short: {len(raw_bytes)} bytes < header size {VECTOR_FIELD_HEADER_SIZE}."
        )

    header = raw_bytes[:VECTOR_FIELD_HEADER_SIZE]
    (
        magic,
        ver,
        flags,
        _res,
        scale,
        lat_start,
        lat_step,
        lat_count,
        lon_start,
        lon_step,
        lon_count,
    ) = struct.unpack(VECTOR_FIELD_HEADER_FMT, header)

    if magic != VECTOR_FIELD_MAGIC:
        raise ValueError(f"Invalid magic header: {magic!r}, expected {VECTOR_FIELD_MAGIC!r}.")
    if ver != VECTOR_FIELD_VERSION:
        raise ValueError(f"Unsupported version: {ver}, expected {VECTOR_FIELD_VERSION}.")
    if flags != VECTOR_FIELD_FLAG_INT16:
        raise ValueError(f"Unsupported flags: {flags}, expected {VECTOR_FIELD_FLAG_INT16}.")

    meta = VectorGridMetadata(
        lat_start=float(lat_start),
        lat_step=float(lat_step),
        lat_count=int(lat_count),
        lon_start=float(lon_start),
        lon_step=float(lon_step),
        lon_count=int(lon_count),
        scale=float(scale),
    )

    num_points = meta.lat_count * meta.lon_count
    expected_payload_len = VECTOR_FIELD_HEADER_SIZE + num_points * 2 * 2
    if len(raw_bytes) != expected_payload_len:
        raise ValueError(
            f"Payload size mismatch: got {len(raw_bytes)} bytes, expected {expected_payload_len} "
            f"for {meta.lat_count}x{meta.lon_count} grid."
        )

    offset = VECTOR_FIELD_HEADER_SIZE
    u_bytes_len = num_points * 2
    u_i16 = np.frombuffer(raw_bytes, dtype="<i2", count=num_points, offset=offset).reshape(
        (meta.lat_count, meta.lon_count)
    )
    offset += u_bytes_len
    v_i16 = np.frombuffer(raw_bytes, dtype="<i2", count=num_points, offset=offset).reshape(
        (meta.lat_count, meta.lon_count)
    )

    u_mps = (u_i16.astype(np.float32) * meta.scale)
    v_mps = (v_i16.astype(np.float32) * meta.scale)

    return u_mps, v_mps, meta


def advect_particle(
    lat: float,
    lon: float,
    u: float,
    v: float,
    *,
    dt_seconds: float = 60.0,
    lat_clamp: float = 85.0,
    earth_radius_m: float = EARTH_RADIUS_METERS,
) -> tuple[float, float]:
    """Advance a geographic particle position given local (u, v) velocity in m/s.

    Accounts for meridional curvature (Earth radius) and zonal convergence
    at high latitudes (cos(lat)), with pole clamp to avoid numerical singularity.

    Args:
        lat: Current latitude in [-90, 90] degrees.
        lon: Current longitude in degrees.
        u: Zonal wind speed in m/s (eastward positive).
        v: Meridional wind speed in m/s (northward positive).
        dt_seconds: Advection step duration in seconds.
        lat_clamp: Latitude threshold in degrees beyond which cos(lat) is clamped.
        earth_radius_m: Mean Earth radius in meters.

    Returns:
        Tuple of (new_latitude, new_longitude) with normalized longitude in [-180, 180].
    """
    if math.isnan(lat) or math.isnan(lon) or math.isnan(u) or math.isnan(v):
        return float("nan"), float("nan")

    # Clamped latitude for numerical stability in 1/cos(lat)
    clamped_lat = max(-lat_clamp, min(lat_clamp, lat))
    rad_lat = math.radians(clamped_lat)
    cos_lat = max(math.cos(rad_lat), math.cos(math.radians(lat_clamp)))

    # Angular displacement (degrees)
    # dlat = (v * dt) / R * (180 / pi)
    dlat_deg = (v * dt_seconds / earth_radius_m) * (180.0 / math.pi)
    # dlon = (u * dt) / (R * cos(lat)) * (180 / pi)
    dlon_deg = (u * dt_seconds / (earth_radius_m * cos_lat)) * (180.0 / math.pi)

    new_lat = max(-lat_clamp, min(lat_clamp, lat + dlat_deg))
    new_lon = lon + dlon_deg

    # Normalize longitude to [-180, 180]
    norm_lon = ((new_lon + 180.0) % 360.0) - 180.0
    if norm_lon == -180.0 and new_lon > 0:
        norm_lon = 180.0

    return new_lat, norm_lon


def sample_vector_bilinear(
    u_grid: npt.NDArray[np.floating[Any]],
    v_grid: npt.NDArray[np.floating[Any]],
    meta: VectorGridMetadata,
    lat: float,
    lon: float,
) -> tuple[float, float]:
    """Bilinear interpolation of U and V velocity components at (lat, lon).

    Args:
        u_grid: 2-D array of shape (lat_count, lon_count).
        v_grid: 2-D array of shape (lat_count, lon_count).
        meta: VectorGridMetadata describing grid coordinates.
        lat: Target latitude.
        lon: Target longitude.

    Returns:
        Interpolated (u, v) in m/s.
    """
    if math.isnan(lat) or math.isnan(lon):
        return float("nan"), float("nan")

    # Row fractional index
    row_f = (lat - meta.lat_start) / meta.lat_step
    row_f = max(0.0, min(float(meta.lat_count - 1), row_f))

    # Longitude alignment into [lon_start, lon_start + span]
    lon_norm = lon % 360.0
    aligned_lon = lon_norm if meta.lon_start >= 0.0 else ((lon + 180.0) % 360.0) - 180.0

    col_f = (aligned_lon - meta.lon_start) / meta.lon_step
    # Periodic longitude wrapping for global grids
    col_f = col_f % float(meta.lon_count)

    r0 = int(math.floor(row_f))
    r1 = min(meta.lat_count - 1, r0 + 1)
    dr = row_f - r0

    c0 = int(math.floor(col_f))
    c1 = (c0 + 1) % meta.lon_count
    dc = col_f - c0

    # 4-corner weights
    w00 = (1.0 - dr) * (1.0 - dc)
    w01 = (1.0 - dr) * dc
    w10 = dr * (1.0 - dc)
    w11 = dr * dc

    u_val = float(
        w00 * u_grid[r0, c0]
        + w01 * u_grid[r0, c1]
        + w10 * u_grid[r1, c0]
        + w11 * u_grid[r1, c1]
    )
    v_val = float(
        w00 * v_grid[r0, c0]
        + w01 * v_grid[r0, c1]
        + w10 * v_grid[r1, c0]
        + w11 * v_grid[r1, c1]
    )

    return u_val, v_val

