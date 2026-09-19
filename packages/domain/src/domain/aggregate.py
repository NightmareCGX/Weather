"""Per-variable aggregate encodings for ensemble forecast stores.

An ensemble's members are replaced by a small set of *statistic fields* computed per grid
cell from the whole member set. Which fields depends on the variable class, because the
classes fail differently:

``mean+std+bins`` (near-Gaussian variables)
    ``MEAN``, ``STD``, then ``n_bins`` normalised histogram bins over
    ``(x - MEAN) / STD``. The normalisation is not cosmetic: a raw histogram would need a
    global value range, and a variable spanning -60..+50 with a local spread of ~2 would
    then have bins far coarser than the local distribution and cells whose local mean sits
    away from the global mean would fall outside the range entirely. Normalising per cell
    puts the bins on the local distribution, and makes MEAN and STD structurally required
    (the shape carries neither location nor scale).

Every field is quantised to a fixed point so the stored integers have the zeroed low bits
that make the whole scheme compress (the member fields they replace inherit that structure
from GRIB2 packing; numpy-computed fields do not).

Reference semantics
-------------------
:func:`compute_aggregate` is the definition of the encoding. Its float arithmetic is
deliberate and load-bearing:

* ``mean``/``std`` (not the ``nan*`` variants) -- global model fields have no missing
  members, and the NaN-aware forms cost 7x and 2.5x more for identical values.
* a bin index from ``np.digitize`` and a **division** for the normalisation -- not a
  shifted ``floor`` and not a multiply by a precomputed reciprocal. Both alternatives were
  measured to disagree with the reference on real data: one rounded 0.49999976 to 0.5 and
  moved 6 of 30 members out of a bin, the other differed on 6 of 33 million cells.

Anything that recomputes an aggregate must reproduce these values bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

#: Number of histogram bins for the near-Gaussian encoding. See the encoding notes: 16 bins
#: suffice through the body of the distribution, 32 extend coverage to the P90-P99.9 range,
#: and 64 buys nothing further on the tail while costing another 1.5x.
DEFAULT_N_BINS: Final[int] = 32

#: Half-width of the normalised bin support. Members beyond this many standard deviations
#: fall outside every bin, so a cell's bins sum to slightly less than 1 in the tail (a
#: property of the convention, not a defect). Documented because it is also why the bins
#: cannot answer very rare exceedance thresholds.
DEFAULT_SIGMA_RANGE: Final[float] = 4.0

#: Fixed-point scales: value units per stored integer step. A scale of ``s`` bounds the
#: quantisation error at ``s/2``.
#:
#: Bins are probabilities -- multiples of ``1/member_count`` -- and the decoded CDF sums
#: them, so an error of ``s/2`` per bin accumulates across bins. At ``0.01`` that is a
#: worst case of 0.16 on a CDF; at ``0.001`` it is 0.016, for 0.3% more bytes.
BIN_SCALE: Final[float] = 0.001

#: MEAN and STD are in the variable's own units and span a wide range, so they get their own
#: scales rather than sharing the bins'. ``0.01`` covers +-327.67, which spans every
#: currently ingested variable's plausible range.
MEAN_SCALE: Final[float] = 0.01
STD_SCALE: Final[float] = 0.01

#: Sentinel for "no value" in a quantised field. int16 minimum, one step below the smallest
#: representable quantised value, so it cannot collide with a real value.
NAN_SENTINEL: Final[int] = -32768

#: Floor applied to the normalisation denominator. Zero-spread cells exist (a member that
#: is identically zero across the grid) and dividing by zero would make every bin NaN.
_STD_FLOOR: Final[float] = 1e-9


class AggregateError(ValueError):
    """Raised when an aggregate cannot be computed or encoded as requested."""


@dataclass(frozen=True)
class AggregateSpec:
    """Which fields an aggregate carries, and how each is quantised.

    Attributes:
        kind: Encoding family, currently only ``"mean_std_bins"``.
        n_bins: Number of normalised histogram bins.
        sigma_range: Bin support half-width in units of the local standard deviation.
    """

    kind: str = "mean_std_bins"
    n_bins: int = DEFAULT_N_BINS
    sigma_range: float = DEFAULT_SIGMA_RANGE

    def __post_init__(self) -> None:
        if self.kind != "mean_std_bins":
            raise AggregateError(f"unknown aggregate kind {self.kind!r}")
        if self.n_bins < 1:
            raise AggregateError(f"n_bins must be positive, got {self.n_bins}")
        if not self.sigma_range > 0.0:
            raise AggregateError(
                f"sigma_range must be positive, got {self.sigma_range}"
            )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in storage order: ``MEAN``, ``STD``, then the bins."""
        return ("MEAN", "STD", *(f"BIN{i:02d}" for i in range(self.n_bins)))

    @property
    def field_scales(self) -> tuple[float, ...]:
        """Fixed-point scale per field, in :attr:`field_names` order."""
        return (MEAN_SCALE, STD_SCALE, *(BIN_SCALE,) * self.n_bins)

    @property
    def n_fields(self) -> int:
        """Total field count, i.e. the number of stored lat/lon planes."""
        return 2 + self.n_bins

    def bin_edges(self) -> npt.NDArray[np.float32]:
        """Bin edges in normalised units, as float32.

        The dtype matters: the reference computes bin membership by comparing float32
        normalised values against these edges, so returning float64 would change which
        side of an edge a value on the boundary lands.
        """
        return np.linspace(
            -self.sigma_range, self.sigma_range, self.n_bins + 1
        ).astype(np.float32)


def compute_aggregate(
    members: npt.NDArray[np.floating],
    spec: AggregateSpec,
) -> npt.NDArray[np.float32]:
    """Compute the aggregate fields from a member stack.

    Args:
        members: Array of shape ``(n_members, lat, lon)``. Members must all be present; a
            NaN in any member makes every field for that cell NaN, because a partially
            observed cell has no well-defined ensemble statistic.
        spec: Field set and quantisation.

    Returns:
        ``(n_fields, lat, lon)`` float32 array in :attr:`AggregateSpec.field_names` order.

    Raises:
        AggregateError: if ``members`` is not a non-empty 3-D array.
    """
    if members.ndim != 3:
        raise AggregateError(
            f"members must be (n_members, lat, lon); got shape {members.shape}"
        )
    if members.shape[0] < 1:
        raise AggregateError("members must contain at least one member")

    stack = members.astype(np.float32, copy=False)
    n_members = stack.shape[0]

    # mean/std over a complete member axis: identical to the NaN-aware forms here, and
    # 7x/2.5x cheaper. Any NaN in the stack propagates to NaN, which is the intended
    # semantics for a cell that is not fully observed.
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)

    # A zero-spread cell (every member identical) has no shape; the floor keeps the
    # normalisation finite without making the bins meaningful, and matches the reference's
    # floor exactly so the two agree bit for bit.
    denominator = np.maximum(std, np.float32(_STD_FLOOR))
    normalised = (stack - mean[None]) / denominator[None]

    edges = spec.bin_edges()
    out = np.empty((spec.n_fields, *mean.shape), dtype=np.float32)
    out[0] = mean
    out[1] = std

    # Membership is decided by exact comparison against the float32 edges -- the same
    # half-open [e_b, e_{b+1}) semantics np.digitize uses. An arithmetic bin index
    # (floor((z - e0) / width)) was measured to disagree with this on real data, because
    # shifting z by e0 rounds in float32 and GRIB quantisation puts many members on
    # identical z, so a single rounded value moved 6 of 30 members into a neighbouring bin.
    for index in range(spec.n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        out[2 + index] = (
            np.sum((normalised >= lower) & (normalised < upper), axis=0) / n_members
        ).astype(np.float32)

    # A NaN member already makes MEAN and STD NaN, but NaN fails every comparison, so the
    # bins would come out as 0 instead -- a cell that decodes as a shape summing to zero
    # rather than as "unknown". Propagate the undefinedness to every field so the whole
    # cell is consistently unobserved.
    undefined = ~np.isfinite(mean)
    if undefined.any():
        out[:, undefined] = np.float32(np.nan)

    return out


def quantise_field(
    field: npt.NDArray[np.floating],
    scale: float,
) -> npt.NDArray[np.int16]:
    """Quantise one field to int16 fixed point, preserving NaN as the sentinel.

    Args:
        field: Values in the field's own units.
        scale: Value units per stored step; must be positive.

    Returns:
        int16 array with the same shape. NaN becomes :data:`NAN_SENTINEL`.
    """
    if not scale > 0.0:
        raise AggregateError(f"scale must be positive, got {scale}")
    limit = np.iinfo(np.int16).max
    finite = np.isfinite(field)
    scaled = np.where(finite, np.round(field / scale), 0.0)
    # Clipping is reported by the caller's range check rather than silently absorbed; the
    # clamp here only keeps the cast defined.
    codes = np.clip(scaled, -limit, limit).astype(np.int16)
    return np.where(finite, codes, np.int16(NAN_SENTINEL)).astype(np.int16)


def dequantise_field(
    codes: npt.NDArray[np.int16],
    scale: float,
) -> npt.NDArray[np.float32]:
    """Invert :func:`quantise_field`, mapping the sentinel back to NaN."""
    values = codes.astype(np.float32) * np.float32(scale)
    return np.where(codes == NAN_SENTINEL, np.float32(np.nan), values).astype(np.float32)


def check_quantisation_range(
    fields: npt.NDArray[np.floating],
    scales: tuple[float, ...],
) -> tuple[str, ...]:
    """Return the names of fields whose values cannot be represented at their scale.

    A field that clips is not merely imprecise -- it is silently wrong, and it compresses
    *better* because the clipped values collapse to a constant. Callers must treat a
    non-empty result as a hard failure rather than a warning.
    """
    if fields.shape[0] != len(scales):
        raise AggregateError(
            f"{fields.shape[0]} fields but {len(scales)} scales"
        )
    limit = np.iinfo(np.int16).max
    overflowing: list[str] = []
    for index, scale in enumerate(scales):
        field = fields[index]
        finite = np.isfinite(field)
        if not finite.any():
            continue
        if np.any(np.abs(field[finite] / scale) > limit):
            overflowing.append(f"field[{index}]")
    return tuple(overflowing)


def encode_aggregate(
    fields: npt.NDArray[np.floating],
    spec: AggregateSpec,
) -> tuple[npt.NDArray[np.int16], ...]:
    """Quantise every aggregate field, rejecting any that would clip.

    Returns:
        One int16 plane per field, in store order.

    Raises:
        AggregateError: if a field overflows its scale. Clipping would make the result
            silently wrong while compressing better, so it is refused rather than warned.
    """
    scales = spec.field_scales
    overflowing = check_quantisation_range(fields, scales)
    if overflowing:
        raise AggregateError(
            "aggregate field(s) exceed the quantisation range and would clip: "
            + ", ".join(overflowing)
            + "; widen the scale or narrow the field"
        )
    return tuple(
        quantise_field(fields[index], scale) for index, scale in enumerate(scales)
    )


def decode_aggregate(
    planes: tuple[npt.NDArray[np.int16], ...] | list[npt.NDArray[np.int16]],
    spec: AggregateSpec,
) -> npt.NDArray[np.float32]:
    """Reconstruct the aggregate fields from their quantised planes.

    Raises:
        AggregateError: if the plane count disagrees with the spec.
    """
    if len(planes) != spec.n_fields:
        raise AggregateError(
            f"expected {spec.n_fields} planes for {spec.kind} with {spec.n_bins} bins, "
            f"got {len(planes)}"
        )
    scales = spec.field_scales
    return np.stack(
        [dequantise_field(plane, scale) for plane, scale in zip(planes, scales, strict=True)]
    )


def exceedance_from_bins(
    mean: npt.NDArray[np.floating],
    std: npt.NDArray[np.floating],
    bins: npt.NDArray[np.floating],
    threshold: float,
    spec: AggregateSpec,
) -> npt.NDArray[np.float32]:
    """Estimate ``P(x > threshold)`` by reading the CDF of the stored shape.

    This is what the stored bins are *for*: an exceedance probability at a threshold the
    store was never told about. The threshold is expressed in the cell's own normalised
    coordinates and looked up in the interpolated cumulative shape, linearly inside the
    crossing bin.

    Caveat, measured on real GFS/GEFS data: this is accurate through the body of the
    distribution and unreliable for rare thresholds, because the bins span a bounded
    multiple of sigma while the shape's quantisation is coarse where the CDF is steep. For
    a product whose thresholds are known and fixed, storing ``P(x > t)`` directly is both
    cheaper and more accurate.

    Args:
        mean: Decoded MEAN plane.
        std: Decoded STD plane.
        bins: ``(n_bins, lat, lon)`` decoded bin probabilities.
        threshold: Threshold in the variable's units.
        spec: The spec the bins were produced with.

    Returns:
        ``P(x > threshold)`` per cell.
    """
    if bins.shape[0] != spec.n_bins:
        raise AggregateError(
            f"expected {spec.n_bins} bins, got {bins.shape[0]}"
        )
    edges = spec.bin_edges()
    width = float(edges[1] - edges[0])

    coordinates = (threshold - mean) / np.maximum(std, np.float32(1e-12))
    positions = (coordinates - float(edges[0])) / width
    indices = np.clip(np.floor(positions).astype(np.int32), 0, spec.n_bins - 1)
    fractions = np.clip(positions - indices, 0.0, 1.0)

    flat = indices.ravel()
    rows = np.arange(flat.size)
    cumulative = np.cumsum(bins, axis=0).reshape(spec.n_bins, -1)
    below = np.where(flat > 0, cumulative[np.maximum(flat - 1, 0), rows], 0.0)
    inside = bins.reshape(spec.n_bins, -1)[flat, rows]
    cdf = np.clip(below + fractions.ravel() * inside, 0.0, 1.0)
    result: npt.NDArray[np.float32] = (1.0 - cdf).reshape(bins.shape[1:]).astype(np.float32)
    return result
