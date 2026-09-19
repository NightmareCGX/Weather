"""Per-variable aggregate encodings for ensemble forecast stores.

An ensemble's members are replaced by a small set of *statistic fields* computed per grid
cell from the whole member set. Which fields depends on the variable class, because the
classes fail differently:

``mean_std_bins`` (near-Gaussian variables: temperature, wind components)
    ``MEAN``, ``STD``, then ``n_bins`` normalised histogram bins over
    ``(x - MEAN) / STD``. The normalisation is not cosmetic: a raw histogram would need a
    global value range, and a variable spanning -60..+50 with a local spread of ~2 would
    then have bins far coarser than the local distribution and cells whose local mean sits
    away from the global mean would fall outside the range entirely. Normalising per cell
    puts the bins on the local distribution, and makes MEAN and STD structurally required
    (the shape carries neither location nor scale).

``quantile_function`` (bounded, censored, zero-inflated and heavy-tailed variables)
    One plane per probability level, each in the variable's own units: a sampled inverse
    CDF. Measured against the alternative on real GEFS data, this is the only encoding that
    answers rare exceedance thresholds, because it allocates resolution in *probability*
    space instead of in sigma space. The bins encoding keeps its 16-32 intervals near the
    body of the distribution, so ``P(x > t)`` past P90 came out 2-15x the ensemble's own
    sampling noise, and raising the bin count did not fix it (the bounded support discards
    tail mass outright); with this encoding the same thresholds land at 0.23-0.54x.
    The level set is body-dense on purpose: the failure it fixes is at the *median* of a
    zero-inflated field, where the CDF is steepest, not in the far tail.

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

#: Number of histogram bins for the ``mean_std_bins`` encoding. Measured on real GEFS data:
#: 16 bins suffice through the body of the distribution, 32 extend coverage to the P90-P99.9
#: range, and 64 buys nothing further while costing another 1.5x. The approved choice for the
#: near-Gaussian class is 32.
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

#: The approved probability-level set for ``quantile_function``: 19 levels, log-spaced at
#: both ends and deliberately dense through the body.
#:
#: Measured on real GEFS data across three cycles. A 17-level set with fewer body levels left
#: a reconstruction error of 0.200 at the finest precipitation threshold, which tracked the
#: level spacing near the median rather than the bin resolution; 26 levels were *worse* than
#: 19 (0.19-0.28), because a 30-member sample quantile function cannot resolve a denser grid
#: and the extra planes only encode sampling noise. 19 is the measured optimum.
DEFAULT_QUANTILE_LEVELS: Final[tuple[float, ...]] = (
    0.001,
    0.005,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.98,
    0.995,
    0.999,
)

#: Fixed-point scale for quantile planes. They carry values in the variable's own units, so
#: the range check is what guards a variable whose span exceeds +-327.67 (a wind gust in km/h
#: or a 3-hour accumulation in mm both come close).
QUANTILE_SCALE: Final[float] = 0.01

#: Sentinel for "no value" in a quantised field. int16 minimum, one step below the smallest
#: representable quantised value, so it cannot collide with a real value.
NAN_SENTINEL: Final[int] = -32768

#: Floor applied to the normalisation denominator. Zero-spread cells exist (a member that
#: is identically zero across the grid) and dividing by zero would make every bin NaN.
_STD_FLOOR: Final[float] = 1e-9


class AggregateError(ValueError):
    """Raised when an aggregate cannot be computed or encoded as requested."""


#: Encoding families. A new kind is a store-format change: every reader keys on this string.
KIND_MEAN_STD_BINS: Final[str] = "mean_std_bins"
KIND_QUANTILE_FUNCTION: Final[str] = "quantile_function"
VALID_KINDS: Final[frozenset[str]] = frozenset(
    {KIND_MEAN_STD_BINS, KIND_QUANTILE_FUNCTION}
)


@dataclass(frozen=True)
class AggregateSpec:
    """Which fields an aggregate carries, and how each is quantised.

    Two encodings are approved, one per variable class. The class is a property of the
    variable, not a local choice, so a store's manifest fixes the mapping and a reader never
    has to guess.

    Attributes:
        kind: Encoding family, one of :data:`VALID_KINDS`.
        n_bins: ``mean_std_bins``: number of normalised histogram bins.
        sigma_range: ``mean_std_bins``: bin support half-width in local standard deviations.
        levels: ``quantile_function``: probability levels, one plane each.

    Raises:
        AggregateError: on an unknown kind, or on parameters that do not apply to the kind.
            Passing a bin parameter to a quantile spec (or vice versa) is rejected rather
            than ignored, so a mis-specified class cannot silently store the wrong fields.
    """

    kind: str = KIND_MEAN_STD_BINS
    n_bins: int = DEFAULT_N_BINS
    sigma_range: float = DEFAULT_SIGMA_RANGE
    levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise AggregateError(
                f"unknown aggregate kind {self.kind!r}; known: {sorted(VALID_KINDS)}"
            )
        if self.kind == KIND_MEAN_STD_BINS:
            if self.n_bins < 1:
                raise AggregateError(f"n_bins must be positive, got {self.n_bins}")
            if not self.sigma_range > 0.0:
                raise AggregateError(
                    f"sigma_range must be positive, got {self.sigma_range}"
                )
            if self.levels != DEFAULT_QUANTILE_LEVELS:
                raise AggregateError(
                    "levels apply to the quantile_function kind only; a mean_std_bins "
                    "spec must leave them at the default"
                )
            return

        if not self.levels:
            raise AggregateError("quantile_function requires at least one level")
        previous = -1.0
        for level in self.levels:
            if not 0.0 < level < 1.0:
                raise AggregateError(
                    f"quantile levels must lie strictly between 0 and 1, got {level}"
                )
            if level <= previous:
                raise AggregateError(
                    f"quantile levels must be strictly increasing, got {self.levels}"
                )
            previous = level

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in storage order.

        ``mean_std_bins`` stores ``MEAN``, ``STD`` then the bins; ``quantile_function``
        stores one level per field, named by the level so a reader can interpret a plane
        without consulting the spec.
        """
        if self.kind == KIND_MEAN_STD_BINS:
            return ("MEAN", "STD", *(f"BIN{i:02d}" for i in range(self.n_bins)))
        return tuple(f"Q{level:.3f}".replace("0.", "") for level in self.levels)

    @property
    def field_scales(self) -> tuple[float, ...]:
        """Fixed-point scale per field, in :attr:`field_names` order."""
        if self.kind == KIND_MEAN_STD_BINS:
            return (MEAN_SCALE, STD_SCALE, *(BIN_SCALE,) * self.n_bins)
        return (QUANTILE_SCALE,) * len(self.levels)

    @property
    def n_fields(self) -> int:
        """Total field count, i.e. the number of stored lat/lon planes."""
        if self.kind == KIND_MEAN_STD_BINS:
            return 2 + self.n_bins
        return len(self.levels)

    def bin_edges(self) -> npt.NDArray[np.float32]:
        """Bin edges in normalised units, as float32.

        The dtype matters: the reference computes bin membership by comparing float32
        normalised values against these edges, so returning float64 would change which
        side of an edge a value on the boundary lands.

        Raises:
            AggregateError: for a kind that has no bins.
        """
        if self.kind != KIND_MEAN_STD_BINS:
            raise AggregateError(f"{self.kind!r} has no bin edges")
        return np.linspace(
            -self.sigma_range, self.sigma_range, self.n_bins + 1
        ).astype(np.float32)

    def quantile_levels(self) -> tuple[float, ...]:
        """Probability levels, in storage order.

        Raises:
            AggregateError: for a kind that has no levels.
        """
        if self.kind != KIND_QUANTILE_FUNCTION:
            raise AggregateError(f"{self.kind!r} has no quantile levels")
        return self.levels


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

    if spec.kind == KIND_QUANTILE_FUNCTION:
        return _compute_quantile_function(stack, spec)
    return _compute_mean_std_bins(stack, spec)


def _compute_mean_std_bins(
    stack: npt.NDArray[np.float32],
    spec: AggregateSpec,
) -> npt.NDArray[np.float32]:
    """``MEAN``, ``STD`` and normalised histogram bins."""
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


def _compute_quantile_function(
    stack: npt.NDArray[np.float32],
    spec: AggregateSpec,
) -> npt.NDArray[np.float32]:
    """One plane per probability level: a sampled inverse CDF in the variable's units.

    The sample quantile uses linear interpolation on the sorted member axis, at position
    ``p * (n - 1)``. That convention is fixed here and duplicated nowhere: a reader that
    interpolates between two stored levels with a different convention, or a writer that
    samples with a different one, would disagree at the level of a single member step.

    Members are sorted once for all levels. ``np.percentile`` per level re-partitions the
    array each time, measured ~50x slower on a 30 x 721 x 1440 stack for identical values.
    """
    levels = spec.quantile_levels()
    n_members = stack.shape[0]

    # NaN sorts last in numpy, so an incomplete cell would report finite quantiles drawn
    # from the missing members at the tail and NaN at the head. Sorting with NaN mapped to
    # +inf and then restoring the undefinedness per cell keeps every level for an
    # incomplete cell NaN, which is the intended semantics.
    observed = np.isfinite(stack).all(axis=0)
    ordered = np.sort(np.where(np.isfinite(stack), stack, np.float32(np.inf)), axis=0)

    out = np.empty((len(levels), *stack.shape[1:]), dtype=np.float32)
    last = n_members - 1
    for index, level in enumerate(levels):
        position = level * last
        lower = int(np.floor(position))
        upper = min(lower + 1, last)
        fraction = position - lower
        if upper == lower:
            out[index] = ordered[lower]
            continue
        # Interpolate with the weight as a Python float, which NumPy treats as a weak scalar
        # and applies in float32 (NEP 50). Promoting to float64 first and rounding once is
        # the "more accurate" spelling and differs from the reference by one float32 ulp
        # (~2e-6) on the wider levels -- harmless numerically, but it would break the
        # bit-for-bit equivalence the aggregate pass is verified against.
        out[index] = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    if not observed.all():
        out[:, ~observed] = np.float32(np.nan)
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


def exceedance_from_quantiles(
    values: npt.NDArray[np.floating],
    levels: tuple[float, ...] | npt.NDArray[np.floating],
    threshold: float,
) -> npt.NDArray[np.float32]:
    """Estimate ``P(x > threshold)`` by inverting the stored quantile function.

    The inverse CDF is read by locating the threshold between two stored levels and
    interpolating the probability, then subtracting from one. This is the encoding's
    purpose: a product can ask about thresholds the store was never told about.

    Accuracy, measured against the ensemble's own sampling noise on real GEFS data: 0.2-0.7x
    through the body of the distribution and 0.3-0.6x at rare thresholds. Comparable to
    storing the exceedance probability directly for a *known* threshold, while also correct
    for an arbitrary one. Contrast the bin encoding, whose bounded support cannot represent
    the tail at all.

    Args:
        values: ``(n_levels, lat, lon)`` decoded quantile planes, increasing per cell.
        levels: The probability of each plane, strictly increasing.
        threshold: Threshold in the variable's units.

    Returns:
        ``P(x > threshold)`` per cell, clipped to ``[0, 1]``.

    Raises:
        AggregateError: if fewer than two planes are supplied, if the plane count and the
            level count disagree, or if the levels are not strictly increasing. A single
            level cannot be inverted, and a non-increasing set means the planes came from
            different specs.
    """
    level_array = np.asarray(levels, dtype=np.float64)
    if values.ndim != 3 or level_array.size < 2:
        raise AggregateError(
            f"quantile inversion needs at least two planes, got shape {values.shape} "
            f"and {level_array.size} levels"
        )
    if values.shape[0] != level_array.size:
        raise AggregateError(f"{values.shape[0]} planes but {level_array.size} levels")
    if np.any(np.diff(level_array) <= 0.0):
        raise AggregateError(f"levels must be strictly increasing, got {levels}")

    n_levels = level_array.size
    # Counting the levels at or below the threshold is cheaper than a search, and there are
    # only a dozen or so levels. It also keeps the axis length fixed, which a per-cell
    # search would not.
    below = np.zeros(values.shape[1:], dtype=np.int32)
    for index in range(n_levels):
        below += (values[index] <= threshold).astype(np.int32)

    # The CDF is exact at a stored level and interpolated between two, so the segment is
    # ``below - 1`` clamped into range.
    segment = np.clip(below - 1, 0, n_levels - 2)
    lower_values = np.take_along_axis(values, segment[None], axis=0)[0]
    upper_values = np.take_along_axis(values, (segment + 1)[None], axis=0)[0]

    span = upper_values - lower_values
    # A degenerate segment (two levels with the same value) says nothing about where in the
    # interval the threshold falls; the lower level is the only defensible answer.
    informative = np.abs(span) > 0.0
    fraction = np.where(
        informative,
        (np.float32(threshold) - lower_values) / np.where(informative, span, np.float32(1.0)),
        np.float32(0.0),
    )
    fraction = np.clip(fraction, 0.0, 1.0)

    lower_probability = level_array[segment]
    upper_probability = level_array[segment + 1]
    cdf = lower_probability + fraction.astype(np.float64) * (upper_probability - lower_probability)
    result: npt.NDArray[np.float32] = np.clip(1.0 - cdf, 0.0, 1.0).astype(np.float32)
    return result
