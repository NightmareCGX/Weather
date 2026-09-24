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

#: Name of the leading field every aggregate carries: how many members were finite at that cell.
#:
#: It exists because collapsing the members destroys the count, and the count is what the API
#: reports as ``member_count`` and what the serving coverage rule is evaluated against. Carrying
#: it as a field rather than in the descriptor is deliberate: the count is **per cell** (one
#: member can be missing at one cell and present at its neighbour), while a descriptor is per
#: container, so a descriptor field could only hold the container-wide total.
#:
#: What it counts, precisely: the members the variable's *own* values are finite for. A
#: supplementary group may cover fewer members than that -- the phase group needs the four flags
#: as well, and a derived variable's rose needs both components -- and a group says so by writing
#: NaN at the cells it cannot describe. So "the count is 30 and this cell's phase support is NaN"
#: is coherent: the count describes the distribution, the group describes itself.
MEMBER_COUNT_FIELD_NAME: Final[str] = "MEMBER_COUNT"

#: Scale of the member-count field. The count is an exact integer that must survive the
#: fixed-point round trip unchanged, so its step is one member.
MEMBER_COUNT_SCALE: Final[float] = 1.0

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

    def statistic_level(self, statistic: str) -> float | None:
        """Probability level that stores ``statistic`` exactly, or ``None`` if interpolated.

        ``mean_std_bins`` stores no levels at all. For ``quantile_function`` the values
        are the spec's own ``levels``, so this reports where a requested quantile lands
        exactly -- which is what tells a reader whether a contract percentile is a stored
        value or an interpolation between two of them.

        Recognised statistic names are ``p<number>`` (e.g. ``p25``) and ``median``; anything
        else returns ``None`` rather than guessing.
        """
        if self.kind != KIND_QUANTILE_FUNCTION:
            return None
        name = statistic.strip().lower()
        if name == "median":
            probability = 0.5
        elif name.startswith("p") and name[1:].replace(".", "", 1).isdigit():
            probability = float(name[1:]) / 100.0
        else:
            return None
        for level in self.levels:
            if abs(level - probability) < 1e-12:
                return level
        return None

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
    *,
    expected_members: int | None = None,
    min_coverage_ratio: float | None = None,
) -> npt.NDArray[np.float32]:
    """Compute the aggregate fields from a member stack.

    Missing members are **skipped**, and a cell is refused only when too few members remain:

    * every cell is computed from the members that are finite there, so one member's gap does
      not discard a cell the rest of the ensemble can describe;
    * a cell whose finite count falls below the platform's coverage floor is entirely NaN,
      because an aggregate over a minority of the ensemble is not the statistic it claims to be.

    That is what the serving paths already do for every variable (the member reader filters to
    finite members and then applies the per-cell rule), so this reproduces their answer instead
    of being stricter than them. The count of finite members is not returned here -- it is
    :func:`finite_member_count`, recorded as its own field -- because a caller that stores these
    fields needs it and a caller that only wants the arithmetic does not.

    Args:
        members: Array of shape ``(n_members, lat, lon)``.
        spec: Field set and quantisation.
        expected_members: The contract's member count the coverage floor is evaluated against.
            Defaults to the number of members supplied, which is only right for an aggregate
            computed from the complete set.
        min_coverage_ratio: Overrides the active configured floor.

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
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    observed = _observed_cells(
        counts,
        expected_members=members.shape[0] if expected_members is None else expected_members,
        min_coverage_ratio=min_coverage_ratio,
    )

    if spec.kind == KIND_QUANTILE_FUNCTION:
        return _compute_quantile_function(stack, spec, observed=observed, counts=counts)
    return _compute_mean_std_bins(stack, spec, observed=observed, counts=counts)


def finite_member_count(members: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """How many members are finite at each cell, as a field ready to store.

    Returned as float32 because every other field is, and the member-count field has to travel
    through the same container and the same fixed-point round trip. Its scale is one member, so
    the value is exact.

    Raises:
        AggregateError: if ``members`` is not a 3-D array, matching :func:`compute_aggregate`.
    """
    if members.ndim != 3:
        raise AggregateError(
            f"members must be (n_members, lat, lon); got shape {members.shape}"
        )
    return np.isfinite(members).sum(axis=0).astype(np.float32)


def _observed_cells(
    counts: npt.NDArray[np.int64],
    *,
    expected_members: int,
    min_coverage_ratio: float | None,
) -> npt.NDArray[np.bool_]:
    """Which cells have enough finite members to be aggregated.

    The floor is the platform's own rule (``domain.coverage.is_cell_statistically_valid``), so
    the aggregate refuses exactly the cells the serving tier would refuse -- not a stricter set.
    """
    from domain.coverage import is_cell_statistically_valid

    return np.asarray(
        is_cell_statistically_valid(
            counts, expected_members, min_coverage_ratio=min_coverage_ratio
        ),
        dtype=np.bool_,
    )


def _cell_moments(
    stack: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Per-cell mean and standard deviation over the finite members.

    Two paths, chosen because they are needed for different inputs rather than for speed:

    * a stack with no non-finite member takes the plain ``mean``/``std``, which are bit-identical
      to the NaN-aware forms *only* there (verified on real shapes, not assumed) and cost 0.005 s
      and 0.038 s against 0.067 s and 0.102 s for a 30 x 721 x 1440 stack -- 12.8x and 2.7x;
    * otherwise the NaN-aware forms, which are the only ones that give the *finite* members'
      moments rather than propagating one gap across the whole cell.

    The common case is the first: a complete member set has no gaps. The check that selects the
    path is one pass, against the two reductions it guards.
    """
    if np.isnan(stack).any():
        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
    return np.mean(stack, axis=0), np.std(stack, axis=0)


def _compute_mean_std_bins(
    stack: npt.NDArray[np.float32],
    spec: AggregateSpec,
    *,
    observed: npt.NDArray[np.bool_],
    counts: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """``MEAN``, ``STD`` and normalised histogram bins over the finite members of each cell."""
    mean, std = _cell_moments(stack)

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
            np.sum((normalised >= lower) & (normalised < upper), axis=0) / counts
        ).astype(np.float32)

    # A cell with too few finite members has no aggregate to report. NaN fails every
    # comparison, so an undefined cell would otherwise decode as a shape summing to zero --
    # indistinguishable from "no member landed in any bin". Every field is set together so the
    # cell is consistently unobserved rather than partially so.
    undefined = ~observed
    if undefined.any():
        out[:, undefined] = np.float32(np.nan)

    return out


def _compute_quantile_function(
    stack: npt.NDArray[np.float32],
    spec: AggregateSpec,
    *,
    observed: npt.NDArray[np.bool_],
    counts: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """One plane per probability level: a sampled inverse CDF in the variable's units.

    The sample quantile uses linear interpolation on the sorted member axis, at position
    ``p * (k - 1)`` for that cell's finite count ``k``. That convention is fixed here and
    duplicated nowhere: a reader that interpolates between two stored levels with a different
    convention, or a writer that samples with a different one, would disagree at the level of a
    single member step. It is also the convention the serving path arrives at once it has
    filtered to finite members, so a cell holding 29 of 30 members reports the quantile of those
    29 rather than of a 30-member sample with a hole in it.

    Non-finite members are excluded per cell. NaN sorts last in numpy, so mapping NaN to +inf
    before sorting puts each cell's finite members in ascending order at the head of its column,
    which is what makes the positions below index only within the finite prefix.

    The per-level interpolation is one vectorised pass rather than a Python loop per level,
    because ``k`` varies by cell and the bracketing indices are therefore arrays. ``np.percentile``
    per level re-partitions the array each time, measured ~50x slower on a 30 x 721 x 1440 stack.
    """
    levels = spec.quantile_levels()

    ordered = np.sort(np.where(np.isfinite(stack), stack, np.float32(np.inf)), axis=0)

    out = np.empty((len(levels), *stack.shape[1:]), dtype=np.float32)
    # A single-member cell has no span to interpolate: every level is that member's value.
    span = np.maximum(counts - 1, 0)
    lat_idx, lon_idx = np.indices(stack.shape[1:])
    for index, level in enumerate(levels):
        position = level * span
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower + 1, span)
        # The two weights are rounded to float32 individually, matching the scalar spelling this
        # replaced: there each weight was a Python float, which NumPy treats as a weak scalar and
        # applies in float32 (NEP 50). Computing `1.0 - fraction` in float32 would instead round
        # the subtrahend first and differ by an ulp on some cells, which is enough to break the
        # bit-for-bit agreement the aggregate pass is verified against.
        fraction = position - lower
        weight_high = fraction.astype(np.float32)
        weight_low = (1.0 - fraction).astype(np.float32)
        # Gather each cell's bracketing pair: lower/upper are per cell, so the take is along
        # the member axis at per-column indices.
        low_vals = ordered[lower, lat_idx, lon_idx]
        high_vals = ordered[upper, lat_idx, lon_idx]
        out[index] = low_vals * weight_low + high_vals * weight_high

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
    *,
    inclusive: bool = False,
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

    ``inclusive`` reads the other side of the threshold and answers a different question:
    ``P(x >= threshold)``, where a value equal to the threshold counts as exceeding it. The
    two differ only on a point mass -- a continuous distribution gives the same answer either
    way -- but a point mass is exactly what a censored or zero-inflated field has, so the
    difference is the whole atom. A partition closed on the left needs this reading: a value on
    a bin edge belongs to the bin *above* it, which is how ``np.histogram`` assigns it and how
    the member histogram counts it. Measured on real GEFS, a frame whose 30 members all sat on
    an interior edge came out one bin low under the default reading and in the right bin under
    this one.

    Args:
        values: ``(n_levels, lat, lon)`` decoded quantile planes, increasing per cell.
        levels: The probability of each plane, strictly increasing.
        threshold: Threshold in the variable's units.
        inclusive: Read ``P(x >= threshold)`` rather than ``P(x > threshold)``.

    Returns:
        The tail probability per cell, clipped to ``[0, 1]``.

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
    # search would not. The strict reading counts the levels strictly below, so a stored value
    # equal to the threshold is not counted as below it and the returned probability includes
    # the atom sitting there.
    below = np.zeros(values.shape[1:], dtype=np.int32)
    for index in range(n_levels):
        if inclusive:
            below += (values[index] < threshold).astype(np.int32)
        else:
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


def quantile_function_moments(
    values: npt.NDArray[np.floating],
    levels: tuple[float, ...] | npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Recover the mean and standard deviation from a stored quantile function.

    For a sampled inverse CDF ``Q(p)``, ``E[X] = integral of Q`` and
    ``Var[X] = integral of (Q - E[X])^2``, both over ``p`` in ``[0, 1]``. Trapezoid on the
    stored levels evaluates those integrals to the resolution the levels provide.

    Why this is acceptable rather than storing MEAN and STD separately: measured on the
    classes the quantile encoding serves (zero-inflated precipitation, heavy-tailed gust,
    bounded cloud cover and humidity), both statistics come back within **0.03-0.27x** the
    ensemble's own sampling noise at 30 members -- indistinguishable from the wobble caused
    by drawing a different 30 members. Storing two more planes would buy precision the
    display cannot show.

    Caveat, measured: the recovery relies on the outermost levels (0.001/0.999) capturing
    the value range. They do for a *sample* quantile function, whose endpoints are the
    sample's extremes. It would not hold if the levels were narrower than the sample range.

    Args:
        values: ``(n_levels, ...)`` decoded quantile planes in increasing value order.
        levels: The probability of each plane, strictly increasing.

    Returns:
        ``(mean, standard_deviation)`` arrays shaped like one plane.
    """
    level_array = np.asarray(levels, dtype=np.float64)
    if values.shape[0] != level_array.size:
        raise AggregateError(
            f"{values.shape[0]} planes but {level_array.size} levels"
        )
    span = float(level_array[-1] - level_array[0])
    if span <= 0.0:
        raise AggregateError(f"levels must span a positive probability range, got {levels}")

    values64 = values.astype(np.float64)
    # asarray around trapezoid: on a 1-D input the stub types the reduction as a scalar,
    # and a plane-shaped result is what every caller and the return type assume.
    mean64 = np.asarray(np.trapezoid(values64, level_array, axis=0) / span)
    centered = values64 - mean64[None]
    variance64 = np.asarray(
        np.trapezoid(centered * centered, level_array, axis=0) / span
    )
    mean: npt.NDArray[np.float32] = mean64.astype(np.float32)
    spread: npt.NDArray[np.float32] = np.sqrt(np.maximum(variance64, 0.0)).astype(np.float32)
    return mean, spread


def quantile_at(
    values: npt.NDArray[np.floating],
    levels: tuple[float, ...] | npt.NDArray[np.floating],
    probability: float,
) -> npt.NDArray[np.float32]:
    """Read any probability off a stored quantile function.

    Exact at a stored level and linear in probability between two, which is the same
    convention the encoder samples with, so a contract percentile that coincides with a
    stored level is reproduced exactly.

    Args:
        values: ``(n_levels, ...)`` decoded quantile planes in increasing value order.
        levels: The probability of each plane, strictly increasing.
        probability: Requested probability in ``[0, 1]``.

    Returns:
        One plane shaped like a single level.

    Raises:
        AggregateError: on a mismatched plane count, non-increasing levels, or a
            probability outside the unit interval.
    """
    level_array = np.asarray(levels, dtype=np.float64)
    if values.shape[0] != level_array.size:
        raise AggregateError(f"{values.shape[0]} planes but {level_array.size} levels")
    if np.any(np.diff(level_array) <= 0.0):
        raise AggregateError(f"levels must be strictly increasing, got {levels}")
    if not 0.0 <= probability <= 1.0:
        raise AggregateError(f"probability must be in [0, 1], got {probability}")

    position = float(np.searchsorted(level_array, probability, side="right") - 1)
    lower = int(min(max(position, 0), level_array.size - 2))
    span = float(level_array[lower + 1] - level_array[lower])
    fraction = 0.0 if span <= 0.0 else (probability - level_array[lower]) / span
    fraction = float(np.clip(fraction, 0.0, 1.0))
    result: npt.NDArray[np.float32] = (
        values[lower] * (1.0 - fraction) + values[lower + 1] * fraction
    ).astype(np.float32)
    return result
