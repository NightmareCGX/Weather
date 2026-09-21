"""Serving ensemble statistics from aggregate shards when a store has them.

The read path's entry point for aggregates. It sits beside ``_gated_member_values`` rather than
replacing it, because the two are alternatives chosen per request:

* a store with a readable aggregate for the resolved ``(variable, lead)`` is served from it --
  four range GETs instead of thirty member reads, and no per-member interpolation;
* anything else falls through to the member path, which remains the reader of record.

What this module must get right
------------------------------
**Fallback, never a partial answer.** Every failure mode here -- no aggregate, an unreadable
container, a statistic the encoding cannot express -- falls through to the members. The caller
cannot distinguish "served from an aggregate" by shape alone, so a silent partial result would
be indistinguishable from a complete one.

**The encoding decides what is exact.** The bin encoding stores MEAN and STD directly, so those
are exact and its percentiles are read off the accumulated bins. The quantile encoding stores
no moments at all: P10/P50/P90 are stored levels and therefore exact, P25/P75 interpolate in
probability, and mean/spread come from integrating the levels (measured at 0.03-0.27x the
ensemble's own sampling noise, so indistinguishable at 30 members). Each path reports which of
its values are exact so a caller can tell how much of the answer is reconstructed.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from domain.aggregate import (
    KIND_MEAN_STD_BINS,
    KIND_QUANTILE_FUNCTION,
    AggregateError,
    AggregateSpec,
    exceedance_from_bins,
    exceedance_from_quantiles,
    quantile_at,
    quantile_function_moments,
)
from domain.field_layout import (
    FieldLayoutError,
    aggregate_fields_for,
    group_field_names,
)
from domain.variable_class import VariableClassError, spec_for

from api.core.aggregate_reader import AggregateGeometry, AggregateShardReader

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: How the rose group addresses its parts, mirroring ``domain.field_layout``'s declaration: 64
#: sector-bucket cells, then four consensus scalars, then the nine bucket edges. Named here rather
#: than imported because the *reader* must not depend on the writer's producer module; the counts
#: are asserted against the layout by a test, so a layout change cannot silently shift them.
ROSE_BUCKETS: int = 8
ROSE_CELL_COUNT: int = 64
ROSE_SCALAR_OFFSET: int = ROSE_CELL_COUNT
ROSE_SCALAR_COUNT: int = 4
ROSE_EDGE_COUNT: int = ROSE_BUCKETS + 1

#: Statistics the platform serves per variable (``EnsembleStatistics``). A caller passing a
#: different name gets a KeyError from the dataclass rather than a silent zero.
#:
#: ``p0.1`` and ``p99.9`` are the **sample extremes' stand-ins**, not a new product: the chart
#: shows a low/high pair beside the percentiles, and the raw members that pair used to be read
#: from are what a container replaces. The quantile encoding stores both levels exactly (0.001 and
#: 0.999 are in its approved level set), so the pair is as accurate as any other stored level; the
#: bin encoding can only read them off its bounded shape, which is the same caveat its other
#: percentiles carry. Reported under percentiles rather than as "min"/"max" because the stored
#: answer is a *quantile of the distribution*, not the sample's extremes -- calling it min would
#: claim a value the store never recorded.
STATISTIC_NAMES: tuple[str, ...] = (
    "mean",
    "median",
    "spread",
    "p0.1",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p99.9",
)

#: Statistics that coincide with a stored quantile level for the quantile encoding, and are
#: therefore reproduced exactly rather than reconstructed.
_EXACT_QUANTILE_STATISTICS: frozenset[str] = frozenset(
    {"median", "p0.1", "p10", "p50", "p90", "p99.9"}
)


@dataclass(frozen=True)
class AggregatePointStatistics:
    """One variable's statistics at a point, served from an aggregate shard.

    Attributes:
        values: Statistic name to value, for the names the encoding can express.
        spec: The spec the aggregate was written with. ``None`` for a variable that stores a
            fraction instead of a distribution -- a 0/1 flag, whose per-cell fraction is its whole
            representation (``domain.variable_class.CLASS_FLAG``).
        exact: Statistics reproduced bit-exactly rather than reconstructed.
        geometry: The container geometry the values were read from.
        member_count: Ensemble members the aggregate was computed from, when the point was
            observed. ``None`` when the container does not carry a count, so a caller falls
            back to the member path rather than reporting an invented number.
    """

    values: dict[str, float]
    spec: AggregateSpec | None
    exact: frozenset[str]
    geometry: AggregateGeometry
    member_count: int | None = None

    def as_ensemble_statistics_kwargs(self) -> dict[str, float | None]:
        """Return the platform's statistics shape, with unexpressible names as ``None``.

        A missing statistic is reported as absent rather than as a fabricated number: the API
        schema already models these as optional, and a caller reading ``None`` can fall back to
        the member path for that field.
        """
        return {name: self.values.get(name) for name in STATISTIC_NAMES}


def _bilinear_from_corners(
    corners: list[npt.NDArray[np.float32]],
    t_row: float,
    t_col: float,
) -> npt.NDArray[np.float32]:
    """Blend four corner stacks, each ``(n_fields, lat, lon)``, at fractional offsets.

    The same convex combination the member reader applies per member, applied per *field*
    instead. Every aggregate value at a point is a statistic of the interpolated member values,
    so applying the interpolation to the stored statistic is the approximation this whole
    encoding rests on -- and the reason the acceptance yardstick is the ensemble's own sampling
    noise rather than a point-value tolerance.
    """
    lower = corners[0] + (corners[1] - corners[0]) * np.float32(t_col)
    upper = corners[2] + (corners[3] - corners[2]) * np.float32(t_col)
    return (lower + (upper - lower) * np.float32(t_row)).astype(np.float32)


def _corner_values(
    stack: npt.NDArray[np.float32],
    row_in_chunk: int,
    col_in_chunk: int,
) -> npt.NDArray[np.float32]:
    """Extract one cell's field vector from a location stack."""
    return stack[:, row_in_chunk, col_in_chunk]


def _statistics_from_bins(
    fields_at_point: npt.NDArray[np.float32],
    spec: AggregateSpec,
) -> tuple[dict[str, float], frozenset[str]]:
    """Read the platform's statistics from a bin-encoded aggregate.

    MEAN and STD are stored planes, so they are exact. The percentiles are read off the
    accumulated normalised bins, which is approximate: the bins bound the shape's resolution,
    and a value outside the +-sigma support is in no bin at all.
    """
    mean = float(fields_at_point[0])
    spread = float(fields_at_point[1])
    bins = fields_at_point[2:]

    values: dict[str, float] = {"mean": mean, "spread": spread}
    if not math.isfinite(mean) or not math.isfinite(spread):
        return values, frozenset({"mean", "spread"})

    edges = spec.bin_edges()
    width = float(edges[1] - edges[0])
    cumulative = np.cumsum(bins.astype(np.float64))
    total = float(cumulative[-1])

    def quantile(probability: float) -> float | None:
        if total <= 0.0:
            return None
        target = probability * total
        index = int(np.searchsorted(cumulative, target))
        index = min(index, spec.n_bins - 1)
        below = float(cumulative[index - 1]) if index > 0 else 0.0
        inside = float(bins[index])
        if inside <= 0.0:
            return float(mean + edges[index] * std_scale)
        fraction = (target - below) / inside
        normalised = float(edges[index]) + fraction * width
        return float(mean + normalised * std_scale)

    std_scale = spread if spread > 0.0 else 1.0
    for name, probability in (
        ("median", 0.50),
        ("p0.1", 0.001),
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p99.9", 0.999),
    ):
        value = quantile(probability)
        if value is not None:
            values[name] = value
    return values, frozenset({"mean", "spread"})


def _statistics_from_quantiles(
    levels_at_point: npt.NDArray[np.float32],
    spec: AggregateSpec,
) -> tuple[dict[str, float], frozenset[str]]:
    """Read the platform's statistics from a quantile-encoded aggregate.

    P10, P50, the median and P90 coincide with stored levels and are exact. P25 and P75 sit
    between stored levels and interpolate in probability; mean and spread come from integrating
    the levels. Both were measured against the ensemble's own sampling noise (0.03-0.28x), so
    neither is reported as exact but neither is distinguishable from a member-derived answer.
    """
    if not np.isfinite(levels_at_point).any():
        return {}, frozenset()

    levels = spec.quantile_levels()
    values: dict[str, float] = {}
    exact: set[str] = set()

    for name in ("p0.1", "p10", "p25", "median", "p50", "p75", "p90", "p99.9"):
        probability = (
            0.50
            if name == "median"
            else float(name.replace("p", "", 1)) / 100.0
        )
        try:
            values[name] = float(np.asarray(quantile_at(levels_at_point, levels, probability)))
        except AggregateError:  # pragma: no cover - the spec and the planes agree by construction
            continue
        if name in _EXACT_QUANTILE_STATISTICS and spec.statistic_level(name) is not None:
            exact.add(name)

    try:
        # The domain helpers speak in planes; a single point is the degenerate 1x1 plane.
        mean, spread = quantile_function_moments(
            levels_at_point.reshape(len(levels), 1, 1), levels
        )
        values["mean"] = float(mean[0, 0])
        values["spread"] = float(spread[0, 0])
    except AggregateError:  # pragma: no cover - guarded by the caller's spec lookup
        pass
    return values, frozenset(exact)


def fraction_statistics(
    fields_at_point: npt.NDArray[np.float32],
    *,
    layout: Any,
    member_count: int,
) -> tuple[dict[str, float], frozenset[str]] | None:
    """The statistics a 0/1 flag's stored fraction implies at one cell.

    A flag's container holds one field -- the share of members that were set -- because a shape
    over two values carries nothing (``domain.variable_class.CLASS_FLAG``). That single number is
    still enough for every statistic the response reports, because the sample behind it is 0/1 and
    the statistics of such a sample are closed forms. With ``k = round(f * n)`` of ``n`` members
    set:

    * mean is ``k/n``, which *is* the stored fraction (to the fraction's own 0.001 step);
    * spread is ``sqrt(f(1-f))``, the population standard deviation of a Bernoulli sample;
    * a percentile is the same ``linear`` interpolation over a sorted two-valued sample that
      ``numpy.percentile`` performs -- position ``p = q(n-1)``, and the sample is ``n-k`` zeros
      followed by ``k`` ones, so the answer is 0 below the run, 1 inside it, and the boundary
      blend only in the single position that straddles it.

    Every one is therefore **exact**, which is what makes a flag's container a replacement for its
    members rather than an approximation of them: the fraction is the only thing the sample had.
    The formulas are verified exhaustively against the member path for every ``k`` at 30 members.

    Args:
        fields_at_point: The cell's field vector.
        layout: The variable's field layout, for the fraction's index.
        member_count: Members that were finite at this cell, from the container's count field.

    Returns:
        ``(values, exact)``, or ``None`` for an unobserved cell -- a NaN fraction means the store
        refused the cell, which is not the same statement as a fraction of zero.
    """
    fraction = float(fields_at_point[layout.group_slice("fraction").start])
    if not math.isfinite(fraction) or member_count <= 0:
        return None
    fraction = min(max(fraction, 0.0), 1.0)
    ones = min(member_count, max(0, int(round(fraction * member_count))))
    values: dict[str, float] = {
        "mean": ones / member_count,
        "spread": math.sqrt(_bernoulli_fraction(ones, member_count) * (1.0 - _bernoulli_fraction(ones, member_count))),
        "median": 0.0 if ones * 2 < member_count else (1.0 if ones * 2 > member_count else 0.5),
    }
    for name, probability in (
        ("p0.1", 0.001),
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p99.9", 0.999),
    ):
        values[name] = _two_valued_percentile(ones, member_count, probability)
    return values, frozenset(values)


def _bernoulli_fraction(ones: int, member_count: int) -> float:
    """The mean of a 0/1 sample, as a float in [0, 1]."""
    return ones / member_count


def _two_valued_percentile(ones: int, member_count: int, probability: float) -> float:
    """A percentile of a sample of ``member_count`` values, ``ones`` of which are 1.

    ``numpy.percentile``'s ``linear`` method, evaluated without materialising the sample: position
    ``p = probability * (n - 1)``, and the sorted sample is ``n - ones`` zeros then ``ones`` ones,
    so the interpolation is 0 below the run of ones, 1 inside it, and a blend only in the one
    position that straddles the boundary.
    """
    if member_count <= 0:
        return 0.0
    position = probability * (member_count - 1)
    lower = math.floor(position)
    started = member_count - ones
    if lower >= started:
        return 1.0
    if lower + 1 <= started - 1:
        return 0.0
    if lower == started - 1:
        return position - lower
    return 1.0


def statistics_from_aggregate(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    chunk_row: int,
    chunk_col: int,
    row_in_chunk: int,
    col_in_chunk: int,
    generation: str | None = None,
    reader: AggregateShardReader | None = None,
) -> AggregatePointStatistics | None:
    """Serve one variable's ensemble statistics at a point from its aggregate shard.

    ``None`` whenever an aggregate cannot answer -- no shard, an unreadable container, an
    unclassified variable, an off-grid location. The caller then uses the member path, so a
    ``None`` here is a missing optimisation and never a partial answer.

    A single location is read rather than a 2x2 window, because the caller has already resolved
    the point to a cell; interpolating between cells would need four reads for a correction
    smaller than the encoding's own reconstruction error.

    Args:
        variable: Variable code.
        store_path: The cycle store.
        lead_time_hours: Forecast lead.
        chunk_row: Chunk-row of the point's cell.
        chunk_col: Chunk-column of the point's cell.
        row_in_chunk: Row of the cell inside its chunk.
        col_in_chunk: Column of the cell inside its chunk.
        generation: Committed-manifest generation, for cache correctness across PATCHes.
        reader: Optional reader to reuse (the caller may hold one per store).

    Returns:
        The statistics and their exactness, or ``None`` to fall through to the members.
    """
    try:
        layout = aggregate_fields_for(variable)
    except FieldLayoutError:
        return None
    try:
        spec = spec_for(variable)
    except VariableClassError:
        # Not an error: a 0/1 flag has no distribution spec, and its fraction answers every
        # statistic the response reports (see :func:`fraction_statistics`).
        spec = None

    active_reader = reader if reader is not None else AggregateShardReader(store_path)
    stack = active_reader.read_location(
        variable,
        lead_time_hours=lead_time_hours,
        chunk_row=chunk_row,
        chunk_col=chunk_col,
        generation=generation,
    )
    if stack is None:
        return None
    # Field 0 is the per-cell member count; the distribution follows it, then the variable's
    # supplementary groups. The count has to match the whole vector, not just the distribution,
    # because a container written for a different layout would decode as plausible nonsense.
    if stack.shape[0] != layout.n_fields:
        logger.warning(
            "aggregate for %s at lead %d holds %d fields, expected %d; using members",
            variable,
            lead_time_hours,
            stack.shape[0],
            layout.n_fields,
        )
        return None

    geometry = active_reader.open(variable, lead_time_hours, generation=generation)
    if geometry is None:  # pragma: no cover - read_location already required it
        return None

    point = _corner_values(stack, int(row_in_chunk), int(col_in_chunk))
    member_count = int(round(float(point[0]))) if math.isfinite(float(point[0])) else None
    if spec is None:
        if member_count is None:
            return None
        fraction = fraction_statistics(point, layout=layout, member_count=member_count)
        if fraction is None:
            return None
        values, exact = fraction
    else:
        # The distribution is the fields between the count and the groups, located by the layout
        # rather than by counting, so a group added to it cannot shift what this reader is handed.
        encoding_fields = point[layout.distribution_slice]
        try:
            if spec.kind == KIND_MEAN_STD_BINS:
                values, exact = _statistics_from_bins(encoding_fields, spec)
            elif spec.kind == KIND_QUANTILE_FUNCTION:
                values, exact = _statistics_from_quantiles(encoding_fields, spec)
            else:  # pragma: no cover - the spec's own validation rejects other kinds
                return None
        except AggregateError as exc:
            logger.warning(
                "aggregate for %s at lead %d could not be interpreted (%s); using members",
                variable,
                lead_time_hours,
                exc,
            )
            return None

    finite = {name: value for name, value in values.items() if math.isfinite(value)}
    if "mean" not in finite:
        # Without a mean the answer would be a partial statistic set; members can do better.
        return None
    return AggregatePointStatistics(
        values=finite,
        spec=spec,
        exact=exact,
        geometry=geometry,
        member_count=member_count,
    )


@dataclass(frozen=True)
class AggregatePointProbability:
    """An exceedance probability at a point, served from an aggregate shard.

    Attributes:
        probability: ``P(x > threshold)`` (or ``P(x < threshold)`` for ``lt``).
        member_count: Members the aggregate was computed from, when the point was observed.
            Carried here rather than fetched separately so the confidence interval the API
            reports costs no extra read: the count rides in the descriptor the geometry read
            already fetched.
    """

    probability: float
    member_count: int | None


def exceedance_probability(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    threshold: float,
    operator: str,
    chunk_row: int,
    chunk_col: int,
    row_in_chunk: int,
    col_in_chunk: int,
    generation: str | None = None,
    reader: AggregateShardReader | None = None,
) -> AggregatePointProbability | None:
    """Serve a strict exceedance probability from an aggregate shard.

    This is what the encoding exists for: the store was never told the threshold, and the
    quantile encoding answers it by inverting the stored levels. The bin encoding can answer
    the same question only through its CDF, so it is supported too, with the tail caveat its
    own docstring records.

    **Only the strict operators are served: ``gt`` and ``lt``.** The domain's ``gte``/``lte``
    count members *at* the threshold, and a stored quantile function cannot resolve that.
    Measured on a zero-inflated field at the natural threshold of 0, where half the members are
    exactly zero: the member path reports P(x >= 0) = 1.0 while P(x > 0) = 0.47, a 0.53
    difference that is the point mass itself. The aggregate's continuous interpolation reports
    0.50, which is neither. So the inclusive operators return ``None`` and the caller uses the
    members, where the distinction is well defined.

    ``None`` whenever an aggregate cannot answer, so the caller falls back to the members.

    Args:
        variable: Variable code.
        store_path: The cycle store.
        lead_time_hours: Forecast lead.
        threshold: Threshold in the variable's canonical units.
        operator: ``gt`` or ``lt``. ``gte``/``lte`` need the member sample (see above) and
            ``between`` needs two thresholds; all three are the caller's to handle.
        chunk_row: Chunk-row of the point's cell.
        chunk_col: Chunk-column of the point's cell.
        row_in_chunk: Row of the cell inside its chunk.
        col_in_chunk: Column of the cell inside its chunk.
        generation: Committed-manifest generation.
        reader: Optional reader to reuse.
    """
    if operator not in ("gt", "lt"):
        return None
    try:
        layout = aggregate_fields_for(variable)
        spec = spec_for(variable)
    except (FieldLayoutError, VariableClassError):
        # A threshold query needs a distribution: a flag's fraction is a probability, not a set of
        # values to compare a threshold against, so `P(X > t)` cannot be read off it.
        return None

    active_reader = reader if reader is not None else AggregateShardReader(store_path)
    stack = active_reader.read_location(
        variable,
        lead_time_hours=lead_time_hours,
        chunk_row=chunk_row,
        chunk_col=chunk_col,
        generation=generation,
    )
    if stack is None or stack.shape[0] != layout.n_fields:
        return None

    # Field 0 is the per-cell member count; the distribution follows, then the groups.
    full_point = _corner_values(stack, int(row_in_chunk), int(col_in_chunk))
    member_count = (
        int(round(float(full_point[0]))) if math.isfinite(float(full_point[0])) else None
    )
    point = full_point[layout.distribution_slice]
    try:
        if spec.kind == KIND_QUANTILE_FUNCTION:
            probability_above = float(
                np.asarray(
                    exceedance_from_quantiles(
                        point.reshape(spec.n_fields, 1, 1),
                        spec.quantile_levels(),
                        threshold,
                    )
                )[0, 0]
            )
        elif spec.kind == KIND_MEAN_STD_BINS:
            mean = point[0].reshape(1)
            spread = point[1].reshape(1)
            bins = point[2:].reshape(spec.n_bins, 1)
            if not (math.isfinite(float(mean[0])) and math.isfinite(float(spread[0]))):
                return None
            probability_above = float(
                np.asarray(exceedance_from_bins(mean, spread, bins, threshold, spec))[0]
            )
        else:  # pragma: no cover - the spec's own validation rejects other kinds
            return None
    except AggregateError as exc:
        logger.warning(
            "aggregate threshold lookup for %s at lead %d failed (%s); using members",
            variable,
            lead_time_hours,
            exc,
        )
        return None

    if not math.isfinite(probability_above):
        return None
    # Only the strict operators reach here; the inclusive ones are refused above because a
    # stored quantile function cannot represent the point mass at the threshold.
    probability = probability_above if operator == "gt" else 1.0 - probability_above
    return AggregatePointProbability(probability=probability, member_count=member_count)


def gated_statistics_from_aggregate(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    generation: str | None = None,
) -> AggregatePointStatistics | None:
    """Serve statistics from an aggregate shard for a geographic point, under the reader gate.

    The point's cell is derived from the store's own coordinate axes, read inside the same
    gate the member path uses. That is deliberate: the aggregate descriptor carries extents
    but not axis *values*, so deriving the cell from the descriptor alone would mean assuming
    the platform's canonical grid -- exactly the kind of hidden assumption the descriptor
    exists to remove. Coordinates plus the gate are already available on this path, so they are
    used instead of assumed.

    ``None`` whenever an aggregate cannot answer, so the caller uses the members.
    """
    from api.core.manifest_reader import manifest_generation
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.services.point_forecast import _derive_grid

    resolved_generation = generation
    if resolved_generation is None:
        resolved_generation = manifest_generation(store_path)

    def select(dataset: Any) -> AggregatePointStatistics | None:
        if variable not in dataset.data_vars:
            return None
        grid, lat_descending, lon_descending = _derive_grid(dataset)
        if grid.rows < 1 or grid.cols < 1:
            return None
        try:
            row_f, col_f = grid.row_col_from_coordinates(latitude, longitude)
        except Exception:  # noqa: BLE001 - an off-grid point falls back to the members
            return None
        row = min(int(math.floor(row_f)), grid.rows - 1)
        col = min(int(math.floor(col_f)), grid.cols - 1)

        def stored(value: int, size: int, descending: bool) -> int:
            return (size - 1 - value) if descending else value

        lat_idx = stored(row, int(dataset.sizes["latitude"]), lat_descending)
        lon_idx = stored(col, int(dataset.sizes["longitude"]), lon_descending)
        return statistics_from_aggregate_at_cell(
            variable,
            store_path=store_path,
            lead_time_hours=lead_time_hours,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            generation=resolved_generation,
        )

    return gated_read_dataset_with_selector(store_path, select)


def statistics_from_aggregate_at_cell(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    lat_idx: int,
    lon_idx: int,
    generation: str | None = None,
    reader: AggregateShardReader | None = None,
) -> AggregatePointStatistics | None:
    """Statistics from an aggregate shard addressed by absolute grid indices.

    Splits a grid index into its chunk and in-chunk cell, which requires knowing the chunk
    extent -- taken from the container's own descriptor, never assumed.

    ``None`` whenever an aggregate cannot answer, or the index falls outside the grid the
    container describes.
    """
    active_reader = reader if reader is not None else AggregateShardReader(store_path)
    geometry = active_reader.open(variable, lead_time_hours, generation=generation)
    if geometry is None:
        return None
    if not (0 <= lat_idx < geometry.grid_lat and 0 <= lon_idx < geometry.grid_lon):
        return None
    chunk_row, row_in_chunk = divmod(lat_idx, geometry.chunk_lat)
    chunk_col, col_in_chunk = divmod(lon_idx, geometry.chunk_lon)
    return statistics_from_aggregate(
        variable,
        store_path=store_path,
        lead_time_hours=lead_time_hours,
        chunk_row=chunk_row,
        chunk_col=chunk_col,
        row_in_chunk=row_in_chunk,
        col_in_chunk=col_in_chunk,
        generation=generation,
        reader=active_reader,
    )


def gated_exceedance_from_aggregate(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    threshold: float,
    operator: str,
    generation: str | None = None,
) -> AggregatePointProbability | None:
    """Serve an exceedance probability from an aggregate shard, under the reader gate.

    The same gate and coordinate derivation as :func:`gated_statistics_from_aggregate`, for the
    probability endpoint. ``None`` whenever an aggregate cannot answer.
    """
    from api.core.manifest_reader import manifest_generation
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.services.point_forecast import _derive_grid

    resolved_generation = generation
    if resolved_generation is None:
        resolved_generation = manifest_generation(store_path)

    def select(dataset: Any) -> AggregatePointProbability | None:
        if variable not in dataset.data_vars:
            return None
        grid, lat_descending, lon_descending = _derive_grid(dataset)
        try:
            row_f, col_f = grid.row_col_from_coordinates(latitude, longitude)
        except Exception:  # noqa: BLE001 - an off-grid point falls back to the members
            return None
        row = min(int(math.floor(row_f)), grid.rows - 1)
        col = min(int(math.floor(col_f)), grid.cols - 1)

        def stored(value: int, size: int, descending: bool) -> int:
            return (size - 1 - value) if descending else value

        lat_idx = stored(row, int(dataset.sizes["latitude"]), lat_descending)
        lon_idx = stored(col, int(dataset.sizes["longitude"]), lon_descending)

        reader = AggregateShardReader(store_path)
        geometry = reader.open(variable, lead_time_hours, generation=resolved_generation)
        if geometry is None:
            return None
        if not (0 <= lat_idx < geometry.grid_lat and 0 <= lon_idx < geometry.grid_lon):
            return None
        chunk_row, row_in_chunk = divmod(lat_idx, geometry.chunk_lat)
        chunk_col, col_in_chunk = divmod(lon_idx, geometry.chunk_lon)
        return exceedance_probability(
            variable,
            store_path=store_path,
            lead_time_hours=lead_time_hours,
            threshold=threshold,
            operator=operator,
            chunk_row=chunk_row,
            chunk_col=chunk_col,
            row_in_chunk=row_in_chunk,
            col_in_chunk=col_in_chunk,
            generation=resolved_generation,
            reader=reader,
        )

    return gated_read_dataset_with_selector(store_path, select)


def cloud_censoring_from_aggregate(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    chunk_row: int,
    chunk_col: int,
    row_in_chunk: int,
    col_in_chunk: int,
    generation: str | None = None,
    reader: AggregateShardReader | None = None,
) -> dict[str, Any] | None:
    """The cloud variables' per-member censoring, read from their container's own fields.

    What this replaces, and why it is not the same as the callers' member path:

    * ``cloud_ceiling`` splits its members into a finite height and an "unlimited" sentinel and
      summarises each separately. The **counts** are the container's censoring fields; the
      conditional statistics are the ``conditional`` group, computed by the writer over exactly
      the finite members. The member path's own ``min_finite``/``min_valid`` thresholds are its
      rules, not the container's, so the answer says what was stored and lets the caller decide.
    * ``cloud_cover_3h`` summarises only the members inside ``[0, 100]``, and its counts have no
      unlimited class -- the container stores that count as zero rather than leaving it out.

    ``None`` whenever an aggregate cannot answer -- no shard, an unreadable container, an
    off-grid location -- so the caller uses the members, which remain the reader of record.

    Returns:
        ``valid_member_count``, ``finite_member_count``, ``unlimited_member_count`` and
        ``unlimited_probability``, plus the seven conditional statistics under ``conditional_*``
        (``None`` where the container has no conditional value for that cell).
    """
    try:
        layout = aggregate_fields_for(variable)
    except FieldLayoutError:
        return None

    active_reader = reader if reader is not None else AggregateShardReader(store_path)
    stack = active_reader.read_location(
        variable,
        lead_time_hours=lead_time_hours,
        chunk_row=chunk_row,
        chunk_col=chunk_col,
        generation=generation,
    )
    if stack is None or stack.shape[0] != layout.n_fields:
        return None

    point = _corner_values(stack, int(row_in_chunk), int(col_in_chunk))

    def count(value: float) -> int | None:
        return int(round(value)) if math.isfinite(value) else None

    censoring = point[layout.group_slice("censoring")]
    valid_count = count(float(censoring[0]))
    finite_count = count(float(censoring[1]))
    unlimited_count = count(float(censoring[2]))
    if valid_count is None:
        # A cell the aggregator refused: the counts are absent together, so there is nothing to
        # report and the members can say more.
        return None
    probability: float | None = None
    if unlimited_count is not None and valid_count > 0:
        probability = unlimited_count / valid_count

    conditional = point[layout.group_slice("conditional")]
    names = group_field_names("conditional")
    payload: dict[str, Any] = {
        "valid_member_count": valid_count,
        "finite_member_count": finite_count,
        "unlimited_member_count": unlimited_count,
        "unlimited_probability": probability,
    }
    # The stored names carry the group's own prefix (``COND_MEAN``); the caller wants the
    # statistics under the names the response schema uses, so the prefix is stripped rather than
    # re-listed -- a second list would be a second place for the two to drift.
    conditional_values: dict[str, float] = {}
    for index, name in enumerate(names):
        value = float(conditional[index])
        if math.isfinite(value):
            conditional_values[name.removeprefix("COND_").lower()] = value
    payload["conditional"] = conditional_values if conditional_values else None
    return payload


def wind_products_from_aggregate(
    *,
    store_path: str,
    lead_time_hours: int,
    chunk_row: int,
    chunk_col: int,
    row_in_chunk: int,
    col_in_chunk: int,
    generation: str | None = None,
    reader: AggregateShardReader | None = None,
) -> dict[str, Any] | None:
    """Serve the wind products -- consensus vector and rose -- from ``wind_10m``'s container.

    Those two products are functions of each member's ``(u, v)`` pair rather than of the speed
    distribution, so they cannot be read off a distribution. They are stored as fields for
    exactly that reason (``domain.field_layout``'s rose group), and this is the read side of
    that decision.

    Two things the container can say that the member path cannot:

    * **the consensus direction's error is bounded and small.** Its angle is stored as a sin/cos
      pair at a 0.001 step, because a degree field near 0/360 is discontinuous and a fixed-point
      angle would need 360/step (past int16 beyond a step of 0.1). Measured over a full sweep,
      the worst-case recovery error is **0.04 degrees** and the mean 0.014 -- inside the 0.1
      degrees the response rounds to.
    * **the calm count is derived, not stored.** The rose's cells sum to the share of *non-calm*
      members -- a calm member has no direction -- so the calm count is the members less the
      non-calm ones the rose accounts for. It is ``None`` when the container has no rose at all.

    ``None`` whenever an aggregate cannot answer.

    Returns:
        ``{"consensus": {...}, "rose": {...}}``, or ``None``. The rose's sector probabilities
        are fractions of the member set and its ``bins``' keys are bucket indices -- the bucket
        *edges* travel as fields, because they are quantiles of this cycle's member set and a
        reader cannot label a bucket without them.
    """
    from domain.models.wind import (
        CARDINAL_DIRECTIONS_8,
        derive_meteorological_direction,
    )

    try:
        layout = aggregate_fields_for("wind_10m")
    except FieldLayoutError:  # pragma: no cover - the variable is registered
        return None

    active_reader = reader if reader is not None else AggregateShardReader(store_path)
    stack = active_reader.read_location(
        "wind_10m",
        lead_time_hours=lead_time_hours,
        chunk_row=chunk_row,
        chunk_col=chunk_col,
        generation=generation,
    )
    if stack is None or stack.shape[0] != layout.n_fields:
        return None

    point = _corner_values(stack, int(row_in_chunk), int(col_in_chunk))
    member_count = float(point[0])
    if not math.isfinite(member_count) or member_count <= 0:
        return None

    rose_fields_point = point[layout.group_slice("rose")]
    scalars = rose_fields_point[
        ROSE_SCALAR_OFFSET : ROSE_SCALAR_OFFSET + ROSE_SCALAR_COUNT
    ]
    edges = rose_fields_point[-ROSE_EDGE_COUNT:]
    consensus_speed_mps = float(scalars[0])
    coherence = float(scalars[1])
    direction_sin = float(scalars[2])
    direction_cos = float(scalars[3])
    if not all(
        math.isfinite(value)
        for value in (consensus_speed_mps, coherence, direction_sin, direction_cos)
    ):
        # No member had a direction here -- an all-calm or all-missing cell -- so there is no
        # consensus to report and the members can say more about why.
        return None

    mean_u = -consensus_speed_mps * direction_sin
    mean_v = -consensus_speed_mps * direction_cos
    direction_deg = derive_meteorological_direction(mean_u, mean_v)
    cardinal = (
        CARDINAL_DIRECTIONS_8[
            int(math.floor(((direction_deg or 0.0) + 22.5) / 45.0)) % len(CARDINAL_DIRECTIONS_8)
        ]
        if direction_deg is not None
        else "CALM"
    )

    cells = rose_fields_point[:ROSE_CELL_COUNT]
    if not np.isfinite(cells).all():
        return None
    # The rose holds a share of the non-calm members, so the calm count is what it leaves over.
    non_calm = float(np.sum(cells)) * member_count
    calm_count = int(round(member_count - non_calm))
    sectors: list[dict[str, Any]] = []
    for sector_index, name in enumerate(CARDINAL_DIRECTIONS_8):
        sector_cells = cells[
            sector_index * ROSE_BUCKETS : (sector_index + 1) * ROSE_BUCKETS
        ]
        probability = float(np.sum(sector_cells))
        sectors.append(
            {
                "sector": name,
                "count": int(round(probability * member_count)),
                "probability": probability,
                "bins": {
                    str(bucket): float(sector_cells[bucket])
                    for bucket in range(ROSE_BUCKETS)
                },
            }
        )
    return {
        "consensus": {
            "speed_mps": consensus_speed_mps,
            "direction_deg": direction_deg,
            "cardinal": cardinal,
            "coherence": coherence,
        },
        "rose": {
            "calm_count": calm_count,
            "calm_probability": calm_count / member_count,
            "member_count": int(round(member_count)),
            "bucket_edges_mps": [float(edge) for edge in edges],
            "sectors": sectors,
        },
    }


def precipitation_phase_from_aggregate(
    *,
    store_path: str,
    lead_time_hours: int,
    chunk_row: int,
    chunk_col: int,
    row_in_chunk: int,
    col_in_chunk: int,
    generation: str | None = None,
    reader: AggregateShardReader | None = None,
) -> dict[str, Any] | None:
    """Serve the precipitation phase support and transition frequencies from the container.

    The phase group stores the six physical-phase supports for the current interval and the six
    for its predecessor, which is what ``phase_support`` reports; the transition group stores the
    twenty categories. Both are functions of the per-member flags rather than of the amount
    distribution, which is why they are fields.

    ``transition_frequency`` is keyed by category value, and the zero-frequency categories are
    **kept**: the member path drops them, but a stored field is a count and a reader that wants
    to know a category is never reached should not have to infer it from an absence.

    ``phase_support`` is keyed by :class:`~domain.models.precipitation.PhysicalPhase` value. A
    cell whose phase planes are all absent (a lead with no predecessor, or no member the
    classifier could read) reports ``None`` for the affected planes rather than zeros: a zero
    would claim the ensemble said "not that phase", and the container's own statement is that it
    has nothing to say.

    ``None`` whenever an aggregate cannot answer.
    """
    from domain.models.precipitation import PhysicalPhase, PrecipitationTransition

    try:
        layout = aggregate_fields_for("precipitation_amount_3h")
    except FieldLayoutError:  # pragma: no cover - the variable is registered
        return None

    active_reader = reader if reader is not None else AggregateShardReader(store_path)
    stack = active_reader.read_location(
        "precipitation_amount_3h",
        lead_time_hours=lead_time_hours,
        chunk_row=chunk_row,
        chunk_col=chunk_col,
        generation=generation,
    )
    if stack is None or stack.shape[0] != layout.n_fields:
        return None

    point = _corner_values(stack, int(row_in_chunk), int(col_in_chunk))
    member_count = float(point[0])
    if not math.isfinite(member_count) or member_count <= 0:
        return None

    phase = point[layout.group_slice("phase")]
    current = phase[: len(PhysicalPhase)]
    previous = phase[len(PhysicalPhase) :]
    support: dict[str, float | None] = {}
    for index, phase_name in enumerate(PhysicalPhase):
        value = float(current[index])
        support[phase_name.value] = value if math.isfinite(value) else None
        previous_value = float(previous[index])
        if math.isfinite(previous_value):
            support[f"previous_{phase_name.value}"] = previous_value
    if all(value is None for value in support.values()):
        return None

    transition = point[layout.group_slice("transition")]
    frequencies = {
        name.value: float(transition[index])
        for index, name in enumerate(PrecipitationTransition)
        if math.isfinite(float(transition[index]))
    }
    return {
        "member_count": int(round(member_count)),
        "phase_support": support,
        "transition_frequency": frequencies,
    }


def cloud_censoring_at_cell(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    lat_idx: int,
    lon_idx: int,
    generation: str | None = None,
) -> dict[str, Any] | None:
    """Censoring from a container addressed by absolute grid indices."""
    active_reader = AggregateShardReader(store_path)
    geometry = active_reader.open(variable, lead_time_hours, generation=generation)
    if geometry is None or not (
        0 <= lat_idx < geometry.grid_lat and 0 <= lon_idx < geometry.grid_lon
    ):
        return None
    chunk_row, row_in_chunk = divmod(lat_idx, geometry.chunk_lat)
    chunk_col, col_in_chunk = divmod(lon_idx, geometry.chunk_lon)
    return cloud_censoring_from_aggregate(
        variable,
        store_path=store_path,
        lead_time_hours=lead_time_hours,
        chunk_row=chunk_row,
        chunk_col=chunk_col,
        row_in_chunk=row_in_chunk,
        col_in_chunk=col_in_chunk,
        generation=generation,
        reader=active_reader,
    )


def gated_wind_products(
    *,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    generation: str | None = None,
) -> dict[str, Any] | None:
    """The wind products for a geographic point, under the reader gate.

    The cell is derived from the store's own coordinate axes inside the same gate the member path
    uses, because the descriptor carries extents but not axis values (see
    :func:`gated_statistics_from_aggregate`).
    """
    from api.core.manifest_reader import manifest_generation
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.services.point_forecast import _derive_grid

    resolved_generation = generation
    if resolved_generation is None:
        resolved_generation = manifest_generation(store_path)

    def select(dataset: Any) -> dict[str, Any] | None:
        if "wind_10m" not in dataset.data_vars and not (
            "wind_u_10m" in dataset.data_vars and "wind_v_10m" in dataset.data_vars
        ):
            return None
        grid, lat_descending, lon_descending = _derive_grid(dataset)
        try:
            row_f, col_f = grid.row_col_from_coordinates(latitude, longitude)
        except Exception:  # noqa: BLE001 - an off-grid point falls back to the members
            return None
        row = min(int(math.floor(row_f)), grid.rows - 1)
        col = min(int(math.floor(col_f)), grid.cols - 1)

        def stored(value: int, size: int, descending: bool) -> int:
            return (size - 1 - value) if descending else value

        lat_idx = stored(row, int(dataset.sizes["latitude"]), lat_descending)
        lon_idx = stored(col, int(dataset.sizes["longitude"]), lon_descending)
        geometry = AggregateShardReader(store_path).open(
            "wind_10m", lead_time_hours, generation=resolved_generation
        )
        if geometry is None or not (
            0 <= lat_idx < geometry.grid_lat and 0 <= lon_idx < geometry.grid_lon
        ):
            return None
        chunk_row, row_in_chunk = divmod(lat_idx, geometry.chunk_lat)
        chunk_col, col_in_chunk = divmod(lon_idx, geometry.chunk_lon)
        return wind_products_from_aggregate(
            store_path=store_path,
            lead_time_hours=lead_time_hours,
            chunk_row=chunk_row,
            chunk_col=chunk_col,
            row_in_chunk=row_in_chunk,
            col_in_chunk=col_in_chunk,
            generation=resolved_generation,
        )

    return gated_read_dataset_with_selector(store_path, select)


def gated_precipitation_phase(
    *,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    generation: str | None = None,
) -> dict[str, Any] | None:
    """The precipitation phase products for a geographic point, under the reader gate."""
    from api.core.manifest_reader import manifest_generation
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.services.point_forecast import _derive_grid

    resolved_generation = generation
    if resolved_generation is None:
        resolved_generation = manifest_generation(store_path)

    def select(dataset: Any) -> dict[str, Any] | None:
        if "precipitation_amount_3h" not in dataset.data_vars:
            return None
        grid, lat_descending, lon_descending = _derive_grid(dataset)
        try:
            row_f, col_f = grid.row_col_from_coordinates(latitude, longitude)
        except Exception:  # noqa: BLE001 - an off-grid point falls back to the members
            return None
        row = min(int(math.floor(row_f)), grid.rows - 1)
        col = min(int(math.floor(col_f)), grid.cols - 1)

        def stored(value: int, size: int, descending: bool) -> int:
            return (size - 1 - value) if descending else value

        lat_idx = stored(row, int(dataset.sizes["latitude"]), lat_descending)
        lon_idx = stored(col, int(dataset.sizes["longitude"]), lon_descending)
        geometry = AggregateShardReader(store_path).open(
            "precipitation_amount_3h", lead_time_hours, generation=resolved_generation
        )
        if geometry is None or not (
            0 <= lat_idx < geometry.grid_lat and 0 <= lon_idx < geometry.grid_lon
        ):
            return None
        chunk_row, row_in_chunk = divmod(lat_idx, geometry.chunk_lat)
        chunk_col, col_in_chunk = divmod(lon_idx, geometry.chunk_lon)
        return precipitation_phase_from_aggregate(
            store_path=store_path,
            lead_time_hours=lead_time_hours,
            chunk_row=chunk_row,
            chunk_col=chunk_col,
            row_in_chunk=row_in_chunk,
            col_in_chunk=col_in_chunk,
            generation=resolved_generation,
        )

    return gated_read_dataset_with_selector(store_path, select)


def gated_cloud_censoring(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    generation: str | None = None,
) -> dict[str, Any] | None:
    """The cloud censoring counts and conditional statistics for a point, under the reader gate."""
    from api.core.manifest_reader import manifest_generation
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.services.point_forecast import _derive_grid

    resolved_generation = generation
    if resolved_generation is None:
        resolved_generation = manifest_generation(store_path)

    def select(dataset: Any) -> dict[str, Any] | None:
        if variable not in dataset.data_vars:
            return None
        grid, lat_descending, lon_descending = _derive_grid(dataset)
        try:
            row_f, col_f = grid.row_col_from_coordinates(latitude, longitude)
        except Exception:  # noqa: BLE001 - an off-grid point falls back to the members
            return None
        row = min(int(math.floor(row_f)), grid.rows - 1)
        col = min(int(math.floor(col_f)), grid.cols - 1)

        def stored(value: int, size: int, descending: bool) -> int:
            return (size - 1 - value) if descending else value

        lat_idx = stored(row, int(dataset.sizes["latitude"]), lat_descending)
        lon_idx = stored(col, int(dataset.sizes["longitude"]), lon_descending)
        return cloud_censoring_at_cell(
            variable,
            store_path=store_path,
            lead_time_hours=lead_time_hours,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            generation=resolved_generation,
        )

    return gated_read_dataset_with_selector(store_path, select)


def _stored_distribution_at_point(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    edges: list[float] | None,
    bins: int,
    generation: str | None,
) -> tuple[list[float], list[int]] | None:
    """The stored distribution at a point, on ``edges`` or on a grid it states itself.

    One implementation for both grid modes, because everything except the grid is identical and
    the gate, the coordinate derivation and the field slicing must not be able to drift between
    two copies. The two callers differ only in whether the members still exist to define a range.

    ``None`` whenever an aggregate cannot answer, the point is off-grid, or the cell is
    unobserved: the caller then has no stored line to draw, which is a state the migration has to
    be able to show rather than hide.
    """
    from api.core.manifest_reader import manifest_generation
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.services.dual_source import stored_edges, stored_exceedance, stored_histogram
    from api.services.point_forecast import _derive_grid
    from domain.variable_class import spec_for

    try:
        spec = spec_for(variable)
        layout = aggregate_fields_for(variable)
    except (VariableClassError, FieldLayoutError):
        return None
    if edges is not None and not edges:
        return None

    resolved_generation = generation
    if resolved_generation is None:
        resolved_generation = manifest_generation(store_path)

    def select(dataset: Any) -> tuple[list[float], list[int]] | None:
        if variable not in dataset.data_vars:
            return None
        grid, lat_descending, lon_descending = _derive_grid(dataset)
        try:
            row_f, col_f = grid.row_col_from_coordinates(latitude, longitude)
        except Exception:  # noqa: BLE001 - an off-grid point has no stored line
            return None
        row = min(int(math.floor(row_f)), grid.rows - 1)
        col = min(int(math.floor(col_f)), grid.cols - 1)

        def stored(value: int, size: int, descending: bool) -> int:
            return (size - 1 - value) if descending else value

        lat_idx = stored(row, int(dataset.sizes["latitude"]), lat_descending)
        lon_idx = stored(col, int(dataset.sizes["longitude"]), lon_descending)
        active_reader = AggregateShardReader(store_path)
        geometry = active_reader.open(
            variable, lead_time_hours, generation=resolved_generation
        )
        if geometry is None or not (
            0 <= lat_idx < geometry.grid_lat and 0 <= lon_idx < geometry.grid_lon
        ):
            return None
        chunk_row, row_in_chunk = divmod(lat_idx, geometry.chunk_lat)
        chunk_col, col_in_chunk = divmod(lon_idx, geometry.chunk_lon)
        stack = active_reader.read_location(
            variable,
            lead_time_hours=lead_time_hours,
            chunk_row=chunk_row,
            chunk_col=chunk_col,
            generation=resolved_generation,
        )
        if stack is None or stack.shape[0] != layout.n_fields:
            return None
        point = _corner_values(stack, int(row_in_chunk), int(col_in_chunk))
        member_count = float(point[0])
        if not math.isfinite(member_count) or member_count <= 0:
            return None
        distribution = point[layout.distribution_slice]
        # The grid, when the caller has none: the distribution's own bounds. Computed here rather
        # than by the caller because it is a function of the *cell being read* -- on the bin
        # encoding the range is that cell's mean and spread, not the container's.
        active_edges = edges
        if active_edges is None:
            active_edges = stored_edges(distribution, spec, bins)
            if not active_edges:
                return None
        tail = stored_exceedance(distribution, spec, active_edges)
        if not tail:
            return None
        counts = stored_histogram(tail, active_edges, int(round(member_count)))
        if counts is None:
            return None
        return active_edges, counts

    return gated_read_dataset_with_selector(store_path, select)


def stored_histogram_at_point(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    edges: list[float],
    generation: str | None = None,
) -> list[int] | None:
    """The stored distribution's mass in each bin of ``edges``, for a geographic point.

    The comparison mode of :func:`_stored_distribution_at_point`: the caller supplies the bin grid
    the member values defined, so the two histograms are on one partition and can be compared bin
    for bin.
    """
    from api.services.dual_source import DEFAULT_BINS

    resolved = _stored_distribution_at_point(
        variable,
        store_path=store_path,
        lead_time_hours=lead_time_hours,
        latitude=latitude,
        longitude=longitude,
        edges=edges,
        bins=DEFAULT_BINS,
        generation=generation,
    )
    return None if resolved is None else resolved[1]


def stored_distribution_at_point(
    variable: str,
    *,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    bins: int | None = None,
    generation: str | None = None,
) -> tuple[list[float], list[int]] | None:
    """The stored distribution at a point **as its own grid**, with no member values needed.

    The mode a store needs once its members are gone. The comparison mode above takes its grid from
    the member values because they are what is being replaced, and that is still the right choice
    while they exist; but a store whose members have been reclaimed has no such values, and the
    chart drawn from the stored fields is then the answer rather than a second opinion about one.
    Asking the members for a range at that point would make the stored line disappear exactly when
    it became the only line there is.

    The grid comes from the distribution's own outermost information -- the outer stored levels for
    the quantile encoding, the mean and spread for the bin encoding -- so it is a statement the
    container already makes about itself. See :func:`api.services.dual_source.stored_edges`.

    Returns:
        ``(edges, counts)`` for one point, or ``None`` when an aggregate cannot answer. The edges
        are returned because they are per cell on one of the two encodings, so a caller cannot
        reconstruct them from the variable alone.
    """
    from api.services.dual_source import DEFAULT_BINS

    return _stored_distribution_at_point(
        variable,
        store_path=store_path,
        lead_time_hours=lead_time_hours,
        latitude=latitude,
        longitude=longitude,
        edges=None,
        bins=DEFAULT_BINS if bins is None else int(bins),
        generation=generation,
    )


#: Variables whose response carries something an aggregate cannot represent, because it is a
#: function of the *per-member values* rather than of the distribution.
#:
#: This set is now **empty**, and that is the point of the supplementary groups: ``wind_10m``'s
#: consensus vector and rose, ``precipitation_amount_3h``'s phase support and transitions, and
#: the two cloud variables' censoring are all stored as fields, computed by the same arithmetic
#: the member path uses (``domain.product_fields``) and read back by the functions above. The set
#: is kept rather than deleted because it is the place a future variable's exception would go,
#: and because the capability check below reads from it -- an empty set is a fact about the
#: design, not an unfinished list.
SPECIAL_PER_MEMBER_VARIABLES: frozenset[str] = frozenset()

#: The strict operators a stored distribution can answer. ``gte``/``lte`` count members *at*
#: the threshold, which needs the atom at that value; see :func:`exceedance_probability`.
STRICT_EXCEEDANCE_OPERATORS: frozenset[str] = frozenset({"gt", "lt"})


def aggregate_can_answer(
    variable: str,
    *,
    operator: str | None = None,
    needs_members: bool = False,
) -> bool:
    """Whether this query can be served from an aggregate at all.

    A capability check, separate from the per-request attempt: the endpoints consult it before
    deciding which path to take, so a variable that can never be served from an aggregate does
    not pay for a probe that is certain to miss. The per-request attempt still exists, because
    a store may simply have no aggregate for a variable that could have one.

    Args:
        variable: Variable code.
        operator: Exceedance operator, for the probability endpoint.
        needs_members: Whether the caller asked for the raw member values, which an aggregate
            has collapsed and therefore cannot return.
    """
    if needs_members:
        return False
    if variable in SPECIAL_PER_MEMBER_VARIABLES:
        return False
    if operator is not None:
        # A threshold query reads the stored distribution, so it needs one: `gt`/`lt` are the
        # operators the encodings can express, and a flag stores a probability rather than a
        # distribution to compare a threshold against.
        if operator not in STRICT_EXCEEDANCE_OPERATORS:
            return False
        try:
            spec_for(variable)
        except VariableClassError:
            return False
        return True
    # A statistics query is answerable whenever the variable has a stored field layout: the
    # distribution encodings answer it from their fields, and a 0/1 flag answers it from its
    # fraction (see :func:`fraction_statistics`).
    try:
        aggregate_fields_for(variable)
    except FieldLayoutError:
        return False
    return True


def aggregate_answers_for_lead(
    variable: str, *, store_path: str, lead_time_hours: int
) -> bool:
    """Whether a readable container exists for this ``(variable, lead)`` in a store.

    The serving-side evidence check, and the API's own: it opens the container through
    :class:`~api.core.aggregate_reader.AggregateShardReader` -- the same reader the products come
    from, so "the probe says yes" and "the read succeeds" cannot disagree -- and applies the same
    refusals (an unknown encoding, a v1 container, a field count that is not this variable's
    layout).

    **A probe, not a read.** It fetches the trailer and descriptor and returns; the payload is
    never touched, which is what makes it affordable on a resolution path that runs per request.

    A caller uses this to decide whether a variable's members are *represented* at a lead -- the
    case where they have already been released and the coverage floor no longer describes
    anything. It must not be used to decide anything a product depends on: the products read the
    container through their own calls and fall through to the members if it is gone.
    """
    try:
        spec_for(variable)
    except VariableClassError:
        return False
    try:
        geometry = AggregateShardReader(store_path).open(variable, lead_time_hours)
    except Exception:  # noqa: BLE001 - an unreadable store is "no aggregate", not a failure
        return False
    return geometry is not None


def try_read_aggregate(attempt: Callable[[], _T | None]) -> _T | None:
    """Run an aggregate read, treating an unreadable store as "no aggregate".

    The aggregate path is an optimisation over the member shards, and it opens the store
    through the reader gate, which raises ``FileNotFoundError`` when the run is not readable.
    Propagating that would turn "this store has no usable aggregate" into a failure of a
    request the member path could have served; the member path is still the reader of record
    and raises its own error if it cannot read either.

    Only that case is absorbed. A damaged container raises ``ShardFormatError`` and a bad
    encoding raises through the domain helpers, and both are real faults that must stay
    visible -- the reader already reports them for exactly that reason.
    """
    try:
        return attempt()
    except FileNotFoundError:
        return None


__all__ = [
    "SPECIAL_PER_MEMBER_VARIABLES",
    "STATISTIC_NAMES",
    "STRICT_EXCEEDANCE_OPERATORS",
    "AggregatePointProbability",
    "AggregatePointStatistics",
    "aggregate_answers_for_lead",
    "aggregate_can_answer",
    "cloud_censoring_at_cell",
    "cloud_censoring_from_aggregate",
    "exceedance_probability",
    "gated_cloud_censoring",
    "gated_exceedance_from_aggregate",
    "gated_precipitation_phase",
    "gated_statistics_from_aggregate",
    "gated_wind_products",
    "precipitation_phase_from_aggregate",
    "statistics_from_aggregate",
    "statistics_from_aggregate_at_cell",
    "stored_distribution_at_point",
    "stored_histogram_at_point",
    "try_read_aggregate",
    "wind_products_from_aggregate",
]
