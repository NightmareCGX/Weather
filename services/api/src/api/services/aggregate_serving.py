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
from domain.field_layout import FieldLayoutError, aggregate_fields_for
from domain.variable_class import VariableClassError, spec_for

from api.core.aggregate_reader import AggregateGeometry, AggregateShardReader

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Statistics the platform serves per variable (``EnsembleStatistics``). A caller passing a
#: different name gets a KeyError from the dataclass rather than a silent zero.
STATISTIC_NAMES: tuple[str, ...] = (
    "mean",
    "median",
    "spread",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
)

#: Statistics that coincide with a stored quantile level for the quantile encoding, and are
#: therefore reproduced exactly rather than reconstructed.
_EXACT_QUANTILE_STATISTICS: frozenset[str] = frozenset({"median", "p10", "p50", "p90"})


@dataclass(frozen=True)
class AggregatePointStatistics:
    """One variable's statistics at a point, served from an aggregate shard.

    Attributes:
        values: Statistic name to value, for the names the encoding can express.
        spec: The spec the aggregate was written with.
        exact: Statistics reproduced bit-exactly rather than reconstructed.
        geometry: The container geometry the values were read from.
        member_count: Ensemble members the aggregate was computed from, when the point was
            observed. ``None`` when the container does not carry a count, so a caller falls
            back to the member path rather than reporting an invented number.
    """

    values: dict[str, float]
    spec: AggregateSpec
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
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
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

    for name in ("p10", "p25", "median", "p50", "p75", "p90"):
        probability = 0.50 if name == "median" else float(name[1:]) / 100.0
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
        spec = spec_for(variable)
        layout = aggregate_fields_for(variable)
    except (VariableClassError, FieldLayoutError):
        return None

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
        spec = spec_for(variable)
        layout = aggregate_fields_for(variable)
    except (VariableClassError, FieldLayoutError):
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


#: Variables whose response carries something an aggregate cannot represent, because it is a
#: function of the *per-member values* rather than of the distribution.
#:
#: ``wind_10m`` returns a consensus vector and a wind rose, both computed from each member's
#: ``(u, v)`` pair; a statistic of the speed distribution does not determine them.
#: ``precipitation_amount_3h`` returns a phase-support map and transition frequencies, built
#: from each member's 0/1 phase flag. ``cloud_ceiling`` splits its members into a finite height
#: and an "unlimited" sentinel before summarising, and ``cloud_cover_3h`` summarises only the
#: members inside ``[0, 100]`` -- both are per-member censoring, which a collapsed distribution
#: has already lost.
#:
#: These are served from the member shards, which the aggregate phase does not replace.
SPECIAL_PER_MEMBER_VARIABLES: frozenset[str] = frozenset(
    {"wind_10m", "precipitation_amount_3h", "cloud_ceiling", "cloud_cover_3h"}
)

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
    if operator is not None and operator not in STRICT_EXCEEDANCE_OPERATORS:
        return False
    try:
        spec_for(variable)
    except VariableClassError:
        return False
    return True


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
    "aggregate_can_answer",
    "exceedance_probability",
    "statistics_from_aggregate",
    "try_read_aggregate",
]
