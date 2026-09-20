"""The two histograms the source migration compares, on one shared bin grid.

Why this exists
---------------
The front end is moving from drawing distributions out of raw ensemble members to drawing them out
of the stored fields (the aggregate containers). A change of source that shows only one line is
unauditable: a difference between the two would look like a rendering change rather than a data
change, and the migration's whole risk is exactly that the stored fields are *nearly* the member
answer. So while both exist, both are delivered, the client draws both, and the member-derived line
is deleted once the two have been seen to agree.

Both lines therefore have to be on **one bin grid**, and the grid has to be the same on both sides
or the comparison is between two different partitions. The grid comes from the member values (their
range, ten equal bins by default) and the stored side is *integrated over that grid* rather than
re-derived from a distribution of its own: a stored quantile function knows its CDF at every point,
and the mass in a bin is the CDF's difference across it.

What each side is, precisely
----------------------------
``member_histogram`` counts raw member values. They are interpolated values at one point, so a
"count" is a count of members, not of anything continuous.

``stored_histogram`` differences the stored distribution's tail probability at the grid's edges.
For the quantile encoding that is an inversion of the stored levels; for the bin encoding it is the
accumulated normalised shape. Both go through the same ``domain.aggregate`` helpers the statistics
reader uses, so this histogram and the percentiles a response reports cannot disagree about the
distribution they came from.

Both are delivery-only. Neither is part of the statistics contract, and both disappear with the
members -- which is the point of the exercise.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from domain.aggregate import (
    KIND_MEAN_STD_BINS,
    KIND_QUANTILE_FUNCTION,
    AggregateError,
    AggregateSpec,
    exceedance_from_bins,
    exceedance_from_quantiles,
)

#: Bins when the caller does not say. Ten is what the front end's own ``histogramBins`` chooses for
#: 30 members (``ceil(log2(30)) + 1``), so the two source lines are comparable bin for bin.
DEFAULT_BINS: int = 10


def shared_edges(values: npt.NDArray[np.floating], bins: int = DEFAULT_BINS) -> list[float]:
    """The bin edges both sides are measured on, from the member values.

    The member values define the grid because they are the thing being replaced: a grid derived
    from the stored distribution could hide a disagreement about the *range* by construction.

    A degenerate range (every member identical, or a single value) is widened by one unit on each
    side, so the grid is always usable and the value lands in a middle bin rather than on an edge.
    Bins are closed on the left and open on the right except the last, which includes its upper
    edge -- otherwise the maximum would fall outside the histogram.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return []
    low = float(finite.min())
    high = float(finite.max())
    if not high > low:
        low -= 1.0
        high += 1.0
    return [low + (high - low) * index / bins for index in range(bins + 1)]


def member_histogram(values: npt.NDArray[np.floating], edges: list[float]) -> list[int]:
    """How many finite member values fall in each bin of ``edges``.

    ``np.histogram``'s convention is the one :func:`shared_edges` states (the last bin includes its
    upper edge), so no manual boundary handling is needed -- and using the same convention on both
    sides is what makes the two lines comparable rather than merely similar.
    """
    if len(edges) < 2:
        return []
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    counts = np.histogram(finite, bins=np.asarray(edges, dtype=np.float64))[0]
    return [int(count) for count in counts]


def stored_exceedance(
    fields_at_point: npt.NDArray[np.float32],
    spec: AggregateSpec,
    edges: list[float],
) -> list[float | None]:
    """``P(X > edge)`` at each edge, read from a cell's stored distribution fields.

    The fields are the container's own, sliced to the distribution by the caller, and ``spec`` is
    the variable's encoding. Both kinds go through the helpers the statistics reader uses, so this
    and the reported percentiles cannot disagree about the distribution they came from.

    Returns:
        One tail probability per edge; ``None`` where the cell is unobserved or the fields are not
        this spec's shape. An empty list means "not this encoding at all".
    """
    if fields_at_point.size != spec.n_fields:
        return []
    point = np.asarray(fields_at_point, dtype=np.float32)
    if not np.isfinite(point).any():
        # An unobserved cell's distribution fields are all absent, and inverting a plane of NaNs
        # *fabricates* a probability rather than refusing to answer: the inversion compares each
        # threshold against the levels, every comparison against NaN is false, and the interpolation
        # then reads the outermost level -- which reports ``P(X > t) = 0.999`` at every threshold.
        # Measured, and the reason the statistics reader refuses the same point up front.
        return []
    out: list[float | None] = []
    for edge in edges:
        try:
            if spec.kind == KIND_QUANTILE_FUNCTION:
                probability = float(
                    np.asarray(
                        exceedance_from_quantiles(
                            point.reshape(spec.n_fields, 1, 1),
                            spec.quantile_levels(),
                            float(edge),
                        )
                    )[0, 0]
                )
            elif spec.kind == KIND_MEAN_STD_BINS:
                probability = float(
                    np.asarray(
                        exceedance_from_bins(
                            point[0].reshape(1),
                            point[1].reshape(1),
                            point[2:].reshape(spec.n_bins, 1),
                            float(edge),
                            spec,
                        )
                    )[0]
                )
            else:  # pragma: no cover - the spec's own validation rejects other kinds
                return []
        except AggregateError:
            probability = float("nan")
        out.append(probability if math.isfinite(probability) else None)
    return out


def stored_histogram(
    exceedance: list[float | None],
    edges: list[float],
    member_count: int,
) -> list[int] | None:
    """The stored distribution's mass in each bin, as a count of members.

    The mass in ``[edges[i], edges[i+1])`` is ``P(X > edges[i]) - P(X > edges[i+1])``, and the
    count is that probability times the member count -- which is what makes the two histograms
    comparable: both are counts of the members the container was computed from.

    **The first bin is the exception, and it is not a detail.** Its lower edge is the sample
    minimum, and a zero-inflated field has a large atom there: ``P(X > min)`` excludes every member
    that *is* the minimum, and the member histogram counts them. Measured on real GEFS
    precipitation, one cell had 30% of its members exactly at zero, so ``P(X > min)`` dropped nine
    of thirty members and the stored line came out a third short in its first bin alone. The first
    bin's mass is therefore ``P(X <= edges[1])``, which is ``1 - P(X > edges[1])`` for a grid whose
    lower edge is the sample minimum -- the caller's grid, by construction
    (:func:`shared_edges`).

    **The counts are apportioned rather than rounded independently.** Each bin's mass is a fraction
    of a member -- a ten-bin histogram of thirty members has 3 members per bin at most -- so
    rounding each bin on its own loses the remainders and the counts came out three short of the
    member set. Largest-remainder apportionment (give every bin its floor, then hand the leftover
    members to the largest remainders) makes the counts sum to the member count exactly, which is
    what lets the two lines be compared bin for bin at all.

    ``None`` when any probability is unavailable (a cell the store refused), because a histogram
    with holes in it is not a comparison of anything.
    """
    if len(edges) < 2 or len(exceedance) != len(edges):
        return None
    if any(value is None or not math.isfinite(value) for value in exceedance):
        return None
    probabilities = [float(value) for value in exceedance if value is not None]
    masses: list[float] = []
    for index in range(len(edges) - 1):
        masses.append(
            1.0 - probabilities[1]
            if index == 0
            else probabilities[index] - probabilities[index + 1]
        )
    # A grid whose lower edge is not the sample minimum is a caller error, and the negative mass it
    # produces would be worse than reading as one: said out loud rather than clipped silently.
    if any(mass < -1e-9 for mass in masses):
        return None
    return _apportion([max(mass, 0.0) for mass in masses], member_count)


def _apportion(masses: list[float], member_count: int) -> list[int]:
    """Turn per-bin probabilities into integer counts that sum to ``member_count``.

    Largest remainder: every bin gets its floor, and the members left over go to the bins with the
    largest fractional parts. The alternative -- rounding each bin on its own -- is what produced a
    histogram three members short of its own member set, because every bin in a coarse histogram has
    a fractional part worth discarding.
    """
    exact = [mass * member_count for mass in masses]
    counts = [int(math.floor(value)) for value in exact]
    remaining = member_count - sum(counts)
    if remaining > 0:
        order = sorted(
            range(len(counts)),
            key=lambda index: (-(exact[index] - counts[index]), index),
        )
        for index in order[:remaining]:
            counts[index] += 1
    return counts


__all__ = [
    "DEFAULT_BINS",
    "member_histogram",
    "shared_edges",
    "stored_exceedance",
    "stored_histogram",
]
