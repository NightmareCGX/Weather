"""The two histograms the source migration compares, on one shared bin grid.

Why this exists
---------------
The front end is moving from drawing distributions out of raw ensemble members to drawing them out
of the stored fields (the aggregate containers). A change of source that shows only one line is
unauditable: a difference between the two would look like a rendering change rather than a data
change, and the migration's whole risk is exactly that the stored fields are *nearly* the member
answer. So while both exist, both are delivered, the client draws both, and the member-derived line
is deleted once the two have been seen to agree.

**Why the stored side can also define the grid** (:func:`stored_edges`), and why that matters more
than the comparison does. While both sources exist the grid comes from the member values, for the
reason below. Once a variable's members have been reclaimed there are no member values to take a
range from -- and the chart is then the *only* thing left to draw, so a stored line that needed the
members to say where its bins went would disappear at exactly the moment it became the answer
(``IMPLEMENTATION.md`` §19.16). A stored distribution can state its own range: the quantile
encoding's outermost stored levels, or the bin encoding's mean and spread, which are the bounds it
was built on. The stored side therefore has two grid modes, and which one is in use is the caller's
choice because it is the caller that knows whether members still exist.

Both lines on **one grid** is what makes them comparable at all, and the grid has to be the same on
both sides or the comparison is between two different partitions. The member grid is the members'
range in ten equal bins by default; the stored side is *integrated over that grid* rather than
re-derived from a distribution of its own, because a stored quantile function knows its CDF at
every point and the mass in a bin is the CDF's difference across it.

What each side is, precisely
----------------------------
``member_histogram`` counts raw member values. They are interpolated values at one point, so a
"count" is a count of members, not of anything continuous.

``stored_histogram`` differences the stored distribution's tail probability at the grid's edges.
For the quantile encoding that is an inversion of the stored levels; for the bin encoding it is the
accumulated normalised shape. Both go through the same ``domain.aggregate`` helpers the statistics
reader uses, so this histogram and the percentiles a response reports cannot disagree about the
distribution they came from.

Both are delivery-only. Neither is part of the statistics contract; the member line disappears with
the members, which is the point of the exercise.
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
    quantile_at,
)

#: Bins used when a caller asks for a count rather than naming one.
#:
#: Six is Sturges' rule for a 30-member sample (``ceil(log2(30)) + 1``), and it is the front end's
#: own ``histogramBins`` count -- so this is the value that puts the delivered lines on the same
#: partition the member bars are drawn on. Measured over 96 sampled (point, lead) pairs on real
#: GEFS, it is also the *most accurate* choice: the stored line's correlation with the curve the
#: chart draws falls from 0.92 at six bins to 0.69 at twenty and 0.62 at thirty-two, because past
#: that resolution a 30-member count histogram has about one member per bin and the curve does not.
#:
#: A caller that wants a fixed resolution can still say so; ``api.core.config``'s
#: ``ENSEMBLE_DUAL_SOURCE_BINS`` is ``0`` by default, meaning "this rule", and its configured
#: value is passed through when it is positive.
DEFAULT_BINS: int = 6


def member_bin_count(member_count: int) -> int:
    """The bin count a client drawing this sample would choose: Sturges' rule.

    Duplicated from the front end's ``histogramBins`` on purpose rather than shared, because the
    two runtimes cannot share code -- and pinned by a test on both sides, so the two cannot drift
    into delivering a partition the other does not draw.
    """
    return max(1, math.ceil(math.log2(max(1, member_count))) + 1)


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


#: Fraction of the stored range by which :func:`stored_edges` widens its own bounds. A grid whose
#: outermost edges are the extreme stored levels puts the entire upper tail's mass in the last bin
#: and nothing outside it, so the shape's ends are flat by construction. Widening by this much
#: gives the ends a bin to fall in -- chosen small because the stored levels are already the
#: distribution's own range and a wider margin would spend resolution on empty bins.
STORED_EDGE_MARGIN: float = 0.05


def stored_edges(
    fields_at_point: npt.NDArray[np.float32],
    spec: AggregateSpec,
    bins: int = DEFAULT_BINS,
) -> list[float]:
    """The bin edges a stored distribution states about itself, or ``[]`` if it cannot.

    The grid for a chart drawn from a store whose members are gone. Unlike :func:`shared_edges`,
    nothing here needs the member values: the range comes from the encoding's own outermost
    information.

    * **quantile function** -- the lowest and highest stored levels are the distribution's own
      outermost quantiles, so they are the range's bounds by definition. The *mean* is deliberately
      not used to centre them: it is a moment, and a heavy-tailed field's mean sits far from where
      its mass does.
    * **mean/std bins** -- the encoding's support is ``mean +- sigma_range * std`` in normalised
      units, and it is bounded by construction, so those are the bounds. Reading the levels instead
      would be wrong: a bin-encoded container stores no quantiles at all.

    The bounds are widened by :data:`STORED_EDGE_MARGIN` of the range so the outermost bins have
    mass in them (see that constant). A degenerate range -- every member identical, which the bin
    encoding represents as a zero spread -- is widened by one unit on each side, the same fallback
    :func:`shared_edges` uses.

    ``[]`` for an unobserved cell or a field vector that is not this spec's shape, so a caller
    reports "no stored line" rather than drawing one at invented bounds.
    """
    if fields_at_point.size != spec.n_fields:
        return []
    point = np.asarray(fields_at_point, dtype=np.float32)
    if not np.isfinite(point).any():
        return []

    if spec.kind == KIND_QUANTILE_FUNCTION:
        levels = spec.quantile_levels()
        try:
            low = float(np.asarray(quantile_at(point, levels, 0.0)))
            high = float(np.asarray(quantile_at(point, levels, 1.0)))
        except AggregateError:  # pragma: no cover - the shapes agree by construction
            return []
    elif spec.kind == KIND_MEAN_STD_BINS:
        mean = float(point[0])
        spread = float(point[1])
        if not math.isfinite(mean) or not math.isfinite(spread):
            return []
        half = float(spec.sigma_range) * spread
        low, high = mean - half, mean + half
    else:  # pragma: no cover - the spec's own validation rejects other kinds
        return []

    if not math.isfinite(low) or not math.isfinite(high):
        return []
    if not high > low:
        low -= 1.0
        high += 1.0
    margin = (high - low) * STORED_EDGE_MARGIN
    low -= margin
    high += margin
    return [low + (high - low) * index / bins for index in range(bins + 1)]


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

    **The outermost bins absorb what lies beyond them, and that is not a detail either.** The
    general form differences ``P(X > edge)`` between consecutive edges, so it counts
    ``(edges[i], edges[i+1]]`` and drops anything outside the grid entirely: the mass at or below
    the first edge, and the mass above the last. Both are real losses on a reconstructed
    distribution. A zero-inflated field has a large atom at its minimum (measured on real GEFS
    precipitation: one cell had 30% of its members exactly at zero, so the first bin came out a
    third short), and a stored distribution built from quantile levels or from a bounded bin
    support always puts *some* mass past the sample's maximum, because the reconstruction is
    smooth where the sample is not -- measured over 96 sampled ``(point, lead)`` pairs on real
    GEFS, the mass above the member grid's top edge is 1.7% of the member set at the median and
    14.5% at the 95th percentile. The first bin is therefore ``1 - P(X > edges[1])`` and the last
    is ``P(X > edges[-2])``: everything at or below the first edge belongs to the first bin, and
    everything at or above the last belongs to the last.

    That form needs one property either way: the first edge must sit at or below the
    distribution's lowest value, so that "at or below the first edge" is the same set as "in the
    first bin". :func:`shared_edges` has it by definition (its lower edge *is* the sample minimum)
    and :func:`stored_edges` by construction (it widens its own bounds).

    **The counts are apportioned rather than rounded independently.** Each bin's mass is a fraction
    of a member -- a ten-bin histogram of thirty members has 3 members per bin at most -- so
    rounding each bin on its own loses the remainders and the counts came out three short of the
    member set. Largest-remainder apportionment (give every bin its floor, then hand the leftover
    members to the largest remainders) makes the counts sum to the member count exactly, which is
    what lets the two lines be compared bin for bin at all.

    ``None`` when any probability is unavailable (a cell the store refused), because a histogram
    with holes in it is not a comparison of anything -- and when the masses cannot carry the
    member count, which is what :func:`_apportion` refuses.
    """
    if len(edges) < 2 or len(exceedance) != len(edges):
        return None
    if any(value is None or not math.isfinite(value) for value in exceedance):
        return None
    probabilities = [float(value) for value in exceedance if value is not None]
    if member_count <= 0:
        return None
    last = len(edges) - 2
    masses: list[float] = []
    for index in range(len(edges) - 1):
        if index == 0:
            masses.append(1.0 - probabilities[1])
        elif index == last:
            masses.append(probabilities[index])
        else:
            masses.append(probabilities[index] - probabilities[index + 1])
    # A grid whose lower edge is not the distribution's minimum is a caller error, and the negative
    # mass it produces would be worse than reading as one: said out loud rather than clipped
    # silently.
    if any(mass < -1e-9 for mass in masses):
        return None
    return _apportion([max(mass, 0.0) for mass in masses], member_count)


def _apportion(masses: list[float], member_count: int) -> list[int] | None:
    """Turn per-bin probabilities into integer counts that sum to ``member_count``.

    Largest remainder: every bin gets its floor, and the members left over go to the bins with the
    largest fractional parts. The alternative -- rounding each bin on its own -- is what produced a
    histogram three members short of its own member set, because every bin in a coarse histogram has
    a fractional part worth discarding.

    ``None`` when the masses cannot carry the member count at all -- they sum to less than the
    whole set, or to more than a bin per leftover can absorb. Callers reach that only by passing
    something other than a partition of the distribution: :func:`stored_histogram` builds its
    masses so they sum to exactly 1, which is what makes the leftover at most one member per bin.
    The guard is here rather than there because handing out one member per bin regardless is what
    turns an empty set of masses into a flat line of ones, which reads as a distribution rather
    than as an absence.
    """
    exact = [mass * member_count for mass in masses]
    counts = [int(math.floor(value)) for value in exact]
    remaining = member_count - sum(counts)
    if remaining < 0 or remaining > len(counts):
        return None
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
    "STORED_EDGE_MARGIN",
    "member_bin_count",
    "member_histogram",
    "shared_edges",
    "stored_edges",
    "stored_exceedance",
    "stored_histogram",
]
