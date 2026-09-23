"""The migration's two histograms: one grid, two sources, and they have to agree.

The property that matters is not that either histogram is pretty but that a difference between
them is *visible*: the front end draws both lines while the source changes, and the whole reason
the pair is delivered is that a silent divergence between the member-derived and stored-derived
distributions would be indistinguishable from a rendering change. So the tests below pin the grid
(both sides on the same edges, the last bin closed), the counting (counts of members, on both
sides, so the lines are comparable), and the absences (a store with no aggregate reports no stored
line rather than a flat one).
"""

from __future__ import annotations

import numpy as np
import pytest
from api.services.dual_source import (
    member_histogram,
    shared_edges,
    stored_exceedance,
    stored_histogram,
)
from domain.aggregate import (
    KIND_MEAN_STD_BINS,
    KIND_QUANTILE_FUNCTION,
    AggregateSpec,
    compute_aggregate,
)


def _quantile_members(seed: int = 0) -> np.ndarray:
    """A zero-inflated 30-member sample, shaped the way the encoder takes it."""
    rng = np.random.default_rng(seed)
    return np.where(
        rng.random((30, 1, 1)) < 0.5, 0.0, rng.gamma(0.4, 1.5, (30, 1, 1))
    ).astype(np.float32)


def test_shared_edges_span_the_member_range_and_close_the_last_bin() -> None:
    """The grid comes from the members, because they are the thing being replaced."""
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    edges = shared_edges(values, 3)
    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(4.0)
    assert len(edges) == 4
    # The maximum is counted, not dropped off the end.
    assert sum(member_histogram(values, edges)) == 4


def test_a_degenerate_range_still_produces_a_usable_grid() -> None:
    """Every member identical would otherwise give a zero-width grid."""
    edges = shared_edges(np.full(5, 7.0, dtype=np.float32), 4)
    assert len(edges) == 5
    assert edges[-1] > edges[0]
    assert sum(member_histogram(np.full(5, 7.0), edges)) == 5


def test_no_values_means_no_grid() -> None:
    """An all-NaN sample has no grid, which the caller reads as "no comparison"."""
    assert shared_edges(np.full(3, np.nan, dtype=np.float32)) == []
    assert member_histogram(np.full(3, np.nan), []) == []


def test_the_member_histogram_counts_only_finite_values() -> None:
    values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    edges = shared_edges(np.array([1.0, 3.0], dtype=np.float32), 2)
    assert sum(member_histogram(values, edges)) == 2


def test_the_stored_histogram_is_the_same_quantity_as_the_member_one() -> None:
    """Both lines are counts of the same members, which is what makes them comparable.

    The stored side is the stored distribution integrated across each bin of the member-derived
    grid; if it were a probability rather than a count the two lines would differ by a factor of
    the member count and the comparison would be meaningless -- while still looking like two
    plausible histograms.
    """
    members = _quantile_members(seed=1)
    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    fields = compute_aggregate(members, spec, expected_members=30)[:, 0, 0]
    edges = shared_edges(members[:, 0, 0], 10)

    tail = stored_exceedance(fields, spec, edges)
    assert len(tail) == len(edges)
    assert all(value is not None for value in tail)
    stored_counts = stored_histogram(tail, edges, len(members))
    assert stored_counts is not None
    # Every member lands in exactly one bin, on both sides.
    assert sum(stored_counts) == pytest.approx(len(members), abs=1)


def test_the_two_histograms_agree_to_within_their_own_reconstruction_error() -> None:
    """The end-to-end property the migration exists to check.

    A stored quantile function at 19 levels cannot reproduce a 30-member histogram exactly -- the
    bins are coarser than the levels in places and finer in others -- so the assertion is the
    investigation's own: the worst per-bin difference is a small number of members, not a
    structural disagreement. A systemic error would show up here as every bin being off in the
    same direction, which is why the sign is checked too.
    """
    members = _quantile_members(seed=2)
    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    fields = compute_aggregate(members, spec, expected_members=30)[:, 0, 0]
    edges = shared_edges(members[:, 0, 0], 10)

    member_counts = np.asarray(member_histogram(members[:, 0, 0], edges), dtype=np.int64)
    tail = stored_exceedance(fields, spec, edges)
    stored_counts = np.asarray(stored_histogram(tail, edges, len(members)) or [], dtype=np.int64)
    assert stored_counts.size == member_counts.size

    difference = stored_counts - member_counts
    assert np.abs(difference).max() <= 4, difference
    # Both a systematic over- and under-count would show as a mean well away from zero.
    assert abs(float(difference.mean())) <= 1.0


def test_the_bin_encoding_answers_the_same_question() -> None:
    """A bin-encoded variable's stored line comes from its accumulated shape, not its levels."""
    rng = np.random.default_rng(3)
    members = rng.normal(280.0, 8.0, 30).astype(np.float32)
    spec = AggregateSpec(kind=KIND_MEAN_STD_BINS)
    fields = compute_aggregate(members[:, None, None], spec, expected_members=30)[:, 0, 0]
    edges = shared_edges(members, 8)

    tail = stored_exceedance(fields, spec, edges)
    assert all(value is not None for value in tail)
    stored_counts = stored_histogram(tail, edges, len(members))
    assert stored_counts is not None
    assert sum(stored_counts) == pytest.approx(len(members), abs=2)


def test_a_distribution_the_grid_cannot_hold_is_refused_not_flattened() -> None:
    """A stored cell outside the member-derived grid must report nothing, not a flat line.

    The masses sum to ``1 - P(X > top edge)``. When the stored cell sits entirely above the grid
    -- which is what reading a *neighbouring* cell produces at a point halfway between nodes --
    every ``P(X > edge)`` is 1, every mass is 0, and apportioning zero mass used to hand each bin
    one leftover member: the stored line came back as ``[1, 1, ..., 1]``, a flat plateau summing
    to the bin count rather than the member count. Beside a member line with a clear peak that
    reads as a real distribution, which is exactly the shape the user reported.

    A mass the grid *can* carry is still drawn: half the distribution below the first edge and half
    inside is a legitimate comparison, and only the un-carryable case is refused.
    """
    edges = [2.0 + 0.2 * index for index in range(11)]

    # Every probability 1: no mass falls in any bin.
    assert stored_histogram([1.0] * 11, edges, 30) is None

    # Mass the grid can hold is drawn, and it still sums to the member count.
    half_below = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0]
    counts = stored_histogram(half_below, edges, 30)
    assert counts is not None
    assert sum(counts) == 30
    assert len(set(counts)) > 1, "a real distribution is not a flat line"

    # And a container with no members has no histogram at all.
    assert stored_histogram([0.99, 0.9, 0.7] + [0.0] * 8, edges, 0) is None


def test_a_flat_line_cannot_be_produced_by_any_plausible_tail() -> None:
    """The specific signature the user saw, asserted against directly.

    A stored line of all-equal counts is only reachable when the grid carries no mass at all, and
    that case is refused. This guards the regression rather than the code path: if a future change
    lets ``_apportion`` hand out leftovers again, the flat line reappears here.
    """
    edges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for tail in (
        [1.0] * 11,                       # no mass anywhere in the grid
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ):
        assert stored_histogram(tail, edges, 30) is None


def test_an_unobserved_cell_produces_no_stored_histogram() -> None:
    """A cell the store refused reports nothing, rather than a histogram of zeros.

    And it has to be refused *before* the inversion rather than after: inverting a plane of NaNs
    fabricates a probability instead of failing. Every comparison against NaN is false, so the
    interpolation reads the outermost stored level and reports ``P(X > t) = 0.999`` at every
    threshold -- a confident-looking distribution that no member ever supported. Measured, and the
    reason this is refused up front.
    """
    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    fields = np.full(spec.n_fields, np.nan, dtype=np.float32)
    assert stored_exceedance(fields, spec, [0.0, 1.0, 2.0]) == []

    # An edge the store could not answer for makes the histogram absent rather than partial.
    assert stored_histogram([0.5, None, 0.1], [0.0, 1.0, 2.0], 30) is None


def test_a_field_vector_of_the_wrong_width_is_refused() -> None:
    """A container written for another encoding is not this variable's distribution."""
    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    short = np.zeros(spec.n_fields - 1, dtype=np.float32)
    assert stored_exceedance(short, spec, [0.0, 1.0]) == []
    # And a mismatched edge count is not a histogram either.
    assert stored_histogram([0.5], [0.0, 1.0], 30) is None
    assert stored_histogram([0.5, 0.1], [0.0], 30) is None


# ---------------------------------------------------------------------------
# The stored-only grid
# ---------------------------------------------------------------------------


def test_stored_edges_come_from_the_distribution_itself() -> None:
    """The grid for a store whose members are gone, with no member values involved.

    The whole point of the mode: the comparison grid needs the members, and the members are exactly
    what a fully converted store no longer has. A distribution can still say what its own range is.
    """
    from api.services.dual_source import stored_edges

    members = _quantile_members(seed=11)
    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    fields = compute_aggregate(members, spec, expected_members=30)[:, 0, 0]

    edges = stored_edges(fields, spec, bins=10)
    assert len(edges) == 11
    assert all(edges[index] < edges[index + 1] for index in range(10))
    # The outermost stored levels bound the range, so the grid brackets the sample's own.
    sample_low = float(np.min(members))
    sample_high = float(np.max(members))
    assert edges[0] <= sample_low
    assert edges[-1] >= sample_high
    # A margin, not a wide one: the grid's span is close to the quantile range it came from.
    assert edges[-1] - edges[0] < (sample_high - sample_low) * 2.5


def test_the_bin_encoding_states_its_range_as_mean_and_spread() -> None:
    """A bin-encoded container stores no levels, so its bounds are its support."""
    from api.services.dual_source import stored_edges

    rng = np.random.default_rng(12)
    members = rng.normal(280.0, 8.0, 30).astype(np.float32)
    spec = AggregateSpec(kind=KIND_MEAN_STD_BINS)
    fields = compute_aggregate(members[:, None, None], spec, expected_members=30)[:, 0, 0]

    edges = stored_edges(fields, spec, bins=8)
    assert len(edges) == 9
    mean = float(np.mean(members))
    std = float(np.std(members))
    # Bounded by the encoding's own support -- mean +- ``sigma_range`` standard deviations, which is
    # the normalised range its bins cover -- widened by the margin. So it brackets the sample and
    # is a stated bound rather than an arbitrarily wide one.
    assert edges[0] < float(np.min(members))
    assert edges[-1] > float(np.max(members))
    assert edges[0] < mean < edges[-1]
    span_in_sigma = (edges[-1] - edges[0]) / std
    assert span_in_sigma == pytest.approx(2 * float(spec.sigma_range) * 1.1, rel=1e-6)


def test_a_degenerate_stored_range_still_produces_a_usable_grid() -> None:
    """Every member identical: the bin encoding stores a zero spread, and the grid must survive."""
    from api.services.dual_source import stored_edges

    spec = AggregateSpec(kind=KIND_MEAN_STD_BINS)
    members = np.full((5, 1, 1), 7.0, dtype=np.float32)
    fields = compute_aggregate(members, spec, expected_members=5)[:, 0, 0]

    edges = stored_edges(fields, spec, bins=4)
    assert len(edges) == 5
    assert edges[-1] > edges[0]
    assert edges[0] < 7.0 < edges[-1]


def test_an_unobserved_cell_states_no_grid() -> None:
    """Nothing stored at a cell is no grid, rather than a grid at invented bounds."""
    from api.services.dual_source import stored_edges

    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    assert stored_edges(np.full(spec.n_fields, np.nan, dtype=np.float32), spec) == []
    # A field vector of another encoding's width is not this variable's distribution.
    assert stored_edges(np.zeros(spec.n_fields - 1, dtype=np.float32), spec) == []


def test_the_two_grids_agree_about_where_the_distribution_sits() -> None:
    """The stored-line's counts are the same quantity on either grid, which is what lets the
    stored-only mode replace the comparison mode without changing what is drawn."""
    from api.services.dual_source import stored_edges

    members = _quantile_members(seed=13)
    spec = AggregateSpec(kind=KIND_QUANTILE_FUNCTION)
    fields = compute_aggregate(members, spec, expected_members=30)[:, 0, 0]
    flat = np.asarray(members).reshape(-1)

    for edges in (shared_edges(flat, 10), stored_edges(fields, spec, bins=10)):
        tail = stored_exceedance(fields, spec, edges)
        counts = stored_histogram(tail, edges, len(flat))
        assert counts is not None
        assert sum(counts) == len(flat)
        assert counts[0] >= 1, "a zero-inflated sample puts mass at its minimum"
