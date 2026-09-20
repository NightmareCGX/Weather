"""Acceptance tests for the product-defined fields.

Each function here has a scalar reference: the serving path already computes the same quantity
from the member set, member by member, and these fields exist so that path can stop reading
members. So the test is not "is this plausible" but "does this equal what the member path
produces", cell by cell, because a reader compares the stored product against the member-derived
one and a silent difference would look like an encoding error rather than a field one.

The rose is compared against ``domain.models.wind.compute_wind_rose``, the phase support and
transition frequencies against the per-member classifier and its two aggregates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from domain.aggregate import AggregateError
from domain.field_layout import aggregate_fields_for
from domain.models.cloud import (
    cloud_ceiling_ensemble_summary,
    cloud_cover_ensemble_summary,
)
from domain.models.precipitation import (
    PhysicalPhase,
    PrecipitationTransition,
    classify_precipitation_phase,
    compute_phase_weights,
    compute_transition_frequencies,
)
from domain.models.wind import CARDINAL_DIRECTIONS_8, compute_wind_rose
from domain.product_fields import (
    _INTERVAL_CODES,
    _SIGNATURE_DTYPE,
    _UNUSABLE_SIGNATURE,
    FLAG_NAMES,
    ROSE_BUCKETS,
    ROSE_SECTORS,
    PrecipitationGroupBuilder,
    _transition_signature,
    cloud_censoring_fields,
    fraction_of_members,
    phase_support_fields,
    rose_fields,
    transition_fields,
    variable_group_fields,
)

N_MEMBERS = 30
LAT, LON = 3, 4

_PHASES = tuple(PhysicalPhase)
_TRANSITIONS = tuple(PrecipitationTransition)


def _members(seed: int = 0):
    """A zero-inflated amount stack and four flag planes, as the decoder would produce them."""
    rng = np.random.default_rng(seed)
    amounts = rng.gamma(0.4, 1.5, (N_MEMBERS, LAT, LON)).astype(np.float32)
    amounts[rng.random((N_MEMBERS, LAT, LON)) < 0.4] = 0.0
    flags = np.stack(
        [(rng.random((N_MEMBERS, LAT, LON)) < 0.5).astype(np.float32) for _ in FLAG_NAMES]
    )
    return amounts, flags


def _reference_state(amounts, flags, member, row, col, *, amount_prev=None, flags_prev=None):
    """One member-cell through the scalar classifier, the way the serving path calls it."""
    def flag_dict(planes, m, r, c):
        return {name: int(planes[i, m, r, c] >= 0.5) for i, name in enumerate(FLAG_NAMES)}

    return classify_precipitation_phase(
        float(amounts[member, row, col]),
        flag_dict(flags, member, row, col),
        amount_prev=None if amount_prev is None else float(amount_prev[member, row, col]),
        flags_prev=None if flags_prev is None else flag_dict(flags_prev, member, row, col),
    )


# ---------------------------------------------------------------------------
# Phase support
# ---------------------------------------------------------------------------


def test_phase_support_matches_the_member_path_cell_by_cell() -> None:
    """The stored support must equal the mean of the members' own weights, not a re-derivation."""
    amounts, flags = _members(seed=1)
    fields = phase_support_fields(amounts, flags)
    assert fields.shape == (12, LAT, LON)

    for row in range(LAT):
        for col in range(LON):
            states = [
                _reference_state(amounts, flags, m, row, col) for m in range(N_MEMBERS)
            ]
            expected = [
                float(np.mean([compute_phase_weights(state)[phase] for state in states]))
                for phase in _PHASES
            ]
            assert fields[:6, row, col] == pytest.approx(expected, abs=1e-6), (row, col)
            # No predecessor supplied, so the previous planes are unobserved rather than zero:
            # a zero would claim every member's predecessor was dry.
            assert np.isnan(fields[6:, row, col]).all()


def test_phase_support_is_exact_at_the_cells_where_members_disagree() -> None:
    """A cell whose members occupy different phases is the case averaging *flags* would break.

    Member 0 is rain and member 1 is snow, both wet. Their mean support is half and half; the
    support of their mean flags is a single mixed phase. The stored field has to be the former.
    """
    amounts = np.full((2, 1, 1), 2.0, dtype=np.float32)
    flags = np.zeros((4, 2, 1, 1), dtype=np.float32)
    flags[0, 0] = 1.0  # member 0: rain
    flags[1, 1] = 1.0  # member 1: snow
    fields = phase_support_fields(amounts, flags)
    rain = _PHASES.index(PhysicalPhase.RAIN)
    snow = _PHASES.index(PhysicalPhase.SNOW)
    assert fields[rain, 0, 0] == pytest.approx(0.5)
    assert fields[snow, 0, 0] == pytest.approx(0.5)


def test_phase_support_reports_no_predecessor_as_unobserved() -> None:
    """An absent predecessor is not a dry one, and the field must not imply it was dry."""
    amounts, flags = _members(seed=2)
    previous_amounts, previous_flags = _members(seed=3)
    with_prev = phase_support_fields(
        amounts, flags, amounts_prev=previous_amounts, flags_prev=previous_flags
    )
    assert np.isfinite(with_prev[6:]).all()
    without = phase_support_fields(amounts, flags)
    assert np.isnan(without[6:]).all()


def test_a_cell_where_one_member_misses_a_flag_still_reports_the_rest() -> None:
    """One member's gap is a gap in that member's classification, not in the cell's.

    The uniform missing-member rule: the fractions divide by the members the cell could classify.
    Dividing by the member set instead would report a cell with one unusable member as 29/30 rain
    rather than as rain, which is a claim about a member nobody classified. A cell where *no*
    member is classifiable is NaN rather than a zero.
    """
    amounts, flags = _members(seed=4)
    flags[2, 5, 1, 2] = np.nan
    fields = phase_support_fields(amounts, flags)
    assert np.isfinite(fields[:6, 1, 2]).all()
    assert fields[:6, 1, 2].sum() == pytest.approx(1.0, abs=1e-6)
    # The absent predecessor is a different statement and stays absent at every cell.
    assert np.isnan(fields[6:, 1, 2]).all()
    assert np.isfinite(fields[:6, 0, 0]).all()


def test_a_cell_no_member_can_be_classified_at_is_absent() -> None:
    """NaN, not zero: no member had a phase, so the cell has not made a claim about one."""
    amounts, flags = _members(seed=6)
    flags[:, :, 1, 3] = np.nan
    fields = phase_support_fields(amounts, flags)
    assert np.isnan(fields[:, 1, 3]).all()
    assert np.isfinite(fields[:6, 0, 0]).all()


# ---------------------------------------------------------------------------
# Transition frequencies
# ---------------------------------------------------------------------------


def test_transition_fields_match_the_member_path_cell_by_cell() -> None:
    amounts, flags = _members(seed=5)
    fields = transition_fields(amounts, flags)
    assert fields.shape == (len(_TRANSITIONS), LAT, LON)

    for row in range(LAT):
        for col in range(LON):
            states = [
                _reference_state(amounts, flags, m, row, col) for m in range(N_MEMBERS)
            ]
            expected = compute_transition_frequencies(states)
            assert fields[:, row, col] == pytest.approx(
                [expected[t] for t in _TRANSITIONS], abs=1e-6
            ), (row, col)
            assert fields[:, row, col].sum() == pytest.approx(1.0, abs=1e-6)


def test_the_transition_group_carries_what_the_phase_group_cannot() -> None:
    """Two ensembles with identical mean phase support, different transition mixes.

    Ensemble A is one rain member and one snow member; B is two members each spanning both. Their
    mean support is the same half-and-half, and their transitions are not -- so a container
    holding only the phase group could not report B's ``mixed_transition``. This is the reason
    the transition group exists rather than being derived on read.
    """
    amounts = np.full((2, 1, 1), 2.0, dtype=np.float32)
    a_flags = np.zeros((4, 2, 1, 1), dtype=np.float32)
    a_flags[0, 0] = 1.0
    a_flags[1, 1] = 1.0
    b_flags = np.zeros((4, 2, 1, 1), dtype=np.float32)
    b_flags[0, :] = 1.0
    b_flags[1, :] = 1.0

    a_support = phase_support_fields(amounts, a_flags)
    b_support = phase_support_fields(amounts, b_flags)
    # Only the current planes are compared: the predecessor planes are NaN in both, which is
    # ``approx``'s own answer for "both undefined".
    assert a_support[:6, 0, 0] == pytest.approx(b_support[:6, 0, 0], abs=1e-6)

    a_transitions = transition_fields(amounts, a_flags)
    b_transitions = transition_fields(amounts, b_flags)
    assert not np.allclose(a_transitions[:, 0, 0], b_transitions[:, 0, 0])
    mixed = _TRANSITIONS.index(PrecipitationTransition.MIXED_TRANSITION)
    assert b_transitions[mixed, 0, 0] == pytest.approx(1.0)
    assert a_transitions[mixed, 0, 0] == pytest.approx(0.0)


def test_a_predecessor_changes_the_transition_but_not_the_current_support() -> None:
    """The predecessor only enters the transition, which is why it has its own planes."""
    amounts = np.full((1, 1, 1), 2.0, dtype=np.float32)
    flags = np.zeros((4, 1, 1, 1), dtype=np.float32)
    flags[0, 0] = 1.0  # rain, present

    wet_prev_amounts = np.full((1, 1, 1), 3.0, dtype=np.float32)
    wet_prev_flags = np.zeros((4, 1, 1, 1), dtype=np.float32)
    wet_prev_flags[1, 0] = 1.0  # snow, previous

    fresh = transition_fields(amounts, flags)
    continuing = transition_fields(
        amounts, flags, amounts_prev=wet_prev_amounts, flags_prev=wet_prev_flags
    )
    assert fresh[_TRANSITIONS.index(PrecipitationTransition.PERSISTENT_RAIN), 0, 0] == 1.0
    assert continuing[_TRANSITIONS.index(PrecipitationTransition.SNOW_TO_RAIN), 0, 0] == 1.0


# ---------------------------------------------------------------------------
# Fractions
# ---------------------------------------------------------------------------


def test_fraction_of_members_is_the_per_cell_mean_of_a_condition() -> None:
    condition = np.zeros((4, 2, 2), dtype=bool)
    condition[0, 0, 0] = True
    condition[1, 0, 0] = True
    condition[3, 1, 1] = True
    fractions = fraction_of_members(condition)
    assert fractions[0, 0] == pytest.approx(0.5)
    assert fractions[1, 1] == pytest.approx(0.25)
    assert fractions[0, 1] == pytest.approx(0.0)

    # With the finiteness mask the denominator is the members that exist rather than all four:
    # two members are finite at this cell and one of them satisfies the condition.
    finite = np.zeros((4, 2, 2), dtype=bool)
    finite[0, 0, 0] = True   # satisfies
    finite[2, 0, 0] = True   # does not
    masked = fraction_of_members(condition, finite=finite)
    assert masked[0, 0] == pytest.approx(0.5)
    # A cell with no finite member has no fraction, not a zero.
    assert np.isnan(masked[1, 1])


def test_fraction_of_members_reports_unobserved_cells_as_absent() -> None:
    """A cell the caller marks unobservable is NaN, not zero: zero is a claim, NaN is not."""
    condition = np.ones((4, 2, 1), dtype=bool)
    observable = np.array([[True], [False]])
    fractions = fraction_of_members(condition, observable=observable)
    assert fractions[0, 0] == pytest.approx(1.0)
    assert np.isnan(fractions[1, 0])


def test_fraction_of_members_rejects_a_non_member_stack() -> None:
    with pytest.raises(AggregateError, match="must be"):
        fraction_of_members(np.ones((2, 3), dtype=bool))
    with pytest.raises(AggregateError, match="same shape as condition"):
        fraction_of_members(
            np.ones((4, 2, 2), dtype=bool), finite=np.ones((4, 2, 3), dtype=bool)
        )


# ---------------------------------------------------------------------------
# Wind rose
# ---------------------------------------------------------------------------


def _uv(speed: float, direction_deg: float) -> tuple[float, float]:
    """u/v for a wind blowing *from* ``direction_deg`` (meteorological convention)."""
    radians = math.radians(direction_deg)
    return -speed * math.sin(radians), -speed * math.cos(radians)


def test_rose_sectors_match_the_member_path() -> None:
    """The sector assignment is the member path's own convention, checked on known bearings."""
    cases = [(5.0, 0.0), (5.0, 90.0), (5.0, 180.0), (5.0, 270.0), (5.0, 45.0), (20.0, 10.0)]
    pairs = [_uv(speed, direction) for speed, direction in cases]
    us = np.array([p[0] for p in pairs] + [0.0] * (N_MEMBERS - len(pairs)), dtype=np.float32)
    vs = np.array([p[1] for p in pairs] + [0.0] * (N_MEMBERS - len(pairs)), dtype=np.float32)

    rose, _, _ = rose_fields(us.reshape(-1, 1, 1), vs.reshape(-1, 1, 1))
    ours: dict[str, int] = {}
    for ordinal in range(ROSE_SECTORS * ROSE_BUCKETS):
        sector, _bucket = divmod(ordinal, ROSE_BUCKETS)
        count = int(round(float(rose[ordinal, 0, 0]) * N_MEMBERS))
        if count:
            name = CARDINAL_DIRECTIONS_8[sector]
            ours[name] = ours.get(name, 0) + count

    reference = compute_wind_rose(us, vs)
    theirs = {s.sector: s.count for s in reference.sectors if s.count}
    assert ours == theirs
    # Calm members are excluded from the rose, matching the member path's treatment.
    assert int(round(float(rose[:, 0, 0].sum()) * N_MEMBERS)) == N_MEMBERS - reference.calm_count


def test_rose_sums_to_the_non_calm_members_and_stays_within_one() -> None:
    rng = np.random.default_rng(7)
    u = rng.normal(3.0, 4.0, (N_MEMBERS, LAT, LON)).astype(np.float32)
    v = rng.normal(2.0, 4.0, (N_MEMBERS, LAT, LON)).astype(np.float32)
    rose, scalars, edges = rose_fields(u, v)
    assert rose.shape == (ROSE_SECTORS * ROSE_BUCKETS, LAT, LON)
    assert scalars.shape == (4, LAT, LON)
    assert edges.shape == (ROSE_BUCKETS + 1,)
    total = rose.sum(axis=0)
    assert (total <= 1.0 + 1e-6).all()
    assert (total >= 0.0).all()
    # The edges are non-decreasing, or a bin lookup could not be a search.
    assert (np.diff(edges) >= 0).all()


def test_rose_buckets_are_equal_frequency_so_no_bucket_is_starved() -> None:
    """Quantile edges spend the resolution where the members are.

    Fixed thresholds were measured to leave the top bucket empty on real data, because a global
    speed range says nothing about the local spread. Equal-frequency edges cannot do that: the
    member set of the grid fills each bucket as evenly as a discrete sample allows.
    """
    rng = np.random.default_rng(8)
    u = np.abs(rng.standard_t(2.5, (N_MEMBERS, LAT, LON))).astype(np.float32)
    v = np.abs(rng.standard_t(2.5, (N_MEMBERS, LAT, LON))).astype(np.float32)
    rose, _, edges = rose_fields(u, v)
    per_bucket = rose.reshape(ROSE_SECTORS, ROSE_BUCKETS, LAT, LON).sum(axis=(0, 2, 3))
    # Every bucket holds a share of the member mass: no bucket is empty across the whole grid.
    assert (per_bucket > 0).all(), per_bucket
    assert np.diff(edges)[0] > 0, "the first bucket must have width"


def test_rose_direction_is_sin_cos_so_no_angle_wraps() -> None:
    """Storing the angle would wrap at 0/360; the pair does not, and atan2 recovers it."""
    member_u = np.array([_uv(5.0, 359.0)[0], _uv(5.0, 1.0)[0]], dtype=np.float32)
    member_v = np.array([_uv(5.0, 359.0)[1], _uv(5.0, 1.0)[1]], dtype=np.float32)
    rose, scalars, _ = rose_fields(member_u.reshape(-1, 1, 1), member_v.reshape(-1, 1, 1))
    # Both members blow from the north, so the consensus is north too. Storing the *angle*
    # would average 359 and 1 into 180 -- the opposite bearing -- while the sin/cos pair keeps
    # the two bearings adjacent, and atan2 of the mean pair recovers north.
    # ``atan2(sin, cos)``, not ``atan2(-u, -v)``: the pair stored is the direction's own sine and
    # cosine, so recovering the bearing is a plain arctangent of the pair.
    direction = math.degrees(math.atan2(scalars[2, 0, 0], scalars[3, 0, 0])) % 360.0
    assert direction == pytest.approx(0.0, abs=2.0) or direction == pytest.approx(
        360.0, abs=2.0
    )
    assert float(np.hypot(scalars[2, 0, 0], scalars[3, 0, 0])) == pytest.approx(1.0, abs=1e-6)
    assert rose[:, 0, 0].sum() == pytest.approx(1.0, abs=1e-6)


def test_rose_scalars_are_absent_where_the_members_are() -> None:
    """A cell with no finite member has no consensus, and reports none rather than zero."""
    u = np.full((N_MEMBERS, 1, 2), 3.0, dtype=np.float32)
    v = np.full((N_MEMBERS, 1, 2), 1.0, dtype=np.float32)
    u[:, 0, 1] = np.nan
    rose, scalars, _ = rose_fields(u, v)
    assert np.isnan(scalars[:, 0, 1]).all()
    assert np.isnan(rose[:, 0, 1]).all()
    assert np.isfinite(scalars[:, 0, 0]).all()


def test_rose_rejects_mismatched_components() -> None:
    with pytest.raises(AggregateError, match="same shape"):
        rose_fields(
            np.zeros((4, 2, 2), dtype=np.float32), np.zeros((4, 2, 3), dtype=np.float32)
        )


def test_phase_fields_reject_mismatched_flags() -> None:
    amounts = np.zeros((4, 2, 2), dtype=np.float32)
    with pytest.raises(AggregateError, match="flags must be"):
        phase_support_fields(amounts, np.zeros((3, 4, 2, 2), dtype=np.float32))
    with pytest.raises(AggregateError, match="but amounts are"):
        phase_support_fields(amounts, np.zeros((4, 4, 2, 3), dtype=np.float32))


def test_a_degenerate_rose_still_produces_usable_edges() -> None:
    """Every member at the same speed collapses the quantile edges; they must stay searchable."""
    u = np.zeros((N_MEMBERS, 1, 1), dtype=np.float32)
    v = np.full((N_MEMBERS, 1, 1), 4.0, dtype=np.float32)
    rose, _, edges = rose_fields(u, v)
    assert edges[-1] > edges[0]
    assert rose[:, 0, 0].sum() == pytest.approx(1.0, abs=1e-6)


def test_a_malformed_flag_stack_is_refused() -> None:
    """The flag stack's leading axis is the four names, and a different count is a caller bug."""
    amounts = np.zeros((4, 2, 2), dtype=np.float32)
    with pytest.raises(AggregateError, match="flags must be"):
        phase_support_fields(amounts, np.zeros((2, 4, 2, 2), dtype=np.float32))
    with pytest.raises(AggregateError, match="flags must be"):
        transition_fields(amounts, np.zeros((4, 4, 2), dtype=np.float32))
    # Two dimensions is neither a stack nor one member's planes.
    with pytest.raises(AggregateError, match="flags must be"):
        phase_support_fields(amounts, np.zeros((4, 2), dtype=np.float32))
    # A predecessor interval is both its amounts and its flags, or neither: half of it would
    # silently classify against a predecessor the caller did not describe.
    flags = np.zeros((4, 4, 2, 2), dtype=np.float32)
    with pytest.raises(AggregateError, match="both its amounts and its flags"):
        phase_support_fields(amounts, flags, amounts_prev=amounts)
    with pytest.raises(AggregateError, match="both its amounts and its flags"):
        transition_fields(amounts, flags, flags_prev=flags)


# ---------------------------------------------------------------------------
# Cloud censoring
# ---------------------------------------------------------------------------


def _ceiling_members(seed: int = 11):
    """Thirty members with roughly 40% of them at the unlimited sentinel."""
    rng = np.random.default_rng(seed)
    unlimited = rng.random((N_MEMBERS, LAT, LON)) < 0.40
    return np.where(
        unlimited, 20.0, rng.uniform(0.2, 12.0, (N_MEMBERS, LAT, LON))
    ).astype(np.float32)


def test_ceiling_censoring_matches_the_member_summary_cell_by_cell() -> None:
    """The stored counts and conditional statistics must equal the member path's own summary.

    The counts are integers and are compared as such: the serving path reports
    ``valid_member_count`` as a count, and a count is what the coverage rule is evaluated against.
    """
    members = _ceiling_members(seed=12)
    fields = cloud_censoring_fields(members, variable="cloud_ceiling")
    assert fields.shape == (10, LAT, LON)

    for row in range(LAT):
        for col in range(LON):
            summary = cloud_ceiling_ensemble_summary(members[:, row, col])
            assert fields[0, row, col] == summary.valid_member_count
            assert fields[1, row, col] == summary.finite_member_count
            assert fields[2, row, col] == summary.unlimited_member_count
            # The three counts account for the whole member set, which is what makes them
            # readable as counts rather than as share-of-something.
            assert (
                fields[1, row, col] + fields[2, row, col] == fields[0, row, col]
            )
            expected = [
                summary.conditional_mean,
                summary.conditional_spread,
                *[
                    summary.conditional_percentiles[level]
                    for level in ("p10", "p25", "p50", "p75", "p90")
                ],
            ]
            assert fields[3:, row, col] == pytest.approx(expected, abs=1e-4), (
                row,
                col,
            )


def test_cover_censoring_matches_the_member_summary_and_has_no_unlimited_class() -> None:
    members = np.random.default_rng(13).uniform(0, 100, (N_MEMBERS, LAT, LON))
    # A tenth of the members are outside the physical range and must be excluded.
    members[np.random.default_rng(14).random((N_MEMBERS, LAT, LON)) < 0.1] = 120.0
    fields = cloud_censoring_fields(members.astype(np.float32), variable="cloud_cover_3h")
    assert (fields[2] == 0.0).all(), "cloud cover has no unlimited class"

    for row in range(LAT):
        for col in range(LON):
            summary = cloud_cover_ensemble_summary(members[:, row, col], min_valid=1)
            assert fields[0, row, col] == summary.valid_member_count
            assert fields[1, row, col] == fields[0, row, col]
            expected = [
                summary.mean,
                summary.spread,
                *[
                    summary.percentiles[level]
                    for level in ("p10", "p25", "p50", "p75", "p90")
                ],
            ]
            assert fields[3:, row, col] == pytest.approx(expected, abs=1e-3), (row, col)


def test_the_censoring_counts_are_stored_exactly_at_their_step() -> None:
    """A count is an integer, and its step is one member, so the round trip is exact.

    This is the property that makes the field representable at all: its group's step is 1.0, so
    anything that is not an integer -- a fraction of the member set, for instance -- quantises to
    zero or one and is lost. Measured through the real writer and reader before the fields became
    counts: a cell with 19 of 30 finite members decoded as ``FINITE_COUNT = 0.0``.
    """
    from domain.aggregate import dequantise_field, quantise_field
    from domain.field_layout import group_field_scales

    members = _ceiling_members(seed=15)
    fields = cloud_censoring_fields(members, variable="cloud_ceiling")
    scale = group_field_scales("censoring")[0]
    for index in range(3):
        stored = dequantise_field(quantise_field(fields[index], scale), scale)
        assert np.array_equal(stored, fields[index]), index


def test_a_cell_with_no_finite_member_has_no_conditional_statistics() -> None:
    """The conditional median needs the finite members: a mixture cannot supply it.

    The unlimited members are a point mass at the top of the range, so a percentile of the
    mixture says how much mass lies below it and nothing about the finite spread below that --
    which is why the conditional fields are stored rather than derived.
    """
    members = np.full((N_MEMBERS, 1, 2), 20.0, dtype=np.float32)
    members[:, 0, 1] = np.nan
    fields = cloud_censoring_fields(members, variable="cloud_ceiling")
    # The all-unlimited cell has counts but no finite member, so the conditionals are absent.
    assert fields[0, 0, 0] == N_MEMBERS
    assert fields[1, 0, 0] == 0
    assert fields[2, 0, 0] == N_MEMBERS
    assert np.isnan(fields[3:, 0, 0]).all()
    # The all-NaN cell has no valid member at all.
    assert fields[0, 0, 1] == 0
    assert np.isnan(fields[3:, 0, 1]).all()


def test_censoring_refuses_a_variable_with_no_rule() -> None:
    members = np.zeros((N_MEMBERS, 1, 1), dtype=np.float32)
    with pytest.raises(AggregateError, match="no censoring rule"):
        cloud_censoring_fields(members, variable="temperature_2m")
    with pytest.raises(AggregateError, match="must be"):
        cloud_censoring_fields(np.zeros((1, 1), dtype=np.float32), variable="cloud_ceiling")


# ---------------------------------------------------------------------------
# The writer's entry point
# ---------------------------------------------------------------------------


def test_group_fields_are_exactly_what_the_layout_declares() -> None:
    """Every variable's group fields match its container layout, field for field.

    The writer takes the field vector's length from the layout and its content from this
    function, so a disagreement is a container whose descriptor describes fields it does not
    hold -- the reader would then decode the wrong planes under the right names.
    """
    rng = np.random.default_rng(21)
    extra = {
        name: (rng.random((N_MEMBERS, LAT, LON)) < 0.5).astype(np.float32)
        for name in FLAG_NAMES
    }
    u = rng.normal(3.0, 4.0, (N_MEMBERS, LAT, LON)).astype(np.float32)
    v = rng.normal(2.0, 4.0, (N_MEMBERS, LAT, LON)).astype(np.float32)
    amounts = np.abs(rng.normal(1.0, 2.0, (N_MEMBERS, LAT, LON))).astype(np.float32)
    ceiling = np.where(
        rng.random((N_MEMBERS, LAT, LON)) < 0.4,
        20.0,
        rng.uniform(0.2, 12.0, (N_MEMBERS, LAT, LON)),
    ).astype(np.float32)
    flag_values = (rng.random((N_MEMBERS, LAT, LON)) < 0.3).astype(np.float32)

    cases = (
        ("temperature_2m", ceiling, None, 0),
        ("wind_10m", u, {"wind_u_10m": u, "wind_v_10m": v}, 77),
        ("precipitation_amount_3h", amounts, extra, 32),
        ("cloud_ceiling", ceiling, None, 10),
        ("cloud_cover_3h", np.abs(ceiling), None, 10),
        ("crain", flag_values, None, 1),
    )
    for variable, members, extras, expected in cases:
        fields = variable_group_fields(
            variable, members=members, extra_members=extras
        )
        layout = aggregate_fields_for(variable)
        assert fields.shape[0] == expected, variable
        # and the layout agrees: its fields are the count, the distribution, then these.
        non_distribution = layout.n_fields - 1 - (
            layout.distribution_slice.stop - layout.distribution_slice.start
        )
        assert fields.shape[0] == non_distribution, variable
        assert fields.shape[1:] == (LAT, LON)


def test_group_fields_are_absent_for_a_variable_with_no_groups() -> None:
    fields = variable_group_fields(
        "temperature_2m", members=np.zeros((N_MEMBERS, LAT, LON), dtype=np.float32)
    )
    assert fields.shape == (0, LAT, LON)


def test_a_group_missing_its_inputs_is_refused_rather_than_zero_filled() -> None:
    """A group built from nothing would publish zeros, which read as a definite answer."""
    members = np.zeros((N_MEMBERS, LAT, LON), dtype=np.float32)
    with pytest.raises(AggregateError, match="reads 'wind_u_10m'"):
        variable_group_fields("wind_10m", members=members)
    with pytest.raises(AggregateError, match="reads all four flags"):
        variable_group_fields("precipitation_amount_3h", members=members)
    # A partial flag stack is worse than none: the classifier would read the missing flag as
    # unset and classify against the wrong phase set.
    partial = {"crain": members, "csnow": members}
    with pytest.raises(AggregateError, match="reads all four flags"):
        variable_group_fields(
            "precipitation_amount_3h", members=members, extra_members=partial
        )
    with pytest.raises(AggregateError, match="no approved aggregate encoding"):
        variable_group_fields("mystery_variable", members=members)


def test_the_rose_edges_travel_as_constant_planes() -> None:
    """A reader needs the bucket edges to label a bucket or build a speed histogram."""
    rng = np.random.default_rng(22)
    u = rng.normal(3.0, 4.0, (N_MEMBERS, LAT, LON)).astype(np.float32)
    v = rng.normal(2.0, 4.0, (N_MEMBERS, LAT, LON)).astype(np.float32)
    fields = variable_group_fields(
        "wind_10m", members=u, extra_members={"wind_u_10m": u, "wind_v_10m": v}
    )
    edges = fields[68:77]
    # Constant across the grid, and non-decreasing.
    assert np.allclose(edges, edges[:, :1, :1], equal_nan=True)
    column = edges[:, 0, 0]
    assert (np.diff(column) >= 0).all()
    assert column[-1] > column[0]


def test_the_predecessor_interval_is_taken_whole_or_not_at_all() -> None:
    """A container built with a predecessor carries its phase planes; without one it does not.

    Supplying a partial predecessor -- the amount but not the flags -- is refused, because the
    classification would read the absent flags as unset and report a transition the members did
    not describe.
    """
    members = np.zeros((N_MEMBERS, LAT, LON), dtype=np.float32)
    flags = {name: members for name in FLAG_NAMES}

    without = variable_group_fields(
        "precipitation_amount_3h", members=members, extra_members=flags
    )
    assert np.isnan(without[6:12]).all(), "no predecessor means the previous planes are absent"

    with_prev = variable_group_fields(
        "precipitation_amount_3h",
        members=members,
        extra_members=flags,
        predecessor_members={**flags, "precipitation_amount_3h": members},
    )
    assert np.isfinite(with_prev[6:12]).all()

    # Half an interval is refused rather than treated as absent: the classification would
    # otherwise read the absent flags as unset and report a transition nobody described.
    with pytest.raises(AggregateError, match="part of a predecessor interval"):
        variable_group_fields(
            "precipitation_amount_3h",
            members=members,
            extra_members=flags,
            predecessor_members={"precipitation_amount_3h": members},
        )
    with pytest.raises(AggregateError, match="part of a predecessor interval"):
        variable_group_fields(
            "precipitation_amount_3h",
            members=members,
            extra_members=flags,
            predecessor_members={"crain": members},
        )


# ---------------------------------------------------------------------------
# The streamed precipitation builder
# ---------------------------------------------------------------------------


def test_streamed_precipitation_groups_equal_the_batched_ones() -> None:
    """The writer streams members; the batched path is the reference. They must agree exactly.

    Streaming is what keeps the pass inside its memory budget -- the groups read ten member
    planes, which as stacks is 1.25 GB at 721x1440 -- so the two implementations exist and have
    to produce the same fields, not merely close ones.
    """
    amounts, flags = _members(seed=31)
    previous_amounts, previous_flags = _members(seed=32)
    extra = {name: flags[i] for i, name in enumerate(FLAG_NAMES)}
    prev_extra = {name: previous_flags[i] for i, name in enumerate(FLAG_NAMES)}
    prev_extra["precipitation_amount_3h"] = previous_amounts

    batched = variable_group_fields(
        "precipitation_amount_3h",
        members=amounts,
        extra_members=extra,
        predecessor_members=prev_extra,
    )
    builder = PrecipitationGroupBuilder(expected_members=N_MEMBERS)
    for member in range(N_MEMBERS):
        builder.add_member(
            amounts=amounts[member],
            flags=np.stack([flags[i][member] for i in range(len(FLAG_NAMES))]),
            amounts_prev=previous_amounts[member],
            flags_prev=np.stack([previous_flags[i][member] for i in range(len(FLAG_NAMES))]),
        )
    assert builder.members_added == N_MEMBERS
    streamed = builder.finish()
    assert streamed.shape == batched.shape
    assert np.array_equal(streamed, batched, equal_nan=True)


def test_streamed_groups_report_an_absent_predecessor_as_absent() -> None:
    """A member with no predecessor contributes nothing, and the planes stay NaN."""
    amounts, flags = _members(seed=33)
    extra = {name: flags[i] for i, name in enumerate(FLAG_NAMES)}
    batched = variable_group_fields(
        "precipitation_amount_3h", members=amounts, extra_members=extra
    )
    builder = PrecipitationGroupBuilder(expected_members=N_MEMBERS)
    for member in range(N_MEMBERS):
        builder.add_member(
            amounts=amounts[member],
            flags=np.stack([flags[i][member] for i in range(len(FLAG_NAMES))]),
        )
    streamed = builder.finish()
    assert np.array_equal(streamed, batched, equal_nan=True)
    assert np.isnan(streamed[6:12]).all()


def test_a_streamed_builder_can_skip_the_transition_group() -> None:
    """A caller wanting only the phase support does not pay for the second accumulator."""
    amounts, flags = _members(seed=34)
    builder = PrecipitationGroupBuilder(
        expected_members=N_MEMBERS, with_transitions=False
    )
    for member in range(N_MEMBERS):
        builder.add_member(
            amounts=amounts[member],
            flags=np.stack([flags[i][member] for i in range(len(FLAG_NAMES))]),
        )
    fields = builder.finish()
    assert fields.shape[0] == 12


def test_a_streamed_builder_refuses_malformed_input_and_finishes_nothing_empty() -> None:
    amounts, flags = _members(seed=35)
    builder = PrecipitationGroupBuilder(expected_members=N_MEMBERS)
    with pytest.raises(AggregateError, match="no members were added"):
        builder.finish()
    with pytest.raises(AggregateError, match="flags must be"):
        builder.add_member(amounts=amounts[0], flags=flags[:3, 0])
    with pytest.raises(AggregateError, match="both its amounts and its flags"):
        builder.add_member(amounts=amounts[0], flags=flags[:, 0], amounts_prev=amounts[0])
    builder.add_member(amounts=amounts[0], flags=flags[:, 0])
    # A later member with a different grid extent would broadcast silently, so it is refused.
    with pytest.raises(AggregateError, match="expected"):
        builder.add_member(amounts=amounts[0, :1], flags=flags[:, 0, :1])


def test_a_streamed_builder_refuses_more_members_than_it_was_told_to_expect() -> None:
    """The record of which member set the fields describe is checked, not merely carried.

    A caller that configured the builder for 30 members and then fed it 31 has a bookkeeping
    error, and the fields would be published as a statistic of a set the caller named differently.
    """
    amounts, flags = _members(seed=36)
    builder = PrecipitationGroupBuilder(expected_members=2)
    for member in range(3):
        builder.add_member(amounts=amounts[member], flags=flags[:, member])
    with pytest.raises(AggregateError, match="told to expect"):
        builder.finish()


def test_a_streamed_builder_divides_by_the_members_each_cell_could_classify() -> None:
    """The masked denominator, checked against the batched path where a member is missing.

    A gap in one member at one cell must not erase that cell: the cell reports the phases of the
    members that had a value. And a member-cell outside the denominator must not be *summed*
    either -- its amount is non-finite, so the classifier reads it as dry, which would otherwise
    inflate another cell's fraction above one.
    """
    amounts, flags = _members(seed=37)
    amounts[4, 1, 2] = np.nan
    flags[3, 7, 0, 1] = np.nan
    extra = {name: flags[i] for i, name in enumerate(FLAG_NAMES)}
    batched = variable_group_fields(
        "precipitation_amount_3h", members=amounts, extra_members=extra
    )
    builder = PrecipitationGroupBuilder(expected_members=N_MEMBERS)
    for member in range(N_MEMBERS):
        builder.add_member(
            amounts=amounts[member],
            flags=np.stack([flags[i][member] for i in range(len(FLAG_NAMES))]),
        )
    streamed = builder.finish()
    assert np.array_equal(streamed, batched, equal_nan=True)
    # The cell one member is missing from still reports a full phase distribution.
    assert streamed[:6, 1, 2].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(streamed[:6, 1, 2]).all()


def test_a_signature_fits_the_dtype_the_builder_stores_it_in() -> None:
    """A wrapped signature would resolve a different table entry, not fail.

    The builder holds one signature plane per member per group, so the dtype is its residency:
    int16 rather than int64 is 62 MB against 250 MB per stack at 721x1440 for thirty members.
    That is only safe while every reachable signature fits, and the set is small and enumerable
    -- so it is enumerated rather than reasoned about.
    """
    widest = 0
    amounts = np.array([[[1.0]], [[1.0]]], dtype=np.float32)
    for current_bits in range(1 << len(FLAG_NAMES)):
        for predecessor_bits in range(1 << len(FLAG_NAMES)):
            bit_values = [
                float((current_bits >> bit) & 1) for bit in range(len(FLAG_NAMES))
            ] + [float((predecessor_bits >> bit) & 1) for bit in range(len(FLAG_NAMES))]
            flags = np.array(bit_values, dtype=np.float32).reshape(
                len(FLAG_NAMES), 2, 1, 1
            )
            signature = _transition_signature(
                amounts, flags, amounts[1:], flags[:, 1:]
            )
            widest = max(widest, int(signature.max()))
    # The dry side of each interval is the low end of the range, so the all-wet pair is the
    # widest case and nothing else can exceed it.
    assert widest == (_INTERVAL_CODES - 1) * (_INTERVAL_CODES + 1) + (_INTERVAL_CODES - 1)
    assert widest <= np.iinfo(np.int16).max
    # The reserved "cannot classify" code must be above every reachable signature, or a real
    # member-cell would be mistaken for an unusable one and silently dropped from the sums.
    assert widest < _UNUSABLE_SIGNATURE <= np.iinfo(_SIGNATURE_DTYPE).max
