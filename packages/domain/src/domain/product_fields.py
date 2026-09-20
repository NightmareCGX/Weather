"""The fields a variable's products need, computed from the member set per grid cell.

Every variable's container holds a per-cell member count and then two kinds of field: the
*distribution* (``domain.aggregate``'s business, one ordering per variable class) and the
*supplementary groups* this module computes. A group exists when a product the platform serves is
a function of the per-member values rather than of the distribution they collapse to, which is
why those fields have to be stored: an aggregate of 30 members describes their spread, not what
any one of them said.

Each group is computed here rather than at serve time for two reasons. It is the same arithmetic
either way, but doing it once per cycle instead of once per request is a cycle's cost against a
request's; and, more importantly, computing it here is what lets the member shards be deleted,
because afterwards there is no member set left to compute from.

What each group holds, and the measurement or argument that settled it, is recorded in
``domain.field_layout`` beside the declaration. This module is only the arithmetic.

Reference semantics are as load-bearing here as in ``domain.aggregate``: each function states the
convention it fixes, and anything that recomputes these fields must reproduce them exactly,
because the acceptance test compares the restored product against the member-derived one.

Vectorisation
-------------
The precipitation classifier is scalar and branches on a handful of discrete inputs -- the
current and predecessor amounts against the trace threshold, and each side's four flags. A
per-cell Python loop over a 721x1440 grid times thirty members is a million classifications per
field, so instead every member-cell is reduced to a small integer *signature* naming its case,
``np.unique`` collapses the distinct cases, and each distinct case is classified once by the same
scalar function the serving path uses. A signature that classifies one way for two different
inputs would be a bug in the signature, not a tolerated approximation, so the encoding is
asserted to be injective with respect to the classifier's inputs in the tests.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from domain.aggregate import AggregateError
from domain.field_layout import (
    FieldLayoutError,
    aggregate_fields_for,
    group_field_names,
)
from domain.models.cloud import CLOUD_CEILING_UNLIMITED_THRESHOLD_KM
from domain.models.precipitation import (
    PhysicalPhase,
    PrecipitationTransition,
    classify_precipitation_phase,
    compute_phase_weights,
)

#: Member axis first, then latitude, longitude.
_MEMBER_AXIS = 0

#: How many direction sectors and speed buckets the rose holds. Eight sectors is the cardinal
#: count every wind rose uses; eight buckets is the approved radial resolution, chosen so the
#: rose's marginal (the speed histogram the distribution chart draws) is as fine as the bins a
#: scalar variable stores.
ROSE_SECTORS: int = 8
ROSE_BUCKETS: int = 8

#: Sector geometry: eight equal sectors centred on the cardinal directions, so sector 0 spans
#: [337.5, 22.5). Matches ``compute_wind_rose``'s own ``floor((deg + 22.5) / 45) % 8``.
ROSE_SECTORS_WIDTH_DEG: float = 360.0 / ROSE_SECTORS
ROSE_SECTORS_HALF_WIDTH_DEG: float = ROSE_SECTORS_WIDTH_DEG / 2.0

#: The four categorical flags, in the order their planes are stacked.
FLAG_NAMES: tuple[str, ...] = ("crain", "csnow", "cfrzr", "cicep")

#: The conditional percentiles a bounded variable's container stores, in the layout's order.
_CONDITIONAL_PERCENTILES: tuple[float, ...] = (10.0, 25.0, 50.0, 75.0, 90.0)

#: A flag plane at or above this counts as set, matching the serving path's threshold.
FLAG_THRESHOLD: float = 0.5

#: Trace threshold for a dry interval, from ``domain.models.precipitation.TRACE_THRESHOLD_MM``.
#: Duplicated as a signature input only -- the classifier itself applies the same value.
_DRY_AMOUNT_MM: float = 0.10

_PHASES: tuple[PhysicalPhase, ...] = tuple(PhysicalPhase)
_TRANSITIONS: tuple[PrecipitationTransition, ...] = tuple(PrecipitationTransition)


def _counts_per_cell(members: npt.NDArray[np.floating]) -> npt.NDArray[np.int64]:
    """How many members are finite at each cell."""
    return np.isfinite(members).sum(axis=_MEMBER_AXIS).astype(np.int64)


def _check_flag_stack(
    flags: npt.NDArray[np.floating], amounts: npt.NDArray[np.floating] | None = None
) -> None:
    """Refuse a flag stack that is not the four named planes shaped like the amounts.

    Raises:
        AggregateError: on a wrong leading axis, a wrong member or grid extent, or a stack that
            is not 4-D. A mis-shaped stack would otherwise broadcast into a plausible-looking
            field over the wrong cells.
    """
    if flags.ndim != 4 or flags.shape[0] != len(FLAG_NAMES):
        raise AggregateError(
            f"flags must be ({len(FLAG_NAMES)}, n_members, lat, lon); got {flags.shape}"
        )
    if amounts is not None and flags.shape[1:] != amounts.shape:
        raise AggregateError(
            f"flags describe {flags.shape[1:]} but amounts are {amounts.shape}"
        )


def _check_single_flag_planes(
    flags: npt.NDArray[np.floating], shape: tuple[int, int]
) -> None:
    """Refuse a per-member flag stack that is not the four planes shaped like one cell grid.

    Checked per member rather than once, because a streamed caller supplies the planes member by
    member and a later member with a different extent would otherwise broadcast silently.

    Raises:
        AggregateError: on a wrong leading axis or a wrong grid extent.
    """
    if flags.ndim != 3 or flags.shape[0] != len(FLAG_NAMES) or flags.shape[1:] != shape:
        raise AggregateError(
            f"flags must be ({len(FLAG_NAMES)}, *grid) matching {shape} for one member; "
            f"got {flags.shape}"
        )


def _usable_members(
    amounts: npt.NDArray[np.floating],
    flags: npt.NDArray[np.floating],
    previous: npt.NDArray[np.floating] | None = None,
    previous_flags: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.bool_]:
    """Member-cells the classifier can read: a finite amount and all four finite flags.

    The phases and the transitions divide by the members each cell could classify, not by the
    member set. That is the platform's uniform missing-member rule -- skip the non-finite members,
    never let one member's gap erase a cell the others describe -- and the serving path already
    works that way: it hands its phase aggregate the members with a finite amount and divides by
    that count.

    It is also what makes the streamed and batched builders agree. The streaming builder is handed
    one member at a time and only learns a cell's usable count once the last member has been
    added, so a builder dividing by the member count would disagree with the batched path at every
    cell with a gap -- and the two are compared field for field by the acceptance test that admits
    the aggregate encoding.

    Args:
        amounts: ``(n_members, lat, lon)`` the interval's amount.
        flags: ``(4, n_members, lat, lon)`` the interval's flags.
        previous: Predecessor amounts, when the group being built reads them. A member without a
            finite predecessor cannot be classified into a transition, so it does not join that
            cell's denominator either.
        previous_flags: Predecessor flags, supplied together with ``previous``.

    Raises:
        AggregateError: if the flag stacks are not the four named planes shaped like the amounts.
    """
    _check_flag_stack(flags, amounts)
    usable = np.isfinite(amounts)
    for index in range(len(FLAG_NAMES)):
        usable &= np.isfinite(flags[index])
    if previous is not None:
        assert previous_flags is not None
        _check_flag_stack(previous_flags, previous)
        usable &= np.isfinite(previous)
        for index in range(len(FLAG_NAMES)):
            usable &= np.isfinite(previous_flags[index])
    return usable


def _fraction_of(
    predicate: npt.NDArray[np.bool_],
    observable: npt.NDArray[np.bool_],
    counts: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Per-cell fraction of members satisfying ``predicate``, restricted to observed cells."""
    total = predicate.sum(axis=_MEMBER_AXIS, dtype=np.float64)
    denominator = np.maximum(counts, 1).astype(np.float64)
    fraction = (total / denominator).astype(np.float32)
    return np.where(observable, fraction, np.float32(np.nan))


def fraction_of_members(
    condition: npt.NDArray[np.bool_],
    *,
    finite: npt.NDArray[np.bool_] | None = None,
    observable: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.float32]:
    """Per-cell fraction of the members satisfying ``condition``.

    The denominator is the member set, not the set that satisfied the condition -- the two are
    the same quantity only when the condition is always true, which is why the caller has to say
    which members exist. A NaN flag is an unknown rather than a "no", so a condition built by
    thresholding a plane must exclude it in both the numerator and the denominator; passing the
    finiteness mask as ``finite`` is how that is stated.

    Args:
        condition: ``(n_members, lat, lon)`` boolean member planes for "satisfies".
        finite: ``(n_members, lat, lon)`` boolean planes for "has a value"; defaults to every
            member-cell, which is right only when the inputs are known complete.
        observable: ``(lat, lon)`` cells to report; defaults to every cell with a member.

    Returns:
        ``(lat, lon)`` fractions in [0, 1], NaN where ``observable`` is false.

    Raises:
        AggregateError: if ``condition`` is not a 3-D member-first array, or ``finite`` does not
            have its shape.
    """
    if condition.ndim != 3:
        raise AggregateError(
            f"condition must be (n_members, lat, lon); got shape {condition.shape}"
        )
    if finite is None:
        counts = np.full(condition.shape[1:], condition.shape[0], dtype=np.int64)
    else:
        if finite.shape != condition.shape:
            raise AggregateError(
                f"finite must have the same shape as condition; got {finite.shape} "
                f"and {condition.shape}"
            )
        counts = finite.sum(axis=_MEMBER_AXIS).astype(np.int64)
        condition = condition & finite
    cells = observable if observable is not None else counts > 0
    return _fraction_of(condition, cells, counts)


def _flag_bits(flags: npt.NDArray[np.floating]) -> npt.NDArray[np.int32]:
    """Pack four flag planes into one integer per cell.

    Accepts either a whole member stack, ``(4, n_members, lat, lon)``, or one member's planes,
    ``(4, lat, lon)``, because both the batched and the streamed paths pack the same bits.

    Raises:
        AggregateError: if the leading axis is not the four named flags.
    """
    if flags.ndim not in (3, 4) or flags.shape[0] != len(FLAG_NAMES):
        raise AggregateError(  # pragma: no cover - both callers validate the shape first
            f"flags must lead with the {len(FLAG_NAMES)} named planes "
            f"({', '.join(FLAG_NAMES)}); got {flags.shape}"
        )
    bits = np.zeros(flags.shape[1:], dtype=np.int32)
    for index, _name in enumerate(FLAG_NAMES):
        plane = flags[index]
        set_here = np.isfinite(plane) & (plane >= FLAG_THRESHOLD)
        bits |= set_here.astype(np.int32) << index
    return bits


def _dry(amounts: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    """Whether an amount is at or below the trace threshold, or not finite."""
    return ~(np.isfinite(amounts) & (amounts > _DRY_AMOUNT_MM))


def _phase_signature(
    amounts: npt.NDArray[np.floating], flags: npt.NDArray[np.floating]
) -> npt.NDArray[np.int64]:
    """Signature naming every input ``compute_phase_weights`` reads for one member-cell.

    ``compute_phase_weights`` reads only the member's own state: its amount against the trace
    threshold, its interval type, and the phases its four flags select. The interval type follows
    from "dry" and "no flags set" (an unknown wet phase), and the phases are the flag bits, so
    those two inputs are the whole input. Sixteen bit patterns times dry/wet is 32 signatures.
    """
    return (_dry(amounts).astype(np.int64) << 4) | _flag_bits(flags).astype(np.int64)


#: Signature ranges for one interval, kept disjoint so the combination stays injective.
#: An interval the classifier sees as dry ignores its flags entirely (it is dry whatever they
#: say), so its signature drops them; a wet interval's flags are what select its phases.
_DRY_CODE = 0
_WET_CODE_BASE = 1

#: Signature reserved for a member-cell this group cannot classify, so it can be folded into the
#: accumulation's own index space instead of masked out of the sum. A transition signature is at
#: most ``(_INTERVAL_CODES - 1) * (_INTERVAL_CODES + 1) + (_INTERVAL_CODES - 1)`` = 304, so 1023
#: is unreachable and staying inside int16 keeps the planes 62 MB each rather than 250 MB.
_UNUSABLE_SIGNATURE: int = 1023

#: The stored dtype of a signature plane, chosen because the streaming builder holds one per
#: member per group and that is its whole residency. A transition signature is
#: ``current * (_INTERVAL_CODES + 1) + predecessor`` with each side in ``0.._INTERVAL_CODES - 1``,
#: so the widest value is 304 -- well inside int16 -- while three such stacks at 721x1440 for
#: thirty members are 62 MB each in int16 against 250 MB each in int64. Measured: the builder
#: holds 292 MB after thirty members. The bound is asserted by a test rather than assumed, because
#: a wrap would resolve the wrong table entry and produce a plausible field instead of a failure,
#: and the test also pins that the reserved code is above every reachable signature.
_SIGNATURE_DTYPE: npt.DTypeLike = np.int16


def _interval_code(
    amounts: npt.NDArray[np.floating], flags: npt.NDArray[np.floating]
) -> npt.NDArray[np.int64]:
    """Signature of one interval: dry, or wet with the flag bits that select its phases."""
    dry = _dry(amounts)
    return np.where(
        dry, _DRY_CODE, _WET_CODE_BASE + _flag_bits(flags).astype(np.int64)
    )


#: A predecessor interval the aggregate cannot supply at all. Distinct from a dry one because
#: the classifier reports ``persistent_rain`` for an absent predecessor where a dry predecessor
#: gives ``dry_to_rain``, and the serving path reaches both -- a missing predecessor object is
#: not the same as one reporting no rain.
_PREDECESSOR_ABSENT: int = 0

#: How many codes one interval can take: dry, then wet with any of the sixteen flag patterns.
_INTERVAL_CODES: int = 1 + (1 << len(FLAG_NAMES))


def _transition_signature(
    amounts: npt.NDArray[np.floating],
    flags: npt.NDArray[np.floating],
    amounts_prev: npt.NDArray[np.floating] | None,
    flags_prev: npt.NDArray[np.floating] | None,
) -> npt.NDArray[np.int64]:
    """Signature naming every input the transition classification reads for one member-cell."""
    current = _interval_code(amounts, flags)
    if amounts_prev is None or flags_prev is None:
        predecessor = np.full(amounts.shape, _PREDECESSOR_ABSENT, dtype=np.int64)
    else:
        predecessor = np.where(
            np.isfinite(amounts_prev),
            _interval_code(amounts_prev, flags_prev),
            _PREDECESSOR_ABSENT,
        )
    return current * (_INTERVAL_CODES + 1) + predecessor


def _weights_for_signature(signature: int) -> tuple[float, ...]:
    """Phase weights of the state a phase signature names.

    The signature's low four bits are the flags and its fifth says dry, which is everything
    :func:`classify_precipitation_phase` reads from one interval.
    """
    dry = bool((signature >> 4) & 1)
    bits = signature & 0b1111
    flags = {
        name: int(bool((bits >> index) & 1)) for index, name in enumerate(FLAG_NAMES)
    }
    amount = None if dry else 1.0
    state = classify_precipitation_phase(amount, flags if not dry else None)
    weights = compute_phase_weights(state)
    return tuple(weights[phase] for phase in _PHASES)


def _one_hot(index: int, size: int) -> tuple[float, ...]:
    """A one-hot row, so a categorical resolution accumulates through the same scatter."""
    row = [0.0] * size
    row[index] = 1.0
    return tuple(row)


def _flags_of(bits: int) -> dict[str, int]:
    """The four flags a bit pattern names."""
    return {name: int(bool((bits >> index) & 1)) for index, name in enumerate(FLAG_NAMES)}


def _transition_for_signature(signature: int) -> tuple[float, ...]:
    """One-hot row of the transition category the signature names."""
    predecessor = signature % (_INTERVAL_CODES + 1)
    current = signature // (_INTERVAL_CODES + 1)
    prev_present = predecessor != _PREDECESSOR_ABSENT

    curr_wet = current != _DRY_CODE
    prev_wet = prev_present and predecessor != _DRY_CODE
    state = classify_precipitation_phase(
        None if not curr_wet else 1.0,
        None if not curr_wet else _flags_of(current - _WET_CODE_BASE),
        amount_prev=(None if not prev_wet else 1.0),
        flags_prev=(None if not prev_wet else _flags_of(predecessor - _WET_CODE_BASE)),
    )
    return _one_hot(_TRANSITIONS.index(state.transition), len(_TRANSITIONS))


def _accumulate_planes(
    accumulator: npt.NDArray[np.float32],
    value_planes: npt.NDArray[np.float32],
) -> None:
    """Add one member's value planes, ``(lat, lon, n_outputs)``, into the accumulator in place.

    In place so a caller looping over members holds only one member's planes at a time. Member-cells
    this group's denominator does not count have already been folded into the zero row by the
    caller, so there is nothing to mask here.
    """
    accumulator += value_planes


def _finish_fractions(
    accumulator: npt.NDArray[np.float32],
    counts: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Turn per-cell member counts into fractions.

    A cell whose count is zero reports NaN rather than zero: zero is a claim about the phases, and
    a cell no member described has not made one.
    """
    denominator = np.maximum(counts, 1).astype(np.float32)[:, :, None]
    out = (accumulator / denominator).transpose(2, 0, 1)
    return np.where(counts > 0, out, np.float32(np.nan))


def _accumulate_by_signature(
    signature: npt.NDArray[np.int64],
    usable: npt.NDArray[np.bool_],
    *,
    n_outputs: int,
    resolve: Callable[[int], tuple[float, ...]],
) -> npt.NDArray[np.float32]:
    """Accumulate a per-signature value vector into per-cell fractions, one member at a time.

    A thin wrapper over :func:`_accumulate_signatures`, which owns the arithmetic, so the batched
    and streamed paths cannot diverge: the batched caller hands over a whole stack and the streamed
    one the planes it collected. The gathered alternative to both -- materialising
    ``(n_members, lat, lon, n_outputs)`` to sum over the member axis -- is 3.0 GB for the phase
    group and 5.0 GB for the transition group at 721x1440, which is not a shape the container can
    take; it works on the test grids and would fail on the first real lead.
    """
    return _accumulate_signatures(
        list(signature), list(usable), n_outputs=n_outputs, resolve=resolve
    )


def phase_support_fields(
    amounts: npt.NDArray[np.floating],
    flags: npt.NDArray[np.floating],
    *,
    amounts_prev: npt.NDArray[np.floating] | None = None,
    flags_prev: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.float32]:
    """Current and predecessor phase support, six planes each.

    The *weights* are stored rather than the flag fractions because the weights are what the
    product displays (``phase_support``), so the stored field is the shown quantity rather than a
    re-derivation of it.

    The predecessor planes duplicate what lead ``L-3``'s own container holds as its current
    planes. That redundancy is deliberate: it makes a lead's product answerable from that lead's
    object alone, rather than requiring a second read whose target may itself have been
    reclaimed, and it is six fields.

    Args:
        amounts: ``(n_members, lat, lon)`` current-interval precipitation.
        flags: ``(4, n_members, lat, lon)`` current flags.
        amounts_prev: Predecessor amounts, or ``None`` when there is none.
        flags_prev: Predecessor flags, or ``None``.

    Returns:
        ``(12, lat, lon)``: the six current phases then the six predecessor phases, in
        :class:`PhysicalPhase` order, as fractions of the member set.

    The fractions divide by the members each cell could classify -- a finite amount and all four
    finite flags -- not by the member set. One member's gap at one cell is a gap in that cell's
    classification and nothing more, and a cell no member could classify is NaN rather than a
    zero that would read as a definite phase. That is the platform's uniform missing-member rule
    and what the serving path's own phase aggregate does, and it is what lets the streamed
    builder (:class:`PrecipitationGroupBuilder`) reproduce these fields exactly.
    """
    _check_flag_stack(flags, amounts)
    if (amounts_prev is None) != (flags_prev is None):
        raise AggregateError(
            "a predecessor interval needs both its amounts and its flags, or neither"
        )
    usable = _usable_members(amounts, flags)
    current = _accumulate_by_signature(
        _phase_signature(amounts, flags),
        usable,
        n_outputs=len(_PHASES),
        resolve=_weights_for_signature,
    )
    if amounts_prev is None or flags_prev is None:
        previous = np.full_like(current, np.nan)
    else:
        previous = _accumulate_by_signature(
            _phase_signature(amounts_prev, flags_prev),
            _usable_members(amounts_prev, flags_prev),
            n_outputs=len(_PHASES),
            resolve=_weights_for_signature,
        )
    return np.concatenate([current, previous], axis=0)


def transition_fields(
    amounts: npt.NDArray[np.floating],
    flags: npt.NDArray[np.floating],
    *,
    amounts_prev: npt.NDArray[np.floating] | None = None,
    flags_prev: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.float32]:
    """Twenty per-cell transition-category frequencies.

    Stored rather than derived from :func:`phase_support_fields`, and the reason is not the
    predecessor's presence: normalised phase weights discard the phase *set*, so two members at
    rain and at snow average to the same support as two members each spanning both, while their
    transitions differ -- ``persistent_rain`` and ``persistent_snow`` against
    ``mixed_transition``. The transition is a function of the set, so a mean weight cannot
    recover it.

    Returns:
        ``(20, lat, lon)`` frequencies in :class:`PrecipitationTransition` order.

    The denominator is the members each cell could classify into a transition -- as
    :func:`phase_support_fields`, and additionally requiring a finite predecessor when one is
    supplied, since a member without a predecessor has no transition to count.
    """
    _check_flag_stack(flags, amounts)
    if (amounts_prev is None) != (flags_prev is None):
        raise AggregateError(
            "a predecessor interval needs both its amounts and its flags, or neither"
        )
    return _accumulate_by_signature(
        _transition_signature(amounts, flags, amounts_prev, flags_prev),
        _usable_members(amounts, flags, amounts_prev, flags_prev),
        n_outputs=len(_TRANSITIONS),
        resolve=_transition_for_signature,
    )


def rose_fields(
    u_members: npt.NDArray[np.floating],
    v_members: npt.NDArray[np.floating],
    *,
    bucket_edges: npt.NDArray[np.floating] | None = None,
    calm_threshold: float = 0.5,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Wind rose planes, consensus scalars, and the bucket edges they were built with.

    The rose is 8 sectors x 8 speed buckets of *member counts* as fractions, laid out
    sector-major so a bucket index is ``sector * ROSE_BUCKETS + bucket``. Its marginal over
    sectors is the speed histogram the distribution chart draws, so a wind variable needs no
    separate distribution fields -- the rose is one.

    Buckets are **quantile** edges over the member set of the whole grid, not fixed thresholds.
    Fixed edges were measured to put almost every member in the lowest two of four buckets (the
    gale bucket was empty), because a gust field's global range says nothing about the local
    spread; equal-frequency edges resolve the same number of fields and spend them where the
    members actually are. The edges travel with the fields because they are per ``(variable,
    lead)``: without them a reader cannot label a bucket or reconstruct a speed histogram.

    Args:
        u_members: ``(n_members, lat, lon)`` u components.
        v_members: ``(n_members, lat, lon)`` v components.
        bucket_edges: Edges to bin with, or ``None`` to derive them from the member speeds.
        calm_threshold: Speed below which a member is calm and its direction undefined.

    Returns:
        ``(rose, scalars, edges)``: the rose as ``(64, lat, lon)``, the four consensus planes as
        ``(4, lat, lon)`` in the layout's order (speed, coherence, direction sin, direction cos),
        and the ``(ROSE_BUCKETS + 1,)`` edge array used.

    Raises:
        AggregateError: if the two components do not have the same shape.
    """
    if u_members.shape != v_members.shape:
        raise AggregateError(
            f"u and v must have the same shape; got {u_members.shape} and {v_members.shape}"
        )
    speed = np.hypot(u_members, v_members)
    counts = _counts_per_cell(speed)
    observable = (
        np.isfinite(u_members).all(axis=_MEMBER_AXIS)
        & np.isfinite(v_members).all(axis=_MEMBER_AXIS)
    )

    if bucket_edges is None:
        finite_speeds = speed[np.isfinite(speed)]
        quantiles = np.linspace(0.0, 1.0, ROSE_BUCKETS + 1)
        bucket_edges = (
            np.quantile(finite_speeds, quantiles).astype(np.float32)
            if finite_speeds.size
            else np.zeros(ROSE_BUCKETS + 1, dtype=np.float32)
        )
        # A degenerate field (every member the same speed) collapses the edges; widening the top
        # by one step keeps every member in the last bucket rather than failing the search.
        if bucket_edges[-1] <= bucket_edges[0]:
            bucket_edges = bucket_edges.copy()
            bucket_edges[-1] = bucket_edges[0] + np.float32(1.0)

    calm = ~np.isfinite(speed) | (speed < calm_threshold)
    rose = _rose_fractions(
        speed,
        u_members,
        v_members,
        bucket_edges,
        observable[None] & ~calm,
        counts,
        observable,
    )
    del calm

    mean_u = _mean_over_members(u_members, counts, observable)
    mean_v = _mean_over_members(v_members, counts, observable)
    consensus_speed = np.hypot(mean_u, mean_v).astype(np.float32)
    mean_speed = _mean_over_members(speed, counts, observable)
    coherence = np.divide(
        consensus_speed,
        mean_speed,
        out=np.zeros_like(consensus_speed),
        where=mean_speed > 0,
    ).astype(np.float32)
    # Direction as sin/cos rather than degrees: an angle near 0/360 is discontinuous and a
    # fixed-point angle field would need a range of 360/step, which at any useful resolution
    # exceeds int16. The pair is continuous and bounded, and atan2 recovers the angle.
    consensus_direction = np.degrees(np.arctan2(-mean_u, -mean_v)) % 360.0
    direction_sin = np.sin(np.radians(consensus_direction)).astype(np.float32)
    direction_cos = np.cos(np.radians(consensus_direction)).astype(np.float32)
    scalars = np.stack(
        [consensus_speed, coherence, direction_sin, direction_cos]
    ).astype(np.float32)
    scalars = np.where(observable[None], scalars, np.float32(np.nan))
    return rose, scalars, bucket_edges.astype(np.float32)


def _rose_fractions(
    speed: npt.NDArray[np.floating],
    u_members: npt.NDArray[np.floating],
    v_members: npt.NDArray[np.floating],
    bucket_edges: npt.NDArray[np.floating],
    live: npt.NDArray[np.bool_],
    counts: npt.NDArray[np.int64],
    observable: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    """Per-cell member fractions in each ``(sector, bucket)``.

    Accumulated one member at a time rather than with a single ``np.bincount`` over every
    member-cell. The gathered form has to name each live member-cell's ``(combo, cell)`` pair in
    one array, which at 721x1440 x 30 members is 249 MB of indices, and has to accumulate in
    int64 across all 64 combos at once -- 531 MB. Measured at the production grid (30 members,
    721x1440, 60% of members live): the per-member form peaks **749 MB above the member stacks
    against 1959 MB** for the gathered one, in the same wall time (~3.1 s against ~3.2 s). Both
    figures are what the builder's residency budget is spent on, and the gathered one is more
    than the sum of every other group's working set.

    Within one member each grid cell carries exactly one sector and one bucket, so every index in
    a member's scatter is distinct and the increment needs no unbuffered accumulation: the counts
    are exact either way, and a member's own contribution to any cell is at most one. Verified
    bit-identical against the gathered implementation on a 30 x 721 x 1440 member set.

    The counts fit int16 (a cell holds at most one value per member), and the division is done in
    float64 over the accumulator's own dtype -- one slice per combo, so its temporary is a
    quarter-megabyte rather than the whole rose.

    A calm member's direction is undefined (``atan2`` of a zero vector is 0), so it is assigned a
    sector but excluded by ``live`` -- the same treatment the member path gives it.
    """
    n_members, lat, lon = speed.shape
    cells = np.arange(lat * lon, dtype=np.int32)
    one = np.int16(1)
    totals = np.zeros(ROSE_SECTORS * ROSE_BUCKETS * lat * lon, dtype=np.int16)

    for member in range(n_members):
        with np.errstate(invalid="ignore"):
            direction = np.degrees(np.arctan2(-u_members[member], -v_members[member])) % 360.0
        direction = np.where(np.isfinite(direction), direction, 0.0)
        sector = np.mod(
            np.floor((direction + ROSE_SECTORS_HALF_WIDTH_DEG) / ROSE_SECTORS_WIDTH_DEG).astype(
                np.int32
            ),
            ROSE_SECTORS,
        )
        # Members beyond the last edge fall in the top bucket rather than off the scale.
        bucket = np.clip(
            np.searchsorted(bucket_edges, speed[member], side="right") - 1,
            0,
            ROSE_BUCKETS - 1,
        ).astype(np.int32)
        combined = (sector * ROSE_BUCKETS + bucket).ravel()
        combined *= np.int32(lat * lon)
        combined += cells
        np.add.at(totals, combined[live[member].ravel()], one)

    denominator = np.maximum(counts, 1).astype(np.float64).reshape(lat * lon)
    rose = np.empty((ROSE_SECTORS * ROSE_BUCKETS, lat * lon), dtype=np.float32)
    totals_by_combo = totals.reshape(ROSE_SECTORS * ROSE_BUCKETS, lat * lon)
    for combo in range(ROSE_SECTORS * ROSE_BUCKETS):
        rose[combo] = (totals_by_combo[combo].astype(np.float64) / denominator).astype(
            np.float32
        )
    del totals, totals_by_combo
    # Marked in place rather than through ``np.where``, which would hold a second copy of a
    # 266 MB field at the peak of an already field-heavy computation for an identical result.
    rose[:, ~observable.ravel()] = np.float32(np.nan)
    return rose.reshape(ROSE_SECTORS * ROSE_BUCKETS, lat, lon)


def _conditional_percentiles(
    masked: npt.NDArray[np.float32],
    counts: npt.NDArray[np.int64],
) -> list[npt.NDArray[np.float32]]:
    """P10, P25, P50, P75 and P90 over the finite members of each cell.

    One sort of the masked stack and a vectorised gather per level, rather than five
    ``np.nanpercentile`` calls. The convention is the one the distribution's quantile encoding
    uses (``domain.aggregate``): linear interpolation at position ``p * (k - 1)`` for that cell's
    finite count ``k``. That is also what the member path's ``np.percentile(..., method="linear")``
    implements, and the two spellings differ by at most **2 float32 ulps** (measured against
    ``np.nanpercentile`` on random stacks with NaN gaps; the NaN pattern is identical). That
    difference cannot survive storage: these fields are written at a 0.01 step, four orders of
    magnitude coarser.

    What it costs to not do this: measured on a 30 x 721 x 1440 masked stack, five
    ``np.nanpercentile`` calls take **202 s** against **0.45 s** here. The NaN-aware form
    re-partitions the whole stack five times, and it does so per (variable, lead) -- so the cost
    is on the aggregate pass's critical path, not a micro-optimisation.

    NaN sorts last, and every gather index is clamped to ``min(lower + 1, k - 1)``, so the
    non-finite tail is never selected; the sort can therefore run on the masked array in place.
    """
    ordered = masked
    ordered.sort(axis=0)
    span = np.maximum(counts - 1, 0)
    lat_idx, lon_idx = np.indices(ordered.shape[1:])
    out: list[npt.NDArray[np.float32]] = []
    for level in _CONDITIONAL_PERCENTILES:
        position = (level / 100.0) * span
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower + 1, span)
        # Both weights are rounded to float32 individually, matching the scalar spelling this
        # replaces: there each weight was a Python float, which NumPy applies in float32.
        fraction = position - lower
        weight_high = fraction.astype(np.float32)
        weight_low = (1.0 - fraction).astype(np.float32)
        low_vals = ordered[lower, lat_idx, lon_idx]
        high_vals = ordered[upper, lat_idx, lon_idx]
        out.append((low_vals * weight_low + high_vals * weight_high).astype(np.float32))
    return out


def _mean_over_members(
    values: npt.NDArray[np.floating],
    counts: npt.NDArray[np.int64],
    observable: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    """Per-cell mean over members, NaN where the cell is not observable.

    A plain sum, not ``nansum``: the mask already guarantees every member is finite where the
    cell is observable, so ignoring NaNs would only hide a cell that should have been excluded --
    and would compute a mean over a different member set than the distribution beside it.
    """
    total = np.sum(values, axis=_MEMBER_AXIS, dtype=np.float64)
    denominator = np.maximum(counts, 1).astype(np.float64)
    out = (total / denominator).astype(np.float32)
    return np.where(observable, out, np.float32(np.nan))


__all__ = [
    "FLAG_NAMES",
    "PrecipitationGroupBuilder",
    "variable_group_fields",
    "cloud_censoring_fields",
    "FLAG_THRESHOLD",
    "ROSE_BUCKETS",
    "ROSE_SECTORS",
    "fraction_of_members",
    "phase_support_fields",
    "rose_fields",
    "transition_fields",
]


def cloud_censoring_fields(
    members: npt.NDArray[np.floating],
    *,
    variable: str,
) -> npt.NDArray[np.float32]:
    """The three counts a cloud variable takes before summarising, and its conditional statistics.

    Returns ``(4, lat, lon)`` for the censoring group (valid, finite, unlimited counts as
    fractions of the member set) followed by the seven conditional fields (mean, spread and the
    five percentiles, over the finite members).

    The counts are stored as **fractions of the member set** rather than as member counts, for the
    same reason the member-count field is a number: a fraction composes with the count field, so a
    reader multiplying the two recovers the count, while a count alone cannot be sanity-checked
    against the member set it came from. ``valid_count`` is the quantity the API reports as
    ``valid_member_count``, and it is what the serving coverage rule reads once the members are
    gone -- an integer count would also do, but the fraction needs no second field to interpret.

    The conditional statistics are stored because they cannot be recovered from a mixture. The
    unlimited members of a ceiling field sit at one value (the 20 km sentinel), so a percentile of
    the mixture says how much mass lies below it and nothing about how the finite mass is spread
    below that; recovering the conditional median needs the finite members themselves.

    Args:
        members: ``(n_members, lat, lon)`` the variable's member values.
        variable: ``"cloud_ceiling"`` or ``"cloud_cover_3h"``, which decide the censoring rule.

    Returns:
        ``(10, lat, lon)``: valid, finite and unlimited fractions, then the conditional mean,
        spread, p10, p25, p50, p75 and p90.

    Raises:
        AggregateError: for an unknown variable, or a member stack that is not 3-D.
    """
    if members.ndim != 3:
        raise AggregateError(
            f"members must be (n_members, lat, lon); got shape {members.shape}"
        )
    if variable == "cloud_ceiling":
        return _ceiling_censoring(members)
    if variable == "cloud_cover_3h":
        return _cover_censoring(members)
    raise AggregateError(
        f"{variable!r} has no censoring rule; this is for the bounded cloud variables"
    )


def _ceiling_censoring(members: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Ceiling: finite below the sentinel, unlimited at or above it, out-of-range excluded."""
    values = members.astype(np.float32, copy=False)
    in_range = np.isfinite(values) & (values >= 0.0)
    unlimited = in_range & (values >= CLOUD_CEILING_UNLIMITED_THRESHOLD_KM)
    finite = in_range & ~unlimited
    return _censoring_payload(in_range, finite, unlimited, values)


def _cover_censoring(members: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Cover: in range is [0, 100]; there is no unlimited class, so that count is zero."""
    values = members.astype(np.float32, copy=False)
    in_range = np.isfinite(values) & (values >= 0.0) & (values <= 100.0)
    unlimited = np.zeros_like(in_range)
    return _censoring_payload(in_range, in_range, unlimited, values)


def _censoring_payload(
    in_range: npt.NDArray[np.bool_],
    finite: npt.NDArray[np.bool_],
    unlimited: npt.NDArray[np.bool_],
    values: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """The three fractions and the seven conditional fields, computed over the finite members.

    A cell with no finite member has no conditional statistics and reports NaN for all seven,
    rather than zeros that would read as a distribution concentrated at zero.
    """
    total = values.shape[0]
    denominator = np.full(values.shape[1:], float(total))
    valid_fraction = in_range.sum(axis=_MEMBER_AXIS) / denominator
    finite_fraction = finite.sum(axis=_MEMBER_AXIS) / denominator
    unlimited_fraction = unlimited.sum(axis=_MEMBER_AXIS) / denominator

    masked = np.where(finite, values, np.nan)
    counts = finite.sum(axis=_MEMBER_AXIS)
    enough = counts >= 1
    # The percentile convention is the member path's: linear interpolation over the finite
    # members, sorted. A cell with no finite member makes the NaN-aware reductions report an
    # empty slice, which numpy announces with a warning even though NaN is exactly the answer
    # wanted here; the warning is silenced rather than the array masked, because masking would
    # need a gather and a scatter to reach the same NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(masked, axis=_MEMBER_AXIS)
        spread = np.nanstd(masked, axis=_MEMBER_AXIS)
        percentiles = _conditional_percentiles(masked, counts)
    conditional_fields = np.stack([mean, spread, *percentiles]).astype(np.float32)
    conditional_fields = np.where(enough[None], conditional_fields, np.float32(np.nan))

    fractions = np.stack(
        [valid_fraction, finite_fraction, unlimited_fraction]
    ).astype(np.float32)
    return np.concatenate([fractions, conditional_fields], axis=0)


def variable_group_fields(
    variable: str,
    *,
    members: npt.NDArray[np.float32],
    extra_members: Mapping[str, npt.NDArray[np.float32]] | None = None,
    predecessor_members: Mapping[str, npt.NDArray[np.float32]] | None = None,
) -> npt.NDArray[np.float32]:
    """Build every supplementary group a variable's container carries, concatenated in order.

    This is the writer's entry point: it maps a variable's declared groups onto the member stacks
    those groups read, in the order :func:`domain.field_layout.aggregate_fields_for` lays them out.

    Args:
        variable: The variable whose container is being built.
        members: ``(n_members, lat, lon)`` the variable's own members.
        extra_members: Member stacks for the *other* variables a group reads, keyed by variable
            code. The rose needs ``wind_u_10m`` and ``wind_v_10m``; the phase and transition
            groups need the four flags. :func:`domain.field_layout.required_member_variables`
            names the set, so a caller can check its own completeness rather than infer it here.
        predecessor_members: The same stacks for the predecessor interval, for a group that reads
            it. Absent means the predecessor interval is not available.

    Returns:
        ``(n_group_fields, lat, lon)``.

    Raises:
        AggregateError: for a variable with no groups, a group whose inputs are missing, or a
            group named by the layout that this function cannot build. The last is deliberate:
            a group with no producer would otherwise be published as zeros, which reads as a
            definite answer.
    """
    name = variable.strip()
    try:
        layout = aggregate_fields_for(name)
    except FieldLayoutError as exc:
        raise AggregateError(str(exc)) from exc
    if not layout.groups:
        return np.zeros((0, *members.shape[1:]), dtype=np.float32)

    extra = extra_members or {}
    previous = predecessor_members or {}
    blocks: list[npt.NDArray[np.float32]] = []
    for group in layout.groups:
        blocks.append(_group_block(name, group, members, extra, previous))
    return np.concatenate(blocks, axis=0)


def _group_block(
    variable: str,
    group: str,
    members: npt.NDArray[np.float32],
    extra: Mapping[str, npt.NDArray[np.float32]],
    previous: Mapping[str, npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    """One group's fields, from the member stacks it reads."""
    if group == "rose":
        u = _require_input(extra, "wind_u_10m", variable, group)
        v = _require_input(extra, "wind_v_10m", variable, group)
        rose, scalars, edges = rose_fields(u, v)
        # The edges travel as constant planes, so the container is self-describing: a reader can
        # label a bucket, or add the rose up into a speed histogram, without a sidecar object.
        edge_planes = np.broadcast_to(
            edges[:, None, None], (edges.size, *members.shape[1:])
        ).astype(np.float32)
        return np.concatenate([rose, scalars, edge_planes], axis=0)

    if group in ("phase", "transition"):
        flags = _required_flag_stack(extra, variable, group)
        prev_amounts, prev_flags = _predecessor_interval(previous, variable, group)
        if group == "phase":
            return phase_support_fields(
                members,
                flags,
                amounts_prev=prev_amounts,
                flags_prev=prev_flags,
            )
        return transition_fields(
            members, flags, amounts_prev=prev_amounts, flags_prev=prev_flags
        )

    if group in ("censoring", "conditional"):
        # One computation produces both groups -- the counts and the statistics they condition --
        # so it is done once and split by each group's own declared width rather than recomputed
        # per group, which would also be a second sorting pass over the members.
        payload = cloud_censoring_fields(members, variable=variable)
        censoring_width = len(group_field_names("censoring"))
        if group == "censoring":
            return payload[:censoring_width]
        return payload[censoring_width:]

    if group == "fraction":
        finite = np.isfinite(members)
        # A flag plane's "set" is the same threshold the serving path applies, and a non-finite
        # member neither satisfies it nor counts towards the denominator: a NaN flag is an
        # unknown, and calling it a "no" would understate the fraction.
        return fraction_of_members((members >= FLAG_THRESHOLD) & finite, finite=finite)[
            None
        ]

    raise AggregateError(  # pragma: no cover - every declared group has a producer here
        f"{variable!r} declares the {group!r} group, but no producer is registered for it"
    )


def _require_input(
    stacks: Mapping[str, npt.NDArray[np.float32]], needed: str, variable: str, group: str
) -> npt.NDArray[np.float32]:
    """One member stack a group reads, or a refusal naming which input is missing.

    Raises:
        AggregateError: if the caller did not supply it. Refusing rather than skipping keeps a
            container from being published with a group silently built from nothing.
    """
    try:
        return stacks[needed]
    except KeyError as exc:
        raise AggregateError(
            f"the {group!r} group of {variable!r} reads {needed!r}, which was not supplied"
        ) from exc


def _required_flag_stack(
    stacks: Mapping[str, npt.NDArray[np.float32]], variable: str, group: str
) -> npt.NDArray[np.float32]:
    """The four flag planes as one ``(4, n_members, lat, lon)`` stack.

    Raises:
        AggregateError: if any of the four is missing. A partial stack is worse than none: the
            classifier would read the absent flag as unset and classify against the wrong phase
            set, which produces a plausible field rather than a failure.
    """
    planes = [stacks.get(name) for name in FLAG_NAMES]
    if any(plane is None for plane in planes):
        missing = [
            name for name, plane in zip(FLAG_NAMES, planes, strict=True) if plane is None
        ]
        raise AggregateError(
            f"the {group!r} group of {variable!r} reads all four flags "
            f"({', '.join(FLAG_NAMES)}), but {missing} were not supplied"
        )
    return np.stack([plane for plane in planes if plane is not None]).astype(np.float32)


def _predecessor_interval(
    stacks: Mapping[str, npt.NDArray[np.float32]], variable: str, group: str
) -> tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32] | None]:
    """The predecessor interval's amounts and flags, or ``(None, None)`` when it is not supplied.

    All or nothing: an interval is its amount *and* its flags, and half of one would classify a
    transition against a predecessor the members never described. A caller that wants "no
    predecessor" supplies none of the five stacks -- which is the normal case for a lead whose
    predecessor does not exist, i.e. lead 0.

    Raises:
        AggregateError: if some of the interval's stacks are supplied and others are not.
    """
    planes = [stacks.get(name) for name in FLAG_NAMES]
    amount = stacks.get(variable)
    supplied = sum(plane is not None for plane in planes) + (amount is not None)
    if supplied == 0:
        return None, None
    if supplied != len(FLAG_NAMES) + 1:
        missing = [
            name for name, plane in zip(FLAG_NAMES, planes, strict=True) if plane is None
        ]
        if amount is None:
            missing.insert(0, variable)
        raise AggregateError(
            f"the {group!r} group of {variable!r} was given part of a predecessor interval; "
            f"missing {missing}. A predecessor is its amount and all four flags, or none of them"
        )
    return amount, np.stack([plane for plane in planes if plane is not None]).astype(np.float32)


class PrecipitationGroupBuilder:
    """The phase and transition groups, accumulated one member at a time.

    The groups read ten member planes: the precipitation amount and four flags for the interval,
    and the same five for its predecessor. Held as stacks that is 1.25 GB at 721x1440 for thirty
    members -- over the container's budget -- so the writer feeds this one member at a time
    instead of assembling a stack, and both groups come out of the same pass. What the builder
    does hold is one narrow signature plane per member per group: 30 x 721 x 1440 x 2 B = 62 MB
    each, measured at 230 MB for all three together, against 790 MB in int64.

    The arithmetic is :func:`phase_support_fields`' and :func:`transition_fields`': each member's
    cell is reduced to a signature, the distinct signatures are resolved once through the same
    scalar classifier the serving path uses, and the resolved value vectors are accumulated. What
    differs is only that the member axis is a loop the caller drives rather than an array
    dimension.

    Args:
        expected_members: The member set the fields describe, used to refuse a builder fed more
            members than it was told to expect. The fractions divide by the members each cell
            could classify, which is the only denominator a streaming caller can know and the
            one the batched path uses -- see :meth:`finish`.
        with_transitions: Whether to accumulate the transition group as well. Both come from the
            same signature, so a caller wanting only the phase group still pays for the flags but
            not for the second accumulator.
    """

    def __init__(
        self, *, expected_members: int | None = None, with_transitions: bool = True
    ) -> None:
        self._expected_members = expected_members
        self._with_transitions = with_transitions
        self._added = 0
        self._shape: tuple[int, int] | None = None
        self._phase_signatures: list[npt.NDArray[np.int64]] = []
        self._previous_signatures: list[npt.NDArray[np.int64]] = []
        self._transition_signatures: list[npt.NDArray[np.int64]] = []
        #: Per member, the member-cells each group's denominator counts. Kept as masks rather than
        #: running counts because a member-cell outside the mask must not be summed either --
        #: a non-finite input still resolves to a definite value vector ("dry"), so counting it in
        #: the denominator of one cell and adding it in the numerator of another would be worse
        #: than either.
        self._phase_masks: list[npt.NDArray[np.bool_]] = []
        self._previous_masks: list[npt.NDArray[np.bool_]] = []
        self._any_predecessor = False

    @property
    def members_added(self) -> int:
        """How many members have been added."""
        return self._added

    def add_member(
        self,
        *,
        amounts: npt.NDArray[np.floating],
        flags: npt.NDArray[np.floating],
        amounts_prev: npt.NDArray[np.floating] | None = None,
        flags_prev: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """Add one member's planes for the interval and, when supplied, its predecessor.

        Args:
            amounts: ``(lat, lon)`` the member's precipitation amount.
            flags: ``(4, lat, lon)`` the member's four flags.
            amounts_prev: ``(lat, lon)`` the predecessor's amount, or ``None``.
            flags_prev: ``(4, lat, lon)`` the predecessor's flags, or ``None``.

        Raises:
            AggregateError: if the planes do not share a shape, if the flag stack is not the four
                planes, or if one side of the predecessor interval is given without the other.
        """
        _check_single_flag_planes(flags, amounts.shape)
        if (amounts_prev is None) != (flags_prev is None):
            raise AggregateError(
                "a predecessor interval needs both its amounts and its flags, or neither"
            )
        if flags_prev is not None and amounts_prev is not None:
            _check_single_flag_planes(flags_prev, amounts_prev.shape)
        if self._shape is None:
            self._shape = amounts.shape
        elif amounts.shape != self._shape:
            raise AggregateError(
                f"member planes have shape {amounts.shape}, expected {self._shape}"
            )

        # The same signature helpers the batched path uses, applied to a one-member stack, so
        # the two cannot disagree about what a member's case is -- which is what makes the
        # streaming and batched results identical rather than merely close. Stored narrow: a
        # signature is at most 304 and the builder holds three of these stacks, one per member.
        self._phase_signatures.append(
            _phase_signature(amounts[None], flags[:, None])[0].astype(_SIGNATURE_DTYPE)
        )
        if self._with_transitions:
            self._transition_signatures.append(
                _transition_signature(
                    amounts[None],
                    flags[:, None],
                    None if amounts_prev is None else amounts_prev[None],
                    None if flags_prev is None else flags_prev[:, None],
                )[0].astype(_SIGNATURE_DTYPE)
            )
        # The masks are the members each cell could classify: taken from the same helper the
        # batched path calls, on a one-member stack, so the two builders cannot disagree about
        # which member-cells count. The phase group needs the interval's own amount and flags; the
        # transition group additionally needs the predecessor's, since a member with no
        # predecessor has no transition to count -- that conjunction is formed at ``finish`` time
        # from these two rather than stored as a third 31 MB set.
        self._phase_masks.append(_usable_members(amounts[None], flags[:, None])[0])
        if amounts_prev is not None and flags_prev is not None:
            self._previous_signatures.append(
                _phase_signature(amounts_prev[None], flags_prev[:, None])[0].astype(
                    _SIGNATURE_DTYPE
                )
            )
            self._previous_masks.append(
                _usable_members(amounts_prev[None], flags_prev[:, None])[0]
            )
            self._any_predecessor = True
        else:
            # A member with no predecessor still needs a placeholder so the per-member loop stays
            # aligned with the others; it contributes nothing because its mask is all false.
            self._previous_signatures.append(
                np.zeros(amounts.shape, dtype=_SIGNATURE_DTYPE)
            )
            self._previous_masks.append(np.zeros(amounts.shape, dtype=np.bool_))
        self._added += 1

    def finish(self) -> npt.NDArray[np.float32]:
        """The concatenated group fields, in the layout's order: phase then transition.

        Each group divides by the members its own cells could classify, which is the same
        denominator :func:`phase_support_fields` and :func:`transition_fields` use and the only
        one a streaming caller can know. A cell nobody could classify reports NaN rather than a
        zero that would read as a definite phase.
        """
        if not self._phase_signatures:
            raise AggregateError("no members were added")
        if self._expected_members is not None and self._added > self._expected_members:
            raise AggregateError(
                f"{self._added} members were added but the builder was told to expect "
                f"{self._expected_members}; the fields would describe a member set they do not "
                "come from"
            )

        phase = _accumulate_signatures(
            self._phase_signatures,
            self._phase_masks,
            n_outputs=len(_PHASES),
            resolve=_weights_for_signature,
        )
        if self._any_predecessor:
            previous = _accumulate_signatures(
                self._previous_signatures,
                self._previous_masks,
                n_outputs=len(_PHASES),
                resolve=_weights_for_signature,
            )
        else:
            # No member had a predecessor: the previous planes are absent, not zero, because a
            # zero would claim every member's predecessor was dry.
            previous = np.full_like(phase, np.nan)
        phase = np.concatenate([phase, previous], axis=0)
        if not self._with_transitions:
            return phase
        transition = _accumulate_signatures(
            self._transition_signatures,
            self._transition_masks(),
            n_outputs=len(_TRANSITIONS),
            resolve=_transition_for_signature,
        )
        return np.concatenate([phase, transition], axis=0)

    def _transition_masks(self) -> Iterator[npt.NDArray[np.bool_]]:
        """The member-cells the transition denominator counts, one plane per member.

        Derived from the other two masks rather than stored, because it is exactly their
        conjunction and a third set of 30 x 721 x 1440 booleans is 31 MB. The conjunction is the
        condition the batched path applies -- an interval the member could classify *and* a
        predecessor it had -- so the two agree by construction rather than by inspection.

        When no member had a predecessor at all, the batch the batched path would build is one
        with ``amounts_prev=None``, whose denominator is the interval's own usable count. That is
        the case a lead whose predecessor does not exist takes, and it reads the same way here.
        """
        if not self._any_predecessor:
            yield from self._phase_masks
            return
        for interval, predecessor in zip(
            self._phase_masks, self._previous_masks, strict=True
        ):
            yield interval & predecessor


def _accumulate_signatures(
    signatures: Sequence[npt.NDArray[np.int64]],
    usable: Iterable[npt.NDArray[np.bool_]],
    *,
    n_outputs: int,
    resolve: Callable[[int], tuple[float, ...]],
) -> npt.NDArray[np.float32]:
    """Accumulate one signature plane per member, resolving each distinct signature once.

    The batched caller hands over a whole stack and the streamed one the planes it collected, so
    the arithmetic exists once. The gathered alternative to both -- materialising
    ``(n_members, lat, lon, n_outputs)`` to sum over the member axis -- is 3.0 GB for the phase
    group and 5.0 GB for the transition group at 721x1440, which is not a shape the container can
    take; it works on the test grids and would fail on the first real lead.

    ``usable`` is the per-member mask of the member-cells this group's denominator counts, and it
    is the caller's rather than the accumulator's because the two groups count differently: a
    transition needs a predecessor where a phase does not.

    An unusable member-cell is *given the reserved code* rather than masked out of the sum. Its
    value vector is otherwise definite -- a signature names which case the classifier saw, and a
    non-finite input reads as "dry", whose weight is a full 1.0 -- so adding it while dividing by a
    denominator that excludes it would report a fraction above one. Folding it into the signature
    costs one int16 plane compare per member, and the reserved code's table row is then zeroed:
    masking the value vectors instead would be an ``(lat, lon, n_outputs)`` float comparison per
    member per group, over a stack that is four to eight times the signature's size. The two are
    within measurement noise of each other on the production grid (both ~7-10 s for the phase
    group at 30 x 721 x 1440, run to run), so the choice is made on the smaller working set rather
    than on speed.
    """
    stacked = np.stack([np.asarray(plane) for plane in signatures])
    counts = np.zeros(stacked.shape[1:], dtype=np.int64)
    masked = False
    for index, member_usable in enumerate(usable):
        counts += member_usable
        if not member_usable.all():
            masked = True
            stacked[index] = np.where(
                member_usable,
                stacked[index],
                np.asarray(_UNUSABLE_SIGNATURE, dtype=stacked.dtype),
            )

    codes, inverse = np.unique(stacked, return_inverse=True)
    table = np.array([resolve(int(code)) for code in codes.tolist()], dtype=np.float32)
    if masked:
        # ``codes`` is sorted, so the reserved code sits at a known position in it: zeroing that
        # one row is the whole exclusion, rather than a per-cell rewrite of the value planes.
        for row in np.flatnonzero(codes == _UNUSABLE_SIGNATURE).tolist():
            table[row] = np.float32(0.0)
    per_member = inverse.reshape(stacked.shape)

    accumulator = np.zeros((*stacked.shape[1:], n_outputs), dtype=np.float32)
    for member in range(stacked.shape[0]):
        _accumulate_planes(accumulator, table[per_member[member]])
    return _finish_fractions(accumulator, counts)
