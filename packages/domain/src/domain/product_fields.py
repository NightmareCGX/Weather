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
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from domain.aggregate import AggregateError
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


def _fully_observed(
    amounts: npt.NDArray[np.floating], flags: npt.NDArray[np.floating]
) -> npt.NDArray[np.bool_]:
    """Cells where every member has both an amount and all four flags.

    A cell with one missing flag is not partially classifiable: the classifier reads all four, so
    a cell missing any of them has no phase at all. Reporting NaN there keeps a product field
    over the same member set as the distribution beside it.
    """
    _check_flag_stack(flags, amounts)
    # ``flags`` is (4, n_members, lat, lon), so finiteness over members and flags means reducing
    # both of its leading axes -- expressed as a reshape because numpy rejects a duplicate axis.
    flag_finite = np.isfinite(flags).reshape(
        len(FLAG_NAMES) * flags.shape[1], *flags.shape[2:]
    )
    return np.isfinite(amounts).all(axis=_MEMBER_AXIS) & flag_finite.all(axis=0)


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
    """Pack four flag planes into one integer per member-cell.

    Raises:
        AggregateError: if the flag stack does not hold exactly the four planes.
    """
    _check_flag_stack(flags)
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


def _scatter_by_signature(
    signature: npt.NDArray[np.int64],
    observable: npt.NDArray[np.bool_],
    counts: npt.NDArray[np.int64],
    *,
    n_outputs: int,
    resolve: Callable[[int], tuple[float, ...]],
) -> npt.NDArray[np.float32]:
    """Accumulate a per-signature value vector into per-cell fractions.

    One pass classifies each *distinct* signature and then gathers the result back, so the
    per-member cost is an integer compare plus a fancy-index rather than a Python call.
    """
    codes, inverse = np.unique(signature, return_inverse=True)
    table = np.array([resolve(int(code)) for code in codes.tolist()], dtype=np.float64)
    # (n_cells * n_members) member-cell rows, each a vector of output values, summed per cell.
    flat = table[inverse.reshape(-1)].reshape(*signature.shape, n_outputs)
    total = flat.sum(axis=_MEMBER_AXIS, dtype=np.float64)
    denominator = np.maximum(counts, 1).astype(np.float64)[:, :, None]
    out = (total / denominator).transpose(2, 0, 1).astype(np.float32)
    return np.where(observable[None, :, :], out, np.float32(np.nan))


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

    A cell where any member is non-finite in amount or flags is NaN, matching the distribution
    beside it: describing a different member set than the distribution would be worse than
    describing none.
    """
    counts = _counts_per_cell(amounts)
    observable = _fully_observed(amounts, flags)
    if (amounts_prev is None) != (flags_prev is None):
        raise AggregateError(
            "a predecessor interval needs both its amounts and its flags, or neither"
        )
    if flags_prev is not None and amounts_prev is not None:
        _check_flag_stack(flags_prev, amounts_prev)
    current = _scatter_by_signature(
        _phase_signature(amounts, flags),
        observable,
        counts,
        n_outputs=len(_PHASES),
        resolve=_weights_for_signature,
    )
    if amounts_prev is None or flags_prev is None:
        previous = np.full_like(current, np.nan)
    else:
        previous = _scatter_by_signature(
            _phase_signature(amounts_prev, flags_prev),
            observable,
            counts,
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
    """
    counts = _counts_per_cell(amounts)
    observable = _fully_observed(amounts, flags)
    if (amounts_prev is None) != (flags_prev is None):
        raise AggregateError(
            "a predecessor interval needs both its amounts and its flags, or neither"
        )
    if amounts_prev is not None and flags_prev is not None:
        _check_flag_stack(flags_prev, amounts_prev)
        observable = observable & np.isfinite(amounts_prev).all(axis=_MEMBER_AXIS)
    return _scatter_by_signature(
        _transition_signature(amounts, flags, amounts_prev, flags_prev),
        observable,
        counts,
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
    # A calm member's direction is undefined (atan2 of a zero vector is 0), so it is assigned a
    # sector but excluded by ``live`` below -- the same treatment the member path gives it.
    with np.errstate(invalid="ignore"):
        direction = np.degrees(np.arctan2(-u_members, -v_members)) % 360.0
    direction = np.where(np.isfinite(direction), direction, 0.0)
    sector = np.floor(
        (direction + ROSE_SECTORS_HALF_WIDTH_DEG) / ROSE_SECTORS_WIDTH_DEG
    )
    sector = np.mod(sector.astype(np.int64), ROSE_SECTORS)
    # Members beyond the last edge fall in the top bucket rather than off the scale.
    bucket = np.clip(
        np.searchsorted(bucket_edges, speed, side="right") - 1, 0, ROSE_BUCKETS - 1
    )
    bucket = np.where(np.isfinite(speed), bucket, 0)
    live = observable[None] & ~calm

    n_members, lat, lon = speed.shape
    # Count members per (sector, bucket, cell) with one bincount over a flattened index: the
    # alternative is a Python loop over 64 combinations of a 1M-cell grid.
    flat_index = (sector * ROSE_BUCKETS + bucket)[live]
    flat_cell = np.broadcast_to(
        np.arange(lat * lon).reshape(lat, lon)[None], (n_members, lat, lon)
    )[live]
    combined = flat_index.astype(np.int64) * (lat * lon) + flat_cell
    totals = np.bincount(
        combined, minlength=ROSE_SECTORS * ROSE_BUCKETS * lat * lon
    ).reshape(ROSE_SECTORS * ROSE_BUCKETS, lat, lon)
    denominator = np.maximum(counts, 1)[None]
    rose = np.where(
        observable[None], (totals / denominator).astype(np.float32), np.float32(np.nan)
    )

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
    finite_member = np.isfinite(values)
    in_range = finite_member & (values >= 0.0)
    unlimited = in_range & (values >= CLOUD_CEILING_UNLIMITED_THRESHOLD_KM)
    finite = in_range & ~unlimited
    conditional = finite.astype(np.float32)
    return _censoring_payload(in_range, finite, unlimited, values, conditional)


def _cover_censoring(members: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Cover: in range is [0, 100]; there is no unlimited class, so that count is zero."""
    values = members.astype(np.float32, copy=False)
    in_range = np.isfinite(values) & (values >= 0.0) & (values <= 100.0)
    unlimited = np.zeros_like(in_range)
    conditional = in_range.astype(np.float32)
    return _censoring_payload(in_range, in_range, unlimited, values, conditional)


def _censoring_payload(
    in_range: npt.NDArray[np.bool_],
    finite: npt.NDArray[np.bool_],
    unlimited: npt.NDArray[np.bool_],
    values: npt.NDArray[np.float32],
    conditional: npt.NDArray[np.float32],
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
        percentiles = [
            np.nanpercentile(masked, level, axis=_MEMBER_AXIS, method="linear")
            for level in (10, 25, 50, 75, 90)
        ]
    conditional_fields = np.stack([mean, spread, *percentiles]).astype(np.float32)
    conditional_fields = np.where(enough[None], conditional_fields, np.float32(np.nan))

    fractions = np.stack(
        [valid_fraction, finite_fraction, unlimited_fraction]
    ).astype(np.float32)
    return np.concatenate([fractions, conditional_fields], axis=0)
