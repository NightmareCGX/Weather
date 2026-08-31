"""Pure meteorological domain functions and types for precipitation classification and ensemble.

Provides:
* Explicit enumeration of categorical precipitation types (PrecipitationType).
* Physical precipitation phase buckets (PhysicalPhase) excluding 'mixed'.
* Discrete phase transition identifiers (PrecipitationTransition).
* Evidence/certainty classification (EvidenceState).
* Immutable member-level precipitation state (PrecipitationPhaseState) preserving
  complete interval active_phases.
* Member-level physical phase weight derivations (equal-split baseline summing to 1.0).
* Ensemble normalized phase-support aggregation.
* Joint exceedance amount × physical phase support calculations.
* Ensemble transition frequency aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

#: Policy threshold below which 3-hour precipitation is classified as dry/none (mm).
#: This is a platform product policy distinct from GRIB de-accumulation packing tolerance.
TRACE_THRESHOLD_MM: float = 0.10


class PrecipitationType(str, Enum):
    """Categorical precipitation types for member and interval descriptions."""

    NONE = "none"
    RAIN = "rain"
    SNOW = "snow"
    FREEZING_RAIN = "freezing_rain"
    ICE_PELLETS = "ice_pellets"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PhysicalPhase(str, Enum):
    """Physical precipitation phase buckets for normalized ensemble support aggregation.

    'mixed' is an interval description, NOT a physical phase bucket.
    """

    DRY = "dry"
    RAIN = "rain"
    SNOW = "snow"
    FREEZING_RAIN = "freezing_rain"
    ICE_PELLETS = "ice_pellets"
    UNKNOWN = "unknown"


class PrecipitationTransition(str, Enum):
    """Categorical phase progression across a 3-hour interval."""

    NONE = "none"
    PERSISTENT_RAIN = "persistent_rain"
    PERSISTENT_SNOW = "persistent_snow"
    PERSISTENT_FREEZING_RAIN = "persistent_freezing_rain"
    PERSISTENT_ICE_PELLETS = "persistent_ice_pellets"
    DRY_TO_RAIN = "dry_to_rain"
    DRY_TO_SNOW = "dry_to_snow"
    DRY_TO_FREEZING_RAIN = "dry_to_freezing_rain"
    DRY_TO_ICE_PELLETS = "dry_to_ice_pellets"
    WET_TO_DRY = "wet_to_dry"
    RAIN_TO_SNOW = "rain_to_snow"
    SNOW_TO_RAIN = "snow_to_rain"
    RAIN_TO_FREEZING_RAIN = "rain_to_freezing_rain"
    FREEZING_RAIN_TO_RAIN = "freezing_rain_to_rain"
    SNOW_TO_FREEZING_RAIN = "snow_to_freezing_rain"
    FREEZING_RAIN_TO_SNOW = "freezing_rain_to_snow"
    SNOW_TO_ICE_PELLETS = "snow_to_ice_pellets"
    ICE_PELLETS_TO_SNOW = "ice_pellets_to_snow"
    MIXED_TRANSITION = "mixed_transition"
    UNKNOWN = "unknown"


class EvidenceState(str, Enum):
    """Classification of evidence certainty for transition-state reconstruction."""

    EXACT = "exact"
    STRONGLY_INFERRED = "strongly_inferred"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class PrecipitationPhaseState:
    """Immutable member-level precipitation phase state for a 3-hour interval.

    Attributes:
        interval_type: Summary phase across the interval.
        start_type: Phase at/carried into the start of the interval.
        end_type: Phase at/near the end of the interval.
        transition: Discrete transition identifier.
        evidence: Evidence certainty tier.
        active_phases: Complete set of physical phases supported by the available
            evidence for the target interval and retained for conservative
            phase-support allocation. Authoritative source for ensemble phase weighting.
        amount_3h_mm: 3-hour precipitation accumulation depth in mm (or None for f000).
    """

    interval_type: PrecipitationType
    start_type: PrecipitationType
    end_type: PrecipitationType
    transition: PrecipitationTransition
    evidence: EvidenceState
    active_phases: frozenset[PhysicalPhase]
    amount_3h_mm: float | None


def _extract_active_physical_phases(
    flags: Mapping[str, int] | tuple[int, int, int, int] | None,
) -> set[PhysicalPhase]:
    """Extract physical phases from categorical flags (crain, csnow, cfrzr, cicep)."""
    if flags is None:
        return set()

    active: set[PhysicalPhase] = set()
    if isinstance(flags, Mapping):
        if bool(flags.get("crain", 0)):
            active.add(PhysicalPhase.RAIN)
        if bool(flags.get("csnow", 0)):
            active.add(PhysicalPhase.SNOW)
        if bool(flags.get("cfrzr", 0)):
            active.add(PhysicalPhase.FREEZING_RAIN)
        if bool(flags.get("cicep", 0)):
            active.add(PhysicalPhase.ICE_PELLETS)
    else:
        r, s, fr, ip = flags
        if bool(r):
            active.add(PhysicalPhase.RAIN)
        if bool(s):
            active.add(PhysicalPhase.SNOW)
        if bool(fr):
            active.add(PhysicalPhase.FREEZING_RAIN)
        if bool(ip):
            active.add(PhysicalPhase.ICE_PELLETS)
    return active


def classify_precipitation_phase(
    amount_curr: float | None,
    flags_curr: Mapping[str, int] | tuple[int, int, int, int] | None,
    *,
    amount_prev: float | None = None,
    flags_prev: Mapping[str, int] | tuple[int, int, int, int] | None = None,
    t2m_start: float | None = None,
    t2m_end: float | None = None,
    trace_threshold_mm: float = TRACE_THRESHOLD_MM,
) -> PrecipitationPhaseState:
    """Classify 3-hour precipitation into an immutable PrecipitationPhaseState.

    Combines current and predecessor precipitation amounts, interval-average
    categorical flags, and supporting 2m temperature tendency without allowing
    temperature to override explicit GRIB microphysical diagnostics.

    Args:
        amount_curr: Precipitation amount for target interval [t-3, t] in mm.
            None or NaN represents analysis time (f000).
        flags_curr: Categorical flags (crain, csnow, cfrzr, cicep) for the current window.
        amount_prev: Optional predecessor precipitation amount [t-6, t-3] in mm.
        flags_prev: Optional predecessor categorical flags [t-6, t-3].
        t2m_start: Optional 2m air temperature at t-3 in °C.
        t2m_end: Optional 2m air temperature at t in °C.
        trace_threshold_mm: Threshold below which precipitation is classified as dry.

    Returns:
        A fully-populated :class:`PrecipitationPhaseState`.
    """
    # Case 1: Analysis lead (amount_curr is None or NaN)
    if amount_curr is None or np.isnan(amount_curr):
        return PrecipitationPhaseState(
            interval_type=PrecipitationType.NONE,
            start_type=PrecipitationType.NONE,
            end_type=PrecipitationType.NONE,
            transition=PrecipitationTransition.NONE,
            evidence=EvidenceState.EXACT,
            active_phases=frozenset(),
            amount_3h_mm=None,
        )

    val_curr = float(amount_curr)

    # Case 2: Dry current interval (amount <= trace_threshold)
    if val_curr <= trace_threshold_mm:
        is_wet_prev = amount_prev is not None and amount_prev > trace_threshold_mm
        return PrecipitationPhaseState(
            interval_type=PrecipitationType.NONE,
            start_type=PrecipitationType.NONE,
            end_type=PrecipitationType.NONE,
            transition=(
                PrecipitationTransition.WET_TO_DRY
                if is_wet_prev
                else PrecipitationTransition.NONE
            ),
            evidence=EvidenceState.STRONGLY_INFERRED if is_wet_prev else EvidenceState.EXACT,
            active_phases=frozenset(),
            amount_3h_mm=val_curr,
        )

    # Current interval is wet (amount > trace_threshold)
    curr_phases = _extract_active_physical_phases(flags_curr)

    # Case 3: Wet amount but no diagnostic flags active -> UNKNOWN
    if not curr_phases:
        return PrecipitationPhaseState(
            interval_type=PrecipitationType.UNKNOWN,
            start_type=PrecipitationType.UNKNOWN,
            end_type=PrecipitationType.UNKNOWN,
            transition=PrecipitationTransition.UNKNOWN,
            evidence=EvidenceState.AMBIGUOUS,
            active_phases=frozenset({PhysicalPhase.UNKNOWN}),
            amount_3h_mm=val_curr,
        )

    prev_phases = _extract_active_physical_phases(flags_prev)
    is_prev_dry = amount_prev is None or amount_prev <= trace_threshold_mm

    # Case 4: Dry predecessor or no predecessor flags (direct / fresh precipitation)
    if is_prev_dry or flags_prev is None:
        if len(curr_phases) == 1:
            p = next(iter(curr_phases))
            t_p = PrecipitationType(p.value)
            is_dry_prev = amount_prev is not None and amount_prev <= trace_threshold_mm
            tr_name = f"dry_to_{p.value}" if is_dry_prev else f"persistent_{p.value}"
            return PrecipitationPhaseState(
                interval_type=t_p,
                start_type=PrecipitationType.NONE if is_dry_prev else t_p,
                end_type=t_p,
                transition=(
                    PrecipitationTransition(tr_name)
                    if tr_name in PrecipitationTransition._value2member_map_
                    else PrecipitationTransition.UNKNOWN
                ),
                evidence=EvidenceState.EXACT,
                active_phases=frozenset(curr_phases),
                amount_3h_mm=val_curr,
            )
        # Multiple flags active in fresh interval -> MIXED
        is_dry_prev = amount_prev is not None and amount_prev <= trace_threshold_mm
        return PrecipitationPhaseState(
            interval_type=PrecipitationType.MIXED,
            start_type=PrecipitationType.NONE if is_dry_prev else PrecipitationType.MIXED,
            end_type=PrecipitationType.MIXED,
            transition=PrecipitationTransition.MIXED_TRANSITION,
            evidence=EvidenceState.STRONGLY_INFERRED,
            active_phases=frozenset(curr_phases),
            amount_3h_mm=val_curr,
        )

    # Case 5: Both predecessor and current intervals are wet
    # 5a. Persistent single phase (F_prev == F_curr and single flag)
    if curr_phases == prev_phases and len(curr_phases) == 1:
        p = next(iter(curr_phases))
        t_p = PrecipitationType(p.value)
        tr_name = f"persistent_{p.value}"
        return PrecipitationPhaseState(
            interval_type=t_p,
            start_type=t_p,
            end_type=t_p,
            transition=(
                PrecipitationTransition(tr_name)
                if tr_name in PrecipitationTransition._value2member_map_
                else PrecipitationTransition.UNKNOWN
            ),
            evidence=EvidenceState.EXACT,
            active_phases=frozenset(curr_phases),
            amount_3h_mm=val_curr,
        )

    # 5b. Phase transition with newly introduced phase (Delta F = curr_phases - prev_phases)
    new_phases = curr_phases - prev_phases
    if len(prev_phases) == 1 and len(new_phases) == 1:
        old_p = next(iter(prev_phases))
        new_p = next(iter(new_phases))
        t_old = PrecipitationType(old_p.value)
        t_new = PrecipitationType(new_p.value)
        tr_name = f"{old_p.value}_to_{new_p.value}"
        return PrecipitationPhaseState(
            interval_type=PrecipitationType.MIXED,
            start_type=t_old,
            end_type=t_new,
            transition=(
                PrecipitationTransition(tr_name)
                if tr_name in PrecipitationTransition._value2member_map_
                else PrecipitationTransition.MIXED_TRANSITION
            ),
            evidence=EvidenceState.STRONGLY_INFERRED,
            active_phases=frozenset(curr_phases),
            amount_3h_mm=val_curr,
        )

    # 5c. Multi-phase coexistence with ambiguous ordering
    start_type = (
        PrecipitationType(next(iter(prev_phases)).value)
        if len(prev_phases) == 1
        else PrecipitationType.MIXED
    )
    end_type = (
        PrecipitationType(next(iter(new_phases)).value)
        if len(new_phases) == 1
        else PrecipitationType.MIXED
    )
    return PrecipitationPhaseState(
        interval_type=PrecipitationType.MIXED,
        start_type=start_type,
        end_type=end_type,
        transition=PrecipitationTransition.MIXED_TRANSITION,
        evidence=EvidenceState.AMBIGUOUS,
        active_phases=frozenset(curr_phases),
        amount_3h_mm=val_curr,
    )


def compute_phase_weights(
    state: PrecipitationPhaseState,
) -> dict[PhysicalPhase, float]:
    """Derive normalized 1.0 physical phase support weights from member state.

    Represents normalized ensemble phase-support allocation (not precipitation
    mass fraction or duration fraction). The authoritative source is
    ``state.active_phases``, ensuring all physical phases represented in the
    interval receive proportional normalized support. 'mixed' is not a physical
    phase bucket.

    Rules:
    * Dry interval: dry = 1.0 (all other phases 0.0).
    * Single active phase: that phase = 1.0.
    * N active phases: each active phase = 1 / N.
    * Unknown wet phase: unknown = 1.0.

    Invariant:
        sum(weights.values()) == 1.0 (within 1e-6 tolerance).

    Args:
        state: The member's :class:`PrecipitationPhaseState`.

    Returns:
        Mapping of :class:`PhysicalPhase` to normalized support weight [0.0, 1.0].
    """
    weights = {p: 0.0 for p in PhysicalPhase}

    # Dry condition
    if (
        state.amount_3h_mm is None
        or state.amount_3h_mm <= TRACE_THRESHOLD_MM
        or state.interval_type == PrecipitationType.NONE
    ):
        weights[PhysicalPhase.DRY] = 1.0
        return weights

    # Unknown wet condition
    if state.interval_type == PrecipitationType.UNKNOWN or not state.active_phases:
        weights[PhysicalPhase.UNKNOWN] = 1.0
        return weights

    # Known physical wet phases
    known_active = [
        p for p in state.active_phases if p not in (PhysicalPhase.UNKNOWN, PhysicalPhase.DRY)
    ]
    if not known_active:
        weights[PhysicalPhase.UNKNOWN] = 1.0
        return weights

    split = 1.0 / len(known_active)
    for p in known_active:
        weights[p] = split
    return weights


def aggregate_ensemble_phase_support(
    member_states: Sequence[PrecipitationPhaseState],
) -> dict[PhysicalPhase, float]:
    """Aggregate member-level phase weights into normalized ensemble phase support.

    Computes:
        P(phase = k) = (1 / N_valid_members) * sum_m w[m, k]

    Invariant:
        sum(support.values()) == 1.0 (when member_states is non-empty).

    Args:
        member_states: Sequence of member :class:`PrecipitationPhaseState` instances.

    Returns:
        Mapping of :class:`PhysicalPhase` to ensemble phase support fraction in [0.0, 1.0].

    Raises:
        ValueError: If member_states is empty.
    """
    if not member_states:
        raise ValueError("Cannot aggregate phase support from empty member_states sequence.")

    n_members = len(member_states)
    support = {p: 0.0 for p in PhysicalPhase}

    for state in member_states:
        w = compute_phase_weights(state)
        for p, weight_val in w.items():
            support[p] += weight_val / n_members

    return support


def compute_joint_amount_phase_support(
    member_states: Sequence[PrecipitationPhaseState],
    *,
    threshold_mm: float,
    phase: PhysicalPhase,
) -> float:
    """Compute normalized ensemble support for amount exceedance joint with a physical phase.

    Formula:
        P(A >= T and phase = k) = (1 / N) * sum_m ( indicator(A_m >= T) * w[m, k] )

    Args:
        member_states: Sequence of member :class:`PrecipitationPhaseState` instances.
        threshold_mm: Precipitation accumulation threshold in mm.
        phase: Target :class:`PhysicalPhase` (e.g. SNOW, RAIN, FREEZING_RAIN).

    Returns:
        Joint exceedance support fraction in [0.0, 1.0].

    Raises:
        ValueError: If member_states is empty or threshold_mm is negative.
    """
    if not member_states:
        raise ValueError("Cannot compute joint support from empty member_states sequence.")
    if threshold_mm < 0.0:
        raise ValueError(f"threshold_mm must be non-negative, got {threshold_mm}.")

    n_members = len(member_states)
    joint_mass = 0.0

    for state in member_states:
        amount = state.amount_3h_mm if state.amount_3h_mm is not None else 0.0
        if amount >= threshold_mm:
            w = compute_phase_weights(state)
            joint_mass += w.get(phase, 0.0)

    return joint_mass / n_members


def compute_transition_frequencies(
    member_states: Sequence[PrecipitationPhaseState],
) -> dict[PrecipitationTransition, float]:
    """Compute member-binary frequency distribution over discrete transition states.

    Formula:
        frequency(T) = count(member.transition == T) / N_members

    Args:
        member_states: Sequence of member :class:`PrecipitationPhaseState` instances.

    Returns:
        Mapping of :class:`PrecipitationTransition` to member frequency in [0.0, 1.0].

    Raises:
        ValueError: If member_states is empty.
    """
    if not member_states:
        raise ValueError("Cannot compute transition frequencies from empty member_states.")

    n_members = len(member_states)
    frequencies = {t: 0.0 for t in PrecipitationTransition}

    for state in member_states:
        frequencies[state.transition] += 1.0 / n_members

    return frequencies
