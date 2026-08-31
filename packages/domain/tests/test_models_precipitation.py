"""Exhaustive unit tests for precipitation phase classification, weighting, and ensemble support."""

from __future__ import annotations

import pytest
from domain.models.precipitation import (
    EvidenceState,
    PhysicalPhase,
    PrecipitationPhaseState,
    PrecipitationTransition,
    PrecipitationType,
    aggregate_ensemble_phase_support,
    classify_precipitation_phase,
    compute_joint_amount_phase_support,
    compute_phase_weights,
    compute_transition_frequencies,
)


def test_classify_lead_zero_analysis() -> None:
    """Analysis lead (amount None or NaN) yields NONE state with None amount."""
    state_none = classify_precipitation_phase(None, None)
    assert state_none.interval_type == PrecipitationType.NONE
    assert state_none.start_type == PrecipitationType.NONE
    assert state_none.end_type == PrecipitationType.NONE
    assert state_none.transition == PrecipitationTransition.NONE
    assert state_none.evidence == EvidenceState.EXACT
    assert state_none.active_phases == frozenset()
    assert state_none.amount_3h_mm is None

    state_nan = classify_precipitation_phase(float("nan"), (1, 0, 0, 0))
    assert state_nan.interval_type == PrecipitationType.NONE
    assert state_nan.amount_3h_mm is None


def test_classify_dry_interval() -> None:
    """Precipitation amount <= TRACE_THRESHOLD_MM is deterministically dry/none."""
    # Dry with flags present (must not trigger false rain)
    state = classify_precipitation_phase(0.04, {"crain": 1, "csnow": 0})
    assert state.interval_type == PrecipitationType.NONE
    assert state.start_type == PrecipitationType.NONE
    assert state.end_type == PrecipitationType.NONE
    assert state.transition == PrecipitationTransition.NONE
    assert state.evidence == EvidenceState.EXACT
    assert state.active_phases == frozenset()
    assert state.amount_3h_mm == 0.04


def test_classify_wet_to_dry_transition() -> None:
    """Predecessor wet and current dry yields wet_to_dry transition."""
    state = classify_precipitation_phase(
        0.0,
        {"crain": 0},
        amount_prev=3.5,
        flags_prev={"crain": 1},
    )
    assert state.interval_type == PrecipitationType.NONE
    assert state.transition == PrecipitationTransition.WET_TO_DRY
    assert state.evidence == EvidenceState.STRONGLY_INFERRED


def test_classify_unknown_wet_phase() -> None:
    """Measurable precipitation without diagnostic flags yields UNKNOWN state."""
    state = classify_precipitation_phase(2.5, {"crain": 0, "csnow": 0, "cfrzr": 0, "cicep": 0})
    assert state.interval_type == PrecipitationType.UNKNOWN
    assert state.start_type == PrecipitationType.UNKNOWN
    assert state.end_type == PrecipitationType.UNKNOWN
    assert state.transition == PrecipitationTransition.UNKNOWN
    assert state.evidence == EvidenceState.AMBIGUOUS
    assert state.active_phases == frozenset({PhysicalPhase.UNKNOWN})
    assert state.amount_3h_mm == 2.5


def test_classify_single_phases_direct() -> None:
    """Direct 3h leads with single active flag classify exact physical types."""
    # Rain
    r = classify_precipitation_phase(4.0, (1, 0, 0, 0))
    assert r.interval_type == PrecipitationType.RAIN
    assert r.start_type == PrecipitationType.RAIN
    assert r.end_type == PrecipitationType.RAIN
    assert r.transition == PrecipitationTransition.PERSISTENT_RAIN
    assert r.active_phases == frozenset({PhysicalPhase.RAIN})

    # Snow
    s = classify_precipitation_phase(1.5, {"csnow": 1})
    assert s.interval_type == PrecipitationType.SNOW
    assert s.start_type == PrecipitationType.SNOW
    assert s.end_type == PrecipitationType.SNOW
    assert s.transition == PrecipitationTransition.PERSISTENT_SNOW
    assert s.active_phases == frozenset({PhysicalPhase.SNOW})

    # Freezing Rain
    fr = classify_precipitation_phase(0.8, {"cfrzr": 1})
    assert fr.interval_type == PrecipitationType.FREEZING_RAIN
    assert fr.start_type == PrecipitationType.FREEZING_RAIN
    assert fr.end_type == PrecipitationType.FREEZING_RAIN
    assert fr.transition == PrecipitationTransition.PERSISTENT_FREEZING_RAIN
    assert fr.active_phases == frozenset({PhysicalPhase.FREEZING_RAIN})

    # Ice Pellets
    ip = classify_precipitation_phase(0.5, {"cicep": 1})
    assert ip.interval_type == PrecipitationType.ICE_PELLETS
    assert ip.start_type == PrecipitationType.ICE_PELLETS
    assert ip.end_type == PrecipitationType.ICE_PELLETS
    assert ip.transition == PrecipitationTransition.PERSISTENT_ICE_PELLETS
    assert ip.active_phases == frozenset({PhysicalPhase.ICE_PELLETS})


def test_classify_dry_to_wet_fresh_precipitation() -> None:
    """Precipitation starting in target interval (prev dry) is dry_to_<phase>."""
    state = classify_precipitation_phase(
        3.2,
        {"csnow": 1},
        amount_prev=0.0,
        flags_prev={"crain": 0, "csnow": 0},
    )
    assert state.interval_type == PrecipitationType.SNOW
    assert state.start_type == PrecipitationType.NONE
    assert state.end_type == PrecipitationType.SNOW
    assert state.transition == PrecipitationTransition.DRY_TO_SNOW
    assert state.evidence == EvidenceState.EXACT
    assert state.active_phases == frozenset({PhysicalPhase.SNOW})


def test_classify_persistent_single_phase() -> None:
    """Both intervals wet with matching single flag is persistent_<phase>."""
    state = classify_precipitation_phase(
        5.0,
        {"crain": 1, "csnow": 0},
        amount_prev=4.0,
        flags_prev={"crain": 1, "csnow": 0},
    )
    assert state.interval_type == PrecipitationType.RAIN
    assert state.start_type == PrecipitationType.RAIN
    assert state.end_type == PrecipitationType.RAIN
    assert state.transition == PrecipitationTransition.PERSISTENT_RAIN
    assert state.evidence == EvidenceState.EXACT
    assert state.active_phases == frozenset({PhysicalPhase.RAIN})


def test_classify_rain_to_snow_transition() -> None:
    """Rain in predecessor, Snow newly introduced in reset window is rain_to_snow."""
    state = classify_precipitation_phase(
        2.0,
        {"crain": 1, "csnow": 1},
        amount_prev=3.0,
        flags_prev={"crain": 1, "csnow": 0},
        t2m_start=1.5,
        t2m_end=-1.0,
    )
    assert state.interval_type == PrecipitationType.MIXED
    assert state.start_type == PrecipitationType.RAIN
    assert state.end_type == PrecipitationType.SNOW
    assert state.transition == PrecipitationTransition.RAIN_TO_SNOW
    assert state.evidence == EvidenceState.STRONGLY_INFERRED
    # Authoritative active_phases preserves both physical phases
    assert state.active_phases == frozenset({PhysicalPhase.RAIN, PhysicalPhase.SNOW})


def test_classify_snow_to_rain_transition() -> None:
    """Snow in predecessor, Rain newly introduced in reset window is snow_to_rain."""
    state = classify_precipitation_phase(
        4.5,
        {"crain": 1, "csnow": 1},
        amount_prev=2.0,
        flags_prev={"crain": 0, "csnow": 1},
        t2m_start=-1.0,
        t2m_end=2.5,
    )
    assert state.interval_type == PrecipitationType.MIXED
    assert state.start_type == PrecipitationType.SNOW
    assert state.end_type == PrecipitationType.RAIN
    assert state.transition == PrecipitationTransition.SNOW_TO_RAIN
    assert state.evidence == EvidenceState.STRONGLY_INFERRED
    assert state.active_phases == frozenset({PhysicalPhase.SNOW, PhysicalPhase.RAIN})


def test_classify_rain_to_freezing_rain_transition() -> None:
    """Rain in predecessor, Freezing Rain introduced is rain_to_freezing_rain."""
    state = classify_precipitation_phase(
        1.8,
        {"crain": 1, "cfrzr": 1},
        amount_prev=1.2,
        flags_prev={"crain": 1, "cfrzr": 0},
    )
    assert state.interval_type == PrecipitationType.MIXED
    assert state.start_type == PrecipitationType.RAIN
    assert state.end_type == PrecipitationType.FREEZING_RAIN
    assert state.transition == PrecipitationTransition.RAIN_TO_FREEZING_RAIN
    assert state.evidence == EvidenceState.STRONGLY_INFERRED
    assert state.active_phases == frozenset({PhysicalPhase.RAIN, PhysicalPhase.FREEZING_RAIN})


def test_classify_freezing_rain_to_rain_transition() -> None:
    """Freezing Rain in predecessor, Rain introduced is freezing_rain_to_rain."""
    state = classify_precipitation_phase(
        1.5,
        {"crain": 1, "cfrzr": 1},
        amount_prev=1.0,
        flags_prev={"crain": 0, "cfrzr": 1},
    )
    assert state.interval_type == PrecipitationType.MIXED
    assert state.start_type == PrecipitationType.FREEZING_RAIN
    assert state.end_type == PrecipitationType.RAIN
    assert state.transition == PrecipitationTransition.FREEZING_RAIN_TO_RAIN
    assert state.active_phases == frozenset({PhysicalPhase.FREEZING_RAIN, PhysicalPhase.RAIN})


def test_classify_snow_to_ice_pellets() -> None:
    """Snow in predecessor, Ice Pellets introduced is snow_to_ice_pellets."""
    state = classify_precipitation_phase(
        0.9,
        {"csnow": 1, "cicep": 1},
        amount_prev=1.5,
        flags_prev={"csnow": 1, "cicep": 0},
    )
    assert state.interval_type == PrecipitationType.MIXED
    assert state.start_type == PrecipitationType.SNOW
    assert state.end_type == PrecipitationType.ICE_PELLETS
    assert state.transition == PrecipitationTransition.SNOW_TO_ICE_PELLETS
    assert state.active_phases == frozenset({PhysicalPhase.SNOW, PhysicalPhase.ICE_PELLETS})


def test_classify_ambiguous_multi_phase() -> None:
    """Multi-phase coexistence with complex overlap yields MIXED_TRANSITION and AMBIGUOUS."""
    state = classify_precipitation_phase(
        3.0,
        {"crain": 1, "csnow": 1, "cfrzr": 1},
        amount_prev=2.0,
        flags_prev={"crain": 1, "csnow": 1},
    )
    assert state.interval_type == PrecipitationType.MIXED
    assert state.transition == PrecipitationTransition.MIXED_TRANSITION
    assert state.evidence == EvidenceState.AMBIGUOUS
    # Preserves all 3 represented physical phases
    assert state.active_phases == frozenset({
        PhysicalPhase.RAIN,
        PhysicalPhase.SNOW,
        PhysicalPhase.FREEZING_RAIN,
    })


# --- Phase Weight Derivation Tests ---


def test_compute_phase_weights_dry() -> None:
    """Dry state yields dry=1.0 and all other phases 0.0."""
    dry_state = classify_precipitation_phase(0.05, {"crain": 1})
    weights = compute_phase_weights(dry_state)
    assert weights[PhysicalPhase.DRY] == 1.0
    assert weights[PhysicalPhase.RAIN] == 0.0
    assert weights[PhysicalPhase.SNOW] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_compute_phase_weights_single_phase() -> None:
    """Single active phase yields that phase=1.0."""
    rain_state = classify_precipitation_phase(4.5, {"crain": 1, "csnow": 0})
    weights = compute_phase_weights(rain_state)
    assert weights[PhysicalPhase.RAIN] == 1.0
    assert weights[PhysicalPhase.DRY] == 0.0
    assert weights[PhysicalPhase.SNOW] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_compute_phase_weights_two_phase_equal_split() -> None:
    """Two active phases (Rain -> Snow) yields equal 0.5 split."""
    state = classify_precipitation_phase(
        2.0,
        {"crain": 1, "csnow": 1},
        amount_prev=2.0,
        flags_prev={"crain": 1, "csnow": 0},
    )
    weights = compute_phase_weights(state)
    assert weights[PhysicalPhase.RAIN] == pytest.approx(0.5, abs=1e-6)
    assert weights[PhysicalPhase.SNOW] == pytest.approx(0.5, abs=1e-6)
    assert weights[PhysicalPhase.DRY] == 0.0
    assert weights[PhysicalPhase.FREEZING_RAIN] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_compute_phase_weights_three_phase_mandatory() -> None:
    """Three active phases yields exact 1/3 split preserving intermediate phase."""
    # State with active_phases = {rain, snow, freezing_rain}
    state = classify_precipitation_phase(
        5.0,
        {"crain": 1, "csnow": 1, "cfrzr": 1},
        amount_prev=2.0,
        flags_prev={"crain": 1, "csnow": 1},
    )
    weights = compute_phase_weights(state)
    assert weights[PhysicalPhase.RAIN] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert weights[PhysicalPhase.SNOW] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert weights[PhysicalPhase.FREEZING_RAIN] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert weights[PhysicalPhase.DRY] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_compute_phase_weights_four_phase_equal_split() -> None:
    """All 4 active wet phases yields exact 0.25 split."""
    state = classify_precipitation_phase(
        6.0,
        {"crain": 1, "csnow": 1, "cfrzr": 1, "cicep": 1},
    )
    weights = compute_phase_weights(state)
    assert weights[PhysicalPhase.RAIN] == pytest.approx(0.25, abs=1e-6)
    assert weights[PhysicalPhase.SNOW] == pytest.approx(0.25, abs=1e-6)
    assert weights[PhysicalPhase.FREEZING_RAIN] == pytest.approx(0.25, abs=1e-6)
    assert weights[PhysicalPhase.ICE_PELLETS] == pytest.approx(0.25, abs=1e-6)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_compute_phase_weights_unknown_wet() -> None:
    """Unknown wet phase yields unknown=1.0 and is not redistributed."""
    state = classify_precipitation_phase(3.0, {"crain": 0, "csnow": 0, "cfrzr": 0, "cicep": 0})
    weights = compute_phase_weights(state)
    assert weights[PhysicalPhase.UNKNOWN] == 1.0
    assert weights[PhysicalPhase.RAIN] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


# --- Ensemble Aggregation Tests ---


def test_aggregate_ensemble_phase_support_worked_example() -> None:
    """4-member worked test: Rain, Rain, Rain->Snow, Rain->Snow -> 75% Rain, 25% Snow."""
    m1 = classify_precipitation_phase(3.0, {"crain": 1})
    m2 = classify_precipitation_phase(2.5, {"crain": 1})
    m3 = classify_precipitation_phase(
        2.0, {"crain": 1, "csnow": 1}, amount_prev=2.0, flags_prev={"crain": 1}
    )
    m4 = classify_precipitation_phase(
        1.8, {"crain": 1, "csnow": 1}, amount_prev=1.5, flags_prev={"crain": 1}
    )

    support = aggregate_ensemble_phase_support([m1, m2, m3, m4])
    assert support[PhysicalPhase.RAIN] == pytest.approx(0.75, abs=1e-6)
    assert support[PhysicalPhase.SNOW] == pytest.approx(0.25, abs=1e-6)
    assert support[PhysicalPhase.DRY] == 0.0
    assert support[PhysicalPhase.FREEZING_RAIN] == 0.0
    assert sum(support.values()) == pytest.approx(1.0, abs=1e-6)


def test_aggregate_ensemble_phase_support_empty_raises() -> None:
    """Empty member sequence raises ValueError."""
    with pytest.raises(ValueError, match="empty member_states"):
        aggregate_ensemble_phase_support([])


def test_joint_amount_phase_support() -> None:
    """Joint amount exceedance x phase support properly weights multi-phase members."""
    # Member 1: Rain, 8mm (exceeds 5mm threshold, rain=1.0)
    m1 = classify_precipitation_phase(8.0, {"crain": 1})
    # Member 2: Rain -> Snow, 6mm (exceeds 5mm threshold, rain=0.5, snow=0.5)
    m2 = classify_precipitation_phase(
        6.0, {"crain": 1, "csnow": 1}, amount_prev=4.0, flags_prev={"crain": 1}
    )
    # Member 3: Snow, 3mm (below 5mm threshold, contributes 0.0)
    m3 = classify_precipitation_phase(3.0, {"csnow": 1})
    # Member 4: Dry, 0mm (below 5mm threshold, contributes 0.0)
    m4 = classify_precipitation_phase(0.0, None)

    members = [m1, m2, m3, m4]
    # Total rain joint support = (1.0 + 0.5 + 0.0 + 0.0) / 4 = 1.5 / 4 = 0.375
    p_rain_5mm = compute_joint_amount_phase_support(
        members, threshold_mm=5.0, phase=PhysicalPhase.RAIN
    )
    assert p_rain_5mm == pytest.approx(0.375, abs=1e-6)

    # Total snow joint support = (0.0 + 0.5 + 0.0 + 0.0) / 4 = 0.5 / 4 = 0.125
    p_snow_5mm = compute_joint_amount_phase_support(
        members, threshold_mm=5.0, phase=PhysicalPhase.SNOW
    )
    assert p_snow_5mm == pytest.approx(0.125, abs=1e-6)

    # Negative threshold raises ValueError
    with pytest.raises(ValueError, match="must be non-negative"):
        compute_joint_amount_phase_support(members, threshold_mm=-1.0, phase=PhysicalPhase.RAIN)


def test_extract_active_physical_phases_tuple_and_unknown_edge_cases() -> None:
    """Tuple flag inputs and unknown active_phases are covered."""
    # Test tuple inputs for snow, freezing rain, ice pellets
    s = classify_precipitation_phase(2.0, (0, 1, 0, 0))
    assert s.active_phases == frozenset({PhysicalPhase.SNOW})

    fr = classify_precipitation_phase(2.0, (0, 0, 1, 0))
    assert fr.active_phases == frozenset({PhysicalPhase.FREEZING_RAIN})

    ip = classify_precipitation_phase(2.0, (0, 0, 0, 1))
    assert ip.active_phases == frozenset({PhysicalPhase.ICE_PELLETS})

    # Test state with only UNKNOWN active_phases in compute_phase_weights
    unknown_state = PrecipitationPhaseState(
        interval_type=PrecipitationType.MIXED,
        start_type=PrecipitationType.UNKNOWN,
        end_type=PrecipitationType.UNKNOWN,
        transition=PrecipitationTransition.UNKNOWN,
        evidence=EvidenceState.AMBIGUOUS,
        active_phases=frozenset({PhysicalPhase.UNKNOWN}),
        amount_3h_mm=3.0,
    )
    w = compute_phase_weights(unknown_state)
    assert w[PhysicalPhase.UNKNOWN] == 1.0


def test_joint_amount_phase_support_empty_raises() -> None:
    """Empty member sequence in compute_joint_amount_phase_support raises ValueError."""
    with pytest.raises(ValueError, match="empty member_states"):
        compute_joint_amount_phase_support([], threshold_mm=1.0, phase=PhysicalPhase.RAIN)


def test_compute_transition_frequencies() -> None:
    """Transition frequency computes unweighted member-binary fractions."""
    m1 = classify_precipitation_phase(3.0, {"crain": 1})  # persistent_rain
    m2 = classify_precipitation_phase(2.0, {"crain": 1})  # persistent_rain
    m3 = classify_precipitation_phase(
        2.0, {"crain": 1, "csnow": 1}, amount_prev=2.0, flags_prev={"crain": 1}
    )  # rain_to_snow
    m4 = classify_precipitation_phase(0.0, None)  # none

    freqs = compute_transition_frequencies([m1, m2, m3, m4])
    assert freqs[PrecipitationTransition.PERSISTENT_RAIN] == pytest.approx(0.50, abs=1e-6)
    assert freqs[PrecipitationTransition.RAIN_TO_SNOW] == pytest.approx(0.25, abs=1e-6)
    assert freqs[PrecipitationTransition.NONE] == pytest.approx(0.25, abs=1e-6)
    assert sum(freqs.values()) == pytest.approx(1.0, abs=1e-6)


def test_compute_transition_frequencies_empty_raises() -> None:
    """Empty member sequence in compute_transition_frequencies raises ValueError."""
    with pytest.raises(ValueError, match="empty member_states"):
        compute_transition_frequencies([])
