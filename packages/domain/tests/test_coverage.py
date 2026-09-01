"""Unit tests for domain coverage rules and expected member contracts."""

import pytest
from domain.coverage import (
    DEFAULT_MIN_COVERAGE_RATIO,
    MODEL_EXPECTED_MEMBERS,
    LeadCoverage,
    build_lead_coverage,
    compute_coverage_ratio,
    get_expected_members,
    get_min_coverage_ratio,
    is_cell_statistically_valid,
    is_lead_servable,
    register_expected_members,
    reset_min_coverage_ratio,
    set_min_coverage_ratio,
)


@pytest.fixture(autouse=True)
def clean_coverage_state():
    reset_min_coverage_ratio()
    yield
    reset_min_coverage_ratio()


def test_constants_and_types() -> None:
    assert DEFAULT_MIN_COVERAGE_RATIO == 0.85
    assert MODEL_EXPECTED_MEMBERS["gfs"] == 1
    assert MODEL_EXPECTED_MEMBERS["gefs"] == 30
    register_expected_members("test_model", 10)
    assert get_expected_members("test_model") == 10
    with pytest.raises(ValueError, match="must be positive"):
        register_expected_members("test_model", 0)
    cov = LeadCoverage(
        lead_time_hours=0,
        available_members=1,
        expected_members=1,
        coverage_ratio=1.0,
        servable=True,
        available_member_indices=(1,),
    )
    assert cov.servable is True


def test_coverage_threshold_configuration_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verify default is 0.85
    reset_min_coverage_ratio()
    assert get_min_coverage_ratio() == 0.85
    assert is_lead_servable(26, 30) is True
    assert is_lead_servable(25, 30) is False

    # Custom override to 0.90 (90%)
    set_min_coverage_ratio(0.90)
    assert get_min_coverage_ratio() == 0.90
    assert is_lead_servable(27, 30) is True   # 27/30 = 90%
    assert is_lead_servable(26, 30) is False  # 26/30 = 86.7% < 90%
    assert is_cell_statistically_valid(27, 30) is True
    assert is_cell_statistically_valid(26, 30) is False

    # Explicit override parameter
    assert is_lead_servable(18, 20, min_coverage_ratio=0.90) is True
    assert is_lead_servable(17, 20, min_coverage_ratio=0.90) is False
    assert is_cell_statistically_valid(17, 20, min_coverage_ratio=0.90) is False

    # NumPy array support for cell validity
    import numpy as np
    counts_arr = np.array([27, 25, 0, 30])
    valid_mask = is_cell_statistically_valid(counts_arr, 30, min_coverage_ratio=0.85)
    assert np.array_equal(valid_mask, np.array([True, False, False, True]))

    # Invalid thresholds raise ValueError
    with pytest.raises(ValueError, match="must be in"):
        set_min_coverage_ratio(0.0)
    with pytest.raises(ValueError, match="must be in"):
        set_min_coverage_ratio(-0.5)
    with pytest.raises(ValueError, match="must be in"):
        set_min_coverage_ratio(1.5)

    # Environment variable parsing
    monkeypatch.setenv("ENSEMBLE_MIN_COVERAGE_RATIO", "0.95")
    reset_min_coverage_ratio()
    assert get_min_coverage_ratio() == 0.95

    monkeypatch.setenv("ENSEMBLE_MIN_COVERAGE_RATIO", "invalid_number")
    with pytest.raises(ValueError, match="Invalid ENSEMBLE_MIN_COVERAGE_RATIO"):
        reset_min_coverage_ratio()

    monkeypatch.setenv("ENSEMBLE_MIN_COVERAGE_RATIO", "0.0")
    with pytest.raises(ValueError, match="must be in"):
        reset_min_coverage_ratio()

    monkeypatch.delenv("ENSEMBLE_MIN_COVERAGE_RATIO", raising=False)
    reset_min_coverage_ratio()
    assert get_min_coverage_ratio() == 0.85

    # Threshold does NOT change expected_members
    assert get_expected_members("gefs") == 30
    assert get_expected_members("gfs") == 1


def test_get_expected_members_known_models() -> None:
    assert get_expected_members("gfs") == 1
    assert get_expected_members("GFS") == 1
    assert get_expected_members("gefs") == 30
    assert get_expected_members("GEFS ") == 30


def test_get_expected_members_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model identifier 'ecmwf'"):
        get_expected_members("ecmwf")


def test_get_expected_members_unknown_model_with_default() -> None:
    assert get_expected_members("ecmwf", default_if_unknown=50) == 50
    assert get_expected_members("custom_det", default_if_unknown=1) == 1


def test_is_lead_servable_gefs_boundaries() -> None:
    # 30-member GEFS: 85% of 30 is 25.5 -> 26 members needed
    assert is_lead_servable(30, 30) is True  # 100%
    assert is_lead_servable(29, 30) is True  # 96.7%
    assert is_lead_servable(27, 30) is True  # 90.0%
    assert is_lead_servable(26, 30) is True  # 86.7%
    assert is_lead_servable(25, 30) is False  # 83.3%
    assert is_lead_servable(0, 30) is False  # 0%


def test_is_lead_servable_exact_85_percent_boundaries() -> None:
    # 20-member model: exactly 17 is 85.0%
    assert is_lead_servable(17, 20) is True
    assert is_lead_servable(16, 20) is False

    # 100-member model: exactly 85 is 85.0%
    assert is_lead_servable(85, 100) is True
    assert is_lead_servable(84, 100) is False

    # 1-member deterministic
    assert is_lead_servable(1, 1) is True
    assert is_lead_servable(0, 1) is False


def test_is_lead_servable_invalid_inputs() -> None:
    assert is_lead_servable(0, 0) is False
    assert is_lead_servable(5, -1) is False
    assert is_lead_servable(-1, 30) is False


def test_is_cell_statistically_valid() -> None:
    # 30 expected members
    assert is_cell_statistically_valid(30, 30) is True
    assert is_cell_statistically_valid(27, 30) is True
    assert is_cell_statistically_valid(26, 30) is True
    assert is_cell_statistically_valid(25, 30) is False
    assert is_cell_statistically_valid(0, 30) is False

    # Invalid inputs
    assert is_cell_statistically_valid(10, 0) is False
    assert is_cell_statistically_valid(-1, 30) is False
    assert is_cell_statistically_valid(10, -5) is False
    import numpy as np
    assert is_cell_statistically_valid(np.array([10, 20]), 0) is False


def test_compute_coverage_ratio() -> None:
    assert compute_coverage_ratio(30, 30) == 1.0
    assert compute_coverage_ratio(29, 30) == 0.9667
    assert compute_coverage_ratio(26, 30) == 0.8667
    assert compute_coverage_ratio(25, 30) == 0.8333
    assert compute_coverage_ratio(0, 30) == 0.0
    assert compute_coverage_ratio(-5, 30) == 0.0
    assert compute_coverage_ratio(35, 30) == 1.0
    assert compute_coverage_ratio(10, 0) == 0.0
    assert compute_coverage_ratio(10, -5) == 0.0


def test_build_lead_coverage() -> None:
    coverage = build_lead_coverage(
        lead_time_hours=6,
        available_member_indices={1, 2, 3, 5, 4},
        expected_members=5,
    )
    assert coverage.lead_time_hours == 6
    assert coverage.available_members == 5
    assert coverage.expected_members == 5
    assert coverage.coverage_ratio == 1.0
    assert coverage.servable is True
    assert coverage.available_member_indices == (1, 2, 3, 4, 5)

    # Sub-threshold lead coverage
    coverage_below = build_lead_coverage(
        lead_time_hours=72,
        available_member_indices=[1, 2, 3],
        expected_members=10,
    )
    assert coverage_below.lead_time_hours == 72
    assert coverage_below.available_members == 3
    assert coverage_below.expected_members == 10
    assert coverage_below.coverage_ratio == 0.3
    assert coverage_below.servable is False
    assert coverage_below.available_member_indices == (1, 2, 3)
