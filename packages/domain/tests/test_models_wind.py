"""Unit tests for domain 10 m wind mathematics and conventions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from domain.exceptions import EmptyEnsembleError, InvalidEnsembleError, InvalidThresholdError
from domain.models.wind import (
    CARDINAL_DIRECTIONS_8,
    compute_consensus_vector,
    compute_directional_probability,
    compute_wind_rose,
    derive_ensemble_mean_scalar_speed,
    derive_ensemble_mean_vector,
    derive_ensemble_member_speeds,
    derive_meteorological_direction,
    derive_meteorological_direction_array,
    derive_wind_speed,
    get_cardinal_direction,
)


def test_derive_wind_speed_scalar() -> None:
    assert derive_wind_speed(0.0, 0.0) == 0.0
    assert derive_wind_speed(3.0, 4.0) == 5.0
    assert derive_wind_speed(-3.0, -4.0) == 5.0
    assert derive_wind_speed(6.0, -8.0) == 10.0
    assert math.isnan(derive_wind_speed(float("nan"), 4.0))
    assert math.isnan(derive_wind_speed(3.0, float("nan")))


def test_derive_wind_speed_array() -> None:
    u = np.array([0.0, 3.0, -3.0, 6.0])
    v = np.array([0.0, 4.0, -4.0, -8.0])
    expected = np.array([0.0, 5.0, 5.0, 10.0])
    result = derive_wind_speed(u, v)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("u", "v", "expected_deg", "expected_cardinal_8", "expected_cardinal_16"),
    [
        # North wind: air flows South (v < 0)
        (0.0, -10.0, 0.0, "N", "N"),
        # NNE wind: direction 22.5°
        (
            -10.0 * math.sin(math.radians(22.5)),
            -10.0 * math.cos(math.radians(22.5)),
            22.5,
            "NE",
            "NNE",
        ),
        # NE wind: air flows SW (u < 0, v < 0)
        (-10.0, -10.0, 45.0, "NE", "NE"),
        # ENE wind: direction 67.5°
        (
            -10.0 * math.sin(math.radians(67.5)),
            -10.0 * math.cos(math.radians(67.5)),
            67.5,
            "E",
            "ENE",
        ),
        # East wind: air flows West (u < 0)
        (-10.0, 0.0, 90.0, "E", "E"),
        # ESE wind: direction 112.5°
        (
            -10.0 * math.sin(math.radians(112.5)),
            -10.0 * math.cos(math.radians(112.5)),
            112.5,
            "SE",
            "ESE",
        ),
        # SE wind: air flows NW (u < 0, v > 0)
        (-10.0, 10.0, 135.0, "SE", "SE"),
        # SSE wind: direction 157.5°
        (
            -10.0 * math.sin(math.radians(157.5)),
            -10.0 * math.cos(math.radians(157.5)),
            157.5,
            "S",
            "SSE",
        ),
        # South wind: air flows North (v > 0)
        (0.0, 10.0, 180.0, "S", "S"),
        # SSW wind: direction 202.5°
        (
            -10.0 * math.sin(math.radians(202.5)),
            -10.0 * math.cos(math.radians(202.5)),
            202.5,
            "SW",
            "SSW",
        ),
        # SW wind: air flows NE (u > 0, v > 0)
        (10.0, 10.0, 225.0, "SW", "SW"),
        # WSW wind: direction 247.5°
        (
            -10.0 * math.sin(math.radians(247.5)),
            -10.0 * math.cos(math.radians(247.5)),
            247.5,
            "W",
            "WSW",
        ),
        # West wind: air flows East (u > 0)
        (10.0, 0.0, 270.0, "W", "W"),
        # WNW wind: direction 292.5°
        (
            -10.0 * math.sin(math.radians(292.5)),
            -10.0 * math.cos(math.radians(292.5)),
            292.5,
            "NW",
            "WNW",
        ),
        # NW wind: air flows SE (u > 0, v < 0)
        (10.0, -10.0, 315.0, "NW", "NW"),
        # NNW wind: direction 337.5°
        (
            -10.0 * math.sin(math.radians(337.5)),
            -10.0 * math.cos(math.radians(337.5)),
            337.5,
            "N",
            "NNW",
        ),
    ],
)
def test_meteorological_direction_and_cardinals(
    u: float,
    v: float,
    expected_deg: float,
    expected_cardinal_8: str,
    expected_cardinal_16: str,
) -> None:
    deg = derive_meteorological_direction(u, v)
    assert deg is not None
    assert pytest.approx(deg, abs=0.01) == expected_deg
    assert get_cardinal_direction(deg, sectors=8) == expected_cardinal_8
    assert get_cardinal_direction(deg, sectors=16) == expected_cardinal_16


def test_calm_wind_threshold() -> None:
    assert derive_meteorological_direction(0.0, 0.0) is None
    assert derive_meteorological_direction(0.2, 0.2) is None
    assert derive_meteorological_direction(0.0, 0.49) is None

    deg_50 = derive_meteorological_direction(0.0, 0.50)
    assert deg_50 is not None
    assert pytest.approx(deg_50) == 180.0

    assert derive_meteorological_direction(0.2, 0.2, calm_threshold=0.2) is not None
    assert derive_meteorological_direction(0.2, 0.2, calm_threshold=1.0) is None


def test_invalid_and_nan_inputs() -> None:
    assert derive_meteorological_direction(float("nan"), 10.0) is None
    assert derive_meteorological_direction(10.0, float("nan")) is None
    assert derive_meteorological_direction(float("inf"), 10.0) is None
    assert derive_meteorological_direction(10.0, float("-inf")) is None

    assert get_cardinal_direction(None) is None
    assert get_cardinal_direction(float("nan")) is None
    assert get_cardinal_direction(float("inf")) is None

    with pytest.raises(ValueError, match="Unsupported sector count"):
        get_cardinal_direction(180.0, sectors=12)


def test_cardinal_direction_boundaries() -> None:
    # 8-point boundaries
    assert get_cardinal_direction(0.0, sectors=8) == "N"
    assert get_cardinal_direction(22.4, sectors=8) == "N"
    assert get_cardinal_direction(22.5, sectors=8) == "NE"
    assert get_cardinal_direction(67.4, sectors=8) == "NE"
    assert get_cardinal_direction(67.5, sectors=8) == "E"
    assert get_cardinal_direction(337.4, sectors=8) == "NW"
    assert get_cardinal_direction(337.5, sectors=8) == "N"
    assert get_cardinal_direction(359.9, sectors=8) == "N"

    # 16-point boundaries
    assert get_cardinal_direction(0.0, sectors=16) == "N"
    assert get_cardinal_direction(11.24, sectors=16) == "N"
    assert get_cardinal_direction(11.25, sectors=16) == "NNE"
    assert get_cardinal_direction(348.74, sectors=16) == "NNW"
    assert get_cardinal_direction(348.75, sectors=16) == "N"


def test_derive_meteorological_direction_array() -> None:
    u = np.array([0.0, -10.0, 10.0, 0.2, np.nan])
    v = np.array([-10.0, 0.0, 10.0, 0.2, 10.0])
    res = derive_meteorological_direction_array(u, v)

    assert pytest.approx(res[0]) == 0.0
    assert pytest.approx(res[1]) == 90.0
    assert pytest.approx(res[2]) == 225.0
    assert np.isnan(res[3])
    assert np.isnan(res[4])


def test_gefs_opposing_members_regression() -> None:
    u_members = np.zeros(30, dtype=np.float64)
    v_members = np.array([-20.0] * 15 + [20.0] * 15, dtype=np.float64)

    member_speeds = derive_ensemble_member_speeds(u_members, v_members)
    assert len(member_speeds) == 30
    np.testing.assert_allclose(member_speeds, 20.0)

    mean_scalar_speed = derive_ensemble_mean_scalar_speed(u_members, v_members)
    assert pytest.approx(mean_scalar_speed) == 20.0

    mean_u, mean_v = derive_ensemble_mean_vector(u_members, v_members)
    assert pytest.approx(mean_u) == 0.0
    assert pytest.approx(mean_v) == 0.0

    mean_vector_mag = math.hypot(float(mean_u), float(mean_v))
    assert pytest.approx(mean_vector_mag) == 0.0
    assert mean_scalar_speed > mean_vector_mag


def test_ensemble_vectorized_multidimensional_grid() -> None:
    u_grid = np.ones((30, 2, 2), dtype=np.float64) * 3.0
    v_grid = np.ones((30, 2, 2), dtype=np.float64) * 4.0

    mean_u, mean_v = derive_ensemble_mean_vector(u_grid, v_grid, axis=0)
    assert isinstance(mean_u, np.ndarray)
    assert isinstance(mean_v, np.ndarray)
    assert mean_u.shape == (2, 2)
    assert mean_v.shape == (2, 2)
    np.testing.assert_allclose(mean_u, 3.0)
    np.testing.assert_allclose(mean_v, 4.0)

    mean_speed_grid = derive_ensemble_mean_scalar_speed(u_grid, v_grid, axis=0)
    assert isinstance(mean_speed_grid, np.ndarray)
    assert mean_speed_grid.shape == (2, 2)
    np.testing.assert_allclose(mean_speed_grid, 5.0)


def test_compute_consensus_vector_aligned() -> None:
    # 30 members all predicting (3, 4) m/s -> SW wind at 5 m/s
    u = [3.0] * 30
    v = [4.0] * 30
    consensus = compute_consensus_vector(u, v)
    assert pytest.approx(consensus.speed_mps) == 5.0
    assert consensus.direction_deg is not None
    assert pytest.approx(consensus.direction_deg) == 216.8698976
    assert consensus.cardinal == "SW"
    assert pytest.approx(consensus.coherence) == 1.0


def test_compute_consensus_vector_cancelling() -> None:
    # 15 members North (0, -10), 15 members South (0, 10)
    u = [0.0] * 30
    v = [-10.0] * 15 + [10.0] * 15
    consensus = compute_consensus_vector(u, v)
    assert pytest.approx(consensus.speed_mps) == 0.0
    assert consensus.direction_deg is None
    assert consensus.cardinal == "CALM"
    assert pytest.approx(consensus.coherence) == 0.0


def test_compute_consensus_vector_errors() -> None:
    with pytest.raises(EmptyEnsembleError):
        compute_consensus_vector([], [])
    with pytest.raises(InvalidEnsembleError):
        compute_consensus_vector([1.0, 2.0], [1.0])
    with pytest.raises(InvalidEnsembleError):
        compute_consensus_vector([1.0, float("nan")], [1.0, 2.0])


def test_compute_wind_rose_distributions() -> None:
    # Create 30 members with known sectors and speed bins:
    # 5 calm members (< 0.5 m/s)
    # 10 South members (u=0, v=10 m/s -> moderate [5.5, 11))
    # 10 North members (u=0, v=-20 m/s -> gale [17, inf))
    # 5 East members (u=-3, v=0 m/s -> light [0.5, 5.5))
    u = [0.0] * 5 + [0.0] * 10 + [0.0] * 10 + [-3.0] * 5
    v = [0.1] * 5 + [10.0] * 10 + [-20.0] * 10 + [0.0] * 5

    rose = compute_wind_rose(u, v)
    assert rose.member_count == 30
    assert rose.calm_count == 5
    assert pytest.approx(rose.calm_probability) == 5 / 30

    by_sec = {s.sector: s for s in rose.sectors}
    assert len(rose.sectors) == 8
    assert set(by_sec.keys()) == set(CARDINAL_DIRECTIONS_8)

    # Check South sector
    south = by_sec["S"]
    assert south.count == 10
    assert pytest.approx(south.probability) == 10 / 30
    assert south.bin_counts["moderate"] == 10
    assert pytest.approx(south.bins["moderate"]) == 10 / 30

    # Check North sector (gale)
    north = by_sec["N"]
    assert north.count == 10
    assert pytest.approx(north.probability) == 10 / 30
    assert north.bin_counts["gale"] == 10

    # Check East sector (light)
    east = by_sec["E"]
    assert east.count == 5
    assert pytest.approx(east.probability) == 5 / 30
    assert east.bin_counts["light"] == 5

    # Total probabilities sum to 1.0 (calm + sum(sector probs))
    total_prob = rose.calm_probability + sum(s.probability for s in rose.sectors)
    assert pytest.approx(total_prob) == 1.0


def test_compute_wind_rose_errors() -> None:
    with pytest.raises(EmptyEnsembleError):
        compute_wind_rose([], [])
    with pytest.raises(InvalidEnsembleError):
        compute_wind_rose([1.0, 2.0], [1.0])
    with pytest.raises(InvalidEnsembleError):
        compute_wind_rose([1.0, float("nan")], [1.0, 2.0])


def test_compute_directional_probability_direction_only() -> None:
    # 20 members SW (u=10, v=10), 10 members N (u=0, v=-10)
    u = [10.0] * 20 + [0.0] * 10
    v = [10.0] * 20 + [-10.0] * 10

    prob_sw, ci_sw = compute_directional_probability(u, v, sector="SW")
    assert pytest.approx(prob_sw) == 20 / 30
    assert ci_sw[0] <= prob_sw <= ci_sw[1]

    prob_n, ci_n = compute_directional_probability(u, v, sector="N")
    assert pytest.approx(prob_n) == 10 / 30

    prob_e, _ = compute_directional_probability(u, v, sector="E")
    assert prob_e == 0.0


def test_compute_directional_probability_joint_speed_and_direction() -> None:
    # 15 members SW with speed 10 m/s (u=7.07, v=7.07)
    # 15 members SW with speed 20 m/s (u=14.14, v=14.14)
    u = [7.071] * 15 + [14.142] * 15
    v = [7.071] * 15 + [14.142] * 15

    # P(SW >= 15 m/s)
    prob_gte, _ = compute_directional_probability(
        u, v, sector="SW", speed_threshold=15.0, operator="gte"
    )
    assert pytest.approx(prob_gte) == 15 / 30

    # P(SW > 10 m/s)
    prob_gt, _ = compute_directional_probability(
        u, v, sector="SW", speed_threshold=10.0, operator="gt"
    )
    assert pytest.approx(prob_gt) == 15 / 30

    # P(SW <= 10 m/s)
    prob_lte, _ = compute_directional_probability(
        u, v, sector="SW", speed_threshold=10.001, operator="lte"
    )
    assert pytest.approx(prob_lte) == 15 / 30

    # P(SW <= 5 m/s) - test le alias with no matches
    prob_le, _ = compute_directional_probability(
        u, v, sector="SW", speed_threshold=5.0, operator="le"
    )
    assert prob_le == 0.0

    # P(SW < 10 m/s) -> 15 members
    prob_lt, _ = compute_directional_probability(
        u, v, sector="SW", speed_threshold=10.0, operator="lt"
    )
    assert pytest.approx(prob_lt) == 15 / 30

    # P(SW < 5 m/s) -> 0 members
    prob_lt_zero, _ = compute_directional_probability(
        u, v, sector="SW", speed_threshold=5.0, operator="<"
    )
    assert prob_lt_zero == 0.0

    # P(SW >= 15 m/s) - test ge alias
    prob_ge, _ = compute_directional_probability(
        u, v, sector="SW", speed_threshold=15.0, operator="ge"
    )
    assert pytest.approx(prob_ge) == 15 / 30


def test_compute_directional_probability_calm_members_excluded() -> None:
    # 5 calm members, 15 SW members (10 m/s), 10 N members (10 m/s)
    u = [0.1] * 5 + [7.071] * 15 + [0.0] * 10
    v = [0.1] * 5 + [7.071] * 15 + [-10.0] * 10

    # Calm members are not in SW sector even if speed < 0.5
    prob_sw, _ = compute_directional_probability(u, v, sector="SW")
    assert pytest.approx(prob_sw) == 15 / 30

    prob_n, _ = compute_directional_probability(u, v, sector="N")
    assert pytest.approx(prob_n) == 10 / 30


def test_compute_directional_probability_errors() -> None:
    with pytest.raises(InvalidThresholdError, match="Invalid directional sector"):
        compute_directional_probability([1.0] * 10, [1.0] * 10, sector="INVALID")
    with pytest.raises(InvalidThresholdError, match="speed_threshold must be a non-negative"):
        compute_directional_probability(
            [1.0] * 10, [1.0] * 10, sector="SW", speed_threshold=-5.0
        )
    with pytest.raises(InvalidThresholdError, match="Unsupported operator"):
        compute_directional_probability(
            [1.0] * 10, [1.0] * 10, sector="SW", speed_threshold=5.0, operator="between"
        )
    with pytest.raises(EmptyEnsembleError):
        compute_directional_probability([], [], sector="N")
    with pytest.raises(InvalidEnsembleError):
        compute_directional_probability([1.0], [1.0, 2.0], sector="N")
    with pytest.raises(InvalidEnsembleError):
        compute_directional_probability([float("nan")], [1.0], sector="N")
