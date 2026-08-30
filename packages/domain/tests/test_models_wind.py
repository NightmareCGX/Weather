"""Unit tests for domain 10 m wind mathematics and conventions."""

import math

import numpy as np
import pytest
from domain.exceptions import (
    EmptyEnsembleError,
    InvalidEnsembleError,
    InvalidGridError,
    InvalidThresholdError,
)
from domain.models.wind import (
    CARDINAL_DIRECTIONS_8,
    EARTH_RADIUS_METERS,
    VECTOR_FIELD_FLAG_INT16,
    VECTOR_FIELD_HEADER_SIZE,
    VECTOR_FIELD_MAGIC,
    VectorGridMetadata,
    advect_particle,
    compute_consensus_vector,
    compute_directional_probability,
    compute_wind_rose,
    decode_vector_field_int16,
    derive_ensemble_mean_scalar_speed,
    derive_ensemble_mean_vector,
    derive_ensemble_member_speeds,
    derive_meteorological_direction,
    derive_meteorological_direction_array,
    derive_wind_speed,
    encode_vector_field_int16,
    get_cardinal_direction,
    sample_vector_bilinear,
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


def test_vector_grid_metadata_validation() -> None:
    meta = VectorGridMetadata(
        lat_start=90.0,
        lat_step=-0.5,
        lat_count=361,
        lon_start=0.0,
        lon_step=0.5,
        lon_count=720,
        scale=0.01,
    )
    assert meta.lat_count == 361
    assert meta.lon_count == 720
    assert meta.scale == 0.01

    with pytest.raises(InvalidGridError, match="at least 2"):
        VectorGridMetadata(
            lat_start=0, lat_step=1, lat_count=1, lon_start=0, lon_step=1, lon_count=10
        )
    with pytest.raises(InvalidGridError, match="at least 2"):
        VectorGridMetadata(
            lat_start=0, lat_step=1, lat_count=10, lon_start=0, lon_step=1, lon_count=1
        )
    with pytest.raises(InvalidGridError, match="non-zero"):
        VectorGridMetadata(
            lat_start=0, lat_step=0, lat_count=10, lon_start=0, lon_step=1, lon_count=10
        )
    with pytest.raises(InvalidGridError, match="non-zero"):
        VectorGridMetadata(
            lat_start=0, lat_step=1, lat_count=10, lon_start=0, lon_step=0, lon_count=10
        )
    with pytest.raises(InvalidThresholdError, match="positive"):
        VectorGridMetadata(
            lat_start=0,
            lat_step=1,
            lat_count=10,
            lon_start=0,
            lon_step=1,
            lon_count=10,
            scale=0.0,
        )
    with pytest.raises(InvalidThresholdError, match="positive"):
        VectorGridMetadata(
            lat_start=0,
            lat_step=1,
            lat_count=10,
            lon_start=0,
            lon_step=1,
            lon_count=10,
            scale=-0.01,
        )
    with pytest.raises(InvalidThresholdError, match="finite"):
        VectorGridMetadata(
            lat_start=0,
            lat_step=1,
            lat_count=10,
            lon_start=0,
            lon_step=1,
            lon_count=10,
            scale=float("nan"),
        )
    with pytest.raises(InvalidGridError, match="finite"):
        VectorGridMetadata(
            lat_start=float("nan"), lat_step=1, lat_count=10, lon_start=0, lon_step=1, lon_count=10
        )
    with pytest.raises(InvalidGridError, match="finite"):
        VectorGridMetadata(
            lat_start=0, lat_step=1, lat_count=10, lon_start=float("inf"), lon_step=1, lon_count=10
        )


def test_vector_field_encode_decode_roundtrip() -> None:
    u = np.array([[10.123, -5.678], [0.0, 45.5]], dtype=np.float32)
    v = np.array([[-12.345, 8.901], [0.49, -0.49]], dtype=np.float32)

    encoded = encode_vector_field_int16(
        u,
        v,
        lat_start=90.0,
        lat_step=-0.5,
        lon_start=0.0,
        lon_step=0.5,
        scale=0.01,
    )
    assert len(encoded) == VECTOR_FIELD_HEADER_SIZE + 2 * 2 * 4

    u_dec, v_dec, meta = decode_vector_field_int16(encoded)
    assert meta.lat_start == 90.0
    assert meta.lat_step == -0.5
    assert meta.lat_count == 2
    assert meta.lon_start == 0.0
    assert meta.lon_step == 0.5
    assert meta.lon_count == 2
    assert pytest.approx(meta.scale) == 0.01

    # Precision tolerance is scale / 2 (0.005 m/s)
    np.testing.assert_allclose(u_dec, u, atol=0.0051)
    np.testing.assert_allclose(v_dec, v, atol=0.0051)


def test_vector_field_encode_errors() -> None:
    u_1d = np.array([1.0, 2.0], dtype=np.float32)
    v_1d = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(InvalidGridError, match="2-dimensional"):
        encode_vector_field_int16(u_1d, v_1d, lat_start=0, lat_step=1, lon_start=0, lon_step=1)

    u_2d = np.ones((2, 2), dtype=np.float32)
    v_mismatch = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(InvalidGridError, match="Shape mismatch"):
        encode_vector_field_int16(
            u_2d, v_mismatch, lat_start=0, lat_step=1, lon_start=0, lon_step=1
        )

    u_nan = np.array([[1.0, float("nan")], [0.0, 0.0]], dtype=np.float32)
    with pytest.raises(InvalidGridError, match="finite numeric values"):
        encode_vector_field_int16(u_nan, u_2d, lat_start=0, lat_step=1, lon_start=0, lon_step=1)


def test_vector_field_decode_errors() -> None:
    with pytest.raises(ValueError, match="Payload too short"):
        decode_vector_field_int16(b"SHORT")

    # Corrupt magic
    bad_magic = b"XXXX" + b"\x00" * (VECTOR_FIELD_HEADER_SIZE - 4)
    with pytest.raises(ValueError, match="Invalid magic"):
        decode_vector_field_int16(bad_magic)

    # Valid magic but bad version
    import struct

    header_bad_ver = struct.pack(
        "<4sBBHf ffIffI",
        VECTOR_FIELD_MAGIC,
        2,  # bad version
        VECTOR_FIELD_FLAG_INT16,
        0,
        0.01,
        90.0,
        -0.5,
        2,
        0.0,
        0.5,
        2,
    )
    with pytest.raises(ValueError, match="Unsupported version"):
        decode_vector_field_int16(header_bad_ver)

    # Bad flags
    header_bad_flags = struct.pack(
        "<4sBBHf ffIffI",
        VECTOR_FIELD_MAGIC,
        1,
        0,  # bad flags
        0,
        0.01,
        90.0,
        -0.5,
        2,
        0.0,
        0.5,
        2,
    )
    with pytest.raises(ValueError, match="Unsupported flags"):
        decode_vector_field_int16(header_bad_flags)

    # Payload size mismatch
    header_valid = struct.pack(
        "<4sBBHf ffIffI",
        VECTOR_FIELD_MAGIC,
        1,
        1,
        0,
        0.01,
        90.0,
        -0.5,
        10,
        0.0,
        0.5,
        10,
    )
    with pytest.raises(ValueError, match="Payload size mismatch"):
        decode_vector_field_int16(header_valid + b"\x00" * 10)


def test_advect_particle() -> None:
    # Pure northward motion from equator (lat 0, lon 0)
    # v = 100 m/s for 3600 seconds -> distance = 360,000 m
    # dlat = 360,000 / 6,371,000 * (180 / pi) = 3.2374 deg
    new_lat, new_lon = advect_particle(0.0, 0.0, 0.0, 100.0, dt_seconds=3600.0)
    expected_dlat = (100.0 * 3600.0 / EARTH_RADIUS_METERS) * (180.0 / math.pi)
    assert pytest.approx(new_lat) == expected_dlat
    assert pytest.approx(new_lon) == 0.0

    # Pure eastward motion across the dateline (lat 0, lon 179)
    # u = 100 m/s for 3600s -> dlon = 3.2374 deg -> new lon = 182.2374 -> wrapped to -177.7626
    new_lat, new_lon = advect_particle(0.0, 179.0, 100.0, 0.0, dt_seconds=3600.0)
    assert pytest.approx(new_lat) == 0.0
    assert pytest.approx(new_lon) == 179.0 + expected_dlat - 360.0

    # Westward motion crossing -180 (lat 0, lon -179)
    new_lat, new_lon = advect_particle(0.0, -179.0, -100.0, 0.0, dt_seconds=3600.0)
    assert pytest.approx(new_lon) == -179.0 - expected_dlat + 360.0

    # High latitude pole clamp
    new_lat, new_lon = advect_particle(
        88.0, 0.0, 100.0, 100.0, dt_seconds=3600.0, lat_clamp=85.0
    )
    assert new_lat == 85.0
    assert not math.isinf(new_lon) and not math.isnan(new_lon)

    # South pole clamp
    new_lat, new_lon = advect_particle(
        -88.0, 0.0, -100.0, -100.0, dt_seconds=3600.0, lat_clamp=85.0
    )
    assert new_lat == -85.0

    # Exact boundary dateline wrap: new_lon lands at +180.0
    new_lat, new_lon = advect_particle(0.0, 180.0, 0.0, 0.0)
    assert new_lat == 0.0
    assert new_lon == 180.0

    # NaN inputs return NaN
    assert math.isnan(advect_particle(float("nan"), 0.0, 1.0, 1.0)[0])
    assert math.isnan(advect_particle(0.0, float("nan"), 1.0, 1.0)[0])
    assert math.isnan(advect_particle(0.0, 0.0, float("nan"), 1.0)[0])
    assert math.isnan(advect_particle(0.0, 0.0, 1.0, float("nan"))[0])


def test_sample_vector_bilinear() -> None:
    meta = VectorGridMetadata(
        lat_start=90.0,
        lat_step=-1.0,
        lat_count=3,  # rows: 90, 89, 88
        lon_start=0.0,
        lon_step=1.0,
        lon_count=4,  # cols: 0, 1, 2, 3
        scale=0.01,
    )
    # U increases with lon, V increases with lat
    u_grid = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ], dtype=np.float32)

    v_grid = np.array([
        [10.0, 10.0, 10.0, 10.0],
        [5.0,  5.0,  5.0,  5.0],
        [0.0,  0.0,  0.0,  0.0],
    ], dtype=np.float32)

    # Exact node sampling
    u_val, v_val = sample_vector_bilinear(u_grid, v_grid, meta, 90.0, 0.0)
    assert pytest.approx(u_val) == 0.0
    assert pytest.approx(v_val) == 10.0

    # Exact midpoint sampling
    u_val, v_val = sample_vector_bilinear(u_grid, v_grid, meta, 89.5, 1.5)
    assert pytest.approx(u_val) == 1.5
    assert pytest.approx(v_val) == 7.5

    # Longitude wrapping: lon = 3.5 wraps to between col 3 (3.0) and col 0 (0.0)
    # u is 3.0 at col 3 and 0.0 at col 0 -> midpoint is 1.5
    u_val, v_val = sample_vector_bilinear(u_grid, v_grid, meta, 90.0, 3.5)
    assert pytest.approx(u_val) == 1.5

    # Negative longitude wrapping: lon = -0.5 is same as 3.5
    u_val, v_val = sample_vector_bilinear(u_grid, v_grid, meta, 90.0, -0.5)
    assert pytest.approx(u_val) == 1.5

    # Target latitude clamped out of bounds
    u_val, v_val = sample_vector_bilinear(u_grid, v_grid, meta, 95.0, 0.0)
    assert pytest.approx(u_val) == 0.0
    assert pytest.approx(v_val) == 10.0

    u_val, v_val = sample_vector_bilinear(u_grid, v_grid, meta, 80.0, 0.0)
    assert pytest.approx(v_val) == 0.0

    # Negative lon_start grid convention [-180, 180]
    meta_neg = VectorGridMetadata(
        lat_start=90.0,
        lat_step=-1.0,
        lat_count=3,
        lon_start=-180.0,
        lon_step=1.0,
        lon_count=4,
        scale=0.01,
    )
    u_val, _ = sample_vector_bilinear(u_grid, v_grid, meta_neg, 90.0, -180.0)
    assert pytest.approx(u_val) == 0.0

    # NaN inputs return NaN
    assert math.isnan(sample_vector_bilinear(u_grid, v_grid, meta, float("nan"), 0.0)[0])
    assert math.isnan(sample_vector_bilinear(u_grid, v_grid, meta, 0.0, float("nan"))[0])

