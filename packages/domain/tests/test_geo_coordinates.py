"""Unit tests for domain.geo.coordinates."""

import math

import pytest
from domain.exceptions import InvalidCoordinatesError
from domain.geo.coordinates import (
    EARTH_RADIUS_KM,
    LATITUDE_MAX,
    LATITUDE_MIN,
    LONGITUDE_MAX,
    LONGITUDE_MIN,
    haversine_distance_km,
    normalize_longitude,
    validate_coordinates,
    validate_latitude,
    validate_longitude,
)


class TestValidateLatitude:
    def test_accepts_valid_values(self) -> None:
        for value in (-90.0, -45.5, 0.0, 45.5, 90.0):
            assert validate_latitude(value) == value

    def test_rejects_below_minimum(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            validate_latitude(LATITUDE_MIN - 0.01)

    def test_rejects_above_maximum(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            validate_latitude(LATITUDE_MAX + 0.01)

    def test_rejects_non_finite(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with pytest.raises(InvalidCoordinatesError):
                validate_latitude(value)


class TestValidateLongitude:
    def test_accepts_valid_values(self) -> None:
        for value in (-180.0, -90.5, 0.0, 90.5, 180.0):
            assert validate_longitude(value) == value

    def test_rejects_below_minimum(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            validate_longitude(LONGITUDE_MIN - 0.01)

    def test_rejects_above_maximum(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            validate_longitude(LONGITUDE_MAX + 0.01)

    def test_rejects_non_finite(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with pytest.raises(InvalidCoordinatesError):
                validate_longitude(value)


class TestValidateCoordinates:
    def test_returns_validated_pair(self) -> None:
        assert validate_coordinates(10.0, 20.0) == (10.0, 20.0)

    def test_rejects_invalid_latitude(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            validate_coordinates(91.0, 0.0)

    def test_rejects_invalid_longitude(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            validate_coordinates(0.0, 181.0)


class TestNormalizeLongitude:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            (0.0, 0.0),
            (180.0, 180.0),
            (-180.0, -180.0),
            (360.0, 0.0),
            (-360.0, 0.0),
            (540.0, 180.0),
            (-540.0, -180.0),
            (45.0, 45.0),
            (-45.0, -45.0),
            (179.0, 179.0),
            (-179.0, -179.0),
            (200.0, -160.0),
            (-200.0, 160.0),
        ],
    )
    def test_wraps_into_range(self, given: float, expected: float) -> None:
        result = normalize_longitude(given)
        assert result == pytest.approx(expected)
        assert LONGITUDE_MIN <= result <= LONGITUDE_MAX

    def test_positive_wraparound_examples(self) -> None:
        assert normalize_longitude(190.0) == pytest.approx(-170.0)
        assert normalize_longitude(170.0) == pytest.approx(170.0)
        assert normalize_longitude(260.0) == pytest.approx(-100.0)
        assert normalize_longitude(360.0 + 190.0) == pytest.approx(-170.0)

    def test_negative_wraparound_examples(self) -> None:
        assert normalize_longitude(-190.0) == pytest.approx(170.0)
        assert normalize_longitude(-260.0) == pytest.approx(100.0)
        assert normalize_longitude(-(360.0 + 190.0)) == pytest.approx(170.0)

    def test_boundary_values_preserved(self) -> None:
        assert normalize_longitude(180.0) == pytest.approx(180.0)
        assert normalize_longitude(-180.0) == pytest.approx(-180.0)

    def test_wraps_congruent_to_180(self) -> None:
        # 540 == 180 (mod 360) and is reached by a positive wrap.
        assert normalize_longitude(540.0) == pytest.approx(180.0)
        # -540 == -180 (mod 360) and is reached by a negative wrap.
        assert normalize_longitude(-540.0) == pytest.approx(-180.0)

    def test_rejects_non_finite(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with pytest.raises(InvalidCoordinatesError):
                normalize_longitude(value)


class TestHaversineDistance:
    def test_zero_distance_for_identical_points(self) -> None:
        assert haversine_distance_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        forward = haversine_distance_km(51.5, -0.1, 40.7128, -74.0060)
        reverse = haversine_distance_km(40.7128, -74.0060, 51.5, -0.1)
        assert forward == pytest.approx(reverse)

    def test_new_york_to_london(self) -> None:
        distance = haversine_distance_km(40.7128, -74.0060, 51.5074, -0.1278)
        assert distance == pytest.approx(5570.0, rel=0.02)

    def test_equator_degree_longitude(self) -> None:
        # One degree of longitude at the equator is ~111.32 km.
        distance = haversine_distance_km(0.0, 0.0, 0.0, 1.0)
        assert distance == pytest.approx(EARTH_RADIUS_KM * math.radians(1.0), rel=1e-3)

    def test_same_longitude_same_latitude_delta(self) -> None:
        # One degree of latitude is always ~111.19 km.
        distance = haversine_distance_km(0.0, 0.0, 1.0, 0.0)
        assert distance == pytest.approx(EARTH_RADIUS_KM * math.radians(1.0), rel=1e-3)

    def test_half_meridian_is_half_circumference(self) -> None:
        distance = haversine_distance_km(0.0, 0.0, 0.0, 180.0)
        assert distance == pytest.approx(math.pi * EARTH_RADIUS_KM, rel=1e-3)

    def test_north_to_south_pole_is_half_circumference(self) -> None:
        distance = haversine_distance_km(90.0, 0.0, -90.0, 0.0)
        assert distance == pytest.approx(math.pi * EARTH_RADIUS_KM, rel=1e-3)

    def test_rejects_invalid_coordinates(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            haversine_distance_km(95.0, 0.0, 0.0, 0.0)
        with pytest.raises(InvalidCoordinatesError):
            haversine_distance_km(0.0, 0.0, 0.0, 185.0)
