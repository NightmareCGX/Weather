"""Unit tests for domain.models.point."""

import math

import pytest
from domain.exceptions import InvalidCoordinatesError
from domain.models.point import (
    RESOLVED_VIA_ADDRESS,
    RESOLVED_VIA_CITY,
    RESOLVED_VIA_COORDINATES,
    RESOLVED_VIA_RESORT,
    RESOLVED_VIA_STATION,
    ForecastPoint,
)


class TestForecastPointValidation:
    def test_minimum_coordinates(self) -> None:
        point = ForecastPoint(latitude=-90.0, longitude=-180.0)
        assert point.latitude == -90.0
        assert point.longitude == -180.0

    def test_maximum_coordinates(self) -> None:
        point = ForecastPoint(latitude=90.0, longitude=180.0)
        assert point.latitude == 90.0
        assert point.longitude == 180.0

    def test_defaults(self) -> None:
        point = ForecastPoint(latitude=39.1911, longitude=-106.8175)
        assert point.elevation_m is None
        assert point.resolved_via == RESOLVED_VIA_COORDINATES
        assert point.id is None
        assert point.name is None

    def test_full_construction(self) -> None:
        point = ForecastPoint(
            latitude=39.1911,
            longitude=-106.8175,
            elevation_m=3417.0,
            resolved_via=RESOLVED_VIA_RESORT,
            id="resort_aspen_mountain",
            name="Aspen Mountain",
        )
        assert point.elevation_m == 3417.0
        assert point.resolved_via == RESOLVED_VIA_RESORT
        assert point.id == "resort_aspen_mountain"
        assert point.name == "Aspen Mountain"

    def test_valid_resolved_via_values(self) -> None:
        for value in (
            RESOLVED_VIA_COORDINATES,
            RESOLVED_VIA_CITY,
            RESOLVED_VIA_RESORT,
            RESOLVED_VIA_STATION,
            RESOLVED_VIA_ADDRESS,
        ):
            point = ForecastPoint(latitude=0.0, longitude=0.0, resolved_via=value)
            assert point.resolved_via == value

    def test_invalid_resolved_via_rejected(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            ForecastPoint(latitude=0.0, longitude=0.0, resolved_via="river")

    def test_latitude_out_of_range_rejected(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            ForecastPoint(latitude=90.1, longitude=0.0)
        with pytest.raises(InvalidCoordinatesError):
            ForecastPoint(latitude=-90.1, longitude=0.0)

    def test_longitude_out_of_range_rejected(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            ForecastPoint(latitude=0.0, longitude=180.1)
        with pytest.raises(InvalidCoordinatesError):
            ForecastPoint(latitude=0.0, longitude=-180.1)

    def test_non_finite_coordinates_rejected(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with pytest.raises(InvalidCoordinatesError):
                ForecastPoint(latitude=value, longitude=0.0)
            with pytest.raises(InvalidCoordinatesError):
                ForecastPoint(latitude=0.0, longitude=value)


class TestForecastPointBehavior:
    def test_equality(self) -> None:
        first = ForecastPoint(latitude=10.0, longitude=20.0, elevation_m=100.0)
        second = ForecastPoint(latitude=10.0, longitude=20.0, elevation_m=100.0)
        assert first == second

    def test_inequality_on_coordinates(self) -> None:
        first = ForecastPoint(latitude=10.0, longitude=20.0)
        second = ForecastPoint(latitude=10.0, longitude=21.0)
        assert first != second

    def test_frozen_raises_on_mutation(self) -> None:
        point = ForecastPoint(latitude=10.0, longitude=20.0)
        with pytest.raises(AttributeError):
            point.latitude = 11.0  # type: ignore[misc]

    def test_hashable_in_set(self) -> None:
        point = ForecastPoint(latitude=10.0, longitude=20.0)
        assert len({point, point}) == 1

    def test_repr_contains_fields(self) -> None:
        point = ForecastPoint(latitude=10.0, longitude=20.0)
        assert "latitude" in repr(point)
        assert "longitude" in repr(point)
