"""Spatial coordinate, grid, and interpolation utilities."""

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
from domain.geo.grid import GRID_BOUNDS_TOLERANCE, GridPoint, RegularGrid
from domain.geo.interpolation import bilinear_interpolate

__all__ = [
    "EARTH_RADIUS_KM",
    "GRID_BOUNDS_TOLERANCE",
    "GridPoint",
    "LATITUDE_MAX",
    "LATITUDE_MIN",
    "LONGITUDE_MAX",
    "LONGITUDE_MIN",
    "RegularGrid",
    "bilinear_interpolate",
    "haversine_distance_km",
    "normalize_longitude",
    "validate_coordinates",
    "validate_latitude",
    "validate_longitude",
]
