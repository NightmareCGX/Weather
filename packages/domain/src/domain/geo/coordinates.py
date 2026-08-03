"""Latitude/longitude validation and great-circle distance helpers."""

import math

from domain.exceptions import InvalidCoordinatesError

#: Valid WGS 84 latitude range in decimal degrees.
LATITUDE_MIN = -90.0
LATITUDE_MAX = 90.0

#: Valid WGS 84 longitude range in decimal degrees.
LONGITUDE_MIN = -180.0
LONGITUDE_MAX = 180.0

#: Mean Earth radius in kilometers (WGS 84 mean radius).
EARTH_RADIUS_KM = 6371.0088


def validate_latitude(latitude: float) -> float:
    """Validate a latitude against the valid WGS 84 range.

    Args:
        latitude: Latitude in decimal degrees.

    Returns:
        The validated latitude.

    Raises:
        InvalidCoordinatesError: If the latitude is not finite or falls
            outside the range [-90, 90].
    """
    if not math.isfinite(latitude) or not (LATITUDE_MIN <= latitude <= LATITUDE_MAX):
        raise InvalidCoordinatesError(
            f"latitude must be between {LATITUDE_MIN} and {LATITUDE_MAX} degrees, got {latitude}"
        )
    return latitude


def validate_longitude(longitude: float) -> float:
    """Validate a longitude against the valid WGS 84 range.

    Args:
        longitude: Longitude in decimal degrees.

    Returns:
        The validated longitude.

    Raises:
        InvalidCoordinatesError: If the longitude is not finite or falls
            outside the range [-180, 180].
    """
    if not math.isfinite(longitude) or not (LONGITUDE_MIN <= longitude <= LONGITUDE_MAX):
        raise InvalidCoordinatesError(
            "longitude must be between "
            f"{LONGITUDE_MIN} and {LONGITUDE_MAX} degrees, got {longitude}"
        )
    return longitude


def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    """Validate a latitude/longitude pair.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.

    Returns:
        The validated ``(latitude, longitude)`` tuple.

    Raises:
        InvalidCoordinatesError: If either coordinate is invalid.
    """
    return validate_latitude(latitude), validate_longitude(longitude)


def normalize_longitude(longitude: float) -> float:
    """Wrap a longitude into the closed interval [-180, 180].

    Both boundary values are preserved: ``180.0`` maps to ``180.0`` and
    ``-180.0`` maps to ``-180.0``. Values congruent to 180 degrees but
    reached by wrapping in the positive direction (e.g. ``540.0``) map to
    ``+180.0``, while those reached in the negative direction (e.g.
    ``-540.0``) map to ``-180.0``.

    Args:
        longitude: Longitude in decimal degrees; may be any finite value.

    Returns:
        The equivalent longitude within [-180, 180].

    Raises:
        InvalidCoordinatesError: If the longitude is not finite.
    """
    if not math.isfinite(longitude):
        raise InvalidCoordinatesError(f"longitude must be finite, got {longitude}")
    normalized = (longitude + LONGITUDE_MAX) % 360.0 - LONGITUDE_MAX
    # The modulo expression returns the representative in the half-open
    # interval [-180, 180). A positive wrap that lands exactly on the -180
    # boundary (e.g. 180, 540) is pushed back to +180 so the interval is
    # closed on both ends; negative wraps (e.g. -180, -540) stay at -180.
    if normalized == -LONGITUDE_MAX and longitude > 0:
        return LONGITUDE_MAX
    return normalized


def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Return the great-circle distance between two points in kilometers.

    Uses the haversine formula with the WGS 84 mean Earth radius.

    Args:
        latitude_1: Latitude of the first point in decimal degrees.
        longitude_1: Longitude of the first point in decimal degrees.
        latitude_2: Latitude of the second point in decimal degrees.
        longitude_2: Longitude of the second point in decimal degrees.

    Returns:
        Great-circle distance in kilometers.

    Raises:
        InvalidCoordinatesError: If either coordinate pair is invalid.
    """
    validate_coordinates(latitude_1, longitude_1)
    validate_coordinates(latitude_2, longitude_2)

    phi_1 = math.radians(latitude_1)
    phi_2 = math.radians(latitude_2)
    delta_phi = math.radians(latitude_2 - latitude_1)
    delta_lambda = math.radians(longitude_2 - longitude_1)

    sin_phi = math.sin(delta_phi / 2.0)
    sin_lambda = math.sin(delta_lambda / 2.0)
    a = sin_phi * sin_phi + math.cos(phi_1) * math.cos(phi_2) * sin_lambda * sin_lambda
    # Clamp to [0, 1] to guard against floating-point overshoot at antipodes.
    a = min(1.0, max(0.0, a))
    central_angle = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * central_angle
