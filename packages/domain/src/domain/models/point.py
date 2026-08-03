"""Geographic forecast point domain structures."""

from dataclasses import dataclass

from domain.exceptions import InvalidCoordinatesError
from domain.geo.coordinates import validate_latitude, validate_longitude

#: Location resolved directly from raw coordinates.
RESOLVED_VIA_COORDINATES = "coordinates"
#: Location resolved from a city record.
RESOLVED_VIA_CITY = "city"
#: Location resolved from a ski resort record.
RESOLVED_VIA_RESORT = "resort"
#: Location resolved from an observation station record.
RESOLVED_VIA_STATION = "station"
#: Location resolved from a street address.
RESOLVED_VIA_ADDRESS = "address"

#: All documented values accepted by ``ForecastPoint.resolved_via``.
VALID_RESOLVED_VIA = frozenset(
    {
        RESOLVED_VIA_COORDINATES,
        RESOLVED_VIA_CITY,
        RESOLVED_VIA_RESORT,
        RESOLVED_VIA_STATION,
        RESOLVED_VIA_ADDRESS,
    }
)


@dataclass(frozen=True)
class ForecastPoint:
    """A geographic location for which a forecast may be requested.

    Attributes:
        latitude: Latitude in decimal degrees within [-90, 90].
        longitude: Longitude in decimal degrees within [-180, 180].
        elevation_m: Optional elevation in meters above sea level.
        resolved_via: How the location was resolved (coordinates, city,
            resort, station, or address).
        id: Optional platform identifier of the underlying place.
        name: Optional human-readable place name.
    """

    latitude: float
    longitude: float
    elevation_m: float | None = None
    resolved_via: str = RESOLVED_VIA_COORDINATES
    id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        validate_latitude(self.latitude)
        validate_longitude(self.longitude)
        if self.resolved_via not in VALID_RESOLVED_VIA:
            raise InvalidCoordinatesError(
                f"resolved_via must be one of {sorted(VALID_RESOLVED_VIA)}, "
                f"got {self.resolved_via!r}"
            )
