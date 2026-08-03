"""Regular latitude/longitude grid definitions and coordinate mapping."""

import math
from dataclasses import dataclass

from domain.exceptions import InvalidGridError, PointOutsideGridError
from domain.geo.coordinates import validate_coordinates

#: Absolute tolerance in degrees applied when comparing against grid bounds so
#: that boundary points affected by floating-point error are not rejected.
GRID_BOUNDS_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GridPoint:
    """A discrete row/column index within a regular grid.

    Attributes:
        row: Zero-based row index.
        col: Zero-based column index.
    """

    row: int
    col: int

    def __post_init__(self) -> None:
        if self.row < 0 or self.col < 0:
            raise InvalidGridError(
                f"Grid indices must be non-negative, got ({self.row}, {self.col})."
            )


@dataclass(frozen=True)
class RegularGrid:
    """A rectilinear latitude/longitude grid with uniform spacing.

    Rows increase with latitude and columns increase with longitude. A field
    sampled on the grid stores the value for node ``(row, col)`` at
    ``(row_latitude(row), col_longitude(col))``.

    Attributes:
        lat_start: Latitude of the first row.
        lon_start: Longitude of the first column.
        lat_step: Latitude spacing between adjacent rows in degrees.
        lon_step: Longitude spacing between adjacent columns in degrees.
        rows: Number of latitude rows.
        cols: Number of longitude columns.
    """

    lat_start: float
    lon_start: float
    lat_step: float
    lon_step: float
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.lat_start)
            and math.isfinite(self.lon_start)
            and math.isfinite(self.lat_step)
            and math.isfinite(self.lon_step)
        ):
            raise InvalidGridError("Grid origin and step values must be finite.")
        if self.lat_step <= 0.0 or self.lon_step <= 0.0:
            raise InvalidGridError("Grid step values must be positive.")
        if self.rows < 1 or self.cols < 1:
            raise InvalidGridError("Grid must define at least one row and one column.")
        validate_coordinates(self.lat_start, self.lon_start)

    @property
    def lat_stop(self) -> float:
        """Latitude of the final grid row."""
        return self.lat_start + (self.rows - 1) * self.lat_step

    @property
    def lon_stop(self) -> float:
        """Longitude of the final grid column."""
        return self.lon_start + (self.cols - 1) * self.lon_step

    def row_latitude(self, row: int) -> float:
        """Return the latitude of a grid row.

        Args:
            row: Zero-based row index.

        Returns:
            The latitude of the row.

        Raises:
            InvalidGridError: If the row index is out of bounds.
        """
        self._validate_index(row, self.rows, "row")
        return self.lat_start + row * self.lat_step

    def col_longitude(self, col: int) -> float:
        """Return the longitude of a grid column.

        Args:
            col: Zero-based column index.

        Returns:
            The longitude of the column.

        Raises:
            InvalidGridError: If the column index is out of bounds.
        """
        self._validate_index(col, self.cols, "column")
        return self.lon_start + col * self.lon_step

    def contains(self, latitude: float, longitude: float) -> bool:
        """Return whether a geographic point lies within the grid bounds.

        Boundary points within ``GRID_BOUNDS_TOLERANCE`` of the edges are
        considered inside.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.

        Returns:
            True if the point is inside the grid, False otherwise.

        Raises:
            InvalidCoordinatesError: If the coordinates are invalid.
        """
        validate_coordinates(latitude, longitude)
        return (
            self.lat_start - GRID_BOUNDS_TOLERANCE
            <= latitude
            <= self.lat_stop + GRID_BOUNDS_TOLERANCE
        ) and (
            self.lon_start - GRID_BOUNDS_TOLERANCE
            <= longitude
            <= self.lon_stop + GRID_BOUNDS_TOLERANCE
        )

    def nearest_grid_index(self, latitude: float, longitude: float) -> GridPoint:
        """Return the grid node whose center is nearest to a geographic point.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.

        Returns:
            The nearest ``GridPoint``.

        Raises:
            PointOutsideGridError: If the point lies outside the grid.
            InvalidCoordinatesError: If the coordinates are invalid.
        """
        if not self.contains(latitude, longitude):
            raise PointOutsideGridError(
                f"Point ({latitude}, {longitude}) is outside grid bounds "
                f"[{self.lat_start}, {self.lat_stop}] x [{self.lon_start}, {self.lon_stop}]."
            )
        # ``round`` applies banker's rounding (round(0.5) == 0); use half-up so
        # the nearest node is picked deterministically for exact midpoints.
        row = math.floor((latitude - self.lat_start) / self.lat_step + 0.5)
        col = math.floor((longitude - self.lon_start) / self.lon_step + 0.5)
        # Guard against rounding overshoot at the far edges of the grid.
        row = min(max(row, 0), self.rows - 1)
        col = min(max(col, 0), self.cols - 1)
        return GridPoint(row=row, col=col)

    def grid_index_to_coordinate(self, point: GridPoint) -> tuple[float, float]:
        """Return the ``(latitude, longitude)`` of a grid node's center.

        Args:
            point: Grid index.

        Returns:
            The ``(latitude, longitude)`` of the node.

        Raises:
            InvalidGridError: If the index is out of bounds.
        """
        return self.row_latitude(point.row), self.col_longitude(point.col)

    def row_col_from_coordinates(
        self, latitude: float, longitude: float
    ) -> tuple[float, float]:
        """Map geographic coordinates to continuous fractional grid positions.

        The returned values are the floating-point row and column positions of
        the point, clamped to the grid, suitable for interpolation.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.

        Returns:
            The ``(row, col)`` fractional positions within ``[0, rows - 1]``
            and ``[0, cols - 1]`` respectively.

        Raises:
            PointOutsideGridError: If the point lies outside the grid.
            InvalidCoordinatesError: If the coordinates are invalid.
        """
        if not self.contains(latitude, longitude):
            raise PointOutsideGridError(
                f"Point ({latitude}, {longitude}) is outside grid bounds "
                f"[{self.lat_start}, {self.lat_stop}] x [{self.lon_start}, {self.lon_stop}]."
            )
        row_f = (latitude - self.lat_start) / self.lat_step
        col_f = (longitude - self.lon_start) / self.lon_step
        # Clamp so boundary points within tolerance map onto the grid.
        row_f = min(max(row_f, 0.0), float(self.rows - 1))
        col_f = min(max(col_f, 0.0), float(self.cols - 1))
        return row_f, col_f

    def _validate_index(self, index: int, size: int, name: str) -> None:
        if index < 0 or index >= size:
            raise InvalidGridError(
                f"{name} index {index} is out of bounds for grid of size {size}."
            )
