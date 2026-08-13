"""Regular latitude/longitude grid definitions and coordinate mapping."""

import math
from dataclasses import dataclass

from domain.exceptions import InvalidGridError, PointOutsideGridError
from domain.geo.coordinates import normalize_longitude, validate_coordinates

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

    def _is_full_circle(self) -> bool:
        """Return whether the grid wraps the entire 360 degrees in longitude.

        A grid whose columns span (or exceed) a full revolution has no hard
        longitudinal boundary: the first and last columns are adjacent across
        the ``0/360`` seam. Such grids accept any valid longitude and column
        indices wrap around.
        """
        return self.cols * self.lon_step >= 360.0 - GRID_BOUNDS_TOLERANCE

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
        considered inside. A grid that wraps the full 360 degrees of longitude
        accepts any valid longitude; the ``0/360`` seam between the last and
        first columns is covered by the wrap.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.

        Returns:
            True if the point is inside the grid, False otherwise.

        Raises:
            InvalidCoordinatesError: If the coordinates are invalid.
        """
        validate_coordinates(latitude, longitude)
        if self._is_full_circle():
            longitude_is_inside = True
        else:
            aligned = self.align_longitude(longitude)
            longitude_is_inside = (
                self.lon_start - GRID_BOUNDS_TOLERANCE
                <= aligned
                <= self.lon_stop + GRID_BOUNDS_TOLERANCE
            )
        return (
            self.lat_start - GRID_BOUNDS_TOLERANCE
            <= latitude
            <= self.lat_stop + GRID_BOUNDS_TOLERANCE
        ) and longitude_is_inside

    def align_longitude(self, longitude: float) -> float:
        """Return ``longitude`` expressed in the grid coordinate convention.

        The query longitude is first wrapped into the closed interval
        ``[-180, 180]`` via :func:`domain.geo.coordinates.normalize_longitude`.
        When the grid stores longitudes in the ``[0, 360]`` convention (e.g.
        native GFS ``0..359.75`` grids, where ``lon_stop`` exceeds 180), a
        negative normalized longitude is shifted by ``+360`` so the point
        falls inside the grid (e.g. ``-106.82 -> 253.18``). For ``[-180, 180]``
        grids the normalized value is used unchanged.

        For a grid that wraps the full 360 degrees, the value is expressed in
        the single-revolution interval ``[lon_start, lon_start + 360)`` using
        modular arithmetic, so a point near the ``0`` seam is not pushed beyond
        the final column (e.g. ``-0.1 -> 359.9`` stays within the grid).

        Args:
            longitude: Longitude in decimal degrees (any finite value).

        Returns:
            The longitude expressed in this grid coordinate convention.

        Raises:
            InvalidCoordinatesError: If the longitude is not finite.
        """
        normalized = normalize_longitude(longitude)
        if self._is_full_circle():
            return self.lon_start + ((normalized - self.lon_start) % 360.0)
        if self.lon_stop > 180.0 and normalized < 0.0:
            return normalized + 360.0
        return normalized

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
        # ``round`` applies banker rounding (round(0.5) == 0); use half-up so
        # the nearest node is picked deterministically for exact midpoints.
        aligned_longitude = self.align_longitude(longitude)
        row = math.floor((latitude - self.lat_start) / self.lat_step + 0.5)
        col = math.floor(
            (aligned_longitude - self.lon_start) / self.lon_step + 0.5
        )
        # Guard against rounding overshoot at the far edges of the grid.
        row = min(max(row, 0), self.rows - 1)
        # On a whole-circle grid the column index wraps around the 0/360 seam
        # (e.g. -0.1 aligns to col 0, not a non-existent col 1440).
        col = (
            col % self.cols
            if self._is_full_circle()
            else min(max(col, 0), self.cols - 1)
        )
        return GridPoint(row=row, col=col)

    def grid_index_to_coordinate(self, point: GridPoint) -> tuple[float, float]:
        """Return the ``(latitude, longitude)`` of a grid node center.

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
        aligned_longitude = self.align_longitude(longitude)
        row_f = (latitude - self.lat_start) / self.lat_step
        col_f = (aligned_longitude - self.lon_start) / self.lon_step
        if self._is_full_circle():
            col_f = col_f % float(self.cols)
        else:
            # Clamp so boundary points within tolerance map onto the grid.
            col_f = min(max(col_f, 0.0), float(self.cols - 1))
        row_f = min(max(row_f, 0.0), float(self.rows - 1))
        return row_f, col_f

    def _validate_index(self, index: int, size: int, name: str) -> None:
        if index < 0 or index >= size:
            raise InvalidGridError(
                f"{name} index {index} is out of bounds for grid of size {size}."
            )
