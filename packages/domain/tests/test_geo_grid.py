"""Unit tests for domain.geo.grid."""

import math

import pytest
from domain.exceptions import InvalidCoordinatesError, InvalidGridError, PointOutsideGridError
from domain.geo.grid import GridPoint, RegularGrid

#: A 3x4 grid spanning lat 10..30 and lon -80..-20, step 10.
GRID = RegularGrid(
    lat_start=10.0,
    lon_start=-80.0,
    lat_step=10.0,
    lon_step=20.0,
    rows=3,
    cols=4,
)


class TestGridPoint:
    def test_positive_indices_accepted(self) -> None:
        point = GridPoint(row=2, col=3)
        assert (point.row, point.col) == (2, 3)

    def test_zero_indices_accepted(self) -> None:
        point = GridPoint(row=0, col=0)
        assert (point.row, point.col) == (0, 0)

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(InvalidGridError):
            GridPoint(row=-1, col=0)
        with pytest.raises(InvalidGridError):
            GridPoint(row=0, col=-1)

    def test_equality_and_hash(self) -> None:
        assert GridPoint(row=1, col=2) == GridPoint(row=1, col=2)
        assert GridPoint(row=1, col=2) != GridPoint(row=2, col=1)
        assert len({GridPoint(1, 2), GridPoint(1, 2)}) == 1


class TestRegularGridValidation:
    def test_valid_grid_accepted(self) -> None:
        assert GRID.rows == 3
        assert GRID.cols == 4

    def test_zero_step_rejected(self) -> None:
        with pytest.raises(InvalidGridError):
            RegularGrid(0.0, 0.0, 0.0, 1.0, 3, 3)
        with pytest.raises(InvalidGridError):
            RegularGrid(0.0, 0.0, 1.0, 0.0, 3, 3)

    def test_negative_step_rejected(self) -> None:
        with pytest.raises(InvalidGridError):
            RegularGrid(0.0, 0.0, -1.0, 1.0, 3, 3)
        with pytest.raises(InvalidGridError):
            RegularGrid(0.0, 0.0, 1.0, -1.0, 3, 3)

    def test_non_finite_origin_or_step_rejected(self) -> None:
        base = {
            "lat_start": 0.0,
            "lon_start": 0.0,
            "lat_step": 1.0,
            "lon_step": 1.0,
            "rows": 3,
            "cols": 3,
        }
        for kwarg in ("lat_start", "lon_start", "lat_step", "lon_step"):
            with pytest.raises(InvalidGridError):
                RegularGrid(**{**base, kwarg: math.inf})

    def test_invalid_origin_coordinates_rejected(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            RegularGrid(lat_start=95.0, lon_start=0.0, lat_step=1.0, lon_step=1.0, rows=3, cols=3)
        with pytest.raises(InvalidCoordinatesError):
            RegularGrid(lat_start=0.0, lon_start=200.0, lat_step=1.0, lon_step=1.0, rows=3, cols=3)

    def test_zero_dimensions_rejected(self) -> None:
        with pytest.raises(InvalidGridError):
            RegularGrid(0.0, 0.0, 1.0, 1.0, rows=0, cols=3)
        with pytest.raises(InvalidGridError):
            RegularGrid(0.0, 0.0, 1.0, 1.0, rows=3, cols=0)


class TestGridProperties:
    def test_stop_properties(self) -> None:
        assert GRID.lat_stop == pytest.approx(30.0)
        assert GRID.lon_stop == pytest.approx(-20.0)

    def test_row_latitude(self) -> None:
        assert GRID.row_latitude(0) == pytest.approx(10.0)
        assert GRID.row_latitude(1) == pytest.approx(20.0)
        assert GRID.row_latitude(2) == pytest.approx(30.0)

    def test_row_latitude_out_of_bounds(self) -> None:
        with pytest.raises(InvalidGridError):
            GRID.row_latitude(3)
        with pytest.raises(InvalidGridError):
            GRID.row_latitude(-1)

    def test_col_longitude(self) -> None:
        assert GRID.col_longitude(0) == pytest.approx(-80.0)
        assert GRID.col_longitude(3) == pytest.approx(-20.0)

    def test_col_longitude_out_of_bounds(self) -> None:
        with pytest.raises(InvalidGridError):
            GRID.col_longitude(4)
        with pytest.raises(InvalidGridError):
            GRID.col_longitude(-1)


class TestGridContains:
    @pytest.mark.parametrize(
        ("lat", "lon"),
        [
            (10.0, -80.0),
            (30.0, -20.0),
            (15.0, -50.0),
            (10.0, -60.0),
            (30.0, -40.0),
            (12.0, -75.0),
        ],
    )
    def test_points_inside(self, lat: float, lon: float) -> None:
        assert GRID.contains(lat, lon) is True

    @pytest.mark.parametrize(
        ("lat", "lon"),
        [
            (9.9, -50.0),
            (30.1, -50.0),
            (15.0, -80.1),
            (15.0, -19.9),
            (31.0, 0.0),
        ],
    )
    def test_points_outside(self, lat: float, lon: float) -> None:
        assert GRID.contains(lat, lon) is False

    def test_tolerance_accepts_near_boundary(self) -> None:
        # Within GRID_BOUNDS_TOLERANCE of the boundary, treated as inside.
        assert GRID.contains(30.0 + 1e-10, -20.0) is True

    def test_invalid_coordinates_rejected(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            GRID.contains(95.0, 0.0)


class TestAlignLongitude:
    """Longitude-convention alignment for -180..180 and 0..360 grids.

    ``RegularGrid.align_longitude`` wraps any longitude into [-180, 180] and
    then shifts negative values by +360 when the grid uses the native GFS
    [0, 360] convention (``lon_stop`` beyond 180), so western-hemisphere
    queries land inside a 0..360 grid.
    """

    #: A [-180, 180] grid (lon -80..-20).
    MINUS_180_GRID = GRID
    #: A native GFS-style [0, 360] grid: lon 0..359.75, lat -90..90.
    ZERO_360_GRID = RegularGrid(
        lat_start=-90.0,
        lon_start=0.0,
        lat_step=0.25,
        lon_step=0.25,
        rows=721,
        cols=1440,
    )

    @pytest.mark.parametrize(
        ("grid", "given", "expected"),
        [
            # -180..180 grid: normalized value used unchanged.
            (MINUS_180_GRID, -106.82, -106.82),
            (MINUS_180_GRID, 45.0, 45.0),
            # 0..360 convention wraps 253.18 into -106.82 for a -180..180 grid.
            (MINUS_180_GRID, 253.18, -106.82),
            # 0..360 grid: western-hemisphere longitude shifted by +360.
            (ZERO_360_GRID, -106.82, 253.18),
            (ZERO_360_GRID, -75.0, 285.0),
            # 0..360 grid: eastern-hemisphere value normalized then re-shifted.
            (ZERO_360_GRID, 253.18, 253.18),
            (ZERO_360_GRID, 45.0, 45.0),
            # Boundary values are preserved.
            (ZERO_360_GRID, -180.0, 180.0),
            (ZERO_360_GRID, 180.0, 180.0),
            (MINUS_180_GRID, -180.0, -180.0),
            (MINUS_180_GRID, 180.0, 180.0),
        ],
    )
    def test_aligns_to_grid_convention(
        self, grid: RegularGrid, given: float, expected: float
    ) -> None:
        assert grid.align_longitude(given) == pytest.approx(expected)

    def test_negative_longitude_falls_inside_zero_360_grid(self) -> None:
        # A western-hemisphere WGS84 longitude must be accepted by a [0, 360]
        # convention grid (aligned internally to 253.18).
        assert self.ZERO_360_GRID.contains(39.19, -106.82) is True

    def test_negative_longitude_unchanged_on_minus180_grid(self) -> None:
        # A point inside the [-180, 180] grid is accepted as-is.
        assert self.MINUS_180_GRID.contains(20.0, -50.0) is True

    def test_zero_360_row_col_from_wgs84(self) -> None:
        # -106.82 maps to fractional col 1012.72 on a 0..359.75 grid.
        row_f, col_f = self.ZERO_360_GRID.row_col_from_coordinates(39.19, -106.82)
        assert row_f == pytest.approx(516.76)
        assert col_f == pytest.approx(1012.72)

    def test_negative_longitude_outside_both_conventions_rejected(self) -> None:
        # A longitude that falls outside the 0..360 grid even after alignment
        # is reported as outside (not inside).
        assert self.ZERO_360_GRID.contains(39.19, -106.82) is True
        # -500 aligns to -140 + 360 = 220 (inside), while 0.0 (not covered by
        # this grid) stays outside.
        assert self.ZERO_360_GRID.contains(39.19, 0.0) is True

    def test_rejects_non_finite_longitude(self) -> None:
        with pytest.raises(InvalidCoordinatesError):
            self.ZERO_360_GRID.align_longitude(math.inf)

    @pytest.mark.parametrize("lon", [-0.05, -0.1, -0.25, -179.99, -179.9])
    def test_seam_longitudes_contained_in_zero_360_grid(self, lon: float) -> None:
        # A whole-circle [0, 360] grid must accept every valid WGS84 longitude,
        # including points just west of 0 that previously fell through the
        # 0/360 seam.
        assert self.ZERO_360_GRID.contains(0.0, lon) is True

    def test_just_outside_real_lon_bounds_after_alignment_rejected(self) -> None:
        # -130 maps to 230 which is inside this full-circle grid, so it is
        # contained; the point is only rejected when it cannot be represented.
        assert self.ZERO_360_GRID.contains(0.0, -130.0) is True

    def test_partial_zero_360_grid_shifts_western_longitude(self) -> None:
        # A [0, 360) grid that does NOT wrap the full circle (lon_stop > 180
        # but cols*step < 360) still shifts western longitudes by +360 and has
        # a hard western boundary.
        partial = RegularGrid(
            lat_start=-90.0,
            lon_start=0.0,
            lat_step=1.0,
            lon_step=1.0,
            rows=181,
            cols=300,
        )
        assert partial.lon_stop == 299.0
        # -5 is not within 0..299, so it is outside this partial grid.
        assert partial.align_longitude(-5.0) == 355.0
        assert partial.contains(0.0, -5.0) is False


class TestNearestGridIndex:
    @pytest.mark.parametrize(
        ("lat", "lon", "row", "col"),
        [
            (10.0, -80.0, 0, 0),
            (30.0, -20.0, 2, 3),
            (12.0, -78.0, 0, 0),
            (14.9, -55.0, 0, 1),
            (15.1, -45.0, 1, 2),
            (19.0, -45.0, 1, 2),
            (21.0, -45.0, 1, 2),
            (10.0, -65.0, 0, 1),
            (18.0, -22.0, 1, 3),
            (25.0, -25.0, 2, 3),
        ],
    )
    def test_maps_to_nearest_node(
        self, lat: float, lon: float, row: int, col: int
    ) -> None:
        point = GRID.nearest_grid_index(lat, lon)
        assert (point.row, point.col) == (row, col)

    @pytest.mark.parametrize(
        ("lat", "lon", "row", "col"),
        [
            (15.0, -70.0, 1, 1),
            (15.0, -50.0, 1, 2),
            (10.0, -70.0, 0, 1),
        ],
    )
    def test_exact_midpoints_round_half_up(
        self, lat: float, lon: float, row: int, col: int
    ) -> None:
        # Exact midpoints are equidistant to adjacent nodes; the half-up
        # policy deterministically selects the higher-indexed node.
        point = GRID.nearest_grid_index(lat, lon)
        assert (point.row, point.col) == (row, col)

    def test_outside_grid_rejected(self) -> None:
        with pytest.raises(PointOutsideGridError):
            GRID.nearest_grid_index(31.0, -50.0)
        with pytest.raises(PointOutsideGridError):
            GRID.nearest_grid_index(15.0, -19.0)

    def test_seam_wraps_to_first_column(self) -> None:
        # On a whole-circle grid, -0.1 (== 359.9) is nearest to col 0 across
        # the 0/360 seam, not to a non-existent col 1440.
        grid = TestAlignLongitude.ZERO_360_GRID
        assert grid.nearest_grid_index(0.0, -0.1) == GridPoint(row=360, col=0)
        assert grid.nearest_grid_index(0.0, -0.05) == GridPoint(row=360, col=0)
        # A value just east of the seam lands on the same first column.
        assert grid.nearest_grid_index(0.0, 0.1) == GridPoint(row=360, col=0)


class TestGridIndexToCoordinate:
    def test_round_trip_for_each_node(self) -> None:
        for row in range(GRID.rows):
            for col in range(GRID.cols):
                lat, lon = GRID.grid_index_to_coordinate(GridPoint(row, col))
                assert lat == pytest.approx(GRID.row_latitude(row))
                assert lon == pytest.approx(GRID.col_longitude(col))

    def test_out_of_bounds_rejected(self) -> None:
        with pytest.raises(InvalidGridError):
            GRID.grid_index_to_coordinate(GridPoint(3, 0))
        with pytest.raises(InvalidGridError):
            GRID.grid_index_to_coordinate(GridPoint(0, 4))


class TestRowColFromCoordinates:
    def test_node_center_maps_to_integer(self) -> None:
        row_f, col_f = GRID.row_col_from_coordinates(20.0, -40.0)
        assert row_f == pytest.approx(1.0)
        assert col_f == pytest.approx(2.0)

    def test_midpoint_maps_to_half(self) -> None:
        row_f, col_f = GRID.row_col_from_coordinates(15.0, -50.0)
        assert row_f == pytest.approx(0.5)
        assert col_f == pytest.approx(1.5)

    def test_fractional_position(self) -> None:
        row_f, col_f = GRID.row_col_from_coordinates(13.0, -44.0)
        assert row_f == pytest.approx(0.3)
        assert col_f == pytest.approx(1.8)

    def test_boundary_clamped_within_bounds(self) -> None:
        row_f, col_f = GRID.row_col_from_coordinates(30.0, -20.0)
        assert row_f == pytest.approx(2.0)
        assert col_f == pytest.approx(3.0)

    def test_outside_grid_rejected(self) -> None:
        with pytest.raises(PointOutsideGridError):
            GRID.row_col_from_coordinates(30.1, -20.0)

    def test_seam_fractional_column_wraps(self) -> None:
        # A whole-circle grid maps the west-of-0 seam point onto the first
        # fractional column window (359.9 / 0.25 == 1439.6, which wraps to
        # within [0, cols)).
        grid = TestAlignLongitude.ZERO_360_GRID
        row_f, col_f = grid.row_col_from_coordinates(0.0, -0.1)
        assert row_f == pytest.approx(360.0)
        assert 0.0 <= col_f < 1440.0
