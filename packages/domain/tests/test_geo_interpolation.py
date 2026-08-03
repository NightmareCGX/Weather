"""Unit tests for domain.geo.interpolation."""

import math

import pytest
from domain.exceptions import InvalidGridError, PointOutsideGridError
from domain.geo.grid import RegularGrid
from domain.geo.interpolation import bilinear_interpolate

GRID = RegularGrid(
    lat_start=10.0,
    lon_start=-80.0,
    lat_step=10.0,
    lon_step=20.0,
    rows=3,
    cols=4,
)


def _linear_field() -> list[list[float]]:
    """A field varying linearly: value = 10 + row + 2 * col."""
    return [[10.0 + row + 2.0 * col for col in range(GRID.cols)] for row in range(GRID.rows)]


def _constant_field(value: float = 7.0) -> list[list[float]]:
    """A constant field."""
    return [[value for _ in range(GRID.cols)] for _ in range(GRID.rows)]


class TestBilinearInterpolateValidation:
    def test_rejects_less_than_two_rows(self) -> None:
        with pytest.raises(InvalidGridError):
            bilinear_interpolate(
                RegularGrid(0.0, 0.0, 1.0, 1.0, rows=1, cols=4),
                [[1.0] * 4],
                0.5,
                0.5,
            )

    def test_rejects_less_than_two_cols(self) -> None:
        with pytest.raises(InvalidGridError):
            bilinear_interpolate(
                RegularGrid(0.0, 0.0, 1.0, 1.0, rows=4, cols=1),
                [[1.0], [1.0], [1.0], [1.0]],
                0.5,
                0.5,
            )

    def test_rejects_field_with_wrong_row_count(self) -> None:
        with pytest.raises(InvalidGridError):
            bilinear_interpolate(GRID, [[0.0] * GRID.cols], 15.0, -50.0)

    def test_rejects_field_with_wrong_column_count(self) -> None:
        with pytest.raises(InvalidGridError):
            bilinear_interpolate(
                GRID, [[0.0] * (GRID.cols - 1) for _ in range(GRID.rows)], 15.0, -50.0
            )

    def test_rejects_point_outside_grid(self) -> None:
        with pytest.raises(PointOutsideGridError):
            bilinear_interpolate(GRID, _constant_field(), 31.0, -50.0)


class TestBilinearInterpolateValues:
    def test_constant_field_returns_constant(self) -> None:
        result = bilinear_interpolate(GRID, _constant_field(7.0), 15.0, -50.0)
        assert result == pytest.approx(7.0)

    def test_linear_field_exact_at_node(self) -> None:
        # At node (1, 2): value = 10 + 1 + 2*2 = 15.
        result = bilinear_interpolate(GRID, _linear_field(), 20.0, -40.0)
        assert result == pytest.approx(15.0)

    def test_linear_field_analytic_at_center(self) -> None:
        # At midpoint (row 0.5, col 1.5): 10 + 0.5 + 2*1.5 = 13.5.
        result = bilinear_interpolate(GRID, _linear_field(), 15.0, -50.0)
        assert result == pytest.approx(13.5)

    def test_linear_field_analytic_fractional(self) -> None:
        # row 0.3, col 1.8: 10 + 0.3 + 2*1.8 = 13.9.
        result = bilinear_interpolate(GRID, _linear_field(), 13.0, -44.0)
        assert result == pytest.approx(13.9)

    def test_linear_field_analytic_at_corner_cell_center(self) -> None:
        # Lower-left cell center (row 0.5, col 0.5): 10 + 0.5 + 2*0.5 = 11.5.
        result = bilinear_interpolate(GRID, _linear_field(), 15.0, -70.0)
        assert result == pytest.approx(11.5)

    def test_corner_points_equal_node_values(self) -> None:
        values = _linear_field()
        result = bilinear_interpolate(GRID, values, 10.0, -80.0)
        assert result == pytest.approx(values[0][0])

    def test_boundary_midpoint_edge(self) -> None:
        # Top edge midpoint between nodes (1, 1) and (1, 2):
        # value = 10 + 1 + 2*1.5 = 14.0.
        result = bilinear_interpolate(GRID, _linear_field(), 20.0, -50.0)
        assert result == pytest.approx(14.0)

    def test_interpolation_is_within_range_of_node_values(self) -> None:
        values = _linear_field()
        result = bilinear_interpolate(GRID, values, 17.0, -35.0)
        # row_f=0.7, col_f=2.25 in cell [(0,2),(1,3)]:
        # v00=14, v01=16, v10=15, v11=17, t_row=0.7, t_col=0.25.
        assert result == pytest.approx(15.2)

    def test_matches_manual_formula(self) -> None:
        values = _linear_field()
        row_f, col_f = 0.7, 1.4
        row_0, col_0 = 0, 1
        t_row, t_col = 0.7, 0.4
        lower = values[row_0][col_0] + (values[row_0][col_0 + 1] - values[row_0][col_0]) * t_col
        upper = values[row_0 + 1][col_0] + (
            values[row_0 + 1][col_0 + 1] - values[row_0 + 1][col_0]
        ) * t_col
        expected = lower + (upper - lower) * t_row

        lat = GRID.lat_start + row_f * GRID.lat_step
        lon = GRID.lon_start + col_f * GRID.lon_step
        result = bilinear_interpolate(GRID, values, lat, lon)
        assert result == pytest.approx(expected)

    def test_far_boundary_cell(self) -> None:
        # Interpolate in the last cell (row 1..2, col 2..3).
        result = bilinear_interpolate(GRID, _linear_field(), 22.0, -30.0)
        assert result == pytest.approx(16.2)

    def test_math_is_close_detection(self) -> None:
        result = bilinear_interpolate(GRID, _linear_field(), 15.0, -50.0)
        assert abs(result - 13.5) < 1e-9
        assert math.isclose(result, 13.5, rel_tol=1e-9, abs_tol=1e-12)


class TestBilinearInterpolateRightEdge:
    """Interpolation along the last row/column exercises the clamp branches.

    A 2x2 unit grid spanning lat/lon 0..1 ensures a point on the top or right
    edge lands on the final row/column, driving ``row_0 == rows - 1`` and
    ``col_0 == cols - 1`` in the implementation.
    """

    UNIT = RegularGrid(
        lat_start=0.0,
        lon_start=0.0,
        lat_step=1.0,
        lon_step=1.0,
        rows=2,
        cols=2,
    )

    def test_top_edge_interpolates_in_last_row_cell(self) -> None:
        # Point on the top edge (row_f == rows - 1 == 1) at col 0.5:
        # value = 2 + 0.5 * 1 = 2.5.
        values = [[1.0, 2.0], [3.0, 4.0]]
        result = bilinear_interpolate(self.UNIT, values, 1.0, 0.5)
        assert result == pytest.approx(3.5)

    def test_right_edge_interpolates_in_last_col_cell(self) -> None:
        # Point on the right edge (col_f == cols - 1 == 1) at row 0.5:
        # value = 2 + 0.5 * (4 - 2) = 3.0.
        values = [[1.0, 2.0], [3.0, 4.0]]
        result = bilinear_interpolate(self.UNIT, values, 0.5, 1.0)
        assert result == pytest.approx(3.0)

    def test_corner_point_on_last_row_col(self) -> None:
        # Point exactly at the top-right corner node (1, 1) = value 4.
        values = [[1.0, 2.0], [3.0, 4.0]]
        result = bilinear_interpolate(self.UNIT, values, 1.0, 1.0)
        assert result == pytest.approx(4.0)
