"""Spatial interpolation helpers for regular latitude/longitude grids."""

import math
from collections.abc import Sequence

from domain.exceptions import InvalidGridError
from domain.geo.grid import RegularGrid


def bilinear_interpolate(
    grid: RegularGrid,
    values: Sequence[Sequence[float]],
    latitude: float,
    longitude: float,
) -> float:
    """Bilinearly interpolate a scalar field at a geographic point.

    The field is sampled on the grid nodes, where ``values[row][col]`` is the
    value at ``(grid.row_latitude(row), grid.col_longitude(col))``.

    Args:
        grid: The regular grid the field is sampled on.
        values: 2-D field shaped ``(grid.rows, grid.cols)``.
        latitude: Target latitude within the grid bounds.
        longitude: Target longitude within the grid bounds.

    Returns:
        The interpolated field value.

    Raises:
        InvalidGridError: If the grid or field shape cannot support bilinear
            interpolation.
        PointOutsideGridError: If the target point lies outside the grid.
    """
    if grid.rows < 2 or grid.cols < 2:
        raise InvalidGridError(
            "Bilinear interpolation requires at least two rows and two columns; "
            f"grid is {grid.rows} x {grid.cols}."
        )
    _validate_values_shape(grid, values)

    row_f, col_f = grid.row_col_from_coordinates(latitude, longitude)

    row_0 = math.floor(row_f)
    col_0 = math.floor(col_f)
    if row_0 == grid.rows - 1:
        row_0 = grid.rows - 2
    if col_0 == grid.cols - 1:
        col_0 = grid.cols - 2
    row_1, col_1 = row_0 + 1, col_0 + 1

    t_row = row_f - row_0
    t_col = col_f - col_0

    value_00 = values[row_0][col_0]
    value_01 = values[row_0][col_1]
    value_10 = values[row_1][col_0]
    value_11 = values[row_1][col_1]

    lower = value_00 + (value_01 - value_00) * t_col
    upper = value_10 + (value_11 - value_10) * t_col
    return lower + (upper - lower) * t_row


def _validate_values_shape(
    grid: RegularGrid, values: Sequence[Sequence[float]]
) -> None:
    """Ensure a field's shape matches the grid dimensions.

    Args:
        grid: The grid the field is expected to cover.
        values: The 2-D field to validate.

    Raises:
        InvalidGridError: If the field shape does not match the grid.
    """
    if len(values) != grid.rows:
        raise InvalidGridError(
            f"Field has {len(values)} rows but grid expects {grid.rows}."
        )
    for index, row in enumerate(values):
        if len(row) != grid.cols:
            raise InvalidGridError(
                f"Field row {index} has {len(row)} columns but grid expects {grid.cols}."
            )
