"""Web-Mercator raster tile rendering for forecast map layers.

This service turns a forecast variable's spatial field (stored as a
regular latitude/longitude grid in Zarr) into MapLibre raster tiles
(``/v1/maps/{model}/{variable}/{level}/{z}/{x}/{y}.png``). Each tile is a
256x256 RGBA PNG where every pixel is the variable value at that geographic
location, mapped to a color by the variable's legend ramp; pixels outside the
grid (or with no data) are transparent.

The math is deliberately small and dependency-free:

* A Web-Mercator tile at ``(z, x, y)`` covers the standard slippy-map bounds.
* Tile pixel centers are inverted back to WGS84 ``(lat, lon)``.
* The forecast dataset's ``latitude``/``longitude`` axes define a regular
  grid; the field is first sliced to the tile's geographic bounds (so only
  the Zarr chunks overlapping the tile are read), then each pixel's value is
  the nearest grid cell to its location (vectorized ``searchsorted``). The
  dataset's longitude convention (``[-180, 180]`` or GFS-native ``[0, 360]``)
  is handled by aligning the pixel's longitude into the dataset's axis range.
* The value is clamped to the variable's fixed data range and mapped through
  the variable's legend color stops (linear interpolation between stop
  colors); values outside the range clamp to the nearest stop color.

No fake forecast data is generated: every tile reads genuine values from the
run's Zarr store selected by the requested model/variable/initial time/lead
time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import numpy.typing as npt
import xarray as xr
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.core.png import encode_rgba_png
from api.core.zarr import read_dataset
from api.models.entities import (
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.point_forecast import _axis_values

#: Tile size in pixels (standard MapLibre raster tile).
TILE_SIZE = 256
#: Minimum zoom level served by the tile endpoint (matches /v1/maps metadata).
MIN_ZOOM = 0
#: Maximum zoom level served by the tile endpoint (matches /v1/maps metadata).
MAX_ZOOM = 9


@dataclass(frozen=True)
class _TileGrid:
    """A dataset's regular latitude/longitude grid.

    Attributes:
        lat_start: Ascending latitude origin.
        lat_step: Uniform latitude spacing.
        lat_count: Latitude axis length.
        lon_start: Ascending longitude origin.
        lon_step: Uniform longitude spacing.
        lon_count: Longitude axis length.
        lat_reversed: Whether the dataset stores latitude descending.
        lon_reversed: Whether the dataset stores longitude descending.
    """

    lat_start: float
    lat_step: float
    lat_count: int
    lon_start: float
    lon_step: float
    lon_count: int
    lat_reversed: bool
    lon_reversed: bool

    @property
    def lon_end(self) -> float:
        """The maximum native longitude coordinate of the axis."""
        return self.lon_start + self.lon_step * (self.lon_count - 1)


def _pixel_lonlat(zoom: int, x: int, y: int, px: int, py: int) -> tuple[float, float]:
    """Return the (lon, lat) of a tile's pixel at (px, py)."""
    n = 2**zoom
    lon = ((x + (px + 0.5) / TILE_SIZE) / n) * 360.0 - 180.0
    lat_rad = math.atan(
        math.sinh(math.pi * (1 - 2 * (y + (py + 0.5) / TILE_SIZE) / n))
    )
    lat = math.degrees(lat_rad)
    return lon, lat


def _native_lon(grid: _TileGrid, lon: float) -> float:
    """Align a WGS84 longitude into the dataset's native axis convention."""
    normalized = (lon % 360.0 + 360.0) % 360.0
    native_min = grid.lon_start
    native_max = grid.lon_end
    span = native_max - native_min
    if span <= 0:
        return normalized
    candidate = normalized
    while candidate > native_max and candidate - 360.0 >= native_min:
        candidate -= 360.0
    while candidate < native_min:
        candidate += 360.0
    return candidate


def _color_stops(variable_code: str) -> list[tuple[float, tuple[int, int, int]]]:
    """Return the (value, RGB) color stops for a variable's display ramp."""
    if variable_code == "precipitation_rate":
        return [
            (0.0, (255, 255, 255)),
            (0.5, (194, 230, 153)),
            (1.0, (120, 198, 121)),
            (2.5, (49, 163, 84)),
            (5.0, (25, 114, 120)),
            (10.0, (49, 76, 143)),
            (20.0, (123, 65, 115)),
            (40.0, (84, 39, 136)),
        ]
    return [
        (-40.0, (49, 54, 149)),
        (-20.0, (69, 117, 180)),
        (-5.0, (116, 173, 209)),
        (5.0, (240, 249, 232)),
        (15.0, (254, 217, 118)),
        (25.0, (254, 153, 41)),
        (35.0, (217, 72, 1)),
        (45.0, (165, 0, 38)),
    ]


def _data_range(variable_code: str) -> tuple[float, float]:
    """Return the fixed (min, max) data range used to normalize a variable."""
    if variable_code == "precipitation_rate":
        return (0.0, 25.0)
    return (-40.0, 45.0)


def _interpolate_color(
    stops: list[tuple[float, tuple[int, int, int]]], value: float
) -> tuple[int, int, int]:
    """Map a value to an RGB color by linear interpolation between stops."""
    if value <= stops[0][0]:
        return stops[0][1]
    for index in range(len(stops) - 1):
        lower_value, lower_color = stops[index]
        upper_value, upper_color = stops[index + 1]
        if value <= upper_value:
            span = upper_value - lower_value
            weight = 0.0 if span <= 0 else (value - lower_value) / span
            channels: list[int] = [
                int(round(lower + (upper - lower) * weight))
                for lower, upper in zip(lower_color, upper_color)
            ]
            return channels[0], channels[1], channels[2]
    return stops[-1][1]


def _derive_grid(dataset: xr.Dataset) -> _TileGrid:
    """Derive a regular grid from a dataset's coordinate arrays.

    Raises:
        ValueError: If an axis is missing, degenerate, or non-uniform.
    """
    lat_raw = _axis_values(dataset, "latitude")
    lon_raw = _axis_values(dataset, "longitude")
    lat_desc = lat_raw[-1] < lat_raw[0]
    lon_desc = lon_raw[-1] < lon_raw[0]
    lat_asc = list(reversed(lat_raw)) if lat_desc else list(lat_raw)
    lon_asc = list(reversed(lon_raw)) if lon_desc else list(lon_raw)

    def _uniform_axis(values: list[float]) -> tuple[float, float, int]:
        if len(values) < 2:
            raise ValueError("Grid axis must have at least two points.")
        step = (values[-1] - values[0]) / (len(values) - 1)
        if step <= 0.0:
            raise ValueError("Grid axis must be ascending and uniform.")
        if not np.allclose(np.diff(values), step):
            raise ValueError("Grid axis must be uniformly spaced.")
        return values[0], step, len(values)

    lat_start, lat_step, lat_count = _uniform_axis(lat_asc)
    lon_start, lon_step, lon_count = _uniform_axis(lon_asc)
    return _TileGrid(
        lat_start=lat_start,
        lat_step=lat_step,
        lat_count=lat_count,
        lon_start=lon_start,
        lon_step=lon_step,
        lon_count=lon_count,
        lat_reversed=lat_desc,
        lon_reversed=lon_desc,
    )


def _nearest_indices(
    values: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
) -> npt.NDArray[np.int_]:
    """Return the nearest index in ascending ``values`` for each target.

    Targets outside the axis clamp to the nearest end. ``values`` must be
    ascending; the result is a same-shaped integer array of valid indices. A
    single-element axis (the out-of-grid fallback field) yields index 0 for
    every target so the caller's transparency mask controls the output.
    """
    if len(values) <= 1:
        return np.zeros_like(targets, dtype=int)
    positions = np.searchsorted(values, targets)
    positions = np.clip(positions, 1, len(values) - 1)
    lower = values[positions - 1]
    upper = values[positions]
    use_lower = np.abs(targets - lower) <= np.abs(upper - targets)
    indices: npt.NDArray[np.int_] = np.where(
        use_lower, positions - 1, positions
    )
    return indices


def render_tile_png(
    db: Session,
    *,
    model: str,
    variable: str,
    level: str,
    zoom: int,
    x: int,
    y: int,
    lead_time_hours: int,
    initial_time: str | None = None,
) -> bytes:
    """Render one PNG tile for a forecast variable and selection.

    The newest ``status='ready'`` run of the model (optionally pinned to a
    specific ``cycle_time`` via ``initial_time``) whose store opens is
    selected, the requested lead is sliced from the variable's field, and the
    tile's pixels are sampled nearest-neighbor and colored by the variable's
    ramp.

    Args:
        db: Database session.
        model: A model identifier.
        variable: A ``forecast_variables.variable_code``.
        level: The vertical level; only ``surface`` is supported.
        zoom: Web-Mercator zoom level.
        x: Web-Mercator tile X.
        y: Web-Mercator tile Y.
        lead_time_hours: Forecast offset hours from the run's cycle time.
        initial_time: Optional ISO 8601 UTC cycle time to pin the run.

    Returns:
        The PNG bytes.

    Raises:
        HTTPException: 404 when no ready run / product / lead, or the variable
            is missing from the dataset.
        ValueError: For invalid tile/grid/field input.
    """
    _validate_tile(zoom, x, y)
    run, dataset, lead = _resolve_run_and_field(
        db,
        model=model,
        variable=variable,
        level=level,
        lead_time_hours=lead_time_hours,
        initial_time=initial_time,
    )
    grid = _derive_grid(dataset)
    stops = _color_stops(variable)
    data_min, data_max = _data_range(variable)

    # Compute the tile's geographic bounds (pixel centers).
    pixel_lats: npt.NDArray[np.float64] = np.empty(
        (TILE_SIZE, TILE_SIZE), dtype=np.float64
    )
    pixel_lons: npt.NDArray[np.float64] = np.empty(
        (TILE_SIZE, TILE_SIZE), dtype=np.float64
    )
    for py in range(TILE_SIZE):
        for px in range(TILE_SIZE):
            pixel_lons[py, px], pixel_lats[py, px] = _pixel_lonlat(zoom, x, y, px, py)

    # Align pixel longitudes into the dataset's native convention up front.
    lon_native: npt.NDArray[np.float64] = np.vectorize(
        lambda value: _native_lon(grid, float(value))
    )(pixel_lons)

    field, lat_axis, lon_axis = _slice_field(
        dataset, variable, lead, grid, pixel_lats, lon_native
    )

    # Nearest grid index per pixel, into the *sliced* ascending axes.
    rows = _nearest_indices(lat_axis, pixel_lats)
    cols = _nearest_indices(lon_axis, lon_native)

    valid = _inside_grid(grid, pixel_lats, lon_native)
    values = field[rows, cols]

    pixels = bytearray()
    finite = np.isfinite(values) & valid
    flat_values = values.ravel()
    flat_finite = finite.ravel()
    for index in range(TILE_SIZE * TILE_SIZE):
        if not flat_finite[index]:
            pixels += b"\x00\x00\x00\x00"
            continue
        clamped = max(data_min, min(data_max, float(flat_values[index])))
        red, green, blue = _interpolate_color(stops, clamped)
        pixels += bytes((red, green, blue, 255))

    return encode_rgba_png(bytes(pixels), TILE_SIZE, TILE_SIZE)


def _slice_field(
    dataset: xr.Dataset,
    variable: str,
    lead: int,
    grid: _TileGrid,
    pixel_lats: npt.NDArray[np.float64],
    lon_native: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the 2-D ascending field and its sliced axes for the tile bounds.

    The field is reversed along any descending axis and then sliced to the
    tile's latitude / native-longitude bounds so only the Zarr chunks
    overlapping the tile are read. The returned axes are the sliced ascending
    ``latitude`` and ``longitude`` arrays, used to index the field with
    nearest-neighbor lookup.

    When the tile does not intersect the grid, a 1x1 NaN field with the full
    axes is returned so the caller's mask renders every pixel transparent.

    Args:
        dataset: The run's Zarr dataset.
        variable: The forecast variable code.
        lead: The lead time to select.
        grid: The derived regular grid.
        pixel_lats: 2-D array of pixel latitudes.
        lon_native: 2-D array of pixel longitudes aligned to the grid's native
            convention.

    Returns:
        A ``(values, lat_axis, lon_axis)`` tuple where ``values`` is the
        ascending ``(lat, lon)`` slice and ``lat_axis``/``lon_axis`` are the
        ascending coordinate arrays of the slice.

    Raises:
        ValueError: If the variable is missing or not a 2-D surface field.
    """
    if variable not in dataset.data_vars:
        raise ValueError(f"Variable '{variable}' is not in the dataset.")
    field = dataset[variable]
    if "lead_time_hours" in field.dims:
        field = field.sel(lead_time_hours=lead)
    if field.ndim != 2:
        raise ValueError(f"Variable '{variable}' is not a 2-D surface field.")

    if grid.lat_reversed:
        field = field.isel(latitude=slice(None, None, -1))
    if grid.lon_reversed:
        field = field.isel(longitude=slice(None, None, -1))

    lat_axis_full = field.latitude.values
    lon_axis_full = field.longitude.values
    lat_min = float(pixel_lats.min())
    lat_max = float(pixel_lats.max())
    lon_native_min = float(lon_native.min())
    lon_native_max = float(lon_native.max())

    lo_lat = float(max(lat_axis_full[0], lat_min))
    hi_lat = float(min(lat_axis_full[-1], lat_max))
    lo_lon = float(max(lon_axis_full[0], lon_native_min))
    hi_lon = float(min(lon_axis_full[-1], lon_native_max))
    if lo_lat > hi_lat or lo_lon > hi_lon:
        # The tile is entirely outside the grid; return a transparent field
        # carrying the full axes so nearest-index lookups stay in bounds.
        return (
            np.full((1, 1), np.nan),
            np.asarray([lat_axis_full[0]]),
            np.asarray([lon_axis_full[0]]),
        )
    sliced = field.sel(
        latitude=slice(lo_lat, hi_lat), longitude=slice(lo_lon, hi_lon)
    )
    return (
        np.asarray(sliced.values, dtype=float),
        np.asarray(sliced.latitude.values, dtype=float),
        np.asarray(sliced.longitude.values, dtype=float),
    )


def _inside_grid(
    grid: _TileGrid,
    pixel_lats: npt.NDArray[np.float64],
    lon_native: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of pixels inside the grid's lat/native-lon box."""
    lat_min = grid.lat_start
    lat_max = grid.lat_start + grid.lat_step * (grid.lat_count - 1)
    return (pixel_lats >= lat_min) & (pixel_lats <= lat_max) & (
        lon_native >= grid.lon_start
    ) & (lon_native <= grid.lon_end)


def check_available(
    db: Session,
    *,
    model: str,
    variable: str,
    level: str,
    lead_time_hours: int,
    initial_time: str | None = None,
) -> None:
    """Validate (without reading Zarr) that a map selection is servable.

    Checks the catalog only: the model/variable exist, a ``ready`` run exists
    for the model (optionally pinned to ``initial_time``), and a matching
    ``forecast_products`` row exists for the run/variable/level/lead. Raises
    404/422 exactly as the tile renderer would, without opening the store —
    the metadata endpoint uses this so a template is only returned for
    combinations that actually exist.

    Raises:
        HTTPException: 422 for an unsupported level, 404 when the selection is
            not available.
    """
    run = _resolve_ready_run(
        db,
        model=model,
        initial_time=initial_time,
    )
    _require_model_variable(db, model, variable)
    _require_product(db, run, variable, level, lead_time_hours)


def _resolve_ready_run(
    db: Session,
    *,
    model: str,
    initial_time: str | None,
) -> ModelRun:
    """Resolve the newest ready run for a model, optionally pinned by cycle.

    Raises:
        HTTPException: 404 when no ready run matches.
    """
    from fastapi import HTTPException

    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status == "ready")
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        cycle = _parse_cycle_time(initial_time)
        stmt = stmt.where(ModelRun.cycle_time == cycle)
    stmt = stmt.order_by(ModelRun.cycle_time.desc())
    runs = list(db.execute(stmt).scalars().all())
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No ready forecast run with data was found for model '{model}'"
                + (f" and initial time '{initial_time}'." if initial_time else ".")
            ),
        )
    return runs[0]


def _resolve_run_and_field(
    db: Session,
    *,
    model: str,
    variable: str,
    level: str,
    lead_time_hours: int,
    initial_time: str | None,
) -> tuple[ModelRun, xr.Dataset, int]:
    """Resolve the ready run and dataset backing a tile request.

    Raises:
        HTTPException: 404 when the model/variable/level/lead combination is
            not available (no ready run, no matching product, or a store that
            cannot be read), 422 for an unsupported level.
    """
    from fastapi import HTTPException

    if level != "surface":
        raise HTTPException(status_code=422, detail=f"Unsupported level '{level}'.")
    _require_model_variable(db, model, variable)

    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status == "ready")
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        cycle = _parse_cycle_time(initial_time)
        stmt = stmt.where(ModelRun.cycle_time == cycle)
    stmt = stmt.order_by(ModelRun.cycle_time.desc())
    runs = list(db.execute(stmt).scalars().all())
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No ready forecast run with data was found for model '{model}'"
                + (f" and initial time '{initial_time}'." if initial_time else ".")
            ),
        )

    for run in runs:
        assert run.zarr_store_path is not None
        try:
            dataset = read_dataset(run.zarr_store_path)
        except Exception:  # noqa: BLE001 - probe store, fall through
            continue
        _require_product(db, run, variable, level, lead_time_hours)
        return run, dataset, lead_time_hours
    raise HTTPException(
        status_code=404,
        detail=f"No readable forecast run with data was found for model '{model}'.",
    )


def _require_model_variable(db: Session, model: str, variable: str) -> None:
    """Raise 404 if the model or variable is not in the catalog."""
    from fastapi import HTTPException

    model_found = db.execute(
        select(Model.model_id).where(Model.model_id == model)
    ).scalar_one_or_none()
    if model_found is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' was not found.")
    variable_found = db.execute(
        select(ForecastVariable.variable_code).where(
            ForecastVariable.variable_code == variable
        )
    ).scalar_one_or_none()
    if variable_found is None:
        raise HTTPException(
            status_code=404, detail=f"Variable '{variable}' was not found."
        )


def _require_product(
    db: Session,
    run: ModelRun,
    variable: str,
    level: str,
    lead_time_hours: int,
) -> None:
    """Raise 404 if no forecast product row exists for the selection."""
    from fastapi import HTTPException

    product = db.execute(
        select(ForecastProduct.id).where(
            ForecastProduct.run_id == run.id,
            ForecastProduct.variable_id == variable,
            ForecastProduct.product_type == level,
            ForecastProduct.lead_time_hours == lead_time_hours,
        )
    ).scalar_one_or_none()
    if product is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No forecast product is available for run '{run.id}', "
                f"variable '{variable}', level '{level}', lead "
                f"'{lead_time_hours}h'."
            ),
        )


def _parse_cycle_time(value: str) -> datetime:
    """Parse an ISO 8601 UTC cycle time string into an aware datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_tile(zoom: int, x: int, y: int) -> None:
    """Validate Web-Mercator tile coordinates."""
    if zoom < MIN_ZOOM or zoom > MAX_ZOOM:
        raise ValueError(f"Zoom {zoom} is outside the supported range.")
    n = 2**zoom
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError(f"Tile ({x}, {y}) is outside zoom {zoom}.")
