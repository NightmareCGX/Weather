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
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import numpy.typing as npt
import xarray as xr
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.core.png import encode_rgba_png
from api.models.entities import (
    EnsembleMemberProduct,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.lifecycle import filter_visible_runs, require_cycle_visible
from api.services.point_forecast import _axis_values

#: ModelRun lifecycle statuses eligible for serving.
SERVING_ELIGIBLE_STATUSES: tuple[str, ...] = ("ready", "processing", "partial")

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


def _color_stops(variable_code: str) -> list[tuple[float, tuple[int, int, int]]]:
    """Return the (value, RGB) color stops for a variable's display ramp."""
    if variable_code in ("precipitation_rate", "precipitation_amount_3h"):
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
    if variable_code == "relative_humidity_2m":
        return [
            (0.0, (140, 81, 10)),
            (20.0, (216, 179, 101)),
            (40.0, (246, 232, 195)),
            (60.0, (199, 234, 229)),
            (80.0, (90, 180, 172)),
            (100.0, (1, 102, 94)),
        ]
    if variable_code == "wind_gust":
        return [
            (0.0, (247, 247, 247)),
            (20.0, (204, 235, 197)),
            (40.0, (168, 221, 181)),
            (60.0, (78, 179, 211)),
            (80.0, (43, 140, 190)),
            (100.0, (8, 88, 158)),
            (120.0, (122, 1, 119)),
            (150.0, (73, 0, 106)),
        ]
    if variable_code in ("wind_10m", "wind_speed_10m"):
        return [
            (0.0, (255, 255, 255)),
            (10.0, (199, 233, 192)),
            (25.0, (116, 196, 118)),
            (40.0, (65, 171, 93)),
            (60.0, (66, 146, 198)),
            (80.0, (8, 81, 156)),
            (100.0, (122, 1, 119)),
            (140.0, (73, 0, 106)),
        ]
    if variable_code == "visibility":
        return [
            (0.0, (73, 0, 106)),
            (500.0, (174, 1, 126)),
            (1000.0, (247, 104, 161)),
            (3000.0, (251, 180, 185)),
            (6000.0, (254, 235, 226)),
            (10000.0, (240, 249, 232)),
            (24000.0, (255, 255, 255)),
        ]
    if variable_code == "snow_depth":
        return [
            (0.0, (255, 255, 255)),
            (0.02, (224, 243, 248)),
            (0.10, (171, 217, 233)),
            (0.25, (116, 173, 209)),
            (0.50, (69, 117, 180)),
            (1.00, (49, 54, 149)),
            (2.50, (26, 0, 64)),
        ]
    if variable_code == "cloud_cover_3h":
        return [
            (0.0, (255, 255, 255)),
            (20.0, (220, 225, 230)),
            (40.0, (180, 190, 205)),
            (60.0, (140, 155, 175)),
            (80.0, (90, 105, 125)),
            (100.0, (50, 60, 75)),
        ]
    if variable_code == "cloud_ceiling":
        return [
            (0.0, (165, 0, 38)),
            (150.0, (215, 48, 39)),
            (300.0, (254, 224, 144)),
            (900.0, (145, 191, 219)),
            (1500.0, (69, 117, 180)),
            (3000.0, (255, 255, 255)),
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
    if variable_code in ("precipitation_rate", "precipitation_amount_3h"):
        return (0.0, 40.0)
    if variable_code == "relative_humidity_2m":
        return (0.0, 100.0)
    if variable_code == "wind_gust":
        return (0.0, 150.0)
    if variable_code in ("wind_10m", "wind_speed_10m"):
        return (0.0, 140.0)
    if variable_code == "visibility":
        return (0.0, 24000.0)
    if variable_code == "snow_depth":
        return (0.0, 2.5)
    if variable_code == "cloud_cover_3h":
        return (0.0, 100.0)
    if variable_code == "cloud_ceiling":
        return (0.0, 3000.0)
    return (-40.0, 45.0)


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


@dataclass(frozen=True)
class TileReadContext:
    """Immutable database-resolved metadata context for tile rendering."""

    model: str
    variable: str
    level: str
    zoom: int
    x: int
    y: int
    lead_time_hours: int
    initial_time: str | None
    store_path: str
    expected_members: int
    latest_retired_iso: str | None


def resolve_tile_read_context(
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
    excluded: set[str] | None = None,
) -> TileReadContext:
    """Resolve all catalog, lifecycle, and store metadata in a short-lived DB scope.

    This function isolates all database operations into a compact scope so the
    request's ORM database connection can be released before manifest reading,
    cache checks, and Zarr storage reads.
    """
    _validate_tile(zoom, x, y)
    if initial_time is not None:
        require_cycle_visible(db, initial_time)

    store_path = _resolve_run_store_path(
        db,
        model=model,
        variable=variable,
        level=level,
        lead_time_hours=lead_time_hours,
        initial_time=initial_time,
        excluded=excluded or set(),
    )
    from domain.coverage import get_expected_members

    expected_members = get_expected_members(model, default_if_unknown=1)

    latest_retired_iso: str | None = None
    if initial_time is None:
        from api.models.entities import ForecastCycleLifecycle

        latest_retired = db.execute(
            select(func.max(ForecastCycleLifecycle.retired_at))
        ).scalar_one_or_none()
        if latest_retired is not None:
            if latest_retired.tzinfo is None:
                latest_retired = latest_retired.replace(tzinfo=timezone.utc)
            latest_retired_iso = latest_retired.astimezone(timezone.utc).isoformat()

    return TileReadContext(
        model=model,
        variable=variable,
        level=level,
        zoom=zoom,
        x=x,
        y=y,
        lead_time_hours=lead_time_hours,
        initial_time=initial_time,
        store_path=store_path,
        expected_members=expected_members,
        latest_retired_iso=latest_retired_iso,
    )


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

    The output is cached server-side keyed by the full forecast identity
    (model, variable, level, lead, cycle, tile coordinates) so identical tile
    requests are served from memory instead of recomputing the PNG
    (ACCEPTANCE_REMEDIATION_PLAN §12). The color mapping is fully vectorized
    (no per-pixel Python loop) and the longitude alignment is vectorized
    (no ``np.vectorize``), removing the dominant rendering costs.

    **Connection lifetime (row liveness).** All database metadata — the
    committed-manifest serving generation, the ready run, and the forecast
    product — is resolved into plain values *first*, while the request's DB
    session is live. The session is then closed, returning its QueuePool
    connection, **before** the expensive Zarr materialize + PNG encode runs.
    The store read itself goes through ``api.core.reader_gate`` which uses its
    own dedicated lock pool and revalidates the run is READY on a fresh core
    query (per ``docs/ARCHITECTURE.md``), so no ORM session is needed after the
    metadata phase. This keeps a single tile request's database connection
    checkout to a few milliseconds of catalog queries instead of the full
    render duration, so a browser viewport's concurrent tile requests cannot
    exhaust the QueuePool (``pool_size=5``/``max_overflow=10``) even when many
    tiles need cold reads of a large store.

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
    from api.services.point_forecast import resolve_serving_generation_for_store

    session = db
    excluded: set[str] = set()
    while True:
        try:
            context = resolve_tile_read_context(
                session,
                model=model,
                variable=variable,
                level=level,
                zoom=zoom,
                x=x,
                y=y,
                lead_time_hours=lead_time_hours,
                initial_time=initial_time,
                excluded=excluded,
            )
        finally:
            # Release this session's DB connection immediately upon context resolution.
            # Even on cache hits, no connection remains checked out during response send.
            session.close()

        serving_generation = resolve_serving_generation_for_store(
            context.store_path, context.latest_retired_iso
        )
        cache_key = _tile_cache_key(
            model,
            variable,
            level,
            zoom,
            x,
            y,
            lead_time_hours,
            initial_time,
            serving_generation,
        )
        cached = _tile_cache_get(cache_key)
        if cached is not None:
            return cached

        from api.core import reader_gate
        from api.core.database import SessionLocal

        try:
            # Phase 1 remediation: only the tile's geographic window is read
            # from the store. The selector receives the lazy dataset, selects
            # the variable/lead (and member-reduces GEFS), crops the spatial
            # window, then materializes that bounded window *inside* the gate.
            windowed = reader_gate.gated_read_dataset_with_selector(
                context.store_path,
                selector=lambda dataset: _select_tile_window(
                    dataset,
                    variable=variable,
                    lead=lead_time_hours,
                    zoom=zoom,
                    x=x,
                    y=y,
                    expected_members=context.expected_members,
                    store_path=context.store_path,
                ),
            )
        except Exception:  # noqa: BLE001 - unreadable/no-longer-ready store
            excluded.add(context.store_path)
            session = SessionLocal()
            continue
        break
    return _render_window_to_png(
        windowed,
        variable=variable,
        zoom=zoom,
        x=x,
        y=y,
        cache_key=cache_key,
    )


@dataclass(frozen=True)
class _TileWindow:
    """A materialized spatial window selected from a forecast store.

    Attributes:
        field: Ascending ``(lat, lon)`` in-memory numpy array of the window.
        lat_axis: Ascending latitude coordinates of ``field`` rows.
        lon_axis: Ascending longitude coordinates of ``field`` columns.
        grid: The dataset's regular grid (:class:`_TileGrid`).
    """

    field: npt.NDArray[np.float64]
    lat_axis: npt.NDArray[np.float64]
    lon_axis: npt.NDArray[np.float64]
    grid: _TileGrid


def _select_tile_window(
    dataset: xr.Dataset,
    *,
    variable: str,
    lead: int,
    zoom: int,
    x: int,
    y: int,
    expected_members: int = 1,
    store_path: str | None = None,
) -> _TileWindow:
    """Gate-time selector: read only the tile's geographic window from the store.

    Runs on the **lazy** dataset *inside* the reader gate. It derives the grid
    (coordinate arrays only), computes the tile's pixel bounds, selects the
    variable + lead (and member-reduces GEFS to the ensemble mean), crops the
    spatial window to the tile bounds, and materializes only that small window
    via ``sliced.values``. Because ``sliced`` is a ``MemoryCachedArray``-backed
    selection, ``.values`` reads only the zarr chunks overlapping the window —
    never the full global field.

    Returns:
        A :class:`_TileWindow` containing the bounded window + its axes + grid.
    """
    grid = _derive_grid(dataset)

    # Compute the tile's geographic bounds (pixel centers), vectorized.
    n = 2**zoom
    px_idx, py_idx = np.meshgrid(
        np.arange(TILE_SIZE, dtype=np.float64),
        np.arange(TILE_SIZE, dtype=np.float64),
        indexing="xy",
    )
    pixel_lons = ((x + (px_idx + 0.5) / TILE_SIZE) / n) * 360.0 - 180.0
    y_merc = y + (py_idx + 0.5) / TILE_SIZE
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y_merc / n)))
    pixel_lats = np.degrees(lat_rad)
    lon_native = _align_longitudes(grid, pixel_lons)

    field, lat_axis, lon_axis = _slice_field(
        dataset,
        variable,
        lead,
        grid,
        pixel_lats,
        lon_native,
        expected_members=expected_members,
        store_path=store_path,
    )
    # ``_slice_field`` already materializes the bounded window. Return the
    # window + its axes + the grid so rendering needs no store access.
    return _TileWindow(
        field=np.asarray(field, dtype=np.float64),
        lat_axis=np.asarray(lat_axis, dtype=np.float64),
        lon_axis=np.asarray(lon_axis, dtype=np.float64),
        grid=grid,
    )


def _render_window_to_png(
    window: _TileWindow,
    *,
    variable: str,
    zoom: int,
    x: int,
    y: int,
    cache_key: tuple[object, ...],
) -> bytes:
    """Rasterize an already-materialized spatial window into a tile PNG (no DB).

    All data access already happened under the reader gate; this function is
    pure CPU (nearest-neighbor pixel sampling + color mapping + PNG encode).
    """
    field_arr = window.field
    lat_axis_arr = window.lat_axis
    lon_axis_arr = window.lon_axis
    stops = _color_stops(variable)
    data_min, data_max = _data_range(variable)

    # Compute the tile's geographic bounds (pixel centers), vectorized.
    n = 2**zoom
    px_idx, py_idx = np.meshgrid(
        np.arange(TILE_SIZE, dtype=np.float64),
        np.arange(TILE_SIZE, dtype=np.float64),
        indexing="xy",
    )
    pixel_lons = ((x + (px_idx + 0.5) / TILE_SIZE) / n) * 360.0 - 180.0
    y_merc = y + (py_idx + 0.5) / TILE_SIZE
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y_merc / n)))
    pixel_lats = np.degrees(lat_rad)

    # Align pixel longitudes into the grid's native convention (the window's
    # lon_axis already matches, so use the grid's start/stop bounds).
    aligned_lons = _align_longitudes(window.grid, pixel_lons)

    # Nearest grid index per pixel, into the *sliced* ascending axes. Columns
    # MUST use the grid-native aligned longitudes just like the selection stage
    # (and the pre-Phase-1 renderer): raw ``[-180, 180)`` pixel longitudes have
    # no consistent displacement within the window's native lon axis and clamp
    # every western-hemisphere pixel to index 0, rendering mirrored-wrong
    # bands. ``_align_longitudes`` shifts each pixel by at most one ±360 step.
    rows = _nearest_indices(lat_axis_arr, pixel_lats)
    cols = _nearest_indices(lon_axis_arr, aligned_lons)
    valid = _inside_grid(window.grid, pixel_lats, aligned_lons)
    values = field_arr[rows, cols]

    # Fully vectorized color mapping: clamp, interpolate across the stop ramp
    # per RGB channel with ``np.interp``, and build the RGBA scanlines with
    # NumPy (no 65,536-iteration Python loop).
    if variable == "cloud_ceiling":
        finite = np.isfinite(values) & valid & (values < 19990.0)
    else:
        finite = np.isfinite(values) & valid
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    rgba[..., 3] = 0  # Default fully transparent (nodata/unlimited/outside grid)
    if np.any(finite):
        clamped = np.clip(values[finite], data_min, data_max)
        stop_values = np.asarray([stop[0] for stop in stops], dtype=np.float64)
        # ``np.interp`` needs a 1-D fp per call, so interpolate each channel.
        red = np.interp(clamped, stop_values, np.asarray([s[1][0] for s in stops], dtype=np.float64))
        green = np.interp(clamped, stop_values, np.asarray([s[1][1] for s in stops], dtype=np.float64))
        blue = np.interp(clamped, stop_values, np.asarray([s[1][2] for s in stops], dtype=np.float64))
        rows_f = np.nonzero(finite)[0]
        cols_f = np.nonzero(finite)[1]
        rgba[rows_f, cols_f, 0] = red
        rgba[rows_f, cols_f, 1] = green
        rgba[rows_f, cols_f, 2] = blue
        rgba[rows_f, cols_f, 3] = 255

    png = encode_rgba_png(rgba.tobytes(), TILE_SIZE, TILE_SIZE)
    _tile_cache_set(cache_key, png)
    return png


def _align_longitudes(grid: _TileGrid, lons: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vectorized longitude alignment into the dataset's native convention.

    Equivalent to applying :func:`_native_lon` per pixel, but as pure array
    arithmetic (no ``np.vectorize`` Python loop) and **bounded**: each
    longitude is shifted by at most one ``±360°`` step so no value can
    oscillate indefinitely. Pixels whose longitude still falls outside the
    native region are masked by ``_inside_grid`` as no-data, so a bounded
    single shift is exactly as correct as the original per-pixel loop for
    in-range values while never hanging on out-of-range ones (a latent hang in
    the original ``_native_lon`` for small native regions).
    """
    normalized = np.mod(np.mod(lons, 360.0) + 360.0, 360.0)
    native_min = grid.lon_start
    native_max = grid.lon_end
    span = native_max - native_min
    if span <= 0:
        return normalized
    out = normalized.copy()
    # One -360 step for values above the native max that would still land
    # within (or near) the region.
    above = out > native_max
    candidate = out - 360.0
    out = np.where(above & (candidate >= native_min), candidate, out)
    # One +360 step for values below the native min.
    below = out < native_min
    out = np.where(below, out + 360.0, out)
    # Any value still outside the region is left as-is (masked as no-data).
    return out


#: Server-side tile LRU cache: ``model/variable/level/z/x/y/lead/initial_time``
#: -> PNG bytes. Bounded so the API process does not grow unbounded; the TTL
#: aligns with the tile ``Cache-Control: max-age=300`` so newly-ingested runs
#: become visible promptly. The cache key carries the full forecast identity
#: (including the cycle via ``initial_time``), so a tile for one forecast run
#: can never satisfy another's request.
_TILE_CACHE_MAX_ENTRIES = 4096
_TILE_CACHE_TTL_SECONDS = 300
_tile_cache: dict[tuple[object, ...], tuple[float, bytes]] = {}


def _tile_cache_key(
    model: str,
    variable: str,
    level: str,
    zoom: int,
    x: int,
    y: int,
    lead_time_hours: int,
    initial_time: str | None,
    serving_generation: str | None,
) -> tuple[object, ...]:
    """Build the tile cache key: the full forecast + spatial + generation."""
    return (
        model,
        variable,
        level,
        zoom,
        x,
        y,
        lead_time_hours,
        initial_time,
        serving_generation,
    )


def _tile_cache_get(key: tuple[object, ...]) -> bytes | None:
    """Return a live cached tile, evicting stale entries."""
    entry = _tile_cache.get(key)
    if entry is None:
        return None
    created, png = entry
    if time.monotonic() - created > _TILE_CACHE_TTL_SECONDS:
        _tile_cache.pop(key, None)
        return None
    return png


def _tile_cache_set(key: tuple[object, ...], png: bytes) -> None:
    """Store a tile, evicting the oldest entry when the cache is full."""
    _tile_cache[key] = (time.monotonic(), png)
    if len(_tile_cache) > _TILE_CACHE_MAX_ENTRIES:
        # Evict the oldest-inserted key (approximate LRU via insertion order).
        try:
            oldest = next(iter(_tile_cache))
            _tile_cache.pop(oldest, None)
        except StopIteration:
            pass


def _slice_field(
    dataset: xr.Dataset,
    variable: str,
    lead: int,
    grid: _TileGrid,
    pixel_lats: npt.NDArray[np.float64],
    lon_native: npt.NDArray[np.float64],
    expected_members: int = 1,
    store_path: str | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the 2-D ascending field and its sliced axes for the tile bounds."""
    from api.core.manifest_reader import manifest_generation, manifest_storage_format
    from api.core.zarr import get_sharded_reader

    format_version = manifest_storage_format(store_path) if store_path else "v2_unsharded"
    if format_version == "sharded_v1" and store_path is not None:
        reader = get_sharded_reader(store_path)
        generation = manifest_generation(store_path)

        lat_axis_full = dataset.coords["latitude"].values
        lon_axis_full = dataset.coords["longitude"].values
        lat_min = float(pixel_lats.min())
        lat_max = float(pixel_lats.max())
        lon_native_min = float(lon_native.min())
        lon_native_max = float(lon_native.max())

        lat_step = abs(float(lat_axis_full[1] - lat_axis_full[0])) if len(lat_axis_full) > 1 else 0.25
        lon_step = abs(float(lon_axis_full[1] - lon_axis_full[0])) if len(lon_axis_full) > 1 else 0.25
        lo_lat = float(max(min(lat_axis_full[0], lat_axis_full[-1]), lat_min - lat_step))
        hi_lat = float(min(max(lat_axis_full[-1], lat_axis_full[0]), lat_max + lat_step))
        lo_lon = float(max(min(lon_axis_full[0], lon_axis_full[-1]), lon_native_min - lon_step))
        hi_lon = float(min(max(lon_axis_full[-1], lon_axis_full[0]), lon_native_max + lon_step))

        if lo_lat > hi_lat or lo_lon > hi_lon:
            return (
                np.full((1, 1), np.nan),
                np.asarray([min(lat_axis_full[0], lat_axis_full[-1])]),
                np.asarray([min(lon_axis_full[0], lon_axis_full[-1])]),
            )

        if grid.lat_reversed:
            lat_idx_0 = int(np.clip(round((lat_axis_full[0] - hi_lat) / lat_step), 0, len(lat_axis_full) - 1))
            lat_idx_1 = int(np.clip(round((lat_axis_full[0] - lo_lat) / lat_step), 0, len(lat_axis_full) - 1))
        else:
            lat_idx_0 = int(np.clip(round((lo_lat - lat_axis_full[0]) / lat_step), 0, len(lat_axis_full) - 1))
            lat_idx_1 = int(np.clip(round((hi_lat - lat_axis_full[0]) / lat_step), 0, len(lat_axis_full) - 1))

        if grid.lon_reversed:
            lon_idx_0 = int(np.clip(round((lon_axis_full[0] - hi_lon) / lon_step), 0, len(lon_axis_full) - 1))
            lon_idx_1 = int(np.clip(round((lon_axis_full[0] - lo_lon) / lon_step), 0, len(lon_axis_full) - 1))
        else:
            lon_idx_0 = int(np.clip(round((lo_lon - lon_axis_full[0]) / lon_step), 0, len(lon_axis_full) - 1))
            lon_idx_1 = int(np.clip(round((hi_lon - lon_axis_full[0]) / lon_step), 0, len(lon_axis_full) - 1))

        lat_min_idx = min(lat_idx_0, lat_idx_1)
        lat_max_idx = max(lat_idx_0, lat_idx_1)
        lon_min_idx = min(lon_idx_0, lon_idx_1)
        lon_max_idx = max(lon_idx_0, lon_idx_1)

        is_ensemble = "member" in dataset.coords or expected_members > 1
        if variable in ("wind_10m", "wind_speed_10m"):
            if is_ensemble:
                members_to_read = tuple(range(1, expected_members + 1))
                u_stack = [
                    reader.read_window(
                        "wind_u_10m",
                        member=m,
                        lead_time_hours=lead,
                        lat_min=lat_min_idx,
                        lat_max=lat_max_idx,
                        lon_min=lon_min_idx,
                        lon_max=lon_max_idx,
                        generation=generation,
                    )
                    for m in members_to_read
                ]
                v_stack = [
                    reader.read_window(
                        "wind_v_10m",
                        member=m,
                        lead_time_hours=lead,
                        lat_min=lat_min_idx,
                        lat_max=lat_max_idx,
                        lon_min=lon_min_idx,
                        lon_max=lon_max_idx,
                        generation=generation,
                    )
                    for m in members_to_read
                ]
                from domain.coverage import is_cell_statistically_valid

                u_arr = np.stack(u_stack, axis=0)
                v_arr = np.stack(v_stack, axis=0)
                finite_mask = np.isfinite(u_arr) & np.isfinite(v_arr)
                finite_counts = np.sum(finite_mask, axis=0)
                valid_cells = is_cell_statistically_valid(finite_counts, expected_members)
                speed_members = np.hypot(u_arr, v_arr)
                with np.errstate(all="ignore"):
                    mean_speed = np.nanmean(speed_members, axis=0) * 3.6
                    values = np.where(valid_cells, mean_speed, np.nan)
            else:
                u_win = reader.read_window(
                    "wind_u_10m",
                    member=None,
                    lead_time_hours=lead,
                    lat_min=lat_min_idx,
                    lat_max=lat_max_idx,
                    lon_min=lon_min_idx,
                    lon_max=lon_max_idx,
                    generation=generation,
                )
                v_win = reader.read_window(
                    "wind_v_10m",
                    member=None,
                    lead_time_hours=lead,
                    lat_min=lat_min_idx,
                    lat_max=lat_max_idx,
                    lon_min=lon_min_idx,
                    lon_max=lon_max_idx,
                    generation=generation,
                )
                values = np.hypot(u_win, v_win) * 3.6
        else:
            if is_ensemble:
                members_to_read = tuple(range(1, expected_members + 1))
                values = reader.read_ensemble_mean_window(
                    variable,
                    members=members_to_read,
                    lead_time_hours=lead,
                    lat_min=lat_min_idx,
                    lat_max=lat_max_idx,
                    lon_min=lon_min_idx,
                    lon_max=lon_max_idx,
                    expected_members=expected_members,
                    generation=generation,
                )
            else:
                values = reader.read_window(
                    variable,
                    member=None,
                    lead_time_hours=lead,
                    lat_min=lat_min_idx,
                    lat_max=lat_max_idx,
                    lon_min=lon_min_idx,
                    lon_max=lon_max_idx,
                    generation=generation,
                )

        lat_sliced = np.asarray(lat_axis_full[lat_min_idx : lat_max_idx + 1], dtype=float)
        lon_sliced = np.asarray(lon_axis_full[lon_min_idx : lon_max_idx + 1], dtype=float)

        if len(lat_sliced) > 1 and lat_sliced[-1] < lat_sliced[0]:
            values = values[::-1, :]
            lat_sliced = lat_sliced[::-1]
        if len(lon_sliced) > 1 and lon_sliced[-1] < lon_sliced[0]:
            values = values[:, ::-1]
            lon_sliced = lon_sliced[::-1]

        return (values, lat_sliced, lon_sliced)

    if variable in ("wind_10m", "wind_speed_10m"):
        if "wind_u_10m" not in dataset.data_vars or "wind_v_10m" not in dataset.data_vars:
            raise ValueError("Variables 'wind_u_10m' and 'wind_v_10m' must be in the dataset.")
        field_u = dataset["wind_u_10m"]
        field_v = dataset["wind_v_10m"]
        if "lead_time_hours" in field_u.dims:
            field_u = field_u.sel(lead_time_hours=lead)
        if "lead_time_hours" in field_v.dims:
            field_v = field_v.sel(lead_time_hours=lead)

        lat_axis_full = field_u.latitude.values
        lon_axis_full = field_u.longitude.values
        lat_min = float(pixel_lats.min())
        lat_max = float(pixel_lats.max())
        lon_native_min = float(lon_native.min())
        lon_native_max = float(lon_native.max())

        lat_step = float(lat_axis_full[1] - lat_axis_full[0]) if len(lat_axis_full) > 1 else 1.0
        lon_step = float(lon_axis_full[1] - lon_axis_full[0]) if len(lon_axis_full) > 1 else 1.0
        lo_lat = float(max(min(lat_axis_full[0], lat_axis_full[-1]), lat_min - abs(lat_step)))
        hi_lat = float(min(max(lat_axis_full[-1], lat_axis_full[0]), lat_max + abs(lat_step)))
        lo_lon = float(max(min(lon_axis_full[0], lon_axis_full[-1]), lon_native_min - abs(lon_step)))
        hi_lon = float(min(max(lon_axis_full[-1], lon_axis_full[0]), lon_native_max + abs(lon_step)))
        if lo_lat > hi_lat or lo_lon > hi_lon:
            return (
                np.full((1, 1), np.nan),
                np.asarray([min(lat_axis_full[0], lat_axis_full[-1])]),
                np.asarray([min(lon_axis_full[0], lon_axis_full[-1])]),
            )
        sliced_u = field_u.sel(
            latitude=slice(hi_lat, lo_lat) if grid.lat_reversed else slice(lo_lat, hi_lat),
            longitude=slice(hi_lon, lo_lon) if grid.lon_reversed else slice(lo_lon, hi_lon),
        )
        sliced_v = field_v.sel(
            latitude=slice(hi_lat, lo_lat) if grid.lat_reversed else slice(lo_lat, hi_lat),
            longitude=slice(hi_lon, lo_lon) if grid.lon_reversed else slice(lo_lon, hi_lon),
        )
        if "member" in sliced_u.dims:
            # Mean scalar wind speed across ensemble members: mean(hypot(u_i, v_i))
            from domain.coverage import is_cell_statistically_valid

            u_vals = np.asarray(sliced_u.values, dtype=float)
            v_vals = np.asarray(sliced_v.values, dtype=float)
            finite_mask = np.isfinite(u_vals) & np.isfinite(v_vals)
            finite_counts = np.sum(finite_mask, axis=0)
            valid_cells = is_cell_statistically_valid(finite_counts, expected_members)
            speed_members = np.hypot(u_vals, v_vals)
            with np.errstate(all="ignore"):
                mean_speed = np.nanmean(speed_members, axis=0) * 3.6
                values = np.where(valid_cells, mean_speed, np.nan)
        else:
            values = np.hypot(sliced_u.values, sliced_v.values) * 3.6

        lat_sliced = np.asarray(sliced_u.latitude.values, dtype=float)
        lon_sliced = np.asarray(sliced_u.longitude.values, dtype=float)
        if values.shape[0] == 0 or values.shape[1] == 0:
            return (
                np.full((1, 1), np.nan),
                np.asarray([min(lat_axis_full[0], lat_axis_full[-1])]),
                np.asarray([min(lon_axis_full[0], lon_axis_full[-1])]),
            )
        if len(lat_sliced) > 1 and lat_sliced[-1] < lat_sliced[0]:
            values = values[::-1, :]
            lat_sliced = lat_sliced[::-1]
        if len(lon_sliced) > 1 and lon_sliced[-1] < lon_sliced[0]:
            values = values[:, ::-1]
            lon_sliced = lon_sliced[::-1]
        return (values, lat_sliced, lon_sliced)

    if variable not in dataset.data_vars:
        raise ValueError(f"Variable '{variable}' is not in the dataset.")

    field = dataset[variable]
    if "lead_time_hours" in field.dims:
        field = field.sel(lead_time_hours=lead)

    lat_axis_full = field.latitude.values
    lon_axis_full = field.longitude.values
    lat_min = float(pixel_lats.min())
    lat_max = float(pixel_lats.max())
    lon_native_min = float(lon_native.min())
    lon_native_max = float(lon_native.max())

    lat_step = float(lat_axis_full[1] - lat_axis_full[0]) if len(lat_axis_full) > 1 else 1.0
    lon_step = float(lon_axis_full[1] - lon_axis_full[0]) if len(lon_axis_full) > 1 else 1.0
    lo_lat = float(max(min(lat_axis_full[0], lat_axis_full[-1]), lat_min - abs(lat_step)))
    hi_lat = float(min(max(lat_axis_full[-1], lat_axis_full[0]), lat_max + abs(lat_step)))
    lo_lon = float(max(min(lon_axis_full[0], lon_axis_full[-1]), lon_native_min - abs(lon_step)))
    hi_lon = float(min(max(lon_axis_full[-1], lon_axis_full[0]), lon_native_max + abs(lon_step)))
    if lo_lat > hi_lat or lo_lon > hi_lon:
        return (
            np.full((1, 1), np.nan),
            np.asarray([min(lat_axis_full[0], lat_axis_full[-1])]),
            np.asarray([min(lon_axis_full[0], lon_axis_full[-1])]),
        )
    sliced = field.sel(
        latitude=slice(hi_lat, lo_lat) if grid.lat_reversed else slice(lo_lat, hi_lat),
        longitude=slice(hi_lon, lo_lon) if grid.lon_reversed else slice(lo_lon, hi_lon),
    )
    if "member" in sliced.dims:
        from domain.coverage import is_cell_statistically_valid

        raw_members = np.asarray(sliced.values, dtype=float)
        if variable == "cloud_ceiling":
            valid_mask = np.isfinite(raw_members) & (raw_members >= 0.0)
            valid_counts = np.sum(valid_mask, axis=0)
            valid_cells = is_cell_statistically_valid(valid_counts, expected_members)
            unlimited_mask = raw_members >= 19990.0
            finite_members = np.where(unlimited_mask, np.nan, raw_members)
            with np.errstate(all="ignore"):
                mean_vals = np.nanmean(finite_members, axis=0)
                values = np.where(valid_cells, mean_vals, np.nan)
        else:
            finite_mask = np.isfinite(raw_members)
            finite_counts = np.sum(finite_mask, axis=0)
            valid_cells = is_cell_statistically_valid(finite_counts, expected_members)
            with np.errstate(all="ignore"):
                mean_vals = np.nanmean(raw_members, axis=0)
                values = np.where(valid_cells, mean_vals, np.nan)
    else:
        values = np.asarray(sliced.values, dtype=float)

    lat_sliced = np.asarray(sliced.latitude.values, dtype=float)
    lon_sliced = np.asarray(sliced.longitude.values, dtype=float)
    if values.shape[0] == 0 or values.shape[1] == 0:
        return (
            np.full((1, 1), np.nan),
            np.asarray([min(lat_axis_full[0], lat_axis_full[-1])]),
            np.asarray([min(lon_axis_full[0], lon_axis_full[-1])]),
        )
    if len(lat_sliced) > 1 and lat_sliced[-1] < lat_sliced[0]:
        values = values[::-1, :]
        lat_sliced = lat_sliced[::-1]
    if len(lon_sliced) > 1 and lon_sliced[-1] < lon_sliced[0]:
        values = values[:, ::-1]
        lon_sliced = lon_sliced[::-1]
    return (values, lat_sliced, lon_sliced)


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

    Checks the catalog only: the model/variable exist, an eligible run exists
    for the model (optionally pinned to ``initial_time``), and a matching
    ``forecast_products`` row exists for the run/variable/level/lead. Raises
    404/422 exactly as the tile renderer would, without opening the store —
    the metadata endpoint uses this so a template is only returned for
    combinations that actually exist.

    Raises:
        HTTPException: 422 for an unsupported level, 404 when the selection is
            not available.
    """
    _resolve_run_store_path(
        db,
        model=model,
        variable=variable,
        level=level,
        lead_time_hours=lead_time_hours,
        initial_time=initial_time,
        excluded=set(),
    )


def _resolve_ready_run(
    db: Session,
    *,
    model: str,
    initial_time: str | None,
) -> ModelRun:
    """Resolve the newest eligible run for a model, optionally pinned by cycle.

    Raises:
        HTTPException: 404 when no eligible run matches.
    """
    from fastapi import HTTPException

    if initial_time is not None:
        require_cycle_visible(db, initial_time)

    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        cycle = _parse_cycle_time(initial_time)
        stmt = stmt.where(ModelRun.cycle_time == cycle)
    stmt = filter_visible_runs(stmt).order_by(ModelRun.cycle_time.desc())
    runs = list(db.execute(stmt).scalars().all())
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No eligible forecast run with data was found for model '{model}'"
                + (f" and initial time '{initial_time}'." if initial_time else ".")
            ),
        )
    return runs[0]


def _resolve_run_store_path(
    db: Session,
    *,
    model: str,
    variable: str,
    level: str,
    lead_time_hours: int,
    initial_time: str | None,
    excluded: set[str],
) -> str:
    """Resolve the eligible run's Zarr store path for a tile request (catalog only).

    This is the **DB metadata phase** of the tile render. It validates the
    model/variable/level, selects the newest eligible run with a store (optionally
    pinned to ``initial_time``), checks ensemble member coverage (>= 85%), and
    confirms a matching ``forecast_products`` row exists. Only cheap catalog
    queries run here; the caller releases the session/connection before any
    Zarr read.

    ``excluded`` holds store paths already tried and found unreadable, so the
    broken-newest-store recovery path falls through to the next-newest eligible
    run.

    Raises:
        HTTPException: 404 when the model/variable/level/lead combination is
            not available, 422 for an unsupported level.
    """
    from fastapi import HTTPException
    from domain.coverage import get_expected_members, is_lead_servable

    if level != "surface":
        raise HTTPException(status_code=422, detail=f"Unsupported level '{level}'.")
    _require_model_variable(db, model, variable)

    if initial_time is not None:
        require_cycle_visible(db, initial_time)

    expected_members = get_expected_members(model, default_if_unknown=1)

    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        cycle = _parse_cycle_time(initial_time)
        stmt = stmt.where(ModelRun.cycle_time == cycle)
    stmt = filter_visible_runs(stmt).order_by(ModelRun.cycle_time.desc())
    runs = list(db.execute(stmt).scalars().all())

    candidates = [
        run for run in runs
        if run.zarr_store_path is not None
        and str(run.zarr_store_path) not in excluded
    ]
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No readable forecast run with data was found for model '{model}'."
                if excluded
                else f"No eligible forecast run with data was found for model '{model}'"
                + (f" and initial time '{initial_time}'." if initial_time else ".")
            ),
        )

    for run in candidates:
        assert run.zarr_store_path is not None
        # Check ensemble member coverage if model is ensemble
        if expected_members > 1:
            member_rows = db.execute(
                select(EnsembleMemberProduct.member_index).where(
                    EnsembleMemberProduct.run_id == run.id,
                    EnsembleMemberProduct.lead_time_hours == lead_time_hours,
                )
            ).scalars().all()
            avail_members = tuple(member_rows)
            # If no pair rows (legacy store / test fixture), allow ready runs
            if not avail_members and run.status == "ready":
                avail_members = tuple(range(1, expected_members + 1))
            if not is_lead_servable(len(avail_members), expected_members):
                if initial_time is not None:
                    raise HTTPException(
                        status_code=404,
                        detail=(
                            f"Lead {lead_time_hours}h for model '{model}' at cycle '{initial_time}' "
                            f"is not servable ({len(avail_members)}/{expected_members} members < 85%)."
                        ),
                    )
                continue

        try:
            _require_product(db, run, variable, level, lead_time_hours)
            return str(run.zarr_store_path)
        except HTTPException:
            if initial_time is not None:
                raise
            continue

    raise HTTPException(
        status_code=404,
        detail=(
            f"No forecast product is available for model '{model}', "
            f"variable '{variable}', level '{level}', lead '{lead_time_hours}h'."
        ),
    )


def _require_model_variable(db: Session, model: str, variable: str) -> None:
    """Raise 404 if the model or variable is not in the catalog."""
    from fastapi import HTTPException

    model_found = db.execute(
        select(Model.model_id).where(Model.model_id == model)
    ).scalar_one_or_none()
    if model_found is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' was not found.")

    check_vars = ["wind_u_10m", "wind_v_10m"] if variable == "wind_10m" else [variable]
    for var in check_vars:
        variable_found = db.execute(
            select(ForecastVariable.variable_code).where(
                ForecastVariable.variable_code == var
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

    check_vars = ["wind_u_10m", "wind_v_10m"] if variable == "wind_10m" else [variable]
    for var in check_vars:
        product = db.execute(
            select(ForecastProduct.id).where(
                ForecastProduct.run_id == run.id,
                ForecastProduct.variable_id == var,
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
