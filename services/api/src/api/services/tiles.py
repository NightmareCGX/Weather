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
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.core.png import encode_rgba_png
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
    _validate_tile(zoom, x, y)
    # Resolve the committed-manifest generation for cache identity before the
    # lookup (the tile cache key includes the generation so a same-set data
    # replacement makes old tiles unreachable). This is a cheap catalog query.
    from api.services.point_forecast import resolve_latest_run_serving_generation

    serving_generation = resolve_latest_run_serving_generation(
        db, model, initial_time
    )
    cache_key = _tile_cache_key(
        model, variable, level, zoom, x, y, lead_time_hours, initial_time, serving_generation
    )
    cached = _tile_cache_get(cache_key)
    if cached is not None:
        return cached

    from api.core import reader_gate
    from api.core.database import SessionLocal

    session = db
    excluded: set[str] = set()
    while True:
        try:
            store_path = _resolve_run_store_path(
                session,
                model=model,
                variable=variable,
                level=level,
                lead_time_hours=lead_time_hours,
                initial_time=initial_time,
                excluded=excluded,
            )
        except BaseException:
            # No usable candidate (404) or a query error: the session's
            # connection must still be returned to the pool before propagating.
            session.close()
            raise
        # Release this session's DB connection BEFORE the expensive Zarr read.
        # The reader gate revalidates the run is READY on its own dedicated
        # lock-pool connection, so no ORM session is needed during the read. On
        # a broken store the catalog is re-queried on a fresh short-lived
        # session (rare recovery path) to fall through to the next candidate.
        session.close()  # return the connection to the QueuePool
        try:
            dataset = reader_gate.gated_read_dataset(store_path)
        except Exception:  # noqa: BLE001 - unreadable/no-longer-ready store
            excluded.add(store_path)
            session = SessionLocal()
            continue
        break
    return _render_dataset_to_png(
        dataset,
        variable=variable,
        zoom=zoom,
        x=x,
        y=y,
        lead_time_hours=lead_time_hours,
        cache_key=cache_key,
    )


def _render_dataset_to_png(
    dataset: xr.Dataset,
    *,
    variable: str,
    zoom: int,
    x: int,
    y: int,
    lead_time_hours: int,
    cache_key: tuple[object, ...],
) -> bytes:
    """Rasterize an already-materialized dataset into a tile PNG (no DB)."""
    grid = _derive_grid(dataset)
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

    # Align pixel longitudes into the dataset's native convention up front.
    lon_native = _align_longitudes(grid, pixel_lons)

    field, lat_axis, lon_axis = _slice_field(
        dataset, variable, lead_time_hours, grid, pixel_lats, lon_native
    )

    # Nearest grid index per pixel, into the *sliced* ascending axes.
    rows = _nearest_indices(lat_axis, pixel_lats)
    cols = _nearest_indices(lon_axis, lon_native)

    valid = _inside_grid(grid, pixel_lats, lon_native)
    values = field[rows, cols]

    # Fully vectorized color mapping: clamp, interpolate across the stop ramp
    # per RGB channel with ``np.interp``, and build the RGBA scanlines with
    # NumPy (no 65,536-iteration Python loop).
    finite = np.isfinite(values) & valid
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    rgba[..., 3] = 255
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
    # Ensemble (GEFS) stores carry a leading ``member`` dimension. A map tile
    # is a single deterministic surface image, so the member axis is reduced by
    # the mean (the platform's documented ensemble aggregate — API.md 5.1
    # derives statistics from all members; member 0 control is not present in
    # the real stores, which hold perturbation members 1..30). Reject only if
    # the field is still not a plain 2-D surface after the reduction.
    if "member" in field.dims:
        field = field.mean(dim="member", keep_attrs=True)
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
    """Resolve the ready run's Zarr store path for a tile request (catalog only).

    This is the **DB metadata phase** of the tile render. It validates the
    model/variable/level, selects the newest ready run with a store (optionally
    pinned to ``initial_time``), and confirms a matching ``forecast_products``
    row exists. Only cheap catalog queries run here; the caller releases the
    session/connection before any Zarr read.

    ``excluded`` holds store paths already tried and found unreadable, so the
    broken-newest-store recovery path falls through to the next-newest ready
    run (mirroring the legacy ``_resolve_run_and_field`` fallback).

    Raises:
        HTTPException: 404 when the model/variable/level/lead combination is
            not available, 422 for an unsupported level.
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
                else f"No ready forecast run with data was found for model '{model}'"
                + (f" and initial time '{initial_time}'." if initial_time else ".")
            ),
        )

    for run in candidates:
        assert run.zarr_store_path is not None
        # ``_require_product`` raises 404 when the newest matching run lacks
        # this exact variable/level/lead product (no fallback to older runs),
        # preserving the legacy selection semantics.
        _require_product(db, run, variable, level, lead_time_hours)
        return str(run.zarr_store_path)
    raise AssertionError("unreachable: every candidate raised in _require_product")


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
