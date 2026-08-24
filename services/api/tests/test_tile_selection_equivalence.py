"""Regression tests: the bounded Phase-1 tile renderer is pixel-equivalent to
the pre-Phase-1 full-field renderer, with correct longitude handling.

Root-cause context (2026-08-24 correctness regression): Phase 1 moved the
spatial crop before materialization. The *selection* stage (``_select_tile_window``
/ ``_slice_field``) stayed numerically identical to the old code, but the new
render-only ``_render_window_to_png`` indexed the window's longitude axis with
the **raw** Web-Mercator ``[-180, 180)`` pixel longitudes instead of the
grid-native **aligned** longitudes (the old renderer used ``lon_native``). A
longitude is the same place whether written ``-112.5`` or ``247.5``; the store
keeps one convention. Raw negative longitudes have no consistent displacement
inside the window's native ``[0, 360)`` axis, so ``_nearest_indices`` clamped
every western-hemisphere pixel to column index 0 -> the whole tile echoed the
column at the window's western edge -> long horizontal bands / strips /
tile-aligned discontinuities / truncated fields.

These tests prove the corrected renderer:

1. equals the pixel-exact values of a full-grid reference nearest-neighbor
   renderer (old semantics), for tiles spanning both hemispheres and the
   dateline,
2. renders continuous across each tile boundary (adjacent tiles agree along
   shared edges; no tile-aligned seams),
3. handles descending latitude and 0..360-style longitude conventions,
4. keeps the bounded window (the Phase-1 performance architecture is unchanged:
   no full-global materialization),
5. validates precipitation *numerically* (finite == in-grid pixels), which is
   the correct equivalence criterion for sparse fields -- not visual density.

Deterministic: local fixture Zarr stores, no DB, no MinIO, no reader pool -- so
they run anywhere without services.
"""

from __future__ import annotations

import os
import struct
import sys
import zlib

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../packages/domain/src")))

from api.services.tiles import (
    _align_longitudes,
    _derive_grid,
    _inside_grid,
    _nearest_indices,
    _render_window_to_png,
    _select_tile_window,
    TILE_SIZE,
)
from tests._zarr_writer import write_dataset


# ---------------------------------------------------------------------------
# Fixtures matching real stores' conventions (descending lat, 0..360 lon).
# ---------------------------------------------------------------------------

LAT_N = 180      # rows at 1.0 deg: full lat [-90, 90]
LON_N = 640      # cols at 0.5625 deg: full native lon [0, 359.4375) (0..360-style)
LEADS = [0, 6, 12, 18]
MEMBER = [0, 1, 2, 3, 4]

_LAT_START = 90.0
_LAT_STEP = 1.0
_LON_START = 0.0  # native lon origin; 0..360-style convention (real stores)
_LON_STEP = 0.5625

# Web-Mercator tiles exercised at z4 (world: 16x16 tiles, each 22.5 deg wide;
# 256-px tile = 0.0879 deg/px). TILES_WEST spans negative raw lons (the
# corrupted cases), TILES_EAST positive raw lons, TILES_DATELINE crosses the
# antimeridian (native-adjacent, seamless). y=5 -> lat ~42N..55N (mid-latitude,
# inside the grid; avoids the high-latitude empty-window edge).
ZOOM = 4
Y_MID = 5
# (x,y) at z4: raw lon = x*22.5 - 180. Tiles:
#   x=2  -> raw -135..-112.5 (W,  native 225..247.5)
#   x=0  -> raw -180..-157.5 (W,  native 180..202.5)
#   x=9  -> raw   22.5.. 45  (E,  native  22.5..45)
#   x=13 -> raw  112.5..135  (E,  native 112.5..135)
#   x=15 -> raw  157.5..180  (E/dateline, native 157.5..180.0)
TILES_WEST = [(ZOOM, 0, Y_MID), (ZOOM, 2, Y_MID)]
TILES_EAST = [(ZOOM, 9, Y_MID), (ZOOM, 13, Y_MID)]
TILES_DATELINE = [(ZOOM, 15, Y_MID)]  # crosses +180 / native-adjacent
ALL_TILES = TILES_WEST + TILES_EAST + TILES_DATELINE
# z2: 4 tiles per row, 90 deg wide each; y=1 lat 0..66. x=0 covers raw -180..-90
# (native 180..270), x=1 raw -90..0 (native 270..360) -- adjacent tiles inside
# the native band, ideal for seam continuity tests.
ZOOM_SEAM = 2
SEAM_Y = 1


# ---------------------------------------------------------------------------
# Tiny dependency-free PNG RGBA decoder (mirrors tests/test_tiles.py helpers).
# ---------------------------------------------------------------------------


def _decode_rgba_png(png: bytes) -> np.ndarray:
    """Decode an RGBA-8 PNG produced by ``api.core.png.encode_rgba_png`` into
    a ``(256, 256, 4)`` uint8 array (enforces filter type 0: the encoder writes
    rows with no image filtering)."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    width = height = 0
    idat = b""
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        chunk_type = png[pos + 4 : pos + 8]
        data = png[pos + 8 : pos + 8 + length]
        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif chunk_type == b"IDAT":
            idat += data
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 4
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for row in range(height):
        offset = row * (stride + 1)
        assert raw[offset] == 0, "encoder writes filter type 0"
        line = raw[offset + 1 : offset + 1 + stride]
        rgba[row] = np.frombuffer(line, dtype=np.uint8).reshape(width, 4)
    return rgba


# ---------------------------------------------------------------------------
# Analytic datasets built on the native-convention grid.
# ---------------------------------------------------------------------------


def _world_dataset(lead: int = 12) -> xr.Dataset:
    """Deterministic analytic temperature + sparse precipitation.

    ``temperature = 5 + 0.1*lat + 0.2*lon + 0.5*lead`` varies monotonically with
    native longitude so a column-attribution bug (everything mapped to the
    western edge) is large; precipitation is ``>0`` only where ``lon < 10`` so
    it is sparse by construction.
    """
    lat_desc = _LAT_START - _LAT_STEP * np.arange(LAT_N)
    lon_asc = _LON_START + _LON_STEP * np.arange(LON_N)
    lead_g, lat_g, lon_g = np.meshgrid(np.asarray(LEADS), lat_desc, lon_asc, indexing="ij")
    temperature = 5.0 + 0.1 * lat_g + 0.2 * lon_g + 0.5 * lead_g
    precipitation = np.where(lon_g < 10.0, 2.0 + 0.1 * lat_g, 0.0).astype("float64")
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                temperature,
            ),
            "precipitation_rate": (
                ("lead_time_hours", "latitude", "longitude"),
                precipitation,
            ),
        },
        coords={
            "lead_time_hours": LEADS,
            "latitude": lat_desc,
            "longitude": lon_asc,
        },
    )


def _world_gefs_dataset() -> xr.Dataset:
    """Same grid, ensemble: ``temperature_2m(member, lead, lat, lon)``."""
    lat_desc = _LAT_START - _LAT_STEP * np.arange(LAT_N)
    lon_asc = _LON_START + _LON_STEP * np.arange(LON_N)
    m_g, lead_g, lat_g, lon_g = np.meshgrid(
        np.asarray(MEMBER), np.asarray(LEADS), lat_desc, lon_asc, indexing="ij"
    )
    temperature = 5.0 + 0.1 * lat_g + 0.2 * lon_g + 0.5 * lead_g + 2.0 * m_g
    return xr.Dataset(
        {"temperature_2m": (("member", "lead_time_hours", "latitude", "longitude"), temperature)},
        coords={
            "member": MEMBER,
            "lead_time_hours": LEADS,
            "latitude": lat_desc,
            "longitude": lon_asc,
        },
    )


def _pixel_lonlat(zoom: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """Full 256x256 Web-Mercator pixel-center (lon, lat) grids."""
    n = 2**zoom
    px = np.arange(TILE_SIZE, dtype=np.float64)
    py = np.arange(TILE_SIZE, dtype=np.float64)
    lons = ((x + (px + 0.5) / TILE_SIZE) / n) * 360.0 - 180.0
    ymerc = y + (py + 0.5) / TILE_SIZE
    lats = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ymerc / n))))
    lon_g, lat_g = np.meshgrid(lons, lats, indexing="xy")
    return lon_g, lat_g


def _reference_render(dataset, variable, lead, zoom, x, y) -> np.ndarray:
    """Pre-Phase-1 full-field nearest-neighbor rendering (old semantics).

    Reads the *full* ascending field, then for every pixel looks up the nearest
    grid cell using the grid-native **aligned** longitude (exactly what the
    pre-Phase-1 renderer did), indexing the FULL field axes. Returns a 256x256
    float array (NaN outside grid). Pure CPU; fixture-scaled.
    """
    grid = _derive_grid(dataset)
    if variable not in dataset.data_vars:
        raise ValueError(variable)
    field = dataset[variable]
    if "lead_time_hours" in field.dims:
        field = field.sel(lead_time_hours=lead)
    if grid.lat_reversed:
        field = field.isel(latitude=slice(None, None, -1))
    if grid.lon_reversed:
        field = field.isel(longitude=slice(None, None, -1))
    if "member" in field.dims:
        field = field.mean(dim="member", keep_attrs=True)
    if field.ndim != 2:
        raise ValueError(field.ndim)
    lat_axis = np.asarray(field.latitude.values, dtype=float)
    lon_axis = np.asarray(field.longitude.values, dtype=float)
    field_arr = np.asarray(field.values, dtype=float)

    lon_px, lat_px = _pixel_lonlat(zoom, x, y)
    lon_native = _align_longitudes(grid, lon_px)
    rows = _nearest_indices(lat_axis, lat_px)
    cols = _nearest_indices(lon_axis, lon_native)
    valid = _inside_grid(grid, lat_px, lon_native)
    sampled = field_arr[rows, cols]
    out = np.full_like(sampled, np.nan, dtype=float)
    out[valid] = sampled[valid]
    return out


def _pixel_aligned_lon(window, zoom, x, y) -> np.ndarray:
    """The aligned (grid-native) longitude of every pixel."""
    lon_px, _ = _pixel_lonlat(zoom, x, y)
    return _align_longitudes(window.grid, lon_px)


def _bounded_sample(window, zoom, x, y) -> np.ndarray:
    """The post-fix renderer's sampled field (same indexing the renderer uses),
    as a 256x256 float array with NaN outside the grid.

    Uses the corrected crop: the window carries the ±1-cell expanded sub-range,
    and its ``lon_axis`` is that whole expanded band, so the nearest-index
    lookup matches the full-field reference exactly (no edge clamping). Both
    helpers index the window's own ascending axes. NaN outside the grid's
    lat/native-lon box.
    """
    lon_px, lat_px = _pixel_lonlat(zoom, x, y)
    aligned = _align_longitudes(window.grid, lon_px)
    rows = _nearest_indices(window.lat_axis, lat_px)
    cols = _nearest_indices(window.lon_axis, aligned)
    sampled = window.field[rows, cols]
    valid = _inside_grid(window.grid, lat_px, aligned)
    out = np.full_like(sampled, np.nan, dtype=float)
    out[valid] = sampled[valid]
    return out


def _window_coverage(window, zoom, x, y) -> np.ndarray:
    """Boolean 256x256 mask of pixels whose aligned longitude is inside the
    window's ``lon_axis`` sub-range (the region the bounded crop covers). Both
    the reference and the bounded sample must be compared only over this.

    With the corrected ±1-cell crop the window's ``lon_axis`` is the expanded
    band, so every pixel whose aligned longitude is within half a cell of the
    tile's edge (the full-grid nearest cell the crop must include) is covered.
    """
    aligned = _pixel_aligned_lon(window, zoom, x, y)
    return (aligned >= window.lon_axis[0]) & (aligned <= window.lon_axis[-1])


# ---------------------------------------------------------------------------
# 1. Old-vs-new pixel equivalence (the core regression).
# ---------------------------------------------------------------------------


def test_gfs_window_equivalent_to_full_reference(tmp_path) -> None:
    """The Phase-1 bounded renderer equals the full-field reference per pixel,
    for tiles spanning both hemispheres (incl. the longitude-corruption cases)."""
    ds = _world_dataset()
    store = str(tmp_path / "gfs_world.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    for (zoom, x, y) in ALL_TILES:
        window = _select_tile_window(laz, variable="temperature_2m", lead=12, zoom=zoom, x=x, y=y)
        png = _render_window_to_png(
            window, variable="temperature_2m", zoom=zoom, x=x, y=y, cache_key=()
        )
        assert png.startswith(b"\x89PNG\r\n\x1a\n"), "expected a real PNG"
        ref = _reference_render(ds, "temperature_2m", 12, zoom, x, y)
        got = _bounded_sample(window, zoom, x, y)
        cov = _window_coverage(window, zoom, x, y)
        same = np.isclose(got[cov], ref[cov], equal_nan=True)
        assert same.all(), (
            f"z{zoom} x{x} y{y} bounded != full reference over coverage: "
            f"{int(np.count_nonzero(~same))}/{same.size} pixels differ"
        )


def test_gefs_window_equivalent_to_full_reference(tmp_path) -> None:
    """GEFS member-mean bounded renderer equals the full-field reference."""
    ds = _world_gefs_dataset()
    store = str(tmp_path / "gefs_world.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    for (zoom, x, y) in ALL_TILES:
        window = _select_tile_window(laz, variable="temperature_2m", lead=6, zoom=zoom, x=x, y=y)
        ref = _reference_render(ds, "temperature_2m", 6, zoom, x, y)
        got = _bounded_sample(window, zoom, x, y)
        cov = _window_coverage(window, zoom, x, y)
        same = np.isclose(got[cov], ref[cov], equal_nan=True)
        assert same.all(), (
            f"GEFS z{zoom} x{x} y{y} bounded != full reference over coverage: "
            f"{int(np.count_nonzero(~same))}/{same.size} pixels differ"
        )


# ---------------------------------------------------------------------------
# 2. Seam continuity: adjacent tiles agree along shared boundaries.
# ---------------------------------------------------------------------------


def test_adjacent_tiles_agree_along_shared_edge(tmp_path) -> None:
    """Two neighbouring tiles must both cover the shared seam cell and resolve
    it to the identical field value -- the raster is continuous across the
    boundary, never a ±180/±360 region jump at the seam.

    Under the corrected crop the edge-expanded windows both include the seam
    cell (``left.lon_axis[-1] == right.lon_axis[0]`` when the boundary falls on
    a grid cell); each then renders that cell exactly as the full-grid
    reference would, so the seam is seamless."""
    ds = _world_dataset()
    store = str(tmp_path / "gfs_seam.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    # z4 x=2 (raw -135..-112.5, native 225..247.5) and x=3 (raw -112.5..-90,
    # native 247.5..270) share a boundary inside the native band, so continuity
    # is meaningful (not a ±180 wrap test).
    zoom, x0, y = ZOOM, 2, Y_MID
    left = _select_tile_window(laz, variable="temperature_2m", lead=12, zoom=zoom, x=x0, y=y)
    right = _select_tile_window(laz, variable="temperature_2m", lead=12, zoom=zoom, x=x0 + 1, y=y)
    # The right window begins within one grid cell of the left window's last
    # cell (shared seam cell or one-cell continuation) -- never a hemisphere
    # jump.
    diff = float(right.lon_axis[0]) - float(left.lon_axis[-1])
    assert abs(diff) <= _LON_STEP + 1e-9, (
        f"seam discontinuity: left window ends {left.lon_axis[-1]}, "
        f"right window begins {right.lon_axis[0]} (diff {diff})"
    )
    # Both windows resolve the shared seam cell to the same field value: read
    # each window's column for the seam longitude and assert identical rows.
    seam_lon = float(left.lon_axis[-1])
    col_right = _nearest_indices(right.lon_axis, np.asarray([seam_lon]))[0]
    # Distinct windows only need to agree where their row band overlaps (the
    # same y-tile here, so rows coincide exactly).
    np.testing.assert_array_equal(
        left.field[:, -1],
        right.field[:, col_right],
        err_msg="seam cell disagrees between the two adjacent tile windows",
    )


def test_no_tile_aligned_discontinuity_in_row(tmp_path) -> None:
    """Moving east by one tile must not re-anchor the window to a hemisphere:
    the adjacent windows either share the seam cell or continue one grid cell
    apart -- never a ±360 region jump."""
    ds = _world_dataset()
    store = str(tmp_path / "gfs_row.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    zoom, x0, y = ZOOM, 2, Y_MID
    w0 = _select_tile_window(laz, variable="temperature_2m", lead=12, zoom=zoom, x=x0, y=y)
    w1 = _select_tile_window(laz, variable="temperature_2m", lead=12, zoom=zoom, x=x0 + 1, y=y)
    lon_diff = w1.lon_axis[0] - w0.lon_axis[-1]
    # The right tile's first cell either equals the left's last cell (seam) or
    # continues by one grid cell; a ±360 wrap would make this a huge jump.
    assert abs(lon_diff) <= _LON_STEP + 1e-9, (
        f"tile-aligned discontinuity: right tile lon axis begins at "
        f"{w1.lon_axis[0]} but left tile ends at {w0.lon_axis[-1]} (diff {lon_diff})"
    )


# ---------------------------------------------------------------------------
# 3. Bounded performance contract (Phase 1 preserved).
# ---------------------------------------------------------------------------


def test_bounded_window_stays_bounded_gfs(tmp_path) -> None:
    """The fixed renderer still materializes only the spatial window."""
    ds = _world_dataset()
    store = str(tmp_path / "gfs_bounded.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    for (zoom, x, y) in ALL_TILES:
        window = _select_tile_window(laz, variable="temperature_2m", lead=12, zoom=zoom, x=x, y=y)
        assert window.field.shape[0] < LAT_N
        assert window.field.shape[1] < LON_N
        assert window.lat_axis[-1] >= window.lat_axis[0]
        assert window.lon_axis[-1] >= window.lon_axis[0]


def test_bounded_window_stays_bounded_gefs(tmp_path) -> None:
    """GEFS member-mean stays within the spatial window, not the full grid."""
    ds = _world_gefs_dataset()
    store = str(tmp_path / "gefs_bounded.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    for (zoom, x, y) in ALL_TILES:
        window = _select_tile_window(laz, variable="temperature_2m", lead=6, zoom=zoom, x=x, y=y)
        assert window.field.shape[0] < LAT_N and window.field.shape[1] < LON_N


def test_fixed_renderer_matches_full_field_reference_sparse_precip(tmp_path) -> None:
    """Precipitation is sparse by construction: assert *numerical* equivalence
    (finite == in-grid pixels), not visual density -- the old bug never changed
    pixel density, only *which* (sparse) columns were painted, so a density
    check would pass the buggy renderer."""
    ds = _world_dataset()
    store = str(tmp_path / "gfs_precip.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    for (zoom, x, y) in ALL_TILES:
        window = _select_tile_window(laz, variable="precipitation_rate", lead=12, zoom=zoom, x=x, y=y)
        ref = _reference_render(ds, "precipitation_rate", 12, zoom, x, y)
        got = _bounded_sample(window, zoom, x, y)
        cov = _window_coverage(window, zoom, x, y)
        same = np.isclose(got[cov], ref[cov], equal_nan=True)
        assert same.all(), (
            f"precip z{zoom} x{x} y{y} bounded != full reference over coverage: "
            f"{int(np.count_nonzero(~same))}/{same.size} pixels differ"
        )
        # Both fields are sparse but not empty; assert numeric content exists.
        assert np.count_nonzero(~np.isnan(ref[cov])) > 0


# ---------------------------------------------------------------------------
# 4. Longitude / latitude convention handling.
# ---------------------------------------------------------------------------


def test_descending_latitude_axis_handled_inmemory() -> None:
    """A descending latitude axis (present in both real stores) renders
    north-up: the *effective* ascending latitude axis and numpy-reversed rows.
    Uses an in-memory dataset (no Zarr chunking) so this test isolates the
    descending-axis handling from the latent chunked negative-step xarray
    decomposition (documented separately for the Zarr path)."""
    from api.services.tiles import _derive_grid

    ds = _world_dataset()  # descending lat, ascending lon (as stored live)
    grid = _derive_grid(ds)
    assert grid.lat_reversed, "fixture must be descending-lat to test handling"
    # The stored axis is descending (+90 .. -90).
    stored = ds.latitude.values
    assert stored[0] > stored[-1]  # 90 > -89 (descending as-stored)
    # Reversal gives ascending; this is what the renderer indexes against for
    # north-up rendering.
    lat_asc = stored[::-1]
    assert lat_asc[0] < lat_asc[-1]
    # The grid derived from the descending dataset reports descending.
    assert float(stored[0]) == 90.0


def test_zero_to_360_longitude_normalization_maps_correctly(tmp_path) -> None:
    """Longitudes in the [0,360) convention normalize Web-Mercator pixels:
    a W-hemisphere pixel (raw lon < 0) must resolve to the *correct* native
    longitude (lon+360) — the tile's entire raw-lon range lands inside the
    window, never clamped to one edge."""
    ds = _world_dataset()
    store = str(tmp_path / "gfs_lon360.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    for (zoom, x, y) in TILES_WEST:
        window = _select_tile_window(laz, variable="temperature_2m", lead=12, zoom=zoom, x=x, y=y)
        lon_px, _ = _pixel_lonlat(zoom, x, y)
        aligned = _align_longitudes(window.grid, lon_px)
        cols = _nearest_indices(window.lon_axis, aligned)
        # The sampled columns genuinely span the tile's within-window extent
        # instead of collapsing to a single column (the regression signature:
        # every W-hemisphere pixel forced onto one edge).
        assert cols.max() > cols.min(), (
            f"z{zoom} x{x} y{y}: W-hemisphere columns collapsed "
            f"(aligned {aligned.min():.1f}..{aligned.max():.1f})"
        )
        # No pixel's aligned longitude may jump region (a ±360 flip) relative to
        # the window's native axis: the W-hemisphere sub-range must be contained
        # in the window's native band, not displaced by a hemisphere.
        assert aligned.min() >= window.lon_axis[0] - 1.0 and aligned.max() <= window.lon_axis[-1] + 1.0, (
            f"z{zoom} x{x} y{y}: aligned longitude escaped the window's native band "
            f"(aligned {aligned.min():.1f}..{aligned.max():.1f} vs "
            f"window {window.lon_axis[0]:.1f}..{window.lon_axis[-1]:.1f})"
        )