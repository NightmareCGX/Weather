"""Contract and integration tests for the map tile endpoint.

These tests verify that `/v1/maps/{model}/{variable}/{level}/{z}/{x}/{y}.png`
renders genuine forecast data from the fixture Zarr stores as a PNG, and that
invalid selections return proper errors. When PostgreSQL is unreachable they
skip, following the existing convention.
"""

import struct
import zlib

import numpy as np


def _png_dimensions(png: bytes) -> tuple[int, int]:
    """Extract the width/height from a PNG's IHDR chunk."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        chunk_type = png[pos + 4 : pos + 8]
        chunk_data = png[pos + 8 : pos + 8 + length]
        if chunk_type == b"IHDR":
            # IHDR data is 13 bytes: width (4) + height (4) + 5 byte fields.
            width, height = struct.unpack(">II", chunk_data[:8])
            return width, height
        pos += 12 + length
    raise AssertionError("No IHDR chunk found")


def _png_has_opaque_pixels(png: bytes) -> bool:
    """Return whether any pixel in the PNG is opaque (alpha == 255)."""
    _, _, idat = _extract_idat(png)
    raw = zlib.decompress(idat)
    # Filter type 0 only; scanline stride is width*4+1.
    opaque = False
    stride = 256 * 4
    for row in range(256):
        offset = row * (stride + 1) + 1
        line = raw[offset : offset + stride]
        for px in range(0, stride, 4):
            if line[px + 3] == 255:
                opaque = True
                break
        if opaque:
            break
    return opaque


def _extract_idat(png: bytes) -> tuple[int, int, bytes]:
    width = height = 0
    idat = b""
    pos = 8
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        chunk_type = png[pos + 4 : pos + 8]
        data = png[pos + 8 : pos + 8 + length]
        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif chunk_type == b"IDAT":
            idat += data
        pos += 12 + length
    return width, height, idat


def test_tile_renders_temperature_png(client):
    # The fixture gfs Zarr store covers lat 38-38.75 / lon -107..-106.25.
    # Tile (8, 51, 98) spans lon -108.28..-106.88 and lat 37.77..38.88, which
    # overlaps the fixture grid, so the tile has opaque (forecast) pixels.
    resp = client.get(
        "/v1/maps/gfs/temperature_2m/surface/8/51/98.png?lead_time_hours=6"
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/png"
    assert resp.headers["Cache-Control"] == "no-cache"
    width, height = _png_dimensions(resp.content)
    assert (width, height) == (256, 256)
    assert _png_has_opaque_pixels(resp.content)


def test_tile_renders_precipitation_png(client):
    resp = client.get(
        "/v1/maps/gfs/precipitation_rate/surface/8/51/98.png?lead_time_hours=6"
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/png"
    assert _png_has_opaque_pixels(resp.content)


def test_tile_with_initial_time_pins_the_run(client):
    resp = client.get(
        "/v1/maps/gfs/temperature_2m/surface/8/51/98.png"
        "?lead_time_hours=6&initial_time=2026-07-21T00:00:00Z"
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/png"


def test_tile_unknown_model_404(client):
    resp = client.get(
        "/v1/maps/nope/temperature_2m/surface/8/51/98.png?lead_time_hours=6"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_tile_unknown_variable_404(client):
    resp = client.get(
        "/v1/maps/gfs/wind_speed/surface/8/51/98.png?lead_time_hours=6"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_tile_unsupported_level_422(client):
    resp = client.get(
        "/v1/maps/gfs/temperature_2m/500hPa/8/51/98.png?lead_time_hours=6"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_tile_out_of_range_422(client):
    # Zoom 15 exceeds the supported max.
    resp = client.get(
        "/v1/maps/gfs/temperature_2m/surface/15/0/0.png?lead_time_hours=6"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_tile_lead_not_available_404(client):
    # The fixture dataset has leads [0, 6, 12, 18]; 24 is not available.
    resp = client.get(
        "/v1/maps/gfs/temperature_2m/surface/8/51/98.png?lead_time_hours=24"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_tile_renders_ensemble_map_png(client):
    """A supported GEFS tile request must render, not 422.

    The fixture gefs store holds ``temperature_2m(member, lead, lat, lon)``;
    the tile endpoint must reduce the ``member`` dimension (ensemble mean) and
    return a real 256x256 PNG. Regression test for the GEFS map-tile 422: the
    renderer previously rejected any field with ndim > 2 after lead selection.
    """
    resp = client.get(
        "/v1/maps/gefs/temperature_2m/surface/8/51/98.png?lead_time_hours=6"
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/png"
    width, height = _png_dimensions(resp.content)
    assert (width, height) == (256, 256)
    assert _png_has_opaque_pixels(resp.content)


def test_tile_renders_ensemble_precipitation_png(client):
    """The ensemble precipitation field is also renderable as a map tile."""
    resp = client.get(
        "/v1/maps/gefs/precipitation_rate/surface/8/51/98.png?lead_time_hours=6"
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/png"
    assert _png_has_opaque_pixels(resp.content)


def test_ensemble_tile_reduces_member_dimension_to_mean():
    """GEFS map tiles render the ensemble mean, not a single member.

    The documented ensemble contract (API.md 5.1) derives statistics from all
    members; the map tile therefore reduces the ``member`` dimension by the
    mean so the rendered field is the deterministic ensemble-mean surface
    (member 0 is not a valid selection — the real stores carry only
    perturbation members 1..30). This asserts the *semantics* of the member
    reduction, not merely HTTP 200.
    """
    import numpy as np

    from api.services.tiles import _slice_field
    from tests.fixtures import (
        LATITUDES,
        LONGITUDES,
        build_ensemble_dataset,
        ensemble_temperature_at,
    )

    dataset = build_ensemble_dataset()
    # Mean over members of the fixture ensemble temperature (analytic field:
    # base + 0.5*lead + 2*member) is base + 0.5*lead + 2*mean([0..4]) = +4.0.
    grid_lat = np.asarray(LATITUDES)
    grid_lon = np.asarray(LONGITUDES)
    lead = 6
    field, lat_axis, lon_axis = _slice_field(
        dataset, "temperature_2m", lead, _derive_grid_fixture(dataset),
        grid_lat, grid_lon,
    )
    # The whole grid is inside the slice window, so the field equals the mean.
    expected = np.array(
        [
            [
                ensemble_temperature_at(member, lat, lon, lead)
                for member in [0, 1, 2, 3, 4]
            ]
            for lat in LATITUDES
            for lon in LONGITUDES
        ]
    ).reshape(len(LATITUDES), len(LONGITUDES), -1).mean(axis=2)
    assert field.shape == expected.shape
    # Mean over members is the additive constant 4.0 above the member-0 field
    # base; assert the rendered grid matches the ensemble-mean surface.
    for i, lat in enumerate(LATITUDES):
        for j, lon in enumerate(LONGITUDES):
            assert abs(field[i, j] - expected[i, j]) < 1e-6


def test_tile_renders_with_valid_time(client):
    """Under Lifecycle V2, raster tiles can be requested via valid_time."""
    resp = client.get(
        "/v1/maps/gfs/temperature_2m/surface/8/51/98.png?valid_time=2026-07-21T06:00:00Z"
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/png"
    width, height = _png_dimensions(resp.content)
    assert (width, height) == (256, 256)
    assert _png_has_opaque_pixels(resp.content)


def _derive_grid_fixture(dataset):
    """Small helper: derive the grid for the fixture ensemble dataset."""
    from api.services.tiles import _derive_grid

    return _derive_grid(dataset)


# --- Raster cache identity (ACCEPTANCE_REMEDIATION_PLAN §12) ---


def test_tile_cache_key_distinguishes_cycles():
    """GFS 00Z tile A must never equal GFS 12Z tile A in the tile cache."""
    from api.services.tiles import _tile_cache_key

    k00 = _tile_cache_key(
        "gfs", "temperature_2m", "surface", 8, 51, 98, 6, "2026-08-13T00:00:00Z", "gen1"
    )
    k12 = _tile_cache_key(
        "gfs", "temperature_2m", "surface", 8, 51, 98, 6, "2026-08-13T12:00:00Z", "gen1"
    )
    assert k00 != k12
    # Same cycle + same tile -> same key (deterministic).
    assert k00 == _tile_cache_key(
        "gfs", "temperature_2m", "surface", 8, 51, 98, 6, "2026-08-13T00:00:00Z", "gen1"
    )


def test_tile_cache_key_distinguishes_leads():
    """Lead 6 must never share a tile cache key with lead 18."""
    from api.services.tiles import _tile_cache_key

    k6 = _tile_cache_key("gfs", "temperature_2m", "surface", 8, 51, 98, 6, None, "gen1")
    k18 = _tile_cache_key("gfs", "temperature_2m", "surface", 8, 51, 98, 18, None, "gen1")
    assert k6 != k18


def test_tile_cache_key_distinguishes_tile_coordinates():
    """Different tile x/y/z must never share a tile cache key."""
    from api.services.tiles import _tile_cache_key

    a = _tile_cache_key("gfs", "temperature_2m", "surface", 8, 51, 98, 6, None, "gen1")
    b = _tile_cache_key("gfs", "temperature_2m", "surface", 8, 52, 98, 6, None, "gen1")
    c = _tile_cache_key("gfs", "temperature_2m", "surface", 9, 51, 98, 6, None, "gen1")
    assert len({a, b, c}) == 3


def test_wind_10m_tile_slice_deterministic():
    import numpy as np
    import xarray as xr
    from api.services.tiles import _derive_grid, _slice_field

    ds = xr.Dataset(
        data_vars={
            "wind_u_10m": (("lead_time_hours", "latitude", "longitude"), np.ones((1, 4, 4)) * 3.0),
            "wind_v_10m": (("lead_time_hours", "latitude", "longitude"), np.ones((1, 4, 4)) * 4.0),
        },
        coords={
            "lead_time_hours": [6],
            "latitude": [38.0, 38.25, 38.5, 38.75],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )
    grid = _derive_grid(ds)
    field, lat_axis, lon_axis = _slice_field(
        ds, "wind_10m", 6, grid, np.array([[38.5]]), np.array([[-106.5]])
    )
    # speed is hypot(3, 4) * 3.6 = 5.0 * 3.6 = 18.0 km/h
    np.testing.assert_allclose(field, 18.0)


def test_wind_10m_tile_slice_ensemble_mean():
    import numpy as np
    import xarray as xr
    from api.services.tiles import _derive_grid, _slice_field

    # 2 members: member 0 (u=3, v=4 -> speed=5), member 1 (u=6, v=8 -> speed=10)
    # mean scalar speed is (5 + 10) / 2 = 7.5 m/s = 27.0 km/h
    u_data = np.zeros((2, 1, 4, 4))
    u_data[0] = 3.0
    u_data[1] = 6.0
    v_data = np.zeros((2, 1, 4, 4))
    v_data[0] = 4.0
    v_data[1] = 8.0

    ds = xr.Dataset(
        data_vars={
            "wind_u_10m": (("member", "lead_time_hours", "latitude", "longitude"), u_data),
            "wind_v_10m": (("member", "lead_time_hours", "latitude", "longitude"), v_data),
        },
        coords={
            "member": [1, 2],
            "lead_time_hours": [6],
            "latitude": [38.0, 38.25, 38.5, 38.75],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )
    grid = _derive_grid(ds)
    field, lat_axis, lon_axis = _slice_field(
        ds, "wind_10m", 6, grid, np.array([[38.5]]), np.array([[-106.5]])
    )
    # mean scalar speed is 27.0 km/h
    np.testing.assert_allclose(field, 27.0)


def test_tile_cache_serves_identical_requests():
    """A repeated identical tile request is served from the server cache."""
    from api.services.tiles import _tile_cache, _tile_cache_get, _tile_cache_set, _tile_cache_key

    _tile_cache.clear()
    key = _tile_cache_key("gfs", "temperature_2m", "surface", 8, 51, 98, 6, None, "gen1")
    assert _tile_cache_get(key) is None
    _tile_cache_set(key, b"PNG-DATA")
    assert _tile_cache_get(key) == b"PNG-DATA"
    _tile_cache.clear()


def test_phase1a_color_stops_and_data_ranges():
    """Phase 1A variables define monotonically increasing color stops and valid data ranges."""
    from api.services.tiles import _color_stops, _data_range

    phase1a_vars = [
        "temperature_2m",
        "precipitation_rate",
        "relative_humidity_2m",
        "wind_gust",
        "visibility",
        "snow_depth",
    ]
    for var in phase1a_vars:
        stops = _color_stops(var)
        assert len(stops) >= 2
        # Values must be strictly increasing
        values = [val for val, rgb in stops]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))
        # Colors must be valid RGB triplets in [0, 255]
        for val, (r, g, b) in stops:
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

        d_min, d_max = _data_range(var)
        assert d_min < d_max


def _extract_rgba(png: bytes) -> np.ndarray:
    """Decode raw RGBA scanlines from PNG bytes (filter type 0)."""
    import numpy as np

    width, height, idat = _extract_idat(png)
    raw = zlib.decompress(idat)
    stride = width * 4
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for row in range(height):
        offset = row * (stride + 1) + 1
        line = raw[offset : offset + stride]
        rgba[row] = np.frombuffer(line, dtype=np.uint8).reshape(width, 4)
    return rgba


def test_periodic_grid_detection():
    """_TileGrid.is_periodic_lon detects global 360° longitude domains."""
    from api.services.tiles import _TileGrid

    gfs = _TileGrid(lat_start=-90.0, lat_step=0.25, lat_count=721, lon_start=0.0, lon_step=0.25, lon_count=1440, lat_reversed=False, lon_reversed=False)
    assert gfs.is_periodic_lon is True

    gefs = _TileGrid(lat_start=-90.0, lat_step=0.5, lat_count=361, lon_start=0.0, lon_step=0.5, lon_count=720, lat_reversed=False, lon_reversed=False)
    assert gefs.is_periodic_lon is True

    global_180 = _TileGrid(lat_start=-90.0, lat_step=0.25, lat_count=721, lon_start=-180.0, lon_step=0.25, lon_count=1440, lat_reversed=False, lon_reversed=False)
    assert global_180.is_periodic_lon is True

    regional = _TileGrid(lat_start=38.0, lat_step=0.25, lat_count=4, lon_start=-107.0, lon_step=0.25, lon_count=4, lat_reversed=False, lon_reversed=False)
    assert regional.is_periodic_lon is False


def test_inside_grid_periodic_and_regional():
    """_inside_grid accepts wrapped longitudes on periodic grids but rejects on regional grids."""
    import numpy as np
    from api.services.tiles import _TileGrid, _inside_grid

    gfs = _TileGrid(lat_start=-90.0, lat_step=0.25, lat_count=721, lon_start=0.0, lon_step=0.25, lon_count=1440, lat_reversed=False, lon_reversed=False)
    # Longitudes in [359.75, 360) are inside the global periodic grid
    assert bool(_inside_grid(gfs, np.array([40.0]), np.array([359.956]))[0]) is True
    # Out-of-bounds latitude is still rejected
    assert bool(_inside_grid(gfs, np.array([95.0]), np.array([359.956]))[0]) is False

    regional = _TileGrid(lat_start=38.0, lat_step=0.25, lat_count=4, lon_start=-107.0, lon_step=0.25, lon_count=4, lat_reversed=False, lon_reversed=False)
    # Regional grid strictly enforces longitude boundaries
    assert bool(_inside_grid(regional, np.array([38.125]), np.array([-105.0]))[0]) is False
    assert bool(_inside_grid(regional, np.array([38.125]), np.array([-106.875]))[0]) is True


def test_gfs_west_of_0_seam_not_transparent():
    """GFS 0.25° tiles immediately west of 0° have no transparent seam."""
    import numpy as np
    import xarray as xr
    from api.services.tiles import _select_tile_window, _render_window_to_png

    lats = np.arange(90.0, -90.25, -0.25)
    lons = np.arange(0.0, 360.0, 0.25)
    data = np.full((1, len(lats), len(lons)), 20.0, dtype=np.float32)
    data[0, :, 0] = 10.0   # lon 0.0
    data[0, :, -1] = 50.0  # lon 359.75

    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), data)},
        coords={"lead_time_hours": [0], "latitude": lats, "longitude": lons},
    )

    for z, x in [(4, 7), (8, 127)]:
        y = 2 ** (z - 1)
        win = _select_tile_window(ds, variable="temperature_2m", lead=0, zoom=z, x=x, y=y)
        png = _render_window_to_png(win, variable="temperature_2m", zoom=z, x=x, y=y, cache_key=())
        rgba = _extract_rgba(png)
        # All columns at the seam must be 100% opaque (alpha == 255)
        assert np.all(rgba[:, -5:, 3] == 255), f"GFS z={z} x={x} right seam contains transparent pixels"


def test_gefs_west_of_0_seam_not_transparent():
    """GEFS 0.50° tiles immediately west of 0° have no transparent seam."""
    import numpy as np
    import xarray as xr
    from api.services.tiles import _select_tile_window, _render_window_to_png

    lats = np.arange(90.0, -90.5, -0.5)
    lons = np.arange(0.0, 360.0, 0.5)
    data = np.full((1, len(lats), len(lons)), 20.0, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), data)},
        coords={"lead_time_hours": [0], "latitude": lats, "longitude": lons},
    )

    for z, x in [(4, 7), (8, 127)]:
        y = 2 ** (z - 1)
        win = _select_tile_window(ds, variable="temperature_2m", lead=0, zoom=z, x=x, y=y)
        png = _render_window_to_png(win, variable="temperature_2m", zoom=z, x=x, y=y, cache_key=())
        rgba = _extract_rgba(png)
        assert np.all(rgba[:, -5:, 3] == 255), f"GEFS z={z} x={x} right seam contains transparent pixels"


def test_periodic_nearest_neighbor_sampling():
    """Pixels near 360° select conceptual wrapped column 360.0° (backed by stored lon 0.0°)."""
    import numpy as np
    import xarray as xr
    from api.services.tiles import _select_tile_window

    lats = np.arange(90.0, -90.25, -0.25)
    lons = np.arange(0.0, 360.0, 0.25)
    data = np.full((1, len(lats), len(lons)), 20.0, dtype=np.float32)
    data[0, :, 0] = 10.0   # lon 0.0 value
    data[0, :, -1] = 50.0  # lon 359.75 value

    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), data)},
        coords={"lead_time_hours": [0], "latitude": lats, "longitude": lons},
    )

    win = _select_tile_window(ds, variable="temperature_2m", lead=0, zoom=8, x=127, y=128)
    # Window longitude axis must expose wrapped column at 360.0
    assert win.lon_axis[-1] == 360.0
    # Wrapped column at 360.0 must carry data from stored lon 0.0 (10.0)
    assert win.field[0, -1] == 10.0
    # Penultimate column must carry data from stored lon 359.75 (50.0)
    assert win.field[0, -2] == 50.0


def test_east_side_of_0_continuous():
    """Tiles immediately east of 0° (x=8) remain valid, opaque, and meet seamlessly."""
    import numpy as np
    import xarray as xr
    from api.services.tiles import _select_tile_window, _render_window_to_png

    lats = np.arange(90.0, -90.25, -0.25)
    lons = np.arange(0.0, 360.0, 0.25)
    data = np.full((1, len(lats), len(lons)), 20.0, dtype=np.float32)
    data[0, :, 0] = 10.0

    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), data)},
        coords={"lead_time_hours": [0], "latitude": lats, "longitude": lons},
    )

    win = _select_tile_window(ds, variable="temperature_2m", lead=0, zoom=4, x=8, y=8)
    png = _render_window_to_png(win, variable="temperature_2m", zoom=4, x=8, y=8, cache_key=())
    rgba = _extract_rgba(png)
    assert np.all(rgba[:, :5, 3] == 255)
    assert win.field[0, 0] == 10.0


def test_plus_minus_180_interior_regression():
    """±180° is an interior grid location on [0, 360) stores and does not trigger wrap."""
    import numpy as np
    import xarray as xr
    from api.services.tiles import _select_tile_window, _render_window_to_png

    lats = np.arange(90.0, -90.25, -0.25)
    lons = np.arange(0.0, 360.0, 0.25)
    data = np.full((1, len(lats), len(lons)), 20.0, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), data)},
        coords={"lead_time_hours": [0], "latitude": lats, "longitude": lons},
    )

    win_180 = _select_tile_window(ds, variable="temperature_2m", lead=0, zoom=4, x=0, y=8)
    assert win_180.lon_axis[-1] <= win_180.grid.lon_end
    png = _render_window_to_png(win_180, variable="temperature_2m", zoom=4, x=0, y=8, cache_key=())
    rgba = _extract_rgba(png)
    assert np.all(rgba[:, :, 3] == 255)


def test_sharded_v1_periodic_wrap():
    """Sharded v1 store format correctly appends column 0 at 360.0 for periodic grids."""
    import json
    import tempfile
    from pathlib import Path
    import numpy as np
    import xarray as xr
    from api.services.tiles import _derive_grid, _slice_field, _align_longitudes, TILE_SIZE
    from tests.test_sharded_reader import _build_test_shard

    latitudes = [90.0 - i * 0.25 for i in range(721)]
    longitudes = [0.0 + j * 0.25 for j in range(1440)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_dir = Path(tmp_dir) / "gefs_test.zarr"
        store_dir.mkdir(parents=True, exist_ok=True)

        manifest_dir = store_dir / "__commit__" / "v1"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(
            json.dumps({
                "manifest_schema_version": 1,
                "generation": "gen_test",
                "storage_format_version": "sharded_v1",
            })
        )

        # Write shard container for member 1, lead 0 of temperature_2m
        # Chunk row 4, col 0 (lon 0.0) has value 4 * 15 + 0 + 10.0 = 70.0
        # Chunk row 4, col 14 (lon 359.75) has value 4 * 15 + 14 + 10.0 = 84.0
        t_shard = _build_test_shard(val_offset=10.0)
        shard_file = store_dir / "temperature_2m" / "shard.mem001_L0000.shard"
        shard_file.parent.mkdir(parents=True, exist_ok=True)
        shard_file.write_bytes(t_shard)

        ds_meta = xr.Dataset(
            data_vars={
                "temperature_2m": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((1, 1, 721, 1440), dtype=np.float32)),
            },
            coords={
                "member": [1],
                "lead_time_hours": [0],
                "latitude": latitudes,
                "longitude": longitudes,
            },
        )
        ds_meta.to_zarr(str(store_dir), mode="a", consolidated=True, zarr_format=2)

        grid = _derive_grid(ds_meta)
        z, x, y = 4, 7, 8
        n = 2 ** z
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
            ds_meta, "temperature_2m", 0, grid, pixel_lats, lon_native, expected_members=1, store_path=str(store_dir)
        )
        assert lon_axis[-1] == 360.0
        # Wrapped column at 360.0 must carry data from stored lon 0.0 (chunk 60 = 70.0)
        assert field[0, -1] == 70.0
        # Penultimate column must carry data from stored lon 359.75 (chunk 74 = 84.0)
        assert field[0, -2] == 84.0
