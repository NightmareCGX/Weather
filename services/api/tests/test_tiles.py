"""Contract and integration tests for the map tile endpoint.

These tests verify that `/v1/maps/{model}/{variable}/{level}/{z}/{x}/{y}.png`
renders genuine forecast data from the fixture Zarr stores as a PNG, and that
invalid selections return proper errors. When PostgreSQL is unreachable they
skip, following the existing convention.
"""

import struct
import zlib


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


def test_tile_cache_serves_identical_requests():
    """A repeated identical tile request is served from the server cache."""
    from api.services.tiles import _tile_cache, _tile_cache_get, _tile_cache_set, _tile_cache_key

    _tile_cache.clear()
    key = _tile_cache_key("gfs", "temperature_2m", "surface", 8, 51, 98, 6, None, "gen1")
    assert _tile_cache_get(key) is None
    _tile_cache_set(key, b"PNG-DATA")
    assert _tile_cache_get(key) == b"PNG-DATA"
    _tile_cache.clear()
