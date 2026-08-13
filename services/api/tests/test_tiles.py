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
    assert resp.headers["Cache-Control"] == "public, max-age=300"
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
