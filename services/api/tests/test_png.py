"""Unit tests for the dependency-free PNG encoder (api/core/png.py)."""

import struct
import zlib

import pytest

from api.core.png import encode_rgba_png


def _chunks(png: bytes) -> list[tuple[bytes, bytes]]:
    """Split a PNG into (type, data) chunks."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    chunks = []
    pos = 8
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        chunk_type = png[pos + 4 : pos + 8]
        data = png[pos + 8 : pos + 8 + length]
        chunks.append((chunk_type, data))
        pos += 12 + length
    return chunks


def _decode(png: bytes) -> tuple[int, int, bytes]:
    """Decode the RGBA scanlines of an encoder-produced PNG."""
    width = height = 0
    idat = b""
    for chunk_type, data in _chunks(png):
        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif chunk_type == b"IDAT":
            idat += data
    raw = zlib.decompress(idat)
    stride = width * 4
    out = bytearray()
    for row in range(height):
        offset = row * (stride + 1)
        assert raw[offset] == 0, "encoder must use filter type 0"
        out += raw[offset + 1 : offset + 1 + stride]
    return width, height, bytes(out)


def test_png_signature_and_ihdr():
    png = encode_rgba_png(bytes(16), 2, 2)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    types = [t for t, _ in _chunks(png)]
    assert types[0] == b"IHDR"
    assert types[-1] == b"IEND"
    assert b"IDAT" in types


def test_png_roundtrip_solid_color():
    width, height = 4, 4
    pixels = bytes([255, 0, 0, 255]) * (width * height)
    png = encode_rgba_png(pixels, width, height)
    w, h, decoded = _decode(png)
    assert (w, h) == (width, height)
    assert decoded == pixels


def test_png_roundtrip_gradient():
    width, height = 2, 3
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels += bytes((x * 100 % 256, y * 80 % 256, (x + y) * 40 % 256, 255))
    png = encode_rgba_png(bytes(pixels), width, height)
    _, _, decoded = _decode(png)
    assert decoded == bytes(pixels)


def test_png_transparent_pixels_preserved():
    width, height = 2, 2
    pixels = bytes([0, 0, 0, 0, 10, 20, 30, 255, 40, 50, 60, 128, 70, 80, 90, 255])
    png = encode_rgba_png(pixels, width, height)
    _, _, decoded = _decode(png)
    assert decoded == pixels


def test_png_rejects_wrong_buffer_length():
    with pytest.raises(ValueError):
        encode_rgba_png(bytes(15), 2, 2)
