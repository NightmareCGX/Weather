"""Minimal dependency-free PNG encoding for map tile rendering.

The API serving tier renders forecast raster tiles from Zarr data. Pillow is
not a declared dependency of the API service, so this module implements a
small, correct PNG encoder for 8-bit RGBA truecolor images using only the
Python standard library (``zlib`` + ``struct``).

Only the subset needed for map tiles is implemented: one RGBA frame, no
interlacing, no palette. The encoder is deterministic and side-effect free so
it can be unit tested without external services.
"""

from __future__ import annotations

import struct
import zlib

#: PNG signature prefix required by the format.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: Color type 6 = truecolor with alpha (RGBA).
_COLOR_TYPE_RGBA = 6
#: Bit depth 8 (one byte per channel).
_BIT_DEPTH = 8


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Encode one PNG chunk (length + type + data + CRC32).

    Args:
        chunk_type: The 4-byte ASCII chunk type (e.g. ``b"IHDR"``).
        data: The chunk payload.

    Returns:
        The complete chunk bytes.
    """
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def encode_rgba_png(pixels: bytes, width: int, height: int) -> bytes:
    """Encode an RGBA pixel buffer as a PNG.

    Args:
        pixels: Raw RGBA bytes, ``width * height * 4`` in length, row-major
            top-to-bottom.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The complete PNG file bytes.

    Raises:
        ValueError: If the pixel buffer length does not match the dimensions.
    """
    expected = width * height * 4
    if len(pixels) != expected:
        raise ValueError(
            f"RGBA buffer length {len(pixels)} does not match "
            f"{width}x{height} ({expected})."
        )

    # IHDR: width, height, bit depth, color type, compression, filter, interlace.
    ihdr = _chunk(
        b"IHDR",
        struct.pack(
            ">IIBBBBB",
            width,
            height,
            _BIT_DEPTH,
            _COLOR_TYPE_RGBA,
            0,  # compression method (deflate)
            0,  # filter method (adaptive)
            0,  # interlace (none)
        ),
    )

    # Each scanline is prefixed with filter type 0 (None).
    stride = width * 4
    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)
        scanlines += pixels[row * stride : (row + 1) * stride]

    idat = _chunk(b"IDAT", zlib.compress(bytes(scanlines), 9))
    iend = _chunk(b"IEND", b"")
    return _PNG_SIGNATURE + ihdr + idat + iend
