"""Experiment 2: per-chunk local (GRIB2-style) packing vs global fixed-point scaling.

Motivation: a *global* scale must cover the whole globe's dynamic range, which forces
int32 at a 0.001 error budget. A *per-chunk* min/max scale only has to cover the range
inside one 100x100 tile, which is a fraction of a degree for smooth NWP fields -- so a
16-bit container can carry far finer resolution than 0.001.

Run:  .venv/Scripts/python.exe artifacts/dtype_bench/bench_local_packing.py
"""

from __future__ import annotations

import struct
import time

import numpy as np
from numcodecs import Zstd

LAT, LON, CHUNK = 721, 1440, 100
LAT_CHUNKS, LON_CHUNKS = 8, 15
ZSTD = Zstd(level=5)
RNG = np.random.default_rng(7)


def spectral_field(shape, slope=2.2, amplitude=1.0, fine_noise=0.0, seed=None):
    """Synthesize a smooth field with a k^-slope power spectrum (realistic NWP look).

    ``fine_noise`` adds white noise on top, which is what sets the float32 mantissa
    entropy and therefore the *baseline* compressibility.
    """
    rng = np.random.default_rng(seed) if seed is not None else RNG
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(ky**2 + kx**2)
    k[0, 0] = 1.0
    spec = amplitude / (k**slope)
    spec[0, 0] = 0.0
    phase = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    field = np.real(np.fft.ifft2(spec * phase))
    field /= field.std()
    if fine_noise:
        field = field + rng.standard_normal(shape) * fine_noise
    return field


def make_temperature(fine_noise, nan_frac=0.0, seed=11):
    """2 m temperature in degC, global dynamic range roughly -55..+50."""
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    lon = np.linspace(0.0, 359.75, LON)[None, :]
    base = 30.0 - 55.0 * np.abs(np.sin(np.deg2rad(lat)))
    field = base + 9.0 * spectral_field((LAT, LON), slope=2.2, seed=seed) + fine_noise * RNG.standard_normal((LAT, LON))
    field = np.asarray(field, dtype=np.float32)
    if nan_frac:
        field[RNG.random((LAT, LON)) < nan_frac] = np.nan
    return field


def chunks_of(arr):
    for r in range(LAT_CHUNKS):
        for c in range(LON_CHUNKS):
            buf = np.full((CHUNK, CHUNK), np.nan, dtype=np.float32)
            r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
            c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
            buf[: r1 - r0, : c1 - c0] = arr[r0:r1, c0:c1]
            yield buf


def measure(arr, encode, decode, label, extra_bytes_per_chunk=0):
    """Return (total_bytes, encode_ms, decode_ms, max_abs_err)."""
    t0 = time.perf_counter()
    payloads = [encode(c) for c in chunks_of(arr)]
    t1 = time.perf_counter()
    total = sum(len(p) for p in payloads)

    out = np.full((LAT, LON), np.nan, dtype=np.float64)
    for idx, p in enumerate(payloads):
        r, c = divmod(idx, LON_CHUNKS)
        r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
        c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
        out[r0:r1, c0:c1] = decode(p)[: r1 - r0, : c1 - c0]
    t2 = time.perf_counter()

    finite = np.isfinite(arr)
    err = float(np.abs(out[finite] - arr[finite]).max()) if finite.any() else 0.0
    if (np.isnan(out) != np.isnan(arr)).any():
        err = float("inf")
    return total, (t1 - t0) * 1e3, (t2 - t1) * 1e3, err


# --------------------------------------------------------------------------------------
# Encoders / decoders
# --------------------------------------------------------------------------------------
def enc_float32(c):
    return ZSTD.encode(c.tobytes(order="C"))


def dec_float32(p):
    return np.frombuffer(ZSTD.decode(p), dtype=np.float32).reshape(CHUNK, CHUNK)


def global_int(scale, np_dtype, sentinel):
    limit = np.iinfo(np_dtype).max

    def enc(c):
        q = np.clip(np.round(np.nan_to_num(c, nan=0.0) / scale), -limit, limit).astype(np_dtype)
        q = np.where(np.isnan(c), sentinel, q).astype(np_dtype)
        return ZSTD.encode(q.tobytes(order="C"))

    def dec(p):
        q = np.frombuffer(ZSTD.decode(p), dtype=np_dtype).reshape(CHUNK, CHUNK)
        vals = q.astype(np.float32) * np.float32(scale)
        return np.where(q == sentinel, np.float32(np.nan), vals).astype(np.float64)

    return enc, dec


# NaN code reserved at the top of the uint16 container.
LOCAL_NAN = np.uint16(65535)
LOCAL_MAX = 65533.0


def local_pack(use_bitmap):
    """Per-chunk min/max packing: rescale each tile to the 16-bit container.

    Layout per chunk: float32 ``lo``, float32 ``scale``, uint32 ``body_len``,
    then the Zstd body (uint16 codes) and, optionally, a packed validity bitmap.
    A per-chunk scale is what makes 16 bits sufficient: a 100x100 tile of a smooth
    NWP field spans a small fraction of the global range, so the quantisation step
    lands far below the 0.001 error budget.
    """

    def enc(c):
        finite = np.isfinite(c)
        if not finite.any():
            return struct.pack("<ffI", 0.0, 0.0, 0) + ZSTD.encode(
                np.full(CHUNK * CHUNK, LOCAL_NAN, dtype=np.uint16).tobytes()
            )
        lo = float(np.nanmin(c))
        span = float(np.nanmax(c)) - lo
        scale = (span / LOCAL_MAX) if span > 0 else 0.0
        vals = np.nan_to_num(c, nan=lo)
        codes = (
            np.clip(np.round((vals - lo) / scale), 0, LOCAL_MAX).astype(np.uint16)
            if scale
            else np.zeros((CHUNK, CHUNK), dtype=np.uint16)
        )
        if use_bitmap:
            mask = ZSTD.encode(np.packbits(finite.ravel()).tobytes())
            # 4 bytes is the packed-bitmap raw length only if we prepend it; keep fixed split.
            body = ZSTD.encode(codes.tobytes(order="C"))
            return struct.pack("<ffI", lo, scale, len(body)) + body + mask
        body = ZSTD.encode(np.where(finite, codes, LOCAL_NAN).astype(np.uint16).tobytes(order="C"))
        return struct.pack("<ffI", lo, scale, len(body)) + body

    def dec(p):
        lo, scale, body_len = struct.unpack_from("<ffI", p, 0)
        if body_len == 0:
            return np.full((CHUNK, CHUNK), np.nan, dtype=np.float64)
        codes = (
            np.frombuffer(ZSTD.decode(p[12 : 12 + body_len]), dtype=np.uint16)
            .reshape(CHUNK, CHUNK)
            .copy()
        )
        vals = lo + codes.astype(np.float64) * scale
        if use_bitmap:
            mask = np.unpackbits(
                np.frombuffer(ZSTD.decode(p[12 + body_len :]), dtype=np.uint8)
            )[: CHUNK * CHUNK].reshape(CHUNK, CHUNK)
            return np.where(mask.astype(bool), vals, np.nan)
        return np.where(codes == LOCAL_NAN, np.nan, vals)

    return enc, dec



def run_field(name, arr, variants):
    raw = arr.size * 4
    print("=" * 112)
    finite = arr[np.isfinite(arr)]
    print(f"{name}   range [{finite.min():.2f}, {finite.max():.2f}]  nan {100 * np.isnan(arr).mean():.1f}%")
    print("=" * 112)
    base = None
    for label, enc, dec in variants:
        total, ems, dms, err = measure(arr, enc, dec, label)
        if base is None:
            base = total
        print(
            f"  {label:<40} {total / 1e6:7.3f} MB   base/zstd {raw / total:5.2f}x   "
            f"vs-current {base / total:5.2f}x   enc {ems:6.1f} ms  dec {dms:6.1f} ms  "
            f"maxerr {err:.7f}"
        )
    print()
    return base


def main():
    z = [
        ("[CURRENT] float32 + Zstd5", enc_float32, dec_float32),
        ("int16 global scale=0.01", *global_int(0.01, np.int16, np.int16(-32768))),
        ("int16 global scale=0.001", *global_int(0.001, np.int16, np.int16(-32768))),
        ("int32 global scale=0.001", *global_int(0.001, np.int32, np.int32(-2147483648))),
        ("LOCAL pack 16-bit + NaN sentinel", *local_pack(use_bitmap=False)),
        ("LOCAL pack 16-bit + NaN bitmap", *local_pack(use_bitmap=True)),
    ]

    print()
    print("### A. Smoothness sensitivity (temperature, no NaN)")
    print("    The repo reports 4.21x on real GFS data; baseline compressibility drives the")
    print("    marginal gain from quantization, so bracket it.\n")
    for fn in (0.0, 0.02, 0.1, 0.4, 1.0):
        arr = make_temperature(fine_noise=fn, nan_frac=0.0, seed=11)
        run_field(f"temperature_2m  fine_noise={fn}", arr, z)

    print("\n### B. With 6% NaN (masked members / undefined cells)\n")
    arr = make_temperature(fine_noise=0.1, nan_frac=0.06, seed=12)
    run_field("temperature_2m  fine_noise=0.1  nan=6%", arr, z)

    print("\n### C. Wide-dynamic-range variables at fine_noise=0.1 (these break int16 global)\n")
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    gust = np.abs(spectral_field((LAT, LON), slope=2.0, seed=21)) * 120.0 + 5.0
    gust = np.asarray(gust, dtype=np.float32)
    run_field("wind_gust_kmh  (0..250 km/h)", gust, z)

    prate = np.abs(spectral_field((LAT, LON), slope=2.6, seed=31)) ** 2 * 40.0
    prate = np.asarray(prate, dtype=np.float32)
    run_field("precipitation_rate_mmh  (0..200 mm/h)", prate, z)

    flag = np.asarray((spectral_field((LAT, LON), slope=2.5, seed=41) > 0.5).astype(np.float32))
    u8 = [
        ("[CURRENT] float32 + Zstd5", enc_float32, dec_float32),
        (
            "uint8 + Zstd5",
            lambda c: ZSTD.encode(np.asarray(np.nan_to_num(c, nan=0.0), dtype=np.uint8).tobytes()),
            lambda p: np.frombuffer(ZSTD.decode(p), dtype=np.uint8).reshape(CHUNK, CHUNK).astype(np.float64),
        ),
        (
            "PackBits + Zstd5",
            lambda c: ZSTD.encode(np.packbits(np.nan_to_num(c, nan=0.0).astype(bool)).tobytes()),
            lambda p: np.unpackbits(np.frombuffer(ZSTD.decode(p), dtype=np.uint8))[: CHUNK * CHUNK]
            .reshape(CHUNK, CHUNK)
            .astype(np.float64),
        ),
    ]
    run_field("crain_csnow_cfrzr_cicep  (0/1 flags)", flag, u8)


if __name__ == "__main__":
    main()
