"""Measure real storage/CPU cost of candidate numeric encodings for the sharded_v1 store.

Mirrors production exactly: 721x1440 field, split into 100x100 chunks (edge chunks NaN-padded
to a full 100x100 buffer, as ingestion/core/zarr_writer.encode_region_sharded_v1 does),
each chunk compressed independently, 120 chunks per shard.

Run:  .venv/Scripts/python.exe artifacts/dtype_bench/bench_dtype_storage.py
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import numpy as np
from numcodecs import Blosc, FixedScaleOffset, Zstd

LAT, LON = 721, 1440
CHUNK = 100
LAT_CHUNKS = (LAT + CHUNK - 1) // CHUNK  # 8
LON_CHUNKS = (LON + CHUNK - 1) // CHUNK  # 15
N_CHUNKS = LAT_CHUNKS * LON_CHUNKS  # 120

I16_NAN = np.int16(-32768)
I32_NAN = np.int32(-2147483648)

RNG = np.random.default_rng(20260918)


# --------------------------------------------------------------------------------------
# Synthetic but meteorologically-shaped fields
# --------------------------------------------------------------------------------------
def _smooth_noise(shape, sigma_cells, passes=3):
    """Low-pass random field: gives synoptic-scale structure like a real NWP analysis."""
    out = np.zeros(shape, dtype=np.float64)
    for _ in range(passes):
        f = RNG.standard_normal(shape)
        # cheap separable box blur via cumsum, applied sigma_cells times
        k = max(1, int(sigma_cells))
        c = np.cumsum(f, axis=0)
        f = (c[k:] - c[:-k]) / k
        c = np.cumsum(f, axis=1)
        f = (c[:, k:] - c[:, :-k]) / k
        pad = np.pad(f, ((0, shape[0] - f.shape[0]), (0, shape[1] - f.shape[1])), mode="edge")
        out += pad / (pad.std() + 1e-12)
    return out / passes


def make_t2m():
    """2 m temperature in degC: strong latitudinal gradient + synoptic waves + fine grain."""
    lat = np.linspace(90.0, -90.0, LAT)
    lon = np.linspace(0.0, 359.75, LON)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    base = 30.0 - 55.0 * np.abs(np.sin(np.deg2rad(lat_g)))
    waves = 6.0 * np.sin(np.deg2rad(3.0 * lon_g)) * np.cos(np.deg2rad(2.0 * lat_g))
    field = base + waves + 8.0 * _smooth_noise((LAT, LON), 12) + RNG.normal(0, 0.15, (LAT, LON))
    # masked cells (land/sea masks, missing members) - production stores NaN, not 0
    field = np.asarray(field, dtype=np.float32)
    field[RNG.random((LAT, LON)) < 0.06] = np.nan
    return field


def make_wind_u():
    """10 m u-component in m/s: narrower range than temperature."""
    lat = np.linspace(90.0, -90.0, LAT)
    lon = np.linspace(0.0, 359.75, LON)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    base = -12.0 * np.sin(np.deg2rad(4.0 * lat_g)) * np.cos(np.deg2rad(2.0 * lon_g))
    field = base + 10.0 * _smooth_noise((LAT, LON), 10) + RNG.normal(0, 0.3, (LAT, LON))
    field = np.asarray(field, dtype=np.float32)
    field[RNG.random((LAT, LON)) < 0.03] = np.nan
    return field


def make_precip_3h():
    """3 h precipitation accumulation in mm: mostly zero, heavy right tail."""
    field = np.abs(_smooth_noise((LAT, LON), 8)) ** 2.5 * 9.0
    field = np.where(field < 0.05, 0.0, field)
    field = np.asarray(field, dtype=np.float32)
    field[RNG.random((LAT, LON)) < 0.10] = np.nan
    return field


def make_flag():
    """Categorical 0/1 field (crain/csnow/cfrzr/cicep)."""
    return np.asarray((_smooth_noise((LAT, LON), 8) > 0.4).astype(np.float32))


FIELDS = {
    "temperature_2m_C": make_t2m,
    "wind_u_10m_ms": make_wind_u,
    "precip_amount_3h_mm": make_precip_3h,
    "flag_0_1": make_flag,
}


# --------------------------------------------------------------------------------------
# Encoding variants. Each returns (chunk_payload_bytes_list, decoder_closure)
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    total_bytes: int
    encode_ms: float
    decode_ms: float
    max_abs_err: float
    worst_value_clipped: bool


def _chunks(arr: np.ndarray):
    """Yield 100x100 windows, NaN-padded to the full buffer (production behaviour)."""
    for r in range(LAT_CHUNKS):
        for c in range(LON_CHUNKS):
            buf = np.full((CHUNK, CHUNK), np.nan, dtype=np.float32)
            r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
            c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
            buf[: r1 - r0, : c1 - c0] = arr[r0:r1, c0:c1]
            yield buf


def _run(name, arr, encode_fn, decode_fn, ref):
    t0 = time.perf_counter()
    payloads = [encode_fn(c) for c in _chunks(arr)]
    t1 = time.perf_counter()
    total = sum(len(p) for p in payloads)
    out = np.empty((LAT, LON), dtype=np.float64)
    out.fill(np.nan)
    clipped = False
    for idx, p in enumerate(payloads):
        r, c = divmod(idx, LON_CHUNKS)
        r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
        c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
        dec, did_clip = decode_fn(p)
        out[r0:r1, c0:c1] = dec[: r1 - r0, : c1 - c0]
        clipped = clipped or did_clip
    t2 = time.perf_counter()
    finite = np.isfinite(ref)
    err = np.abs(out[finite] - ref[finite]).max() if finite.any() else 0.0
    # NaN placement must survive too
    nan_mismatch = int((np.isnan(out) != np.isnan(ref)).sum())
    if nan_mismatch:
        err = max(err, float("inf"))
    return Result(name, total, (t1 - t0) * 1e3, (t2 - t1) * 1e3, float(err), clipped)


def variants(arr: np.ndarray):
    zstd = Zstd(level=5)
    out = []

    # --- V1: production baseline - float32 + Zstd(5), NaN kept in the float payload
    out.append(
        _run(
            "V1 float32 + Zstd5 [CURRENT]",
            arr,
            lambda c: zstd.encode(c.tobytes(order="C")),
            lambda p: (np.frombuffer(zstd.decode(p), dtype=np.float32).reshape(CHUNK, CHUNK), False),
            arr,
        )
    )

    # --- V2/V3: byte shuffle / bitshuffle before zstd, still lossless float32
    for label, shuffle in (
        ("V2 float32 + Blosc(zstd5, SHUFFLE)", Blosc.SHUFFLE),
        ("V3 float32 + Blosc(zstd5, BITSHUFFLE)", Blosc.BITSHUFFLE),
    ):
        bl = Blosc(cname="zstd", clevel=5, shuffle=shuffle)
        out.append(
            _run(
                label,
                arr,
                lambda c, bl=bl: bl.encode(c.tobytes(order="C")),
                lambda p, bl=bl: (
                    np.frombuffer(bl.decode(p), dtype=np.float32).reshape(CHUNK, CHUNK),
                    False,
                ),
                arr,
            )
        )

    # --- V4-V6: integer scaling with an in-band sentinel for NaN
    for label, np_dtype, sentinel, scale in (
        ("V4 int16 scale=0.01 + Zstd5", np.int16, I16_NAN, 0.01),
        ("V5 int16 scale=0.001 + Zstd5", np.int16, I16_NAN, 0.001),
        ("V6 int32 scale=0.001 + Zstd5", np.int32, I32_NAN, 0.001),
    ):
        limit = np.iinfo(np_dtype).max
        clipped_flag = [False]

        def enc(c, np_dtype=np_dtype, sentinel=sentinel, scale=scale, limit=limit, cf=clipped_flag):
            vals = np.nan_to_num(c, nan=0.0)
            over = bool((np.abs(vals / scale) > limit).any())
            q = np.clip(np.round(vals / scale), -limit, limit).astype(np.int64).astype(np_dtype)
            q = np.where(np.isnan(c), sentinel, q).astype(np_dtype)
            if over:
                cf[0] = True
            return zstd.encode(q.tobytes(order="C"))

        def dec(p, np_dtype=np_dtype, sentinel=sentinel, scale=scale):
            q = np.frombuffer(zstd.decode(p), dtype=np_dtype).reshape(CHUNK, CHUNK)
            vals = q.astype(np.float32) * np.float32(scale)
            return np.where(q == sentinel, np.float32(np.nan), vals), False

        r = _run(label, arr, enc, dec, arr)
        r.worst_value_clipped = clipped_flag[0]
        out.append(r)

    # --- V7/V8: FixedScaleOffset (the numcodecs-native integer scaling filter) + zstd
    for label, scale, astype in (
        ("V7 FixedScaleOffset(s=1000,i4) + Zstd5", 1000, "i4"),
        ("V8 FixedScaleOffset(s=100,i2) + Zstd5", 100, "i2"),
    ):
        fso = FixedScaleOffset(offset=0.0, scale=float(scale), dtype="f4", astype=astype)
        z = Zstd(level=5)

        def enc(c, fso=fso, z=z):
            filled = np.nan_to_num(c, nan=0.0)
            return z.encode(fso.encode(filled.tobytes(order="C")))

        def dec(p, fso=fso, z=z):
            raw = z.decode(p)
            vals = np.frombuffer(fso.decode(raw), dtype="f4").reshape(CHUNK, CHUNK)
            return vals, False

        out.append(_run(label, arr, enc, dec, arr))

    # --- V9: int16 value plane + packed 1-bit validity mask (GRIB-style bitmap)
    for label, scale in (("V9 int16 s=0.01 + NaN bitmap", 0.01), ("V10 int16 s=0.001 + NaN bitmap", 0.001)):
        def enc(c, scale=scale):
            valid = np.isfinite(c)
            q = np.clip(np.round(np.nan_to_num(c, nan=0.0) / scale), -32767, 32767).astype(np.int16)
            mask = np.packbits(valid.ravel(order="C"))
            qc = zstd.encode(q.tobytes(order="C"))
            mc = zstd.encode(mask.tobytes())
            return struct.pack("<I", len(qc)) + qc + mc

        def dec(p):
            n = struct.unpack_from("<I", p, 0)[0]
            q = np.frombuffer(zstd.decode(p[4 : 4 + n]), dtype=np.int16).reshape(CHUNK, CHUNK)
            mask = np.unpackbits(np.frombuffer(zstd.decode(p[4 + n :]), dtype=np.uint8))[
                : CHUNK * CHUNK
            ].reshape(CHUNK, CHUNK)
            vals = q.astype(np.float32) * np.float32(scale)
            return np.where(mask.astype(bool), vals, np.float32(np.nan)), False

        out.append(_run(label, arr, enc, dec, arr))

    # --- V11: raw uint8 for categorical fields, vs production's float32 coercion
    if np.array_equal(np.unique(arr[np.isfinite(arr)]), np.array([0.0, 1.0], dtype=np.float32)):
        u8 = np.asarray(np.nan_to_num(arr, nan=0.0), dtype=np.uint8)
        out.append(
            _run(
                "V11 uint8 + Zstd5 (flag field)",
                arr,
                lambda c: zstd.encode(np.asarray(np.nan_to_num(c, nan=0.0), dtype=np.uint8).tobytes()),
                lambda p: (
                    np.frombuffer(zstd.decode(p), dtype=np.uint8).reshape(CHUNK, CHUNK).astype(np.float32),
                    False,
                ),
                arr,
            )
        )
    return out


def main():
    print(f"grid {LAT}x{LON}  chunks {LAT_CHUNKS}x{LON_CHUNKS} = {N_CHUNKS} per shard")
    print(f"raw float32 shard = {LAT * LON * 4 / 1e6:.2f} MB (padded: {N_CHUNKS * CHUNK * CHUNK * 4 / 1e6:.2f} MB)")
    print()
    for field_name, factory in FIELDS.items():
        arr = factory()
        finite = arr[np.isfinite(arr)]
        print("=" * 108)
        print(
            f"{field_name}: range [{finite.min():.3f}, {finite.max():.3f}]  "
            f"nan {100.0 * np.isnan(arr).mean():.1f}%"
        )
        print("=" * 108)
        base = None
        rows = variants(arr)
        for r in rows:
            if base is None:
                base = r.total_bytes
            ratio = base / r.total_bytes if r.total_bytes else 0.0
            warn = "  CLIPPED" if r.worst_value_clipped else ""
            print(
                f"  {r.name:<42} {r.total_bytes / 1e6:7.2f} MB  "
                f"vs-base {ratio:5.2f}x  enc {r.encode_ms:7.1f} ms  dec {r.decode_ms:7.1f} ms  "
                f"maxerr {r.max_abs_err:.6f}{warn}"
            )
        print()


if __name__ == "__main__":
    main()
