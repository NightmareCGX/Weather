"""Experiment 6: a relaxed 0.005 error budget, and quantisation x framing interaction.

Two questions from the review:

  Q1  The 0.001 budget blocks int16 (it needs int32, i.e. no width saving). If 0.005
      is acceptable instead, int16 becomes usable across 12 of 15 variables. How much
      does the answer actually change? And how sensitive is it to the exact scale?

  Q2  Framing (grouping several chunks per Zstd stream) won 1.18-1.33x losslessly.
      Does quantisation *stack* with framing, or do they exploit the same redundancy?

Plus the tile-path cost question: a grouped read must decode a whole group and slice
the chunks out. How does that compare with decoding the same chunks one by one?

Run:  .venv/Scripts/python.exe benches/bench_budget_and_framing.py
"""

from __future__ import annotations

import time

import numpy as np
from numcodecs import Zstd

from bench_local_packing import CHUNK, LAT, LAT_CHUNKS, LON, LON_CHUNKS, spectral_field
from bench_shard_and_ensemble import chunk_buffers, grib_pack, make_temperature

Z5 = Zstd(level=5)
Z19 = Zstd(level=19)
RNG = np.random.default_rng(4242)

I16_NAN = np.int16(-32768)
I32_NAN = np.int32(-2147483648)


# --------------------------------------------------------------------------------------
# field synthesis
# --------------------------------------------------------------------------------------
def make_gust(nan_frac=0.0, seed=21):
    """Wind gust in km/h: the widest-range variable (m/s x 3.6)."""
    f = np.abs(spectral_field((LAT, LON), slope=2.0, seed=seed)) * 120.0 + 5.0
    a = np.asarray(f, dtype=np.float32)
    if nan_frac:
        a[RNG.random(a.shape) < nan_frac] = np.nan
    return a


def make_prate(nan_frac=0.0, seed=31):
    """Precipitation rate in mm/h: kg m-2 s-1 x 3600, heavy right tail."""
    f = np.abs(spectral_field((LAT, LON), slope=2.6, seed=seed)) ** 2 * 40.0
    a = np.asarray(f, dtype=np.float32)
    if nan_frac:
        a[RNG.random(a.shape) < nan_frac] = np.nan
    return a


def make_rr2(nan_frac=0.0, seed=51):
    """Relative humidity in %: mid-range variable, still exceeds int16 at 0.001."""
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    f = 60.0 + 25.0 * np.cos(np.deg2rad(lat)) + 15.0 * spectral_field((LAT, LON), slope=2.2, seed=seed)
    a = np.asarray(np.clip(f, 0.0, 100.0), dtype=np.float32)
    if nan_frac:
        a[RNG.random(a.shape) < nan_frac] = np.nan
    return a


# --------------------------------------------------------------------------------------
# encoders (quantise -> optional bit packing -> group -> zstd)
# --------------------------------------------------------------------------------------
def quantise(arr, np_dtype, sentinel, scale):
    limit = np.iinfo(np_dtype).max
    q = np.clip(np.round(np.nan_to_num(arr, nan=0.0) / scale), -limit, limit).astype(np_dtype)
    return np.where(np.isnan(arr), sentinel, q).astype(np_dtype)


def pack_local_12bit(buf):
    """Per-chunk min/max into a 12-bit container, bit-packed 2 values per 3 bytes."""
    finite = np.isfinite(buf)
    if not finite.any():
        # Fixed-size so chunks later in the same group stay byte-aligned.
        return bytes(CHUNK * CHUNK // 2 * 3), None, None
    lo = float(np.nanmin(buf))
    span = float(np.nanmax(buf)) - lo
    scale = (span / 4095.0) if span > 0 else 0.0
    flat = np.nan_to_num(buf, nan=lo).ravel()
    q = np.clip(np.round((flat - lo) / scale), 0, 4095).astype(np.uint16) if scale else np.zeros(flat.size, np.uint16)
    if q.size % 2:
        q = np.concatenate([q, np.zeros(1, np.uint16)])
    a, b = q[0::2].astype(np.uint32), q[1::2].astype(np.uint32)
    out = np.empty((a.size, 3), dtype=np.uint8)
    out[:, 0] = a & 0xFF
    out[:, 1] = ((a >> 8) & 0x0F) | ((b & 0x0F) << 4)
    out[:, 2] = (b >> 4) & 0xFF
    return out.tobytes(), lo, scale


def unpack_local_12bit(blob, lo, scale):
    u = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    a = u[:, 0] | ((u[:, 1] & 0x0F) << 8)
    b = (u[:, 1] >> 4) | (u[:, 2] << 4)
    q = np.empty(a.size * 2, dtype=np.uint32)
    q[0::2] = a
    q[1::2] = b
    q = q[: CHUNK * CHUNK]
    return (lo + q.astype(np.float64) * scale).reshape(CHUNK, CHUNK)


def make_variant(kind, scale):
    """Return (chunk_encoder, group_decoder_factory)."""

    def enc_chunk(buf):
        if kind == "float32":
            return buf.tobytes(order="C")
        if kind == "i16":
            return quantise(buf, np.int16, I16_NAN, scale).tobytes(order="C")
        if kind == "i32":
            return quantise(buf, np.int32, I32_NAN, scale).tobytes(order="C")
        if kind == "local12":
            return pack_local_12bit(buf)
        raise ValueError(kind)

    return enc_chunk


def group_encode(bufs, group, kind, scale, z=Z5):
    """Encode chunk buffers into one Zstd stream per `group` chunks."""
    blobs = []
    meta = []
    for i in range(0, len(bufs), group):
        slab = bufs[i : i + group]
        if kind == "float32":
            payload = np.concatenate([b.ravel() for b in slab]).astype(np.float32).tobytes(order="C")
            blobs.append(z.encode(payload))
            meta.append(None)
        elif kind in ("i16", "i32"):
            payload = np.concatenate([quantise(b, np.int16 if kind == "i16" else np.int32,
                                               I16_NAN if kind == "i16" else I32_NAN,
                                               scale).ravel() for b in slab]).tobytes(order="C")
            blobs.append(z.encode(payload))
            meta.append(None)
        elif kind == "local12":
            parts, hdrs = [], []
            for b in slab:
                p = pack_local_12bit(b)
                parts.append(p[0] if p[0] is not None else b"")
                hdrs.append((p[1], p[2]))
            blobs.append(z.encode(b"".join(parts)))
            meta.append(hdrs)
    return blobs, meta


def group_decode_metrics(blobs, meta, group, kind, scale, z=Z5):
    """Decode every group and slice out the chunks; return (ms, chunk_count, sizes)."""
    chunks = []
    t0 = time.perf_counter()
    for i, blob in enumerate(blobs):
        raw = z.decode(blob)
        if kind == "float32":
            arr = np.frombuffer(raw, dtype=np.float32)
            for k in range(len(arr) // (CHUNK * CHUNK)):
                chunks.append(arr[k * CHUNK * CHUNK : (k + 1) * CHUNK * CHUNK].reshape(CHUNK, CHUNK))
        elif kind in ("i16", "i32"):
            dt = np.int16 if kind == "i16" else np.int32
            sentinel = I16_NAN if kind == "i16" else I32_NAN
            arr = np.frombuffer(raw, dtype=dt)
            ngroups_chunks = len(arr) // (CHUNK * CHUNK)
            for k in range(ngroups_chunks):
                q = arr[k * CHUNK * CHUNK : (k + 1) * CHUNK * CHUNK].reshape(CHUNK, CHUNK)
                vals = q.astype(np.float32) * np.float32(scale)
                chunks.append(np.where(q == sentinel, np.float32(np.nan), vals))
        elif kind == "local12":
            blen = CHUNK * CHUNK // 2 * 3
            for k, (lo, sc) in enumerate(meta[i]):
                piece = raw[k * blen : (k + 1) * blen]
                chunks.append(
                    unpack_local_12bit(piece, lo, sc) if lo is not None else np.full((CHUNK, CHUNK), np.nan)
                )
    t1 = time.perf_counter()
    return 1000 * (t1 - t0), chunks


def error_of(chunks, arr, group):
    out = np.full((LAT, LON), np.nan, dtype=np.float64)
    idx = 0
    for r in range(LAT_CHUNKS):
        for c in range(LON_CHUNKS):
            r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
            c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
            out[r0:r1, c0:c1] = chunks[idx][: r1 - r0, : c1 - c0]
            idx += 1
    finite = np.isfinite(arr)
    e = float(np.abs(out[finite] - arr[finite]).max()) if finite.any() else 0.0
    if (np.isnan(out) != np.isnan(arr)).any():
        e = float("inf")
    return e


# --------------------------------------------------------------------------------------
# Q1: scale sensitivity under a 0.005 budget
# --------------------------------------------------------------------------------------
def q1():
    print("=" * 112)
    print("Q1. int16 scale sensitivity, temperature at upstream GRIB 12 bits/view")
    print("    (0.005 budget => max |error| = scale/2, so scale 0.01 is the coarsest allowed)")
    print("=" * 112)
    arr = grib_pack(make_temperature(), 12)
    bufs = chunk_buffers(arr)
    raw = sum(b.size * 4 for b in bufs)

    base, _ = group_encode(bufs, 1, "float32", 0.0)
    base_bytes = sum(len(b) for b in base)
    print(f"{'variant':<44} {'MB':>8} {'vs current':>11} {'maxerr':>10} {'clip?':>7}")
    blobs, meta = group_encode(bufs, 1, "float32", 0.0)
    print(f"{'float32 + Zstd5 [CURRENT]':<44} {base_bytes / 1e6:8.3f} {1.0:10.2f}x {0.0:10.7f} {'-':>7}")
    for scale in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        blobs, meta = group_encode(bufs, 1, "i16", scale)
        tot = sum(len(b) for b in blobs)
        _, chunks = group_decode_metrics(blobs, meta, 1, "i16", scale)
        e = error_of(chunks, arr, 1)
        clipped = "YES" if e > 5 * scale else "no"
        print(f"{'int16 global scale=' + str(scale):<44} {tot / 1e6:8.3f} {base_bytes / tot:10.2f}x {e:10.7f} {clipped:>7}")
    print()

    print("    Same sweep for the wide-range variables (km/h and mm/h exceed int16 at 0.001):")
    for name, field in (("wind_gust_kmh", make_gust()), ("precip_rate_mmh", make_prate()), ("relative_humidity_%", make_rr2())):
        a = grib_pack(field, 12)
        bf = chunk_buffers(a)
        bb = sum(len(b) for b in group_encode(bf, 1, "float32", 0.0)[0])
        line = [f"      {name:<22} current {bb / 1e6:7.3f} MB  "]
        for scale in (0.005, 0.01, 0.02):
            bl, mt = group_encode(bf, 1, "i16", scale)
            tot = sum(len(b) for b in bl)
            _, ch = group_decode_metrics(bl, mt, 1, "i16", scale)
            e = error_of(ch, a, 1)
            tag = "CLIP" if e > 5 * scale else "ok"
            line.append(f"| s={scale}: {bb / tot:4.2f}x err {e:.4f} {tag} ")
        print("".join(line))
    print()


# --------------------------------------------------------------------------------------
# Q2: does quantisation stack with framing?
# --------------------------------------------------------------------------------------
def q2():
    print("=" * 112)
    print("Q2. Quantisation x framing at upstream GRIB 12 bits/view (temperature)")
    print("=" * 112)
    arr = grib_pack(make_temperature(), 12)
    bufs = chunk_buffers(arr)

    print(f"{'framing':<26} {'float32':>18} {'int16@0.01':>18} {'local 12-bit':>18} {'int32@0.001':>18}")
    print("-" * 112)
    ref = None
    for group in (1, 15, 24, 120):
        cells = []
        for kind, scale in (("float32", 0.0), ("i16", 0.01), ("local12", 0.0), ("i32", 0.001)):
            bl, mt = group_encode(bufs, group, kind, scale)
            cells.append(sum(len(b) for b in bl))
        if ref is None:
            ref = cells[0]
        label = f"group={group}" + (" (current)" if group == 1 else "")
        print(
            f"{label:<26} "
            + " ".join(f"{c / 1e6:9.3f} {ref / c:5.2f}x" for c in cells)
        )
    print()
    print("    Cells are 'MB  gain-vs-current'. The gain columns show whether the two")
    print("    effects stack: compare int16@0.01's gain at group=1 against its gain at group=120.")
    print()

    # error check for the 0.005 budget under local 12-bit
    bl, mt = group_encode(bufs, 15, "local12", 0.0)
    _, ch = group_decode_metrics(bl, mt, 15, "local12", 0.0)
    print(f"    local 12-bit max |error| = {error_of(ch, arr, 15):.7f}  (budget 0.005)")
    bl, mt = group_encode(bufs, 15, "i16", 0.01)
    _, ch = group_decode_metrics(bl, mt, 15, "i16", 0.01)
    print(f"    int16@0.01  max |error| = {error_of(ch, arr, 15):.7f}")
    print()
    print("=" * 112)
    print("    Zstd19 on top of the best framing:")
    print("=" * 112)
    for kind, scale in (("float32", 0.0), ("i16", 0.01), ("local12", 0.0)):
        for group in (15, 120):
            bl, _ = group_encode(bufs, group, kind, scale, z=Z19)
            tot = sum(len(b) for b in bl)
            print(f"      {kind:<12} group={group:<4} Zstd19  {tot / 1e6:7.3f} MB  {ref / tot:5.2f}x")
    print()


# --------------------------------------------------------------------------------------
# Tile-path cost: grouped decode + slice vs per-chunk decode
# --------------------------------------------------------------------------------------
def tile_path():
    print("=" * 112)
    print("Tile path cost: decoding a 12-chunk window (the z=2 case) grouped vs per-chunk")
    print("=" * 112)
    arr = grib_pack(make_temperature(), 12)
    bufs = chunk_buffers(arr)

    # A z=2 tile window spans a 3x4 neighbourhood of chunks (12 chunks, per config.py).
    window = [r * LON_CHUNKS + c for r in (2, 3, 4) for c in (4, 5, 6, 7)]

    print(f"{'framing':<22} {'fetch calls':>12} {'KB fetched':>11} {'decode ms':>10} {'note':<28}")
    print("-" * 112)

    for group in (1, 15, 24, 120):
        bl, mt = group_encode(bufs, group, "float32", 0.0)
        touched = sorted({i // group for i in window})
        fetched = sum(len(bl[g]) for g in touched)
        # decode only the touched groups, then slice the 12 chunks out
        t0 = time.perf_counter()
        got = 0
        for g in touched:
            raw = Z5.decode(bl[g])
            for k in range(group):
                if g * group + k in window:
                    _ = np.frombuffer(raw, dtype=np.float32)[k * CHUNK * CHUNK : (k + 1) * CHUNK * CHUNK]
                    got += 1
        t1 = time.perf_counter()
        note = f"{got} chunks sliced"
        if group == 1:
            note = "current: one stream per chunk"
        elif group == 120:
            note = "whole shard"
        print(
            f"group={group:<16} {len(touched):>12} {fetched / 1e3:11.1f} {1000 * (t1 - t0):10.3f} {note:<28}"
        )

    print()
    print("    'fetch calls' is what matters on the current 4-core host: API_CHUNK_FETCH_WORKERS=2,")
    print("    so 12 per-chunk fetches serialise into 6 rounds of RTT while 1 group fetch is 1 round.")
    print()


if __name__ == "__main__":
    q1()
    q2()
    tile_path()
