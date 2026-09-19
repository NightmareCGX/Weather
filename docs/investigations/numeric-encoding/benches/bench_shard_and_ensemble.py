"""Experiment 4: the two levers that are not numeric-type changes.

The dtype experiments (bench_grib_realism) showed integer scaling buys only ~1.1-1.2x
on data that already compresses ~4x, because Zstd is already harvesting the redundancy
that quantisation would remove. This experiment measures the two structural levers that
remain:

  A. Compress the whole 900 KB shard as ONE Zstd stream instead of 120 separate 40 KB
     chunks. Adjacent 100x100 tiles of the same field are spatially adjacent, so the
     cross-chunk redundancy is large and per-chunk framing throws it away.
  B. For GEFS (30 members), store members as residuals from the ensemble mean. GEFS is
     ~96% of the platform's bytes, and members differ only by small perturbations.

Run:  .venv/Scripts/python.exe artifacts/dtype_bench/bench_shard_and_ensemble.py
"""

from __future__ import annotations

import time

import numpy as np
from numcodecs import Zstd

from bench_local_packing import (
    CHUNK,
    LAT,
    LAT_CHUNKS,
    LON,
    LON_CHUNKS,
    dec_float32,
    enc_float32,
    spectral_field,
)

RNG = np.random.default_rng(2026)
Z5 = Zstd(level=5)
Z19 = Zstd(level=19)
Z22 = Zstd(level=22)


def grib_pack(field, nbits, nan_frac=0.0):
    finite = np.isfinite(field)
    lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
    levels = float((1 << nbits) - 1)
    scale = (hi - lo) / levels if hi > lo else 0.0
    vals = np.nan_to_num(field, nan=lo)
    q = np.round((vals - lo) / scale) if scale else np.zeros_like(vals)
    out = np.asarray(lo + q * scale, dtype=np.float32)
    if nan_frac:
        out[RNG.random(out.shape) < nan_frac] = np.nan
    return out


def make_temperature(seed=11, nan_frac=0.0):
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    base = 30.0 - 55.0 * np.abs(np.sin(np.deg2rad(lat)))
    field = np.asarray(base + 9.0 * spectral_field((LAT, LON), slope=2.2, seed=seed), dtype=np.float32)
    if nan_frac:
        field[RNG.random(field.shape) < nan_frac] = np.nan
    return field


def chunk_buffers(arr):
    """The exact 100x100 NaN-padded buffers production compresses."""
    bufs = []
    for r in range(LAT_CHUNKS):
        for c in range(LON_CHUNKS):
            buf = np.full((CHUNK, CHUNK), np.nan, dtype=np.float32)
            r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
            c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
            buf[: r1 - r0, : c1 - c0] = arr[r0:r1, c0:c1]
            bufs.append(buf)
    return bufs


# --------------------------------------------------------------------------------------
# A. per-chunk framing vs whole-shard framing
# --------------------------------------------------------------------------------------
def lever_a(arr, nbits):
    bufs = chunk_buffers(arr)
    raw = sum(b.size * 4 for b in bufs)

    print(f"  --- upstream GRIB {nbits} bits/view ---")

    # current: 120 independent Zstd-5 streams
    t0 = time.perf_counter()
    per_chunk = [Z5.encode(b.tobytes(order="C")) for b in bufs]
    t1 = time.perf_counter()
    tot_chunk = sum(len(p) for p in per_chunk)
    print(
        f"    per-chunk Zstd5 (CURRENT)      {tot_chunk / 1e6:6.3f} MB  {raw / tot_chunk:5.2f}x  "
        f"enc {1000 * (t1 - t0):6.1f} ms"
    )

    # whole shard, one stream
    whole = np.concatenate([b.ravel() for b in bufs]).astype(np.float32)
    for label, z in (("Zstd5", Z5), ("Zstd19", Z19), ("Zstd22", Z22)):
        t0 = time.perf_counter()
        blob = z.encode(whole.tobytes(order="C"))
        t1 = time.perf_counter()
        # full-shard decode cost (what a whole-shard reader would pay per access)
        t2 = time.perf_counter()
        z.decode(blob)
        t3 = time.perf_counter()
        print(
            f"    whole-shard {label:<9}          {len(blob) / 1e6:6.3f} MB  {raw / len(blob):5.2f}x  "
            f"enc {1000 * (t1 - t0):6.1f} ms  shard-decode {1000 * (t3 - t2):5.2f} ms"
        )

    # single-chunk access cost under each framing
    t0 = time.perf_counter()
    Z5.decode(per_chunk[60])
    t1 = time.perf_counter()
    print(
        f"    single-chunk access: per-chunk {1000 * (t1 - t0):5.2f} ms"
        f"   |  whole-shard {1000 * (t3 - t2):5.2f} ms (must decode whole shard)"
    )
    print()


# --------------------------------------------------------------------------------------
# B. GEFS members: independent vs residual-from-mean
# --------------------------------------------------------------------------------------
def lever_b(nbits, n_members=30):
    print(f"  --- {n_members}-member ensemble, upstream GRIB {nbits} bits/view ---")
    mean_field = make_temperature(seed=101)

    # Perturbations: smooth, spatially correlated, with a realistic spread.
    members = []
    for m in range(n_members):
        spread = spectral_field((LAT, LON), slope=2.0, seed=500 + m)
        spread = spread / (spread.std() + 1e-12) * 1.6
        members.append(grib_pack(np.asarray(mean_field + spread, dtype=np.float32), nbits))

    base_arr = grib_pack(mean_field, nbits)

    def shard_bytes(arr):
        return sum(len(Z5.encode(b.tobytes(order="C"))) for b in chunk_buffers(arr))

    t0 = time.perf_counter()
    indep = sum(shard_bytes(a) for a in members)
    t1 = time.perf_counter()
    print(f"    A) {n_members} members stored independently   {indep / 1e6:7.2f} MB   enc {1000 * (t1 - t0):6.0f} ms")

    # Residual design: one mean shard + N residual shards. The platform also stores the
    # upstream geavg product already, so the mean is not necessarily extra bytes.
    mean_bytes = shard_bytes(base_arr)
    t0 = time.perf_counter()
    resid_bytes = 0
    max_spread = 0.0
    resid_arrays = []
    for a in members:
        r = np.asarray(a - base_arr, dtype=np.float32)
        max_spread = max(max_spread, float(np.nanmax(np.abs(r))))
        resid_arrays.append(r)
        resid_bytes += shard_bytes(r)
    t1 = time.perf_counter()
    total_resid = mean_bytes + resid_bytes
    print(
        f"    B) mean + {n_members} residual shards          {total_resid / 1e6:7.2f} MB   "
        f"enc {1000 * (t1 - t0):6.0f} ms"
    )
    print(
        f"       -> {indep / total_resid:5.2f}x smaller; residual range +/-{max_spread:.2f} degC "
        f"(int16@0.01 covers +/-327 -> OK)"
    )

    # C) residual quantised to int16 (residuals have a small dynamic range, so 16 bits
    #    is now plenty even at fine resolution)
    Z = Zstd(level=5)

    def shard_bytes_i16(arr, scale):
        tot = 0
        for b in chunk_buffers(arr):
            q = np.clip(np.round(np.nan_to_num(b, nan=0.0) / scale), -32767, 32767).astype(np.int16)
            tot += len(Z.encode(q.tobytes(order="C")))
        return tot

    t0 = time.perf_counter()
    q_bytes = shard_bytes_i16(base_arr, 0.01) + sum(shard_bytes_i16(r, 0.001) for r in resid_arrays)
    t1 = time.perf_counter()
    print(
        f"    C) mean int16@0.01 + residuals int16@0.001 {q_bytes / 1e6:7.2f} MB   "
        f"enc {1000 * (t1 - t0):6.0f} ms   -> {indep / q_bytes:5.2f}x smaller than A"
    )
    print()


def main():
    print()
    print("### Lever A: per-chunk framing vs whole-shard framing (temperature, no NaN)")
    print()
    for nbits in (12, 14, 16):
        lever_a(grib_pack(make_temperature(), nbits), nbits)

    print("### Lever B: GEFS ensemble members as residuals")
    print()
    for nbits in (12, 14, 16):
        lever_b(nbits)


if __name__ == "__main__":
    main()
