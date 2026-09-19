"""Experiment 3: does quantisation actually help on data that is already GRIB-packed?

cfgrib expands GRIB2's bit-packed integers into float32. The low mantissa bits are
therefore zeros, and Zstd already exploits those zero runs -- which is why the repo
measured 4.21x on real GFS data while raw synthetic float32 fields only reach ~1.15x.

This experiment emulates GRIB2 packing at N bits/view (N = 10..16, the range NOAA
actually uses) *before* encoding, so the baseline ratio matches reality and the
marginal gain from integer scaling can be read off honestly.

Run:  .venv/Scripts/python.exe artifacts/dtype_bench/bench_grib_realism.py
"""

from __future__ import annotations

import numpy as np

from bench_local_packing import (
    CHUNK,
    LAT,
    LON,
    ZSTD,
    dec_float32,
    enc_float32,
    global_int,
    local_pack,
    measure,
    spectral_field,
)

RNG = np.random.default_rng(99)


def grib_pack(field: np.ndarray, nbits: int, nan_frac: float = 0.0) -> np.ndarray:
    """Emulate GRIB2 simple packing: quantise to ``nbits`` over the field's range.

    This is what cfgrib hands the ingestion pipeline (as float32 with zeroed
    low mantissa bits).
    """
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


def make_temperature(nan_frac=0.0, seed=11):
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    base = 30.0 - 55.0 * np.abs(np.sin(np.deg2rad(lat)))
    field = base + 9.0 * spectral_field((LAT, LON), slope=2.2, seed=seed)
    field = np.asarray(field, dtype=np.float32)
    if nan_frac:
        field[RNG.random(field.shape) < nan_frac] = np.nan
    return field


VARIANTS = [
    ("[CURRENT] float32 + Zstd5", enc_float32, dec_float32),
    ("int16 global scale=0.01", *global_int(0.01, np.int16, np.int16(-32768))),
    ("int32 global scale=0.001", *global_int(0.001, np.int32, np.int32(-2147483648))),
    ("LOCAL pack 16-bit + bitmap", *local_pack(use_bitmap=True)),
]


def report(title, arr):
    raw = arr.size * 4
    finite = arr[np.isfinite(arr)]
    err_pack = abs(finite).max()
    print("=" * 116)
    print(f"{title}")
    print("=" * 116)
    base = None
    for label, enc, dec in VARIANTS:
        total, ems, dms, err = measure(arr, enc, dec, label)
        if base is None:
            base = total
        print(
            f"  {label:<32} {total / 1e6:7.3f} MB   raw/zstd {raw / total:6.2f}x   "
            f"vs float32 {base / total:5.2f}x   enc {ems:6.1f} ms  dec {dms:6.1f} ms  maxerr {err:.7f}"
        )
    print()


def main():
    print()
    print("### Temperature after GRIB packing at N bits/view (no NaN)")
    print()
    for nbits in (8, 10, 12, 14, 16, 24):
        arr = grib_pack(make_temperature(), nbits)
        report(f"temperature_2m  GRIB {nbits} bits/view", arr)

    print("\n### Temperature, GRIB 14 bits, with 6% NaN\n")
    report("temperature_2m  GRIB 14 bits  nan=6%", grib_pack(make_temperature(), 14, nan_frac=0.06))


if __name__ == "__main__":
    main()
