"""Experiment 29 (REAL DATA): resources of the LOCKED design.

The locked design is:
    A  near-Gaussian (temperature, wind u/v)   MEAN + STD + 32 normalised bins
    B  zero-inflated  (precip, snow_depth, gust)  absolute quantile function, 19 body-dense
    C  bounded/censored (cover, ceiling, vis, RH) same as B (measured in section 16)
    D  flags (4)                                  per-cell fractions
    no neighbour covariances
    plus Layer 1: compression framing + Zstd level 7 (lossless)

This measures, per variable, the aggregate-step CPU and peak memory for the two final
encodings, on real 30-member data, so the per-cycle figures are measured rather than
extrapolated. Decode concurrency and the encode cost are reported alongside.

Run:  .venv/Scripts/python.exe bench_locked_design.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import (  # noqa: E402
    CACHE, K_RANGE, compress_field, decode, quantise,
)
from bench_tail_paths import quantiles_fast  # noqa: E402
from gefs_fetch_c import decode as decode_c  # noqa: E402

PROC = psutil.Process()
LEADS, VARS = 81, 14
UNITS = LEADS * VARS
NB = 32
QF19 = (0.005, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999)


def rss():
    gc.collect()
    return PROC.memory_info().rss


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = 1000 * (time.perf_counter() - t0)
    print(f"      {label:<38} {dt:8.1f} ms")
    return out, dt


def main():
    cases = (
        ("2t", 2, 273.15, "A: MEAN+STD+32 bins", "A"),
        ("tp", 0, 0.0, "B: qfunc 19 body-dense", "B"),
        ("tcc", 0, 0.0, "C: qfunc 19 body-dense", "C"),
    )
    print()
    print("=" * 104)
    print("LOCKED DESIGN -- aggregate-step CPU and memory, one variable, 30 real members")
    print("=" * 104)

    summary = []
    for short, level, offset, label, klass in cases:
        paths = [
            os.path.join(CACHE, f"2026091800_gep{m:02d}_f006_{short}.grib2")
            for m in range(1, 31)
        ]
        if not all(os.path.exists(p) for p in paths):
            print(f"  {label}: MISSING data")
            continue
        base = rss()
        c_class = short in ("vis", "r2", "tcc", "gh")

        def load_field(p):
            if c_class:
                return decode_c(p, short)[3].astype(np.float32) - np.float32(offset)
            return decode(p, short, level) - np.float32(offset)

        stack = np.stack([load_field(p) for p in paths])
        after_load = rss()
        print()
        print(f"  {label}   ({short})   stack {stack.nbytes / 1e6:.0f} MB   "
              f"RSS after load {after_load / 1e6:.0f} MB (+{(after_load - base) / 1e6:.0f})")

        peak = [after_load]

        def tick():
            peak[0] = max(peak[0], PROC.memory_info().rss)

        if klass == "A":
            mean_f, t_mean = timed("MEAN (nanmean over members)",
                                   lambda: np.nanmean(stack, axis=0))
            std_f, t_std = timed("STD (nanstd over members)",
                                 lambda: np.nanstd(stack, axis=0))
            def bins_fn():
                z = (stack - mean_f[None]) / np.maximum(std_f[None], 1e-9)
                edges = np.linspace(-K_RANGE, K_RANGE, NB + 1)
                return [np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0)
                        for b in range(NB)]
            bins, t_bins = timed(f"{NB} normalised bins", bins_fn)
            tick()
            fields = [mean_f, std_f] + bins
            nfields = len(fields)
            t_agg = t_mean + t_std + t_bins
        else:
            qf, t_qf = timed(f"quantiles_fast ({len(QF19)} levels)", lambda: quantiles_fast(stack, QF19))
            tick()
            fields = qf
            nfields = len(fields)
            t_agg = t_qf

        nbytes, t_enc = timed(f"encode {nfields} fields (int16 + Zstd5)",
                              lambda: sum(compress_field(quantise(f), np.int16) for f in fields))
        base_mb = sum(compress_field(stack[m]) for m in range(30)) / 1e6
        print(f"      -> {nfields} fields, {nbytes / 1e6:.3f} MB vs baseline {base_mb:.3f} MB "
              f"= {base_mb / (nbytes / 1e6):.2f}x   aggregate {t_agg:.0f} ms, encode {t_enc:.0f} ms")
        summary.append((label, klass, nfields, nbytes / 1e6, base_mb, base_mb / (nbytes / 1e6),
                        t_agg, t_enc, peak[0] - base))
        del stack, fields
        gc.collect()

    print()
    print("=" * 104)
    print("Per-cycle projection (14 variables x 81 leads)")
    print("=" * 104)
    print(f"{'class':<8} {'fields':>7} {'ratio':>8} {'agg ms/var':>12} {'enc ms/var':>12}")
    print("-" * 104)
    for label, klass, nf, mb, base_mb, ratio, t_agg, t_enc, mem in summary:
        print(f"{klass:<8} {nf:>7} {ratio:7.2f}x {t_agg:12.0f} {t_enc:12.0f}")
    # GEFS mix: A=3 (temp, u, v), B=3 (precip_amt, gust, snow_depth), C=4, D=4 flags
    a = next(r for r in summary if r[1] == "A")
    b = next(r for r in summary if r[1] == "B")
    c = next(r for r in summary if r[1] == "C")
    agg_per_cycle = (3 * a[6] + 3 * b[6] + 4 * c[6]) * LEADS / 1000 / 3600
    enc_per_cycle = (3 * a[7] + 3 * b[7] + 4 * c[7]) * LEADS / 1000 / 3600
    print()
    print(f"  GEFS mix assumed: A=3 vars, B=3, C=4, D=4 flags(negligible)")
    print(f"  aggregate CPU/cycle: {agg_per_cycle:.2f} h single-core  "
          f"({agg_per_cycle / 2:.2f} h on 2 cores)")
    print(f"  encode CPU/cycle   : {enc_per_cycle:.2f} h single-core  "
          f"({enc_per_cycle / 2:.2f} h on 2 cores)")
    print(f"  baseline encode for comparison: ~0.9-1.1 s/var/lead -> "
          f"{14 * LEADS * 1.0 / 3600:.2f} h single-core")
    print()
    for label, klass, nf, mb, base_mb, ratio, t_agg, t_enc, mem in summary:
        print(f"  peak RSS above start, {klass} ({label.split(':')[0]}): {mem / 1e6:+.0f} MB")
    print()
    print("  NOTE: the aggregate step needs all 30 members resident (the stack), so the")
    print("  pipeline must collect them per variable. Peak RSS above counts the stack plus")
    print("  the working copies; it is NOT the whole-pipeline footprint.")
    print()


if __name__ == "__main__":
    main()
