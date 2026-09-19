"""Experiment 23 (REAL DATA): long lead times, and the CPU/memory breakdown under
production-like concurrency.

Two things the review asked for that section 9.14 could not answer:

  (A) LONG LEADS. Section 9.14 measured f006 only. Ensemble spread grows with lead time,
      so the non-Gaussianity -- and therefore the reconstruction error and the
      +-4-sigma normalisation failure -- can only get worse. This re-runs the diagnostics
      at f240 in the SAME cycle (20260918 00Z has the full 81 leads; 18Z was only
      published to f213), and cross-checks f006 00Z against f006 18Z as a reproducibility
      control.

  (B) CONCURRENCY. Section 9.14's CPU and memory numbers are single-process, single
      variable. GEFS ingests members at concurrency 4, so this measures decode throughput
      and peak RSS at 1/2/4 workers, and breaks the aggregate step down by component so
      the cost is attributable. This is a single-node simulation of the aggregate stage,
      NOT the real pipeline.

Run:  .venv/Scripts/python.exe bench_longlead_concurrency.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import (  # noqa: E402
    CACHE, N_BINS, K_RANGE, bundle_fields, compress_field, decode, quantise,
)

PROC = psutil.Process()
LEADS_PER_CYCLE, GEFS_VARS = 81, 14
UNITS = LEADS_PER_CYCLE * GEFS_VARS


def paths(cycle_date, cycle, lead, short):
    return [
        os.path.join(CACHE, f"{cycle_date}{cycle}_gep{m:02d}_{lead}_{short}.grib2")
        for m in range(1, 31)
    ]


def rss():
    gc.collect()
    return PROC.memory_info().rss


def load(paths_, short, level, offset=0.0):
    return np.stack([decode(p, short, level) - np.float32(offset) for p in paths_])


def tail_diagnostics(stack, label):
    """How heavy is the tail, and does the +-4sigma shape recover the thresholds?"""
    mean_f, std_f, bins, _, _ = bundle_fields(stack, False)
    sd_mean = float(np.nanmean(std_f))
    mu_mean = float(np.nanmean(mean_f))
    hi = float(np.nanmax(stack))
    zero = float((stack == 0).mean())
    print(f"    {label}")
    print(f"      mean {mu_mean:8.3f}   sigma(mean) {sd_mean:8.3f}   max {hi:9.3f}   "
          f"max/sigma {hi / max(sd_mean, 1e-9):8.1f}   zero-fraction {zero:.3f}")

    n_pts = 800
    rng = np.random.default_rng(5)
    rows = rng.uniform(1, 718, n_pts)
    cols = rng.uniform(2, 1437, n_pts)
    r0, c0 = rows.astype(int), cols.astype(int)
    tr, tc = rows - r0, cols - c0

    def bil(f):
        g00, g01 = f[r0, c0], f[r0, c0 + 1]
        g10, g11 = f[r0 + 1, c0], f[r0 + 1, c0 + 1]
        lo = g00 + (g01 - g00) * tc
        up = g10 + (g11 - g10) * tc
        return lo + (up - lo) * tr

    interp = np.stack([bil(stack[m]) for m in range(30)])
    mix = np.stack([bil(b) for b in bins])
    cum = np.cumsum(mix, axis=0)
    edges = np.linspace(-K_RANGE, K_RANGE, N_BINS + 1)
    width = edges[1] - edges[0]
    ar = np.arange(n_pts)
    mu_a, sd_a = bil(mean_f), bil(std_f)
    half = 15

    # thresholds expressed as a percentile of the local distribution, so they are
    # comparable across leads: exceedance of the 50th / 90th / 99th percentile
    out = []
    for q in (50.0, 90.0, 99.0):
        t = np.nanpercentile(interp, q, axis=0)  # per-point threshold
        truth = np.mean(interp > t[None, :], axis=0)
        nz = float(
            np.max(
                np.abs(
                    np.mean(interp[:half] > t[None, :], axis=0)
                    - np.mean(interp[half:] > t[None, :], axis=0)
                )
            )
        )
        zt = (t - mu_a) / np.maximum(sd_a, 1e-12)
        pos = (zt - edges[0]) / width
        idx = np.clip(np.floor(pos).astype(int), 0, N_BINS - 1)
        frac = np.clip(pos - idx, 0.0, 1.0)
        lo_cdf = np.where(idx > 0, cum[np.maximum(idx - 1, 0), ar], 0.0)
        cdf = np.clip(lo_cdf + frac * mix[idx, ar], 0.0, 1.0)
        e = float(np.max(np.abs((1.0 - cdf) - truth)))
        out.append(e / nz if nz else 0.0)
    print(f"      exceedance of the local P50/P90/P99, err/noise = "
          + " / ".join(f"{v:.2f}" for v in out)
          + f"   worst {max(out):.2f} {'OK' if max(out) < 1 else 'FAILS'}")
    return max(out)


def main():
    print()
    print("=" * 108)
    print("(A) LONG LEADS -- same cycle 20260918 00Z, f006 vs f240")
    print("=" * 108)

    for short, level, name in (("tp", 0, "APCP precip"), ("2t", 2, "2t temperature")):
        print()
        print(f"  {name}")
        for lead in ("f006", "f240"):
            p = paths("20260918", "00", lead, short)
            if not all(os.path.exists(x) for x in p):
                print(f"    {lead}: MISSING (run gefs_fetch.py 20260918 00 {lead})")
                continue
            st = load(p, short, level, offset=273.15 if short == "2t" else 0.0)
            ratio = tail_diagnostics(st, f"{lead}  ({st.nbytes / 1e6:.0f} MB stack)")
            if lead == "f006":
                base = sum(compress_field(st[m]) for m in range(30)) / 1e6
                mf, sf, bn, cv, _ = bundle_fields(st, True)
                bun = sum(compress_field(quantise(f), np.int16)
                          for f in [mf, sf] + bn) / 1e6
                print(f"      baseline 30 members {base:.3f} MB -> bundle {bun:.3f} MB "
                      f"({base / bun:.2f}x)")
            del st
            gc.collect()

    # reproducibility control: f006 18Z vs f006 00Z
    print()
    print("  reproducibility control, f006 from two independent cycles:")
    for cyc in ("18", "00"):
        p = paths("20260918", cyc, "f006", "tp")
        if all(os.path.exists(x) for x in p):
            st = load(p, "tp", 0)
            base = sum(compress_field(st[m]) for m in range(30)) / 1e6
            print(f"    {cyc}Z f006 APCP baseline = {base:.3f} MB per (variable, lead), "
                  f"per shard {base / 30 * 1e3:.1f} KB")
            del st
            gc.collect()

    # -------------------------------------------------------------------------------
    print()
    print("=" * 108)
    print("(B) DECODE CONCURRENCY and the aggregate step's CPU breakdown")
    print("=" * 108)
    p = paths("20260918", "00", "f006", "2t")
    if not all(os.path.exists(x) for x in p):
        print("  MISSING data")
        return

    print()
    print(f"{'workers':>8} {'decode wall':>13} {'MB/s':>8} {'peak RSS':>11} {'over start':>12}")
    print("-" * 108)
    base_rss = rss()
    for workers in (1, 2, 4):
        gc.collect()
        before = rss()
        t0 = time.perf_counter()
        if workers == 1:
            fields = [decode(x, "2t", 2) for x in p]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fields = list(ex.map(lambda x: decode(x, "2t", 2), p))
        t1 = time.perf_counter()
        st = np.stack(fields)
        peak = max(rss(), before)
        print(f"{workers:>8} {t1 - t0:12.2f}s {30 * st[0].nbytes / 1e6 / (t1 - t0):8.1f} "
              f"{peak / 1e6:10.0f}MB {(peak - base_rss) / 1e6:11.0f}MB")
        del fields, st
        gc.collect()

    # aggregate step breakdown
    st = np.stack([decode(x, "2t", 2) for x in p])
    print()
    print("  aggregate step breakdown (one variable, 30 members, 125 MB stack)")
    print(f"    {'component':<34} {'wall ms':>9} {'MB/field':>10}")
    print("-" * 108)
    timings = []

    def timed(label, fn):
        t0 = time.perf_counter()
        r = fn()
        dt = 1000 * (time.perf_counter() - t0)
        timings.append((label, dt))
        print(f"    {label:<34} {dt:9.1f}")
        return r

    mean_f = timed("MEAN  (nanmean over members)", lambda: np.nanmean(st, axis=0))
    std_f = timed("STD   (nanstd over members)", lambda: np.nanstd(st, axis=0))
    z = timed("normalise  (z = (x-mean)/std)",
              lambda: (st - mean_f[None]) / np.maximum(std_f[None], 1e-9))
    edges = np.linspace(-K_RANGE, K_RANGE, N_BINS + 1)
    bins = timed(f"{N_BINS} normalised bins",
                 lambda: [np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0)
                          for b in range(N_BINS)])
    covs = timed("4 neighbour covariances", lambda: [
        np.nanmean((st - mean_f[None])[:, :, :-1] * (st - mean_f[None])[:, :, 1:], axis=0),
        np.nanmean((st - mean_f[None])[:, :-1, :] * (st - mean_f[None])[:, 1:, :], axis=0),
        np.nanmean((st - mean_f[None])[:, :-1, :-1] * (st - mean_f[None])[:, 1:, 1:], axis=0),
        np.nanmean((st - mean_f[None])[:, 1:, :-1] * (st - mean_f[None])[:, :-1, 1:], axis=0),
    ])
    total = sum(d for _, d in timings)
    print(f"    {'TOTAL':<34} {total:9.1f}")
    no_cov = sum(d for l, d in timings if l != "4 neighbour covariances")
    print()
    print(f"  without covariances: {no_cov:.0f} ms   with: {total:.0f} ms")
    print(f"  per cycle (14 vars x 81 leads), WITHOUT covariances: "
          f"{no_cov * UNITS / 1000 / 3600:.2f} h single-core, "
          f"{no_cov * UNITS / 1000 / 3600 / 2:.2f} h on 2 cores")
    print(f"  per cycle WITH covariances:                          "
          f"{total * UNITS / 1000 / 3600:.2f} h single-core, "
          f"{total * UNITS / 1000 / 3600 / 2:.2f} h on 2 cores")
    print()
    print("  NOTE: single-node simulation of the aggregate stage only. The real pipeline's")
    print("  behaviour after restructuring to 'collect all 30 members per variable' may")
    print("  differ; and these are wall-clock timings on one machine, not a benchmark of")
    print("  the deployed 4-core ARM64 host.")
    print()


if __name__ == "__main__":
    main()
