"""Experiment 27 (REAL DATA): 16 vs 32 normalised bins for near-Gaussian variables.

The review recalls a 16-vs-32 bin comparison. The only one in the report is section
9.13.2, which ran on SYNTHETIC fields and measured QUANTILE reconstruction, not
exceedance thresholds. This measures the real thing:

  * byte cost of the bin block and of the whole bundle (MEAN + STD + bins)
  * saving vs the real 30-member baseline
  * error at BODY thresholds (fixed absolute values cutting the middle of the local
    distribution) and at RARE thresholds (P50/P90/P99/P99.9 of the local distribution)

for 2t (near-Gaussian) and APCP (zero-inflated) at f006 and f240.

Yardstick: max |estimate - truth| / the ensemble's own sampling noise at n = 30.

Run:  .venv/Scripts/python.exe bench_bin_count.py
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import (  # noqa: E402
    CACHE, K_RANGE, compress_field, decode, quantise,
)

LEADS_PER_CYCLE, GEFS_VARS = 81, 14
UNITS = LEADS_PER_CYCLE * GEFS_VARS
BIN_COUNTS = (16, 32, 64)
RARE_Q = (50.0, 90.0, 99.0, 99.9)

#: fixed absolute body thresholds per variable
BODY = {
    "tp": (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0),
    "2t": None,  # filled in from the field's own mean/sigma
}
BODY_Z = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def main():
    for cycle_date, cycle, lead in (("20260918", "00", "f006"),
                                    ("20260918", "00", "f240")):
        for short, level, name, offset in (("2t", 2, "2t temperature (degC)", 273.15),
                                           ("tp", 0, "APCP precip (mm)", 0.0)):
            paths = [
                os.path.join(CACHE, f"{cycle_date}{cycle}_gep{m:02d}_{lead}_{short}.grib2")
                for m in range(1, 31)
            ]
            if not all(os.path.exists(p) for p in paths):
                continue
            stack = np.stack([decode(p, short, level) - np.float32(offset) for p in paths])
            base_mb = sum(compress_field(stack[m]) for m in range(30)) / 1e6

            n_pts = 800
            rng = np.random.default_rng(29)
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
            half = 15
            mean_f, std_f = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
            z = (stack - mean_f[None]) / np.maximum(std_f[None], 1e-9)
            mu_bar, sd_bar = float(np.nanmean(mean_f)), float(np.nanmean(std_f))
            body = BODY[short] or tuple(mu_bar + zz * sd_bar for zz in BODY_Z)

            print()
            print("=" * 116)
            print(f"{name}   {cycle_date} {cycle}Z {lead}   baseline 30 members "
                  f"{base_mb:.3f} MB   sigma_bar {sd_bar:.4f}")
            print("=" * 116)
            print(f"{'bins':>5} {'binMB':>8} {'bundleMB':>9} {'vs base':>8} "
                  f"| {'BODY: max err/noise at':>22} | {'RARE: err/noise at':>26}")
            print(f"{'':>5} {'':>8} {'':>9} {'':>8} | "
                  + " ".join(f"{t:>6.2f}" for t in body) + " | "
                  + " ".join(f"{q:>6.0f}" for q in RARE_Q))
            print("-" * 116)

            for nb in BIN_COUNTS:
                edges = np.linspace(-K_RANGE, K_RANGE, nb + 1)
                bins = [np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0)
                        for b in range(nb)]
                bin_mb = sum(compress_field(quantise(f), np.int16) for f in bins) / 1e6
                bun_mb = bin_mb + sum(
                    compress_field(quantise(f), np.int16) for f in (mean_f, std_f)
                ) / 1e6
                mix = np.stack([bil(f) for f in bins])
                cum = np.cumsum(mix, axis=0)
                width = edges[1] - edges[0]
                ar = np.arange(n_pts)

                def exceed_from_shape(t):
                    coord = (t - bil(mean_f)) / np.maximum(bil(std_f), 1e-12)
                    pos = (coord - edges[0]) / width
                    idx = np.clip(np.floor(pos).astype(int), 0, nb - 1)
                    frac = np.clip(pos - idx, 0.0, 1.0)
                    lo_cdf = np.where(idx > 0, cum[np.maximum(idx - 1, 0), ar], 0.0)
                    return 1.0 - np.clip(lo_cdf + frac * mix[idx, ar], 0.0, 1.0)

                body_errs = []
                for t in body:
                    truth = np.mean(interp > t, axis=0)
                    nz = float(np.max(np.abs(
                        np.mean(interp[:half] > t, axis=0)
                        - np.mean(interp[half:] > t, axis=0))))
                    e = float(np.max(np.abs(exceed_from_shape(t) - truth)))
                    body_errs.append(e / nz if nz else 0.0)

                rare_errs = []
                for q in RARE_Q:
                    t = np.nanpercentile(interp, q, axis=0)
                    truth = np.mean(interp > t[None, :], axis=0)
                    nz = float(np.max(np.abs(
                        np.mean(interp[:half] > t[None, :], axis=0)
                        - np.mean(interp[half:] > t[None, :], axis=0))))
                    e = float(np.max(np.abs(exceed_from_shape(t) - truth)))
                    rare_errs.append(e / nz if nz else 0.0)

                print(f"{nb:>5} {bin_mb:8.3f} {bun_mb:9.3f} {base_mb / bun_mb:7.2f}x | "
                      + " ".join(f"{v:6.2f}" for v in body_errs) + " | "
                      + " ".join(f"{v:6.2f}" for v in rare_errs))
            del stack, interp
            gc.collect()
    print()
    print("  BODY = fixed absolute thresholds through the middle of the local distribution;")
    print("  RARE = percentiles of the local distribution (P50/P90/P99/P99.9).")
    print("  Errors are the max over 800 points, relative to the ensemble's own n=30 noise.")
    print("  < 1.00 means the reconstruction is indistinguishable from the ensemble's own")
    print("  wobble; > 1.00 means the difference would be visible.")
    print()


if __name__ == "__main__":
    main()
