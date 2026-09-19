"""Experiment 25 (REAL DATA): explicit P(x > t) fields -- path 2 done properly.

Section 9.15.4 measured explicit exceedance fields at 1.235 MB for 7 thresholds
(11.8x cheaper than the members) but dismissed the path because the worst threshold came
out at 1.28x the sampling noise. Batch 3 then showed the SHAPE encodings fail much worse
(2-15x) on tail thresholds. Put together, the explicit-field path looks like the winner
for a product whose threshold set is FIXED AND KNOWN -- which `/v1/probabilities` is.

This measures it across a threshold ladder that spans the body and the tail, for the two
variable classes and two lead times, reporting BOTH:
  * the ratio to the ensemble's own sampling noise (the yardstick used throughout), and
  * the ABSOLUTE probability error, because at rare thresholds the noise floor itself is
    tiny and a large ratio can still be a trivial absolute error.

Run:  .venv/Scripts/python.exe bench_explicit_thresholds.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import (  # noqa: E402
    CACHE, compress_field, decode, quantise,
)

LEADS_PER_CYCLE, GEFS_VARS = 81, 14
UNITS = LEADS_PER_CYCLE * GEFS_VARS

#: (variable, cfgrib level, display name, offset, threshold ladder)
CASES = (
    ("tp", 0, "APCP precip (mm)", 0.0,
     (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)),
    ("2t", 2, "2t temperature (degC)", 273.15,
     (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0)),
)

LEADS = (("20260918", "18", "f006"), ("20260918", "00", "f006"),
         ("20260918", "00", "f240"))


def main():
    for cycle_date, cycle, lead in LEADS:
        for short, level, name, offset, thresholds in CASES:
            paths = [
                os.path.join(CACHE, f"{cycle_date}{cycle}_gep{m:02d}_{lead}_{short}.grib2")
                for m in range(1, 31)
            ]
            if not all(os.path.exists(p) for p in paths):
                continue
            stack = np.stack([decode(p, short, level) - np.float32(offset) for p in paths])

            n_pts = 800
            rng = np.random.default_rng(17)
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

            # the stored fields: per-cell fraction of members above each threshold
            fields = [np.nanmean(stack > t, axis=0) for t in thresholds]
            mb = sum(compress_field(quantise(f), np.int16) for f in fields) / 1e6
            base_mb = sum(compress_field(stack[m]) for m in range(30)) / 1e6

            print()
            print("=" * 112)
            print(f"{name}   {cycle_date} {cycle}Z {lead}   "
                  f"baseline {base_mb:.3f} MB   explicit fields {mb:.3f} MB "
                  f"({base_mb / mb:.2f}x) for {len(thresholds)} thresholds")
            print("=" * 112)
            print(f"{'threshold':>10} {'truth P':>8} {'est P':>8} {'bias':>8} "
                  f"{'|e| p50':>8} {'|e| p90':>8} {'|e| p99':>8} {'|e| max':>8} "
                  f"{'noise':>7} {'max/noise':>10}")
            print("-" * 112)
            worst_ratio = 0.0
            for i, t in enumerate(thresholds):
                truth = np.mean(interp > t, axis=0)
                est = bil(fields[i])
                d = np.abs(est - truth)
                nz = float(np.max(np.abs(
                    np.mean(interp[:half] > t, axis=0) - np.mean(interp[half:] > t, axis=0))))
                ratio = float(d.max()) / nz if nz else 0.0
                worst_ratio = max(worst_ratio, ratio)
                print(f"{t:>10.2f} {truth.mean():8.4f} {est.mean():8.4f} "
                      f"{est.mean() - truth.mean():+8.4f} "
                      f"{np.percentile(d, 50):8.4f} {np.percentile(d, 90):8.4f} "
                      f"{np.percentile(d, 99):8.4f} {d.max():8.4f} {nz:7.4f} {ratio:10.2f}")
            print(f"  worst max/noise {worst_ratio:.2f}   "
                  f"{'OK' if worst_ratio < 1 else 'MARGINAL/FAILS'}")
            del stack, interp
            import gc
            gc.collect()


if __name__ == "__main__":
    main()
