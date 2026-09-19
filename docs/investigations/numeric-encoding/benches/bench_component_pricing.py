"""Experiment 21 (REAL DATA): price every bundle component per cycle.

The review noticed that "16 bins" and "5 explicit quantiles" both come to roughly
5-6 GB/cycle, which looks like a coincidence worth explaining. This prices each group on
the real GEFS members, per field and per cycle, so the arithmetic is visible.

It also clarifies what the bins are FOR once the per-member dots are gone and arbitrary
exceedance thresholds must still be served: the normalised bins are not decoration, they
are the stored CDF. P(x > t) for an ARBITRARY t is answered by looking up
z = (t - interp(MEAN)) / interp(STD) in the interpolated shape's CDF, with linear
interpolation inside the crossing bin. So the bins are REQUIRED by the threshold
requirement, and the 5 explicit quantile fields become an optional precision refinement
rather than a necessity.

Run:  .venv/Scripts/python.exe bench_component_pricing.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import (  # noqa: E402
    CACHE, DATE, CYCLE, LEAD, N_BINS, K_RANGE, VARIABLES, bundle_fields, compress_field,
    decode, quantise,
)

LEADS, GEFS_VARS = 81, 14
UNITS_PER_CYCLE = LEADS * GEFS_VARS


def main():
    print()
    print("=" * 104)
    print(f"REAL DATA component pricing: GEFS {DATE} {CYCLE}Z {LEAD}, {N_BINS} normalised bins")
    print("=" * 104)

    for label, var, lvl_tok, short, level, unit in VARIABLES:
        paths = [
            os.path.join(CACHE, f"{DATE}{CYCLE}_gep{m:02d}_{LEAD}_{short}.grib2")
            for m in range(1, 31)
        ]
        stack = np.stack([decode(p, short, level) for p in paths])

        mean_f, std_f, bins, covs, z = bundle_fields(stack, True)
        quants = [np.nanpercentile(stack, q, axis=0) for q in (10, 25, 50, 75, 90)]

        base_mb = sum(compress_field(stack[m]) for m in range(30)) / 1e6

        groups = [
            ("MEAN", [mean_f]),
            ("STD", [std_f]),
            (f"{N_BINS} normalised bins", bins),
            ("5 explicit quantiles", quants),
            ("4 covariances", covs),
        ]

        print()
        print(f"{label}   ({unit})   baseline 30 members = {base_mb:.3f} MB per (variable, lead)")
        print(f"{'component':<26} {'fields':>7} {'MB':>9} {'MB/field':>10} "
              f"{'GB/cycle':>10} {'vs baseline':>12}")
        print("-" * 104)
        tot_mb = 0.0
        for gname, gf in groups:
            b = sum(compress_field(quantise(f), np.int16) for f in gf) / 1e6
            tot_mb += b
            print(f"{gname:<26} {len(gf):>7} {b:9.3f} {b / len(gf):10.3f} "
                  f"{b * UNITS_PER_CYCLE / 1e3:10.2f} {base_mb / b:11.2f}x")
        print("-" * 104)
        print(f"{'ALL':<26} {sum(len(g) for _, g in groups):>7} {tot_mb:9.3f} "
              f"{tot_mb / sum(len(g) for _, g in groups):10.3f} "
              f"{tot_mb * UNITS_PER_CYCLE / 1e3:10.2f} {base_mb / tot_mb:11.2f}x")
        ms = sum(compress_field(quantise(f), np.int16) for f in [mean_f, std_f]) / 1e6
        print(f"{'MEAN + STD + bins':<26} {2 + N_BINS:>7} {ms + sum(compress_field(quantise(f), np.int16) for f in bins) / 1e6:9.3f}")
        print()

    # ---------------------------------------------------------------------------------
    # Arbitrary-threshold exceedance from the shape: where is it reliable?
    # ---------------------------------------------------------------------------------
    print("=" * 104)
    print("Arbitrary-threshold exceedance P(x > t) answered from the 16-bin shape")
    print("=" * 104)
    print(f"{'variable':<12} {'threshold (z)':>14} {'truth P':>10} {'from shape':>11} "
          f"{'abs err':>9} {'sampling noise':>15} {'err/noise':>10}")
    print("-" * 104)
    rng = np.random.default_rng(7)
    for label, var, lvl_tok, short, level, unit in VARIABLES:
        paths = [os.path.join(CACHE, f"{DATE}{CYCLE}_gep{m:02d}_{LEAD}_{short}.grib2")
                 for m in range(1, 31)]
        stack = np.stack([decode(p, short, level) for p in paths])
        mean_f, std_f, bins, covs, zstack = bundle_fields(stack, True)
        pts = 1500
        rows = rng.uniform(1, 718, pts); cols = rng.uniform(2, 1437, pts)
        r0, c0 = rows.astype(int), cols.astype(int)
        tr, tc = rows - r0, cols - c0
        def bil(f):
            g00,g01 = f[r0,c0], f[r0,c0+1]; g10,g11 = f[r0+1,c0], f[r0+1,c0+1]
            lo = g00+(g01-g00)*tc; up = g10+(g11-g10)*tc
            return lo+(up-lo)*tr
        interp = np.stack([bil(stack[m]) for m in range(30)])
        mix = np.stack([bil(b) for b in bins])
        cdf = np.cumsum(mix, axis=0)
        edges = np.linspace(-K_RANGE, K_RANGE, N_BINS + 1); width = edges[1]-edges[0]
        ar = np.arange(pts)
        half = 15
        for zt in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):
            def exceed(a):
                mu = np.nanmean(a, axis=0); sd = np.nanstd(a, axis=0)
                zz = (a - mu[None]) / np.maximum(sd[None], 1e-12)
                return np.mean(zz > zt, axis=0)
            truth = exceed(interp)
            nz = float(np.max(np.abs(exceed(interp[:half]) - exceed(interp[half:]))))
            mu_a = bil(mean_f); sd_a = bil(std_f)
            zt_at = (mu_a + zt * sd_a - mu_a) / np.maximum(sd_a, 1e-12)
            idx = np.clip((zt_at - edges[0]) / width, 0, N_BINS - 1).astype(int)
            cdf_lo = np.where(idx > 0, cdf[np.maximum(idx-1,0), ar], 0.0)
            frac = np.clip((zt_at - edges[idx]) / width, 0.0, 1.0)
            cdf_at = cdf_lo + frac * mix[idx, ar]
            shape_ex = 1.0 - np.clip(cdf_at, 0.0, 1.0)
            e = float(np.max(np.abs(shape_ex - truth)))
            print(f"{label.split()[0]:<12} {zt:>14.1f} {truth.mean():>10.4f} {shape_ex.mean():>11.4f} "
                  f"{e:>9.4f} {nz:>15.4f} {e/nz if nz else 0:>10.3f}")
    print()
    print("  P(x > t) is answered by looking z = (t - MEAN)/STD up in the shape's CDF, so the")
    print("  stored bins ARE the mechanism for arbitrary thresholds. Accuracy is good in the")
    print("  body of the distribution; in the far tail the 30-member ensemble itself cannot")
    print("  resolve probabilities below ~1/30, so the bins are not the binding limit there.")
    print()
    print("=" * 104)
    print("Why '16 bins' and '5 quantiles' land in the same place")
    print("=" * 104)
    print("  Per FIELD the bins are far cheaper (bounded fractions in [0,1] compress hard)")
    print("  but there are ~3x more of them than the quantiles, and the quantiles are")
    print("  full-range value fields. The two effects very nearly cancel, which is why both")
    print("  land at the same per-cycle cost -- visible as MB/field above.")
    print()


if __name__ == "__main__":
    main()
