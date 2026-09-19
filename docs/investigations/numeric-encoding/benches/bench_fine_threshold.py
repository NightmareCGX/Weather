"""Experiment 26 (REAL DATA): fixing the FINEST precipitation threshold.

The measured problem: explicit P(x > t) fields are accurate everywhere except the finest
precipitation threshold (0.1 mm), where they carry a systematic -0.079 bias (-17%
relative). The cause is that 47% of the cells are EXACTLY zero and 0.1 mm cuts right at
the zero/non-zero boundary, so interpolating a probability smooths across the point mass.

Candidate fixes compared at that threshold (and across the ladder), all on real APCP:

  A  explicit P(x>t) fields                     (bias -0.079, the problem)
  B  +-4sigma normalised 16 bins                 (shape route)
  C  absolute quantile function, 17 log levels   (best measured so far, 0.44)
  D  TWO-PART zero-inflated model: store the zero fraction plus the mean/std/shape of the
     POSITIVE part, and reconstruct P(x>t) = (1 - z0) * P_pos(x>t | x>0). One extra field
     over B, and it anchors the point mass instead of smoothing over it.

Yardstick: err relative to the ensemble's own sampling noise, plus bias and the
|err| distribution (p50/p90/p99/max) because a single worst point misleads.

Run:  .venv/Scripts/python.exe bench_fine_threshold.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import (  # noqa: E402
    CACHE, N_BINS, K_RANGE, compress_field, decode, quantise,
)
from bench_tail_paths import QF_LEVELS, quantiles_fast  # noqa: E402

LEADS_PER_CYCLE, GEFS_VARS = 81, 14
UNITS = LEADS_PER_CYCLE * GEFS_VARS
THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
K = K_RANGE


def main():
    for cycle_date, cycle, lead in (("20260918", "18", "f006"),
                                    ("20260918", "00", "f006"),
                                    ("20260918", "00", "f240")):
        paths = [
            os.path.join(CACHE, f"{cycle_date}{cycle}_gep{m:02d}_{lead}_tp.grib2")
            for m in range(1, 31)
        ]
        if not all(os.path.exists(p) for p in paths):
            continue
        stack = np.stack([decode(p, "tp", 0) for p in paths])

        n_pts = 800
        rng = np.random.default_rng(23)
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
        pos = np.where(stack > 0, stack, np.nan)

        # ---------------- encodings ----------------
        # A: explicit exceedance fields
        fA = [np.nanmean(stack > t, axis=0) for t in THRESHOLDS]

        # B: +-4sigma normalised bins
        mean_f, std_f = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
        z = (stack - mean_f[None]) / np.maximum(std_f[None], 1e-9)
        edges = np.linspace(-K, K, N_BINS + 1)
        fB = [np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0)
              for b in range(N_BINS)]

        # C: absolute quantile function, original levels
        fC = quantiles_fast(stack, QF_LEVELS)
        # C2: same, but with the level set densified in the BODY of the CDF. C's own
        # p99 error is pinned at 0.200 across all three cycles, which is the spacing of its
        # probability grid near the median -- and the finest precipitation threshold sits
        # at P~0.45, i.e. in the body, not the tail. So the fix is body levels.
        levels_body = (0.001, 0.005, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50,
                       0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.995, 0.999)
        fC2 = quantiles_fast(stack, levels_body)
        levels_rich = (0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22,
                       0.30, 0.38, 0.44, 0.50, 0.56, 0.62, 0.70, 0.78, 0.85, 0.90,
                       0.94, 0.97, 0.99, 0.995, 0.998, 0.999)
        fC3 = quantiles_fast(stack, levels_rich)

        # D: two-part zero-inflated model
        z0 = np.nanmean(stack == 0, axis=0)                      # point mass at 0
        pmean = np.nanmean(pos, axis=0)
        pstd = np.nanstd(pos, axis=0)
        zp = (pos - pmean[None]) / np.maximum(pstd[None], 1e-9)
        fD = [z0, pmean, pstd] + [
            np.nanmean((zp >= edges[b]) & (zp < edges[b + 1]), axis=0)
            for b in range(N_BINS)
        ]
        # an all-NaN positive shape bin would poison the interpolation; zero it, since
        # those cells have no positive mass to describe
        fD = [np.nan_to_num(f, nan=0.0) if f.ndim == 2 else f for f in fD]

        def predict(kind, t):
            if kind == "A":
                return bil(fA[THRESHOLDS.index(t)])
            if kind == "B":
                mix = np.stack([bil(f) for f in fB])
                cum = np.cumsum(mix, axis=0)
                coord = (t - bil(mean_f)) / np.maximum(bil(std_f), 1e-12)
                w = edges[1] - edges[0]
                p_ = (coord - edges[0]) / w
                idx = np.clip(np.floor(p_).astype(int), 0, N_BINS - 1)
                fr = np.clip(p_ - idx, 0.0, 1.0)
                ar = np.arange(len(coord))
                lo_cdf = np.where(idx > 0, cum[np.maximum(idx - 1, 0), ar], 0.0)
                return 1.0 - np.clip(lo_cdf + fr * mix[idx, ar], 0.0, 1.0)
            if kind in ("C", "C2", "C3"):
                flds, lv = (fC, QF_LEVELS) if kind == "C" else (
                    (fC2, levels_body) if kind == "C2" else (fC3, levels_rich))
                xs = np.stack([bil(f) for f in flds])
                return 1.0 - np.array(
                    [np.interp(t, xs[:, i], lv) for i in range(xs.shape[1])]
                )
            # D: two-part. Cells where EVERY member is exactly zero are common for
            # precipitation, so the positive part is undefined there and P(x>t)=0 for t>0.
            z0a = np.clip(bil(fD[0]), 0.0, 1.0)
            if t <= 0:
                return np.ones_like(z0a)
            pm_raw, ps_raw = bil(fD[1]), bil(fD[2])
            valid = np.isfinite(pm_raw) & np.isfinite(ps_raw)
            pma = np.where(valid, np.nan_to_num(pm_raw, nan=0.0), 0.0)
            psd = np.maximum(np.where(valid, np.nan_to_num(ps_raw, nan=1.0), 1.0), 1e-12)
            mix = np.stack([bil(f) for f in fD[3:]])
            cum = np.cumsum(mix, axis=0)
            coord = (t - pma) / psd
            w = edges[1] - edges[0]
            p_ = (coord - edges[0]) / w
            idx = np.clip(np.floor(p_).astype(int), 0, N_BINS - 1)
            fr = np.clip(p_ - idx, 0.0, 1.0)
            ar = np.arange(len(coord))
            lo_cdf = np.where(idx > 0, cum[np.maximum(idx - 1, 0), ar], 0.0)
            cdf_pos = np.clip(lo_cdf + fr * mix[idx, ar], 0.0, 1.0)
            est = np.where(coord < edges[0], 1.0 - z0a, (1.0 - z0a) * (1.0 - cdf_pos))
            return np.where(valid, est, 0.0)

        labels = {
            "A": ("A explicit P(x>t) fields", fA),
            "B": ("B +-4sigma 16 bins", fB),
            "C": ("C qfunc 17 levels", fC),
            "C2": ("C2 qfunc 19 levels (body-dense)", fC2),
            "C3": ("C3 qfunc 26 levels (rich)", fC3),
            "D": ("D two-part (z0 + positive shape)", fD),
        }
        base_mb = sum(compress_field(stack[m]) for m in range(30)) / 1e6

        print()
        print("=" * 116)
        print(f"APCP precip   {cycle_date} {cycle}Z {lead}   baseline 30 members {base_mb:.3f} MB   "
              f"zero-fraction {(stack == 0).mean():.3f}")
        print("=" * 116)
        print(f"{'encoding':<34} {'fields':>6} {'MB':>8} {'vs base':>8} "
              f"{'thr':>6} {'truth':>7} {'est':>7} {'bias':>8} "
              f"{'p50':>7} {'p90':>7} {'p99':>7} {'max':>7} {'max/noise':>10}")
        print("-" * 116)
        for kind in ("A", "B", "C", "C2", "C3", "D"):
            label, fields = labels[kind]
            mb = sum(compress_field(quantise(f), np.int16) for f in fields) / 1e6
            for t in THRESHOLDS:
                truth = np.mean(interp > t, axis=0)
                est = predict(kind, t)
                d = np.abs(est - truth)
                nz = float(np.max(np.abs(
                    np.mean(interp[:half] > t, axis=0) - np.mean(interp[half:] > t, axis=0))))
                ratio = float(d.max()) / nz if nz else 0.0
                flag = "  <-- finest" if t == THRESHOLDS[0] else ""
                print(f"{label if t == THRESHOLDS[0] else '':<34} "
                      f"{len(fields) if t == THRESHOLDS[0] else 0:>6} "
                      f"{mb if t == THRESHOLDS[0] else 0:8.3f} "
                      f"{base_mb / mb if t == THRESHOLDS[0] else 0:7.2f}x "
                      f"{t:6.2f} {truth.mean():7.4f} {est.mean():7.4f} "
                      f"{est.mean() - truth.mean():+8.4f} "
                      f"{np.percentile(d, 50):7.4f} {np.percentile(d, 90):7.4f} "
                      f"{np.percentile(d, 99):7.4f} {d.max():7.4f} {ratio:10.2f}{flag}")
            print()
        del stack, interp, pos
        import gc
        gc.collect()


if __name__ == "__main__":
    main()
