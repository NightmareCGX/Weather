"""Experiment 22 (REAL DATA): the three candidate paths for the precipitation
upper-tail threshold failure, plus a fourth encoding that should do better.

Measured problem (section 9.15.2): answering an arbitrary exceedance threshold
P(x > t) from a +-4-sigma NORMALISED histogram fails badly for precipitation, because the
tail extends to ~100 sigma while the normalised support covers only +-4.

Encodings compared, all on the same real 30 members:
  1. +-4sigma normalised 16 bins              (the failing baseline)
  2. absolute range 32 bins over [0, P99.9]   (path 1: variable-adapted shape)
  3. log1p then +-4sigma normalised 16 bins   (path 1 variant: log scale)
  4. quantile function at 17 log-spaced levels (the candidate: same information as a
     histogram but parameterised in PROBABILITY space, so the tail gets real resolution)
  5. explicit P(x > t) fields at the tested thresholds (path 2, upper bound on quality)
  6. members (reference -- truth is computed from the interpolated members)

Paths 1-4 are per-cell fields interpolated to the query point; the threshold is answered
from the interpolated representation. Yardstick: the ensemble's own sampling noise (n=30).

Run:  .venv/Scripts/python.exe bench_tail_paths.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import (  # noqa: E402
    CACHE, DATE, CYCLE, N_BINS, K_RANGE, compress_field, decode, quantise,
)
from bench_real_gefs_bundle import bundle_fields  # noqa: E402

LEADS_PER_CYCLE, GEFS_VARS = 81, 14
UNITS = LEADS_PER_CYCLE * GEFS_VARS

#: thresholds in mm, spanning the body and the upper tail of the precipitation field
THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)

#: log-spaced probability levels for the quantile-function encoding
QF_LEVELS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50,
             0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)


def quantiles_fast(stack, levels):
    """All quantile levels from ONE sort along the member axis.

    np.nanpercentile with a list of levels re-partitions per level and is ~50x slower on
    a (30, 721, 1440) array; one sorted array plus linear interpolation at each level
    gives the same values cheaply. NaN sorts to the end in numpy, so it is mapped to +inf
    first to keep the member axis clean.
    """
    srt = np.sort(np.where(np.isnan(stack), np.inf, stack), axis=0)
    n = srt.shape[0]
    out = []
    for p in levels:
        pos = p * (n - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        v = srt[lo] * (1 - frac) + srt[hi] * frac
        out.append(v.astype(np.float32))
    return out


def bilinear(field, r0, c0, tr, tc):
    g00, g01 = field[r0, c0], field[r0, c0 + 1]
    g10, g11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
    lo = g00 + (g01 - g00) * tc
    up = g10 + (g11 - g10) * tc
    return lo + (up - lo) * tr


def main():
    for var, short, level, unit in (("apcp (6h precip) mm", "tp", 0, "mm"),
                                    ("2t (temperature) degC", "2t", 2, "K")):
        paths = [
            os.path.join(CACHE, f"{DATE}{CYCLE}_gep{m:02d}_f006_{short}.grib2")
            for m in range(1, 31)
        ]
        if not all(os.path.exists(p) for p in paths):
            paths = [
                os.path.join(CACHE, f"2026091818_gep{m:02d}_f006_{short}.grib2")
                for m in range(1, 31)
            ]
        stack = np.stack([decode(p, short, level) for p in paths])
        if short == "2t":
            stack = stack - np.float32(273.15)

        n_pts = 1200
        rng = np.random.default_rng(11)
        rows = rng.uniform(1, 718, n_pts)
        cols = rng.uniform(2, 1437, n_pts)
        r0, c0 = rows.astype(int), cols.astype(int)
        tr, tc = rows - r0, cols - c0
        bil = lambda f: bilinear(f, r0, c0, tr, tc)  # noqa: E731

        interp = np.stack([bil(stack[m]) for m in range(30)])
        half = 15

        def true_exceed(t):
            return np.mean(interp > t, axis=0)

        def noise_exceed(t):
            return float(
                np.max(
                    np.abs(
                        np.mean(interp[:half] > t, axis=0) - np.mean(interp[half:] > t, axis=0)
                    )
                )
            )

        # thresholds: physical ones for precip, z-scaled ones for temperature
        mean_f, std_f, bins_norm, covs, _ = bundle_fields(stack, True)
        if short == "tp":
            thresholds = THRESHOLDS
        else:
            mu = float(np.nanmean(stack))
            sd = float(np.nanmean(std_f))
            thresholds = tuple(mu + z * sd for z in (-2.5, -1.0, 0.0, 1.0, 2.0, 3.0))

        # ---- build every encoding once
        enc = {}

        # 1. normalised +-4sigma bins (already in bundle_fields)
        enc["1. +-4sigma normalised 16 bins"] = ("hist_norm", bins_norm,
                                                 np.linspace(-K_RANGE, K_RANGE, N_BINS + 1))

        # 2. absolute-range bins over [min, P99.9]
        hi = float(np.nanpercentile(stack, 99.9))
        lo = float(np.nanmin(stack))
        edges_abs = np.linspace(lo, hi, 33)
        enc["2. absolute 32 bins [min, P99.9]"] = (
            "hist_abs",
            [np.nanmean((stack >= edges_abs[b]) & (stack < edges_abs[b + 1]), axis=0)
             for b in range(32)],
            edges_abs,
        )

        # 3. log1p then normalised
        zs = np.log1p(np.maximum(stack, 0.0))
        lmean = np.nanmean(zs, axis=0)
        lstd = np.nanstd(zs, axis=0)
        zz = (zs - lmean[None]) / np.maximum(lstd[None], 1e-9)
        edges_l = np.linspace(-K_RANGE, K_RANGE, N_BINS + 1)
        enc["3. log1p + normalised 16 bins"] = (
            "hist_log",
            [np.nanmean((zz >= edges_l[b]) & (zz < edges_l[b + 1]), axis=0)
             for b in range(N_BINS)],
            edges_l,
        )

        # 4. quantile function at log-spaced levels
        enc["4. quantile function (17 log levels)"] = (
            "qfunc",
            quantiles_fast(stack, QF_LEVELS),
            np.array(QF_LEVELS),
        )

        # 5. explicit P(x > t) fields at the tested thresholds
        enc["5. explicit P(x>t) fields"] = (
            "pexp",
            [np.nanmean(stack > t, axis=0) for t in thresholds],
            np.array(thresholds),
        )

        # ---- storage
        print()
        print("=" * 112)
        print(f"{var}")
        print("=" * 112)
        base_mb = sum(compress_field(stack[m]) for m in range(30)) / 1e6
        print(f"  baseline 30 members: {base_mb:.3f} MB per (variable, lead)")
        print()
        print(f"{'encoding':<38} {'fields':>7} {'MB':>9} {'vs base':>9} "
              f"{'max err/threshold, as a fraction of sampling noise':>0}")
        print("-" * 112)

        # ---- answer thresholds from each encoding
        def hist_exceed(kind, fields, edges):
            """P(x > t) from a stored histogram, with within-bin linear interpolation.

            The bins are interpolated to the query point first (they are per-cell fields),
            then the CDF is read off at the threshold's coordinate in that histogram's own
            space: z-space for the normalised/log encodings, physical units for the
            absolute one.
            """
            mix = np.stack([bil(f) for f in fields])
            cum = np.cumsum(mix, axis=0)
            ar = np.arange(mix.shape[1])
            width = edges[1] - edges[0]
            out = []
            for t in thresholds:
                if kind == "hist_norm":
                    coord = (t - bil(mean_f)) / np.maximum(bil(std_f), 1e-12)
                elif kind == "hist_log":
                    coord = (np.log1p(max(t, 0.0)) - bil(lmean)) / np.maximum(bil(lstd), 1e-12)
                else:
                    coord = np.full(mix.shape[1], t, dtype=float)
                pos = (coord - edges[0]) / width
                idx = np.clip(np.floor(pos).astype(int), 0, len(fields) - 1)
                frac = np.clip(pos - idx, 0.0, 1.0)
                lo_cdf = np.where(idx > 0, cum[np.maximum(idx - 1, 0), ar], 0.0)
                cdf = np.clip(lo_cdf + frac * mix[idx, ar], 0.0, 1.0)
                out.append(1.0 - cdf)
            return out

        def qfunc_exceed(fields):
            out = []
            xs = np.stack([bil(f) for f in fields])  # (nlevels, npts)
            for t in thresholds:
                # F(t): invert the quantile function by interpolating p against x
                f = np.array([np.interp(t, xs[:, i], QF_LEVELS) for i in range(xs.shape[1])])
                out.append(1.0 - f)
            return out

        for name, (kind, fields, extra) in enc.items():
            if kind in ("hist_norm", "hist_abs", "hist_log"):
                pred = hist_exceed(kind, fields, extra)
            elif kind == "qfunc":
                pred = qfunc_exceed(fields)
            else:
                pred = [bil(f) for f in fields]
            errs = []
            for i, t in enumerate(thresholds):
                nz = noise_exceed(t)
                e = float(np.max(np.abs(pred[i] - true_exceed(t))))
                errs.append(e / nz if nz else 0.0)
            nf = len(fields)
            mb = sum(compress_field(quantise(f), np.int16) for f in fields) / 1e6
            worst = max(errs)
            flag = "OK" if worst < 1.0 else "FAILS"
            print(f"{name:<38} {nf:>7} {mb:9.3f} {base_mb / mb:8.2f}x   "
                  + " ".join(f"{e:5.2f}" for e in errs) + f"   worst {worst:5.2f} {flag}")
        print()
        print("  thresholds tested: " + ", ".join(f"{t:.2f}" for t in thresholds))
        print("  each column is err / sampling-noise for that threshold; < 1.00 means hidden")
        print()


if __name__ == "__main__":
    main()
