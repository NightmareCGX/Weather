"""Experiment 24 (REAL DATA): can the bundles satisfy RARE-event thresholds?

The review's question: if precipitation-like variables carry ALL 27 fields
(MEAN + STD + 16 bins + P10..P90 + 4 covariances), does that satisfy the arbitrary
exceedance-threshold requirement?

It cannot, structurally: the five stored quantiles are P10..P90, which do not reach a P99
threshold. The bound is set by the SHAPE encoding, not by the quantile fields. This
measures it, and compares the encodings that could actually reach the tail:

  A. +-4sigma normalised 16 bins          (the failing baseline)
  B. log1p + normalised 16 bins           (path 1 variant)
  C. quantile function, 17 log-spaced levels incl. 0.99/0.995/0.998/0.999  (the candidate)
  D. C plus the 5 body quantiles          (what "all 27 fields" amounts to, shape-wise)

Thresholds are expressed as percentiles of the local distribution (P50/P90/P99/P99.9), so
they are comparable across leads. Errors are relative to the ensemble's own sampling
noise at n = 30, which is ALSO the resolution floor: with 30 members an exceedance
probability below ~1/30 cannot be measured by the ensemble at all.

Run:  .venv/Scripts/python.exe bench_rare_thresholds.py
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
Q_LEVELS = (50.0, 90.0, 99.0, 99.9)


def main():
    for cycle_date, cycle, lead in (("20260918", "18", "f006"),
                                    ("20260918", "00", "f006"),
                                    ("20260918", "00", "f240")):
        for short, level, name, offset in (("tp", 0, "APCP precip", 0.0),
                                           ("2t", 2, "2t temperature", 273.15)):
            paths = [
                os.path.join(CACHE, f"{cycle_date}{cycle}_gep{m:02d}_{lead}_{short}.grib2")
                for m in range(1, 31)
            ]
            if not all(os.path.exists(p) for p in paths):
                continue
            stack = np.stack([decode(p, short, level) - np.float32(offset) for p in paths])

            n_pts = 800
            rng = np.random.default_rng(3)
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
            mu_p = np.nanmean(interp, axis=0)
            sd_p = np.nanstd(interp, axis=0)

            # ---------- encodings ----------
            mean_f = np.nanmean(stack, axis=0)
            std_f = np.nanstd(stack, axis=0)
            z = (stack - mean_f[None]) / np.maximum(std_f[None], 1e-9)
            edges = np.linspace(-K_RANGE, K_RANGE, N_BINS + 1)
            bins = [np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0)
                    for b in range(N_BINS)]

            lz = np.log1p(np.maximum(stack, 0.0))
            lm = np.nanmean(lz, axis=0)
            ls = np.nanstd(lz, axis=0)
            lzn = (lz - lm[None]) / np.maximum(ls[None], 1e-9)
            lbins = [np.nanmean((lzn >= edges[b]) & (lzn < edges[b + 1]), axis=0)
                     for b in range(N_BINS)]

            qf = quantiles_fast(stack, QF_LEVELS)
            # E: the SAME quantile function but stored as dimensionless z-scores, so the
            # fields are O(1) smooth numbers instead of full-range value fields. This is
            # the parameterisation that should make the tail-capable encoding affordable.
            qf_z = [(q - mean_f) / np.maximum(std_f, 1e-9) for q in qf]
            # F: coarser level set (11 levels, still log-spaced at both ends)
            levels11 = (0.005, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98, 0.995, 0.999)
            qf11 = quantiles_fast(stack, levels11)
            qf11_z = [(q - mean_f) / np.maximum(std_f, 1e-9) for q in qf11]

            enc = {
                "A +-4sigma 16 bins": ("hist", bins, None),
                "B log1p 16 bins": ("hist_log", lbins, None),
                "C qfunc 17 levels (absolute)": ("qf", qf, QF_LEVELS),
                "E qfunc 17 levels (z-normalised)": ("qfz", qf_z, np.array(QF_LEVELS)),
                "F qfunc 11 levels (z-normalised)": ("qfz", qf11_z, np.array(levels11)),
            }

            print()
            print("=" * 112)
            print(f"{name}   {cycle_date} {cycle}Z {lead}"
                  f"   sigma(mean) {float(np.nanmean(std_f)):.3f}"
                  f"   surface zero-fraction {(stack == 0).mean():.3f}")
            print("=" * 112)
            print(f"{'encoding':<32} {'fields':>7} {'MB':>8} {'vs base':>8}   "
                  + "  ".join(f"P{q:g} >" for q in Q_LEVELS) + "   worst")
            print("-" * 112)
            base_mb = sum(compress_field(stack[m]) for m in range(30)) / 1e6

            for label, (kind, fields, extra) in enc.items():
                errs = []
                for q in Q_LEVELS:
                    t = np.nanpercentile(interp, q, axis=0)          # per-point threshold
                    truth = np.mean(interp > t[None, :], axis=0)
                    nz = float(np.max(np.abs(
                        np.mean(interp[:half] > t[None, :], axis=0)
                        - np.mean(interp[half:] > t[None, :], axis=0))))
                    if kind in ("hist", "hist_log"):
                        if kind == "hist":
                            coord = (t - bil(mean_f)) / np.maximum(bil(std_f), 1e-12)
                        else:
                            coord = ((np.log1p(np.maximum(t, 0.0)) - bil(lm))
                                     / np.maximum(bil(ls), 1e-12))
                        mix = np.stack([bil(f) for f in fields])
                        cum = np.cumsum(mix, axis=0)
                        width = edges[1] - edges[0]
                        pos = (coord - edges[0]) / width
                        idx = np.clip(np.floor(pos).astype(int), 0, N_BINS - 1)
                        frac = np.clip(pos - idx, 0.0, 1.0)
                        ar = np.arange(pos.size)
                        lo_cdf = np.where(idx > 0, cum[np.maximum(idx - 1, 0), ar], 0.0)
                        cdf = np.clip(lo_cdf + frac * mix[idx, ar], 0.0, 1.0)
                        est = 1.0 - cdf
                    elif kind == "qf":
                        xs = np.stack([bil(f) for f in fields])
                        est = 1.0 - np.array(
                            [np.interp(t[i], xs[:, i], extra) for i in range(xs.shape[1])]
                        )
                    else:  # qfz: unnormalise the stored z-scores first
                        zs = np.stack([bil(f) for f in fields])
                        mu_a, sd_a = bil(mean_f), bil(std_f)
                        xs = mu_a[None, :] + zs * sd_a[None, :]
                        est = 1.0 - np.array(
                            [np.interp(t[i], xs[:, i], extra) for i in range(xs.shape[1])]
                        )
                    errs.append(float(np.max(np.abs(est - truth))) / nz if nz else 0.0)
                nf = len(fields)
                mb = sum(compress_field(quantise(f), np.int16) for f in fields) / 1e6
                worst = max(errs)
                print(f"{label:<32} {nf:>7} {mb:8.3f} {base_mb / mb:7.2f}x   "
                      + "   ".join(f"{e:5.2f}" for e in errs)
                      + f"   {worst:5.2f} {'OK' if worst < 1 else 'FAILS'}")
            print(f"  (baseline 30 members = {base_mb:.3f} MB; 1/30 sample floor = 0.033)")
            del stack, interp
            import gc
            gc.collect()


if __name__ == "__main__":
    main()
