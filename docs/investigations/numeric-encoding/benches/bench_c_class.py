"""Experiment 28 (REAL DATA): the C-class pre-adoption diagnostic.

C class = bounded / truncated / bimodal, and the one-member probe already found the
dominant structural feature: a large point mass at a CAP.

    visibility            0 .. 24.1 km, only 242 distinct values, 71.75% AT the cap
    cloud_ceiling         0.004 .. 20 km,           54.66% AT the cap ("Unlimited")
    cloud_cover_3h        0 .. 100 %, integer, 8.11% at 0 AND 24.91% at 100
    relative_humidity_2m  5.07 .. 99.97 %,          3.44% at max  (the mildest)

This measures, per variable, three encodings against the real 30 members:

    E1  MEAN + STD + 32 normalised bins      (the A-class choice)
    E2  absolute quantile function, 19 body-dense levels  (the B-class choice)
    E3  TWO-PART at the cap: p_cap + the mean/std/shape of the SUB-CAP part
        (the upper-end analogue of the zero-inflated model that fixed precipitation)

Product-relevant exceedance directions differ by variable -- aviation cares about LOW
visibility and LOW ceiling, cloudiness about HIGH cover, humidity about HIGH RH -- so each
variable gets (a) a fixed absolute threshold ladder in its own direction and (b) a local
percentile ladder in the same direction.

Yardstick: max |estimate - truth| / the ensemble's own sampling noise at n = 30.

Run:  .venv/Scripts/python.exe bench_c_class.py
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import K_RANGE, compress_field, quantise  # noqa: E402
from gefs_fetch_c import decode as decode_c  # noqa: E402
from bench_tail_paths import quantiles_fast  # noqa: E402

DATE, CYCLE, LEAD = "20260918", "00", "f006"
K = K_RANGE
NB = 32
QF = (0.005, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50,
      0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999)

#: (cache short, display name, cfgrib level, canonical divisor, direction, absolute ladder)
VARS = (
    ("vis", "visibility (km)", 0, 1000.0, "below", (0.5, 1.0, 2.0, 5.0, 10.0, 15.0)),
    ("gh", "cloud_ceiling (km)", 0, 1000.0, "below", (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)),
    ("tcc", "cloud_cover_3h (%)", 0, 1.0, "above", (10.0, 25.0, 50.0, 75.0, 90.0)),
    ("r2", "relative_humidity_2m (%)", 0, 1.0, "above", (60.0, 70.0, 80.0, 90.0, 95.0)),
)
LOCAL_PCTL = {"below": (1.0, 5.0, 10.0, 25.0), "above": (75.0, 90.0, 95.0, 99.0)}


def tail_exceed(values, t, direction):
    """P(x < t) for 'below', P(x > t) for 'above'."""
    return np.mean(values < t[None, :], axis=0) if direction == "below" \
        else np.mean(values > t[None, :], axis=0)


def main():
    for short, name, level, div, direction, abs_ladder in VARS:
        paths = [
            os.path.join(
                os.environ.get("WEATHER_REALDATA_CACHE",
                               os.path.join(__import__("tempfile").gettempdir(),
                                            "weather_realdata")),
                f"{DATE}{CYCLE}_gep{m:02d}_{LEAD}_{short}.grib2",
            )
            for m in range(1, 31)
        ]
        if not all(os.path.exists(p) for p in paths):
            print(f"{name}: MISSING data")
            continue
        stack = np.stack(
            [decode_c(p, short)[3].astype(np.float32) / np.float32(div) for p in paths]
        )

        # ---------------- structural diagnostics ----------------
        finite = stack[np.isfinite(stack)]
        cap = float(np.nanmax(stack))
        at_cap = float((stack == cap).mean())
        at_zero = float((stack == 0).mean())
        sd_bar = float(np.nanmean(np.nanstd(stack, axis=0)))
        print()
        print("=" * 116)
        print(f"{name}")
        print(f"  range [{finite.min():.4f}, {cap:.4f}]  distinct={np.unique(finite).size}  "
              f"sigma_bar={sd_bar:.4f}  zero-frac={at_zero:.4f}  "
              f"at-cap-frac={at_cap:.4f}  max/sigma_bar={cap / max(sd_bar, 1e-9):.1f}")
        print("=" * 116)

        n_pts = 800
        rng = np.random.default_rng(41)
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

        # ---------------- encodings ----------------
        mean_f, std_f = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
        z = (stack - mean_f[None]) / np.maximum(std_f[None], 1e-9)
        edges = np.linspace(-K, K, NB + 1)
        E1 = [mean_f, std_f] + [
            np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0) for b in range(NB)
        ]

        E2 = quantiles_fast(stack, QF)

        # E3: two-part at the cap. p_cap = fraction AT the cap; the rest is the sub-cap
        # part, described by its own mean/std/shape.
        p_cap = np.nanmean(stack >= cap, axis=0)
        sub = np.where(stack < cap, stack, np.nan)
        smean = np.nanmean(sub, axis=0)
        sstd = np.nanstd(sub, axis=0)
        sz = (sub - smean[None]) / np.maximum(sstd[None], 1e-9)
        E3 = [p_cap, smean, sstd] + [
            np.nanmean((sz >= edges[b]) & (sz < edges[b + 1]), axis=0) for b in range(NB)
        ]
        E3 = [np.nan_to_num(f, nan=0.0) if f.ndim == 2 else f for f in E3]

        base_mb = sum(compress_field(stack[m]) for m in range(30)) / 1e6

        def as_pts(t):
            """Normalise a threshold -- scalar or per-point array -- to shape (n_pts,)."""
            tt = np.asarray(t, dtype=float)
            return np.full(n_pts, float(tt)) if tt.ndim == 0 else tt

        def exceed_E1(t):
            t = as_pts(t)
            mix = np.stack([bil(f) for f in E1[2:]])
            cum = np.cumsum(mix, axis=0)
            coord = (t - bil(mean_f)) / np.maximum(bil(std_f), 1e-12)
            w = edges[1] - edges[0]
            pos = (coord - edges[0]) / w
            idx = np.clip(np.floor(pos).astype(int), 0, NB - 1)
            fr = np.clip(pos - idx, 0.0, 1.0)
            ar = np.arange(len(np.atleast_1d(coord)))
            lo = np.where(idx > 0, cum[np.maximum(idx - 1, 0), ar], 0.0)
            cdf = np.clip(lo + fr * mix[idx, ar], 0.0, 1.0)
            return cdf if direction == "below" else 1.0 - cdf

        def exceed_E2(t):
            # np.interp must be called per point with a SCALAR x; passing the whole
            # per-point array as x returns the wrong shape entirely.
            tt = as_pts(t)
            xs = np.stack([bil(f) for f in E2])          # (nlevels, n_pts)
            p = np.array([np.interp(tt[i], xs[:, i], QF) for i in range(xs.shape[1])])
            p = np.clip(p, 0.0, 1.0)
            return p if direction == "below" else 1.0 - p

        def exceed_E3(t):
            """Two-part at the cap.

            For x > t with t < cap: the capped members ALL exceed t, so
                P(x>t) = p_cap + (1-p_cap) * P_sub(x>t);
            for t >= cap it is 0 (the cap is the maximum).
            For x < t with t <= cap the capped members never count, so
                P(x<t) = (1-p_cap) * P_sub(x<t);
            for t > cap it is 1.
            """
            tt = as_pts(t)
            pc = np.clip(bil(E3[0]), 0.0, 1.0)
            if direction == "above":
                return np.where(tt >= cap, 0.0, pc + (1.0 - pc) * (1.0 - exceed_sub(tt)))
            return np.where(tt > cap, 1.0, (1.0 - pc) * exceed_sub(tt))

        def exceed_sub(t):
            t = as_pts(t)
            mix = np.stack([bil(f) for f in E3[3:]])
            cum = np.cumsum(mix, axis=0)
            coord = (t - bil(E3[1])) / np.maximum(bil(E3[2]), 1e-12)
            w = edges[1] - edges[0]
            pos = (coord - edges[0]) / w
            idx = np.clip(np.floor(pos).astype(int), 0, NB - 1)
            fr = np.clip(pos - idx, 0.0, 1.0)
            ar = np.arange(len(np.atleast_1d(coord)))
            lo = np.where(idx > 0, cum[np.maximum(idx - 1, 0), ar], 0.0)
            sub = np.clip(lo + fr * mix[idx, ar], 0.0, 1.0)
            return sub if direction == "below" else 1.0 - sub

        encodings = (("E1 32 norm bins", E1, exceed_E1),
                     ("E2 qfunc 19 body-dense", E2, exceed_E2),
                     ("E3 two-part at cap", E3, exceed_E3))

        # threshold sets: fixed absolute ladder, then local percentiles in the same tail
        pctl = LOCAL_PCTL[direction]
        if direction == "below":
            local = [np.nanpercentile(interp, q, axis=0) for q in pctl]
        else:
            local = [np.nanpercentile(interp, q, axis=0) for q in pctl]

        print(f"  direction: P(x {'<' if direction == 'below' else '>'} t)   "
              f"baseline 30 members {base_mb:.3f} MB")
        print()
        print(f"{'encoding':<26} {'fields':>6} {'MB':>8} {'vs base':>8} | "
              f"{'ABS ladder: max err/noise':>26} | {'LOCAL pctl: max err/noise':>26}")
        print(f"{'':<26} {'':>6} {'':>8} {'':>8} | "
              + " ".join(f"{t:>6.2f}" for t in abs_ladder) + " | "
              + " ".join(f"{q:>6.0f}" for q in pctl))
        print("-" * 116)
        for label, fields, fn in encodings:
            mb = sum(compress_field(quantise(f), np.int16) for f in fields) / 1e6
            ae, le = [], []
            for t in abs_ladder:
                truth = tail_exceed(interp, np.full(n_pts, t), direction)
                nz = float(np.max(np.abs(
                    tail_exceed(interp[:half], np.full(n_pts, t), direction)
                    - tail_exceed(interp[half:], np.full(n_pts, t), direction))))
                e = float(np.max(np.abs(fn(np.float64(t)) - truth)))
                ae.append(e / nz if nz else 0.0)
            for t in local:
                truth = tail_exceed(interp, t, direction)
                nz = float(np.max(np.abs(
                    tail_exceed(interp[:half], t, direction)
                    - tail_exceed(interp[half:], t, direction))))
                e = float(np.max(np.abs(fn(t) - truth)))
                le.append(e / nz if nz else 0.0)
            print(f"{label:<26} {len(fields):>6} {mb:8.3f} {base_mb / mb:7.2f}x | "
                  + " ".join(f"{v:6.2f}" for v in ae) + " | "
                  + " ".join(f"{v:6.2f}" for v in le))
        print()
        # saturation guard: show the worst local-percentile point in absolute terms so a
        # ratio blow-up cannot hide behind a tiny denominator
        for t in local:
            truth = tail_exceed(interp, t, direction)
            for label, _, fn in encodings:
                est = fn(t)
                d = np.abs(est - truth)
                j = int(np.argmax(d))
                if d[j] > 0.5:
                    print(f"  [saturation?] {label}: point {j} t={t[j]:.4f} truth={truth[j]:.4f} "
                          f"est={est[j]:.4f} |err|={d[j]:.4f}")
        print("  LOCAL percentile thresholds, ABSOLUTE probability error (the ratio above can")
        print("  blow up when both truth and estimate are near zero -- judge on these):")
        print(f"  {'encoding':<26} " + " ".join(f"P{q:>4.0f}: truth/err/nz"
                                             for q in pctl))
        for label, fields, fn in encodings:
            cells = []
            for t in local:
                truth = tail_exceed(interp, t, direction)
                nz = float(np.max(np.abs(
                    tail_exceed(interp[:half], t, direction)
                    - tail_exceed(interp[half:], t, direction))))
                e = float(np.max(np.abs(fn(t) - truth)))
                cells.append(f"{truth.mean():.4f}/{e:.4f}/{nz:.4f}")
            print(f"  {label:<26} " + "  ".join(cells))
        print()
        del stack, interp
        gc.collect()


if __name__ == "__main__":
    main()
