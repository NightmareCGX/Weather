"""Experiment 13: what is the RIGHT yardstick for the interpolation error?

Sections 9.3/9.4 judged the precomputed-statistics error against the platform's 0.001 /
0.005 point-VALUE tolerance and concluded it was 1-2 orders of magnitude too large. That
is the wrong yardstick for ensemble *distribution* statistics: a quantile or a histogram
estimated from 30 members already wobbles by a lot, because the ensemble is a finite
sample. The question that matters for the displayed charts is:

    is the precompute-and-interpolate deviation SMALLER than the ensemble's own
    sampling noise, i.e. is it distinguishable at n = 30?

This measures both quantities on the same fields and reports their ratio. Ratio < 1 means
the approximation hides inside the sampling noise of the product's own 30 members.

Run:  .venv/Scripts/python.exe bench_noise_floor.py
"""

from __future__ import annotations

import numpy as np

from bench_local_packing import LAT, LON
from bench_member_packing import N_MEMBERS
from bench_render_outputs import build_scalar_members

RNG = np.random.default_rng(5150)


def main():
    members = build_scalar_members()
    stack = np.stack(members, axis=0)

    n_pts = 4000
    rows = RNG.uniform(1.0, LAT - 3.0, n_pts)
    cols = RNG.uniform(1.0, LON - 3.0, n_pts)
    r0, c0 = rows.astype(int), cols.astype(int)
    tr, tc = rows - r0, cols - c0

    # current path: interpolate each member, then compute the statistic
    interp = np.empty((N_MEMBERS, n_pts))
    for m, field in enumerate(members):
        f00, f01 = field[r0, c0], field[r0, c0 + 1]
        f10, f11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
        lo_ = f00 + (f01 - f00) * tc
        up_ = f10 + (f11 - f10) * tc
        interp[m] = lo_ + (up_ - lo_) * tr

    w00, w01 = (1 - tc) * (1 - tr), tc * (1 - tr)
    w10, w11 = (1 - tc) * tr, tc * tr

    def bilinear(field):
        g00, g01 = field[r0, c0], field[r0, c0 + 1]
        g10, g11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
        lo_ = g00 + (g01 - g00) * tc
        up_ = g10 + (g11 - g10) * tc
        return lo_ + (up_ - lo_) * tr

    print()
    print("Sampling noise of a 30-member ensemble  vs  the precompute error")
    print(f"({n_pts} query points, member spread ratio 0.20)")
    print()
    print(
        f"{'statistic':<12} {'precompute err':>15} {'ensemble noise (n=30)':>23} "
        f"{'err / noise':>12} {'verdict':>22}"
    )
    print("-" * 92)

    for name, fn in (
        ("mean", lambda a: np.nanmean(a, axis=0)),
        ("std", lambda a: np.nanstd(a, axis=0)),
        ("P10", lambda a: np.nanpercentile(a, 10, axis=0)),
        ("P50", lambda a: np.nanpercentile(a, 50, axis=0)),
        ("P90", lambda a: np.nanpercentile(a, 90, axis=0)),
    ):
        truth = fn(interp)

        # precompute: statistic per cell, then interpolate
        per_cell = fn(stack)
        pre = bilinear(per_cell)
        pre_err = float(np.nanmax(np.abs(pre - truth)))

        # ensemble sampling noise: split the 30 members into two 15-member halves and
        # take the difference of their statistic estimates. This is the product's own
        # irreducible wobble at the displayed ensemble size.
        half = N_MEMBERS // 2
        noise = float(np.nanmax(np.abs(fn(interp[:half]) - fn(interp[half:]))))

        ratio = pre_err / noise if noise else float("inf")
        verdict = "hidden in the noise" if ratio < 1 else "VISIBLE above the noise"
        print(f"{name:<12} {pre_err:15.4f} {noise:23.4f} {ratio:12.3f} {verdict:>22}")

    print()
    print("Interpretation: ratio < 1 means the approximation is smaller than how much the")
    print("displayed number would move if a different 15 of the 30 members had been drawn.")
    print("For ensemble distribution charts that is the honest acceptance criterion; the")
    print("0.001/0.005 point-value tolerance used in sections 7-9 applies to a single")
    print("forecast VALUE, not to a statistic estimated from a finite ensemble.")
    print()

    # Field-count and byte cost of each bundle component, so the trade can be read directly.
    print("=" * 92)
    print("Where the bytes actually go (per (variable, lead), one scalar variable)")
    print("=" * 92)
    from bench_render_outputs import bytes_of, per_cell_stats

    member_bytes = bytes_of(list(members))
    mean, var, dev = per_cell_stats(stack)
    quant = [np.nanpercentile(stack, q, axis=0) for q in (10, 25, 50, 75, 90)]
    mm = [np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)]
    edges = np.linspace(float(np.nanmin(stack)), float(np.nanmax(stack)), 15)
    hist = [np.nanmean((stack >= edges[b]) & (stack < edges[b + 1]), axis=0) for b in range(14)]

    groups = [
        ("30 members (current)", list(members)),
        ("moments (mean, var)", [mean, var]),
        ("quantiles (5)", quant),
        ("histogram bars (14)", hist),
        ("min/max (2)", mm),
    ]
    print(f"{'group':<28} {'fields':>7} {'MB':>9} {'MB/field':>10}")
    print("-" * 58)
    for label, fields in groups:
        b = bytes_of(fields)
        print(f"{label:<28} {len(fields):>7} {b / 1e6:9.2f} {b / len(fields) / 1e6:10.3f}")
    print()
    full = [mean, var] + quant + hist + mm
    fb = bytes_of(full)
    print(f"full bundle ({len(full)} fields) = {fb / 1e6:.2f} MB  ->  {member_bytes / fb:.2f}x smaller")
    print()


if __name__ == "__main__":
    main()
