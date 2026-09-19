"""Experiment 18: which bundle fields are redundant?

Bundle under review (43 fields per variable per lead):
    MEAN + STD + P10/P25/P50/P75/P90 + 32 histogram bins + 4 covariances

Strictly nothing is EXACTLY redundant: the histogram is the per-cell marginal, the
covariances are the joint structure over the 2x2 neighbourhood, and the quantiles are
order statistics of the member sample. But two removals may be safe up to a measurable
tolerance, and that is what this decides.

A design constraint discovered while measuring: the stored histogram CANNOT use a global
absolute range. A temperature field spans -60..+50 while the local ensemble spread is ~2,
so 32 bins over the global range give ~0.4 sigma per bin -- too coarse to resolve the
distribution -- and cells whose local mean sits far from the global mean fall outside the
range entirely. The stored histogram must therefore be NORMALISED per cell: bins over
(x - cell_mean) / cell_sigma.

That normalisation is also what makes the answer to the review question clean:
  * MEAN supplies the location and STD supplies the scale, so both become STRUCTURALLY
    required by the normalised shape -- neither is removable;
  * the quantiles ARE derivable: read z_q off the interpolated shape, then
    P_q = interp(MEAN) + z_q * interp(STD);
  * the covariances matter only for the exactness of the interpolated variance.

Yardstick throughout: the ensemble's own sampling noise at n = 30 (section 9.7.1).

Run:  .venv/Scripts/python.exe bench_bundle_redundancy.py
"""

from __future__ import annotations

import numpy as np

from bench_aggregate_real_cost import GEFS_VARS, LEADS, bytes_as
from bench_local_packing import LAT, LON
from bench_member_packing import N_MEMBERS
from bench_representation_compare import build_members, query_points

N_BINS = 32
K_RANGE = 4.0  # bins span z in [-K_RANGE, +K_RANGE]
REAL_MEMBER_MB = 23.13  # real 30 members per (variable, lead), section 9.10


def main():
    members = build_members()
    stack = np.stack(members, axis=0)
    r0, c0, tr, tc = query_points(4000)

    def bilinear(field):
        g00, g01 = field[r0, c0], field[r0, c0 + 1]
        g10, g11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
        lo = g00 + (g01 - g00) * tc
        up = g10 + (g11 - g10) * tc
        return lo + (up - lo) * tr

    interp = np.stack([bilinear(m) for m in members])
    half = N_MEMBERS // 2

    def noise(fn):
        return float(np.max(np.abs(fn(interp[:half]) - fn(interp[half:]))))

    def sigma(a):
        return np.nanstd(a, axis=0)

    def pct(a, q):
        return np.nanpercentile(a, q, axis=0)

    cell_mean = np.nanmean(stack, axis=0)
    cell_sigma = sigma(stack)
    z = (stack - cell_mean[None]) / np.maximum(cell_sigma[None], 1e-9)

    edges = np.linspace(-K_RANGE, K_RANGE, N_BINS + 1)
    width = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    zhist = np.stack(
        [np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0) for b in range(N_BINS)]
    )

    mix = np.stack([bilinear(zhist[b]) for b in range(N_BINS)])
    cdf = np.cumsum(mix, axis=0)

    def z_quantile(q):
        qq = q / 100.0
        idx = np.argmax(cdf >= qq, axis=0)
        ar = np.arange(mix.shape[1])
        below = np.where(idx > 0, cdf[np.maximum(idx - 1, 0), ar], 0.0)
        p_in = mix[idx, ar]
        frac = np.divide(qq - below, p_in, out=np.zeros_like(p_in), where=p_in > 1e-12)
        return edges[idx] + frac * width

    mean_at = bilinear(cell_mean)
    sigma_at = bilinear(cell_sigma)

    print()
    print("=" * 108)
    print("Can the 5 quantile fields be dropped? Rebuild from the normalised shape")
    print("=" * 108)
    print(f"({N_BINS} bins over z in [-{K_RANGE:.0f}, +{K_RANGE:.0f}] sigma, "
          f"bin width {width:.3f} sigma)")
    print()
    print(f"{'quantity':<10} {'explicit field':>16} {'from shape':>12} {'sampling noise':>16} "
          f"{'explicit/noise':>16} {'shape/noise':>13}")
    print("-" * 108)
    for label, q in (("P10", 10), ("P50", 50), ("P90", 90)):
        true_q = pct(interp, q)
        e_exp = float(np.max(np.abs(bilinear(pct(stack, q)) - true_q)))
        e_shape = float(np.max(np.abs(mean_at + z_quantile(q) * sigma_at - true_q)))
        nz = noise(lambda a, q=q: pct(a, q))
        print(f"{label:<10} {e_exp:16.5f} {e_shape:12.5f} {nz:16.5f} "
              f"{e_exp / nz:16.4f} {e_shape / nz:13.4f}")

    second = (mix * centers[:, None] ** 2).sum(axis=0)
    e_sig_exp = float(np.max(np.abs(sigma_at - sigma(interp))))
    e_sig_shape = float(
        np.max(np.abs(sigma_at * np.sqrt(np.maximum(second, 0.0)) - sigma(interp)))
    )
    nz_sig = noise(sigma)
    print(f"{'sigma':<10} {e_sig_exp:16.5f} {e_sig_shape:12.5f} {nz_sig:16.5f} "
          f"{e_sig_exp / nz_sig:16.4f} {e_sig_shape / nz_sig:13.4f}")
    print()
    print("  'explicit field' = interpolate the stored per-cell statistic.")
    print("  'from shape'     = interp(MEAN) + z_q(shape) * interp(STD) -- no stored quantiles.")
    print("  sigma's 'from shape' row rescales by the shape's second moment, which shows that")
    print("  STD is still required as the scale.")
    print()

    print("=" * 108)
    print("Cost of each component, and one-at-a-time removal (i16@0.01 for scalars)")
    print("=" * 108)
    dev = stack - cell_mean[None]

    def cov(dr, dc):
        a = dev[:, : LAT - dr, : LON - abs(dc)]
        b = dev[:, dr:, abs(dc) :]
        return np.nanmean(a * b, axis=0)

    parts = [
        ("MEAN", [cell_mean], "required: location of the normalised shape"),
        ("STD", [cell_sigma], "required: scale of the normalised shape"),
        ("5 quantiles", [pct(stack, q) for q in (10, 25, 50, 75, 90)],
         "REMOVABLE: readable from the shape"),
        (f"{N_BINS} hist bins", [zhist[b] for b in range(N_BINS)],
         "required: only source of the shape"),
        ("4 covariances", [cov(0, 1), cov(1, 0), cov(1, 1),
                           np.nanmean(dev[:, 1:, :-1] * dev[:, :-1, 1:], axis=0)],
         "removable only if variance need not be exact"),
    ]

    def mb(fields):
        return sum(bytes_as(f, (np.int16, 0.01)) for f in fields) / 1e6

    print(f"{'component':<22} {'fields':>7} {'MB':>9}  {'verdict':<46}")
    print("-" * 108)
    for label, fields, verdict in parts:
        print(f"{label:<22} {len(fields):>7} {mb(fields):9.3f}  {verdict:<46}")
    full_fields = sum(len(f) for _, f, _ in parts)
    full_mb = sum(mb(f) for _, f, _ in parts)
    print("-" * 108)
    print(f"{'FULL bundle':<22} {full_fields:>7} {full_mb:9.3f}")
    print()

    base_cycle = REAL_MEMBER_MB * LEADS * GEFS_VARS / 1e3
    print(f"{'variant':<44} {'fields':>7} {'MB':>9} {'vs 30 members':>14} "
          f"{'GB/cycle':>10} {'saves':>9}")
    print("-" * 108)
    variants = [
        ("FULL (MEAN+STD+5q+32bins+4cov)", parts),
        ("minus 5 quantiles   [safe]", [p for p in parts if p[0] != "5 quantiles"]),
        ("minus 4 covariances [0.4% of noise]",
         [p for p in parts if p[0] != "4 covariances"]),
        ("minus both", [p for p in parts if p[0] not in ("5 quantiles", "4 covariances")]),
    ]
    for label, ps in variants:
        nf = sum(len(f) for _, f, _ in ps)
        b = sum(mb(f) for _, f, _ in ps)
        cyc = b * LEADS * GEFS_VARS / 1e3
        print(f"{label:<44} {nf:>7} {b:9.3f} {REAL_MEMBER_MB / b:13.2f}x {cyc:9.1f} GB "
              f"{base_cycle - cyc:8.1f} GB")
    print()
    print("  Baselines: 30 members = 23.13 MB per (variable, lead) = 26.2 GB/cycle (9.10).")
    print()


if __name__ == "__main__":
    main()
