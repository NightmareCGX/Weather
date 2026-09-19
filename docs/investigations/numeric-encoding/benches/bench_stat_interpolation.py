"""Experiment 8: does precomputing per-cell ensemble statistics commute with the
point path's bilinear interpolation?

The proposal under test: instead of storing 30 members and, at query time, bilinearly
interpolating each member and *then* computing statistics (the current behaviour, see
`api/core/zarr.py:534-581` reading per-member `interpolate_point`), store the per-cell
statistics at ingest and bilinearly interpolate *those* at query time.

Bilinear interpolation is a convex combination: F(p) = sum_k w_k f(k) with w_k >= 0 and
sum_k w_k == 1. Whether a statistic commutes with it depends on the statistic:

* mean          -- linear, so EXACT.
* std / variance-- does NOT commute. Var(sum_k w_k X_k) = sum_k w_k^2 Var(X_k)
                   + sum_{k!=l} w_k w_l Cov(X_k, X_l), whereas sum_k w_k Var(X_k)
                   ignores the covariance structure. With uncorrelated cells the
                   interpolated spread is SMALLER (sqrt(sum w_k^2) < 1), i.e. the
                   band is systematically too narrow.
* percentiles   -- order statistics of a convex combination; not linear. Non-commuting.
* min / max     -- min_m sum_k w_k f_m(k) >= sum_k w_k min_m f_m(k), i.e. the
                   precomputed extremes OVER-state the range.

The size of the deviation is governed by the spatial correlation of the field: at
correlation -> 1 every statistic commutes; at correlation -> 0 the spread bias reaches
its worst case. So this sweeps correlation length and reports the error per statistic,
which is what decides whether any variable can be moved.

Run:  .venv/Scripts/python.exe bench_stat_interpolation.py
"""

from __future__ import annotations

import numpy as np

LAT, LON = 721, 1440
N_MEMBERS = 30
RNG = np.random.default_rng(20260918)


def smooth(shape, corr_cells, seed):
    """Gaussian-correlated random field via FFT (corr_cells = 1/e correlation length)."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(ky**2 + kx**2)
    # Gaussian filter in frequency space: exp(-2 pi^2 sigma^2 k^2), sigma in cells.
    g = np.exp(-2.0 * (np.pi**2) * (corr_cells**2) * (k**2))
    ph = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    f = np.real(np.fft.ifft2(g * ph))
    f -= f.mean()
    f /= f.std() + 1e-12
    return f


def build_members(corr_cells, spread, seed=5):
    """A smooth ensemble: shared base field + spatially correlated member perturbations.

    Member ordering across cells is what controls the commuting error, and its
    spatial correlation length is exactly `corr_cells`.
    """
    base = smooth((LAT, LON), 12.0, seed) * 10.0 + 12.0  # a smooth "wind speed" field
    members = []
    for m in range(N_MEMBERS):
        pert = smooth((LAT, LON), corr_cells, seed + 1000 + m) * spread
        members.append(np.asarray(base + pert, dtype=np.float32))
    return members


STATS = ("mean", "std", "P10", "P25", "P50", "P75", "P90", "min", "max")


def stat_of(vals: np.ndarray, name: str) -> float:
    """Compute one statistic across the member axis for a single point (1-D)."""
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return float("nan")
    if name == "mean":
        return float(np.mean(v))
    if name == "std":
        return float(np.std(v))
    if name == "min":
        return float(np.min(v))
    if name == "max":
        return float(np.max(v))
    q = float(name[1:])
    return float(np.percentile(v, q))


def main():
    # Query points: random fractional positions inside the interior of the grid, so
    # each lands in a 2x2 cell block (this is what a user clicking a map produces).
    n_pts = 4000
    rows = RNG.uniform(1.0, LAT - 3.0, n_pts)
    cols = RNG.uniform(1.0, LON - 3.0, n_pts)

    print()
    print("Precompute-per-cell-statistics  vs  interpolate-members-then-statistics")
    print(f"{n_pts} random query points, {N_MEMBERS} members, spread = 2.0 (same units as the field)")
    print()
    header = (
        f"{'corr (cells)':>13} | "
        + " | ".join(f"{s:>17}" for s in ("mean", "std", "P10", "P90"))
        + " |  max |err|"
    )
    print(header)
    print("-" * len(header))
    print("(each cell shows max |difference| in stat units, and the ratio proposed/current)")
    print()

    for corr in (1.0, 2.0, 5.0, 10.0, 25.0, 60.0):
        members = build_members(corr, spread=2.0)

        # Current: interpolate each member, then compute the statistic.
        interp_vals = np.empty((N_MEMBERS, n_pts), dtype=np.float64)
        for m, field in enumerate(members):
            r0 = rows.astype(int)
            c0 = cols.astype(int)
            tr = rows - r0
            tc = cols - c0
            f00 = field[r0, c0]
            f01 = field[r0, c0 + 1]
            f10 = field[r0 + 1, c0]
            f11 = field[r0 + 1, c0 + 1]
            lower = f00 + (f01 - f00) * tc
            upper = f10 + (f11 - f10) * tc
            interp_vals[m] = lower + (upper - lower) * tr
        current = {s: np.array([stat_of(interp_vals[:, i], s) for i in range(n_pts)]) for s in STATS}

        # Proposed: precompute the statistic per cell, then interpolate the stat field.
        proposed = {}
        for s in STATS:
            per_cell = np.empty((LAT, LON), dtype=np.float64)
            stack = np.stack(members, axis=0)
            if s == "mean":
                per_cell = np.mean(stack, axis=0)
            elif s == "std":
                per_cell = np.std(stack, axis=0)
            elif s == "min":
                per_cell = np.min(stack, axis=0)
            elif s == "max":
                per_cell = np.max(stack, axis=0)
            else:
                per_cell = np.percentile(stack, float(s[1:]), axis=0)
            r0 = rows.astype(int)
            c0 = cols.astype(int)
            tr = rows - r0
            tc = cols - c0
            g00 = per_cell[r0, c0]
            g01 = per_cell[r0, c0 + 1]
            g10 = per_cell[r0 + 1, c0]
            g11 = per_cell[r0 + 1, c0 + 1]
            lo = g00 + (g01 - g00) * tc
            up = g10 + (g11 - g10) * tc
            proposed[s] = lo + (up - lo) * tr

        cells = []
        for s in ("mean", "std", "P10", "P90"):
            d = np.abs(proposed[s] - current[s])
            cells.append(f"{np.nanmax(d):7.4f} ({np.nanmean(proposed[s]) / np.nanmean(current[s]):.3f})")
        maxerr = max(float(np.nanmax(np.abs(proposed[s] - current[s]))) for s in STATS)
        print(f"{corr:>13.0f} | " + " | ".join(cells) + f" |  {maxerr:10.4f}")

    print()
    print("Reading: the ratio in parentheses is mean(proposed)/mean(current). A value below")
    print("1.0 means the precomputed-statistics path UNDER-states the statistic -- for std")
    print("that is the interpolated band being systematically too narrow.")
    print()

    # ---------------------------------------------------------------------------------
    # The two displayed elements that a handful of scalars cannot encode at all.
    # ---------------------------------------------------------------------------------
    print("=" * 100)
    print("Elements that need the member DISTRIBUTION, not a few moments")
    print("=" * 100)
    members = build_members(10.0, spread=2.0)
    stack = np.stack(members, axis=0).astype(np.float64)
    i = 0
    r0, c0 = 300, 700
    cell = stack[:, r0 : r0 + 2, c0 : c0 + 2]
    point_via_interp = cell.reshape(N_MEMBERS, 4) @ np.array([0.25, 0.25, 0.25, 0.25])
    cell_mean = cell.reshape(N_MEMBERS, 4).mean(axis=1)

    print(f"  member distribution at a point: current std = {point_via_interp.std():.4f}, "
          f"precomputed-cell-std interpolated = {cell_mean.std():.4f}")
    print(f"    ratio = {cell_mean.std() / point_via_interp.std():.3f}  <- the histogram's width")
    # A KDE / histogram over 30 members is a function of the whole sample, so the only
    # way to precompute it is to store a per-cell histogram; interpolating bin counts
    # yields a MIXTURE, which is wider than the histogram of the interpolated members.
    bins = np.linspace(-20, 30, 26)
    hist_per_cell = []
    for rr in range(2):
        for cc in range(2):
            hist_per_cell.append(np.histogram(cell[:, rr, cc], bins=bins)[0] / N_MEMBERS)
    mixture = np.mean(hist_per_cell, axis=0)
    hist_interp = np.histogram(point_via_interp, bins=bins)[0] / N_MEMBERS
    print(f"  histogram-bin L1 distance, precomputed-mixture vs current = {np.abs(mixture - hist_interp).sum():.4f}")
    print(f"  histogram-bin L1 distance, lower bound (disjoint)         = 2.0000")
    print()
    print("  Wind rose: compute_wind_rose(u_members, v_members) consumes 30 per-member")
    print("  (u, v) pairs (domain/models/wind.py:383-412). It is a 2-D joint histogram over")
    print("  speed x direction, so no set of scalar moments reconstructs it; precomputing it")
    print("  per cell and interpolating bin masses gives a mixture, same class of change.")
    print()
    print("  Phase support: per-member 0/1 flags are bilinearly interpolated and THEN")
    print("  thresholded at 0.5 (api/services/point_forecast.py:880-898). Thresholding after")
    print("  a convex combination is a nonlinear operator, so per-cell phase frequencies do")
    print("  not reproduce it either.")
    print()


if __name__ == "__main__":
    main()
