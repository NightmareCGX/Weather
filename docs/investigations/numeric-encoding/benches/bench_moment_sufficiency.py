"""Experiment 11: exact moments from local covariance fields -- how few fields replace 30 members?

Section 9.4 showed that storing the per-cell statistic and interpolating it does NOT
reproduce the point-path statistic, because bilinear interpolation mixes the 2x2
neighbourhood. That is unavoidable for ORDER statistics: the quantile at an interpolated
point depends on the joint distribution (copula) of the 2x2 member values, not just the
per-cell marginals, so no set of per-cell scalars is sufficient.

But moments are different. For weights w over the 4 cells,

    Var_m( sum_k w_k f_m(k) ) = sum_k sum_l w_k w_l Cov_m( f_m(k), f_m(l) )

which needs only the *pairwise* covariances between cells -- not the full member sample.
So storing, per cell, the mean plus the local second-moment structure (variance and the
four distinct neighbour covariances) should reproduce the interpolated mean and variance
EXACTLY, while the per-cell-variance-only approach cannot.

Fields needed (6 instead of 30 members):
    MEAN(i,j)   = mean_m f_m(i,j)
    VAR(i,j)    = Var_m  f_m(i,j)
    COV_H(i,j)  = Cov_m( f_m(i,j), f_m(i,j+1) )
    COV_V(i,j)  = Cov_m( f_m(i,j), f_m(i+1,j) )
    COV_D(i,j)  = Cov_m( f_m(i,j), f_m(i+1,j+1) )
    COV_A(i,j)  = Cov_m( f_m(i,j), f_m(i+1,j-1) )   [anti-diagonal]

This measures whether they are exact, and how the 6 aggregate fields compress relative to
the 30 member fields (they are smoother, so the effective saving may beat 6/30).

Run:  .venv/Scripts/python.exe bench_moment_sufficiency.py
"""

from __future__ import annotations

import numpy as np
from numcodecs import Zstd

from bench_local_packing import CHUNK, LAT, LAT_CHUNKS, LON, LON_CHUNKS
from bench_member_packing import N_MEMBERS, chunk_buffers
from bench_shard_and_ensemble import grib_pack
from bench_stat_interpolation import smooth

Z5 = Zstd(level=5)
RNG = np.random.default_rng(31337)


def build_members(corr=8.0, spread_ratio=0.20, with_nan=False):
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    base = 12.0 + 9.0 * smooth((LAT, LON), 12.0, 11)
    members = []
    for m in range(N_MEMBERS):
        pert = smooth((LAT, LON), corr, 4000 + m) * (base.std() * spread_ratio)
        f = np.asarray(base + pert, dtype=np.float32)
        if with_nan:
            # a member missing for a whole (member, lead) unit -> NaN for every cell,
            # which is how the reader represents an uncommitted member.
            if m % 7 == 0:
                f = np.full((LAT, LON), np.nan, dtype=np.float32)
        members.append(f)
    return members


def aggregate_fields(stack):
    """Per-cell mean + local second moments, computed with NaN-aware member reduction."""
    mean = np.nanmean(stack, axis=0)
    dev = stack - mean[None, :, :]
    var = np.nanmean(dev * dev, axis=0)

    def cov(shift_r, shift_c):
        """Cov_m(f(i,j), f(i+shift_r, j+shift_c)) on the overlapping grid."""
        a = dev[:, : LAT - shift_r, : LON - abs(shift_c)]
        b = dev[:, shift_r:, abs(shift_c) :]
        return np.nanmean(a * b, axis=0)

    return {
        "MEAN": mean,
        "VAR": var,
        "COV_H": cov(0, 1),
        "COV_V": cov(1, 0),
        "COV_D": cov(1, 1),
        # Cov_m(f(i,j), f(i+1,j-1)) for j >= 1
        "COV_A": np.nanmean(dev[:, 1:, :-1] * dev[:, :-1, 1:], axis=0),
    }


def main():
    n_pts = 4000
    rows = RNG.uniform(1.0, LAT - 3.0, n_pts)
    cols = RNG.uniform(2.0, LON - 3.0, n_pts)

    print()
    print("Can 6 aggregate fields replace 30 members for mean and std at a query point?")
    print()
    for label, with_nan in (("all 30 members present", False), ("with uncommitted members (NaN)", True)):
        members = build_members(with_nan=with_nan)
        stack = np.stack(members, axis=0)

        # ground truth: interpolate each member, then compute the statistic (current path)
        r0 = rows.astype(int)
        c0 = cols.astype(int)
        tr = rows - r0
        tc = cols - c0
        interp = np.empty((N_MEMBERS, n_pts))
        for m, field in enumerate(members):
            f00, f01 = field[r0, c0], field[r0, c0 + 1]
            f10, f11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
            lo = f00 + (f01 - f00) * tc
            up = f10 + (f11 - f10) * tc
            interp[m] = lo + (up - lo) * tr
        true_mean = np.nanmean(interp, axis=0)
        true_std = np.nanstd(interp, axis=0)

        agg = aggregate_fields(stack)

        # --- approach A: store per-cell mean / std, interpolate them
        def bilinear(field):
            g00, g01 = field[r0, c0], field[r0, c0 + 1]
            g10, g11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
            lo = g00 + (g01 - g00) * tc
            up = g10 + (g11 - g10) * tc
            return lo + (up - lo) * tr

        # --- approach B: reconstruct Var at the point from stored local covariances
        w00 = (1 - tc) * (1 - tr)
        w01 = tc * (1 - tr)
        w10 = (1 - tc) * tr
        w11 = tc * tr
        V = agg["VAR"]
        var_at_p = (
            w00**2 * V[r0, c0]
            + w01**2 * V[r0, c0 + 1]
            + w10**2 * V[r0 + 1, c0]
            + w11**2 * V[r0 + 1, c0 + 1]
            + 2 * w00 * w01 * agg["COV_H"][r0, c0]
            + 2 * w00 * w10 * agg["COV_V"][r0, c0]
            + 2 * w00 * w11 * agg["COV_D"][r0, c0]
            + 2 * w01 * w10 * agg["COV_A"][r0, c0]
            + 2 * w01 * w11 * agg["COV_V"][r0, c0 + 1]
            + 2 * w10 * w11 * agg["COV_H"][r0 + 1, c0]
        )
        std_b = np.sqrt(np.maximum(var_at_p, 0.0))

        print("-" * 92)
        print(f"{label}")
        print("-" * 92)
        print(f"  mean  | per-cell mean interpolated     err = {np.nanmax(np.abs(bilinear(agg['MEAN']) - true_mean)):.3e}")
        print(f"  std   | per-cell std interpolated (1 field)  err = {np.nanmax(np.abs(bilinear(np.sqrt(np.maximum(agg['VAR'],0))) - true_std)):.3e}")
        print(f"  std   | local covariance fields  (6 fields) err = {np.nanmax(np.abs(std_b - true_std)):.3e}")
        rel = np.nanmax(np.abs(std_b - true_std) / np.where(true_std > 0, true_std, 1))
        print(f"        relative error of the 6-field reconstruction = {rel:.3e}")
        print()

    # -------------------------------------------------------------------------------
    print("=" * 92)
    print("Does the saving survive compression? (bytes per (variable, lead), GRIB-12 fields)")
    print("=" * 92)
    members = build_members()
    member_bytes = sum(
        len(Z5.encode(b.tobytes(order="C"))) for m in members for b in chunk_buffers(m)
    )
    agg = aggregate_fields(np.stack(members, axis=0))
    agg_bytes = 0
    for name, field in agg.items():
        f = np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0)
        padded = np.zeros((LAT, LON), dtype=np.float32)
        padded[: f.shape[0], : f.shape[1]] = f
        agg_bytes += sum(len(Z5.encode(b.tobytes(order="C"))) for b in chunk_buffers(padded))

    print(f"  30 member fields      : {member_bytes / 1e6:8.2f} MB   (1.000x)")
    print(f"  6 aggregate fields    : {agg_bytes / 1e6:8.2f} MB   ({member_bytes / agg_bytes:.3f}x)")
    print(f"  field-count ratio     : {6 / 30:.3f}x")
    print()
    print("  The aggregate fields are smoother than the members, so the byte ratio can beat")
    print("  the field-count ratio. Only the MEAN and the STD are reproduced by these six;")
    print("  quantiles, min/max, the wind rose and the member histogram are NOT.")
    print()


if __name__ == "__main__":
    main()
