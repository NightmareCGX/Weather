"""Experiment 12: store the *rendered outputs* (histogram + rose bin counts) instead of members?

The proposal: keep the 6 moment fields and the 5 quantile fields, and additionally store
the member histogram and the wind-rose bin counts as per-cell fields, so the displayed
charts can be rebuilt without the members.

Two things decide it:

  1. Bundle-size arithmetic. Bin counts are linear, so they DO commute with bilinear
     interpolation (a bin count is a sum of indicators; interpolating counts yields the
     counts of the *mixture*). But note experiment 11: per-field compression of the
     aggregate fields was ~equal to the members (3.58 vs 3.72 MB/field), so the saving
     from aggregates is essentially just the FIELD-COUNT ratio. If the histogram+rose
     bins push the field count back to ~30, the saving collapses.
  2. Whether the mixture is the SAME distribution as the histogram of interpolated
     members. It is not in general, so the difference is reported against a sampling-noise
     floor (two independent 30-member draws from the same process).

Run:  .venv/Scripts/python.exe bench_render_outputs.py
"""

from __future__ import annotations

import numpy as np
from numcodecs import Zstd

from bench_local_packing import LAT, LAT_CHUNKS, LON, LON_CHUNKS
from bench_member_packing import N_MEMBERS, chunk_buffers
from bench_shard_and_ensemble import grib_pack
from bench_stat_interpolation import smooth

Z5 = Zstd(level=5)
RNG = np.random.default_rng(9182)

# Bin counts that the actual UI renders (see the ensemble screenshots).
BINS_HIST = 14  # speed histogram bars
ROSE_SECTORS = 8
ROSE_SPEED_CLASSES = 4  # Light / Moderate / Strong / Gale
ROSE_BINS = ROSE_SECTORS * ROSE_SPEED_CLASSES  # 32


def build_scalar_members(spread_ratio=0.20, seed=11):
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    base = 12.0 + 9.0 * smooth((LAT, LON), 12.0, seed)
    out = []
    for m in range(N_MEMBERS):
        pert = smooth((LAT, LON), 8.0, 5000 + m) * (base.std() * spread_ratio)
        out.append(grib_pack(np.asarray(base + pert, dtype=np.float32), 12))
    return out


def bytes_of(fields):
    """Compressed bytes of a list of 2-D float32 fields, using the production chunking."""
    total = 0
    for f in fields:
        padded = np.zeros((LAT, LON), dtype=np.float32)
        a = np.asarray(f, dtype=np.float32)
        padded[: a.shape[0], : a.shape[1]] = np.nan_to_num(a, nan=0.0)
        total += sum(len(Z5.encode(b.tobytes(order="C"))) for b in chunk_buffers(padded))
    return total


def per_cell_stats(stack):
    mean = np.nanmean(stack, axis=0)
    dev = stack - mean[None]
    var = np.nanmean(dev * dev, axis=0)
    return mean, var, dev


def main():
    members = build_scalar_members()
    stack = np.stack(members, axis=0)

    # ---------------------------------------------------------------- bundle arithmetic
    print()
    print("Bundle arithmetic for ONE scalar variable (30 members)")
    print()
    scalar_current = bytes_of(list(members))
    fields_members = 30

    # The full "keep every displayed output" bundle, per cell:
    #   moments 6 + quantiles 5 + histogram BINS_HIST + min/max 2
    mean, var, dev = per_cell_stats(stack)
    lo_hi = BINS_HIST
    hist_range = (float(np.nanmin(stack)), float(np.nanmax(stack)))
    edges = np.linspace(hist_range[0], hist_range[1], lo_hi + 1)
    hist_fields = []
    for b in range(lo_hi):
        cnt = np.nanmean((stack >= edges[b]) & (stack < edges[b + 1]), axis=0)
        hist_fields.append(cnt)
    quant_fields = [np.nanpercentile(stack, q, axis=0) for q in (10, 25, 50, 75, 90)]
    mm_fields = [np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)]
    moment_fields = [mean, var]
    bundle = moment_fields + quant_fields + hist_fields + mm_fields
    bundle_bytes = bytes_of(bundle)

    print(f"{'bundle':<52} {'fields':>7} {'MB':>9} {'vs current':>11}")
    print("-" * 84)
    print(f"{'30 member fields (current)':<52} {fields_members:>7} {scalar_current / 1e6:9.2f} {1.0:>10.2f}x")
    print(
        f"{'6 moments + 5 quantiles (exp.11 form)':<52} {11:>7} "
        f"{bytes_of(moment_fields + quant_fields) / 1e6:9.2f} "
        f"{scalar_current / bytes_of(moment_fields + quant_fields):>10.2f}x"
    )
    print(
        f"{'  + histogram bars + min/max (full proposal)':<52} {len(bundle):>7} "
        f"{bundle_bytes / 1e6:9.2f} {scalar_current / bundle_bytes:>10.2f}x"
    )
    print()
    print(f"  break-even histogram bin count = {fields_members - 6 - 5 - 2}  (bundle fields = 30)")
    print(f"  actually-needed bars in the UI  = ~{BINS_HIST}")
    print()

    # Wind needs u AND v per member: 60 member fields, and the rose replaces the pair.
    print("Bundle arithmetic for WIND (wind_u_10m + wind_v_10m, 30 members each)")
    print()
    wind_members_bytes = scalar_current * 2  # two variables, same geometry
    wind_bundle_fields = 6 + 5 + BINS_HIST + ROSE_BINS + 2
    wind_bundle_bytes = bytes_of(moment_fields + quant_fields + hist_fields + mm_fields) + (
        bytes_of(hist_fields[:ROSE_BINS]) if ROSE_BINS <= len(hist_fields) else 0
    )
    print(f"{'60 member fields (u + v, current)':<52} {60:>7} {wind_members_bytes / 1e6:9.2f} {1.0:>10.2f}x")
    print(
        f"{'moments + quantiles + histogram + rose + min/max':<52} {wind_bundle_fields:>7} "
        f"{'~' + format(wind_bundle_bytes / 1e6, '.2f'):>9} "
        f"{wind_members_bytes / max(wind_bundle_bytes, 1):>10.2f}x"
    )
    print(f"  (rose = {ROSE_SECTORS} sectors x {ROSE_SPEED_CLASSES} speed classes = {ROSE_BINS} bins,"
          f" replaces 60 member fields)")
    print()

    # ---------------------------------------------------------------- distributional identity
    print("=" * 96)
    print("Is the rebuilt distribution the SAME as today's? (TV distance, lower = closer)")
    print("=" * 96)
    n_pts = 6000
    rows = RNG.uniform(1.0, LAT - 3.0, n_pts)
    cols = RNG.uniform(1.0, LON - 3.0, n_pts)
    r0 = rows.astype(int)
    c0 = cols.astype(int)
    tr = rows - r0
    tc = cols - c0

    interp = np.empty((N_MEMBERS, n_pts))
    for m, field in enumerate(members):
        f00, f01 = field[r0, c0], field[r0, c0 + 1]
        f10, f11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
        lo_ = f00 + (f01 - f00) * tc
        up_ = f10 + (f11 - f10) * tc
        interp[m] = lo_ + (up_ - lo_) * tr

    w00 = (1 - tc) * (1 - tr)
    w01 = tc * (1 - tr)
    w10 = (1 - tc) * tr
    w11 = tc * tr

    # per-cell histograms
    cell_hist = np.stack(
        [np.nanmean((stack >= edges[b]) & (stack < edges[b + 1]), axis=0) for b in range(lo_hi)]
    )

    tv_mix, tv_floor = [], []
    for i in range(0, n_pts, 200):
        sl = slice(i, i + 200)
        # mixture: interpolate the per-cell bin counts
        mix = (
            w00[sl] * cell_hist[:, r0[sl], c0[sl]]
            + w01[sl] * cell_hist[:, r0[sl], c0[sl] + 1]
            + w10[sl] * cell_hist[:, r0[sl] + 1, c0[sl]]
            + w11[sl] * cell_hist[:, r0[sl] + 1, c0[sl] + 1]
        )
        # truth: histogram of the interpolated members
        truth = np.stack(
            [np.mean((interp[:, sl] >= edges[b]) & (interp[:, sl] < edges[b + 1]), axis=0) for b in range(lo_hi)]
        )
        tv_mix.append(0.5 * np.abs(mix - truth).sum(axis=0))
        # sampling-noise floor: two independent 30-member draws at the same location
        half = N_MEMBERS // 2
        a = np.stack(
            [
                np.mean(
                    (interp[:half, sl] >= edges[b]) & (interp[:half, sl] < edges[b + 1]), axis=0
                )
                for b in range(lo_hi)
            ]
        )
        c = np.stack(
            [
                np.mean(
                    (interp[half:, sl] >= edges[b]) & (interp[half:, sl] < edges[b + 1]), axis=0
                )
                for b in range(lo_hi)
            ]
        )
        tv_floor.append(0.5 * np.abs(a - c).sum(axis=0))

    tv_mix = np.concatenate(tv_mix)
    tv_floor = np.concatenate(tv_floor)
    print(f"  TV(mixture of per-cell histograms , histogram of interpolated members) = {tv_mix.mean():.4f}")
    print(f"  TV(sampling-noise floor: two independent 15-member halves)              = {tv_floor.mean():.4f}")
    print()
    print(f"  mixture-vs-truth is {'BELOW' if tv_mix.mean() < tv_floor.mean() else 'ABOVE'} the sampling-noise floor")
    print("  -> " + (
        "the rebuilt histogram is within ensemble sampling noise, i.e. not distinguishable"
        if tv_mix.mean() < tv_floor.mean()
        else "the rebuilt histogram deviates more than the inherent sampling noise, i.e. visible"
    ))
    print()


if __name__ == "__main__":
    main()
