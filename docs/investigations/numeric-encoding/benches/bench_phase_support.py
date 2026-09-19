"""Experiment 15: precipitation PHASE SUPPORT under the stored-aggregate bundle.

Section 9.7's bundle omitted phase support. It is the one displayed element whose
underlying data is categorical 0/1 flags -- the most spatially intermittent fields in the
platform -- so its interpolation fidelity cannot be assumed from the wind-rose result and
has to be measured.

The current ensemble path (api/services/ensemble_data.py:1262-1273) interpolates each
member's 0/1 flag bilinearly and THEN thresholds at 0.5, then
`aggregate_ensemble_phase_support` averages the member phase weights. The proposed path
stores the per-cell member fraction and interpolates that. Those are different operators,
so this measures the difference against the ensemble's own sampling noise -- the yardstick
established in section 9.7.1.

Run:  .venv/Scripts/python.exe bench_phase_support.py
"""

from __future__ import annotations

import numpy as np

from bench_local_packing import LAT, LON
from bench_member_packing import N_MEMBERS
from bench_render_outputs import bytes_of
from bench_stat_interpolation import smooth

RNG = np.random.default_rng(2718)


def flag_members(corr_cells, coverage=0.25, seed=31):
    """30 members of an intermittent 0/1 field (like crain/csnow/cfrzr/cicep).

    Built by thresholding a smooth field so the field is patchy with a tunable spatial
    correlation length -- precipitation flags are the least smooth fields in the system.
    """
    out = []
    for m in range(N_MEMBERS):
        f = smooth((LAT, LON), corr_cells, seed + 6000 + m)
        # shift the threshold so ~`coverage` of cells are wet
        thr = np.quantile(f, 1.0 - coverage)
        out.append((f > thr).astype(np.float32))
    return out


def main():
    n_pts = 3000
    print()
    print("Phase support: per-member threshold-then-average (current)  vs")
    print("per-cell fraction then interpolate (proposed)")
    print(f"({n_pts} query points, 30 members, ~25% wet coverage)")
    print()
    print(
        f"{'corr (cells)':>13} {'precompute err':>15} {'ensemble noise':>15} "
        f"{'err/noise':>11} {'verdict':>22}"
    )
    print("-" * 82)

    for corr in (1.0, 2.0, 4.0, 8.0, 16.0):
        members = flag_members(corr)
        stack = np.stack(members, axis=0)

        rows = RNG.uniform(1.0, LAT - 3.0, n_pts)
        cols = RNG.uniform(1.0, LON - 3.0, n_pts)
        r0, c0 = rows.astype(int), cols.astype(int)
        tr, tc = rows - r0, cols - c0

        # truth: interpolate each member's flag, threshold at 0.5, then average over members
        per_member = np.empty((N_MEMBERS, n_pts))
        for m, field in enumerate(members):
            f00, f01 = field[r0, c0], field[r0, c0 + 1]
            f10, f11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
            lo_ = f00 + (f01 - f00) * tc
            up_ = f10 + (f11 - f10) * tc
            per_member[m] = lo_ + (up_ - lo_) * tr
        truth = np.mean(per_member >= 0.5, axis=0)

        # proposed: per-cell member fraction, interpolated
        per_cell = np.mean(stack, axis=0)
        g00, g01 = per_cell[r0, c0], per_cell[r0, c0 + 1]
        g10, g11 = per_cell[r0 + 1, c0], per_cell[r0 + 1, c0 + 1]
        lo_ = g00 + (g01 - g00) * tc
        up_ = g10 + (g11 - g10) * tc
        proposed = lo_ + (up_ - lo_) * tr

        pre_err = float(np.max(np.abs(proposed - truth)))
        half = N_MEMBERS // 2
        noise = float(
            np.max(
                np.abs(
                    np.mean(per_member[:half] >= 0.5, axis=0)
                    - np.mean(per_member[half:] >= 0.5, axis=0)
                )
            )
        )
        ratio = pre_err / noise if noise else float("inf")
        verdict = "hidden in the noise" if ratio < 1 else "VISIBLE above the noise"
        print(f"{corr:>13.0f} {pre_err:15.4f} {noise:15.4f} {ratio:11.3f} {verdict:>22}")

    print()
    print("Flags are the most intermittent fields in the platform, so the low-correlation")
    print("rows are the realistic ones for convective precipitation; the high-correlation")
    print("rows bound the benign case (stratiform / large-scale coverage).")
    print()

    # storage cost of the support fields
    print("=" * 84)
    print("Storage cost of the phase-support fields (count-type, so cheap)")
    print("=" * 84)
    members = flag_members(2.0)
    stack = np.stack(members, axis=0)
    frac = np.mean(stack, axis=0)
    dry = 1.0 - frac
    fields = [frac, dry, frac * 0.0, frac * 0.0, frac * 0.0, frac * 0.0][:6]
    b = bytes_of(fields)
    print(f"  6 support fields (dry/rain/snow/fzra/icep/unknown) : {b / 1e6:.2f} MB "
          f"({b / 6 / 1e6:.3f} MB/field)")
    print(f"  for reference, 30 flag member fields              : "
          f"{bytes_of(list(members)) / 1e6:.2f} MB")
    print(f"  4 raw flag member fields is what the current store holds for these")
    print()


if __name__ == "__main__":
    main()
