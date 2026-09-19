"""Experiment 14: the full "store the rendered outputs" bundle, measured for WIND.

Experiment 12/13 found:
  * derived fields are NOT cheaper per field than members (quantiles cost 3.71 MB/field
    vs 2.33 for members) -- because members carry GRIB-packing zero low-mantissa bits
    while numpy-computed derived fields are full-entropy float32;
  * but BOUNDED COUNT fields (histogram bars, rose bins) cost 0.175 MB/field, 13x
    cheaper than members, because most bins are empty almost everywhere;
  * and the precompute error is 20-30x SMALLER than the ensemble's own sampling noise.

So keeping the histogram and the wind rose as stored count fields is nearly free, which
is what makes the "store outputs, not members" bundle work. This measures the wind case
end to end instead of modelling it: u and v each need 30 members, but the rose replaces
the whole (u, v) pair.

Run:  .venv/Scripts/python.exe bench_wind_bundle.py
"""

from __future__ import annotations

import numpy as np

from bench_local_packing import LAT, LON
from bench_member_packing import N_MEMBERS
from bench_render_outputs import bytes_of
from bench_shard_and_ensemble import grib_pack
from bench_stat_interpolation import smooth

ROSE_SECTORS = 8
ROSE_SPEED_CLASSES = 4
ROSE_BINS = ROSE_SECTORS * ROSE_SPEED_CLASSES
BINS_HIST = 14
RNG = np.random.default_rng(606)


def build_wind_members():
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    u_base = 6.0 + 5.0 * smooth((LAT, LON), 12.0, 71)
    v_base = -2.0 + 5.0 * smooth((LAT, LON), 12.0, 72)
    us, vs = [], []
    for m in range(N_MEMBERS):
        du = smooth((LAT, LON), 8.0, 7000 + m) * 2.0
        dv = smooth((LAT, LON), 8.0, 8000 + m) * 2.0
        us.append(grib_pack(np.asarray(u_base + du, dtype=np.float32), 12))
        vs.append(grib_pack(np.asarray(v_base + dv, dtype=np.float32), 12))
    return us, vs


def main():
    us, vs = build_wind_members()
    u_stack = np.stack(us, axis=0)
    v_stack = np.stack(vs, axis=0)
    speed = np.hypot(u_stack, v_stack)  # (30, LAT, LON)
    direction = (np.degrees(np.arctan2(-u_stack, -v_stack)) + 360.0) % 360.0  # met. direction

    current = bytes_of(list(us)) + bytes_of(list(vs))

    # --- bundle: everything the screenshots display, as stored per-cell fields
    mean_s = np.nanmean(speed, axis=0)
    std_s = np.nanstd(speed, axis=0)
    quant_s = [np.nanpercentile(speed, q, axis=0) for q in (10, 25, 50, 75, 90)]
    mm_s = [np.nanmin(speed, axis=0), np.nanmax(speed, axis=0)]

    edges = np.linspace(float(np.nanmin(speed)), float(np.nanmax(speed)), BINS_HIST + 1)
    hist_s = [
        np.nanmean((speed >= edges[b]) & (speed < edges[b + 1]), axis=0)
        for b in range(BINS_HIST)
    ]

    # Rose: 8 sectors x 4 speed classes, as member-probability fields in [0, 1].
    # Class edges mirror the UI legend: Light / Moderate / Strong / Gale.
    speed_edges = np.array([0.0, 20.0, 40.0, 60.0, 1e9])  # km/h bands as shown
    sector_idx = np.floor(direction / (360.0 / ROSE_SECTORS)).astype(int) % ROSE_SECTORS
    rose = []
    for s in range(ROSE_SECTORS):
        for c in range(ROSE_SPEED_CLASSES):
            in_bin = (sector_idx == s) & (speed >= speed_edges[c]) & (speed < speed_edges[c + 1])
            rose.append(np.nanmean(in_bin, axis=0))

    bundle = [mean_s, std_s] + quant_s + hist_s + mm_s + rose

    print()
    print("WIND: 30 members of u + 30 of v  vs  the stored-output bundle")
    print()
    print(f"{'field group':<34} {'fields':>7} {'MB':>9} {'MB/field':>10}")
    print("-" * 64)
    groups = [
        ("u members (current)", list(us)),
        ("v members (current)", list(vs)),
        ("speed moments (mean, std)", [mean_s, std_s]),
        ("speed quantiles (5)", quant_s),
        ("speed histogram bars (14)", hist_s),
        ("speed min/max (2)", mm_s),
        (f"wind rose ({ROSE_SECTORS}x{ROSE_SPEED_CLASSES}=32)", rose),
    ]
    for label, fields in groups:
        b = bytes_of(fields)
        print(f"{label:<34} {len(fields):>7} {b / 1e6:9.2f} {b / len(fields) / 1e6:10.3f}")
    print()
    bundle_bytes = bytes_of(bundle)
    print(f"  current (60 member fields)          : {current / 1e6:8.2f} MB  (1.00x)")
    print(f"  bundle  ({len(bundle)} fields)             : {bundle_bytes / 1e6:8.2f} MB  ({current / bundle_bytes:.2f}x smaller)")
    print()
    print("  The bundle keeps: speed mean/std/min/max, all five quantiles, the 14-bar")
    print("  histogram and the full 8x4 wind rose. What it cannot keep is the discrete")
    print("  per-member sample (the dots the chart plots) nor arbitrary-threshold")
    print("  exceedance probabilities beyond the fixed bin edges.")
    print()

    # --- fidelity: rebuild the rose probabilities at a point both ways.
    #
    # Truth follows the platform's method (api/core/zarr.py interpolate_members ->
    # domain/models/wind.py compute_wind_rose): bilinearly interpolate the u and v
    # COMPONENTS per member, then derive speed/direction and bin each member.
    #
    # Proposed: bilinearly interpolate the stored per-cell rose probabilities, which
    # yields the MIXTURE of the neighbouring cells' rose distributions.
    n_pts = 3000
    rows = RNG.uniform(1.0, LAT - 3.0, n_pts)
    cols = RNG.uniform(1.0, LON - 3.0, n_pts)
    r0, c0 = rows.astype(int), cols.astype(int)
    tr, tc = rows - r0, cols - c0
    w00, w01 = (1 - tc) * (1 - tr), tc * (1 - tr)
    w10, w11 = (1 - tc) * tr, tc * tr

    def bilinear(field):
        g00, g01 = field[r0, c0], field[r0, c0 + 1]
        g10, g11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
        lo_ = g00 + (g01 - g00) * tc
        up_ = g10 + (g11 - g10) * tc
        return lo_ + (up_ - lo_) * tr

    # per-member interpolated components at the points
    u_p = np.stack([bilinear(u_stack[m]) for m in range(N_MEMBERS)])  # (30, n_pts)
    v_p = np.stack([bilinear(v_stack[m]) for m in range(N_MEMBERS)])
    spd_p = np.hypot(u_p, v_p)
    dir_p = (np.degrees(np.arctan2(-u_p, -v_p)) + 360.0) % 360.0
    sec_p = np.floor(dir_p / (360.0 / ROSE_SECTORS)).astype(int) % ROSE_SECTORS

    def rose_truth(mask_members, sector, cls):
        """Fraction of the given members landing in (sector, speed class) at each point."""
        sel = (sec_p[mask_members] == sector) & (
            (spd_p[mask_members] >= speed_edges[cls]) & (spd_p[mask_members] < speed_edges[cls + 1])
        )
        return sel.mean(axis=0)

    diffs, noise = [], []
    for s in range(ROSE_SECTORS):
        for c in range(ROSE_SPEED_CLASSES):
            truth = rose_truth(np.ones(N_MEMBERS, dtype=bool), s, c)
            pre = bilinear(rose[s * ROSE_SPEED_CLASSES + c])
            diffs.append(float(np.max(np.abs(pre - truth))))
            half = N_MEMBERS // 2
            m_all = np.ones(N_MEMBERS, dtype=bool)
            m_lo = np.zeros(N_MEMBERS, dtype=bool)
            m_lo[:half] = True
            m_hi = ~m_lo
            noise.append(
                float(np.max(np.abs(rose_truth(m_lo, s, c) - rose_truth(m_hi, s, c))))
            )
    print("  Fidelity: wind-rose bin probabilities (all 32 bins, max over")
    print(f"  {n_pts} points)")
    print(f"    precompute-and-interpolate error : {max(diffs):.4f}")
    print(f"    ensemble sampling noise (n=30)   : {max(noise):.4f}")
    print(f"    ratio                            : {max(diffs) / max(noise):.3f}"
          f"   ({'hidden in the noise' if max(diffs) < max(noise) else 'VISIBLE'})")
    print()


if __name__ == "__main__":
    main()
