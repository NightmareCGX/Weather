"""Experiment 17: aggregate-field cost measured on its own terms, rescaled to real GEFS.

Section 9.10 established the real per-shard baseline: 740-771 KB per (variable, member,
lead), i.e. 30 members cost 22.2-23.1 MB per (variable, lead).

But every benchmark in this report generated its member fields by EMULATING GRIB-12
packing, and those synthetic members compress to 2.32 MB/shard -- 3.1x WORSE than real
GEFS members. That asymmetry matters enormously, because:

  * member fields inherit GRIB2's packed-integer representation, whose low mantissa bits
    are zero, and that is what Zstd compresses;
  * every aggregate/derived field is computed by numpy on float32 and is therefore
    FULL-ENTROPY -- it does not have that structure.

So any scheme whose output is derived fields loses the packing advantage, and every
"vs current" ratio reported for such schemes in sections 9.7-9.9 is inflated by roughly
that 3.1x factor. This measures the aggregate fields' own cost (which is source
independent) and rescales against the REAL member baseline.

It also tests the fix: quantising the aggregate fields to a fixed-point grid restores the
zero-low-mantissa structure, which is exactly what the integer-scaling work in sections
2-3 was for. That work was useless for storing members (they are already packed) but it
is the key enabler for storing aggregates.

Run:  .venv/Scripts/python.exe bench_aggregate_real_cost.py
"""

from __future__ import annotations

import numpy as np
from numcodecs import Zstd

from bench_local_packing import CHUNK, LAT, LAT_CHUNKS, LON, LON_CHUNKS
from bench_member_packing import N_MEMBERS, chunk_buffers
from bench_representation_compare import build_members
from bench_render_outputs import bytes_of

Z5 = Zstd(level=5)

# --- the real per-shard baseline, from section 9.10 / the repo's own measurements -----
REAL_SHARD_LO_MB = 740 / 1e3
REAL_SHARD_HI_MB = 771 / 1e3
LEADS = 81
GEFS_VARS = 14
GEFS_MEMBERS = 30


def quantise(field, np_dtype, scale, sentinel_zero=False):
    """Quantise a float32 field to a fixed-point grid, zeroing the low mantissa bits."""
    limit = np.iinfo(np_dtype).max
    q = np.clip(np.round(np.nan_to_num(field, nan=0.0) / scale), -limit, limit).astype(np_dtype)
    return q


def bytes_as(field, quant=None):
    """Compressed bytes of one field, optionally quantised to `quant` = (dtype, scale)."""
    if quant is None:
        arr = np.asarray(field, dtype=np.float32)
    else:
        dt, scale = quant
        arr = quantise(field, dt, scale)
    padded = np.zeros((LAT, LON), dtype=arr.dtype)
    a = np.asarray(arr)
    padded[: a.shape[0], : a.shape[1]] = a
    # chunk_buffers pads float32; here we compress the raw buffer of whatever dtype
    total = 0
    for r in range(LAT_CHUNKS):
        for c in range(LON_CHUNKS):
            r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
            c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
            buf = np.zeros((CHUNK, CHUNK), dtype=arr.dtype)
            buf[: r1 - r0, : c1 - c0] = padded[r0:r1, c0:c1]
            total += len(Z5.encode(buf.tobytes(order="C")))
    return total


def main():
    members = build_members()
    stack = np.stack(members, axis=0)
    mean = np.nanmean(stack, axis=0)
    var = np.nanstd(stack, axis=0)  # store sigma, not sigma^2, so the scale is comparable

    dev = stack - mean[None]

    def cov(dr, dc):
        a = dev[:, : LAT - dr, : LON - abs(dc)]
        b = dev[:, dr:, abs(dc) :]
        return np.nanmean(a * b, axis=0)

    covs = [cov(0, 1), cov(1, 0), cov(1, 1), np.nanmean(dev[:, 1:, :-1] * dev[:, :-1, 1:], axis=0)]

    print()
    print("=" * 100)
    print("Aggregate-field cost, and the real-baseline rescaling")
    print("=" * 100)
    synthetic_members_mb = bytes_of(list(members)) / 1e6
    real_members_mb = GEFS_MEMBERS * REAL_SHARD_HI_MB
    print(f"  synthetic 30 members (this benchmark) : {synthetic_members_mb:8.2f} MB per (variable, lead)")
    print(f"  REAL 30 members (repo measurement)    : {real_members_mb:8.2f} MB per (variable, lead)")
    print(f"  -> the benchmark baseline is {synthetic_members_mb / real_members_mb:.2f}x worse-compressing")
    print("     than the real thing, so every 'vs current' ratio below is corrected by that.")
    print()
    print(f"{'field':<22} {'float32 MB':>11} {'i16@0.01 MB':>13} {'i32@0.001 MB':>14} {'i16 gain':>10}")
    print("-" * 100)
    fields = [("MEAN", mean), ("STD (sigma)", var)] + [(f"COV_{i}", c) for i, c in enumerate(covs)]
    tot = {"f32": 0.0, "i16": 0.0, "i32": 0.0}
    for label, f in fields:
        b_f = bytes_as(f) / 1e6
        b_16 = bytes_as(f, (np.int16, 0.01)) / 1e6
        b_32 = bytes_as(f, (np.int32, 0.001)) / 1e6
        tot["f32"] += b_f
        tot["i16"] += b_16
        tot["i32"] += b_32
        print(f"{label:<22} {b_f:11.3f} {b_16:13.3f} {b_32:14.3f} {b_f / b_16:9.2f}x")
    print("-" * 100)
    print(f"{'6-field total':<22} {tot['f32']:11.3f} {tot['i16']:13.3f} {tot['i32']:14.3f} "
          f"{tot['f32'] / tot['i16']:9.2f}x")
    print()
    print("  (quantising an aggregate is safe for MEAN/STD at scale 0.01 -- that is the 0.005")
    print("   error budget you already accepted. The covariance fields are signed and small;")
    print("   scale 0.01 on them needs a range check before use.)")
    print()

    print("=" * 100)
    print("Per (variable, lead), corrected to the REAL member baseline")
    print("=" * 100)
    print(f"{'scheme':<34} {'fields':>7} {'MB':>9} {'vs 30 real members':>19}")
    print("-" * 100)
    schemes = [
        ("30 members (current, REAL)", 30, real_members_mb),
        ("MEAN + STD, float32", 2, tot["f32"] / 6 * 2),
        ("MEAN + STD, i16@0.01", 2, tot["i16"] / 6 * 2),
        ("MEAN + STD + 4 cov, float32", 6, tot["f32"]),
        ("MEAN + STD + 4 cov, i16@0.01", 6, tot["i16"]),
    ]
    for label, nf, mb in schemes:
        print(f"{label:<34} {nf:>7} {mb:9.2f} {real_members_mb / mb:18.2f}x")
    print()

    print("=" * 100)
    print("Per GEFS cycle: 30 members, all 14 variables, 81 leads")
    print("=" * 100)
    units = LEADS * GEFS_VARS
    member_total = REAL_SHARD_HI_MB * GEFS_MEMBERS * units / 1e3
    print(f"  current (30 members)                  : {member_total:7.1f} GB   (matches 9.10)")
    for label, mb in (
        ("MEAN + STD, float32", tot["f32"] / 6 * 2),
        ("MEAN + STD, i16@0.01", tot["i16"] / 6 * 2),
        ("MEAN + STD + 4 cov, float32", tot["f32"]),
        ("MEAN + STD + 4 cov, i16@0.01", tot["i16"]),
    ):
        per_cycle = mb * units / 1e3
        print(f"  {label:<37}: {per_cycle:7.1f} GB   ({member_total / per_cycle:.2f}x smaller, "
              f"saves {member_total - per_cycle:.1f} GB/cycle)")
    print()
    print("  NOTE: the mean shards (0.87 GB/cycle) become redundant in all of these, since")
    print("  the mean is one of the stored fields -- so the true saving is slightly larger.")
    print()

    # ---------------------------------------------------------------------------------
    # MEAN+STD gives no band. The chart's P10-P90 band must then be ASSUMED Gaussian
    # (mean +/- 1.28 sigma). If the band must come from real sample quantiles, those are
    # extra fields -- this measures what they add.
    # ---------------------------------------------------------------------------------
    print("=" * 100)
    print("Keeping the P10-P90 band: cost of storing the quantiles explicitly")
    print("=" * 100)
    quant = [np.nanpercentile(stack, q, axis=0) for q in (10, 25, 50, 75, 90)]
    print(f"{'group':<30} {'float32 MB':>11} {'i16@0.01 MB':>13} {'i16 gain':>10}")
    print("-" * 100)
    q_f = sum(bytes_as(f) for f in quant) / 1e6
    q_16 = sum(bytes_as(f, (np.int16, 0.01)) for f in quant) / 1e6
    print(f"{'5 quantiles (P10/P25/P50/P75/P90)':<30} {q_f:11.3f} {q_16:13.3f} {q_f / q_16:9.2f}x")

    edges = np.linspace(float(np.nanmin(stack)), float(np.nanmax(stack)), 33)
    hist = [np.nanmean((stack >= edges[b]) & (stack < edges[b + 1]), axis=0) for b in range(32)]
    h_f = sum(bytes_as(f) for f in hist) / 1e6
    h_16 = sum(bytes_as(f, (np.int16, 0.01)) for f in hist) / 1e6
    print(f"{'32 histogram bars':<30} {h_f:11.3f} {h_16:13.3f} {h_f / h_16:9.2f}x")
    print()

    print(f"{'bundle':<44} {'MB':>9} {'GB/cycle':>10} {'vs current':>11} {'saves':>10}")
    print("-" * 100)
    combos = [
        ("MEAN + STD (band assumed Gaussian)", tot["i16"] / 6 * 2),
        ("MEAN + STD + 4 cov (exact variance)", tot["i16"]),
        ("MEAN + STD + 5 quantiles", tot["i16"] / 6 * 2 + q_16),
        ("MEAN + STD + 5 quantiles + 32 hist bars", tot["i16"] / 6 * 2 + q_16 + h_16),
        ("MEAN + STD + 4 cov + 5 quantiles + 32 bars", tot["i16"] + q_16 + h_16),
    ]
    for label, mb in combos:
        cyc = mb * units / 1e3
        print(f"{label:<44} {mb:9.2f} {cyc:10.1f} {member_total / cyc:10.2f}x "
              f"{member_total - cyc:9.1f} GB")
    print()
    print("  All rows use i16@0.01 for the stored fields (max error 0.005, the budget you")
    print("  already accepted). The band is the deciding cost: assuming a Gaussian band is")
    print("  what makes the 2-field form cheap, and the sample ensemble is visibly")
    print("  non-Gaussian in the KDE, so that assumption needs validating per variable.")
    print()


if __name__ == "__main__":
    main()
