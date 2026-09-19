"""Experiment 5: framing granularity curve for the sharded_v1 container.

Whole-shard compression won 1.15-1.33x at Zstd-5 (bench_shard_and_ensemble), but it
forces every random access to decode the entire 900 KB shard, which is brutal for the
point-forecast path (it reads a 2x2 corner = ~7 KB today).

This measures the trade-off: group N chunks per Zstd stream, for N = 1..120, and report
compressed size alongside (a) single-group decode cost and (b) the read amplification a
point query pays, so a defensible operating point can be picked.

Run:  .venv/Scripts/python.exe artifacts/dtype_bench/bench_framing_curve.py
"""

from __future__ import annotations

import time

import numpy as np
from numcodecs import Zstd

from bench_local_packing import LAT_CHUNKS, LON_CHUNKS
from bench_shard_and_ensemble import chunk_buffers, grib_pack, make_temperature

Z5 = Zstd(level=5)
Z12 = Zstd(level=12)
Z19 = Zstd(level=19)

SHARDS_PER_GEFS_CYCLE = 30 * 81 * 14  # members x leads x variables
SHARDS_PER_GFS_CYCLE = 1 * 81 * 15


def framing(arr, group):
    bufs = chunk_buffers(arr)
    raw = sum(b.size * 4 for b in bufs)
    groups = [np.concatenate([b.ravel() for b in bufs[i : i + group]]) for i in range(0, len(bufs), group)]
    return groups, raw


def point_amplification(group):
    """Compressed-byte amplification for a point query's 2x2 corner chunk window.

    The reader fetches up to 4 chunks (2x2 bilinear corners). Under grouping, that
    window may land in 1..4 groups. Averaged over every valid corner position, in
    compressed bytes, relative to the current one-stream-per-chunk layout.
    """
    touched = []
    for r in range(LAT_CHUNKS - 1):
        for c in range(LON_CHUNKS - 1):
            idxs = (r * LON_CHUNKS + c, r * LON_CHUNKS + c + 1, (r + 1) * LON_CHUNKS + c, (r + 1) * LON_CHUNKS + c + 1)
            touched.append(len({i // group for i in idxs}))
    return float(np.mean(touched)), float(np.max(touched))


def main():
    print()
    print("Framing granularity: group N of 120 chunks per Zstd stream")
    print("(upstream GRIB 12 bits/view, which matches the repo's measured 4.21x baseline)")
    print()
    arr = grib_pack(make_temperature(), 12)
    bufs = chunk_buffers(arr)
    raw = sum(b.size * 4 for b in bufs)

    header = (
        f"{'group':>6} {'streams':>8} {'grpKB':>7} | {'Z5 MB':>7} {'vs now':>7} {'dec ms':>7} | "
        f"{'Z12 MB':>7} {'vs now':>7} | {'Z19 MB':>7} {'vs now':>7} {'dec ms':>7} | "
        f"{'pt KB':>7} {'amp':>6}"
    )
    print(header)
    print("-" * len(header))

    baseline = None
    baseline_pt = None
    for group in (1, 2, 3, 5, 8, 15, 24, 40, 60, 120):
        groups, _ = framing(arr, group)
        cells = []
        for z in (Z5, Z12, Z19):
            t0 = time.perf_counter()
            blobs = [z.encode(g.tobytes(order="C")) for g in groups]
            t1 = time.perf_counter()
            t2 = time.perf_counter()
            z.decode(blobs[len(blobs) // 2])
            t3 = time.perf_counter()
            cells.append((sum(len(b) for b in blobs), 1000 * (t3 - t2), 1000 * (t1 - t0)))
        if baseline is None:
            baseline = cells[0][0]

        mean_groups, _ = point_amplification(group)
        grp_compressed = cells[0][0] / len(groups)
        pt_kb = mean_groups * grp_compressed / 1e3
        if baseline_pt is None:
            baseline_pt = pt_kb
        amp = pt_kb / baseline_pt

        print(
            f"{group:>6} {len(groups):>8} {grp_compressed / 1e3:7.1f} | "
            f"{cells[0][0] / 1e6:7.3f} {baseline / cells[0][0]:6.2f}x {cells[0][1]:6.2f} | "
            f"{cells[1][0] / 1e6:7.3f} {baseline / cells[1][0]:6.2f}x | "
            f"{cells[2][0] / 1e6:7.3f} {baseline / cells[2][0]:6.2f}x {cells[2][1]:6.2f} | "
            f"{pt_kb:7.1f} {amp:5.1f}x"
        )

    print()
    print("grpKB = compressed size of one framing group; pt KB = compressed bytes a point")
    print("query must fetch; amp = that relative to the current per-chunk layout.")
    print()
    print("Projected shard counts per cycle (framing factor is per-shard, so it multiplies):")
    print(f"  GEFS {SHARDS_PER_GEFS_CYCLE:,} shards/cycle   GFS {SHARDS_PER_GFS_CYCLE:,} shards/cycle")


if __name__ == "__main__":
    main()
