"""Experiment 7: the affordable lossless ceiling.

Experiment 4 dismissed Zstd-19 on encode cost (1.1 s/shard at whole-shard framing), but
never swept the *intermediate* levels at the recommended grouping. Higher levels only
cost encode time -- decode cost and the reader are unchanged -- so if level 9-15 lands
between the Zstd-5 and Zstd-19 ratios at an affordable encode cost, the lossless ceiling
is higher than reported.

This sweeps level x framing and projects the encode hours onto the real GEFS cycle
(34,020 shards/cycle, 2 available ingestion cores per the deployment config), so
"affordable" is decided against the actual 6-hour cycle budget, not by eyeball.

Run:  .venv/Scripts/python.exe bench_level_sweep.py
"""

from __future__ import annotations

import time

import numpy as np
from numcodecs import Zstd

from bench_local_packing import LAT_CHUNKS, LON_CHUNKS
from bench_shard_and_ensemble import chunk_buffers, grib_pack, make_temperature

LEVELS = (3, 5, 7, 9, 12, 15, 19)
GROUPINGS = (15, 120)

# Real GEFS per-cycle shard count: 30 members x 81 leads x 14 variables.
GEFS_SHARDS_PER_CYCLE = 30 * 81 * 14
# The deployment runs ingestion with cpus: "2.0" (docker-compose.yml:143) and
# WEATHER_INGEST_DECODE_CONCURRENCY: "2" -- compression is on those same cores.
INGEST_CORES = 2
# The GEFS cycle currently completes in ~38m50s (docs/MONITORING.md:219) inside a 6h
# cycle. Allow the extra compression work to stay under 1h of wall clock.
ENCODE_BUDGET_SECONDS = 3600


def encode_all(bufs, group, level):
    """Return (total_bytes, encode_ms_per_shard)."""
    z = Zstd(level=level)
    t0 = time.perf_counter()
    total = 0
    for i in range(0, len(bufs), group):
        slab = bufs[i : i + group]
        payload = np.concatenate([b.ravel() for b in slab]).astype(np.float32).tobytes(order="C")
        total += len(z.encode(payload))
    t1 = time.perf_counter()
    return total, 1000 * (t1 - t0)


def main():
    print()
    print("Sweeping Zstd level x framing (temperature, upstream GRIB 12 bits/view)")
    print()

    arr = grib_pack(make_temperature(), 12)
    bufs = chunk_buffers(arr)
    cur, cur_ms = encode_all(bufs, 1, 5)

    print("=" * 112)
    print(f"Reference: shipped config (Zstd5, one stream per chunk) = {cur / 1e6:.3f} MB, {cur_ms:.0f} ms/shard")
    print(f"Cycle budget: {GEFS_SHARDS_PER_CYCLE:,} GEFS shards/cycle on {INGEST_CORES} ingestion cores;")
    print(f"             allowing <{ENCODE_BUDGET_SECONDS / 3600:.0f} h of wall clock (current GEFS cycle is ~39 min).")
    print("=" * 112)
    print()

    for group in GROUPINGS:
        print(f"group = {group} chunks/stream  ({LAT_CHUNKS * LON_CHUNKS // group} streams per shard)")
        print(
            f"{'level':>6} {'MB':>9} {'vs current':>11} {'enc ms/shard':>13} "
            f"{'GEFS encode h @2 cores':>23} {'verdict':>10}"
        )
        print("-" * 112)
        for level in LEVELS:
            total, enc_ms = encode_all(bufs, group, level)
            hours = enc_ms / 1000 * GEFS_SHARDS_PER_CYCLE / INGEST_CORES / 3600
            verdict = "OK" if hours <= ENCODE_BUDGET_SECONDS / 3600 else "TOO SLOW"
            print(
                f"{level:>6} {total / 1e6:9.3f} {cur / total:10.2f}x {enc_ms:13.1f} {hours:23.2f} {verdict:>10}"
            )
        print()


if __name__ == "__main__":
    main()
