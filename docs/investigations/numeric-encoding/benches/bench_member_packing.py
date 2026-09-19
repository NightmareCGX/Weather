"""Experiment 9: is the member axis a better compression-grouping axis than the spatial axis?

Proposal under test: change the physical layout from one shard per
(cycle, variable, member, lead) to one shard per (cycle, variable, lead) holding members
1..30.

Key mechanical point this experiment exists to settle: merely packing 30 members into one
container while keeping per-chunk independent Zstd streams saves essentially NOTHING --
each 40 KB chunk is still its own stream, so no byte ever shares context with another
member's. The only thing that can pay is putting several members' bytes into the SAME
Zstd stream. So the real question is whether cross-member grouping beats cross-spatial
grouping (which §4 measured at 1.18-1.39x).

Note this is NOT the same as §5's ensemble-residual experiment, which explicitly
subtracted a mean and reintroduced float32 cancellation noise. Here nothing is
subtracted: Zstd's own match finder exploits the cross-member similarity.

Layouts compared (all with the production 100x100 float32 chunk geometry, 120 chunks per
variable per lead):

  A  current          30 members x 120 independent streams      (40 KB raw each)
  B  spatial only     30 members x   1 stream per member        (4.8 MB raw each)
  C  member-packed    120 streams, each = 1 chunk x 30 members   (1.2 MB raw each)
  D  everything       1 stream                                  (144 MB raw)
  E  hybrid           8 streams, each = 15 chunks x 30 members   (18 MB raw each)

Also reported: what a single-member point query must decode under each layout, since
that is what decides whether the point path survives the change.

Run:  .venv/Scripts/python.exe bench_member_packing.py
"""

from __future__ import annotations

import time

import numpy as np
from numcodecs import Zstd

from bench_local_packing import CHUNK, LAT, LAT_CHUNKS, LON, LON_CHUNKS
from bench_shard_and_ensemble import grib_pack
from bench_stat_interpolation import smooth

N_MEMBERS = 30
N_CHUNKS = LAT_CHUNKS * LON_CHUNKS  # 120
CHUNK_BYTES = CHUNK * CHUNK * 4  # 40000

Z5 = Zstd(level=5)
RNG = np.random.default_rng(7)


def build_members():
    """30 realistic members: the analysis field plus spatially correlated perturbations."""
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    base = 12.0 + 9.0 * smooth((LAT, LON), 12.0, 11)
    members = []
    for m in range(N_MEMBERS):
        pert = smooth((LAT, LON), 8.0, 2000 + m) * 1.8
        field = grib_pack(np.asarray(base + pert, dtype=np.float32), 12)
        members.append(field)
    return members


def chunk_buffers(arr):
    """The exact 100x100 NaN-padded buffers production compresses."""
    bufs = []
    for r in range(LAT_CHUNKS):
        for c in range(LON_CHUNKS):
            buf = np.full((CHUNK, CHUNK), np.nan, dtype=np.float32)
            r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
            c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
            buf[: r1 - r0, : c1 - c0] = arr[r0:r1, c0:c1]
            bufs.append(buf)
    return bufs


def main():
    members = build_members()
    # bufs[member][chunk]
    bufs = [chunk_buffers(m) for m in members]
    raw_per_var_lead = N_MEMBERS * N_CHUNKS * CHUNK_BYTES

    print()
    print("Physical layout comparison, 30 members x 1 (variable, lead)")
    print(f"production geometry: {N_CHUNKS} chunks of {CHUNK}x{CHUNK} float32 per member")
    print(f"raw (NaN-padded): {raw_per_var_lead / 1e6:.1f} MB per (variable, lead)")
    print()

    results = {}

    # ---- A: current -- per member, independent per-chunk streams ----------------
    t0 = time.perf_counter()
    total = 0
    for mb in bufs:
        for b in mb:
            total += len(Z5.encode(b.tobytes(order="C")))
    t1 = time.perf_counter()
    results["A current (per-member, per-chunk)"] = (total, 1000 * (t1 - t0), CHUNK_BYTES)

    # ---- B: spatial only -- one stream per member over its 120 chunks -----------
    t0 = time.perf_counter()
    total = 0
    for mb in bufs:
        payload = np.concatenate([b.ravel() for b in mb]).astype(np.float32).tobytes(order="C")
        total += len(Z5.encode(payload))
    t1 = time.perf_counter()
    results["B spatial only (per-member, 1 stream)"] = (
        total,
        1000 * (t1 - t0),
        N_CHUNKS * CHUNK_BYTES,
    )

    # ---- C: member-packed, chunk-granular -- 120 streams, each = chunk x 30 members
    t0 = time.perf_counter()
    total = 0
    for k in range(N_CHUNKS):
        payload = np.concatenate([bufs[m][k].ravel() for m in range(N_MEMBERS)]).astype(np.float32)
        total += len(Z5.encode(payload.tobytes(order="C")))
    t1 = time.perf_counter()
    results["C member-packed (chunk x 30 members)"] = (
        total,
        1000 * (t1 - t0),
        N_MEMBERS * CHUNK_BYTES,
    )

    # ---- D: everything in one stream. Two orderings, since Zstd's match window
    #         only reaches so far: member-major puts a member's twin chunk 900 KB
    #         away, chunk-major puts it 40 KB away.
    for label, order in (("D all-in-one, member-major", "m"), ("D all-in-one, chunk-major", "c")):
        parts = []
        if order == "m":
            for m in range(N_MEMBERS):
                for k in range(N_CHUNKS):
                    parts.append(bufs[m][k].ravel())
        else:
            for k in range(N_CHUNKS):
                for m in range(N_MEMBERS):
                    parts.append(bufs[m][k].ravel())
        payload = np.concatenate(parts).astype(np.float32)
        t0 = time.perf_counter()
        total = len(Z5.encode(payload.tobytes(order="C")))
        t1 = time.perf_counter()
        results[label] = (total, 1000 * (t1 - t0), payload.nbytes)

    # ---- E: hybrid -- 15 chunks x 30 members per stream -------------------------
    group = 15
    t0 = time.perf_counter()
    total = 0
    for g in range(0, N_CHUNKS, group):
        parts = []
        for k in range(g, min(g + group, N_CHUNKS)):
            for m in range(N_MEMBERS):
                parts.append(bufs[m][k].ravel())
        payload = np.concatenate(parts).astype(np.float32)
        total += len(Z5.encode(payload.tobytes(order="C")))
    t1 = time.perf_counter()
    results["E hybrid (15 chunks x 30 members)"] = (
        total,
        1000 * (t1 - t0),
        group * N_MEMBERS * CHUNK_BYTES,
    )

    baseline = results["A current (per-member, per-chunk)"][0]
    print(
        f"{'layout':<42} {'MB':>8} {'vs current':>11} {'encode ms':>10} "
        f"{'point-query decode':>19}"
    )
    print("-" * 96)
    for label, (total, enc_ms, one_stream_bytes) in results.items():
        print(
            f"{label:<42} {total / 1e6:8.2f} {baseline / total:10.3f}x {enc_ms:10.0f} "
            f"{one_stream_bytes / 1e6:15.2f} MB"
        )
    print()
    print("'point-query decode' = the raw bytes a single-member point query must")
    print("decompress under that layout. Today it is one 40 KB chunk; the point path")
    print("currently costs ~2.5 ms per chunk fetch (api/core/config.py:228-232).")
    print()

    # Actual decode timing for the layout that matters most: C.
    k = 60
    payload_c = np.concatenate([bufs[m][k].ravel() for m in range(N_MEMBERS)]).astype(np.float32)
    blob_c = Z5.encode(payload_c.tobytes(order="C"))
    blob_a = Z5.encode(bufs[0][k].tobytes(order="C"))
    t0 = time.perf_counter()
    for _ in range(200):
        Z5.decode(blob_a)
    t1 = time.perf_counter()
    dec_a = 1000 * (t1 - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        Z5.decode(blob_c)
    t1 = time.perf_counter()
    dec_c = 1000 * (t1 - t0) / 200
    print(f"measured decode, one member's chunk (layout A): {dec_a:.3f} ms")
    print(f"measured decode, 30 members' chunk (layout C):  {dec_c:.3f} ms  ({dec_c / dec_a:.1f}x)")
    print()

    # -------------------------------------------------------------------------------
    # Reshape pass feasibility (the user's second-phase coalescing idea)
    # -------------------------------------------------------------------------------
    print("=" * 96)
    print("Reshape pass: read 30 member shards, re-encode, write 1 packed shard")
    print("=" * 96)
    per_group_read = baseline  # compressed bytes of 30 member shards for one (variable, lead)
    per_group_write = results["C member-packed (chunk x 30 members)"][0]
    groups = 81 * 14  # leads x variables for GEFS
    print(f"  per (variable, lead): read {per_group_read / 1e6:.2f} MB, write {per_group_write / 1e6:.2f} MB")
    print(f"  GEFS groups per cycle: 81 leads x 14 vars = {groups}")
    print(f"  total I/O per cycle: {(per_group_read + per_group_write) * groups / 1e9:.1f} GB")
    enc_ms = results["C member-packed (chunk x 30 members)"][1]
    print(f"  re-encode CPU: {enc_ms:.0f} ms/group x {groups} = {enc_ms * groups / 1000 / 3600:.2f} h single-core")
    print()


if __name__ == "__main__":
    main()
