"""Experiment 10: can cross-member redundancy be reached with a bigger Zstd window,
and how similar do members have to be for member-packing to win?

Experiment 9 found member-packing compresses WORSE (0.859x) than the spatial grouping
(1.171x). Two possible explanations, with very different implications:

  (1) There is no cross-member redundancy worth having; or
  (2) It exists but Zstd-5's window cannot reach it -- concatenating members
      member-major puts member 2's twin of chunk 0 ~900 KB after member 1's, and
      interleaving chunk-major destroys the within-member spatial locality that pays.

Only (2) would be fixable (bigger window / different ordering), so this separates them:
  * sweep the Zstd level (level 19 has a multi-MB window) with member-major ordering;
  * sweep how similar members are, to find the crossover point, if any.

Run:  .venv/Scripts/python.exe bench_member_window.py
"""

from __future__ import annotations

import time

import numpy as np
from numcodecs import Zstd

from bench_local_packing import CHUNK, LAT, LAT_CHUNKS, LON, LON_CHUNKS
from bench_member_packing import N_CHUNKS, build_members, chunk_buffers
from bench_shard_and_ensemble import grib_pack
from bench_stat_interpolation import smooth

N_MEMBERS = 30
Z = {lvl: Zstd(level=lvl) for lvl in (5, 12, 19)}


def encode_flat(parts, z):
    payload = np.concatenate(parts).astype(np.float32)
    return len(z.encode(payload.tobytes(order="C")))


def main():
    bufs = [[b for b in chunk_buffers(m)] for m in build_members()]

    print()
    print("Part 1: does a bigger Zstd window reach cross-member redundancy?")
    print("(all figures are MB for 30 members x 1 variable/lead; lower is better)")
    print()
    print(f"{'level':>6} {'A current':>12} {'B spatial':>12} {'D member-major':>16} {'C chunk-interleaved':>20}")
    print("-" * 72)
    for lvl, z in Z.items():
        a = sum(len(z.encode(b.tobytes(order="C"))) for mb in bufs for b in mb) / 1e6
        b = sum(encode_flat([x.ravel() for x in mb], z) for mb in bufs) / 1e6
        d = encode_flat([bufs[m][k].ravel() for m in range(N_MEMBERS) for k in range(N_CHUNKS)], z) / 1e6
        c = sum(
            encode_flat([bufs[m][k].ravel() for m in range(N_MEMBERS)], z) for k in range(N_CHUNKS)
        ) / 1e6
        print(f"{lvl:>6} {a:12.2f} {b:12.2f} {d:16.2f} {c:20.2f}")
    print()
    print("If D (member-major, big window) were clearly below B, cross-member redundancy")
    print("would be reachable and the proposal would have a variant worth pursuing.")
    print()

    # -------------------------------------------------------------------------------
    print("=" * 84)
    print("Part 2: crossover vs member similarity")
    print("=" * 84)
    print("A spread/base ratio near 0 means members are near-identical. Real GEFS 10 m wind")
    print("has a spread of roughly 2-8 km/h against a ~12-30 km/h mean, i.e. ratio ~0.1-0.5.")
    print()
    print(f"{'spread/base':>12} {'B spatial MB':>14} {'C packed MB':>13} {'C/B':>7} {'verdict':>14}")
    print("-" * 66)
    base = 12.0 + 9.0 * smooth((LAT, LON), 12.0, 11)
    for ratio in (0.0, 0.02, 0.05, 0.15, 0.3, 0.6, 1.5):
        members = []
        for m in range(N_MEMBERS):
            pert = smooth((LAT, LON), 8.0, 3000 + m) * (base.std() * ratio)
            members.append(grib_pack(np.asarray(base + pert, dtype=np.float32), 12))
        mb = [chunk_buffers(x) for x in members]
        z = Z[5]
        b = sum(encode_flat([x.ravel() for x in one], z) for one in mb) / 1e6
        c = (
            sum(encode_flat([mb[m][k].ravel() for m in range(N_MEMBERS)], z) for k in range(N_CHUNKS))
            / 1e6
        )
        verdict = "packing wins" if c < b else "spatial wins"
        print(f"{ratio:>12.2f} {b:14.2f} {c:13.2f} {c / b:7.3f} {verdict:>14}")
    print()


if __name__ == "__main__":
    main()
