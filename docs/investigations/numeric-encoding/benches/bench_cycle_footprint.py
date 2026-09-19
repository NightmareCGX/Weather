"""How much disk does ONE cycle's 240 h forecast occupy?

Derivation from documented constants and the repo's own measured shard sizes. Every
constant is annotated with its source so the number is auditable and can be re-run
against a real store.

Two independent measurements of the compressed shard size are available and they agree:
  * PHASE4B:369-370 -- 10.36 MB compressed per region, where 1 region = 14 shards
    (PHASE4F section 7: "1 Region = 14 physical shard objects") -> 740 KB/shard.
  * PHASE4F:165-167 -- 30 regions = 323.98 MB over 420 shards -> 771 KB/shard.
The 4% spread is the difference between an average over all 14 variables and a
temperature-heavy sample, so this brackets the answer rather than picking one.

Also flags a documented inconsistency: PHASE4B reports "1,680 chunks (43.57 MB float32)"
for 1,680 padded 100x100 float32 chunks, which actually weigh 67.2 MB. So the quoted
"4.21x compression ratio" is not consistent with the shard geometry (5.6-6.5x implied),
and neither figure should be quoted as the store's compression ratio.

Run:  .venv/Scripts/python.exe bench_cycle_footprint.py
"""

from __future__ import annotations

# --- structural constants, each with its source ---------------------------------------
GRID_LAT, GRID_LON = 721, 1440          # ARCHITECTURE.md:161-163 (0.25 deg global)
CHUNK = 100                             # zarr_writer.py:37-48 DEFAULT_CHUNKS
LAT_CHUNKS, LON_CHUNKS = 8, 15          # 721/100 and 1440/100, ceil
CHUNKS_PER_SHARD = LAT_CHUNKS * LON_CHUNKS        # 120
CHUNK_BYTES = CHUNK * CHUNK * 4                   # padded to full 100x100, float32
GEFS_MEMBERS = 30                       # coverage.py:30-33 MODEL_EXPECTED_MEMBERS
GEFS_VARS = 14                          # reclamation.py:321-336
GFS_VARS = 15                           # reclamation.py:302-318
LEADS = 81                              # horizon.py:14-15 (0..240 h at 3 h)
SHARDS_PER_REGION = 14                  # PHASE4F section 7

# --- measured compressed shard size ---------------------------------------------------
SHARD_LO_MB = 10.36 / SHARDS_PER_REGION         # PHASE4B:369-370
SHARD_HI_MB = 323.98 / (30 * SHARDS_PER_REGION)  # PHASE4F:165-167


def shards(members: int, variables: int, leads: int) -> int:
    return members * variables * leads


def report():
    padded_shard_mb = CHUNKS_PER_SHARD * CHUNK_BYTES / 1e6
    unpadded_field_mb = GRID_LAT * GRID_LON * 4 / 1e6

    print()
    print("=" * 96)
    print("Geometry")
    print("=" * 96)
    print(f"  grid                      : {GRID_LAT} x {GRID_LON}  (0.25 deg global)")
    print(f"  chunk                     : {CHUNK} x {CHUNK} x 4 B = {CHUNK_BYTES / 1e3:.0f} KB (NaN-padded)")
    print(f"  chunks per shard          : {LAT_CHUNKS} x {LON_CHUNKS} = {CHUNKS_PER_SHARD}")
    print(f"  padded shard payload      : {padded_shard_mb:.2f} MB")
    print(f"  one field, unpadded       : {unpadded_field_mb:.3f} MB")
    print(f"  measured compressed shard : {SHARD_LO_MB:.3f} - {SHARD_HI_MB:.3f} MB")
    print()

    members_gefs = shards(GEFS_MEMBERS, GEFS_VARS, LEADS)
    means_gefs = shards(1, GEFS_VARS, LEADS)
    gfs = shards(1, GFS_VARS, LEADS)

    print("=" * 96)
    print("Object counts per cycle")
    print("=" * 96)
    print(f"  GEFS member shards : {GEFS_MEMBERS} members x {GEFS_VARS} vars x {LEADS} leads = {members_gefs:,}")
    print(f"  GEFS mean shards   : 1 x {GEFS_VARS} vars x {LEADS} leads              = {means_gefs:,}")
    print(f"  GFS  det shards    : 1 x {GFS_VARS} vars x {LEADS} leads              = {gfs:,}")
    print(f"  total              : {members_gefs + means_gefs + gfs:,} objects per cycle")
    print()

    print("=" * 96)
    print("Disk per cycle")
    print("=" * 96)
    print(f"{'component':<24} {'shards':>9} {'@740 KB':>11} {'@771 KB':>11}")
    print("-" * 96)
    rows = [
        ("GEFS 30 members", members_gefs),
        ("GEFS ensemble mean", means_gefs),
        ("GFS deterministic", gfs),
    ]
    tot_lo = tot_hi = 0.0
    for label, n in rows:
        lo, hi = n * SHARD_LO_MB / 1e3, n * SHARD_HI_MB / 1e3
        tot_lo += lo
        tot_hi += hi
        print(f"{label:<24} {n:>9,} {lo:>10.2f} GB {hi:>10.2f} GB")
    print("-" * 96)
    print(f"{'TOTAL per cycle':<24} {members_gefs + means_gefs + gfs:>9,} "
          f"{tot_lo:>10.2f} GB {tot_hi:>10.2f} GB")
    print()
    print("=" * 96)
    print("The GEFS ensemble alone (what the review asked about)")
    print("=" * 96)
    print(f"  30 members, 14 variables, 81 leads : "
          f"{members_gefs * SHARD_LO_MB / 1e3:.1f} - {members_gefs * SHARD_HI_MB / 1e3:.1f} GB")
    print(f"  plus the ensemble mean shards      : "
          f"{means_gefs * SHARD_LO_MB / 1e3:.2f} - {means_gefs * SHARD_HI_MB / 1e3:.2f} GB")
    print(f"  -> ensemble-related footprint      : "
          f"{(members_gefs + means_gefs) * SHARD_LO_MB / 1e3:.1f} - "
          f"{(members_gefs + means_gefs) * SHARD_HI_MB / 1e3:.1f} GB per cycle")
    print()
    print("  useful per-unit breakdown (at 771 KB/shard):")
    per = SHARD_HI_MB / 1e3
    print(f"    per member            : {members_gefs / GEFS_MEMBERS * per:.2f} GB")
    print(f"    per lead time         : {members_gefs / LEADS * per * 1e3:.0f} MB")
    print(f"    per variable          : {members_gefs / GEFS_VARS * per:.2f} GB")
    print(f"    per (member, lead)    : {GEFS_VARS * per * 1e3:.1f} MB  "
          f"(PHASE4B measured 10.36 MB)")
    print(f"    per shard             : {SHARD_HI_MB * 1e3:.0f} KB")
    print()

    print("=" * 96)
    print("Cross-checks")
    print("=" * 96)
    print(f"  raw vs compressed, GEFS members: "
          f"{members_gefs * padded_shard_mb / 1e3:.0f} GB -> "
          f"{members_gefs * SHARD_HI_MB / 1e3:.1f} GB  = "
          f"{padded_shard_mb / SHARD_HI_MB:.1f}x aggregate (padding included)")
    print(f"  same, unpadded field size      : "
          f"{members_gefs * unpadded_field_mb / 1e3:.0f} GB -> "
          f"{members_gefs * SHARD_HI_MB / 1e3:.1f} GB  = "
          f"{unpadded_field_mb / SHARD_HI_MB:.1f}x")
    print(f"  PHASE4B quotes 4.21x, computed here 5.6-6.5x -> PHASE4B's uncompressed")
    print(f"  figure (43.57 MB for 1,680 padded chunks that weigh 67.2 MB) is internally")
    print(f"  inconsistent; do not quote either as the store's ratio.")
    print()
    print(f"  daily (4 cycles): {4 * tot_lo:.0f} - {4 * tot_hi:.0f} GB/day")
    print("  MONITORING.md:206-247 sample shows MinIO at ~61 GB used with Postgres")
    print("  separately at 42.1 GB -> ~2 cycles' worth at this rate, which is consistent.")
    print()
    print("  NOTE: derived from documented constants and the repo's measured shard sizes,")
    print("  NOT from a real store (none is available here). Verify by listing the actual")
    print("  objects for one cycle prefix and summing their sizes.")
    print()


if __name__ == "__main__":
    report()
