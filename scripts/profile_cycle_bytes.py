"""Per-variable byte profile of a forecast cycle store.

Read-only. Lists the objects under one cycle's store prefix and aggregates bytes by
(variable, target kind), which is what capacity planning needs: the platform's whole-cycle
footprint has only ever been estimated from two sampled variables, and the per-variable
compressed size varies by 1.75x across the variables measured so far
(``temperature_2m`` 852.7 KB/shard vs ``apcp`` 486.3 KB/shard).

Usage (S3 / MinIO, reads MINIO_* / AWS credentials from the environment):
    python scripts/profile_cycle_bytes.py \
        --store s3://weather-data/gefs/2026-09-18/00/cycle.zarr

Usage (local directory)::

    python scripts/profile_cycle_bytes.py --store /data/gefs/2026-09-18/00/cycle.zarr

Exits non-zero if the store cannot be listed, so it is safe to use as a gate.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field

#: Object-name shapes the store uses, mirroring the shard key conventions.
#: ``<variable>/shard.<kind>...`` for sharded stores, ``<variable>/<key>`` for legacy v2.
#: ``__commit__/`` and ``.markers/`` are metadata and are reported separately.
METADATA_PREFIXES = ("__commit__/", ".markers/")

KIND_LABELS = {
    "det": "deterministic",
    "mean": "ensemble mean",
    "mem": "ensemble member",
    "agg": "aggregate (statistics)",
}


@dataclass
class Group:
    """Byte and object counts for one (variable, kind) pair."""

    objects: int = 0
    bytes: int = 0
    sizes: list[int] = field(default_factory=list)

    def add(self, size: int) -> None:
        self.objects += 1
        self.bytes += size
        self.sizes.append(size)


def classify(key: str) -> tuple[str, str]:
    """Return ``(variable, kind)`` for a store-relative object key.

    Metadata objects are reported under ``("", "metadata")`` so they are visible in the
    totals without being mistaken for forecast data.
    """
    if key.startswith(METADATA_PREFIXES):
        return "", "metadata"
    parts = key.split("/")
    variable = parts[0] if parts else key
    name = parts[-1] if len(parts) > 1 else ""
    for prefix, kind in (
        ("shard.det", "det"),
        ("shard.mean", "mean"),
        ("shard.agg", "agg"),
        ("shard.mem", "mem"),
    ):
        if name.startswith(prefix):
            return variable, kind
    return variable, "v2_unsharded"


def iter_objects(store: str):
    """Yield ``(key, size)`` for every object under the store prefix."""
    if store.startswith("s3://"):
        import s3fs  # type: ignore[import-untyped]

        fs = s3fs.S3FileSystem(anon=False)
        prefix = store[len("s3://") :].rstrip("/")
        for path in fs.find(prefix, detail=True).values():
            yield path["name"][len(prefix) + 1 :], int(path["size"])
    else:
        import os

        root = os.path.abspath(store)
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                yield os.path.relpath(full, root).replace(os.sep, "/"), os.path.getsize(full)


def human(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.3f} {unit}"
    return f"{n} B"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", required=True, help="s3://bucket/prefix or a local path")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="show only the N heaviest variables (0 = all)",
    )
    args = parser.parse_args(argv)

    groups: dict[tuple[str, str], Group] = defaultdict(Group)
    total_objects = 0
    total_bytes = 0
    try:
        for key, size in iter_objects(args.store):
            variable, kind = classify(key)
            groups[(variable, kind)].add(size)
            total_objects += 1
            total_bytes += size
    except Exception as exc:  # noqa: BLE001 - a listing failure must be a non-zero exit
        print(f"error: cannot list {args.store!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not groups:
        print(f"error: no objects under {args.store!r}", file=sys.stderr)
        return 1

    # ---- per (variable, kind) ----
    print(f"store: {args.store}")
    print(f"objects: {total_objects}   total: {human(total_bytes)}")
    print()
    header = f"{'variable':<26} {'kind':<22} {'objects':>9} {'bytes':>12} {'mean':>10} {'share':>7}"
    print(header)
    print("-" * len(header))
    for (variable, kind), g in sorted(groups.items(), key=lambda kv: -kv[1].bytes):
        label = KIND_LABELS.get(kind, kind)
        mean = g.bytes // g.objects if g.objects else 0
        share = 100.0 * g.bytes / total_bytes if total_bytes else 0.0
        print(
            f"{(variable or '(store metadata)'):<26} {label:<22} {g.objects:>9} "
            f"{human(g.bytes):>12} {human(mean):>10} {share:6.2f}%"
        )

    # ---- per variable, all kinds summed: the number capacity planning actually uses ----
    per_var: dict[str, int] = defaultdict(int)
    per_var_objects: dict[str, int] = defaultdict(int)
    for (variable, _kind), g in groups.items():
        per_var[variable] += g.bytes
        per_var_objects[variable] += g.objects

    ranked = sorted(per_var.items(), key=lambda kv: -kv[1])
    if args.top:
        ranked = ranked[: args.top]
    print()
    print(f"{'variable (all kinds)':<26} {'objects':>9} {'bytes':>12} {'share':>7}")
    print("-" * 58)
    for variable, nbytes in ranked:
        share = 100.0 * nbytes / total_bytes if total_bytes else 0.0
        print(
            f"{(variable or '(store metadata)'):<26} {per_var_objects[variable]:>9} "
            f"{human(nbytes):>12} {share:6.2f}%"
        )

    # ---- the outlier signal: spread of per-variable compressed shard size ----
    shard_sizes = [
        g.bytes // g.objects
        for (_v, kind), g in groups.items()
        if kind in ("det", "mem", "mean", "agg") and g.objects
    ]
    if len(shard_sizes) >= 2:
        lo, hi = min(shard_sizes), max(shard_sizes)
        print()
        print(
            f"per-shard compressed size range: {human(lo)} .. {human(hi)} "
            f"({hi / lo:.2f}x spread across variables)"
        )
        print(
            "  -> a whole-cycle projection built from one sampled variable carries at least "
            "this much error; use the per-variable table above instead."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
