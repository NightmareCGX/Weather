"""Operator CLI for sharded_v2 shadow validation.

Subcommands:
  validate  Re-encode a canonical store's committed shards into a shadow-v2
            store and compare storage/numerical/serving output. Exit 0 only
            when no semantic regression is detected.
  cleanup   Delete a shadow store (guarded to the shadow-v2 namespace only;
            refuses arbitrary production prefixes; supports --older-than-days
            and --dry-run).

The shadow store is never registered in the catalog: it is invisible to
canonical serving selection, the reclamation queue, and tombstones. Its
lifetime is managed exclusively by this tool.

This script sits outside the per-package type-check/test environments on
purpose: the serving comparison needs BOTH the ingestion and api packages,
which no single package CI job installs together (the cross-package contract
suite in ``tests/contracts`` covers it with ``uv sync --all-packages``).

Examples:
  python scripts/shadow_v2.py validate --store s3://weather-data/gfs/2026-09-23/18/cycle.zarr
  python scripts/shadow_v2.py validate --store ./cycle.zarr --sample-points 100
  python scripts/shadow_v2.py cleanup --shadow ./cycle.shadow-v2.zarr --older-than-days 2
  python scripts/shadow_v2.py cleanup --shadow ./cycle.shadow-v2.zarr --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the workspace packages importable when run from a repo checkout without
# an editable install (the uv root venv normally already provides them).
_REPO = Path(__file__).resolve().parent.parent
for _src in (
    _REPO / "packages" / "domain" / "src",
    _REPO / "services" / "ingestion" / "src",
    _REPO / "services" / "api" / "src",
):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from domain.storage_dtype import resolve_storage_dtype  # noqa: E402
from ingestion.core.shadow import (  # noqa: E402
    _CEILING_THRESHOLD_KM,
    _PRECIP_THRESHOLD_MM,
    _iter_local_shards,
    cleanup_shadow_stores,
    compare_stores_numerical,
    default_shadow_store_path,
    write_shadow_store,
)


def compare_serving(
    source_store: str,
    shadow_store: str,
    *,
    lead_time_hours: int = 6,
    sample_points: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare point-interpolation serving output between both stores.

    Uses the same grid sample for both readers; the reader class is chosen per
    store by the committed manifest, exactly as production dispatch would.
    Requires both the ingestion and api packages (script context).
    """
    from api.core.zarr import get_sharded_reader

    rng = np.random.default_rng(seed)
    vars_seen = sorted(_iter_local_shards(source_store).keys())
    lat_indices = rng.integers(0, 720, size=sample_points)
    lon_indices = rng.integers(0, 1440, size=sample_points)
    per_var: dict[str, dict[str, Any]] = {}
    for var_name in vars_seen:
        resolved = resolve_storage_dtype("sharded_v2", var_name, "det")
        member = (
            1
            if any(
                shard.name.startswith("shard.mem")
                for shard in _iter_local_shards(source_store)[var_name]
            )
            else None
        )
        v1_reader = get_sharded_reader(source_store)
        v2_reader = get_sharded_reader(shadow_store)
        max_diff = 0.0
        flips = 0
        n = 0
        for lat_idx, lon_idx in zip(lat_indices, lon_indices):
            a = v1_reader.read_point_value(
                var_name,
                member=member,
                lead_time_hours=lead_time_hours,
                lat_idx=int(lat_idx),
                lon_idx=int(lon_idx),
            )
            b = v2_reader.read_point_value(
                var_name,
                member=member,
                lead_time_hours=lead_time_hours,
                lat_idx=int(lat_idx),
                lon_idx=int(lon_idx),
            )
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            n += 1
            max_diff = max(max_diff, abs(b - a))
            if var_name == "precipitation_amount_3h":
                flips += int(
                    (b <= _PRECIP_THRESHOLD_MM) != (a <= _PRECIP_THRESHOLD_MM)
                )
            if var_name == "cloud_ceiling":
                flips += int(
                    (b >= _CEILING_THRESHOLD_KM) != (a >= _CEILING_THRESHOLD_KM)
                )
        per_var[var_name] = {
            "points_compared": n,
            "max_abs_diff": max_diff,
            "threshold_flips": flips,
            "storage_dtype": resolved.str,
        }
    return {"sample_points": sample_points, "per_variable": per_var}


def run_shadow_validation(
    source_store: str,
    shadow_store: str | None = None,
    *,
    lead_time_hours: int = 6,
    sample_points: int = 25,
) -> dict[str, Any]:
    """Full shadow validation: re-encode, compare storage/numerical/serving.

    Returns a JSON-serializable report. ``semantic_regression_free`` is False
    unless every f32 exception is byte-identical and both threshold predicates
    show zero flips across the numerical and serving comparisons.
    """
    shadow = shadow_store or default_shadow_store_path(source_store)
    storage = write_shadow_store(source_store, shadow)
    numerical = compare_stores_numerical(source_store, shadow)
    serving = compare_serving(
        source_store, shadow, lead_time_hours=lead_time_hours, sample_points=sample_points
    )

    semantic_regression_free = True
    for report in numerical:
        if report.max_abs_diff > report.tolerance:
            semantic_regression_free = False
        if report.precip_threshold_flips or report.ceiling_threshold_flips:
            semantic_regression_free = False
        if report.notes:
            semantic_regression_free = False
    for var_report in serving["per_variable"].values():
        if var_report["threshold_flips"]:
            semantic_regression_free = False

    return {
        "storage": storage,
        "numerical": [vars(r) for r in numerical],
        "serving": serving,
        "semantic_regression_free": semantic_regression_free,
    }


def _cmd_validate(args: argparse.Namespace) -> int:
    report = run_shadow_validation(
        args.store, args.out, lead_time_hours=args.lead, sample_points=args.sample_points
    )
    print(json.dumps(report, indent=2))
    if not report["semantic_regression_free"]:
        print(
            "SHADOW VALIDATION FAILED: semantic regression detected "
            "(see per-variable flips/notes above).",
            file=sys.stderr,
        )
        return 1
    if not report["storage"]["shards_reencoded"]:
        print("SHADOW VALIDATION INCONCLUSIVE: no shards were re-encoded.", file=sys.stderr)
        return 2
    print("SHADOW VALIDATION PASSED: no semantic regression detected.")
    return 0


def _cmd_cleanup(args: argparse.Namespace) -> int:
    result = cleanup_shadow_stores(
        args.shadow,
        older_than_days=args.older_than_days,
        dry_run=not args.apply,
    )
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate", help="Run shadow-v2 validation against a canonical store."
    )
    p_validate.add_argument(
        "--store", required=True, help="Canonical store path (local dir or s3:// URL)."
    )
    p_validate.add_argument(
        "--out",
        default=None,
        help="Shadow store path (default: derived shadow-v2 namespace).",
    )
    p_validate.add_argument(
        "--lead", type=int, default=6, help="Lead used for serving point samples."
    )
    p_validate.add_argument(
        "--sample-points", type=int, default=25, help="Point-serving comparison samples."
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_cleanup = sub.add_parser("cleanup", help="Delete a shadow store (namespace-guarded).")
    p_cleanup.add_argument(
        "--shadow", required=True, help="Shadow store path (must contain shadow-v2)."
    )
    p_cleanup.add_argument(
        "--older-than-days", type=float, default=None, help="Only delete if older than N days."
    )
    p_cleanup.add_argument(
        "--apply", action="store_true", help="Actually delete (default is dry-run)."
    )
    p_cleanup.set_defaults(func=_cmd_cleanup)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
