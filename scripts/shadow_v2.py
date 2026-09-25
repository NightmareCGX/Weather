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


def _cmd_validate(args: argparse.Namespace) -> int:
    from ingestion.core.shadow import run_shadow_validation

    report = run_shadow_validation(args.store, args.out, sample_points=args.sample_points)
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
    from ingestion.core.shadow import cleanup_shadow_stores

    result = cleanup_shadow_stores(
        args.shadow,
        older_than_days=args.older_than_days,
        dry_run=not args.apply,
    )
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Run shadow-v2 validation against a canonical store.")
    p_validate.add_argument("--store", required=True, help="Canonical store path (local dir or s3:// URL).")
    p_validate.add_argument("--out", default=None, help="Shadow store path (default: derived shadow-v2 namespace).")
    p_validate.add_argument("--sample-points", type=int, default=25, help="Point-serving comparison samples.")
    p_validate.set_defaults(func=_cmd_validate)

    p_cleanup = sub.add_parser("cleanup", help="Delete a shadow store (namespace-guarded).")
    p_cleanup.add_argument("--shadow", required=True, help="Shadow store path (must contain shadow-v2).")
    p_cleanup.add_argument("--older-than-days", type=float, default=None, help="Only delete if older than N days.")
    p_cleanup.add_argument("--apply", action="store_true", help="Actually delete (default is dry-run).")
    p_cleanup.set_defaults(func=_cmd_cleanup)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
