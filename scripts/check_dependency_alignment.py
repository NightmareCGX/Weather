#!/usr/bin/env python3
"""Shared runtime dependency alignment CI gate.

Verifies that any third-party runtime package declared in two or more active
workspace packages shares the exact same version constraint expression across
all PEP 621 manifests. This guards constraint *ranges*; resolved-version
consistency is guaranteed structurally by the single root ``uv.lock`` and
enforced in CI via ``uv lock --check``.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

MANIFESTS = [
    WORKSPACE_ROOT / "pyproject.toml",
    WORKSPACE_ROOT / "packages" / "domain" / "pyproject.toml",
    WORKSPACE_ROOT / "packages" / "config" / "pyproject.toml",
    WORKSPACE_ROOT / "packages" / "contracts" / "pyproject.toml",
    WORKSPACE_ROOT / "services" / "api" / "pyproject.toml",
    WORKSPACE_ROOT / "services" / "ingestion" / "pyproject.toml",
]

# python is implicit (requires-python); the weather-platform-* names are
# intra-workspace edges (resolved via [tool.uv.sources] workspace = true),
# not third-party constraints.
EXEMPT_DEPS = {
    "weather-platform-domain",
    "weather-platform-contracts",
    "weather-platform-config",
    "weather-platform-ingestion",
    "weather-platform-api",
}

# PEP 508 requirement: name (with optional extras) followed by the constraint
# expression and/or environment marker, e.g. "numpy>=2.0.0,<3.0.0".
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$")


def _split_requirement(req: str) -> tuple[str, str] | None:
    match = _REQ_NAME_RE.match(req)
    if not match:
        return None
    name = match.group(1).lower().replace("_", "-")
    constraint = match.group(3).strip()
    return name, constraint


def check_alignment() -> int:
    manifest_deps: dict[str, dict[str, str]] = {}

    for manifest_path in MANIFESTS:
        if not manifest_path.exists():
            continue
        rel_path = manifest_path.relative_to(WORKSPACE_ROOT).as_posix()
        with open(manifest_path, "rb") as fh:
            data = tomllib.load(fh)

        requirements = data.get("project", {}).get("dependencies", [])
        normalized: dict[str, str] = {}
        for req in requirements:
            if not isinstance(req, str):
                continue
            parsed = _split_requirement(req)
            if parsed is None:
                continue
            dep, constraint = parsed
            if dep in EXEMPT_DEPS:
                continue
            normalized[dep] = constraint
        manifest_deps[rel_path] = normalized

    # Map each dependency to {project_path: constraint}
    occurrences: dict[str, dict[str, str]] = {}
    for proj, deps in manifest_deps.items():
        for dep, constraint in deps.items():
            occurrences.setdefault(dep, {})[proj] = constraint

    drift_errors: list[str] = []
    aligned_count = 0

    for dep, proj_map in sorted(occurrences.items()):
        if len(proj_map) < 2:
            continue
        constraints = set(proj_map.values())
        if len(constraints) > 1:
            details = "\n".join(f"    {p}: {c}" for p, c in sorted(proj_map.items()))
            drift_errors.append(f"  - '{dep}' has drifted version constraints:\n{details}")
        else:
            aligned_count += 1

    if drift_errors:
        print("ERROR: Shared dependency version drift detected!\n", file=sys.stderr)
        print("\n".join(drift_errors), file=sys.stderr)
        print("Expected identical repository-wide constraints for all shared dependencies.", file=sys.stderr)
        return 1

    print(f"PASS: {aligned_count} shared runtime dependencies are aligned across {len(manifest_deps)} packages.")
    return 0


if __name__ == "__main__":
    sys.exit(check_alignment())
