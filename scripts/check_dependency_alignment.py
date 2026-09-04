#!/usr/bin/env python3
"""Shared runtime dependency alignment CI gate.

Verifies that any third-party runtime package declared in two or more active
workspace packages shares the exact same version constraint expression across
all projects.
"""

from __future__ import annotations

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

EXEMPT_DEPS = {"python", "weather-platform-domain"}


def check_alignment() -> int:
    manifest_deps: dict[str, dict[str, str]] = {}

    for manifest_path in MANIFESTS:
        if not manifest_path.exists():
            continue
        rel_path = manifest_path.relative_to(WORKSPACE_ROOT).as_posix()
        with open(manifest_path, "rb") as fh:
            data = tomllib.load(fh)

        main_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        normalized: dict[str, str] = {}
        for dep, spec in main_deps.items():
            if isinstance(spec, str):
                normalized[dep.lower()] = spec
            elif isinstance(spec, dict) and "version" in spec:
                normalized[dep.lower()] = str(spec["version"])
        manifest_deps[rel_path] = normalized

    # Map each dependency to {project_path: constraint}
    occurrences: dict[str, dict[str, str]] = {}
    for proj, deps in manifest_deps.items():
        for dep, constraint in deps.items():
            if dep in EXEMPT_DEPS:
                continue
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
        print("\nExpected identical repository-wide constraints for all shared dependencies.", file=sys.stderr)
        return 1

    print(f"PASS: {aligned_count} shared runtime dependencies are aligned across {len(manifest_deps)} packages.")
    return 0


if __name__ == "__main__":
    sys.exit(check_alignment())
