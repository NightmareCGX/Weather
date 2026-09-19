#!/usr/bin/env python
"""Validation-environment probe for the Windows + Linux engineering gate.

The repository's gate (CLAUDE.md §5) requires validating every affected scope on both the
Windows host and a Linux/CI-equivalent environment, and forbids counting a skipped check
as green. This script exists so that readiness is *proven* rather than assumed: a gap in
the Linux side (no daemon, missing mount scope, wrong interpreter) shows up here instead
of silently producing an empty test run.

Two modes:

  python scripts/check_validation_env.py
      Structural preflight plus the Windows tool chain. Fast, no network beyond the
      ``python:3.12-slim`` image if it is not yet cached.

  python scripts/check_validation_env.py --deep --scope tests/contracts
      Additionally builds a Linux virtual environment inside a ``python:3.12-slim``
      container from the bind-mounted ``uv.lock`` and runs the given scope there. This is
      the CI-equivalent run; it needs network access on first use.

Every line is ``PASS``, ``FAIL`` or ``SKIP`` with a reason, and any ``FAIL`` exits non-zero.
A ``SKIP`` never counts as a pass.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Host paths bind-mounted into the Linux container.
MOUNT_SCOPES = (
    "packages",
    "services",
    "tests",
    "scripts",
    "monitoring",
    "docs",
    "pyproject.toml",
    "uv.lock",
)

EXPECTED_PYTHON = "3.12"
CONTAINER_IMAGE = "python:3.12-slim"

#: uv version pinned by CLAUDE.md §4 as the CI toolchain.
UV_VERSION = "0.12.13"


def _run(cmd: list[str], *, timeout: int = 900) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def _mount_args() -> list[str]:
    """Bind the scopes READ-ONLY: the container must never mutate the working tree."""
    args: list[str] = []
    for scope in MOUNT_SCOPES:
        args += ["-v", f"{REPO_ROOT / scope}:/repo/{scope}:ro"]
    return args


def check_windows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    rows.append(
        (
            "Windows: interpreter",
            "PASS" if version == EXPECTED_PYTHON else "FAIL",
            f"python {version} (expected {EXPECTED_PYTHON})",
        )
    )
    for tool in ("ruff", "mypy", "pytest"):
        code, out = _run([sys.executable, "-m", tool, "--version"], timeout=120)
        rows.append(
            (
                f"Windows: {tool}",
                "PASS" if code == 0 else "FAIL",
                out.splitlines()[0] if out else "not importable",
            )
        )
    code, out = _run(["uv", "--version"], timeout=120)
    rows.append(
        (
            "Windows: uv",
            "PASS" if code == 0 else "FAIL",
            out.splitlines()[0] if out else "uv not on PATH",
        )
    )
    return rows


def check_linux_preflight() -> tuple[list[tuple[str, str, str]], bool]:
    rows: list[tuple[str, str, str]] = []
    if shutil.which("docker") is None:
        rows.append(("Linux: docker CLI", "FAIL", "docker not on PATH"))
        return rows, False

    code, out = _run(["docker", "info", "--format", "{{.OSType}}"], timeout=120)
    linux_ok = code == 0 and out.strip() == "linux"
    rows.append(
        (
            "Linux: docker daemon",
            "PASS" if linux_ok else "FAIL",
            out.strip() or "daemon unreachable",
        )
    )
    if not linux_ok:
        return rows, False

    missing = [s for s in MOUNT_SCOPES if not (REPO_ROOT / s).exists()]
    rows.append(
        (
            "Linux: mount scopes",
            "PASS" if not missing else "FAIL",
            "all present" if not missing else f"missing {missing}",
        )
    )

    code, out = _run(
        ["docker", "run", "--rm", CONTAINER_IMAGE, "python", "-c",
         "import sys;print('%d.%d' % sys.version_info[:2])"],
        timeout=600,
    )
    rows.append(
        (
            "Linux: container interpreter",
            "PASS" if code == 0 and out.endswith(EXPECTED_PYTHON) else "FAIL",
            out.splitlines()[-1] if out else "container did not run",
        )
    )
    return rows, not missing and code == 0


def run_linux_scope(scope: str, test_args: list[str]) -> list[tuple[str, str, str]]:
    """Build a Linux venv from the mounted lock and run ``scope`` inside the container."""
    # uv is copied in from the official image so the container needs no distro packages.
    # The published uv image is ``FROM scratch`` -- it has no shell -- so uv is installed
    # into python:3.12-slim instead, matching the runtime the service images use.
    # uv writes the environment to /opt/venv (UV_PROJECT_ENVIRONMENT), never into the
    # read-only mounted repo, and --frozen forbids touching uv.lock.
    script = (
        "set -e; "
        "pip install --quiet --disable-pip-version-check "
        f"'uv=={UV_VERSION}'; "
        "cd /repo; "
        "export UV_PROJECT_ENVIRONMENT=/opt/venv; "
        "uv sync --all-packages --frozen --no-progress; "
        f"/opt/venv/bin/python -m pytest {scope} -q {' '.join(test_args)}"
    )
    cmd = ["docker", "run", "--rm"]
    cmd += _mount_args()
    cmd += ["--workdir", "/repo", CONTAINER_IMAGE, "sh", "-c", script]
    code, out = _run(cmd, timeout=1800)
    if code == 0:
        # pytest -q ends with its summary line; surface that, not the resolver noise.
        summary = [ln for ln in out.splitlines() if " passed" in ln or " failed" in ln]
        detail = summary[-1] if summary else "ok"
    else:
        detail = "\n".join(out.splitlines()[-12:])
    return [(f"Linux: pytest {scope}", "PASS" if code == 0 else "FAIL", detail)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also build a Linux venv and run a scope in the container (needs network)",
    )
    parser.add_argument(
        "--scope",
        default="tests/contracts",
        help="pytest target for --deep (default: tests/contracts)",
    )
    args = parser.parse_args(argv)

    rows = check_windows()
    linux_rows, linux_ready = check_linux_preflight()
    rows += linux_rows

    if args.deep:
        if linux_ready:
            rows += run_linux_scope(args.scope, [])
        else:
            rows.append(
                (
                    f"Linux: pytest {args.scope}",
                    "SKIP",
                    "Linux prerequisites unmet; not counted as a pass",
                )
            )

    width = max(len(label) for label, _, _ in rows)
    failures = 0
    for label, status, detail in rows:
        if status == "FAIL":
            failures += 1
        first = detail.splitlines()[0] if detail else ""
        print(f"{status}  {label:<{width}}  {first[:160]}")
        for extra in detail.splitlines()[1:]:
            print(f"      {'':<{width}}  {extra[:160]}")

    print()
    if failures:
        print(f"{failures} FAIL -- validation environment is NOT ready.")
        return 1
    if any(status == "SKIP" for _, status, _ in rows):
        print("Preflight clean, but a SKIP remains; that scope is NOT validated.")
        return 0
    if not args.deep:
        print(
            "Preflight clean. Structural readiness only -- run with --deep to actually "
            "execute a scope on the Linux side."
        )
        return 0
    print("Windows + Linux/CI-equivalent validation ran and passed for the given scope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
