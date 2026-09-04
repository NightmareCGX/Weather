"""Pytest configuration for cross-package contract integration tests (Stage 7D).

Adds packages/domain, services/api, and services/ingestion to sys.path so that
contract tests can verify producer/consumer compatibility without contaminating
individual package dependency definitions.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

domain_src = str(ROOT / "packages" / "domain" / "src")
api_src = str(ROOT / "services" / "api" / "src")
ingestion_src = str(ROOT / "services" / "ingestion" / "src")

for p in (domain_src, api_src, ingestion_src):
    if p not in sys.path:
        sys.path.insert(0, p)
