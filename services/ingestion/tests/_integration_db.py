"""Integration-test database selection (test isolation, lifecycle doc §11-10).

Integration tests must NEVER fall back to the live compose database URL. The
historical hardcoded default
(``postgresql://weather_user:weather_password@localhost:5432/weather_db``)
silently pointed local pytest runs at the live database, where destructive
fixtures (``DROP SCHEMA public CASCADE``) and seeded ``model_runs``
``zarr_store_path`` rows pointing at pytest temp dirs polluted production data.

Rule: integration tests run only against an EXPLICITLY configured test
database — ``TEST_DATABASE_URL`` (preferred) or ``DATABASE_URL`` from the
environment. CI sets ``DATABASE_URL`` for its own service containers, so CI
is unaffected. With neither variable set, DB-backed tests skip instead of
guessing.
"""

from __future__ import annotations

import os

import pytest

_SKIP_MESSAGE = (
    "Integration test needs an isolated test database: set TEST_DATABASE_URL "
    "(preferred) or DATABASE_URL. Refusing to guess a default because the old "
    "hardcoded URL pointed at the live compose DB (pytest pollution incident, "
    "lifecycle doc §11 item 10)."
)


def integration_db_url() -> str:
    """Return the explicitly configured integration DB URL, or skip the test."""
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        pytest.skip(_SKIP_MESSAGE)
    return url


def integration_db_url_or_skip_module() -> str:
    """Module-level variant: skip the whole module when no DB URL is configured.

    For test modules that resolve their database URL at import time.
    """
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        pytest.skip(_SKIP_MESSAGE, allow_module_level=True)
    return url
