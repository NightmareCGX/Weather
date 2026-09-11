"""Authoritative current-time provider for API serving paths.

Provides the reference current UTC wall-clock time for determining the
authoritative forecast serving-window left boundary (Data Lifecycle V3 Phase 1).
Supports test overrides via WEATHER_SIMULATED_NOW environment variable or
FastAPI dependency injection (app.dependency_overrides[get_current_time]).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def get_current_time() -> datetime:
    """Return the current reference datetime in UTC.

    In production, returns ``datetime.now(timezone.utc)``.
    If the ``WEATHER_SIMULATED_NOW`` environment variable is set (ISO 8601 string),
    it parses and returns that simulated time for reproducible testing.
    """
    env_time = os.getenv("WEATHER_SIMULATED_NOW")
    if env_time:
        parsed = datetime.fromisoformat(env_time.strip())
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)
