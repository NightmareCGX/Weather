"""Unit tests for the vector-field cache prewarm pass logic.

Covers the serving-window grid, warm-entry skipping, per-pass compute budget,
throttling, and per-entry failure isolation (404 skips and unexpected errors
must never abort the pass or raise).
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from api.services import vector_field
from api.services import vector_prewarm
from api.services.resolver import ResolvedForecastSource
from api.services.vector_field import (
    _vector_cache,
    _vector_cache_key,
    _vector_cache_set,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 13, 7, 15, tzinfo=UTC)  # floors to 06:00 grid


def _source(valid_time: datetime, lead: int = 6) -> ResolvedForecastSource:
    return ResolvedForecastSource(
        model="gfs",
        valid_time=valid_time,
        cycle_time=valid_time - timedelta(hours=lead),
        lead_time_hours=lead,
        run_id="run-1",
        store_path="s3://bucket/run",
        serving_generation="gen1",
        member_indices=None,
    )


def _mirror_cache_key(model: str, source: ResolvedForecastSource) -> tuple[object, ...]:
    """Build the same key the pass probes, for test-side cache prepopulation."""
    return _vector_cache_key(
        model,
        "wind_10m",
        source.lead_time_hours,
        source.cycle_time.isoformat().replace("+00:00", "Z"),
        source.serving_generation,
        2,
        valid_time=source.valid_time.isoformat().replace("+00:00", "Z"),
        member_fingerprint=source.member_fingerprint,
    )


class _FakeSession:
    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clean_vector_cache(monkeypatch):
    """Isolate each test from the L1 dict and the (possibly live) shared Redis L2.

    These tests exercise pass logic against the process-local layer only; the
    Redis tier has its own tests in test_vector_cache_redis.py.
    """
    _vector_cache.clear()
    monkeypatch.setattr(vector_field, "_get_redis_client", lambda: None)
    yield
    _vector_cache.clear()


def test_serving_window_grid_alignment():
    """Valid times start on the 3h serving grid and span the horizon inclusive."""
    times = vector_prewarm.serving_window_valid_times(NOW, horizon_hours=6)
    assert times[0] == datetime(2026, 9, 13, 6, 0, tzinfo=UTC)
    assert times == [
        datetime(2026, 9, 13, 6, 0, tzinfo=UTC),
        datetime(2026, 9, 13, 9, 0, tzinfo=UTC),
        datetime(2026, 9, 13, 12, 0, tzinfo=UTC),
    ]

    full = vector_prewarm.serving_window_valid_times(NOW, horizon_hours=48)
    assert len(full) == 17
    assert full[-1] == datetime(2026, 9, 15, 6, 0, tzinfo=UTC)


def test_prewarm_computes_missing_entries_with_budget_and_throttle(monkeypatch):
    """Cold entries render (budget-capped, throttled); warm entries are free skips."""
    calls: list[datetime] = []
    sleeps: list[float] = []

    def fake_resolve(db, model, valid_time, now):
        return _source(valid_time)

    def fake_render(db, model, valid_time, now):
        calls.append(valid_time)
        return b"payload"

    monkeypatch.setattr(vector_prewarm, "_resolve_valid_time", fake_resolve)
    monkeypatch.setattr(vector_prewarm, "_render_valid_time", fake_render)
    monkeypatch.setattr(vector_prewarm, "_open_session", _FakeSession)

    result = vector_prewarm.run_prewarm_pass(
        models=["gfs"],
        now=NOW,
        horizon_hours=6,
        max_computes=2,
        throttle_seconds=2.0,
        sleeper=sleeps.append,
    )

    assert result.considered == 3
    assert result.computed == 2  # budget caps the third
    assert result.already_cached == 0
    assert result.failed == 0
    assert len(calls) == 2
    assert sleeps == [2.0, 2.0]


def test_prewarm_skips_cached_entries_without_rendering(monkeypatch):
    """A cache-warm pass renders nothing and reports every entry as already cached."""
    renders: list[datetime] = []

    def fake_resolve(db, model, valid_time, now):
        return _source(valid_time)

    def fake_render(db, model, valid_time, now):
        renders.append(valid_time)
        return b"payload"

    for valid_time in vector_prewarm.serving_window_valid_times(NOW, horizon_hours=6):
        _vector_cache_set(_mirror_cache_key("gfs", _source(valid_time)), b"payload")

    monkeypatch.setattr(vector_prewarm, "_resolve_valid_time", fake_resolve)
    monkeypatch.setattr(vector_prewarm, "_render_valid_time", fake_render)
    monkeypatch.setattr(vector_prewarm, "_open_session", _FakeSession)

    result = vector_prewarm.run_prewarm_pass(
        models=["gfs"], now=NOW, horizon_hours=6, throttle_seconds=0.0
    )

    assert result.considered == 3
    assert result.already_cached == 3
    assert result.computed == 0
    assert renders == []


def test_prewarm_isolates_unavailable_and_failed_entries(monkeypatch):
    """HTTP 404 valid times are skipped and unexpected errors never abort the pass."""
    times = vector_prewarm.serving_window_valid_times(NOW, horizon_hours=6)

    def fake_resolve(db, model, valid_time, now):
        if valid_time == times[0]:
            raise HTTPException(status_code=404, detail="unavailable")
        if valid_time == times[1]:
            raise RuntimeError("boom")
        return _source(valid_time)

    def fake_render(db, model, valid_time, now):
        return b"payload"

    monkeypatch.setattr(vector_prewarm, "_resolve_valid_time", fake_resolve)
    monkeypatch.setattr(vector_prewarm, "_render_valid_time", fake_render)
    monkeypatch.setattr(vector_prewarm, "_open_session", _FakeSession)

    result = vector_prewarm.run_prewarm_pass(
        models=["gfs"], now=NOW, horizon_hours=6, throttle_seconds=0.0
    )

    assert result.skipped_unavailable == 1
    assert result.failed == 1
    assert result.computed == 1


def test_prewarm_budget_stops_across_models(monkeypatch):
    """Once the per-pass budget is spent, later models are not touched."""
    resolved_models: list[str] = []

    def fake_resolve(db, model, valid_time, now):
        resolved_models.append(model)
        return _source(valid_time)

    def fake_render(db, model, valid_time, now):
        return b"payload"

    monkeypatch.setattr(vector_prewarm, "_resolve_valid_time", fake_resolve)
    monkeypatch.setattr(vector_prewarm, "_render_valid_time", fake_render)
    monkeypatch.setattr(vector_prewarm, "_open_session", _FakeSession)

    result = vector_prewarm.run_prewarm_pass(
        models=["gfs", "gefs"], now=NOW, horizon_hours=6, max_computes=1,
        throttle_seconds=0.0,
    )

    assert result.computed == 1
    assert set(resolved_models) == {"gfs"}
