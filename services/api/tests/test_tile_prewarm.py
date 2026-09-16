"""Unit tests for the low-zoom map tile cache prewarm pass logic (P3).

Covers:
1. Low-zoom tile coordinate generator (Zoom 0..2 = 21 tiles).
2. Cold entries compute all 21 tiles with throttling; warm entries are free skips.
3. Multi-worker deduplication via Redis atomic claim (NX EX).
4. Graceful handling of unavailable runs (404 skipped without raising).
5. Per-tile error isolation (failures do not abort the pass).
6. Gateway HTTP loopback path.
7. Admin endpoint POST /v1/admin/prewarm-tiles.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.main import app
from api.services import tile_prewarm
from api.services.resolver import ResolvedForecastSource
from api.services.tile_prewarm import (
    generate_low_zoom_tile_coords,
    run_tile_prewarm_pass,
)
from api.services.tiles import (
    _tile_cache,
    _tile_cache_key,
    _tile_cache_set,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _mock_source(model: str = "gfs", variable: str = "temperature_2m", lead: int = 0) -> ResolvedForecastSource:
    valid_dt = NOW
    cycle_dt = NOW - timedelta(hours=lead)
    return ResolvedForecastSource(
        model=model,
        valid_time=valid_dt,
        cycle_time=cycle_dt,
        lead_time_hours=lead,
        run_id="run-p3-test",
        store_path="s3://weather-data/test.zarr",
        serving_generation="gen-p3",
        member_indices=None,
    )


@pytest.fixture(autouse=True)
def _clean_tile_cache():
    _tile_cache.clear()
    yield
    _tile_cache.clear()


def test_generate_low_zoom_tile_coords():
    """Zoom 0..2 must produce exactly 21 coordinates covering the globe."""
    coords = generate_low_zoom_tile_coords(max_zoom=2)
    assert len(coords) == 21
    # Zoom 0: 1 tile
    z0 = [c for c in coords if c[0] == 0]
    assert z0 == [(0, 0, 0)]
    # Zoom 1: 4 tiles
    z1 = [c for c in coords if c[0] == 1]
    assert len(z1) == 4
    # Zoom 2: 16 tiles
    z2 = [c for c in coords if c[0] == 2]
    assert len(z2) == 16


def test_tile_prewarm_computes_all_21_tiles(monkeypatch):
    """Cold entries compute all 21 tiles (Zoom 0..2) and write to cache."""
    rendered_tiles: list[tuple[int, int, int]] = []

    def mock_resolve(db, model, valid_time, variable, now):
        return _mock_source(model=model, variable=variable)

    def mock_render(db, *, model, variable, level, zoom, x, y, valid_time, initial_time, now):
        rendered_tiles.append((zoom, x, y))
        cache_key = _tile_cache_key(
            model, variable, level, zoom, x, y, 0, initial_time, "gen-p3", valid_time
        )
        _tile_cache_set(cache_key, b"png_data")
        return b"png_data"

    monkeypatch.setattr("api.services.resolver.resolve_valid_time_source", mock_resolve)
    monkeypatch.setattr("api.services.tiles.render_tile_png", mock_render)
    monkeypatch.setattr(tile_prewarm, "_try_claim_tile_prewarm", lambda key: True)

    result = run_tile_prewarm_pass(
        now=NOW,
        models=["gfs"],
        variables=["temperature_2m"],
        max_zoom=2,
        throttle_seconds=0.0,
    )

    assert result.considered == 21
    assert result.computed == 21
    assert result.already_cached == 0
    assert result.failed == 0
    assert len(rendered_tiles) == 21

    # Second pass immediately sees all 21 tiles already cached!
    result2 = run_tile_prewarm_pass(
        now=NOW,
        models=["gfs"],
        variables=["temperature_2m"],
        max_zoom=2,
        throttle_seconds=0.0,
    )
    assert result2.considered == 21
    assert result2.computed == 0
    assert result2.already_cached == 21


def test_tile_prewarm_redis_claim_deduplication(monkeypatch):
    """If another worker already claimed this cycle, pass skips cleanly without rendering."""
    render_called = False

    def mock_resolve(db, model, valid_time, variable, now):
        return _mock_source(model=model, variable=variable)

    def mock_render(*args, **kwargs):
        nonlocal render_called
        render_called = True
        return b"png"

    monkeypatch.setattr("api.services.resolver.resolve_valid_time_source", mock_resolve)
    monkeypatch.setattr("api.services.tiles.render_tile_png", mock_render)
    # Simulate Redis SET NX returning False (already claimed by Worker 1)
    monkeypatch.setattr(tile_prewarm, "_try_claim_tile_prewarm", lambda key: False)

    result = run_tile_prewarm_pass(
        now=NOW,
        models=["gfs"],
        variables=["temperature_2m"],
        max_zoom=2,
    )

    assert result.computed == 0
    assert result.already_cached == 21
    assert render_called is False


def test_tile_prewarm_skips_unavailable_source_gracefully(monkeypatch):
    """When a model/variable is not ready yet (404), tiles are skipped without failure."""
    def mock_resolve(db, model, valid_time, variable, now):
        raise HTTPException(status_code=404, detail="Not ready")

    monkeypatch.setattr("api.services.resolver.resolve_valid_time_source", mock_resolve)

    result = run_tile_prewarm_pass(
        now=NOW,
        models=["gfs"],
        variables=["temperature_2m"],
        max_zoom=2,
    )

    assert result.skipped_unavailable == 21
    assert result.computed == 0
    assert result.failed == 0


def test_tile_prewarm_handles_render_error_gracefully(monkeypatch):
    """An unexpected render failure on 1 tile does not abort the remaining 20 tiles."""
    def mock_resolve(db, model, valid_time, variable, now):
        return _mock_source(model=model, variable=variable)

    def mock_render(db, *, model, variable, level, zoom, x, y, valid_time, initial_time, now):
        if zoom == 0:
            raise RuntimeError("Disk I/O error on root tile")
        return b"png"

    monkeypatch.setattr("api.services.resolver.resolve_valid_time_source", mock_resolve)
    monkeypatch.setattr("api.services.tiles.render_tile_png", mock_render)
    monkeypatch.setattr(tile_prewarm, "_try_claim_tile_prewarm", lambda key: True)

    result = run_tile_prewarm_pass(
        now=NOW,
        models=["gfs"],
        variables=["temperature_2m"],
        max_zoom=2,
        throttle_seconds=0.0,
    )

    assert result.considered == 21
    assert result.computed == 20  # 20 tiles succeeded
    assert result.failed == 1      # zoom 0 failed gracefully


def test_tile_prewarm_gateway_url_path(monkeypatch):
    """When gateway_url is configured, tiles are requested via the gateway HTTP helper."""
    http_calls: list[dict[str, object]] = []

    def mock_resolve(db, model, valid_time, variable, now):
        return _mock_source(model=model, variable=variable)

    def mock_gateway_render(gw_url, *, model, variable, zoom, x, y, valid_time, initial_time):
        http_calls.append({"gw": gw_url, "z": zoom, "x": x, "y": y})

    monkeypatch.setattr("api.services.resolver.resolve_valid_time_source", mock_resolve)
    monkeypatch.setattr(tile_prewarm, "_render_tile_via_gateway", mock_gateway_render)
    monkeypatch.setattr(tile_prewarm, "_try_claim_tile_prewarm", lambda key: True)

    result = run_tile_prewarm_pass(
        now=NOW,
        models=["gfs"],
        variables=["temperature_2m"],
        max_zoom=1,  # z=0 (1) + z=1 (4) = 5 tiles
        gateway_url="http://weather_gateway",
        throttle_seconds=0.0,
    )

    assert result.considered == 5
    assert result.computed == 5
    assert len(http_calls) == 5
    assert all(c["gw"] == "http://weather_gateway" for c in http_calls)


def test_admin_endpoint_trigger_tile_prewarm(monkeypatch):
    """POST /v1/admin/prewarm-tiles triggers one prewarm pass and returns counters."""
    def mock_pass(**kwargs):
        return tile_prewarm.TilePrewarmPassResult(
            considered=21,
            already_cached=0,
            computed=21,
            skipped_unavailable=0,
            failed=0,
        )

    monkeypatch.setattr("api.services.tile_prewarm.run_tile_prewarm_pass", mock_pass)

    client = TestClient(app)
    response = client.post("/v1/admin/prewarm-tiles")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"]["considered"] == 21
    assert data["data"]["computed"] == 21
    assert data["data"]["failed"] == 0
