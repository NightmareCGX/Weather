"""Generation-aware cache-key tests.

A same-set same-cycle data replacement changes the committed-manifest
generation, making old cache entries unreachable (the cache key includes the
generation). A pure catalog retry with unchanged serving state preserves the
generation (same key). No cross-process LRU invalidation is required.
"""

from __future__ import annotations

from api.services.cache import (
    build_ensemble_cache_key,
    build_point_cache_key,
    build_probability_cache_key,
)


def _point_key(gen: str | None) -> str:
    return build_point_cache_key(
        model="gfs",
        latitude=38.5,
        longitude=-106.5,
        resolved_via="coords",
        location_id=None,
        cycle_time="2026-07-22T00:00:00Z",
        serving_generation=gen,
        variables=("temperature_2m",),
        units="metric",
        start_lead_time_hours=None,
        end_lead_time_hours=None,
        cross_cycle=True,
    )


def _prob_key(gen: str | None) -> str:
    return build_probability_cache_key(
        model="gfs",
        latitude=38.5,
        longitude=-106.5,
        variable="temperature_2m",
        threshold=0.0,
        operator="gt",
        lead_time_hours=6,
        threshold_max=None,
        cycle_time="2026-07-22T00:00:00Z",
        serving_generation=gen,
    )


def _ens_key(gen: str | None) -> str:
    return build_ensemble_cache_key(
        model="gefs",
        latitude=38.5,
        longitude=-106.5,
        variable="temperature_2m",
        lead_time_hours=6,
        include_members=False,
        cycle_time="2026-07-22T00:00:00Z",
        serving_generation=gen,
    )


def test_point_generation_change_rejects_old_entry() -> None:
    old = _point_key("gen1")
    new = _point_key("gen2")
    assert old != new  # same-set replacement -> new generation -> unreachable


def test_point_catalog_retry_preserves_generation() -> None:
    assert _point_key("gen1") == _point_key("gen1")  # stable


def test_probability_generation_change_rejects_old_entry() -> None:
    assert _prob_key("gen1") != _prob_key("gen2")


def test_probability_catalog_retry_preserves_generation() -> None:
    assert _prob_key("gen1") == _prob_key("gen1")


def test_ensemble_generation_change_rejects_old_entry() -> None:
    assert _ens_key("gen1") != _ens_key("gen2")


def test_ensemble_catalog_retry_preserves_generation() -> None:
    assert _ens_key("gen1") == _ens_key("gen1")


def test_legacy_null_generation_distinct_from_real() -> None:
    # A legacy token (None) must never collide with a real generation.
    assert _point_key(None) != _point_key("gen1")
