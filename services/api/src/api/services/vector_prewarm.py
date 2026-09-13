"""Background warm-up of the wind vector-field serving cache.

The vector field payload is expensive to compute cold (full-globe u/v shard
reads per valid time), and the serving cache
(:mod:`api.services.vector_field`: process-local L1 + shared Redis L2) is
filled passively — so after each forecast-cycle publication the first real
user of every valid time paid the full computation. This module closes that
gap with a best-effort background scan that keeps the serving window warm
proactively. The shared Redis L2 makes the warm-up fleet-wide: a payload
computed by any worker (or this loop) is immediately visible to all others.

Design:

* A lifespan-owned asyncio task runs a scan pass every
  ``API_VECTOR_PREWARM_INTERVAL_SECONDS``. A pass enumerates the serving
  window's 3h valid times over ``API_VECTOR_PREWARM_HORIZON_HOURS`` for each
  configured model, resolves each valid time through the **same resolver the
  endpoint uses** (``resolve_valid_time_source``), and computes only the
  cache-missing entries by calling ``render_vector_field_binary`` — reusing
  its internal cache write. Serving splices per valid time (newest cycle
  wins), so resolving per valid time tracks cycle publication incrementally,
  including partial ingest waves, without any ingestion-side notification.
* Passes are bounded (``API_VECTOR_PREWARM_MAX_COMPUTES_PER_PASS``) and
  throttled (``API_VECTOR_PREWARM_COMPUTE_THROTTLE_SECONDS`` between
  computes) so a fresh cycle never saturates S3/disk bandwidth; the next pass
  picks up the remainder. Cache hits make warm passes nearly free.
* Heavy compute runs in a worker thread (``asyncio.to_thread`` at the loop
  level) so the event loop is never blocked. All reads participate in the
  SHARED store gate, so prewarm never blocks or starves online reads.
* Failures are logged per entry and never fatal: a prewarm problem must not
  affect serving. With uvicorn ``--workers > 1`` each process prewarms its
  own cache from the same catalog state (duplicated I/O, bounded by the
  pass budget); scale ``API_VECTOR_PREWARM_*`` accordingly.

Manual trigger: ``POST /v1/admin/prewarm-vector`` runs one pass on demand.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from domain.temporal import serving_start_valid_time

from api.services.vector_field import (
    _vector_cache_get,
    _vector_cache_key,
    render_vector_field_binary,
)

logger = logging.getLogger(__name__)

#: Serving grid cadence (hours) — matches the canonical valid-time grid.
PREWARM_CADENCE_HOURS = 3


@dataclass(frozen=True)
class PrewarmPassResult:
    """Outcome counters of one prewarm scan pass."""

    #: Valid times examined across all models.
    considered: int
    #: Entries already present in the process cache (free skips).
    already_cached: int
    #: Entries computed and written to the cache this pass.
    computed: int
    #: Entries skipped because no source serves the valid time (404).
    skipped_unavailable: int
    #: Entries that raised unexpectedly during resolve/render.
    failed: int


def _resolve_valid_time(db: Any, model: str, valid_time: datetime, now: datetime | None) -> Any:
    """Indirection over ``resolve_valid_time_source`` (monkeypatch seam for tests)."""
    from api.services.resolver import resolve_valid_time_source

    return resolve_valid_time_source(db, model, valid_time, variable="wind_10m", now=now)


def _render_valid_time(db: Any, model: str, valid_time: datetime, now: datetime | None) -> bytes:
    """Indirection over ``render_vector_field_binary`` (monkeypatch seam for tests)."""
    return render_vector_field_binary(
        db, model=model, valid_time=valid_time.isoformat(), now=now
    )


def _open_session() -> Any:
    """Indirection over ``SessionLocal`` (monkeypatch seam for tests)."""
    from api.core.database import SessionLocal

    return SessionLocal()


def serving_window_valid_times(
    now: datetime,
    horizon_hours: int,
    cadence_hours: int = PREWARM_CADENCE_HOURS,
) -> list[datetime]:
    """List the serving grid's valid times from the window start through the horizon."""
    start = serving_start_valid_time(now)
    steps = max(0, int(horizon_hours // cadence_hours))
    return [start + timedelta(hours=cadence_hours * i) for i in range(steps + 1)]


def _cache_entry_cached(model: str, source: Any) -> bool:
    """Mirror the endpoint's valid_time-path cache key and probe the process cache."""
    key = _vector_cache_key(
        model,
        "wind_10m",
        source.lead_time_hours,
        source.cycle_time.isoformat().replace("+00:00", "Z"),
        source.serving_generation,
        2,
        valid_time=source.valid_time.isoformat().replace("+00:00", "Z"),
        member_fingerprint=source.member_fingerprint,
    )
    return _vector_cache_get(key) is not None


def run_prewarm_pass(
    *,
    models: Sequence[str] | None = None,
    now: datetime | None = None,
    horizon_hours: int | None = None,
    max_computes: int | None = None,
    throttle_seconds: float | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> PrewarmPassResult:
    """Run one bounded prewarm scan pass. Never raises.

    Args:
        models: Model ids to warm (default: ``API_VECTOR_PREWARM_MODELS``).
        now: Override current time (tests); default server clock.
        horizon_hours: How far ahead of the serving window to warm.
        max_computes: Per-pass budget for cache-miss computes.
        throttle_seconds: Sleep between computes (bandwidth contention guard).
        sleeper: Sleep callable (tests inject a no-op recorder).
    """
    from api.core.config import settings

    model_ids = list(models if models is not None else settings.API_VECTOR_PREWARM_MODELS)
    horizon = int(horizon_hours if horizon_hours is not None else settings.API_VECTOR_PREWARM_HORIZON_HOURS)
    budget = int(max_computes if max_computes is not None else settings.API_VECTOR_PREWARM_MAX_COMPUTES_PER_PASS)
    throttle = float(
        throttle_seconds if throttle_seconds is not None else settings.API_VECTOR_PREWARM_COMPUTE_THROTTLE_SECONDS
    )
    from fastapi import HTTPException

    current_time = now
    if current_time is None:
        from api.core.time import get_current_time

        current_time = get_current_time()

    considered = already_cached = computed = skipped_unavailable = failed = 0
    exhausted_budget = False

    for model in model_ids:
        if exhausted_budget:
            break
        for valid_time in serving_window_valid_times(current_time, horizon):
            considered += 1
            try:
                with _open_session() as db:
                    source = _resolve_valid_time(db, model, valid_time, current_time)
            except HTTPException:
                # Nothing serves this valid time yet (window/partial wave) —
                # it becomes warmable on a later pass.
                skipped_unavailable += 1
                continue
            except Exception:  # noqa: BLE001
                logger.exception("prewarm resolve failed for %s @ %s", model, valid_time.isoformat())
                failed += 1
                continue

            try:
                if _cache_entry_cached(model, source):
                    already_cached += 1
                    continue
                if computed >= budget:
                    exhausted_budget = True
                    break
                with _open_session() as db:
                    _render_valid_time(db, model, valid_time, current_time)
                computed += 1
                if throttle > 0:
                    sleeper(throttle)
            except HTTPException:
                skipped_unavailable += 1
            except Exception:  # noqa: BLE001
                logger.exception("prewarm compute failed for %s @ %s", model, valid_time.isoformat())
                failed += 1

    return PrewarmPassResult(
        considered=considered,
        already_cached=already_cached,
        computed=computed,
        skipped_unavailable=skipped_unavailable,
        failed=failed,
    )


async def vector_prewarm_loop() -> None:
    """Lifespan-owned scan loop: sleep first, then one pass per interval."""
    from api.core.config import settings

    interval = float(settings.API_VECTOR_PREWARM_INTERVAL_SECONDS)
    # Sleep before the first pass so process boot (DB pools, store caches)
    # settles and short-lived test apps never spawn prewarm work.
    await asyncio.sleep(interval)
    while True:
        try:
            result = await asyncio.to_thread(run_prewarm_pass)
            if result.computed or result.failed:
                logger.info("vector-field prewarm pass: %s", asdict(result))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("vector-field prewarm pass crashed")
        await asyncio.sleep(interval)
