"""Background warm-up of low-zoom forecast map tiles (P3 cold-load optimization).

Pre-renders Zoom 0..2 raster map tiles (1 + 4 + 16 = 21 tiles) for the primary
default forecast model and variable (e.g. GFS 2m Temperature at lead 0).
Prewarmed tiles reside in memory (and optionally in Nginx tmpfs RAM cache),
eliminating cold compute latency when users first open the weather map.

Design:
* A lifespan-owned asyncio task runs a prewarm pass every
  ``API_TILE_PREWARM_INTERVAL_SECONDS``.
* Multi-worker deduplication: each pass attempts an atomic Redis claim
  (``SET prewarm:tile:{model}:{variable}:{cycle} 1 NX EX 600``). Only the
  winning worker executes the prewarm; others skip immediately.
* Smooth throttling: small sleep (e.g. 20ms) between tiles to avoid CPU spikes.
* Dual-track path: defaults to safe in-process ``render_tile_png``; optionally
  fetches via ``API_TILE_PREWARM_GATEWAY_URL`` when configured.
* Error containment: failures are logged per tile and never fatal.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from domain.temporal import serving_start_valid_time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TilePrewarmPassResult:
    """Outcome counters of one tile prewarm scan pass."""

    #: Total tiles examined across all configured models and variables.
    considered: int
    #: Tiles already present in cache (free skips).
    already_cached: int
    #: Tiles computed and written to cache this pass.
    computed: int
    #: Tiles skipped because the run/shard is unavailable (404).
    skipped_unavailable: int
    #: Tiles that raised unexpectedly during render.
    failed: int


def generate_low_zoom_tile_coords(max_zoom: int = 2) -> list[tuple[int, int, int]]:
    """Generate (zoom, x, y) tuples from zoom 0 up to max_zoom inclusive.

    For max_zoom=2:
      z=0: 1 tile  (0, 0, 0)
      z=1: 4 tiles (1, x, y) for x, y in [0, 1]
      z=2: 16 tiles (2, x, y) for x, y in [0..3]
      Total = 21 tiles.
    """
    coords: list[tuple[int, int, int]] = []
    for z in range(max_zoom + 1):
        dim = 2**z
        for x in range(dim):
            for y in range(dim):
                coords.append((z, x, y))
    return coords


def _try_claim_tile_prewarm(claim_key: str, ttl_seconds: int = 600) -> bool:
    """Try to claim a prewarm job via Redis atomic SET NX EX.

    Returns True if successfully claimed (or if Redis is unavailable).
    Returns False if already claimed by another worker.
    """
    from api.core.config import settings

    try:
        import redis as redis_lib

        client = redis_lib.from_url(  # type: ignore[no-untyped-call]
            settings.REDIS_URL,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        try:
            return bool(client.set(claim_key, "1", nx=True, ex=ttl_seconds))
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - redis optional for local/offline
        logger.debug("Redis prewarm claim check bypassed: %s", exc)
        return True


def _render_tile_via_gateway(
    gateway_url: str,
    *,
    model: str,
    variable: str,
    zoom: int,
    x: int,
    y: int,
    valid_time: str,
    initial_time: str,
) -> None:
    """Fetch a tile via HTTP through the gateway to populate Nginx tmpfs cache."""
    import urllib.request

    url = (
        f"{gateway_url.rstrip('/')}/v1/maps/{model}/{variable}/surface/"
        f"{zoom}/{x}/{y}.png?valid_time={valid_time}&initial_time={initial_time}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "WeatherPlatform-Prewarm/1.0"})
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        resp.read()


def run_tile_prewarm_pass(
    *,
    now: datetime | None = None,
    models: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
    max_zoom: int | None = None,
    throttle_seconds: float | None = None,
    gateway_url: str | None = None,
) -> TilePrewarmPassResult:
    """Execute one pass of low-zoom tile prewarming across configured targets."""
    from fastapi import HTTPException
    from api.core.config import settings
    from api.core.database import SessionLocal
    from api.services.resolver import resolve_valid_time_source
    from api.services.tiles import (
        _tile_cache_get,
        _tile_cache_key,
        render_tile_png,
    )

    if not settings.API_TILE_PREWARM_ENABLED:
        return TilePrewarmPassResult(0, 0, 0, 0, 0)

    current_time = now
    if current_time is None:
        from api.core.time import get_current_time

        current_time = get_current_time()

    if models is None:
        cfg_models = settings.API_TILE_PREWARM_MODELS
        models = [cfg_models] if isinstance(cfg_models, str) else list(cfg_models)

    if variables is None:
        cfg_vars = settings.API_TILE_PREWARM_VARIABLES
        variables = [cfg_vars] if isinstance(cfg_vars, str) else list(cfg_vars)

    zoom_limit = max_zoom if max_zoom is not None else int(settings.API_TILE_PREWARM_MAX_ZOOM)
    throttle = (
        throttle_seconds
        if throttle_seconds is not None
        else float(settings.API_TILE_PREWARM_THROTTLE_SECONDS)
    )
    gw_url = gateway_url if gateway_url is not None else settings.API_TILE_PREWARM_GATEWAY_URL

    coords = generate_low_zoom_tile_coords(zoom_limit)
    considered = already_cached = computed = skipped_unavailable = failed = 0

    valid_time_dt = serving_start_valid_time(current_time)

    for model in models:
        for variable in variables:
            # 1. Resolve serving source metadata for this model & variable
            try:
                with SessionLocal() as db:
                    source = resolve_valid_time_source(
                        db, model, valid_time_dt, variable=variable, now=current_time
                    )
            except HTTPException:
                skipped_unavailable += len(coords)
                continue
            except Exception:  # noqa: BLE001
                logger.exception("tile prewarm resolve failed for %s/%s", model, variable)
                failed += len(coords)
                continue

            cycle_iso = source.cycle_time.isoformat().replace("+00:00", "Z")
            valid_iso = source.valid_time.isoformat().replace("+00:00", "Z")

            # 2. Redis atomic claim for multi-worker deduplication
            claim_key = f"prewarm:tile:{model}:{variable}:{cycle_iso}"
            if not _try_claim_tile_prewarm(claim_key):
                # Another worker already claimed or finished this cycle
                already_cached += len(coords)
                continue

            # 3. Prewarm the 21 tiles
            for z, x, y in coords:
                considered += 1
                cache_key = _tile_cache_key(
                    model,
                    variable,
                    "surface",
                    z,
                    x,
                    y,
                    source.lead_time_hours,
                    cycle_iso,
                    source.serving_generation,
                    valid_iso,
                )

                if _tile_cache_get(cache_key) is not None:
                    already_cached += 1
                    continue

                try:
                    if gw_url:
                        _render_tile_via_gateway(
                            gw_url,
                            model=model,
                            variable=variable,
                            zoom=z,
                            x=x,
                            y=y,
                            valid_time=valid_iso,
                            initial_time=cycle_iso,
                        )
                    else:
                        with SessionLocal() as render_db:
                            render_tile_png(
                                render_db,
                                model=model,
                                variable=variable,
                                level="surface",
                                zoom=z,
                                x=x,
                                y=y,
                                valid_time=valid_iso,
                                initial_time=cycle_iso,
                                now=current_time,
                            )
                    computed += 1
                    if throttle > 0:
                        time.sleep(throttle)
                except HTTPException:
                    skipped_unavailable += 1
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "tile prewarm render failed for %s/%s @ z=%d x=%d y=%d",
                        model,
                        variable,
                        z,
                        x,
                        y,
                    )
                    failed += 1

    return TilePrewarmPassResult(
        considered=considered,
        already_cached=already_cached,
        computed=computed,
        skipped_unavailable=skipped_unavailable,
        failed=failed,
    )


async def tile_prewarm_loop() -> None:
    """Lifespan-owned scan loop: sleep first, then one pass per interval."""
    from api.core.config import settings

    interval = float(settings.API_TILE_PREWARM_INTERVAL_SECONDS)
    # Sleep before the first pass so process boot settles
    await asyncio.sleep(min(5.0, interval))
    while True:
        try:
            result = await asyncio.to_thread(run_tile_prewarm_pass)
            if result.computed or result.failed:
                logger.info(
                    "tile prewarm pass: considered=%d computed=%d cached=%d skipped=%d failed=%d",
                    result.considered,
                    result.computed,
                    result.already_cached,
                    result.skipped_unavailable,
                    result.failed,
                )
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001
            logger.exception("unexpected error in tile prewarm loop")

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
