"""Vector field extraction and caching for animated wind map layers.

Extracts canonical 10 m zonal (u) and meridional (v) wind velocity components
from forecast Zarr stores (GFS deterministic or GEFS ensemble), derives the
appropriate consensus vector field, downsamples to the target visualization
grid (default 0.50°), and encodes to quantized Int16 binary format.

For GEFS:
  Flow represents the ensemble consensus mean vector (mean(u_i), mean(v_i))
  across all members, while the background scalar raster represents expected
  member wind speed magnitude mean(hypot(u_i, v_i)). On sharded_v1 stores the
  official precomputed geavg mean shards (``shard.mean_L####``) are read
  directly — 2 full-globe shard reads, same cost as GFS — instead of reading
  30 member shards per component and averaging; the member-wise mean is only
  a fallback for stores without official mean shards.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import redis as redis_lib
import xarray as xr
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.services.point_forecast import (
    resolve_latest_run_store_path,
    resolve_serving_generation_for_store,
)
from api.services.tiles import (
    _resolve_run_store_path,
    check_available,
)
from domain.models.wind import encode_vector_field_int16
from domain.storage_dtype import is_sharded_payload_format

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)

#: L1 process-local cache: a small read-through front for the shared Redis
#: cache so warm entries avoid a network round trip and serving degrades to
#: today's behavior when Redis is unavailable. Bounded so API process memory
#: stays controlled (~16 x ~1 MB payloads).
_VECTOR_CACHE_MAX_ENTRIES = 16
_VECTOR_CACHE_TTL_SECONDS = 1800
_vector_cache: dict[tuple[object, ...], tuple[float, bytes]] = {}

#: L2 shared Redis cache: entries written by any worker are visible to all,
#: so the background prewarm (api/services/vector_prewarm.py) computes each
#: valid time once fleet-wide instead of once per worker. Keys are namespaced
#: by payload format; values are the raw quantized bytes with a TTL matching
#: the L1 freshness window.
_VECTOR_REDIS_KEY_PREFIX = "vectorfield:v1_i16:"

#: Best-effort Redis client (lazy singleton). A Redis outage must never fail
#: serving: reads miss and recompute, writes are dropped, and the state
#: transition is logged once rather than per request.
_redis_client: Redis | None = None
_redis_client_lock = threading.Lock()
_redis_degraded = False


def _get_redis_client() -> Redis | None:
    """Return the shared Redis client, or None when Redis caching is disabled."""
    global _redis_client
    from api.core.config import settings

    if not settings.API_VECTOR_CACHE_REDIS_ENABLED:
        return None
    if _redis_client is None:
        with _redis_client_lock:
            if _redis_client is None:
                # ``redis_lib.from_url`` is untyped in the redis stubs (see
                # api/services/cache.py for the same boundary treatment).
                _redis_client = redis_lib.from_url(  # type: ignore[no-untyped-call]
                    settings.REDIS_URL,
                    decode_responses=False,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
    return _redis_client


def _note_redis_error(operation: str) -> None:
    """Log the degraded-state transition once per Redis outage period."""
    global _redis_degraded
    if not _redis_degraded:
        logger.warning(
            "vector-field Redis cache degraded on %s; entries recompute per process",
            operation,
        )
        _redis_degraded = True


def _note_redis_success() -> None:
    """Log recovery when a Redis operation succeeds after a degraded period."""
    global _redis_degraded
    if _redis_degraded:
        logger.info("vector-field Redis cache recovered")
        _redis_degraded = False


def _vector_redis_key(key: tuple[object, ...]) -> str:
    """Serialize a cache-key tuple into a stable namespaced Redis string key."""
    canonical = json.dumps(key, default=str, separators=(",", ":"))
    return _VECTOR_REDIS_KEY_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _vector_cache_key(
    model: str,
    variable: str,
    lead_time_hours: int,
    initial_time: str | None,
    serving_generation: str | None,
    stride: int,
    valid_time: str | None = None,
    member_fingerprint: str | None = None,
) -> tuple[object, ...]:
    """Build the vector field cache key with full forecast and generation identity."""
    return (
        model,
        variable,
        lead_time_hours,
        initial_time,
        serving_generation,
        stride,
        valid_time,
        member_fingerprint,
        "v1_i16",
    )


def _evict_l1() -> None:
    """Evict the oldest L1 entry when the process-local budget is exceeded."""
    while len(_vector_cache) > _VECTOR_CACHE_MAX_ENTRIES:
        try:
            oldest = next(iter(_vector_cache))
            _vector_cache.pop(oldest, None)
        except StopIteration:
            break


def _vector_cache_get(key: tuple[object, ...]) -> bytes | None:
    """Return a live cached vector field payload (L1, then shared Redis L2).

    Stale L1 entries are evicted. Redis errors are treated as a miss so an
    outage degrades to per-process recomputation instead of failing requests.
    """
    entry = _vector_cache.get(key)
    if entry is not None:
        created, payload = entry
        if time.monotonic() - created <= _VECTOR_CACHE_TTL_SECONDS:
            return payload
        _vector_cache.pop(key, None)

    client = _get_redis_client()
    if client is None:
        return None
    try:
        # The redis stub types ``.get`` as ``Awaitable[Any] | Any`` even for a
        # synchronous client (see api/services/cache.py); with
        # ``decode_responses=False`` the value is ``bytes | None`` at runtime.
        raw = cast("bytes | None", client.get(_vector_redis_key(key)))
    except redis_lib.RedisError:
        _note_redis_error("read")
        return None
    if raw is None:
        return None
    _note_redis_success()
    payload = bytes(raw)
    _vector_cache[key] = (time.monotonic(), payload)
    _evict_l1()
    return payload


def _vector_cache_set(key: tuple[object, ...], payload: bytes) -> None:
    """Store a payload in the L1 cache and, best-effort, the shared Redis L2.

    The L2 write uses the shared TTL so entries expire uniformly across
    workers; a Redis outage drops the write without failing the request.
    """
    _vector_cache[key] = (time.monotonic(), payload)
    _evict_l1()

    client = _get_redis_client()
    if client is None:
        return
    try:
        client.setex(_vector_redis_key(key), _VECTOR_CACHE_TTL_SECONDS, payload)
        _note_redis_success()
    except redis_lib.RedisError:
        _note_redis_error("write")


def _select_and_encode_vector_field(
    dataset: xr.Dataset,
    *,
    lead: int,
    stride: int = 2,
    store_path: str | None = None,
) -> bytes:
    """Gate-time selector: extract and encode U/V components under the SHARED lock.

    For GFS: encodes canonical (u, v) for the requested lead.
    For GEFS: reads the official precomputed mean shards (geavg) when present,
    falling back to the member-wise consensus mean (mean(u_i), mean(v_i)).
    """
    from api.core.manifest_reader import manifest_generation, manifest_storage_format
    from api.core.zarr import get_sharded_reader

    if "wind_u_10m" not in dataset.data_vars or "wind_v_10m" not in dataset.data_vars:
        raise ValueError("Variables 'wind_u_10m' and 'wind_v_10m' must be in the dataset.")

    field_u = dataset["wind_u_10m"]
    field_v = dataset["wind_v_10m"]

    lat_raw = np.asarray(dataset.coords["latitude"].values, dtype=float)
    lon_raw = np.asarray(dataset.coords["longitude"].values, dtype=float)

    lat_stride = lat_raw[::stride]
    lon_stride = lon_raw[::stride]

    format_version = manifest_storage_format(store_path) if store_path else "v2_unsharded"
    if is_sharded_payload_format(format_version) and store_path is not None:
        reader = get_sharded_reader(store_path)
        generation = manifest_generation(store_path)
        is_ensemble = "member" in dataset.coords or "member" in field_u.dims
        if is_ensemble:
            if reader.has_mean_shard(
                "wind_u_10m", lead, generation=generation
            ) and reader.has_mean_shard("wind_v_10m", lead, generation=generation):
                # Official precomputed geavg mean shards: the consensus flow
                # vector is served directly — 2 full-globe reads, same cost as
                # the GFS deterministic path. Matches the raster tile path.
                u_win = reader.read_window(
                    "wind_u_10m",
                    member=None,
                    lead_time_hours=lead,
                    lat_min=0,
                    lat_max=len(lat_raw) - 1,
                    lon_min=0,
                    lon_max=len(lon_raw) - 1,
                    generation=generation,
                    is_mean=True,
                )[::stride, ::stride]
                v_win = reader.read_window(
                    "wind_v_10m",
                    member=None,
                    lead_time_hours=lead,
                    lat_min=0,
                    lat_max=len(lat_raw) - 1,
                    lon_min=0,
                    lon_max=len(lon_raw) - 1,
                    generation=generation,
                    is_mean=True,
                )[::stride, ::stride]
                u_val = np.where(np.isfinite(u_win), u_win, 0.0)
                v_val = np.where(np.isfinite(v_win), v_win, 0.0)
            else:
                # Store without official mean shards: compute the member-wise
                # consensus mean from the per-member shards.
                members_to_read = (
                    [int(v) for v in np.atleast_1d(dataset.coords["member"].values).reshape(-1)]
                    if "member" in dataset.coords
                    else list(range(1, 31))
                )
                u_members = [
                    reader.read_window(
                        "wind_u_10m",
                        member=m,
                        lead_time_hours=lead,
                        lat_min=0,
                        lat_max=len(lat_raw) - 1,
                        lon_min=0,
                        lon_max=len(lon_raw) - 1,
                        generation=generation,
                    )[::stride, ::stride]
                    for m in members_to_read
                ]
                v_members = [
                    reader.read_window(
                        "wind_v_10m",
                        member=m,
                        lead_time_hours=lead,
                        lat_min=0,
                        lat_max=len(lat_raw) - 1,
                        lon_min=0,
                        lon_max=len(lon_raw) - 1,
                        generation=generation,
                    )[::stride, ::stride]
                    for m in members_to_read
                ]
                with np.errstate(all="ignore"):
                    u_val = np.nanmean(u_members, axis=0)
                    v_val = np.nanmean(v_members, axis=0)
                u_val = np.where(np.isfinite(u_val), u_val, 0.0)
                v_val = np.where(np.isfinite(v_val), v_val, 0.0)
        else:
            u_win = reader.read_window(
                "wind_u_10m",
                member=None,
                lead_time_hours=lead,
                lat_min=0,
                lat_max=len(lat_raw) - 1,
                lon_min=0,
                lon_max=len(lon_raw) - 1,
                generation=generation,
            )[::stride, ::stride]
            v_win = reader.read_window(
                "wind_v_10m",
                member=None,
                lead_time_hours=lead,
                lat_min=0,
                lat_max=len(lat_raw) - 1,
                lon_min=0,
                lon_max=len(lon_raw) - 1,
                generation=generation,
            )[::stride, ::stride]
            u_val = np.where(np.isfinite(u_win), u_win, 0.0)
            v_val = np.where(np.isfinite(v_win), v_win, 0.0)
    else:
        if "lead_time_hours" in field_u.dims:
            field_u = field_u.sel(lead_time_hours=lead)
        if "lead_time_hours" in field_v.dims:
            field_v = field_v.sel(lead_time_hours=lead)

        if "member" in field_u.dims:
            # GEFS consensus vector: mean_u = mean(u_i), mean_v = mean(v_i)
            u_mem_arr = np.asarray(field_u.values[:, ::stride, ::stride], dtype=float)
            v_mem_arr = np.asarray(field_v.values[:, ::stride, ::stride], dtype=float)
            with np.errstate(all="ignore"):
                u_val = np.nanmean(u_mem_arr, axis=0)
                v_val = np.nanmean(v_mem_arr, axis=0)
            u_val = np.where(np.isfinite(u_val), u_val, 0.0)
            v_val = np.where(np.isfinite(v_val), v_val, 0.0)
        else:
            # GFS deterministic flow
            u_val = np.asarray(field_u.values[::stride, ::stride], dtype=float)
            v_val = np.asarray(field_v.values[::stride, ::stride], dtype=float)
            u_val = np.where(np.isfinite(u_val), u_val, 0.0)
            v_val = np.where(np.isfinite(v_val), v_val, 0.0)

    lat_step = float((lat_raw[-1] - lat_raw[0]) / (len(lat_raw) - 1) * stride) if len(lat_raw) > 1 else 1.0
    lon_step = float((lon_raw[-1] - lon_raw[0]) / (len(lon_raw) - 1) * stride) if len(lon_raw) > 1 else 1.0

    return encode_vector_field_int16(
        u_val,
        v_val,
        lat_start=float(lat_stride[0]),
        lat_step=lat_step,
        lon_start=float(lon_stride[0]),
        lon_step=lon_step,
        scale=0.01,
    )


def render_vector_field_binary(
    db: Session,
    *,
    model: str,
    lead_time_hours: int | None = None,
    valid_time: str | None = None,
    initial_time: str | None = None,
    stride: int = 2,
    now: datetime | None = None,
) -> bytes:
    """Render the quantized Int16 binary vector field for the given forecast selection.

    Under Lifecycle V2, supports either ``valid_time`` or ``lead_time_hours`` (with optional ``initial_time``).
    """
    from fastapi import HTTPException
    from api.services.resolver import resolve_valid_time_source

    resolved_valid_iso: str | None = None
    resolved_initial: str | None = None
    resolved_lead: int = 0
    store_path: str | None = None
    serving_generation: str | None = None

    if valid_time is not None:
        if initial_time is not None:
            raise HTTPException(
                status_code=422,
                detail="Provide either valid_time or initial_time, not both.",
            )
        source = resolve_valid_time_source(db, model, valid_time, variable="wind_10m", now=now)
        store_path = source.store_path
        resolved_lead = source.lead_time_hours
        resolved_initial = source.cycle_time.isoformat().replace("+00:00", "Z")
        resolved_valid_iso = source.valid_time.isoformat().replace("+00:00", "Z")
        serving_generation = source.serving_generation
        db.close()
    else:
        if lead_time_hours is None:
            raise HTTPException(
                status_code=422,
                detail="Either valid_time or lead_time_hours is required.",
            )
        resolved_lead = lead_time_hours
        resolved_initial = initial_time

        if initial_time is not None:
            from api.services.lifecycle import require_cycle_visible

            require_cycle_visible(db, initial_time, model_id=model)

        check_available(
            db,
            model=model,
            variable="wind_10m",
            level="surface",
            lead_time_hours=resolved_lead,
            initial_time=resolved_initial,
        )

        store_path = resolve_latest_run_store_path(
            db, model, resolved_initial
        )
        # Release DB connection immediately before S3 manifest read and cache check.
        db.close()

        serving_generation = resolve_serving_generation_for_store(store_path)

    member_fingerprint = source.member_fingerprint if valid_time is not None else None
    cache_key = _vector_cache_key(
        model,
        "wind_10m",
        resolved_lead,
        resolved_initial,
        serving_generation,
        stride,
        valid_time=resolved_valid_iso,
        member_fingerprint=member_fingerprint,
    )
    cached = _vector_cache_get(cache_key)
    if cached is not None:
        return cached

    from api.core import reader_gate
    from api.core.database import SessionLocal

    session: Session | None = None
    excluded: set[str] = set()
    current_store_path = store_path or ""
    while True:
        if not current_store_path or current_store_path in excluded:
            session = SessionLocal()
            try:
                current_store_path = _resolve_run_store_path(
                    session,
                    model=model,
                    variable="wind_10m",
                    level="surface",
                    lead_time_hours=resolved_lead,
                    initial_time=resolved_initial,
                    excluded=excluded,
                )
            finally:
                session.close()

        try:
            payload = reader_gate.gated_read_dataset_with_selector(
                current_store_path,
                selector=lambda dataset: _select_and_encode_vector_field(
                    dataset,
                    lead=resolved_lead,
                    stride=stride,
                    store_path=current_store_path,
                ),
            )
        except Exception:  # noqa: BLE001
            excluded.add(current_store_path)
            continue
        break

    # Post-read validation: verify wind components did not transition to deleting/deleted during read
    try:
        from api.models.entities import ReclamationQueue
        from domain.reclamation import make_shard_relative_key

        t_kind = "mean" if model == "gefs" else "det"
        u_key = make_shard_relative_key("wind_u_10m", t_kind, resolved_lead)
        v_key = make_shard_relative_key("wind_v_10m", t_kind, resolved_lead)
        with SessionLocal() as check_session:
            fenced_shards = check_session.execute(
                select(ReclamationQueue.id).where(
                    ReclamationQueue.store_path == current_store_path,
                    ReclamationQueue.physical_key.in_([u_key, v_key]),
                    ReclamationQueue.status.in_(("deleting", "deleted", "failed")),
                )
            ).scalars().all()
            if fenced_shards:
                raise HTTPException(status_code=404, detail="Wind component shards became unavailable during read.")
    except HTTPException:
        raise
    except Exception:
        pass

    _vector_cache_set(cache_key, payload)
    return payload
