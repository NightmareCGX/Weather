"""Vector field extraction and caching for animated wind map layers.

Extracts canonical 10 m zonal (u) and meridional (v) wind velocity components
from forecast Zarr stores (GFS deterministic or GEFS ensemble), derives the
appropriate consensus vector field, downsamples to the target visualization
grid (default 0.50°), and encodes to quantized Int16 binary format.

For GEFS:
  Flow represents the ensemble consensus mean vector (mean(u_i), mean(v_i))
  across all members, while the background scalar raster represents expected
  member wind speed magnitude mean(hypot(u_i, v_i)).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from sqlalchemy.orm import Session

from api.services.point_forecast import (
    resolve_latest_run_store_path_and_retirement,
    resolve_serving_generation_for_store,
)
from api.services.tiles import (
    _resolve_run_store_path,
    check_available,
)
from domain.models.wind import encode_vector_field_int16

if TYPE_CHECKING:
    pass

#: Server-side vector field in-memory LRU cache.
#: Bounded so the API process memory remains strictly controlled.
_VECTOR_CACHE_MAX_ENTRIES = 128
_VECTOR_CACHE_TTL_SECONDS = 300
_vector_cache: dict[tuple[object, ...], tuple[float, bytes]] = {}


def _vector_cache_key(
    model: str,
    variable: str,
    lead_time_hours: int,
    initial_time: str | None,
    serving_generation: str | None,
    stride: int,
    valid_time: str | None = None,
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
        "v1_i16",
    )


def _vector_cache_get(key: tuple[object, ...]) -> bytes | None:
    """Return a live cached vector field payload, evicting stale entries."""
    entry = _vector_cache.get(key)
    if entry is None:
        return None
    created, payload = entry
    if time.monotonic() - created > _VECTOR_CACHE_TTL_SECONDS:
        _vector_cache.pop(key, None)
        return None
    return payload


def _vector_cache_set(key: tuple[object, ...], payload: bytes) -> None:
    """Store a vector field payload, evicting the oldest entry when full."""
    _vector_cache[key] = (time.monotonic(), payload)
    if len(_vector_cache) > _VECTOR_CACHE_MAX_ENTRIES:
        try:
            oldest = next(iter(_vector_cache))
            _vector_cache.pop(oldest, None)
        except StopIteration:
            pass


def _select_and_encode_vector_field(
    dataset: xr.Dataset,
    *,
    lead: int,
    stride: int = 2,
    store_path: str | None = None,
) -> bytes:
    """Gate-time selector: extract and encode U/V components under the SHARED lock.

    For GFS: encodes canonical (u, v) for the requested lead.
    For GEFS: computes consensus mean vector (mean(u_i), mean(v_i)) across members.
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
    if format_version == "sharded_v1" and store_path is not None:
        reader = get_sharded_reader(store_path)
        generation = manifest_generation(store_path)
        is_ensemble = "member" in dataset.coords or "member" in field_u.dims
        if is_ensemble:
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
        source = resolve_valid_time_source(db, model, valid_time, variable="wind_10m")
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

        store_path, latest_retired_iso = resolve_latest_run_store_path_and_retirement(
            db, model, resolved_initial
        )
        # Release DB connection immediately before S3 manifest read and cache check.
        db.close()

        serving_generation = resolve_serving_generation_for_store(
            store_path, latest_retired_iso
        )

    cache_key = _vector_cache_key(
        model,
        "wind_10m",
        resolved_lead,
        resolved_initial,
        serving_generation,
        stride,
        valid_time=resolved_valid_iso,
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
    _vector_cache_set(cache_key, payload)
    return payload
