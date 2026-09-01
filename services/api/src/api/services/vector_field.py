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

from api.services.point_forecast import resolve_latest_run_serving_generation
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
) -> tuple[object, ...]:
    """Build the vector field cache key with full forecast and generation identity."""
    return (
        model,
        variable,
        lead_time_hours,
        initial_time,
        serving_generation,
        stride,
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
) -> bytes:
    """Gate-time selector: extract and encode U/V components under the SHARED lock.

    For GFS: encodes canonical (u, v) for the requested lead.
    For GEFS: computes consensus mean vector (mean(u_i), mean(v_i)) across members.
    """
    if "wind_u_10m" not in dataset.data_vars or "wind_v_10m" not in dataset.data_vars:
        raise ValueError("Variables 'wind_u_10m' and 'wind_v_10m' must be in the dataset.")

    field_u = dataset["wind_u_10m"]
    field_v = dataset["wind_v_10m"]

    if "lead_time_hours" in field_u.dims:
        field_u = field_u.sel(lead_time_hours=lead)
    if "lead_time_hours" in field_v.dims:
        field_v = field_v.sel(lead_time_hours=lead)

    lat_raw = np.asarray(field_u.latitude.values, dtype=float)
    lon_raw = np.asarray(field_u.longitude.values, dtype=float)

    lat_stride = lat_raw[::stride]
    lon_stride = lon_raw[::stride]

    if "member" in field_u.dims:
        # GEFS consensus vector: mean_u = mean(u_i), mean_v = mean(v_i)
        u_members = np.asarray(field_u.values[:, ::stride, ::stride], dtype=float)
        v_members = np.asarray(field_v.values[:, ::stride, ::stride], dtype=float)
        with np.errstate(all="ignore"):
            u_val = np.nanmean(u_members, axis=0)
            v_val = np.nanmean(v_members, axis=0)
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
    lead_time_hours: int,
    initial_time: str | None = None,
    stride: int = 2,
) -> bytes:
    """Render the quantized Int16 binary vector field for the given forecast selection.

    Args:
        db: Database session.
        model: Model identifier (e.g. 'gfs', 'gefs').
        lead_time_hours: Forecast offset hours from cycle time.
        initial_time: Optional ISO 8601 UTC cycle time.
        stride: Grid subsampling stride (default 2 -> 0.50° from 0.25° native).

    Returns:
        Encoded binary bytes.

    Raises:
        HTTPException: 404 if model/product/lead is not found or unreadable.
        ValueError: If dataset is malformed.
    """
    check_available(
        db,
        model=model,
        variable="wind_10m",
        level="surface",
        lead_time_hours=lead_time_hours,
        initial_time=initial_time,
    )

    serving_generation = resolve_latest_run_serving_generation(
        db, model, initial_time
    )
    cache_key = _vector_cache_key(
        model, "wind_10m", lead_time_hours, initial_time, serving_generation, stride
    )
    cached = _vector_cache_get(cache_key)
    if cached is not None:
        return cached

    from api.core import reader_gate
    from api.core.database import SessionLocal

    session = db
    excluded: set[str] = set()
    while True:
        try:
            store_path = _resolve_run_store_path(
                session,
                model=model,
                variable="wind_10m",
                level="surface",
                lead_time_hours=lead_time_hours,
                initial_time=initial_time,
                excluded=excluded,
            )
        except BaseException:
            session.close()
            raise

        session.close()
        try:
            payload = reader_gate.gated_read_dataset_with_selector(
                store_path,
                selector=lambda dataset: _select_and_encode_vector_field(
                    dataset,
                    lead=lead_time_hours,
                    stride=stride,
                ),
            )
        except Exception:  # noqa: BLE001
            excluded.add(store_path)
            session = SessionLocal()
            continue
        break

    _vector_cache_set(cache_key, payload)
    return payload
