"""Production ingestion orchestration: parse GRIB2 -> write Zarr -> record catalog.

This is the thin runtime boundary that wires the ingestion library modules
together into a real production call path. It deliberately contains **no**
network/scheduling logic (the CLI owns download access) and **no** persistence
details (``catalog.py`` owns PostgreSQL), keeping every layer single-purpose:

* ``parse_grib2``  -> decode + normalize a GRIB2 file (``providers/noaa/parser.py``)
* ``write_dataset`` -> persist the dataset to Zarr (``core/zarr_writer.py``)
* ``record_ingested_dataset`` -> upsert the PostgreSQL catalog as a ``ready`` run
  (``core/catalog.py``)

One ``model_runs`` row represents a full forecast cycle (``UNIQUE(model_version_id,
cycle_time)`` per DATABASE.md), and its Zarr store accumulates every lead. NOMADS
serves one GRIB2 file per lead, so :func:`ingest_grib_file` merges each new lead
into the cycle store along ``lead_time_hours`` (re-ingesting a lead replaces it)
before recording the run.

The entry point :func:`ingest_grib_file` is the reusable unit a scheduler or
worker (e.g. a future Celery task) can call; the console entrypoint
``weather-ingest`` (``ingestion.cli``) adds the network download step on top.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from domain.models.cloud import (
    CLOUD_COVER_RECONSTRUCTION_TOLERANCE_PERCENT,
    reconstruct_cloud_cover_3h,
)
from ingestion.core.base import (
    CycleStoreMismatchError,
    DEACCUMULATION_CLAMP_BOUND_MM,
    DeaccumulationError,
    IngestionError,
    LeadTimeMismatchError,
    LiveStoreOverwriteError,
    MissingPredecessorLeadError,
    StoreSchemaMismatchError,
)
from ingestion.core.catalog import (
    CommittedState,
    ModelRunRecord,
    RunCatalogSpec,
    VariableSpec,
    is_ready_run_store,
    record_ingested_dataset,
)
from ingestion.core.zarr_writer import (
    commit_region,
    prepare_run_store,
    read_dataset,
    read_slice,
    store_exists,
)
from ingestion.providers.noaa.parser import parse_grib2

logger = logging.getLogger(__name__)


class UnitNormalizationError(IngestionError):
    """Raised when a source GRIB unit cannot be safely canonicalized.

    A mapped variable whose source ``units`` attribute is neither already equal
    to the target canonical unit nor a known source unit for that target is
    rejected rather than silently stored under a possibly-wrong label.
    """


def _session_local() -> "Session":
    """Return a new catalog Session for the live-store check (library path).

    The unlocked library path uses this only to check whether the target store
    belongs to a live run before refusing to bypass the concurrency protocol.
    Tests monkeypatch ``_live_store_session_factory`` to route this to their
    SQLite catalog engine.
    """
    from sqlalchemy.orm import Session

    return Session(bind=_live_store_session_factory())


def _default_live_store_engine() -> "Engine":
    """Return the configured ingestion catalog engine (live-store guard)."""
    from ingestion.core.db import engine

    return engine


#: Injectable engine factory for the unlocked-library-path live-store check.
#: Production uses the configured ingestion engine; tests replace this with an
#: in-memory SQLite engine so the library path can be exercised without PG.
_live_store_session_factory = _default_live_store_engine


def read_committed_state(
    store_path: str,
    *,
    is_ensemble: bool,
) -> CommittedState:
    """Return the actual committed lead/member state of a cycle store.

    A cycle store's ``lead_time_hours`` coordinate is **pre-allocated** to the
    full expected axis (NaN-filled), so mere coordinate membership does NOT mean
    a lead is committed. This function inspects the store's **data regions**:
    for each data variable with a ``lead_time_hours`` dimension (and, for
    ensemble, a ``member`` dimension), it reduces over the spatial axes with
    ``notnull().any()`` and collects the coordinates whose region holds any
    non-NaN forecast value.

    For deterministic runs the result is the committed lead set. For ensemble
    runs it is the committed ``(member, lead)`` pair set; the committed member
    set is derived from those pairs (a member is committed iff at least one of
    its pairs has data).

    The reduction only loads the small lead/member axes (the chunk grid is
    ``(lead, lat, lon)`` / ``(member, lead, lat, lon)`` with one chunk per
    region), not the full forecast grid.

    Args:
        store_path: The cycle store path/URL.
        is_ensemble: Whether the store has a ``member`` axis (ensemble).

    Returns:
        The committed state derived from real data regions.

    Raises:
        ValueError: If the store cannot be read.
    """
    dataset = read_dataset(store_path)
    if "lead_time_hours" not in dataset.coords:
        raise ValueError(
            f"Cannot derive committed state from {store_path!r}: no "
            "'lead_time_hours' coordinate."
        )
    lead_values = dataset.coords["lead_time_hours"].values
    # The store's real variable set is the source of truth for catalog ↔ store
    # variable honesty: the catalog must only advertise products for variables
    # the store actually carries (e.g. a GEFS store never has
    # ``precipitation_rate`` because GEFS pgrb2s has no instant prate field).
    # Recording the store's data variables here lets reconciliation prune stale
    # products for store-absent variables and never restore them.
    store_variables = {str(name) for name in dataset.data_vars}

    if not is_ensemble:
        committed_leads: set[int] = set()
        for name in dataset.data_vars:
            field = dataset[name]
            if "lead_time_hours" not in field.dims:
                continue
            has = field.notnull().any(dim=("latitude", "longitude"))
            for idx, value in enumerate(has.values):
                if bool(value):
                    committed_leads.add(int(lead_values[idx]))
        return CommittedState.deterministic(committed_leads, variables=store_variables)

    if "member" not in dataset.coords:
        raise ValueError(
            f"Cannot derive committed state from {store_path!r}: expected an "
            "ensemble store but no 'member' coordinate is present."
        )
    member_values = dataset.coords["member"].values
    committed_pairs: set[tuple[int, int]] = set()
    for name in dataset.data_vars:
        field = dataset[name]
        if "member" not in field.dims or "lead_time_hours" not in field.dims:
            continue
        has = field.notnull().any(dim=("latitude", "longitude"))
        for mi, member_val in enumerate(member_values):
            for li, lead_val in enumerate(lead_values):
                if bool(has.values[mi, li]):
                    committed_pairs.add((int(member_val), int(lead_val)))
    committed_members = {member for member, _ in committed_pairs}
    return CommittedState.ensemble(
        committed_pairs, committed_members, variables=store_variables
    )


def guard_full_overwrite(db: Session, store_path: str) -> None:
    """Reject a full overwrite of a store that belongs to a live ``model_runs``.

    The low-level Zarr full-write helpers (``write_dataset``,
    ``write_dataset_atomic``, ``prepare_run_store`` with ``mode="w"``) rebuild a
    store's coordinate axis and would silently shrink/replace the contents of a
    run's store without catalog reconciliation — recreating the stale
    ``forecast_products`` debt. This guard runs at the orchestration boundary
    (NOT inside ``zarr_writer.py``, which stays DB-free): before any such full
    write, the caller checks whether the target belongs to a **ready** run and,
    if so, refuses. New/non-live store creation is unaffected.

    Live-store semantics: a full overwrite destroys content only when the
    target has committed content. A ``ready`` row is the only
    committed-and-quiescent state. A ``processing``/``partial`` row is a
    placeholder from an in-flight or aborted first ingestion whose store may be
    genuinely absent (a region-write failure before any data committed);
    cold-start initializing such a store is the legitimate recovery path and
    must NOT be refused — otherwise a fresh cycle whose first attempt failed
    would self-block on retry. A ``ready`` run whose store is missing is instead
    an external shrink/corruption condition and is refused.

    Args:
        db: Database session used to check ``model_runs`` ownership.
        store_path: The store path/URL that would be fully overwritten.

    Raises:
        LiveStoreOverwriteError: If the target is referenced by a ready
            ``model_runs`` row.
    """
    if is_ready_run_store(db, store_path):
        raise LiveStoreOverwriteError(
            f"Refusing to fully overwrite {store_path!r}: the store belongs to "
            "a live model_runs row. A full overwrite would silently shrink or "
            "replace the run's contents without catalog reconciliation. "
            "Re-ingest individual leads/members instead, or route through a "
            "coordinated replacement path."
        )


def _unit_token(unit: str) -> str:
    """Return a whitespace/power-symbol normalized token for a unit string.

    GRIB ``units`` strings vary in spelling across producers and eccodes
    versions (e.g. ``"kg m-2 s-1"`` vs ``"kg m**-2 s**-1"`` vs
    ``"kg m^-2 s^-1"``). Stripping spaces, ``**``, and ``^`` maps equivalent
    spellings to one token for table lookup.

    Args:
        unit: A unit string (e.g. ``"kg m-2 s-1"``).

    Returns:
        A normalized token (e.g. ``"kgm-2s-1"``).
    """
    return unit.replace(" ", "").replace("**", "").replace("^", "").lower()


#: Source GRIB unit → canonical (target) unit value transforms.
#: Keyed by the target canonical unit string (the ``VariableSpec.unit``), then
#: by the normalized source ``units`` token. Each transform maps a source-unit
#: array to the equivalent canonical-unit array. Only the currently implemented
#: platform mappings are present (ENGINEERING_CONTRACT: no speculative units).
_SOURCE_TO_CANONICAL: dict[
    str, dict[str, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]
] = {
    # GRIB 2-metre temperature ``t`` is Kelvin; canonical is Celsius.
    "°C": {"k": lambda array: array - 273.15},
    # GFS ``prate`` is a precipitation rate in kg m-2 s-1; canonical is mm/h.
    # For liquid water 1 kg m-2 == 1 mm water-equivalent depth, so the
    # rate conversion is a pure ×3600 (s-1 → h-1). This is a rate conversion,
    # not an accumulation conversion.
    "mm/h": {"kgm-2s-1": lambda array: array * 3600.0},
    # Wind gust ``gust`` is m/s; canonical is km/h (x3.6).
    "km/h": {
        "ms-1": lambda array: array * 3.6,
        "m/s": lambda array: array * 3.6,
    },
    # Relative humidity and cloud cover are %; canonical is %.
    "%": {"%": lambda array: array},
    # Visibility, snow depth, and cloud ceiling are meters; canonical is meters.
    "m": {
        "m": lambda array: array,
        "gpm": lambda array: array,
    },
    # Wind vector components are m/s; canonical is m/s.
    "m/s": {
        "ms-1": lambda array: array,
        "m/s": lambda array: array,
    },
    # Total precipitation ``tp`` is kg m-2; canonical is mm (liquid water equivalent, x1.0).
    "mm": {
        "kgm-2": lambda array: array,
        "kg/m2": lambda array: array,
        "kg/m^2": lambda array: array,
        "mm": lambda array: array,
    },
    # Categorical precipitation flags (crain, csnow, cfrzr, cicep) are binary indicators (Code table 4.222).
    "flag": {
        "(codetable4.222)": lambda array: np.asarray(array, dtype=np.uint8),
        "codetable4.222": lambda array: np.asarray(array, dtype=np.uint8),
        "flag": lambda array: np.asarray(array, dtype=np.uint8),
        "0/1flag": lambda array: np.asarray(array, dtype=np.uint8),
        "numeric": lambda array: np.asarray(array, dtype=np.uint8),
    },
}


def deaccumulate_precipitation(
    current_accum: npt.NDArray[Any],
    predecessor_accum: npt.NDArray[Any],
    *,
    tolerance: float = DEACCUMULATION_CLAMP_BOUND_MM,
) -> npt.NDArray[np.float32]:
    """Derive 3-hour precipitation increment by subtracting predecessor accumulation.

    Computes ``increment = current_accum - predecessor_accum`` elementwise:
    * Non-negative increments (residual >= 0.0 mm) are preserved as-is.
    * Negative residuals within ``[-tolerance, 0.0)`` mm (exact bound at ``-0.50 mm``)
      caused by upstream GRIB packing scale differences between 3h and 6h files are
      clamped to ``0.0 mm``.
    * Negative residuals strictly below ``-tolerance`` (< -0.50 mm) violate physical
      bounds and are set to ``NaN`` elementwise without failing the task.
    * Existing NaNs in input arrays are preserved as ``NaN``.
    * Input arrays are never mutated.

    Args:
        current_accum: Current interval accumulation array (e.g. [t-6, t]).
        predecessor_accum: Predecessor interval accumulation array (e.g. [t-6, t-3]).
        tolerance: Clamping bound in mm for negative residuals (default: 0.50 mm).

    Returns:
        The normalized 3-hour precipitation increment array (dtype float32).

    Raises:
        DeaccumulationError: If shape mismatch between current and predecessor arrays.
    """
    curr = np.asarray(current_accum, dtype=np.float64)
    pred = np.asarray(predecessor_accum, dtype=np.float64)
    if curr.shape != pred.shape:
        raise DeaccumulationError(
            f"Cannot de-accumulate precipitation: shape mismatch between current "
            f"{curr.shape} and predecessor {pred.shape}."
        )
    diff = curr - pred
    result = np.full_like(diff, np.nan, dtype=np.float32)
    valid_mask = ~np.isnan(diff)

    # 1. Non-negative residuals preserved as-is
    ge_zero = valid_mask & (diff >= 0.0)
    result[ge_zero] = np.asarray(diff[ge_zero], dtype=np.float32)

    # 2. Negative residuals within [-tolerance, 0.0) clamped to 0.0 mm (exact bound at -0.50 mm)
    bound = float(tolerance)
    clamped_mask = valid_mask & (diff < 0.0) & (diff >= -bound)
    result[clamped_mask] = 0.0

    # 3. Negative residuals < -bound remain NaN in result
    invalidated_mask = valid_mask & (diff < -bound)

    # QC Observability logging
    clamped_count = int(np.count_nonzero(clamped_mask))
    invalidated_count = int(np.count_nonzero(invalidated_mask))
    if clamped_count > 0 or invalidated_count > 0:
        min_residual = float(np.min(diff[valid_mask]))
        logger.warning(
            "Precipitation de-accumulation negative residuals detected: "
            "clamped_count=%d (in [-%0.2f, 0.0) mm -> 0.0 mm), "
            "invalidated_count=%d (< -%0.2f mm -> NaN), min_residual=%.4f mm",
            clamped_count,
            bound,
            invalidated_count,
            bound,
            min_residual,
        )

    return result


def read_predecessor_precipitation(
    store_path: str,
    lead_time_hours: int,
    *,
    member: int | None = None,
) -> npt.NDArray[np.float32]:
    """Read a committed precipitation_amount_3h slice from a cycle's Zarr store.

    Used by the ingestion normalizer when deriving 3-hour increments for
    6-hour-reset leads (t=6, 12, 18, 24, ...).

    Args:
        store_path: Path/URL to the target Zarr store.
        lead_time_hours: The predecessor lead time (e.g. lead - 3).
        member: Ensemble member index (None for deterministic).

    Returns:
        2D numpy array of predecessor precipitation amounts (dtype float32).

    Raises:
        MissingPredecessorLeadError: If store does not exist, lead is missing,
            or slice is completely uncommitted (all NaN).
    """
    if not store_exists(store_path):
        raise MissingPredecessorLeadError(
            f"Cannot read predecessor precipitation: store {store_path!r} does not exist."
        )
    vals = read_slice(
        store_path,
        "precipitation_amount_3h",
        lead_time_hours=lead_time_hours,
        member=member,
    )
    if vals is None:
        raise MissingPredecessorLeadError(
            f"Cannot read predecessor precipitation: variable 'precipitation_amount_3h' "
            f"at lead {lead_time_hours} (member={member}) is missing from store {store_path!r}."
        )
    if np.all(np.isnan(vals)):
        raise MissingPredecessorLeadError(
            f"Predecessor lead {lead_time_hours} (member={member}) in {store_path!r} "
            "is uncommitted (contains only NaN values)."
        )
    return vals


def _normalize_precipitation_increments(
    dataset: xr.Dataset,
    variables: tuple[VariableSpec, ...],
    *,
    store_path: str | None = None,
    predecessor_array: npt.NDArray[Any] | None = None,
    member: int | None = None,
) -> xr.Dataset:
    """Normalize precipitation accumulation fields to canonical 3-hour increments.

    Canonical contract:
    * lead == 0: precipitation_amount_3h is NaN (no 3h interval precedes analysis).
    * lead % 6 == 3 (e.g. 3, 9, 15, ...): direct upstream APCP is [t-3, t].
    * lead % 6 == 0 and lead > 0 (e.g. 6, 12, 18, ...): upstream APCP is [t-6, t].
      Derived as amount_3h(t) = APCP[t-6, t] - APCP[t-6, t-3] using predecessor.

    Args:
        dataset: The decoded dataset (carrying raw shortNames or mapped codes).
        variables: Run's catalog variable specifications.
        store_path: Optional store path for predecessor lookup at 6h leads.
        predecessor_array: Optional explicit predecessor 2D array.
        member: Optional member identity for ensemble predecessor lookup.

    Returns:
        The dataset with normalized precipitation_amount_3h values.
    """
    has_precip_spec = any(
        v.code in ("precipitation_amount_3h", "crain", "csnow", "cfrzr", "cicep")
        for v in variables
    )
    if not has_precip_spec:
        return dataset

    if "lead_time_hours" not in dataset.coords:
        return dataset
    lead_val = int(np.asarray(dataset.coords["lead_time_hours"].values).reshape(-1)[0])

    precip_var_name = None
    if "precipitation_amount_3h" in dataset.data_vars:
        precip_var_name = "precipitation_amount_3h"
    elif "tp" in dataset.data_vars:
        precip_var_name = "tp"

    # Case 1: Analysis time (lead == 0)
    if lead_val == 0:
        if precip_var_name is not None:
            dataset[precip_var_name].values = np.full_like(
                dataset[precip_var_name].values, np.nan, dtype=np.float32
            )
            dataset[precip_var_name].attrs["units"] = "mm"
        else:
            if dataset.data_vars:
                ref_var = next(iter(dataset.data_vars.values()))
                nan_arr = np.full_like(ref_var.values, np.nan, dtype=np.float32)
                dataset["tp"] = xr.DataArray(
                    nan_arr,
                    dims=ref_var.dims,
                    coords=ref_var.coords,
                    attrs={"units": "mm", "long_name": "3-Hour Precipitation Amount"},
                )
        for cat_code in ("crain", "csnow", "cfrzr", "cicep"):
            if any(v.code == cat_code for v in variables):
                if cat_code in dataset.data_vars:
                    dataset[cat_code].values = np.zeros_like(
                        dataset[cat_code].values, dtype=np.uint8
                    )
                    dataset[cat_code].attrs["units"] = "flag"
                elif dataset.data_vars:
                    ref_var = next(iter(dataset.data_vars.values()))
                    dataset[cat_code] = xr.DataArray(
                        np.zeros_like(ref_var.values, dtype=np.uint8),
                        dims=ref_var.dims,
                        coords=ref_var.coords,
                        attrs={"units": "flag"},
                    )
        return dataset

    if precip_var_name is None:
        return dataset

    # Case 2: Direct 3-hour lead (lead % 6 == 3)
    if lead_val % 6 == 3:
        dataset[precip_var_name].values = np.asarray(
            dataset[precip_var_name].values, dtype=np.float32
        )
        dataset[precip_var_name].attrs["units"] = "mm"
        return dataset

    # Case 3: Differenced 6-hour lead (lead % 6 == 0)
    if lead_val % 6 == 0:
        pred_lead = lead_val - 3
        if predecessor_array is None:
            if store_path is None:
                raise MissingPredecessorLeadError(
                    f"Cannot de-accumulate lead {lead_val}: no store_path or predecessor_array provided."
                )
            predecessor_array = read_predecessor_precipitation(
                store_path, pred_lead, member=member
            )

        curr_vals = dataset[precip_var_name].values
        orig_shape = curr_vals.shape
        curr_2d = np.squeeze(curr_vals)
        pred_2d = np.squeeze(predecessor_array)

        diff_2d = deaccumulate_precipitation(curr_2d, pred_2d)
        dataset[precip_var_name].values = diff_2d.reshape(orig_shape)
        dataset[precip_var_name].attrs["units"] = "mm"
        return dataset

    return dataset


def read_predecessor_cloud_cover(
    store_path: str,
    lead_time_hours: int,
    *,
    member: int | None = None,
) -> npt.NDArray[np.float32]:
    """Read a committed cloud_cover_3h slice from a cycle's Zarr store.

    Used by the ingestion normalizer when reconstructing 3-hour interval averages for
    6-hour-reset leads (t=6, 12, 18, 24, ...).

    Args:
        store_path: Path/URL to the target Zarr store.
        lead_time_hours: The predecessor lead time (e.g. lead - 3).
        member: Ensemble member index (None for deterministic).

    Returns:
        2D numpy array of predecessor cloud cover percentages (dtype float32).

    Raises:
        MissingPredecessorLeadError: If store does not exist, lead is missing,
            or slice is completely uncommitted (all NaN).
    """
    if not store_exists(store_path):
        raise MissingPredecessorLeadError(
            f"Cannot read predecessor cloud cover: store {store_path!r} does not exist."
        )
    vals = read_slice(
        store_path,
        "cloud_cover_3h",
        lead_time_hours=lead_time_hours,
        member=member,
    )
    if vals is None:
        raise MissingPredecessorLeadError(
            f"Cannot read predecessor cloud cover: variable 'cloud_cover_3h' "
            f"at lead {lead_time_hours} (member={member}) is missing from store {store_path!r}."
        )
    if np.all(np.isnan(vals)):
        raise MissingPredecessorLeadError(
            f"Predecessor lead {lead_time_hours} (member={member}) in {store_path!r} "
            "is uncommitted (contains only NaN values)."
        )
    return vals


def _normalize_cloud_cover_intervals(
    dataset: xr.Dataset,
    variables: tuple[VariableSpec, ...],
    *,
    store_path: str | None = None,
    predecessor_array: npt.NDArray[Any] | None = None,
    member: int | None = None,
) -> xr.Dataset:
    """Normalize cloud cover interval-average fields to canonical preceding 3-hour averages.

    Canonical contract:
    * lead == 0: cloud_cover_3h is NaN (no preceding 3h interval exists).
    * lead % 6 == 3 (e.g. 3, 9, 15, ...): direct upstream TCDC is already [t-3, t].
    * lead % 6 == 0 and lead > 0 (e.g. 6, 12, 18, ...): upstream TCDC is [t-6, t].
      Derived as C_3h(t) = 2 * TCDC[t-6, t] - C_3h[t-6, t-3] using predecessor,
      guarded by CLOUD_COVER_RECONSTRUCTION_TOLERANCE_PERCENT (±5%).

    Args:
        dataset: The decoded dataset (carrying raw shortNames or mapped codes).
        variables: Run's catalog variable specifications.
        store_path: Optional store path for predecessor lookup at 6h leads.
        predecessor_array: Optional explicit predecessor 2D array.
        member: Optional member identity for ensemble predecessor lookup.

    Returns:
        The dataset with normalized cloud_cover_3h values.
    """
    has_cloud_spec = any(v.code == "cloud_cover_3h" for v in variables)
    if not has_cloud_spec:
        return dataset

    if "lead_time_hours" not in dataset.coords:
        return dataset
    lead_val = int(np.asarray(dataset.coords["lead_time_hours"].values).reshape(-1)[0])

    cloud_var_name = None
    if "cloud_cover_3h" in dataset.data_vars:
        cloud_var_name = "cloud_cover_3h"
    elif "tcc" in dataset.data_vars:
        cloud_var_name = "tcc"

    # Case 1: Analysis time (lead == 0)
    if lead_val == 0:
        if cloud_var_name is not None:
            dataset[cloud_var_name].values = np.full_like(
                dataset[cloud_var_name].values, np.nan, dtype=np.float32
            )
            dataset[cloud_var_name].attrs["units"] = "%"
        else:
            if dataset.data_vars:
                ref_var = next(iter(dataset.data_vars.values()))
                nan_arr = np.full_like(ref_var.values, np.nan, dtype=np.float32)
                dataset["tcc"] = xr.DataArray(
                    nan_arr,
                    dims=ref_var.dims,
                    coords=ref_var.coords,
                    attrs={"units": "%", "long_name": "3-Hour Total Cloud Cover"},
                )
        return dataset

    if cloud_var_name is None:
        return dataset

    # Case 2: Direct 3-hour lead (lead % 6 == 3)
    if lead_val % 6 == 3:
        raw_vals = np.asarray(dataset[cloud_var_name].values, dtype=np.float32)
        clipped = np.clip(raw_vals, 0.0, 100.0)
        dataset[cloud_var_name].values = clipped
        dataset[cloud_var_name].attrs["units"] = "%"
        return dataset

    # Case 3: Reconstructed 6-hour reset lead (lead % 6 == 0)
    if lead_val % 6 == 0:
        pred_lead = lead_val - 3
        if predecessor_array is None:
            if store_path is None:
                raise MissingPredecessorLeadError(
                    f"Cannot reconstruct cloud cover for lead {lead_val}: "
                    "no store_path or predecessor_array provided."
                )
            predecessor_array = read_predecessor_cloud_cover(
                store_path, pred_lead, member=member
            )

        curr_vals = dataset[cloud_var_name].values
        orig_shape = curr_vals.shape
        curr_2d = np.squeeze(curr_vals)
        pred_2d = np.squeeze(predecessor_array)

        reconstructed_2d = reconstruct_cloud_cover_3h(
            curr_2d, pred_2d, tolerance=CLOUD_COVER_RECONSTRUCTION_TOLERANCE_PERCENT
        )
        dataset[cloud_var_name].values = np.asarray(
            reconstructed_2d, dtype=np.float32
        ).reshape(orig_shape)
        dataset[cloud_var_name].attrs["units"] = "%"
        return dataset

    return dataset


def _apply_variable_mapping(
    dataset: xr.Dataset,
    variables: tuple[VariableSpec, ...],
) -> xr.Dataset:
    """Rename a parsed dataset's data variables to the platform vocabulary.

    The GRIB2 decoder emits raw ``shortName`` data variables (e.g. ``t`` for
    2-metre temperature). Each :class:`VariableSpec` in ``variables`` may carry
    a ``source_code`` naming the raw variable it maps from; matching variables
    are renamed to the platform ``code`` so the Zarr store is self-consistent
    with the ``forecast_variables``/``forecast_products`` catalog rows.

    Args:
        dataset: The parsed dataset (raw GRIB2 variable names).
        variables: The run's :class:`VariableSpec` catalog metadata.

    Returns:
        The dataset with raw variable names renamed to platform codes. Unknown
        raw variables are left untouched.
    """
    rename: dict[str, str] = {}
    for variable in variables:
        source = variable.source_code or variable.code
        if source in dataset.data_vars:
            rename[source] = variable.code
    if not rename:
        return dataset
    return dataset.rename(rename)


def _normalize_canonical_units(
    dataset: xr.Dataset,
    variables: tuple[VariableSpec, ...],
) -> xr.Dataset:
    """Convert each mapped variable's values to the platform canonical unit.

    The platform stores canonical units in Zarr (Model A, ``docs/API.md``
    section 2.6 / ``docs/MODELS.md`` section 3): GRIB ``K`` temperature becomes
    ``°C`` and GFS ``prate`` (``kg m-2 s-1``) becomes ``mm/h``. The conversion
    is driven by the *actual* GRIB ``units`` attribute on each mapped data
    variable (not by the variable name alone), so a variable that already
    carries the canonical unit is left numerically untouched.

    For each mapped variable present in the dataset:

    * if the ``units`` attribute is absent, the variable is left untouched
      (synthetic in-memory datasets have no GRIB provenance to convert);
    * if the source unit equals the target canonical unit, values are kept and
      the ``units`` attribute is normalized to the canonical string;
    * if the source unit is a known source for the target canonical unit, the
      values are transformed and the ``units`` attribute set to the canonical;
    * otherwise the mapping is rejected with :class:`UnitNormalizationError`
      rather than silently storing values under a possibly-wrong label.

    Args:
        dataset: The mapped dataset (platform variable names).
        variables: The run's :class:`VariableSpec` catalog metadata.

    Returns:
        The dataset with canonical-unit values and ``units`` attributes.

    Raises:
        UnitNormalizationError: If a mapped variable's source unit is present
            but is neither the canonical unit nor a known source for it.
    """
    for variable in variables:
        code = variable.code
        if code not in dataset.data_vars:
            continue
        data_array = dataset[code]
        source_unit = data_array.attrs.get("units")
        if source_unit is None:
            continue
        source_token = _unit_token(str(source_unit))
        target_unit = variable.unit
        if source_token == _unit_token(target_unit):
            data_array.attrs["units"] = target_unit
            continue
        transforms = _SOURCE_TO_CANONICAL.get(target_unit)
        if transforms is None or source_token not in transforms:
            raise UnitNormalizationError(
                f"Cannot canonicalize variable '{code}' from source unit "
                f"'{source_unit}' to canonical unit '{target_unit}': no "
                "supported conversion is defined."
            )
        data_array.values = transforms[source_token](data_array.values)
        data_array.attrs["units"] = target_unit
    return dataset


def ingest_grib_file(
    spec: RunCatalogSpec,
    grib_path: str | Path,
    store_path: str,
    *,
    requested_lead_time_hours: int | None = None,
    member: int | None = None,
) -> ModelRunRecord:
    """Parse a GRIB2 file, write it to a Zarr store, and record it in the catalog.

    This is the production orchestration call path for an already-downloaded
    forecast file. ``store_path`` is the store of the run's *cycle*: one
    ``model_runs`` row represents a full forecast cycle and its store
    accumulates every lead/member (DATABASE.md ``UNIQUE(model_version_id,
    cycle_time)``).

    Write path: the serving store is pre-allocated (a full store covering the
    run's expected leads and, for GEFS, members) and each file is committed to
    its own region:

        1. ``parse_grib2`` decodes and normalizes the GRIB2 file.
        2. Data variables are renamed to the platform vocabulary (per
           ``spec.variables``).
        3. Values are normalized to the platform canonical units.
        4. When ``requested_lead_time_hours`` is given and differs from the
           file's parsed lead, the ingest is aborted with a
           :class:`LeadTimeMismatchError` (the file's decoded lead is
           authoritative).
        5. The single-lead (and, for GEFS, single-member) dataset is written
           to the store via a targeted ``region`` write at the coordinate-
           driven (lead, member) position. Re-ingesting a file replaces only
           its own region; other leads/members are never read or rewritten.
        6. ``record_ingested_dataset`` records the file's product/member rows
           without marking the whole run ready (run-level readiness is decided
           separately by completeness).

    Args:
        spec: The catalog metadata of the run (model, version, cycle, grid,
            variables).
        grib_path: Path to the downloaded GRIB2 file.
        store_path: Zarr store path/URL of the run's cycle.
        requested_lead_time_hours: The lead the caller requested (e.g. the
            CLI's ``--lead-time-hours``). When provided, ingestion fails
            fast if it does not match the file's parsed lead.
        member: The upstream GEFS member identity (``1..30``) for a
            per-member file. ``None`` for deterministic models or combined
            files. The member identity maps to the store's ``member``
            coordinate position, so ``gep17`` lands in member 17 regardless of
            completion order.

    Returns:
        The recorded :class:`ModelRunRecord` (in its current, possibly
        non-ready, state).

    Raises:
        GribParsingError: If the GRIB2 file cannot be decoded.
        UnitNormalizationError: If a mapped variable's source unit cannot be
            safely canonicalized.
        LeadTimeMismatchError: If ``requested_lead_time_hours`` is provided
            and does not match the file's parsed lead.
        ValueError: If the Zarr store target is unsupported.
        sqlalchemy error: If the catalog write fails (no row is committed).
    """
    # The unlocked library path must not mutate a live-run store. The CLI uses
    # the coordinator (which acquires the advisory-lock store gate); library
    # callers of ingest_grib_file that target a live store are refused rather
    # than silently bypassing the concurrency protocol.
    from ingestion.core.base import CycleTombstonedError
    from ingestion.core.catalog import (
        is_cycle_fenced_or_deleted,
        is_live_run_store,
    )

    with _session_local() as db:
        if is_cycle_fenced_or_deleted(db, spec.cycle_time):
            raise CycleTombstonedError(
                f"Refusing ingestion for cycle {spec.cycle_time.isoformat()}: "
                "cycle is claimed for deletion or already tombstoned."
            )
        if is_live_run_store(db, store_path):
            raise LiveStoreOverwriteError(
                f"Refusing to ingest via the unlocked library path into {store_path!r}: "
                "the store belongs to a live model_runs row. Use the coordinated "
                "ingestion path (the weather-ingest CLI coordinator) so the "
                "region-write concurrency protocol is enforced."
            )

    dataset = parse_grib2(grib_path)
    dataset = _normalize_precipitation_increments(
        dataset,
        spec.variables,
        store_path=store_path,
        member=member,
    )
    dataset = _normalize_cloud_cover_intervals(
        dataset,
        spec.variables,
        store_path=store_path,
        member=member,
    )
    dataset = _apply_variable_mapping(dataset, spec.variables)
    dataset = _normalize_canonical_units(dataset, spec.variables)
    # Record the model so the Zarr store is self-describing about its forecast
    # run (model + cycle) independently of the S3 path (ACCEPTANCE_REMEDIATION
    # PLAN §4). ``cycle_time`` is set by the parser from the GRIB ``time``
    # coordinate.
    dataset.attrs["model_id"] = spec.model_id
    _validate_requested_lead(dataset, requested_lead_time_hours)
    _commit_region(
        dataset,
        store_path,
        member=member,
        expected_lead_time_hours=spec.expected_lead_time_hours,
        expected_members=spec.expected_members,
    )
    # Read the actual committed state AFTER the store write (the source of
    # truth for catalog reconciliation and the store↔catalog READY gate). The
    # per-run region writes are serialized (CLI store lock), so this read
    # observes a stable post-write snapshot. When the store cannot be read
    # (e.g. a stubbed/no-op store in tests, or a transient read failure), the
    # committed state is omitted so reconciliation and the store↔catalog gate
    # are skipped — the existing completeness-only status is preserved.
    committed_state: CommittedState | None = None
    try:
        committed_state = read_committed_state(
            store_path,
            is_ensemble=spec.is_ensemble,
        )
    except Exception:  # noqa: BLE001 - a readable store is required for the gate, never fatal
        committed_state = None
    return record_ingested_dataset(
        spec,
        dataset,
        effective_store_path=store_path,
        member=member,
        committed_state=committed_state,
    )


def _validate_requested_lead(
    dataset: xr.Dataset,
    requested_lead_time_hours: int | None,
) -> None:
    """Fail fast when a requested lead disagrees with the parsed dataset.

    The GRIB file's decoded lead is the source of truth (``parser.py`` derives
    ``lead_time_hours`` from the ``step`` coordinate), so a mismatch with the
    requested lead aborts ingestion rather than silently re-ingesting an
    unexpected file or overwriting a lead the caller did not ask for.

    Args:
        dataset: The normalized parsed dataset.
        requested_lead_time_hours: The lead the caller requested, or ``None``
            when no request bound is asserted (library callers without a lead
            concept).

    Raises:
        LeadTimeMismatchError: When the requested lead is provided and does
            not match the dataset's parsed lead.
    """
    if requested_lead_time_hours is None:
        return
    lead_values = dataset["lead_time_hours"].values
    parsed = int(lead_values[0] if np.ndim(lead_values) != 0 else lead_values)
    if parsed != requested_lead_time_hours:
        raise LeadTimeMismatchError(
            f"Downloaded GRIB2 file decodes to lead time {parsed}h, but the "
            f"requested lead time is {requested_lead_time_hours}h. The file "
            "does not match the requested forecast; aborting instead of "
            "ingesting an unexpected lead."
        )


def _validate_requested_member(
    dataset: xr.Dataset,
    requested_member: int | None,
) -> None:
    """Fail fast when a requested GEFS member disagrees with the parsed dataset.

    Args:
        dataset: The normalized parsed dataset.
        requested_member: The member identity the caller requested (e.g. 1..30),
            or None for deterministic models.

    Raises:
        StoreSchemaMismatchError: When the requested member is provided and does
            not match the dataset's parsed member coordinate.
    """
    if requested_member is None:
        return
    if "member" in dataset.coords or "member" in dataset.dims:
        member_values = dataset["member"].values
        parsed = int(member_values[0] if np.ndim(member_values) != 0 else member_values)
        if parsed != requested_member:
            raise StoreSchemaMismatchError(
                f"Downloaded GEFS file decodes to member {parsed}, but the "
                f"requested member is {requested_member}. The file does not "
                "match the requested forecast member; aborting."
            )


def _merge_lead(dataset: xr.Dataset, store_path: str) -> xr.Dataset:
    """Merge a single-lead dataset into a cycle's Zarr store.

    The parser emits ``lead_time_hours`` as a scalar coordinate (one GRIB file
    holds one lead), while a cycle store needs it as a dimension. The dataset
    is expanded first so subsequent leads can be concatenated along
    ``lead_time_hours``. When the cycle store already exists, the new lead is
    merged in and any previous copy of the same lead is replaced (re-ingest
    idempotency); the merged dataset is returned so the caller writes the
    complete cycle back to the store.

    A Zarr store represents exactly one forecast cycle
    (``UNIQUE(model_version_id, cycle_time)`` per DATABASE.md), so **before**
    merging the incoming dataset's cycle identity is validated against the
    existing store's identity. A mismatch is a hard error
    (:class:`CycleStoreMismatchError`) — the merge is refused, never silently
    bypassed. Same-cycle leads are also validated for structural compatibility
    (grid axes, member axis, per-variable dimensions).

    Args:
        dataset: The normalized single-lead dataset for this GRIB file.
        store_path: The cycle store to merge into (may not exist yet).

    Returns:
        The dataset for the whole cycle: the new lead alone on a fresh ingest,
        otherwise the accumulated cycle with the new lead merged in.

    Raises:
        CycleStoreMismatchError: If the incoming dataset's cycle differs from
            the existing store's cycle, or either identity cannot be
            established.
        StoreSchemaMismatchError: If the incoming lead is structurally
            incompatible with the existing store.
    """
    if "lead_time_hours" not in dataset.dims:
        dataset = dataset.expand_dims("lead_time_hours")
    if not store_exists(store_path):
        return dataset

    existing = read_dataset(store_path)
    _validate_store_identity(dataset, existing, store_path)
    _validate_lead_schema(dataset, existing, store_path)
    if "lead_time_hours" not in existing.dims:
        return dataset
    new_lead = int(dataset["lead_time_hours"].values[0])
    keep = (existing["lead_time_hours"] != new_lead).values
    merged = xr.concat(
        [existing.isel(lead_time_hours=keep), dataset],
        dim="lead_time_hours",
        coords="minimal",
    )
    return merged.sortby("lead_time_hours")


def _commit_region(
    dataset: xr.Dataset,
    store_path: str,
    *,
    member: int | None = None,
    expected_lead_time_hours: tuple[int, ...] = (),
    expected_members: tuple[int, ...] = (),
    snapshot: Any | None = None,
) -> None:
    """Commit a single-lead (and optional single-member) file to the store.

    The file's ``lead_time_hours`` (and, for a GEFS per-member file, the
    ``member`` coordinate value) determines the exact region of the
    pre-allocated serving store that is written. Only that region is touched:
    existing data in other leads/members is never read or rewritten.

    When ``snapshot`` is supplied (from the wave coordinator), schema and identity
    validation are performed in-memory and positional indices are passed directly,
    avoiding all redundant remote store opens.

    Args:
        dataset: The normalized single-lead dataset for this GRIB file.
        store_path: The cycle store to commit into (may not exist yet).
        member: The upstream GEFS member identity (``1..30``). ``None`` for
            deterministic models.
        expected_lead_time_hours: The full lead set the run is expected to
            serve; used to pre-allocate the store on the first write.
        expected_members: The full GEFS member set; used to pre-allocate the
            store on the first write.
        snapshot: Optional immutable StoreMetadataSnapshot from the coordinator.

    Raises:
        CycleStoreMismatchError: If the store already exists and the incoming
            cycle differs from the store's cycle.
        StoreSchemaMismatchError: If the incoming lead/member is structurally
            incompatible with the store.
    """
    # Ensure the lead is a dimension (length 1) so region writes map it
    # positionally; the parser emits it as a scalar coordinate.
    if "lead_time_hours" not in dataset.dims:
        dataset = dataset.expand_dims("lead_time_hours")

    if member is not None:
        if "member" not in dataset.dims and "member" not in dataset.coords:
            raise StoreSchemaMismatchError(
                "A GEFS member identity was supplied but the parsed dataset "
                "has no member coordinate or dimension."
            )
        # Pin the member coordinate to the real upstream identity so the
        # region mapping below is coordinate-driven (gep17 -> member 17).
        dataset = dataset.assign_coords(member=[int(member)])
        # Promote a scalar member coordinate to a length-1 dimension AND
        # broadcast each data variable onto it, so the region write maps the
        # member positionally and the data var dims match the store's
        # ``(member, ...)`` layout.
        if "member" not in dataset.dims:
            dataset = dataset.expand_dims("member")
        # If the data vars still lack the member dim (a scalar-member file
        # decodes t2m as ``(latitude, longitude)``), broadcast them onto it.
        need_member = [
            name
            for name in dataset.data_vars
            if "member" not in dataset[name].dims
        ]
        if need_member:
            dataset = dataset.assign(
                {
                    name: dataset[name].expand_dims("member")
                    for name in need_member
                }
            )

    if snapshot is not None:
        _validate_store_identity_from_snapshot(dataset, snapshot, store_path)
        _validate_lead_schema_from_snapshot(dataset, snapshot, store_path)
        lead_val = int(dataset["lead_time_hours"].values[0])
        lead_idx = snapshot.lead_index_map.get(lead_val)
        member_idx = snapshot.member_index_map.get(member) if member is not None else None
        commit_region(
            dataset,
            store_path,
            lead_time_hours=lead_val,
            member=member,
            lead_index=lead_idx,
            member_index=member_idx,
        )
        return

    if not store_exists(store_path):
        # First write: pre-allocate the full serving store with the run's
        # expected leads (and, for GEFS, members), NaN-filled, then commit this
        # file's own region so the store immediately carries real data.
        prepare_run_store(
            dataset,
            store_path,
            expected_lead_time_hours=expected_lead_time_hours,
            expected_members=expected_members,
        )
        commit_region(dataset, store_path)
        return

    existing = read_dataset(store_path)
    _validate_store_identity(dataset, existing, store_path)
    _validate_lead_schema(dataset, existing, store_path)

    commit_region(dataset, store_path)


def _validate_store_identity_from_snapshot(
    dataset: xr.Dataset,
    snapshot: Any,
    store_path: str,
) -> None:
    """In-memory validation of incoming forecast identity against snapshot."""
    requested = _resolve_cycle_time(dataset)
    stored = snapshot.cycle_time
    if requested is None:
        raise CycleStoreMismatchError(
            f"Refusing to merge into {store_path!r}: the incoming forecast has "
            "no cycle/reference time, so its forecast-run identity cannot be "
            "established."
        )
    if stored is None:
        raise CycleStoreMismatchError(
            f"Refusing to merge into {store_path!r}: the existing store carries "
            "no cycle/reference time, so it cannot be identified as a valid "
            "cycle store."
        )
    if requested != stored:
        raise CycleStoreMismatchError(
            f"Refusing to merge: the incoming forecast is cycle {requested}, "
            f"but the store at {store_path!r} already contains cycle {stored}. "
            "A Zarr store represents exactly one forecast cycle; the cycles "
            "must match."
        )


def _validate_lead_schema_from_snapshot(
    dataset: xr.Dataset,
    snapshot: Any,
    store_path: str,
) -> None:
    """In-memory validation of incoming dataset schema against snapshot."""
    for axis in ("latitude", "longitude"):
        if axis in snapshot.coords_values and axis in dataset.coords:
            stored_vals = snapshot.coords_values[axis]
            incoming_vals = dataset.coords[axis].values
            if len(stored_vals) != len(incoming_vals) or not np.allclose(
                np.asarray(stored_vals, dtype=np.float32),
                np.asarray(incoming_vals, dtype=np.float32),
            ):
                raise StoreSchemaMismatchError(
                    f"Refusing to merge into {store_path!r}: the incoming file's "
                    f"'{axis}' axis differs from the store's '{axis}' axis. A "
                    "cycle store must have one consistent grid."
                )

    for code in set(dataset.data_vars) & set(snapshot.data_var_dims):
        incoming_dims = set(dataset[code].dims)
        existing_dims = set(snapshot.data_var_dims[code])
        incoming_dims.discard("member")
        existing_dims.discard("member")
        if incoming_dims != existing_dims:
            raise StoreSchemaMismatchError(
                f"Refusing to merge into {store_path!r}: variable '{code}' has "
                f"dimensions {tuple(dataset[code].dims)} in the incoming file "
                f"but {snapshot.data_var_dims[code]} in the store."
            )


def _resolve_cycle_time(dataset: xr.Dataset) -> str | None:
    """Return a dataset's forecast-run cycle/reference time as a UTC string.

    The authoritative identity is the ``cycle_time`` attribute written by the
    parser from the GRIB ``time`` coordinate. Legacy stores written before that
    attribute existed still carry the ``time`` coordinate (the parser always
    kept it), which is used as the fallback so old stores remain validable.

    Args:
        dataset: A normalized dataset (incoming lead or existing store).

    Returns:
        The cycle time as an ISO 8601 UTC string, or ``None`` when the dataset
        carries no cycle identity.
    """
    if "cycle_time" in dataset.attrs:
        return str(dataset.attrs["cycle_time"])
    if "time" in dataset.coords:
        value = dataset.coords["time"].values
        item = value.item() if np.ndim(value) != 0 else value
        # ``np.datetime_as_string`` on a scalar returns a 0-d ndarray;
        # ``item()`` extracts the plain ``str`` so the type is ``str | None``.
        return str(np.datetime_as_string(np.asarray(item, dtype="datetime64[ns]"), unit="s").item())
    return None


def _validate_store_identity(
    dataset: xr.Dataset,
    existing: xr.Dataset,
    store_path: str,
) -> None:
    """Refuse to merge a dataset whose cycle differs from the store's cycle.

    Fail-fast correctness: a Zarr store represents one forecast cycle and must
    never silently accept data belonging to another cycle. The merge is gated
    on the incoming and stored identities matching exactly; if either identity
    cannot be established the merge is refused (rather than guessing).

    Args:
        dataset: The incoming single-lead dataset.
        existing: The existing cycle store dataset.
        store_path: The store path (used in the error message).

    Raises:
        CycleStoreMismatchError: If the cycles differ or an identity is missing.
    """
    requested = _resolve_cycle_time(dataset)
    stored = _resolve_cycle_time(existing)
    if requested is None:
        raise CycleStoreMismatchError(
            f"Refusing to merge into {store_path!r}: the incoming forecast has "
            "no cycle/reference time, so its forecast-run identity cannot be "
            "established."
        )
    if stored is None:
        raise CycleStoreMismatchError(
            f"Refusing to merge into {store_path!r}: the existing store carries "
            "no cycle/reference time, so it cannot be identified as a valid "
            "cycle store."
        )
    if requested != stored:
        raise CycleStoreMismatchError(
            f"Refusing to merge: the incoming forecast is cycle {requested}, "
            f"but the store at {store_path!r} already contains cycle {stored}. "
            "A Zarr store represents exactly one forecast cycle; the cycles "
            "must match."
        )


def _validate_lead_schema(
    dataset: xr.Dataset,
    existing: xr.Dataset,
    store_path: str,
) -> None:
    """Validate that a same-cycle lead/member is structurally compatible.

    Same-cycle files committed into a cycle store must share the same spatial
    grid (latitude/longitude axes); a variable present in both must have the
    same dimensions (except that a GEFS per-member file's ``member`` dimension
    of length 1 is compatible with a multi-member store — the region write
    targets exactly one member). Adding a previously-absent variable is
    allowed (a lead file may omit a field the parser skips); redefining an
    existing variable's structure is not.

    Args:
        dataset: The incoming single-lead (optionally single-member) dataset.
        existing: The existing cycle store dataset.
        store_path: The store path (used in the error message).

    Raises:
        StoreSchemaMismatchError: If the incoming file is incompatible.
    """
    for axis in ("latitude", "longitude"):
        if axis in existing.coords and axis in dataset.coords:
            if not np.array_equal(existing.coords[axis].values, dataset.coords[axis].values):
                raise StoreSchemaMismatchError(
                    f"Refusing to merge into {store_path!r}: the incoming file's "
                    f"'{axis}' axis differs from the store's '{axis}' axis. A "
                    "cycle store must have one consistent grid."
                )
    for code in set(dataset.data_vars) & set(existing.data_vars):
        # A per-member GEFS file has a length-1 member dim; the store has the
        # full member axis. Allow the member dim to differ (region write
        # targets one member), but all other dims must match exactly.
        incoming_dims = set(dataset[code].dims)
        existing_dims = set(existing[code].dims)
        if "member" in incoming_dims and "member" not in existing_dims:
            raise StoreSchemaMismatchError(
                f"Refusing to merge into {store_path!r}: variable '{code}' has "
                f"a member dimension in the incoming file but the store does not."
            )
        if "member" in existing_dims and "member" in incoming_dims:
            # The non-member dims must match; the member axis length differs by
            # design (single-member file into multi-member store).
            incoming_dims.discard("member")
            existing_dims.discard("member")
        if incoming_dims != existing_dims:
            raise StoreSchemaMismatchError(
                f"Refusing to merge into {store_path!r}: variable '{code}' has "
                f"dimensions {tuple(dataset[code].dims)} in the incoming file "
                f"but {tuple(existing[code].dims)} in the store."
            )
