"""Point forecast construction: Zarr slicing, grid interpolation, and units.

This service builds a ``PointForecastData`` for an already-resolved
location. It does not geocode addresses: locations are resolved by the
``/v1/search`` endpoint or provided directly as coordinates or platform ids
(see the ``/v1/points`` router).

The grid geometry is derived from the forecast dataset's own
``latitude``/``longitude`` coordinate arrays (origin, step, row/column
counts), assuming a regular, uniformly spaced rectilinear grid. The schema
stores only ``resolution_km``; the approved design documents do not define
grid origin/dimensions, so deriving them from the data avoids introducing
undocumented platform conventions. Non-uniform or non-surface data is out
of scope for Milestone 9 and raises a clear error.
"""

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import xarray as xr
from domain.exceptions import (
    InvalidCoordinatesError,
    InvalidGridError,
    PointOutsideGridError,
)
from domain.geo.coordinates import validate_coordinates
from domain.geo.grid import RegularGrid
from domain.models.precipitation import classify_precipitation_phase
from domain.models.wind import (
    CALM_WIND_THRESHOLD_MPS,
    derive_meteorological_direction,
    get_cardinal_direction,
)
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.models.entities import (
    City,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
    SkiResort,
)
from api.schemas import ForecastLocationOut, ForecastSeries, PointForecastData
from api.services.elevation import get_elevation_provider

logger = logging.getLogger(__name__)

#: ModelRun statuses eligible for serving candidate discovery.
SERVING_ELIGIBLE_STATUSES: tuple[str, ...] = ("ready", "processing", "partial")

#: ``resolved_via`` value for a location resolved from raw coordinates.
RESOLVED_VIA_COORDINATES = "coordinates"
#: ``resolved_via`` value for a location resolved from a city record.
RESOLVED_VIA_CITY = "city"
#: ``resolved_via`` value for a location resolved from a ski resort record.
RESOLVED_VIA_RESORT = "resort"


@dataclass(frozen=True)
class ResolvedLocation:
    """A geographic location resolved from a point-forecast spatial specifier.

    Attributes:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        elevation_m: Elevation in meters, when the resolved record defines one.
        resolved_via: How the location was resolved (coordinates, city, or
            resort).
        id: Stable identity of the resolved location record (the ``cities``
            or ``ski_resorts`` primary key), or ``None`` when resolved from
            raw coordinates. Used as a cache-key discriminator so distinct
            records that share coordinates cannot collide.
    """

    latitude: float
    longitude: float
    elevation_m: float | None
    resolved_via: str
    id: str | None = None


#: SI -> imperial conversions applied when ``units=imperial`` (API.md 2.6).
#: Conversion applies only when the variable's registered unit matches a
#: known pair; unknown units are returned unconverted.
_SI_TO_IMPERIAL: dict[str, tuple[str, Callable[[float], float]]] = {
    "°C": ("°F", lambda celsius: celsius * 9.0 / 5.0 + 32.0),
    "mm/h": ("in/h", lambda mm: mm / 25.4),
    "mm": ("in", lambda mm: mm / 25.4),
    "km/h": ("mph", lambda kmh: kmh * 0.621371),
    "%": ("%", lambda rh: rh),
    "m": ("mi", lambda m: m / 1609.344),
}

#: Variable-specific conversions taking precedence over generic unit matching.
_VARIABLE_IMPERIAL_CONVERSIONS: dict[str, tuple[str, Callable[[float], float]]] = {
    "snow_depth": ("in", lambda m: m * 39.3700787),
    "visibility": ("mi", lambda m: m / 1609.344),
    "wind_10m": ("mph", lambda kmh: kmh * 0.621371),
    "precipitation_amount_3h": ("in", lambda mm: mm / 25.4),
    "cloud_ceiling": ("ft", lambda m: m * 3.28084),
}


def resolve_location(
    db: Session,
    *,
    lat: float | None = None,
    lon: float | None = None,
    city_id: str | None = None,
    resort_id: str | None = None,
) -> ResolvedLocation:
    """Resolve exactly one spatial specifier to a geographic location.

    The specifier must be exactly one of: a ``lat``/``lon`` pair, a
    ``city_id``, or a ``resort_id``. Providing none, more than one, or a
    partial coordinate pair is rejected. ``address`` is intentionally not
    accepted: this endpoint serves forecasts for already-resolved locations
    (API.md section 2.1 lists ``address`` but the schema has no geocoding
    table; see the milestone spec gap).

    Args:
        db: Database session.
        lat: Latitude (required with ``lon``).
        lon: Longitude (required with ``lat``).
        city_id: A ``cities.id`` primary key.
        resort_id: A ``ski_resorts.id`` primary key.

    Returns:
        The resolved location.

    Raises:
        HTTPException: 422 for an invalid or ambiguous specifier, 404 if a
            referenced city or ski resort does not exist.
    """
    if (lat is None) != (lon is None):
        raise HTTPException(
            status_code=422,
            detail="lat and lon must be provided together.",
        )
    specifier_count = (
        int(lat is not None and lon is not None)
        + int(city_id is not None)
        + int(resort_id is not None)
    )
    if specifier_count == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Exactly one spatial specifier is required: lat and lon, "
                "city_id, or resort_id."
            ),
        )
    if specifier_count > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide exactly one spatial specifier: lat and lon, "
                "city_id, or resort_id."
            ),
        )

    if lat is not None and lon is not None:
        try:
            validate_coordinates(lat, lon)
        except InvalidCoordinatesError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ResolvedLocation(
            latitude=lat,
            longitude=lon,
            elevation_m=_elevation_for(lat, lon),
            resolved_via=RESOLVED_VIA_COORDINATES,
        )

    if city_id is not None:
        return _resolve_city(db, city_id)

    # The specifier_count guard above guarantees resort_id is non-None when we
    # reach here (it is the only remaining specifier that could count to 1).
    # Assert to narrow the type and document the invariant without a type-ignore.
    assert resort_id is not None
    return _resolve_ski_resort(db, resort_id)


def _resolve_city(db: Session, city_id: str) -> ResolvedLocation:
    row = db.execute(
        select(City, func.ST_X(City.geom), func.ST_Y(City.geom)).where(
            City.id == city_id
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"City '{city_id}' was not found.")
    return ResolvedLocation(
        latitude=float(row[2]),
        longitude=float(row[1]),
        # Cities have no elevation column in the Milestone 3 schema, so the
        # elevation is resolved from the coordinate via the elevation provider.
        elevation_m=_elevation_for(float(row[2]), float(row[1])),
        resolved_via=RESOLVED_VIA_CITY,
        id=row[0].id,
    )


def _elevation_for(latitude: float, longitude: float) -> float | None:
    """Resolve terrain elevation (meters) for a coordinate, or ``None``.

    Delegates to the configured elevation provider (a local/server-side DEM by
    default). The provider returns ``None`` for no-data/ocean/unavailable, so
    this never fabricates a value and never raises for a missing elevation.
    """
    return get_elevation_provider().get_elevation(latitude, longitude)


def _resolve_ski_resort(db: Session, resort_id: str) -> ResolvedLocation:
    row = db.execute(
        select(SkiResort, func.ST_X(SkiResort.geom), func.ST_Y(SkiResort.geom)).where(
            SkiResort.id == resort_id
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"Ski resort '{resort_id}' was not found."
        )
    return ResolvedLocation(
        latitude=float(row[2]),
        longitude=float(row[1]),
        elevation_m=row[0].summit_elevation_m,
        resolved_via=RESOLVED_VIA_RESORT,
        id=row[0].id,
    )


def build_point_forecast(
    db: Session,
    *,
    location: ResolvedLocation,
    model: str,
    variables: list[str] | None,
    units: str,
    start_lead_time_hours: int | None,
    end_lead_time_hours: int | None,
) -> PointForecastData:
    """Build a point forecast payload for a resolved location and model.

    This is a **cross-cycle** deterministic time series: for every
    ``valid_time``, the payload selects the READY deterministic forecast record
    with the **minimum ``lead_time_hours``** across all READY cycles of the
    model (DATABASE.md: ``valid_time = cycle_time + lead_time_hours``). Because
    cycles are 00/06/12/18 and lead differences are multiples of 6, the minimum
    lead for a fixed valid_time is the newest cycle that covers it.

    The winner selection is done **entirely from catalog metadata** (a
    ``model_runs`` ⋈ ``forecast_products`` query) — no large Zarr dataset is
    opened to decide which record wins. Only the winning runs' Zarr stores are
    opened, one per distinct winning cycle, and reused across their selected
    leads. Each forecast entry carries its source ``cycle_time`` so the
    provenance of a mixed-cycle series is unambiguous.

    Args:
        db: Database session.
        location: The resolved location.
        model: A single model identifier.
        variables: Requested variable codes, or ``None`` to return the
            documented ``forecast_variables`` catalog entries present in the
            dataset.
        units: ``metric`` (default) or ``imperial``.
        start_lead_time_hours: Inclusive lower bound of the lead-time window.
        end_lead_time_hours: Inclusive upper bound of the lead-time window.

    Returns:
        The point forecast payload.

    Raises:
        HTTPException: 404 when no ready run, no data for the location, or an
            unknown variable is encountered; 422/500 for invalid data.
    """
    candidates = _select_min_lead_winners(
        db,
        model,
        start_lead_time_hours=start_lead_time_hours,
        end_lead_time_hours=end_lead_time_hours,
    )
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No ready forecast run with data was found for model '{model}'."
            ),
        )

    # Phase 1 remediation: metadata (lead/var sets) and point interpolations are
    # read through *bounded* gate selectors — never the full store. A broken
    # newest store, or a store whose lead coordinate lacks the selected lead
    # (stale forecast_products metadata after a same-cycle re-ingest),
    # therefore falls back to the next READY candidate covering the same
    # valid_time instead of dropping the record or raising a KeyError.
    by_cycle: dict[datetime, _CycleMetadata] = {}

    def _open_cycle(cycle_time: datetime) -> _CycleMetadata | None:
        if cycle_time in by_cycle:
            return by_cycle[cycle_time]
        try:
            store_path = _resolve_cycle_store_path(db, model, cycle_time)
        except HTTPException:
            return None
        if store_path is None:
            return None
        try:
            metadata = gated_cycle_metadata(store_path)
        except Exception:  # noqa: BLE001 - unreadable store
            return None
        by_cycle[cycle_time] = metadata
        return metadata

    resolved: dict[datetime, tuple[datetime, int]] = {}
    for valid_time, pairs in candidates.items():
        for cycle_time, lead in pairs:
            metadata = _open_cycle(cycle_time)
            if metadata is None or lead not in metadata.lead_times:
                continue
            resolved[valid_time] = (cycle_time, lead)
            break

    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"No readable forecast data was found for model '{model}'.",
        )

    # Resolve variables from the union of the opening cycles' var sets.
    merged_names = _merge_var_names(by_cycle)
    var_codes = _resolve_variables(
        db,
        _CycleMetadata(lead_times=frozenset(), var_names=frozenset(merged_names)),
        variables,
    )
    units_by_code = _variable_units(db, var_codes)

    forecasts: list[ForecastSeries] = []
    for valid_time, (cycle_time, lead) in sorted(resolved.items()):
        store_path = _resolve_cycle_store_path(db, model, cycle_time)
        if store_path is None:
            continue
        # One bounded gate session interpolates every requested variable at the
        # point and lead (the small 2x2 neighborhood read under the SHARED lock).
        values_by_var = gated_point_interpolations(
            store_path,
            var_codes=tuple(var_codes),
            lead=lead,
            latitude=location.latitude,
            longitude=location.longitude,
        )
        if values_by_var is None:
            # The store became unreadable between winner resolution and
            # interpolation; drop this record rather than failing the request.
            continue
        entry: dict[str, Any] = {
            "lead_time_hours": lead,
            "valid_time": valid_time,
            # Cross-cycle provenance: the cycle that produced this entry.
            "cycle_time": cycle_time,
        }
        for var_code in var_codes:
            if var_code == "wind_10m":
                raw_kmh = float(values_by_var["wind_10m"])
                converted_speed = _convert_value(
                    raw_kmh, "km/h", units, var_code="wind_10m"
                )
                entry["wind_10m"] = converted_speed
                direction_deg = values_by_var.get("_wind_direction_10m")
                entry["wind_direction_10m"] = (
                    round(float(direction_deg), 1)
                    if direction_deg is not None and not math.isnan(float(direction_deg))
                    else None
                )
                entry["wind_cardinal_10m"] = str(values_by_var.get("_wind_cardinal_10m", "CALM"))
            elif var_code == "precipitation_amount_3h":
                raw_mm = values_by_var.get("precipitation_amount_3h")
                if raw_mm is None or (isinstance(raw_mm, float) and math.isnan(raw_mm)):
                    entry["precipitation_amount_3h"] = None
                    entry["precipitation_type"] = "none"
                    entry["precipitation_transition"] = "none"
                    entry["precipitation_start_type"] = "none"
                    entry["precipitation_end_type"] = "none"
                    entry["precipitation_evidence"] = "exact"
                else:
                    converted = _convert_value(
                        float(raw_mm), "mm", units, var_code="precipitation_amount_3h"
                    )
                    entry["precipitation_amount_3h"] = converted
                    entry["precipitation_type"] = str(values_by_var.get("_precipitation_type", "none"))
                    entry["precipitation_transition"] = str(values_by_var.get("_precipitation_transition", "none"))
                    entry["precipitation_start_type"] = str(values_by_var.get("_precipitation_start_type", "none"))
                    entry["precipitation_end_type"] = str(values_by_var.get("_precipitation_end_type", "none"))
                    entry["precipitation_evidence"] = str(values_by_var.get("_precipitation_evidence", "exact"))
            elif var_code == "cloud_cover_3h":
                raw_cc = values_by_var.get("cloud_cover_3h")
                if raw_cc is None or (isinstance(raw_cc, float) and math.isnan(raw_cc)):
                    entry["cloud_cover_3h"] = None
                else:
                    entry["cloud_cover_3h"] = round(float(raw_cc), 1)
            elif var_code == "cloud_ceiling":
                raw_ceil = values_by_var.get("cloud_ceiling")
                if raw_ceil is None or (isinstance(raw_ceil, float) and math.isnan(raw_ceil)):
                    entry["cloud_ceiling"] = None
                    entry["cloud_ceiling_unlimited"] = False
                else:
                    val_m = float(raw_ceil)
                    if val_m >= 19990.0:
                        entry["cloud_ceiling"] = None
                        entry["cloud_ceiling_unlimited"] = True
                    else:
                        converted = _convert_value(
                            val_m, "m", units, var_code="cloud_ceiling"
                        )
                        entry["cloud_ceiling"] = round(converted, 1)
                        entry["cloud_ceiling_unlimited"] = False
            else:
                value = float(values_by_var[var_code])
                entry[var_code] = _convert_value(
                    value, units_by_code[var_code], units, var_code=var_code
                )
        forecasts.append(ForecastSeries(**entry))

    if not forecasts:
        raise HTTPException(
            status_code=404,
            detail=f"No readable forecast data was found for model '{model}'.",
        )

    # ``generated_at`` is the newest winning cycle (the "generation time" of the
    # series), keeping the payload deterministic.
    newest_cycle = max(by_cycle)
    return PointForecastData(
        location=ForecastLocationOut(
            latitude=location.latitude,
            longitude=location.longitude,
            elevation_m=location.elevation_m,
            resolved_via=location.resolved_via,
        ),
        generated_at=newest_cycle,
        model=model,
        forecasts=forecasts,
    )


def _select_min_lead_winners(
    db: Session,
    model: str,
    *,
    start_lead_time_hours: int | None = None,
    end_lead_time_hours: int | None = None,
) -> dict[datetime, list[tuple[datetime, int]]]:
    """Return the minimum-lead candidate(s) per valid_time.

    For every READY run of the model with a store, the candidate forecast data
    is discovered from two sources:

    * ``forecast_products`` rows when present (the fast metadata path); and
    * the run's Zarr ``lead_time_hours`` coordinate when no product rows exist
      (a READY run with a readable store remains servable — the legacy
      contract).

    For each ``valid_time``, the candidates are ordered by ascending lead (the
    minimum lead is the newest cycle that covers it). ``build_point_forecast``
    tries them in that order, so a broken newest store falls back to the next
    READY readable candidate for the same valid_time instead of dropping the
    record. Only READY runs participate; partial/ingesting/failed runs are never
    candidates.

    Args:
        db: Database session.
        model: A single deterministic model identifier.
        start_lead_time_hours: Inclusive lower lead bound.
        end_lead_time_hours: Inclusive upper lead bound.

    Returns:
        A mapping ``valid_time -> list[(cycle_time, lead_time_hours)]`` ordered
        by ascending lead (best candidate first).
    """
    # Fast path: valid_times derivable from forecast_products metadata.
    stmt = (
        select(
            ModelRun.cycle_time,
            ForecastProduct.lead_time_hours,
        )
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .join(ForecastProduct, ForecastProduct.run_id == ModelRun.id)
        .where(Model.model_id == model)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    candidates_by_valid: dict[datetime, set[tuple[datetime, int]]] = {}
    # Cycle times that supplied at least one candidate via forecast_products.
    product_cycles: set[datetime] = set()

    def _add(cycle_time: datetime, lead: int) -> None:
        if cycle_time.tzinfo is None:
            cycle_time = cycle_time.replace(tzinfo=timezone.utc)
        if start_lead_time_hours is not None and lead < start_lead_time_hours:
            return
        if end_lead_time_hours is not None and lead > end_lead_time_hours:
            return
        valid_time = cycle_time + timedelta(hours=lead)
        candidates_by_valid.setdefault(valid_time, set()).add((cycle_time, lead))

    for cycle_time, lead in db.execute(stmt).all():
        if cycle_time.tzinfo is None:
            cycle_time = cycle_time.replace(tzinfo=timezone.utc)
        product_cycles.add(cycle_time)
        _add(cycle_time, lead)

    # Fallback discovery: READY runs with a store but no forecast_products rows
    # still serve via their Zarr lead coordinate. Read the store's lead axis
    # once per distinct cycle (metadata only; cheap) to enumerate candidates.
    # A cycle is skipped only when it ALREADY contributed product candidates;
    # a ready run with no products must still be discovered from its store.
    runs_stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    for run in db.execute(runs_stmt).scalars().all():
        cycle_time = run.cycle_time
        if cycle_time.tzinfo is None:
            cycle_time = cycle_time.replace(tzinfo=timezone.utc)
        if cycle_time in product_cycles:
            # Products already supplied this cycle's candidate leads.
            continue
        assert run.zarr_store_path is not None
        try:
            # Candidate discovery reads only the store's lead coordinate
            # metadata (bounded), under the reader gate so a store mid-re-ingest
            # is never opened — never the full gridded data.
            metadata = gated_cycle_metadata(str(run.zarr_store_path))
        except Exception as exc:  # noqa: BLE001 - unreadable store
            logger.warning(
                "Skipping unreadable Zarr store for run %s: %s", run.id, exc
            )
            continue
        leads = sorted(metadata.lead_times)
        for lead in leads:
            _add(cycle_time, lead)

    # Order each valid_time's candidates by ascending lead (best first).
    return {
        valid_time: sorted(pairs, key=lambda pair: pair[1])
        for valid_time, pairs in candidates_by_valid.items()
    }


def _resolve_cycle_store_path(
    db: Session, model: str, cycle_time: datetime
) -> str | None:
    """Return the READY run's Zarr store path for a cycle, or ``None``.

    Catalog-only; no store I/O.
    """
    run = (
        db.execute(
            select(ModelRun.zarr_store_path)
            .join(ModelRun.model_version)
            .join(ModelVersion.model)
            .where(Model.model_id == model)
            .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
            .where(ModelRun.zarr_store_path.isnot(None))
            .where(ModelRun.cycle_time == cycle_time)
        )
        .scalars()
        .one_or_none()
    )
    return str(run) if run is not None else None


@dataclass(frozen=True)
class _CycleMetadata:
    """Metadata read from a cycle's store (no gridded data materialized).

    Attributes:
        lead_times: The store's ``lead_time_hours`` coordinate values.
        var_names: The store's data-variable names.
    """

    lead_times: frozenset[int]
    var_names: frozenset[str]


def gated_cycle_metadata(store_path: str) -> _CycleMetadata:
    """Read a cycle store's lead-time and variable metadata under the gate.

    Reads only coordinate/data-var names (tiny) — never the gridded variables —
    so a point request never materializes the global field just to discover
    what the store carries.
    """
    from api.core.reader_gate import gated_read_dataset_with_selector

    def select_metadata(dataset: xr.Dataset) -> _CycleMetadata:
        leads: set[int] = set()
        if "lead_time_hours" in dataset.coords:
            coord = dataset.coords["lead_time_hours"].values
            if np.ndim(coord) == 0:
                leads.add(int(coord))
            else:
                leads = {int(v) for v in coord}
        names = {str(name) for name in dataset.data_vars}
        return _CycleMetadata(lead_times=frozenset(leads), var_names=frozenset(names))

    return gated_read_dataset_with_selector(store_path, select_metadata)


def gated_point_interpolations(
    store_path: str,
    *,
    var_codes: tuple[str, ...],
    lead: int,
    latitude: float,
    longitude: float,
) -> dict[str, Any] | None:
    """Interpolate every requested variable at a point/lead under the gate.

    A single SHARED gate session opens the lazy store, derives the grid
    (coordinates), and for each variable crops the 2x2 interpolation
    neighborhood around the point, reduces the member axis (GEFS), and
    materializes only that tiny window. Returns ``{var_code: value}``, or
    ``None`` if the store is unreadable (caller drops the record).
    """
    from api.core.reader_gate import gated_read_dataset_with_selector

    def select_and_interpolate(dataset: xr.Dataset) -> dict[str, Any]:
        grid, lat_desc, lon_desc = _derive_grid(dataset)
        out: dict[str, Any] = {}
        for var_code in var_codes:
            if var_code == "wind_10m":
                if "wind_u_10m" not in dataset.data_vars or "wind_v_10m" not in dataset.data_vars:
                    raise HTTPException(
                        status_code=404,
                        detail=(
                            "Variable 'wind_10m' requires 'wind_u_10m' and 'wind_v_10m' in the forecast dataset."
                        ),
                    )
                field_u = dataset["wind_u_10m"]
                field_v = dataset["wind_v_10m"]
                if "lead_time_hours" in field_u.dims:
                    field_u = field_u.sel(lead_time_hours=lead)
                if "lead_time_hours" in field_v.dims:
                    field_v = field_v.sel(lead_time_hours=lead)
                if field_u.ndim not in (2, 3) or field_v.ndim not in (2, 3):
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Variable 'wind_10m' components are not 2-D/3-D (member) surface fields; "
                            "vertical-level variables are not supported."
                        ),
                    )
                u_val = float(
                    _interpolate_neighborhood(
                        field_u, grid, lat_desc, lon_desc, latitude, longitude
                    )
                )
                v_val = float(
                    _interpolate_neighborhood(
                        field_v, grid, lat_desc, lon_desc, latitude, longitude
                    )
                )
                speed_mps = math.hypot(u_val, v_val)
                speed_kmh = speed_mps * 3.6
                direction_deg = derive_meteorological_direction(
                    u_val, v_val, calm_threshold=CALM_WIND_THRESHOLD_MPS
                )
                cardinal_str = get_cardinal_direction(direction_deg) if direction_deg is not None else "CALM"
                out["wind_10m"] = speed_kmh
                out["_wind_direction_10m"] = direction_deg if direction_deg is not None else float("nan")
                out["_wind_cardinal_10m"] = cardinal_str
                continue

            if var_code == "precipitation_amount_3h":
                if "precipitation_amount_3h" not in dataset.data_vars:
                    raise HTTPException(
                        status_code=404,
                        detail="Variable 'precipitation_amount_3h' is not available in the forecast dataset.",
                    )
                field_p = dataset["precipitation_amount_3h"]
                if "lead_time_hours" in field_p.dims:
                    field_p = field_p.sel(lead_time_hours=lead)
                amt_val = float(
                    _interpolate_neighborhood(
                        field_p, grid, lat_desc, lon_desc, latitude, longitude
                    )
                )
                out["precipitation_amount_3h"] = amt_val

                # Optional categorical flags interpolation
                flags_curr: dict[str, int] = {}
                for f_code in ("crain", "csnow", "cfrzr", "cicep"):
                    if f_code in dataset.data_vars:
                        f_field = dataset[f_code]
                        if "lead_time_hours" in f_field.dims:
                            f_field = f_field.sel(lead_time_hours=lead)
                        f_val = float(
                            _interpolate_neighborhood(
                                f_field, grid, lat_desc, lon_desc, latitude, longitude
                            )
                        )
                        flags_curr[f_code] = 1 if f_val >= 0.5 else 0

                # Optional t2m
                t2m_val: float | None = None
                if "temperature_2m" in dataset.data_vars:
                    t_field = dataset["temperature_2m"]
                    if "lead_time_hours" in t_field.dims:
                        t_field = t_field.sel(lead_time_hours=lead)
                    t2m_val = float(
                        _interpolate_neighborhood(
                            t_field, grid, lat_desc, lon_desc, latitude, longitude
                        )
                    )

                # Predecessor contextual evidence for 6-hour reset leads (t=6, 12, 18, 24, ...)
                amt_prev: float | None = None
                flags_prev: dict[str, int] | None = None
                t2m_start: float | None = None

                if lead % 6 == 0 and lead > 0:
                    pred_lead = lead - 3
                    leads_in_ds = [
                        int(v)
                        for v in np.atleast_1d(dataset.coords["lead_time_hours"].values).reshape(-1)
                    ]
                    if pred_lead in leads_in_ds:
                        p_field_prev = dataset["precipitation_amount_3h"].sel(
                            lead_time_hours=pred_lead
                        )
                        amt_prev = float(
                            _interpolate_neighborhood(
                                p_field_prev, grid, lat_desc, lon_desc, latitude, longitude
                            )
                        )
                        f_prev = {}
                        for f_code in ("crain", "csnow", "cfrzr", "cicep"):
                            if f_code in dataset.data_vars:
                                f_field_p = dataset[f_code].sel(lead_time_hours=pred_lead)
                                f_p_val = float(
                                    _interpolate_neighborhood(
                                        f_field_p, grid, lat_desc, lon_desc, latitude, longitude
                                    )
                                )
                                f_prev[f_code] = 1 if f_p_val >= 0.5 else 0
                        if f_prev:
                            flags_prev = f_prev

                        if "temperature_2m" in dataset.data_vars:
                            t_field_p = dataset["temperature_2m"].sel(lead_time_hours=pred_lead)
                            t2m_start = float(
                                _interpolate_neighborhood(
                                    t_field_p, grid, lat_desc, lon_desc, latitude, longitude
                                )
                            )

                phase_state = classify_precipitation_phase(
                    amt_val,
                    flags_curr if flags_curr else None,
                    amount_prev=amt_prev,
                    flags_prev=flags_prev,
                    t2m_start=t2m_start,
                    t2m_end=t2m_val,
                )
                out["_precipitation_type"] = phase_state.interval_type.value
                out["_precipitation_transition"] = phase_state.transition.value
                out["_precipitation_start_type"] = phase_state.start_type.value
                out["_precipitation_end_type"] = phase_state.end_type.value
                out["_precipitation_evidence"] = phase_state.evidence.value
                continue

            if var_code not in dataset.data_vars:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Variable '{var_code}' is not available in the forecast dataset."
                    ),
                )
            field = dataset[var_code]
            if "lead_time_hours" in field.dims:
                field = field.sel(lead_time_hours=lead)
            # Phase 1: member-mean is performed by _interpolate_neighborhood
            # AFTER the 2x2 crop, so only the window's chunks are read.
            if field.ndim not in (2, 3):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Variable '{var_code}' is not a 2-D/3-D (member) surface field; "
                        "vertical-level variables are not supported."
                    ),
                )
            out[var_code] = float(
                _interpolate_neighborhood(
                    field, grid, lat_desc, lon_desc, latitude, longitude
                )
            )
        return out

    from api.core.reader_gate import ReaderGateTimeout

    try:
        return gated_read_dataset_with_selector(store_path, select_and_interpolate)
    except HTTPException:
        raise
    except ReaderGateTimeout:
        raise
    except PointOutsideGridError as exc:
        # The point is outside the grid: a 404 (the historical contract).
        raise HTTPException(
            status_code=404,
            detail=(f"No forecast data covers the requested location: {exc}"),
        ) from exc
    except InvalidGridError as exc:
        raise HTTPException(
            status_code=500,
            detail="The forecast dataset grid is invalid.",
        ) from exc
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - unreadable store
        return None


def _merge_var_names(by_cycle: dict[datetime, _CycleMetadata]) -> Iterable[str]:
    """Union the data-variable names across the opening cycles' metadata."""
    names: set[str] = set()
    for metadata in by_cycle.values():
        names.update(metadata.var_names)
    return sorted(names)


def resolve_latest_run_serving_generation(
    db: Session, model: str, initial_time: str | None = None
) -> str | None:
    """Return the committed-manifest generation for the latest ready run.

    The generation is the cache-generation discriminator: a same-set same-cycle
    data replacement changes the manifest generation, making old cache entries
    unreachable. Returns ``None`` when no manifest exists (legacy store) or when
    no ready run exists.

    Returns:
        The serving generation string, or ``None`` when no ready run exists, no
        manifest exists, or the store path cannot be resolved.
    """
    from api.core.manifest_reader import ManifestReadError, manifest_generation

    stmt = (
        select(ModelRun.zarr_store_path)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        stmt = stmt.where(ModelRun.cycle_time == _parse_cycle_time(initial_time))
    stmt = stmt.order_by(ModelRun.cycle_time.desc())
    path = db.execute(stmt).scalars().first()
    if path is None:
        return None
    try:
        return manifest_generation(path)
    except ManifestReadError:
        # A malformed manifest fails closed: do not serve a stale cache key.
        return None


def resolve_latest_run_cycle_time(
    db: Session, model: str, initial_time: str | None = None
) -> str | None:
    """Return the resolved run's cycle time for a model, or ``None``.

    The cache key for a point/probability/ensemble request must include the
    resolved forecast run's cycle so a cached response for one cycle never
    satisfies a request for another (ACCEPTANCE_REMEDIATION_PLAN §9). This is a
    lightweight DB lookup (the newest eligible run with a store, optionally
    pinned to a specific cycle via ``initial_time``); the heavy store open
    happens in the compute path.

    Args:
        db: Database session.
        model: A single model identifier.
        initial_time: Optional ISO 8601 UTC cycle time pinning the run. When
            provided, the run at exactly that cycle is resolved (GAP-2).

    Returns:
        The run's ``cycle_time`` as an ISO 8601 UTC string, or ``None`` when no
        matching run exists.
    """
    stmt = (
        select(ModelRun.cycle_time)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        stmt = stmt.where(ModelRun.cycle_time == _parse_cycle_time(initial_time))
    stmt = stmt.order_by(ModelRun.cycle_time.desc())
    value = db.execute(stmt).scalars().first()
    if value is None:
        return None
    cycle = value
    if cycle.tzinfo is None:
        cycle = cycle.replace(tzinfo=timezone.utc)
    return cycle.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_cycle_time(value: str) -> datetime:
    """Parse an ISO 8601 UTC cycle time string into an aware UTC datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_ready_dataset(
    db: Session, model_id: str, initial_time: str | None = None
) -> tuple[ModelRun, _CycleMetadata]:
    """Return the newest eligible run for a model whose Zarr store opens.

    Runs in ready, processing, and partial statuses are ordered newest-first;
    each candidate's store is probed in turn (bounded metadata read under the
    SHARED gate) and the first readable one is returned with its cycle metadata.
    A corrupted, truncated, or momentarily-unreachable store on the newest run
    therefore falls through to the next-newest readable run instead of failing
    the request.

    Args:
        db: Database session.
        model_id: A single model identifier.
        initial_time: Optional ISO 8601 UTC cycle time pinning the run. When
            provided, only the run at that cycle is considered (GAP-2).

    Returns:
        A ``(run, metadata)`` pair for the first readable eligible run, where
        ``metadata`` exposes the store's lead times and variable names (no
        gridded data materialized).

    Raises:
        HTTPException: 404 when no eligible run exists or none of the runs
            has a readable store.
    """
    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model_id)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        stmt = stmt.where(ModelRun.cycle_time == _parse_cycle_time(initial_time))
    stmt = stmt.order_by(ModelRun.cycle_time.desc())
    runs = list(db.execute(stmt).scalars().all())
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No ready forecast run with data was found for model '{model_id}'."
            ),
        )
    for run in runs:
        assert run.zarr_store_path is not None
        try:
            # Reader-gate: SHARED store gate + fresh Core revalidation so a
            # store mid-re-ingest is never read. Reads only coordinate/var-name
            # metadata (bounded), never the gridded variables.
            metadata = gated_cycle_metadata(str(run.zarr_store_path))
        except Exception as exc:  # noqa: BLE001 - probe store, fall through
            logger.warning(
                "Skipping unreadable Zarr store for run %s (%s): %s",
                run.id,
                run.zarr_store_path,
                exc,
            )
            continue
        return run, metadata
    raise HTTPException(
        status_code=404,
        detail=(
            f"No readable forecast run with data was found for model '{model_id}'."
        ),
    )


def _resolve_lead_times(
    source: _CycleMetadata | xr.Dataset,
    start: int | None,
    end: int | None,
) -> list[int]:
    available: list[int]
    if isinstance(source, _CycleMetadata):
        available = sorted(source.lead_times)
        if not available:
            raise HTTPException(
                status_code=404,
                detail="The forecast dataset has no lead_time_hours coordinate.",
            )
    else:
        if "lead_time_hours" not in source.coords:
            raise HTTPException(
                status_code=404,
                detail="The forecast dataset has no lead_time_hours coordinate.",
            )
        coord = source.coords["lead_time_hours"].values
        if np.ndim(coord) == 0:
            available = [int(coord)]
        else:
            available = [int(value) for value in coord]
    selected = [
        lead
        for lead in sorted(available)
        if (start is None or lead >= start) and (end is None or lead <= end)
    ]
    if not selected:
        raise HTTPException(
            status_code=404,
            detail="No forecast data is available for the requested lead-time range.",
        )
    return selected


#: Internal platform variables that are not returned in default point forecasts.
INTERNAL_VARIABLES: frozenset[str] = frozenset({"wind_u_10m", "wind_v_10m"})


def _resolve_variables(
    db: Session,
    source: _CycleMetadata | xr.Dataset,
    variables: list[str] | None,
) -> list[str]:
    """Resolve the requested variable codes.

    When ``variables`` is ``None`` the default set is the documented
    ``forecast_variables`` catalog intersected with the variables present in
    the dataset/store, excluding internal platform dependency variables (e.g.
    raw vector components). This explicit allowlist ensures auxiliary or
    non-surface dataset variables are never accidentally exposed or
    interpolated (API.md does not define a default variable list; the catalog
    is the platform's documented forecast-variable vocabulary). Provided codes
    are validated against the ``forecast_variables`` catalog.
    """
    if variables is None:
        catalog = set(_catalog_variable_codes(db) - INTERNAL_VARIABLES)
        present = (
            set(source.var_names)
            if isinstance(source, _CycleMetadata)
            else {str(name) for name in source.data_vars}
        )
        if "wind_u_10m" in present and "wind_v_10m" in present:
            present.add("wind_10m")
            catalog.add("wind_10m")
        return sorted(catalog.intersection(present))
    missing = _missing_catalog_variables(db, variables)
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown variable(s): {', '.join(sorted(missing))}.",
        )
    return list(variables)


def _catalog_variable_codes(db: Session) -> set[str]:
    """Return the set of documented forecast variable codes."""
    stmt = select(ForecastVariable.variable_code)
    return set(db.execute(stmt).scalars().all())


def _missing_catalog_variables(db: Session, variables: list[str]) -> list[str]:
    stmt = select(ForecastVariable.variable_code).where(
        ForecastVariable.variable_code.in_(variables)
    )
    known = set(db.execute(stmt).scalars().all())
    if "wind_10m" in variables:
        u_v_count = db.execute(
            select(ForecastVariable.variable_code).where(
                ForecastVariable.variable_code.in_(["wind_u_10m", "wind_v_10m", "wind_10m"])
            )
        ).scalars().all()
        if len(u_v_count) >= 2 or "wind_10m" in u_v_count:
            known.add("wind_10m")
    return [code for code in variables if code not in known]


def _variable_units(db: Session, var_codes: list[str]) -> dict[str, str | None]:
    if not var_codes:
        return {}
    stmt = select(ForecastVariable.variable_code, ForecastVariable.unit).where(
        ForecastVariable.variable_code.in_(var_codes)
    )
    units: dict[str, str | None] = {}
    for code, unit in db.execute(stmt).all():
        units[code] = unit
    if "wind_10m" in var_codes:
        units["wind_10m"] = "km/h"
    return {code: units.get(code) for code in var_codes}


def _interpolate_neighborhood(
    field: xr.DataArray,
    grid: RegularGrid,
    lat_descending: bool,
    lon_descending: bool,
    latitude: float,
    longitude: float,
) -> float:
    """Bilinearly interpolate ``field`` at a point using only the 2x2 window.

    The fractional row/col is computed from the derived ascending ``grid``
    (``row_col_from_coordinates``), then the four surrounding stored indices
    are located (mapping the ascending row/col back into the stored axis
    orientation when the axis is descending; the window arrives in ascending
    row/col ORDER because list indexers preserve order). Only that 2x2 window
    is read via ``.isel(...).values`` and interpolated with the same formulas as
    :func:`domain.geo.interpolation.bilinear_interpolate` (which would
    otherwise materialize the full 2-D grid).

    Raises:
        PointOutsideGridError: If the point lies outside the grid.
        InvalidGridError: If the grid cannot support interpolation.
    """
    if grid.rows < 2 or grid.cols < 2:
        raise InvalidGridError(
            "Bilinear interpolation requires at least two rows and two columns; "
            f"grid is {grid.rows} x {grid.cols}."
        )
    row_f, col_f = grid.row_col_from_coordinates(latitude, longitude)
    row_0 = math.floor(row_f)
    col_0 = math.floor(col_f)
    if row_0 == grid.rows - 1:
        row_0 = grid.rows - 2
    if col_0 == grid.cols - 1:
        col_0 = grid.cols - 2
    row_1, col_1 = row_0 + 1, col_0 + 1
    t_row = row_f - row_0
    t_col = col_f - col_0

    # Map ascending grid rows/cols to STORED indices (reverse when stored
    # descending so the stored window reads the correct rows/columns). xarray
    # ``isel`` with a LIST indexer returns elements in LIST order, so the
    # window arrives ordered ``[row_0, row_1]`` / ``[col_0, col_1]`` in the
    # ascending-grid sense regardless of stored-axis direction -- NO further
    # in-memory reversal is needed (reversing here would swap the corners and
    # bias the interpolation by up to ~(1 - 2*t) * cell gradient).
    def _stored(value: int, size: int, descending: bool) -> int:
        return (size - 1 - value) if descending else value

    lat_size = int(field.sizes["latitude"])
    lon_size = int(field.sizes["longitude"])
    lat_idx = ([_stored(row_0, lat_size, lat_descending),
                _stored(row_1, lat_size, lat_descending)])
    lon_idx = ([_stored(col_0, lon_size, lon_descending),
                _stored(col_1, lon_size, lon_descending)])

    # Phase 1: crop to the 2x2 neighborhood FIRST, then reduce the member
    # axis. For a GEFS (member, lat, lon) field this reads only the tiny
    # window's chunk(s) per member instead of the full global grid — matching
    # the tile path's crop-before-mean ordering (numerically identical).
    window = field.isel(latitude=lat_idx, longitude=lon_idx)
    if "member" in window.dims:
        window = window.mean(dim="member", keep_attrs=True)
    values = np.asarray(window.values, dtype=float)
    if values.ndim == 0:
        values = np.full((1, 1), float(values))
    # Corner layout (ascending-grid sense): values[0, 0] = (row_0, col_0),
    # values[0, 1] = (row_0, col_1), values[1, 0] = (row_1, col_0),
    # values[1, 1] = (row_1, col_1).

    value_00 = values[0, 0]
    value_01 = values[0, 1]
    value_10 = values[1, 0]
    value_11 = values[1, 1]

    lower = value_00 + (value_01 - value_00) * t_col
    upper = value_10 + (value_11 - value_10) * t_col
    return float(lower + (upper - lower) * t_row)


def _derive_grid(
    dataset: xr.Dataset,
) -> tuple[RegularGrid, bool, bool]:
    """Derive a regular grid from the dataset's coordinate arrays.

    Returns the grid along with flags indicating whether the latitude and
    longitude axes were stored in descending order and therefore must be
    reversed to align with the domain's ascending-row/column convention.
    Longitudes are normalized into the WGS84 ``[-180, 180]`` range where the
    stored axis is a fully-western ``0..360`` axis (see
    :func:`_normalize_grid_longitudes`).
    """
    lat_raw = _axis_values(dataset, "latitude")
    lon_raw = _axis_values(dataset, "longitude")
    latitudes, lat_descending = _ascending(lat_raw)
    longitudes, lon_descending = _ascending(lon_raw)
    longitudes = _normalize_grid_longitudes(longitudes)
    if len(latitudes) < 2 or len(longitudes) < 2:
        raise HTTPException(
            status_code=500,
            detail="The forecast dataset grid must have at least two points per axis.",
        )
    lat_step = (latitudes[-1] - latitudes[0]) / (len(latitudes) - 1)
    lon_step = (longitudes[-1] - longitudes[0]) / (len(longitudes) - 1)
    if lat_step <= 0.0 or lon_step <= 0.0:
        raise HTTPException(
            status_code=500,
            detail="The forecast dataset grid must be uniformly spaced.",
        )
    # The step is derived from the endpoints only; a genuinely non-uniform
    # axis (e.g. a Gaussian latitude grid) would otherwise silently
    # interpolate against a false uniform step. Verify the interior spacing
    # matches before building the grid.
    if not np.allclose(np.diff(latitudes), lat_step) or not np.allclose(
        np.diff(longitudes), lon_step
    ):
        raise HTTPException(
            status_code=500,
            detail="The forecast dataset grid must be uniformly spaced.",
        )
    grid = RegularGrid(
        lat_start=latitudes[0],
        lon_start=longitudes[0],
        lat_step=lat_step,
        lon_step=lon_step,
        rows=len(latitudes),
        cols=len(longitudes),
    )
    return grid, lat_descending, lon_descending


def _normalize_grid_longitudes(longitudes: list[float]) -> list[float]:
    """Map a fully-western 0-360 longitude axis into the WGS84 [-180, 180] range.

    GRIB decoding (``cfgrib``) always exposes a valid grid's longitudes in the
    native ``[0, 360]`` convention regardless of how the file stores them. A
    grid confined to the western hemisphere (e.g. a small GFS subset covering
    ``lon 250..259``) therefore arrives as a ``0..360`` axis whose origin
    exceeds 180. ``RegularGrid`` validates longitude against ``[-180, 180]``,
    so such an axis cannot be represented directly. Subtracting 360 from every
    coordinate maps the axis into ``[-180, 180]`` without changing the grid
    geometry (a uniform axis stays uniform; ordering and spacing are exact).

    Only a *fully* western axis (every longitude greater than 180) is shifted.
    A global axis that spans the antimeridian (e.g. ``0..340``) is left
    unchanged so ``RegularGrid.align_longitude`` can map western-hemisphere
    query longitudes into the ``0..360`` store as documented in API.md section
    2.1. An axis already in ``[-180, 180]`` is returned unchanged.

    Args:
        longitudes: The grid's longitude axis in ascending order.

    Returns:
        The axis mapped into the WGS84 ``[-180, 180]`` convention where
        possible, preserving order and spacing.
    """
    if longitudes and all(value > 180.0 for value in longitudes):
        return [value - 360.0 for value in longitudes]
    return longitudes


def _axis_values(dataset: xr.Dataset, name: str) -> list[float]:
    if name not in dataset.coords:
        raise HTTPException(
            status_code=500,
            detail=f"The forecast dataset has no '{name}' coordinate.",
        )
    return [float(value) for value in dataset.coords[name].values]


def _ascending(values: list[float]) -> tuple[list[float], bool]:
    """Return an ascending copy of an axis and whether it was reversed."""
    if values[-1] < values[0]:
        return list(reversed(values)), True
    return list(values), False


def _convert_value(
    value: float,
    si_unit: str | None,
    units: str,
    var_code: str | None = None,
) -> float:
    """Convert a value to imperial units when requested and supported.

    Conversion is applied only when ``units=imperial`` and either a
    variable-specific conversion is defined or the registered unit matches a
    known SI/imperial pair; otherwise the value is returned unconverted.
    """
    if units != "imperial":
        return value
    if var_code is not None and var_code in _VARIABLE_IMPERIAL_CONVERSIONS:
        return float(_VARIABLE_IMPERIAL_CONVERSIONS[var_code][1](value))
    if si_unit is None:
        return value
    conversion = _SI_TO_IMPERIAL.get(si_unit)
    if conversion is None:
        return value
    return float(conversion[1](value))
