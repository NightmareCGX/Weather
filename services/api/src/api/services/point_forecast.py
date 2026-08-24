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
from domain.geo.interpolation import bilinear_interpolate
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
    # km/h → mph. No km/h variable is currently implemented, but the entry must
    # label the converted value as mph (the conversion factor is applied).
    "km/h": ("mph", lambda kmh: kmh * 0.621371),
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

    # For each valid_time, try the candidate cycles in ascending-lead order and
    # use the first whose store is readable AND actually contains the requested
    # lead. A broken newest store, or a store whose lead coordinate lacks the
    # selected lead (stale forecast_products metadata after a same-cycle
    # re-ingest), therefore falls back to the next READY candidate covering the
    # same valid_time instead of dropping the record or raising a KeyError.
    by_cycle: dict[datetime, xr.Dataset] = {}

    def _open_cycle(cycle_time: datetime) -> xr.Dataset | None:
        if cycle_time in by_cycle:
            return by_cycle[cycle_time]
        try:
            dataset = _open_run_store(db, model, cycle_time)
        except HTTPException:
            return None
        by_cycle[cycle_time] = dataset
        return dataset

    def _store_has_lead(dataset: xr.Dataset, lead: int) -> bool:
        """Return whether the opened store actually carries the requested lead.

        ``forecast_products`` is the fast metadata-first candidate-discovery
        path, but a catalog candidate is only usable when the current Zarr store
        contains the lead (a same-cycle re-ingest with a different lead set can
        leave stale product rows pointing at leads the store no longer holds).
        The check reads only the coordinate, never the full gridded variable.
        """
        if "lead_time_hours" not in dataset.coords:
            return False
        coord = dataset.coords["lead_time_hours"].values
        if np.ndim(coord) == 0:
            return int(coord) == lead
        return lead in {int(v) for v in coord}

    resolved: dict[datetime, tuple[datetime, int]] = {}
    for valid_time, pairs in candidates.items():
        for cycle_time, lead in pairs:
            dataset = _open_cycle(cycle_time)
            if dataset is None or not _store_has_lead(dataset, lead):
                continue
            resolved[valid_time] = (cycle_time, lead)
            break

    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"No readable forecast data was found for model '{model}'.",
        )

    # Resolve variables from the union of the opened winning stores.
    var_codes = _resolve_variables(db, _merge_var_sets(by_cycle.values()), variables)
    units_by_code = _variable_units(db, var_codes)

    forecasts: list[ForecastSeries] = []
    for valid_time, (cycle_time, lead) in sorted(resolved.items()):
        dataset = by_cycle[cycle_time]
        entry: dict[str, Any] = {
            "lead_time_hours": lead,
            "valid_time": valid_time,
            # Cross-cycle provenance: the cycle that produced this entry.
            "cycle_time": cycle_time,
        }
        for var_code in var_codes:
            value = _interpolate_variable(
                dataset, var_code, lead, location.latitude, location.longitude
            )
            entry[var_code] = _convert_value(value, units_by_code[var_code], units)
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


def _merge_var_sets(datasets: Iterable[xr.Dataset]) -> xr.Dataset:
    """Return a minimal dataset whose data-variable names union the inputs.

    Used to resolve the default variable set across multiple winning runs of a
    cross-cycle series without loading any gridded data.

    Args:
        datasets: The opened winning run datasets.

    Returns:
        A dataset exposing the union of the input data-variable names.
    """
    names: set[str] = set()
    for dataset in datasets:
        names.update(str(name) for name in dataset.data_vars)
    return xr.Dataset(
        {name: (("latitude", "longitude"), np.empty((0, 0))) for name in sorted(names)}
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
        .where(ModelRun.status == "ready")
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
        .where(ModelRun.status == "ready")
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
            # Candidate discovery reads the store's lead axis; use the reader
            # gate so a store mid-re-ingest is never opened.
            from api.core.reader_gate import gated_read_dataset

            dataset = gated_read_dataset(str(run.zarr_store_path))
        except Exception as exc:  # noqa: BLE001 - unreadable store
            logger.warning(
                "Skipping unreadable Zarr store for run %s: %s", run.id, exc
            )
            continue
        if "lead_time_hours" not in dataset.coords:
            continue
        coord = dataset.coords["lead_time_hours"].values
        if np.ndim(coord) == 0:
            leads = [int(coord)]
        else:
            leads = [int(v) for v in coord]
        for lead in leads:
            _add(cycle_time, lead)

    # Order each valid_time's candidates by ascending lead (best first).
    return {
        valid_time: sorted(pairs, key=lambda pair: pair[1])
        for valid_time, pairs in candidates_by_valid.items()
    }


def _gated_read_dataset(store_path: str) -> xr.Dataset:
    """Read a forecast Zarr store under the SHARED reader gate.

    The reader participates in the same PostgreSQL store gate as the ingestion
    writer, so it never observes a store mid-re-ingest. The dataset is fully
    materialized (xarray open + load) before the gate is released.

    Falls back to a direct read when the reader-gate pool is not initialized
    (test/development path).

    Raises:
        FileNotFoundError: If the run is no longer READY (revalidation fails).
    """
    from api.core.reader_gate import gated_read_dataset

    result = gated_read_dataset(store_path)
    if not isinstance(result, xr.Dataset):
        raise FileNotFoundError(f"run {store_path!r} did not materialize a dataset")
    return result


def _open_run_store(db: Session, model: str, cycle_time: datetime) -> xr.Dataset:
    """Open the Zarr store of the given READY run, or raise 404.

    Args:
        db: Database session.
        model: A single deterministic model identifier.
        cycle_time: The run's cycle time.

    Returns:
        The opened dataset.

    Raises:
        HTTPException: 404 if the run does not exist or its store is unreadable.
    """
    run = (
        db.execute(
            select(ModelRun)
            .join(ModelRun.model_version)
            .join(ModelVersion.model)
            .where(Model.model_id == model)
            .where(ModelRun.status == "ready")
            .where(ModelRun.zarr_store_path.isnot(None))
            .where(ModelRun.cycle_time == cycle_time)
        )
        .scalars()
        .one_or_none()
    )
    if run is None or run.zarr_store_path is None:
        raise HTTPException(status_code=404, detail=f"Run for cycle {cycle_time} not found.")
    try:
        # Reader-gate: participate in the SHARED store gate + fresh Core
        # revalidation so the run is still READY and its store is stable while
        # this request materializes the dataset.
        return _gated_read_dataset(str(run.zarr_store_path))
    except FileNotFoundError as exc:
        logger.warning("Skipping no-longer-ready Zarr store for run %s", run.id)
        raise HTTPException(
            status_code=404,
            detail=f"Run store for cycle {cycle_time} is no longer ready.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - a broken store is "unavailable"
        logger.warning("Skipping unreadable Zarr store for run %s: %s", run.id, exc)
        raise HTTPException(
            status_code=404, detail=f"Run store for cycle {cycle_time} is unreadable."
        ) from exc


def resolve_latest_run_serving_generation(
    db: Session, model: str, initial_time: str | None = None
) -> str | None:
    """Return the committed-manifest generation for the latest ready run.

    The generation is the cache-generation discriminator: a same-set same-cycle
    data replacement changes the manifest generation, making old cache entries
    unreachable. Falls back to the legacy deterministic token when no manifest
    exists.

    Returns:
        The serving generation string, or ``None`` when no ready run exists or
        the store path cannot be resolved.
    """
    from api.core.manifest_reader import ManifestReadError, manifest_generation

    stmt = (
        select(ModelRun.zarr_store_path)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status == "ready")
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
    """Return the resolved ready run's cycle time for a model, or ``None``.

    The cache key for a point/probability/ensemble request must include the
    resolved forecast run's cycle so a cached response for one cycle never
    satisfies a request for another (ACCEPTANCE_REMEDIATION_PLAN §9). This is a
    lightweight DB lookup (the newest ``ready`` run with a store, optionally
    pinned to a specific cycle via ``initial_time``); the heavy store open
    happens in the compute path.

    Args:
        db: Database session.
        model: A single model identifier.
        initial_time: Optional ISO 8601 UTC cycle time pinning the run. When
            provided, the run at exactly that cycle is resolved (GAP-2).

    Returns:
        The run's ``cycle_time`` as an ISO 8601 UTC string, or ``None`` when no
        matching ready run exists.
    """
    stmt = (
        select(ModelRun.cycle_time)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status == "ready")
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
) -> tuple[ModelRun, xr.Dataset]:
    """Return the newest ready run for a model whose Zarr store opens.

    Ready runs are ordered newest-first; each candidate's store is opened in
    turn and the first one that reads successfully is returned with its
    dataset. A corrupted, truncated, or momentarily-unreachable store on the
    newest run therefore falls through to the next-newest readable run instead
    of failing the request. The store is re-probed per request, so a broken
    run is skipped only for requests while it remains unreadable.

    Args:
        db: Database session.
        model_id: A single model identifier.
        initial_time: Optional ISO 8601 UTC cycle time pinning the run. When
            provided, only the run at that cycle is considered (GAP-2).

    Returns:
        A ``(run, dataset)`` pair for the first readable ready run.

    Raises:
        HTTPException: 404 when no ready run exists or none of the ready runs
            has a readable store.
    """
    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model_id)
        .where(ModelRun.status == "ready")
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
            # store mid-re-ingest is never read.
            from api.core.reader_gate import gated_read_dataset

            dataset = gated_read_dataset(str(run.zarr_store_path))
        except Exception as exc:  # noqa: BLE001 - probe store, fall through
            logger.warning(
                "Skipping unreadable Zarr store for run %s (%s): %s",
                run.id,
                run.zarr_store_path,
                exc,
            )
            continue
        return run, dataset
    raise HTTPException(
        status_code=404,
        detail=(
            f"No readable forecast run with data was found for model '{model_id}'."
        ),
    )


def _resolve_lead_times(
    dataset: xr.Dataset,
    start: int | None,
    end: int | None,
) -> list[int]:
    if "lead_time_hours" not in dataset.coords:
        raise HTTPException(
            status_code=404,
            detail="The forecast dataset has no lead_time_hours coordinate.",
        )
    coord = dataset.coords["lead_time_hours"].values
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


def _resolve_variables(
    db: Session,
    dataset: xr.Dataset,
    variables: list[str] | None,
) -> list[str]:
    """Resolve the requested variable codes.

    When ``variables`` is ``None`` the default set is the documented
    ``forecast_variables`` catalog intersected with the variables present in
    the dataset. This explicit allowlist ensures auxiliary or non-surface
    dataset variables are never accidentally exposed or interpolated (API.md
    does not define a default variable list; the catalog is the platform's
    documented forecast-variable vocabulary). Provided codes are validated
    against the ``forecast_variables`` catalog.
    """
    if variables is None:
        catalog = _catalog_variable_codes(db)
        return sorted(catalog.intersection(str(name) for name in dataset.data_vars))
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
    return {code: units.get(code) for code in var_codes}


def _reduce_surface_field(
    field: xr.DataArray,
    *,
    lead: int,
) -> xr.DataArray:
    """Return a surface field from a (possibly ensemble) dataset variable.

    GEFS stores carry a leading ``member`` dimension. A single-valued surface
    (a point forecast or a map tile) is the ensemble mean — the platform's
    documented ensemble aggregate (API.md 5.1 derives statistics from all
    members). This is the **shared** ensemble-reduction semantic used by the
    map tile renderer (:mod:`api.services.tiles`) and the point forecast, so
    both surfaces agree on the ensemble-mean field. Deterministic (GFS) fields
    have no ``member`` axis and are returned unchanged.

    Only the reduction is shared: the caller validates the resulting field is a
    plain 2-D surface and maps a non-surface field to its own error semantics
    (tiles render it as a 422, the point forecast as a 500 — the historical
    per-surface contract).

    Args:
        field: The dataset's data variable for the requested forecast variable.
        lead: The lead time to select.

    Returns:
        The ``(latitude, longitude)`` surface field for the lead, with the
        ``member`` axis (when present) reduced by the ensemble mean.
    """
    if "lead_time_hours" in field.dims:
        field = field.sel(lead_time_hours=lead)
    if "member" in field.dims:
        # Ensemble stores carry a leading ``member`` axis; the surface renders
        # the ensemble mean (matching the map tile and the ensemble statistics
        # mean).
        field = field.mean(dim="member", keep_attrs=True)
    return field


def _interpolate_variable(
    dataset: xr.Dataset,
    var_code: str,
    lead: int,
    latitude: float,
    longitude: float,
) -> float:
    if var_code not in dataset.data_vars:
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{var_code}' is not available in the forecast dataset.",
        )
    field = dataset[var_code]
    field = _reduce_surface_field(field, lead=lead)
    if field.ndim != 2:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Variable '{var_code}' is not a 2-D surface field; "
                "vertical-level variables are not supported."
            ),
        )
    grid, lat_descending, lon_descending = _derive_grid(dataset)
    values = _field_values(field, lat_descending, lon_descending)
    try:
        return float(bilinear_interpolate(grid, values, latitude, longitude))
    except PointOutsideGridError as exc:
        raise HTTPException(
            status_code=404,
            detail=(f"No forecast data covers the requested location: {exc}"),
        ) from exc
    except InvalidGridError as exc:
        raise HTTPException(
            status_code=500,
            detail="The forecast dataset grid is invalid.",
        ) from exc


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


def _field_values(
    field: xr.DataArray,
    lat_descending: bool,
    lon_descending: bool,
) -> list[list[float]]:
    if lat_descending:
        field = field.isel(latitude=slice(None, None, -1))
    if lon_descending:
        field = field.isel(longitude=slice(None, None, -1))
    # ``numpy.asarray(...).tolist()`` is typed ``Any`` (numpy's ``Any``
    # overloads), which would trip ``no-any-return`` under strict mode. The
    # runtime value is always a nested ``list[float]``, so narrow it through a
    # typed intermediate.
    values: list[list[float]] = np.asarray(field.values, dtype=float).tolist()
    return values


def _convert_value(value: float, si_unit: str | None, units: str) -> float:
    """Convert a value to imperial units when requested and supported.

    Conversion is applied only when ``units=imperial`` and the variable's
    registered unit matches a known SI/imperial pair; otherwise the value is
    returned unconverted.
    """
    if units != "imperial" or si_unit is None:
        return value
    conversion = _SI_TO_IMPERIAL.get(si_unit)
    if conversion is None:
        return value
    return float(conversion[1](value))
