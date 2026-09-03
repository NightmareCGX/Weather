"""Verification data construction: observation pairing and metric aggregation.

This service backs the ``/v1/verifications`` endpoint (API.md section 7.1). It
retrieves ``verification_observations`` rows for the requested model and date
window, pairs each observation with the requested model's forecast value at the
observation's valid time and station (drawn from the model's Zarr stores), and
aggregates RMSE/bias/MAE per variable via ``domain.verification``.

Pairing (a deterministic implementation decision approved for Milestone 11 and
documented in docs/API.md section 7.1): every forecast product of the model
whose ``cycle_time + lead_time_hours`` equals an observation's ``valid_time``
contributes one ``(observed, forecast)`` pair, and all pairs are pooled per
variable. No single cycle or lead time is selected over another.
"""

import logging
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import cast

import xarray as xr
from domain.exceptions import InvalidGridError, PointOutsideGridError
from domain.verification import bias, mean_absolute_error, root_mean_squared_error
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.models.entities import (
    ForecastProduct,
    Model,
    ModelRun,
    ModelVersion,
    Station,
    VerificationObservation,
)
from api.schemas import VerificationPeriod, VerificationReportData
from api.services.lifecycle import filter_visible_runs
from api.services.point_forecast import (
    _derive_grid,
    _interpolate_neighborhood,
)

logger = logging.getLogger(__name__)


def build_verification_report(
    db: Session,
    *,
    model: str,
    start_date: date,
    end_date: date,
) -> VerificationReportData:
    """Build a verification report for a model and an inclusive date window.

    Observations with ``valid_time`` in ``[start_date, end_date]`` (UTC,
    inclusive) are paired with the model's forecast values. For each
    observation, every ready forecast product of the model whose
    ``cycle_time + lead_time_hours`` equals the observation's ``valid_time``
    contributes one ``(observed, forecast)`` pair. RMSE, bias, and MAE are then
    computed per variable over the pooled sample of all eligible pairs.

    A variable with no eligible pairs contributes no metrics keys, so the
    report's ``metrics`` may be empty. This empty-result behavior is a
    deterministic implementation decision approved for Milestone 11 (docs/API.md
    section 7.1).

    Args:
        db: Database session.
        model: A model identifier.
        start_date: Inclusive start date of the verification window.
        end_date: Inclusive end date of the verification window.

    Returns:
        The verification report payload.

    Raises:
        HTTPException: 400 when ``start_date`` is after ``end_date``; 404 when
            the model is unknown.
    """
    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must not be after end_date.",
        )
    _require_model(db, model)

    window_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    window_end = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    observations = _observations_in_window(db, window_start, window_end)

    runs = _ready_runs(db, model)
    products = _products_for_runs(db, runs)
    # ORM attributes are typed as ``Column``; cast to their Python types when
    # used as plain values (see the ``cast`` convention in point_forecast.py).
    runs_by_id = {cast(str, run.id): run for run in runs}

    station_coords: dict[str, tuple[float, float] | None] = {}
    pairs_by_variable: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for observation in observations:
        station_code = cast(str, observation.station_id)
        variable_code = cast(str, observation.variable_code)
        observed_value = cast(float, observation.observed_value)
        valid_time = cast(datetime, observation.valid_time)

        if station_code not in station_coords:
            station_coords[station_code] = _station_coordinates(db, station_code)
        coords = station_coords[station_code]
        if coords is None:
            continue
        latitude, longitude = coords

        for product in products:
            product_variable = cast(str, product.variable_id)
            product_lead = cast(int, product.lead_time_hours)
            product_run_id = cast(str, product.run_id)
            if product_variable != variable_code:
                continue
            run = runs_by_id.get(product_run_id)
            if run is None:
                continue
            run_cycle_time = cast(datetime, run.cycle_time)
            if run_cycle_time + timedelta(hours=product_lead) != valid_time:
                continue
            assert run.zarr_store_path is not None
            # Phase 1 remediation: interpolate the single observation point via
            # a bounded gated read (only the 2x2 neighborhood is materialized,
            # never the full grid), under the SHARED reader gate.
            forecast_value = _gated_interpolate_candidate(
                str(run.zarr_store_path),
                product_variable,
                product_lead,
                latitude,
                longitude,
            )
            if forecast_value is None:
                continue
            pairs_by_variable[variable_code].append((observed_value, forecast_value))

    metrics: dict[str, float] = {}
    for variable_code, pairs in pairs_by_variable.items():
        if not pairs:
            continue
        observed_values = [pair[0] for pair in pairs]
        forecast_values = [pair[1] for pair in pairs]
        metrics[f"{variable_code}_rmse"] = root_mean_squared_error(
            observed_values, forecast_values
        )
        metrics[f"{variable_code}_bias"] = bias(observed_values, forecast_values)
        metrics[f"{variable_code}_mae"] = mean_absolute_error(
            observed_values, forecast_values
        )

    return VerificationReportData(
        model=model,
        period=VerificationPeriod(start=start_date, end=end_date),
        metrics=metrics,
    )


def _require_model(db: Session, model: str) -> None:
    """Reject an unknown model identifier with a 404.

    Args:
        db: Database session.
        model: A model identifier.

    Raises:
        HTTPException: 404 when the model does not exist.
    """
    row = (
        db.execute(select(Model.id).where(Model.model_id == model))
        .scalars()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' was not found.")


def _observations_in_window(
    db: Session, window_start: datetime, window_end: datetime
) -> list[VerificationObservation]:
    """Return observations whose valid time falls in the half-open window.

    Args:
        db: Database session.
        window_start: Inclusive lower bound (UTC).
        window_end: Exclusive upper bound (UTC).

    Returns:
        The matching observations ordered by valid time.
    """
    stmt = (
        select(VerificationObservation)
        .where(VerificationObservation.valid_time >= window_start)
        .where(VerificationObservation.valid_time < window_end)
        .order_by(VerificationObservation.valid_time.asc())
    )
    return list(db.execute(stmt).scalars().all())


def _ready_runs(db: Session, model: str) -> list[ModelRun]:
    """Return the model's ready runs with a non-null store path.

    Runs are ordered newest cycle first. The store itself is opened lazily per
    run by :func:`_dataset_for_run`, so an unreadable store only skips that
    run's candidates rather than failing the request.

    Args:
        db: Database session.
        model: A model identifier.

    Returns:
        The eligible runs.
    """
    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status == "ready")
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    stmt = filter_visible_runs(stmt).order_by(ModelRun.cycle_time.desc())
    return list(db.execute(stmt).scalars().all())


def _products_for_runs(
    db: Session, runs: list[ModelRun]
) -> list[ForecastProduct]:
    """Return every forecast product of the given runs.

    Args:
        db: Database session.
        runs: The ready runs whose products are loaded.

    Returns:
        The forecast products ordered by lead time.
    """
    run_ids = [run.id for run in runs]
    if not run_ids:
        return []
    stmt = (
        select(ForecastProduct)
        .where(ForecastProduct.run_id.in_(run_ids))
        .order_by(ForecastProduct.lead_time_hours.asc())
    )
    return list(db.execute(stmt).scalars().all())


def _station_coordinates(
    db: Session, station_code: str
) -> tuple[float, float] | None:
    """Return a station's ``(latitude, longitude)`` from its PostGIS point.

    Args:
        db: Database session.
        station_code: A ``stations.station_code`` value.

    Returns:
        The ``(latitude, longitude)`` pair, or ``None`` when the station does
        not exist (the observation is then skipped).
    """
    row = db.execute(
        select(func.ST_X(Station.geom), func.ST_Y(Station.geom)).where(
            Station.station_code == station_code
        )
    ).one_or_none()
    if row is None:
        logger.warning(
            "Verification station '%s' was not found; skipping its observations.",
            station_code,
        )
        return None
    return (float(row[1]), float(row[0]))


def _gated_interpolate_candidate(
    store_path: str,
    var_code: str,
    lead: int,
    latitude: float,
    longitude: float,
) -> float | None:
    """Interpolate a forecast field at a station via a bounded gated read.

    A single SHARED gate session opens the lazy store, selects the variable +
    lead, crops the 2x2 neighborhood around the station, and materializes only
    that tiny window. A candidate is skipped (returning ``None``) when the
    variable is absent from the dataset, the requested lead time is absent, the
    field is not a 2-D surface field, the station falls outside the grid, or
    the store is unreadable. Invalid grid data propagates as a server error
    (matching the point-forecast behavior).

    Args:
        store_path: The run's Zarr store path.
        var_code: A forecast variable code.
        lead: The lead time hours of the candidate product.
        latitude: Station latitude.
        longitude: Station longitude.

    Returns:
        The interpolated forecast value, or ``None`` when the candidate is
        skipped.
    """
    from api.core.reader_gate import gated_read_dataset_with_selector

    def select_and_interpolate(dataset: xr.Dataset) -> float:
        if var_code not in dataset.data_vars:
            raise _SkipCandidate()
        field = dataset[var_code]
        if "lead_time_hours" in field.dims:
            try:
                field = field.sel(lead_time_hours=lead)
            except KeyError:
                raise _SkipCandidate() from None
        if "member" in field.dims:
            field = field.mean(dim="member", keep_attrs=True)
        if field.ndim != 2:
            raise _SkipCandidate()
        grid, lat_descending, lon_descending = _derive_grid(dataset)
        return _interpolate_neighborhood(
            field, grid, lat_descending, lon_descending, latitude, longitude
        )

    try:
        return gated_read_dataset_with_selector(store_path, select_and_interpolate)
    except _SkipCandidate:
        return None
    except (PointOutsideGridError, InvalidGridError) as exc:
        logger.warning(
            "Skipping verification candidate %s at lead %s for (%s, %s): %s",
            var_code,
            lead,
            latitude,
            longitude,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - unreadable store
        logger.warning(
            "Skipping unreadable verification store %s: %s", store_path, exc
        )
        return None


class _SkipCandidate(Exception):
    """Internal signal: a candidate is not servable (skip, not error)."""
