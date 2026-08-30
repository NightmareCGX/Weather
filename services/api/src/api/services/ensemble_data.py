"""Ensemble data construction: per-member Zarr slicing and domain math.

This service backs the ``/v1/probabilities`` and ``/v1/ensembles`` endpoints
(API.md sections 3.1 and 5.1). It reuses the pure, stable helpers from
``api.services.point_forecast`` (run resolution, lead-time resolution,
variable resolution, grid derivation) and adds per-member interpolation so the
Milestone 7 domain math in ``domain.ensemble`` can operate on a flat 1-D array
of member values at a point.

The ``member`` axis is read from the run's Zarr dataset; ``member_count`` is
the length of that coordinate, which drives every calculation. The
``ensemble_members`` catalog rows are metadata only and are not consulted
here.
"""

from typing import Any, Literal

import xarray as xr
from domain.ensemble import (
    ensemble_mean,
    ensemble_median,
    ensemble_percentile,
    ensemble_spread,
    estimate_ensemble_pdf,
    probability_above_threshold,
    probability_at_or_above_threshold,
    probability_at_or_below_threshold,
    probability_below_threshold,
    probability_between_thresholds,
    probability_confidence_interval,
)
from domain.exceptions import (
    EmptyEnsembleError,
    InvalidCoordinatesError,
    InvalidEnsembleError,
    InvalidGridError,
    InvalidPercentileError,
    InvalidThresholdError,
    PointOutsideGridError,
)
from domain.geo.coordinates import validate_coordinates
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models.entities import Model
from api.schemas import (
    EnsemblePDF,
    EnsembleStatistics,
    EnsembleStatisticsData,
    ProbabilityForecastData,
    ProbabilityLocation,
)
from api.services.point_forecast import (
    _derive_grid,
    _interpolate_neighborhood,
    _resolve_lead_times,
    _resolve_ready_dataset,
    _resolve_variables,
)

#: Error status code for invalid probability/ensemble inputs.
_STATUS_INVALID_INPUT = 422


def _validate_coordinates(latitude: float, longitude: float) -> None:
    """Validate WGS 84 coordinates, mapping the domain error to 422.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.

    Raises:
        HTTPException: 422 when the coordinates fall outside the valid WGS 84
            bounds.
    """
    try:
        validate_coordinates(latitude, longitude)
    except InvalidCoordinatesError as exc:
        raise HTTPException(status_code=_STATUS_INVALID_INPUT, detail=str(exc)) from exc


def build_probability_forecast(
    db: Session,
    *,
    latitude: float,
    longitude: float,
    variable: str,
    threshold: float,
    operator: Literal["gt", "gte", "lt", "lte", "between"],
    lead_time_hours: int,
    model: str,
    threshold_max: float | None = None,
    initial_time: str | None = None,
) -> ProbabilityForecastData:
    """Build an exceedance probability forecast for a resolved point.

    The newest ``status='ready'`` run with a non-null ``zarr_store_path`` is
    selected for the (ensemble) model. Member values are bilinearly
    interpolated to the location at the requested lead time, and the empirical
    exceedance probability plus its Wilson 95% confidence interval are
    computed by ``domain.ensemble``.

    Args:
        db: Database session.
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        variable: A documented ``forecast_variables`` catalog code.
        threshold: The probability threshold (lower bound for ``between``).
        operator: ``gt``, ``lt``, or ``between``.
        lead_time_hours: Forecast offset hours from the run's cycle time.
        model: A single ensemble model identifier.
        threshold_max: The upper bound of ``between``; required only when
            ``operator == "between"``.

    Returns:
        The probability forecast payload.

    Raises:
        HTTPException: 422 for invalid input (domain math failures), 404 when
            no ready run, unknown model/variable, absent lead time, or the
            location is outside the grid, 500 for invalid grid data.
    """
    _validate_coordinates(latitude, longitude)
    _require_ensemble_model(db, model)
    run, metadata = _resolve_ready_dataset(db, model, initial_time=initial_time)
    leads = _resolve_lead_times(metadata, lead_time_hours, lead_time_hours)
    _resolve_variables(db, metadata, [variable])
    assert run.zarr_store_path is not None

    members = _gated_member_values(
        str(run.zarr_store_path), variable, leads[0], latitude, longitude
    )
    probability = _probability(members, threshold, operator, threshold_max)
    lower, upper = probability_confidence_interval(probability, len(members))
    data: dict[str, Any] = {
        "location": ProbabilityLocation(latitude=latitude, longitude=longitude),
        "variable": variable,
        "threshold": threshold,
        "operator": operator,
        "lead_time_hours": lead_time_hours,
        "probability": probability,
        "confidence_interval_95": [lower, upper],
    }
    if operator == "between":
        data["threshold_max"] = threshold_max
    return ProbabilityForecastData(**data)


def build_ensemble_statistics(
    db: Session,
    *,
    latitude: float,
    longitude: float,
    variable: str,
    model: str,
    lead_time_hours: int,
    include_members: bool = False,
    initial_time: str | None = None,
) -> EnsembleStatisticsData:
    """Build ensemble statistics for a resolved point.

    The newest ``status='ready'`` run with a non-null ``zarr_store_path`` is
    selected for the (ensemble) model. Member values are bilinearly
    interpolated to the location at the requested lead time, and the mean,
    median, population spread, and P10/P25/P50/P75/P90 percentiles are
    computed by ``domain.ensemble``.

    When ``include_members`` is true, the genuine raw member values (the exact
    same array the statistics are computed from, in dataset ``member``-
    coordinate order) are attached to the payload. This is an opt-in,
    additive extension for the Ensemble Distribution View: statistics-only
    requests (the default) stay lightweight and omit the member array.

    Args:
        db: Database session.
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        variable: A documented ``forecast_variables`` catalog code.
        model: A single ensemble model identifier.
        lead_time_hours: Forecast offset hours from the run's cycle time.
        include_members: Whether to attach the raw member values used to
            compute the statistics.

    Returns:
        The ensemble statistics payload.

    Raises:
        HTTPException: 422 for invalid input (domain math failures), 404 when
            no ready run, unknown model/variable, absent lead time, or the
            location is outside the grid, 500 for invalid grid data.
    """
    _validate_coordinates(latitude, longitude)
    _require_ensemble_model(db, model)
    run, metadata = _resolve_ready_dataset(db, model, initial_time=initial_time)
    leads = _resolve_lead_times(metadata, lead_time_hours, lead_time_hours)
    _resolve_variables(db, metadata, [variable])
    assert run.zarr_store_path is not None

    members = _gated_member_values(
        str(run.zarr_store_path), variable, leads[0], latitude, longitude
    )
    pdf_payload: EnsemblePDF | None = None
    if include_members:
        domain_pdf = estimate_ensemble_pdf(members)
        if domain_pdf is not None:
            pdf_payload = EnsemblePDF(x=domain_pdf.x, density=domain_pdf.density)

    return EnsembleStatisticsData(
        model=model,
        lead_time_hours=lead_time_hours,
        member_count=len(members),
        statistics=EnsembleStatistics(
            mean=ensemble_mean(members),
            median=ensemble_median(members),
            spread=ensemble_spread(members),
            p10=ensemble_percentile(members, 10),
            p25=ensemble_percentile(members, 25),
            p50=ensemble_percentile(members, 50),
            p75=ensemble_percentile(members, 75),
            p90=ensemble_percentile(members, 90),
        ),
        members=members if include_members else None,
        pdf=pdf_payload if include_members else None,
    )


def _require_ensemble_model(db: Session, model: str) -> None:
    """Reject deterministic models that have no member axis."""
    model_row = (
        db.execute(select(Model).where(Model.model_id == model)).scalars().one_or_none()
    )
    if model_row is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' was not found.")
    if not model_row.is_ensemble:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{model}' is not an ensemble model; use /v1/points "
                "for deterministic forecasts."
            ),
        )


def _probability(
    members: list[float],
    threshold: float,
    operator: Literal["gt", "gte", "lt", "lte", "between"],
    threshold_max: float | None,
) -> float:
    """Compute the empirical exceedance probability for the operator.

    Domain exceptions are mapped to 422 here so the router stays thin.

    Raises:
        HTTPException: 422 when the domain math rejects the input (empty or
            invalid member array, invalid threshold, or ``upper < lower``).
    """
    try:
        if operator == "gt":
            return probability_above_threshold(members, threshold)
        if operator == "gte":
            return probability_at_or_above_threshold(members, threshold)
        if operator == "lt":
            return probability_below_threshold(members, threshold)
        if operator == "lte":
            return probability_at_or_below_threshold(members, threshold)
        if threshold_max is None:
            raise HTTPException(
                status_code=_STATUS_INVALID_INPUT,
                detail=("threshold_max is required when operator is 'between'."),
            )
        return probability_between_thresholds(members, threshold, threshold_max)
    except (
        EmptyEnsembleError,
        InvalidEnsembleError,
        InvalidThresholdError,
        InvalidPercentileError,
    ) as exc:
        raise HTTPException(status_code=_STATUS_INVALID_INPUT, detail=str(exc)) from exc


def _gated_member_values(
    store_path: str,
    var_code: str,
    lead: int,
    latitude: float,
    longitude: float,
) -> list[float]:
    """Interpolate each ensemble member's field at a point and lead time.

    Phase 1 remediation: a single SHARED gate session opens the lazy store and
    interpolates every member's 2x2 neighborhood around the point — reading
    only the tiny spatial window per member, never the full global ensemble
    field.

    Returns a flat 1-D list of member values in the dataset's ``member``
    coordinate order, suitable for the ``domain.ensemble`` functions.

    Raises:
        HTTPException: 404 when the variable is absent from the dataset or the
            location is outside the grid, 500 for invalid grid data or a
            non-2-D per-member field.
    """
    from api.core.reader_gate import gated_read_dataset_with_selector

    def select_and_interpolate(dataset: xr.Dataset) -> list[float]:
        if var_code not in dataset.data_vars:
            raise HTTPException(
                status_code=404,
                detail=f"Variable '{var_code}' is not available in the forecast dataset.",
            )
        field = dataset[var_code]
        if "member" not in field.dims:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Variable '{var_code}' has no ensemble member dimension in "
                    "the forecast dataset."
                ),
            )
        if "lead_time_hours" in field.dims:
            field = field.sel(lead_time_hours=lead)

        grid, lat_descending, lon_descending = _derive_grid(dataset)
        member_count = int(dataset.coords["member"].size)
        values: list[float] = []
        for member_index in range(member_count):
            member_field = field.isel(member=member_index)
            if member_field.ndim != 2:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Variable '{var_code}' is not a 2-D surface field; "
                        "vertical-level variables are not supported."
                    ),
                )
            try:
                values.append(
                    _interpolate_neighborhood(
                        member_field,
                        grid,
                        lat_descending,
                        lon_descending,
                        latitude,
                        longitude,
                    )
                )
            except PointOutsideGridError as exc:
                raise HTTPException(
                    status_code=404,
                    detail=f"No forecast data covers the requested location: {exc}",
                ) from exc
            except InvalidGridError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="The forecast dataset grid is invalid.",
                ) from exc
        return values

    return gated_read_dataset_with_selector(store_path, select_and_interpolate)
