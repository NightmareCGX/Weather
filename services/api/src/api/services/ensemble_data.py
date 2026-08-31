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

import math
from typing import Any, Literal

import numpy as np
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
from domain.models.cloud import (
    cloud_ceiling_ensemble_summary,
    cloud_cover_ensemble_summary,
    compute_low_ceiling_probability,
)
from domain.models.precipitation import (
    PhysicalPhase,
    PrecipitationPhaseState,
    aggregate_ensemble_phase_support,
    classify_precipitation_phase,
    compute_joint_amount_phase_support,
    compute_transition_frequencies,
)
from domain.models.wind import (
    compute_consensus_vector,
    compute_directional_probability,
    compute_wind_rose,
)
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models.entities import Model
from api.schemas import (
    ConsensusVectorOut,
    EnsemblePDF,
    EnsembleStatistics,
    EnsembleStatisticsData,
    ProbabilityForecastData,
    ProbabilityLocation,
    WindRoseOut,
    WindRoseSectorOut,
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
    direction_sector: str | None = None,
    phase: str | None = None,
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
        operator: ``gt``, ``gte``, ``lt``, ``lte``, or ``between``.
        lead_time_hours: Forecast offset hours from the run's cycle time.
        model: A single ensemble model identifier.
        threshold_max: The upper bound of ``between``; required only when
            ``operator == "between"``.
        direction_sector: Optional 8-point cardinal direction sector for wind
            probabilities (e.g. 'SW', 'N').
        phase: Optional physical precipitation phase (e.g. 'snow', 'rain',
            'freezing_rain', 'ice_pellets') for joint exceedance support.
        initial_time: Optional cycle pinning string.

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

    if variable == "wind_10m":
        u_members, v_members = _gated_wind_member_vectors(
            str(run.zarr_store_path), leads[0], latitude, longitude
        )
        if direction_sector is not None:
            # Convert speed threshold in km/h to m/s for canonical directional probability
            speed_mps = threshold / 3.6
            try:
                probability, (lower, upper) = compute_directional_probability(
                    u_members,
                    v_members,
                    sector=direction_sector,
                    speed_threshold=speed_mps,
                    operator=operator,
                )
            except InvalidThresholdError as exc:
                raise HTTPException(status_code=_STATUS_INVALID_INPUT, detail=str(exc)) from exc
        else:
            members_kmh = [math.hypot(u, v) * 3.6 for u, v in zip(u_members, v_members, strict=True)]
            probability = _probability(members_kmh, threshold, operator, threshold_max)
            lower, upper = probability_confidence_interval(probability, len(members_kmh))
    elif variable == "precipitation_amount_3h" and phase is not None:
        members, precip_states = _gated_precipitation_member_states(
            str(run.zarr_store_path), leads[0], latitude, longitude
        )
        try:
            target_phase = PhysicalPhase(phase.lower())
        except ValueError as exc:
            raise HTTPException(
                status_code=_STATUS_INVALID_INPUT,
                detail=f"Unknown physical phase {phase!r}. Valid phases: {[p.value for p in PhysicalPhase]}",
            ) from exc
        try:
            probability = compute_joint_amount_phase_support(
                precip_states, threshold_mm=threshold, phase=target_phase
            )
        except ValueError as exc:
            raise HTTPException(status_code=_STATUS_INVALID_INPUT, detail=str(exc)) from exc
        lower, upper = probability_confidence_interval(probability, len(precip_states))
    elif variable == "cloud_ceiling" and operator in ("lt", "lte"):
        members = _gated_member_values(
            str(run.zarr_store_path), variable, leads[0], latitude, longitude
        )
        prob = compute_low_ceiling_probability(members, threshold_m=threshold)
        if prob is None:
            raise HTTPException(
                status_code=_STATUS_INVALID_INPUT,
                detail="Insufficient valid members for cloud ceiling probability (requires >= 21).",
            )
        valid_n = sum(1 for m in members if not math.isnan(m) and float(m) >= 0.0)
        lower, upper = probability_confidence_interval(prob, valid_n)
        probability = prob
    elif variable == "cloud_cover_3h":
        members = _gated_member_values(
            str(run.zarr_store_path), variable, leads[0], latitude, longitude
        )
        valid_members = [float(m) for m in members if not math.isnan(m) and 0.0 <= float(m) <= 100.0]
        if len(valid_members) < 21:
            raise HTTPException(
                status_code=_STATUS_INVALID_INPUT,
                detail="Insufficient valid members for cloud cover probability (requires >= 21).",
            )
        probability = _probability(valid_members, threshold, operator, threshold_max)
        lower, upper = probability_confidence_interval(probability, len(valid_members))
    else:
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
    if direction_sector is not None:
        data["direction_sector"] = direction_sector
    if phase is not None:
        data["phase"] = phase
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
        initial_time: Optional cycle pinning string.

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

    consensus_payload: ConsensusVectorOut | None = None
    wind_rose_payload: WindRoseOut | None = None
    phase_support_payload: dict[str, float] | None = None
    transition_freq_payload: dict[str, float] | None = None
    valid_member_count_payload: int | None = None
    unlimited_prob_payload: float | None = None
    finite_member_count_payload: int | None = None
    unlimited_member_count_payload: int | None = None
    stats: EnsembleStatistics | None = None

    if variable == "wind_10m":
        u_members, v_members = _gated_wind_member_vectors(
            str(run.zarr_store_path), leads[0], latitude, longitude
        )
        members = [math.hypot(u, v) * 3.6 for u, v in zip(u_members, v_members, strict=True)]

        # Consensus flow vector in km/h
        consensus = compute_consensus_vector(u_members, v_members)
        consensus_payload = ConsensusVectorOut(
            speed=round(consensus.speed_mps * 3.6, 2),
            direction=round(consensus.direction_deg, 1) if consensus.direction_deg is not None else None,
            cardinal=consensus.cardinal,
            coherence=round(consensus.coherence, 4),
        )

        # 8-Sector Wind Rose
        rose = compute_wind_rose(u_members, v_members)
        wind_rose_payload = WindRoseOut(
            calm_percentage=round(rose.calm_probability * 100.0, 1),
            calm_count=rose.calm_count,
            sectors=[
                WindRoseSectorOut(
                    sector=s.sector,
                    count=s.count,
                    probability=round(s.probability, 4),
                    bins={k: round(v, 4) for k, v in s.bins.items()},
                )
                for s in rose.sectors
            ],
        )
    elif variable == "precipitation_amount_3h":
        amounts, precip_states = _gated_precipitation_member_states(
            str(run.zarr_store_path), leads[0], latitude, longitude
        )
        # At lead 0 (or all-NaN accumulation), compute statistics against 0.0 baseline
        if all(math.isnan(m) for m in amounts):
            members = [0.0] * len(amounts)
        else:
            members = amounts

        support_map = aggregate_ensemble_phase_support(precip_states)
        phase_support_payload = {p.value: round(v, 4) for p, v in support_map.items()}

        freq_map = compute_transition_frequencies(precip_states)
        transition_freq_payload = {
            t.value: round(v, 4) for t, v in freq_map.items() if v > 0.0
        }
    elif variable == "cloud_cover_3h":
        members = _gated_member_values(
            str(run.zarr_store_path), variable, leads[0], latitude, longitude
        )
        min_v = 21 if len(members) >= 21 else 1
        cc_summary = cloud_cover_ensemble_summary(members, min_valid=min_v)
        if cc_summary is not None:
            stats = EnsembleStatistics(
                mean=float(cc_summary.mean),
                median=float(cc_summary.median),
                spread=float(cc_summary.spread),
                p10=float(cc_summary.percentiles["p10"]),
                p25=float(cc_summary.percentiles["p25"]),
                p50=float(cc_summary.percentiles["p50"]),
                p75=float(cc_summary.percentiles["p75"]),
                p90=float(cc_summary.percentiles["p90"]),
            )
            valid_member_count_payload = cc_summary.valid_member_count
        else:
            stats = EnsembleStatistics(
                mean=None, median=None, spread=None,
                p10=None, p25=None, p50=None, p75=None, p90=None
            )
            valid_member_count_payload = sum(
                1 for m in members if not math.isnan(m) and 0.0 <= float(m) <= 100.0
            )
    elif variable == "cloud_ceiling":
        members = _gated_member_values(
            str(run.zarr_store_path), variable, leads[0], latitude, longitude
        )
        min_v = 21 if len(members) >= 21 else 1
        ceil_summary = cloud_ceiling_ensemble_summary(
            members,
            min_finite=10,
            min_valid=min_v,
        )
        if ceil_summary is not None:
            unlimited_prob_payload = round(ceil_summary.unlimited_probability, 4)
            valid_member_count_payload = ceil_summary.valid_member_count
            finite_member_count_payload = ceil_summary.finite_member_count
            unlimited_member_count_payload = ceil_summary.unlimited_member_count
            if ceil_summary.conditional_percentiles_m is not None:
                stats = EnsembleStatistics(
                    mean=float(ceil_summary.conditional_mean_m) if ceil_summary.conditional_mean_m is not None else None,
                    median=float(ceil_summary.conditional_median_m) if ceil_summary.conditional_median_m is not None else None,
                    spread=float(ceil_summary.conditional_spread_m) if ceil_summary.conditional_spread_m is not None else None,
                    p10=float(ceil_summary.conditional_percentiles_m["p10"]),
                    p25=float(ceil_summary.conditional_percentiles_m["p25"]),
                    p50=float(ceil_summary.conditional_percentiles_m["p50"]),
                    p75=float(ceil_summary.conditional_percentiles_m["p75"]),
                    p90=float(ceil_summary.conditional_percentiles_m["p90"]),
                )
            else:
                stats = EnsembleStatistics(
                    mean=None, median=None, spread=None,
                    p10=None, p25=None, p50=None, p75=None, p90=None
                )
        else:
            stats = EnsembleStatistics(
                mean=None, median=None, spread=None,
                p10=None, p25=None, p50=None, p75=None, p90=None
            )
            unlimited_prob_payload = None
            valid_member_count_payload = 0
            finite_member_count_payload = 0
            unlimited_member_count_payload = 0
    else:
        members = _gated_member_values(
            str(run.zarr_store_path), variable, leads[0], latitude, longitude
        )

    pdf_payload: EnsemblePDF | None = None
    if include_members:
        # Filter non-finite/sentinel values for PDF if needed
        valid_pdf_members = [m for m in members if not math.isnan(m) and m < 19990.0] if variable == "cloud_ceiling" else [m for m in members if not math.isnan(m)]
        domain_pdf = estimate_ensemble_pdf(valid_pdf_members) if len(valid_pdf_members) >= 2 else None
        if domain_pdf is not None:
            pdf_payload = EnsemblePDF(x=domain_pdf.x, density=domain_pdf.density)

    if stats is None:
        stats = EnsembleStatistics(
            mean=ensemble_mean(members),
            median=ensemble_median(members),
            spread=ensemble_spread(members),
            p10=ensemble_percentile(members, 10),
            p25=ensemble_percentile(members, 25),
            p50=ensemble_percentile(members, 50),
            p75=ensemble_percentile(members, 75),
            p90=ensemble_percentile(members, 90),
        )

    return EnsembleStatisticsData(
        model=model,
        lead_time_hours=lead_time_hours,
        member_count=len(members),
        statistics=stats,
        members=members if include_members else None,
        pdf=pdf_payload if include_members else None,
        consensus_vector=consensus_payload,
        wind_rose=wind_rose_payload,
        phase_support=phase_support_payload,
        transition_frequency=transition_freq_payload,
        valid_member_count=valid_member_count_payload,
        unlimited_probability=unlimited_prob_payload,
        finite_member_count=finite_member_count_payload,
        unlimited_member_count=unlimited_member_count_payload,
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


def _gated_wind_member_vectors(
    store_path: str,
    lead: int,
    latitude: float,
    longitude: float,
) -> tuple[list[float], list[float]]:
    """Interpolate ensemble member u and v vectors at a point and lead time.

    Returns:
        A tuple of (u_members, v_members) where each is a 1-D list in m/s across
        ensemble members in coordinate order.
    """
    from api.core.reader_gate import gated_read_dataset_with_selector

    def select_and_interpolate(dataset: xr.Dataset) -> tuple[list[float], list[float]]:
        if "wind_u_10m" not in dataset.data_vars or "wind_v_10m" not in dataset.data_vars:
            raise HTTPException(
                status_code=404,
                detail="Variables 'wind_u_10m' and 'wind_v_10m' are not available in the forecast dataset.",
            )
        field_u = dataset["wind_u_10m"]
        field_v = dataset["wind_v_10m"]
        if "member" not in field_u.dims or "member" not in field_v.dims:
            raise HTTPException(
                status_code=422,
                detail="Wind variables have no ensemble member dimension in the forecast dataset.",
            )
        if "lead_time_hours" in field_u.dims:
            field_u = field_u.sel(lead_time_hours=lead)
        if "lead_time_hours" in field_v.dims:
            field_v = field_v.sel(lead_time_hours=lead)

        grid, lat_descending, lon_descending = _derive_grid(dataset)
        member_count = int(dataset.coords["member"].size)
        u_vals: list[float] = []
        v_vals: list[float] = []
        for member_index in range(member_count):
            mf_u = field_u.isel(member=member_index)
            mf_v = field_v.isel(member=member_index)
            if mf_u.ndim != 2 or mf_v.ndim != 2:
                raise HTTPException(
                    status_code=500,
                    detail="Wind variables are not 2-D surface fields per member.",
                )
            try:
                u_vals.append(
                    _interpolate_neighborhood(
                        mf_u,
                        grid,
                        lat_descending,
                        lon_descending,
                        latitude,
                        longitude,
                    )
                )
                v_vals.append(
                    _interpolate_neighborhood(
                        mf_v,
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
        return u_vals, v_vals

    return gated_read_dataset_with_selector(store_path, select_and_interpolate)


def _gated_precipitation_member_states(
    store_path: str,
    lead: int,
    latitude: float,
    longitude: float,
) -> tuple[list[float], list[PrecipitationPhaseState]]:
    """Extract per-member precipitation amounts and classify their phase states under the gate."""
    from api.core.reader_gate import gated_read_dataset_with_selector

    def select_and_classify(dataset: xr.Dataset) -> tuple[list[float], list[PrecipitationPhaseState]]:
        if "precipitation_amount_3h" not in dataset.data_vars:
            raise HTTPException(
                status_code=404,
                detail="Variable 'precipitation_amount_3h' is not available in the forecast dataset.",
            )
        field_amt = dataset["precipitation_amount_3h"]
        if "member" not in field_amt.dims:
            raise HTTPException(
                status_code=422,
                detail="Variable 'precipitation_amount_3h' has no ensemble member dimension in the forecast dataset.",
            )
        if "lead_time_hours" in field_amt.dims:
            field_amt = field_amt.sel(lead_time_hours=lead)

        grid, lat_descending, lon_descending = _derive_grid(dataset)
        member_count = int(dataset.coords["member"].size)

        cat_fields = {}
        for c in ("crain", "csnow", "cfrzr", "cicep"):
            if c in dataset.data_vars:
                f = dataset[c]
                if "lead_time_hours" in f.dims:
                    f = f.sel(lead_time_hours=lead)
                cat_fields[c] = f

        t2m_field = None
        if "temperature_2m" in dataset.data_vars:
            t = dataset["temperature_2m"]
            if "lead_time_hours" in t.dims:
                t = t.sel(lead_time_hours=lead)
            t2m_field = t

        # Predecessor fields for 6-hour reset leads (t=6, 12, 18, 24, ...)
        pred_fields_avail = False
        pred_field_amt = None
        pred_cat_fields = {}
        pred_t2m_field = None

        if lead % 6 == 0 and lead > 0:
            pred_lead = lead - 3
            leads_in_ds = [
                int(v)
                for v in np.atleast_1d(dataset.coords["lead_time_hours"].values).reshape(-1)
            ]
            if pred_lead in leads_in_ds:
                pred_fields_avail = True
                pred_field_amt = dataset["precipitation_amount_3h"].sel(lead_time_hours=pred_lead)
                for c in ("crain", "csnow", "cfrzr", "cicep"):
                    if c in dataset.data_vars:
                        pred_cat_fields[c] = dataset[c].sel(lead_time_hours=pred_lead)
                if "temperature_2m" in dataset.data_vars:
                    pred_t2m_field = dataset["temperature_2m"].sel(lead_time_hours=pred_lead)

        amounts: list[float] = []
        states: list[PrecipitationPhaseState] = []

        for member_index in range(member_count):
            amt_val = float(
                _interpolate_neighborhood(
                    field_amt.isel(member=member_index),
                    grid,
                    lat_descending,
                    lon_descending,
                    latitude,
                    longitude,
                )
            )
            amounts.append(amt_val)

            flags: dict[str, int] = {}
            for c, c_field in cat_fields.items():
                c_val = float(
                    _interpolate_neighborhood(
                        c_field.isel(member=member_index),
                        grid,
                        lat_descending,
                        lon_descending,
                        latitude,
                        longitude,
                    )
                )
                flags[c] = 1 if c_val >= 0.5 else 0

            t_val: float | None = None
            if t2m_field is not None:
                t_val = float(
                    _interpolate_neighborhood(
                        t2m_field.isel(member=member_index),
                        grid,
                        lat_descending,
                        lon_descending,
                        latitude,
                        longitude,
                    )
                )

            amt_prev: float | None = None
            flags_prev: dict[str, int] | None = None
            t_start_val: float | None = None

            if pred_fields_avail and pred_field_amt is not None:
                amt_prev = float(
                    _interpolate_neighborhood(
                        pred_field_amt.isel(member=member_index),
                        grid,
                        lat_descending,
                        lon_descending,
                        latitude,
                        longitude,
                    )
                )
                if pred_cat_fields:
                    f_p = {}
                    for c, c_p_field in pred_cat_fields.items():
                        c_p_val = float(
                            _interpolate_neighborhood(
                                c_p_field.isel(member=member_index),
                                grid,
                                lat_descending,
                                lon_descending,
                                latitude,
                                longitude,
                            )
                        )
                        f_p[c] = 1 if c_p_val >= 0.5 else 0
                    flags_prev = f_p
                if pred_t2m_field is not None:
                    t_start_val = float(
                        _interpolate_neighborhood(
                            pred_t2m_field.isel(member=member_index),
                            grid,
                            lat_descending,
                            lon_descending,
                            latitude,
                            longitude,
                        )
                    )

            st = classify_precipitation_phase(
                amt_val,
                flags if flags else None,
                amount_prev=amt_prev,
                flags_prev=flags_prev,
                t2m_start=t_start_val,
                t2m_end=t_val,
            )
            states.append(st)

        return amounts, states

    return gated_read_dataset_with_selector(store_path, select_and_classify)
