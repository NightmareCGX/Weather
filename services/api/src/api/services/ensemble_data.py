"""Ensemble data construction: per-member Zarr slicing and domain math.

This service backs the ``/v1/probabilities`` and ``/v1/ensembles`` endpoints
(API.md sections 3.1 and 5.1). It enforces the 85% member-coverage lead and cell
serving thresholds, queries committed member indices from the catalog, filters out
unavailable members, and operates strictly on finite participating samples.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Literal

import numpy as np
import xarray as xr
from domain.coverage import (
    get_expected_members,
    is_cell_statistically_valid,
    is_lead_servable,
)
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

from api.models.entities import (
    EnsembleMemberProduct,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.lifecycle import filter_visible_runs, require_cycle_visible
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
    _CycleMetadata,
    _derive_grid,
    _interpolate_neighborhood,
    _parse_cycle_time,
    _resolve_variables,
    gated_cycle_metadata,
)

logger = logging.getLogger(__name__)

#: Error status code for invalid probability/ensemble inputs.
_STATUS_INVALID_INPUT = 422

#: ModelRun lifecycle statuses eligible for serving.
SERVING_ELIGIBLE_STATUSES: tuple[str, ...] = ("ready", "processing", "partial")


def _validate_coordinates(latitude: float, longitude: float) -> None:
    """Validate WGS 84 coordinates, mapping the domain error to 422."""
    try:
        validate_coordinates(latitude, longitude)
    except InvalidCoordinatesError as exc:
        raise HTTPException(status_code=_STATUS_INVALID_INPUT, detail=str(exc)) from exc


def _resolve_eligible_ensemble_run_and_members(
    db: Session,
    model: str,
    lead_time_hours: int,
    initial_time: str | None = None,
) -> tuple[ModelRun, _CycleMetadata, tuple[int, ...]]:
    """Return the newest eligible ensemble run and its committed member indices.

    For the given model and lead_time_hours, finds the newest run in ('ready',
    'processing', 'partial') where the lead is servable (coverage >= 85% of
    expected_members).

    When initial_time is provided, pins to that exact cycle; if that cycle is not
    eligible or store is unreadable, raises HTTP 404.
    When initial_time is omitted, searches newest-to-oldest eligible cycles and
    returns the first readable one.

    Returns:
        (run, metadata, available_member_indices)

    Raises:
        HTTPException: 404 if no eligible run is found.
    """
    if initial_time is not None:
        require_cycle_visible(db, initial_time)

    expected_members = get_expected_members(model, default_if_unknown=30)
    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        stmt = stmt.where(ModelRun.cycle_time == _parse_cycle_time(initial_time))
    stmt = filter_visible_runs(stmt).order_by(ModelRun.cycle_time.desc())
    runs = list(db.execute(stmt).scalars().all())
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast run with data was found for model '{model}'"
            + (f" and initial time '{initial_time}'." if initial_time else "."),
        )

    for run in runs:
        member_rows = db.execute(
            select(EnsembleMemberProduct.member_index).where(
                EnsembleMemberProduct.run_id == run.id,
                EnsembleMemberProduct.lead_time_hours == lead_time_hours,
            )
        ).scalars().all()
        avail_members = tuple(sorted(int(m) for m in member_rows))

        # If no EnsembleMemberProduct rows (legacy store or test mock without pair rows),
        # probe store to see if lead coordinate exists and run is ready
        if not avail_members:
            assert run.zarr_store_path is not None
            try:
                metadata = gated_cycle_metadata(str(run.zarr_store_path))
                if lead_time_hours in metadata.lead_times and run.status == "ready":
                    avail_members = tuple(range(1, expected_members + 1))
            except Exception:
                continue

        # Filter out physically fenced members from reclamation_queue
        try:
            from api.models.entities import ReclamationQueue

            fenced_rows = db.execute(
                select(ReclamationQueue.member_index).where(
                    ReclamationQueue.run_id == run.id,
                    ReclamationQueue.lead_time_hours == lead_time_hours,
                    ReclamationQueue.target_kind == "mem",
                    ReclamationQueue.status.in_(("deleting", "deleted", "failed")),
                )
            ).scalars().all()
            fenced_set = {int(m) for m in fenced_rows}
            if fenced_set:
                avail_members = tuple(m for m in avail_members if m not in fenced_set)
        except Exception:
            pass

        if not is_lead_servable(len(avail_members), expected_members):
            if initial_time is not None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Forecast lead {lead_time_hours}h for model '{model}' at cycle '{initial_time}' "
                        f"is not servable ({len(avail_members)}/{expected_members} members < 85% threshold)."
                    ),
                )
            continue

        assert run.zarr_store_path is not None
        try:
            metadata = gated_cycle_metadata(str(run.zarr_store_path))
        except Exception as exc:
            logger.warning("Skipping unreadable Zarr store for run %s: %s", run.id, exc)
            continue

        if lead_time_hours not in metadata.lead_times:
            if initial_time is not None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Lead {lead_time_hours}h not found in store for cycle '{initial_time}'.",
                )
            continue

        return run, metadata, avail_members

    raise HTTPException(
        status_code=404,
        detail=(
            f"No eligible forecast run with sufficient member coverage (>= 85%) "
            f"was found for model '{model}' at lead {lead_time_hours}h."
        ),
    )


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
    source: Any | None = None,
) -> ProbabilityForecastData:
    """Build an exceedance probability forecast for a resolved point."""
    _validate_coordinates(latitude, longitude)
    _require_ensemble_model(db, model)
    expected_members = get_expected_members(model, default_if_unknown=30)
    if source is not None:
        avail_members = (
            source.member_indices
            if source.member_indices is not None
            else tuple(range(1, expected_members + 1))
        )
        store_path_str = str(source.store_path)
        lead_time_hours = source.lead_time_hours
        provenance_run_id = str(source.run_id)
        metadata = gated_cycle_metadata(store_path_str)
    else:
        run, metadata, avail_members = _resolve_eligible_ensemble_run_and_members(
            db, model, lead_time_hours, initial_time=initial_time
        )
        assert run.zarr_store_path is not None
        store_path_str = str(run.zarr_store_path)
        provenance_run_id = str(run.id)
    _resolve_variables(db, metadata, [variable])
    # Release ORM DB connection before storage reads.
    db.close()

    # Prefer the aggregate shard when the store has a usable one. It is the storage
    # representation going forward, and it is markedly cheaper: one byte range per point
    # instead of a read and an interpolation per member. The member path below remains the
    # reader of record, so ``None`` here simply means "no aggregate answered".
    served_probability = _aggregate_probability(
        variable=variable,
        store_path=store_path_str,
        lead_time_hours=lead_time_hours,
        latitude=latitude,
        longitude=longitude,
        threshold=threshold,
        operator=operator,
        threshold_max=threshold_max,
        direction_sector=direction_sector,
        phase=phase,
        expected_members=expected_members,
    )
    if served_probability is not None:
        probability, lower, upper = served_probability
        served_data: dict[str, Any] = {
            "location": ProbabilityLocation(latitude=latitude, longitude=longitude),
            "variable": variable,
            "threshold": threshold,
            "operator": operator,
            "lead_time_hours": lead_time_hours,
            "probability": probability,
            "confidence_interval_95": [lower, upper],
        }
        if operator == "between":
            served_data["threshold_max"] = threshold_max
        if direction_sector is not None:
            served_data["direction_sector"] = direction_sector
        if phase is not None:
            served_data["phase"] = phase
        return ProbabilityForecastData(**served_data)

    if variable == "wind_10m":
        u_members, v_members = _gated_wind_member_vectors(
            store_path_str, lead_time_hours, latitude, longitude, avail_members
        )
        finite_pairs = [
            (u, v)
            for u, v in zip(u_members, v_members, strict=True)
            if math.isfinite(u) and math.isfinite(v)
        ]
        if not is_cell_statistically_valid(len(finite_pairs), expected_members):
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data covers the requested location with sufficient member coverage (finite {len(finite_pairs)}/{expected_members} < 85%).",
            )
        u_finite = [p[0] for p in finite_pairs]
        v_finite = [p[1] for p in finite_pairs]
        if direction_sector is not None:
            speed_mps = threshold / 3.6
            try:
                probability, (lower, upper) = compute_directional_probability(
                    u_finite,
                    v_finite,
                    sector=direction_sector,
                    speed_threshold=speed_mps,
                    operator=operator,
                )
            except InvalidThresholdError as exc:
                raise HTTPException(status_code=_STATUS_INVALID_INPUT, detail=str(exc)) from exc
        else:
            members_kmh = [math.hypot(u, v) * 3.6 for u, v in zip(u_finite, v_finite, strict=True)]
            probability = _probability(members_kmh, threshold, operator, threshold_max)
            lower, upper = probability_confidence_interval(probability, len(members_kmh))
    elif variable == "precipitation_amount_3h" and phase is not None:
        amounts, precip_states = _gated_precipitation_member_states(
            store_path_str, lead_time_hours, latitude, longitude, avail_members
        )
        finite_states = [st for amt, st in zip(amounts, precip_states, strict=True) if math.isfinite(amt)]
        if not is_cell_statistically_valid(len(finite_states), expected_members):
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data covers the requested location with sufficient member coverage (finite {len(finite_states)}/{expected_members} < 85%).",
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
                finite_states, threshold_mm=threshold, phase=target_phase
            )
        except ValueError as exc:
            raise HTTPException(status_code=_STATUS_INVALID_INPUT, detail=str(exc)) from exc
        lower, upper = probability_confidence_interval(probability, len(finite_states))
    elif variable == "cloud_ceiling" and operator in ("lt", "lte"):
        members = _gated_member_values(
            store_path_str, variable, lead_time_hours, latitude, longitude, avail_members
        )
        finite_members = [m for m in members if math.isfinite(m)]
        if not is_cell_statistically_valid(len(finite_members), expected_members):
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data covers the requested location with sufficient member coverage (finite {len(finite_members)}/{expected_members} < 85%).",
            )
        prob = compute_low_ceiling_probability(finite_members, threshold_km=threshold)
        if prob is None:
            raise HTTPException(
                status_code=404,
                detail="Insufficient valid members for cloud ceiling probability.",
            )
        valid_n = sum(1 for m in finite_members if float(m) >= 0.0)
        lower, upper = probability_confidence_interval(prob, valid_n)
        probability = prob
    elif variable == "cloud_cover_3h":
        members = _gated_member_values(
            store_path_str, variable, lead_time_hours, latitude, longitude, avail_members
        )
        valid_members = [float(m) for m in members if math.isfinite(m) and 0.0 <= float(m) <= 100.0]
        if not is_cell_statistically_valid(len(valid_members), expected_members):
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data covers the requested location with sufficient member coverage (valid {len(valid_members)}/{expected_members} < 85%).",
            )
        probability = _probability(valid_members, threshold, operator, threshold_max)
        lower, upper = probability_confidence_interval(probability, len(valid_members))
    else:
        members = _gated_member_values(
            store_path_str, variable, lead_time_hours, latitude, longitude, avail_members
        )
        finite_members = [m for m in members if math.isfinite(m)]
        if not is_cell_statistically_valid(len(finite_members), expected_members):
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data covers the requested location with sufficient member coverage (finite {len(finite_members)}/{expected_members} < 85%).",
            )
        probability = _probability(finite_members, threshold, operator, threshold_max)
        lower, upper = probability_confidence_interval(probability, len(finite_members))

    # Post-read validation: verify participating member shards did not transition to deleting/deleted
    _revalidate_member_fencing(
        db,
        provenance_run_id=provenance_run_id,
        lead_time_hours=lead_time_hours,
        member_indices=avail_members,
    )

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
    source: Any | None = None,
    series_mode: bool = False,
) -> EnsembleStatisticsData:
    """Build ensemble statistics for a resolved point."""
    _validate_coordinates(latitude, longitude)
    _require_ensemble_model(db, model)
    expected_members = get_expected_members(model, default_if_unknown=30)
    if source is not None:
        avail_members = (
            source.member_indices
            if source.member_indices is not None
            else tuple(range(1, expected_members + 1))
        )
        store_path_str = str(source.store_path)
        lead_time_hours = source.lead_time_hours
        provenance_run_id = str(source.run_id)
        metadata = gated_cycle_metadata(store_path_str)
    else:
        run, metadata, avail_members = _resolve_eligible_ensemble_run_and_members(
            db, model, lead_time_hours, initial_time=initial_time
        )
        assert run.zarr_store_path is not None
        store_path_str = str(run.zarr_store_path)
        provenance_run_id = str(run.id)
    _resolve_variables(db, metadata, [variable])
    # Release ORM DB connection before storage reads.
    db.close()

    # Prefer the aggregate shard when the store has a usable one for this variable and lead.
    # It is the storage representation going forward, and it is markedly cheaper: one byte
    # range per field group instead of a read and an interpolation per member. ``None`` means
    # "no aggregate answers this", and the member path below stays the reader of record.
    #
    # Two responses are still the member path's alone: the raw members and the KDE built from
    # them. Both are what ``include_members`` asks for, and an aggregate has collapsed the
    # members by construction -- there is no reading of a stored distribution that returns them.
    # Every other field in the response is stored: the statistics are the distribution, and the
    # consensus vector, rose, phase support, transition frequencies and censoring counts are the
    # supplementary groups (``domain.field_layout``), computed once per cycle by the same
    # arithmetic this module applies per request.
    if not series_mode:
        from api.services.aggregate_serving import (
            aggregate_can_answer,
            gated_cloud_censoring,
            gated_precipitation_phase,
            gated_statistics_from_aggregate,
            gated_wind_products,
            try_read_aggregate,
        )

        served_products: dict[str, Any] | None = None
        if variable == "wind_10m":
            served_products = try_read_aggregate(
                lambda: gated_wind_products(
                    store_path=str(store_path_str),
                    lead_time_hours=lead_time_hours,
                    latitude=latitude,
                    longitude=longitude,
                )
            )
        elif variable == "precipitation_amount_3h":
            served_products = try_read_aggregate(
                lambda: gated_precipitation_phase(
                    store_path=str(store_path_str),
                    lead_time_hours=lead_time_hours,
                    latitude=latitude,
                    longitude=longitude,
                )
            )
        elif variable in ("cloud_ceiling", "cloud_cover_3h"):
            served_products = try_read_aggregate(
                lambda: gated_cloud_censoring(
                    variable,
                    store_path=str(store_path_str),
                    lead_time_hours=lead_time_hours,
                    latitude=latitude,
                    longitude=longitude,
                )
            )

        # The raw members and the KDE built from them are the one thing an aggregate cannot
        # answer: it has collapsed the members by construction. A request that asks for them
        # takes the member path for the whole response, rather than mixing a stored product with
        # an absent member list.
        if not include_members and aggregate_can_answer(variable):
            served = try_read_aggregate(
                lambda: gated_statistics_from_aggregate(
                    variable,
                    store_path=str(store_path_str),
                    lead_time_hours=lead_time_hours,
                    latitude=latitude,
                    longitude=longitude,
                )
            )
            if served is not None and served.member_count is not None:
                _revalidate_member_fencing(
                    db,
                    provenance_run_id=provenance_run_id,
                    lead_time_hours=lead_time_hours,
                    member_indices=avail_members,
                )
                served_stats = EnsembleStatistics(**served.as_ensemble_statistics_kwargs())
                return _ensemble_payload_from_aggregate(
                    model=model,
                    lead_time_hours=lead_time_hours,
                    member_count=served.member_count,
                    statistics=served_stats,
                    products=served_products,
                    source_cycle=None,
                )

    consensus_payload: ConsensusVectorOut | None = None
    wind_rose_payload: WindRoseOut | None = None
    phase_support_payload: dict[str, float] | None = None
    transition_freq_payload: dict[str, float] | None = None
    valid_member_count_payload: int | None = None
    unlimited_prob_payload: float | None = None
    finite_member_count_payload: int | None = None
    unlimited_member_count_payload: int | None = None
    stats: EnsembleStatistics | None = None
    participating_members: list[float] = []

    if variable == "wind_10m":
        u_members, v_members = _gated_wind_member_vectors(
            store_path_str, lead_time_hours, latitude, longitude, avail_members
        )
        finite_pairs = [
            (u, v)
            for u, v in zip(u_members, v_members, strict=True)
            if math.isfinite(u) and math.isfinite(v)
        ]
        u_fin = [p[0] for p in finite_pairs]
        v_fin = [p[1] for p in finite_pairs]
        participating_members = [math.hypot(u, v) * 3.6 for u, v in zip(u_fin, v_fin, strict=True)]
        valid_cell = is_cell_statistically_valid(len(participating_members), expected_members)

        if valid_cell and u_fin and not series_mode:
            consensus = compute_consensus_vector(u_fin, v_fin)
            consensus_payload = ConsensusVectorOut(
                speed=round(consensus.speed_mps * 3.6, 2),
                direction=round(consensus.direction_deg, 1) if consensus.direction_deg is not None else None,
                cardinal=consensus.cardinal,
                coherence=round(consensus.coherence, 4),
            )
            rose = compute_wind_rose(u_fin, v_fin)
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
            store_path_str, lead_time_hours, latitude, longitude, avail_members
        )
        if all(math.isnan(m) for m in amounts):
            participating_members = [0.0] * len(amounts)
            finite_states = precip_states
        else:
            finite_pairs_p = [
                (amt, st) for amt, st in zip(amounts, precip_states, strict=True) if math.isfinite(amt)
            ]
            participating_members = [p[0] for p in finite_pairs_p]
            finite_states = [p[1] for p in finite_pairs_p]

        valid_cell = is_cell_statistically_valid(len(participating_members), expected_members)
        if valid_cell and finite_states:
            support_map = aggregate_ensemble_phase_support(finite_states)
            phase_support_payload = {p.value: round(v, 4) for p, v in support_map.items()}
            if not series_mode:
                freq_map = compute_transition_frequencies(finite_states)
                transition_freq_payload = {
                    t.value: round(v, 4) for t, v in freq_map.items() if v > 0.0
                }
    elif variable == "cloud_cover_3h":
        members = _gated_member_values(
            store_path_str, variable, lead_time_hours, latitude, longitude, avail_members
        )
        participating_members = [float(m) for m in members if math.isfinite(m) and 0.0 <= float(m) <= 100.0]
        valid_cell = is_cell_statistically_valid(len(participating_members), expected_members)
        if valid_cell:
            min_v = 21 if len(participating_members) >= 21 else 1
            cc_summary = cloud_cover_ensemble_summary(participating_members, min_valid=min_v)
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
                valid_member_count_payload = len(participating_members)
        else:
            stats = EnsembleStatistics(
                mean=None, median=None, spread=None,
                p10=None, p25=None, p50=None, p75=None, p90=None
            )
            valid_member_count_payload = len(participating_members)
    elif variable == "cloud_ceiling":
        members = _gated_member_values(
            store_path_str, variable, lead_time_hours, latitude, longitude, avail_members
        )
        participating_members = [float(m) for m in members if math.isfinite(m) and float(m) >= 0.0]
        valid_cell = is_cell_statistically_valid(len(participating_members), expected_members)
        if valid_cell:
            min_v = 21 if len(participating_members) >= 21 else 1
            ceil_summary = cloud_ceiling_ensemble_summary(
                participating_members,
                min_finite=10,
                min_valid=min_v,
            )
            if ceil_summary is not None:
                unlimited_prob_payload = round(ceil_summary.unlimited_probability, 4)
                valid_member_count_payload = ceil_summary.valid_member_count
                finite_member_count_payload = ceil_summary.finite_member_count
                unlimited_member_count_payload = ceil_summary.unlimited_member_count
                if ceil_summary.conditional_percentiles is not None:
                    stats = EnsembleStatistics(
                        mean=float(ceil_summary.conditional_mean) if ceil_summary.conditional_mean is not None else None,
                        median=float(ceil_summary.conditional_median) if ceil_summary.conditional_median is not None else None,
                        spread=float(ceil_summary.conditional_spread) if ceil_summary.conditional_spread is not None else None,
                        p10=float(ceil_summary.conditional_percentiles["p10"]),
                        p25=float(ceil_summary.conditional_percentiles["p25"]),
                        p50=float(ceil_summary.conditional_percentiles["p50"]),
                        p75=float(ceil_summary.conditional_percentiles["p75"]),
                        p90=float(ceil_summary.conditional_percentiles["p90"]),
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
        else:
            stats = EnsembleStatistics(
                mean=None, median=None, spread=None,
                p10=None, p25=None, p50=None, p75=None, p90=None
            )
    else:
        members = _gated_member_values(
            store_path_str, variable, lead_time_hours, latitude, longitude, avail_members
        )
        participating_members = [m for m in members if math.isfinite(m)]
        valid_cell = is_cell_statistically_valid(len(participating_members), expected_members)

    pdf_payload: EnsemblePDF | None = None
    if include_members:
        valid_pdf_members = (
            [m for m in participating_members if m < 19.99]
            if variable == "cloud_ceiling"
            else participating_members
        )
        domain_pdf = (
            estimate_ensemble_pdf(valid_pdf_members)
            if len(valid_pdf_members) >= 2
            else None
        )
        if domain_pdf is not None:
            pdf_payload = EnsemblePDF(x=domain_pdf.x, density=domain_pdf.density)

    if stats is None:
        valid_cell = is_cell_statistically_valid(len(participating_members), expected_members)
        if valid_cell and participating_members:
            stats = EnsembleStatistics(
                mean=ensemble_mean(participating_members),
                median=ensemble_median(participating_members),
                spread=ensemble_spread(participating_members),
                p10=ensemble_percentile(participating_members, 10),
                p25=ensemble_percentile(participating_members, 25),
                p50=ensemble_percentile(participating_members, 50),
                p75=ensemble_percentile(participating_members, 75),
                p90=ensemble_percentile(participating_members, 90),
            )
        else:
            stats = EnsembleStatistics(
                mean=None, median=None, spread=None,
                p10=None, p25=None, p50=None, p75=None, p90=None
            )

    # Post-read validation: verify participating member shards did not transition to deleting/deleted
    _revalidate_member_fencing(
        db,
        provenance_run_id=provenance_run_id,
        lead_time_hours=lead_time_hours,
        member_indices=avail_members,
    )

    return EnsembleStatisticsData(
        model=model,
        lead_time_hours=lead_time_hours,
        member_count=len(participating_members),
        statistics=stats,
        members=participating_members if include_members else None,
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


def _ensemble_payload_from_aggregate(
    *,
    model: str,
    lead_time_hours: int,
    member_count: int,
    statistics: EnsembleStatistics,
    products: dict[str, Any] | None,
    source_cycle: datetime | None,
) -> EnsembleStatisticsData:
    """Assemble a response from a container's stored fields, with no member read.

    The stored-field shapes are the response's own shapes, so this is a translation and nothing
    else: which supplementary group answers which response field is decided by the *variable*,
    because that is what the group was computed for. A field the container does not carry stays
    ``None`` -- the response schema models every one of them as optional, and ``None`` is how it
    says "this store's container has nothing for this lead", which is a different statement from
    a zero.
    """
    consensus_payload: ConsensusVectorOut | None = None
    wind_rose_payload: WindRoseOut | None = None
    phase_support_payload: dict[str, float] | None = None
    transition_freq_payload: dict[str, float] | None = None
    valid_member_count_payload: int | None = None
    unlimited_prob_payload: float | None = None
    finite_member_count_payload: int | None = None
    unlimited_member_count_payload: int | None = None

    if products is not None:
        if "consensus" in products:
            consensus = products["consensus"]
            consensus_payload = ConsensusVectorOut(
                speed=round(float(consensus["speed_mps"]) * 3.6, 2),
                direction=(
                    round(float(consensus["direction_deg"]), 1)
                    if consensus["direction_deg"] is not None
                    else None
                ),
                cardinal=str(consensus["cardinal"]),
                coherence=round(float(consensus["coherence"]), 4),
            )
        if "rose" in products:
            rose = products["rose"]
            wind_rose_payload = WindRoseOut(
                calm_percentage=round(float(rose["calm_probability"]) * 100.0, 1),
                calm_count=int(rose["calm_count"]),
                sectors=[
                    WindRoseSectorOut(
                        sector=str(sector["sector"]),
                        count=int(sector["count"]),
                        probability=round(float(sector["probability"]), 4),
                        bins={
                            key: round(float(value), 4)
                            for key, value in sector["bins"].items()
                        },
                    )
                    for sector in rose["sectors"]
                ],
            )
        if "phase_support" in products:
            support = products["phase_support"]
            # The container's phase planes are its own names, including the predecessor's; the
            # response reports the current interval's support, which is what the member path's
            # ``aggregate_ensemble_phase_support`` returns.
            phase_support_payload = {
                name: round(float(value), 4)
                for name, value in support.items()
                if value is not None and not name.startswith("previous_")
            }
        if "transition_frequency" in products:
            transition_freq_payload = {
                name: round(float(value), 4)
                for name, value in products["transition_frequency"].items()
            }
        if "valid_member_count" in products:
            valid_member_count_payload = products["valid_member_count"]
            unlimited_prob_payload = products["unlimited_probability"]
            finite_member_count_payload = products["finite_member_count"]
            unlimited_member_count_payload = products["unlimited_member_count"]
            conditional = products["conditional"]
            if conditional:
                statistics = EnsembleStatistics(
                    mean=conditional.get("mean"),
                    median=conditional.get("p50"),
                    spread=conditional.get("spread"),
                    p10=conditional.get("p10"),
                    p25=conditional.get("p25"),
                    p50=conditional.get("p50"),
                    p75=conditional.get("p75"),
                    p90=conditional.get("p90"),
                )

    return EnsembleStatisticsData(
        model=model,
        lead_time_hours=lead_time_hours,
        member_count=member_count,
        statistics=statistics,
        consensus_vector=consensus_payload,
        wind_rose=wind_rose_payload,
        phase_support=phase_support_payload,
        transition_frequency=transition_freq_payload,
        valid_member_count=valid_member_count_payload,
        unlimited_probability=unlimited_prob_payload,
        finite_member_count=finite_member_count_payload,
        unlimited_member_count=unlimited_member_count_payload,
    )


def _aggregate_probability(
    *,
    variable: str,
    store_path: str,
    lead_time_hours: int,
    latitude: float,
    longitude: float,
    threshold: float,
    operator: str,
    threshold_max: float | None,
    direction_sector: str | None,
    phase: str | None,
    expected_members: int,
) -> tuple[float, float, float] | None:
    """Serve ``(probability, lower, upper)`` from an aggregate, or ``None`` for the members.

    ``None`` covers three distinct situations, and they all mean the same thing to the caller:
    the store has no aggregate for this ``(variable, lead)``; the query asks something a
    stored distribution cannot answer; or the aggregate's point is unobserved.

    What a stored distribution cannot answer, and why:

    * the inclusive operators. ``gte``/``lte`` count members *at* the threshold, which needs
      the atom at that value -- a continuous quantile function or a plus-minus-sigma histogram
      has smoothed it away. Measured on a half-zero precipitation field at the natural
      threshold of zero, the members give ``P(x >= 0) = 1.0`` (the atom is 0.53) where the
      interpolated quantile function gives 0.5, and neither is what the members say;
    * a directional or phase-conditioned query, which is a function of each member's vector or
      flag rather than of the distribution;
    * a ``between`` query, which is two inclusive thresholds at once.

    The confidence interval is the same Wilson interval the member path reports, computed from
    the stored member count. It is a statement about the *sample size*, not about the encoding's
    reconstruction error, so it is exactly as valid here as there -- and it is the interval the
    contract already reports for this endpoint.
    """
    from api.services.aggregate_serving import (
        aggregate_can_answer,
        gated_exceedance_from_aggregate,
        try_read_aggregate,
    )

    if threshold_max is not None:
        return None
    if direction_sector is not None or phase is not None:
        return None
    if not aggregate_can_answer(variable, operator=operator):
        return None

    served = try_read_aggregate(
        lambda: gated_exceedance_from_aggregate(
            variable,
            store_path=store_path,
            lead_time_hours=lead_time_hours,
            latitude=latitude,
            longitude=longitude,
            threshold=threshold,
            operator=operator,
        )
    )
    if served is None or served.member_count is None:
        return None

    # The interval is the same Wilson score interval the member path reports, from the stored
    # member count. It quantifies the *sample size*, not the encoding's reconstruction error,
    # so it means the same thing here -- and it is what this endpoint already returns.
    lower, upper = probability_confidence_interval(
        served.probability, max(served.member_count, 1)
    )
    return served.probability, float(lower), float(upper)


def _revalidate_member_fencing(
    db: Session,
    *,
    provenance_run_id: str,
    lead_time_hours: int,
    member_indices: tuple[int, ...],
) -> None:
    """Fail the request if a participating member shard was fenced while it was being read.

    Reclamation runs concurrently with serving, so a read can span a member's transition to
    deleting/deleted and silently mix pre- and post-reclamation data. Both read paths call
    this after their read, for the same reason: the aggregate path reads the members' *derived*
    statistics, so a member reclaimed mid-aggregation would serve a value that no longer has
    the members it came from.

    A database failure here is logged and ignored: revalidation protects against a race, and
    failing every request when the catalog is briefly unavailable would be a larger outage
    than the race it guards.
    """
    try:
        from api.models.entities import ReclamationQueue

        fenced_during_read = db.execute(
            select(ReclamationQueue.id).where(
                ReclamationQueue.run_id == provenance_run_id,
                ReclamationQueue.lead_time_hours == lead_time_hours,
                ReclamationQueue.target_kind == "mem",
                ReclamationQueue.member_index.in_(member_indices),
                ReclamationQueue.status.in_(("deleting", "deleted", "failed")),
            )
        ).scalars().all()
        if fenced_during_read:
            raise HTTPException(
                status_code=404,
                detail="Ensemble member shards became unavailable during read.",
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - revalidation must never break serving
        logger.error(
            "ensemble_post_read_fence_revalidation_failed: run_id=%s lead=%s error=%s",
            provenance_run_id,
            lead_time_hours,
            exc,
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


def _require_variable_in_store(metadata: _CycleMetadata, variable: str, model: str) -> None:
    """Reject a variable the resolved cycle's store does not actually carry.

    :func:`_resolve_variables` validates the request against the *catalog*; this
    validates it against the *store*. The two can disagree — ``precipitation_rate``,
    for example, is a real catalog entry that GFS carries and GEFS does not — and the
    series builder must reject that once, up front. Otherwise every lead raises the same
    store-level 404 inside the per-lead loop, where the generic per-lead handler treats
    it as a recoverable transient and the endpoint answers 200 with an empty series,
    which the UI can only present as "data is not yet available".
    """
    present = set(metadata.var_names)
    # The catalog exposes 10m wind as a single derived variable, but the store only
    # carries its two vector components (mirrors _resolve_variables).
    if variable == "wind_10m":
        available = "wind_u_10m" in present and "wind_v_10m" in present
    else:
        available = variable in present
    if not available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Variable '{variable}' is not available in the forecast dataset for "
                f"model '{model}'."
            ),
        )


def build_ensemble_statistics_series(
    db: Session,
    *,
    latitude: float,
    longitude: float,
    variable: str,
    model: str,
    leads: list[int] | None = None,
    include_members: bool = False,
    initial_time: str | None = None,
    now: Any | None = None,
) -> list[EnsembleStatisticsData]:
    """Build ensemble statistics across multiple leads for a resolved point."""
    from collections import defaultdict
    from datetime import datetime, timedelta
    from api.core.time import get_current_time
    from domain.temporal import serving_start_valid_time

    _validate_coordinates(latitude, longitude)
    _require_ensemble_model(db, model)
    expected_members = get_expected_members(model, default_if_unknown=30)

    if initial_time is not None:
        require_cycle_visible(db, initial_time)

    stmt = (
        select(ModelRun)
        .join(ModelRun.model_version)
        .join(ModelVersion.model)
        .where(Model.model_id == model)
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
        .where(ModelRun.zarr_store_path.isnot(None))
    )
    if initial_time is not None:
        stmt = stmt.where(ModelRun.cycle_time == _parse_cycle_time(initial_time))
    stmt = filter_visible_runs(stmt).order_by(ModelRun.cycle_time.desc())
    runs = list(db.execute(stmt).scalars().all())
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast run with data was found for model '{model}'"
            + (f" and initial time '{initial_time}'." if initial_time else "."),
        )
    run = runs[0]
    assert run.zarr_store_path is not None
    store_path_str = str(run.zarr_store_path)
    provenance_run_id = str(run.id)

    metadata = gated_cycle_metadata(store_path_str)
    _resolve_variables(db, metadata, [variable])
    _require_variable_in_store(metadata, variable, model)

    if leads is not None:
        target_leads = [ld for ld in leads if ld in metadata.lead_times]
    else:
        target_leads = sorted(metadata.lead_times)

    now_utc = now if isinstance(now, datetime) else get_current_time()
    min_vt = serving_start_valid_time(now_utc)
    target_leads = [
        ld for ld in target_leads
        if (run.cycle_time + timedelta(hours=ld)) >= min_vt
    ]

    if not target_leads:
        return []

    member_rows = db.execute(
        select(EnsembleMemberProduct.lead_time_hours, EnsembleMemberProduct.member_index).where(
            EnsembleMemberProduct.run_id == run.id,
            EnsembleMemberProduct.lead_time_hours.in_(target_leads),
        )
    ).all()
    members_by_lead: dict[int, list[int]] = defaultdict(list)
    for ld, m_idx in member_rows:
        members_by_lead[ld].append(int(m_idx))

    from api.models.entities import ReclamationQueue

    fenced_rows = db.execute(
        select(ReclamationQueue.lead_time_hours, ReclamationQueue.member_index).where(
            ReclamationQueue.run_id == run.id,
            ReclamationQueue.lead_time_hours.in_(target_leads),
            ReclamationQueue.target_kind == "mem",
            ReclamationQueue.status.in_(("deleting", "deleted", "failed")),
        )
    ).all()
    fenced_by_lead: dict[int, set[int]] = defaultdict(set)
    for ld, m_idx in fenced_rows:
        fenced_by_lead[ld].add(int(m_idx))

    results: list[EnsembleStatisticsData] = []
    for ld in target_leads:
        avail_list = members_by_lead.get(ld, [])
        if not avail_list and run.status == "ready":
            avail_list = list(range(1, expected_members + 1))
        fenced = fenced_by_lead.get(ld, set())
        avail_members = tuple(sorted(m for m in avail_list if m not in fenced))

        if not is_lead_servable(len(avail_members), expected_members):
            continue

        try:
            source = type(
                "Source",
                (),
                {
                    "member_indices": avail_members,
                    "store_path": store_path_str,
                    "lead_time_hours": ld,
                    "run_id": provenance_run_id,
                },
            )()
            item = build_ensemble_statistics(
                db,
                latitude=latitude,
                longitude=longitude,
                variable=variable,
                model=model,
                lead_time_hours=ld,
                include_members=include_members,
                initial_time=initial_time,
                source=source,
                series_mode=True,
            )
            results.append(item)
        except Exception as exc:
            logger.warning("Failed computing ensemble lead %d: %s", ld, exc)
            continue

    return results


def _probability(
    members: list[float],
    threshold: float,
    operator: Literal["gt", "gte", "lt", "lte", "between"],
    threshold_max: float | None,
) -> float:
    """Compute the empirical exceedance probability for the operator."""
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
    available_member_indices: tuple[int, ...] | None = None,
) -> list[float]:
    """Interpolate each committed ensemble member's field at a point and lead time."""
    from api.core.manifest_reader import manifest_generation, manifest_storage_format
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.core.zarr import get_sharded_reader

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
        grid, lat_descending, lon_descending = _derive_grid(dataset)
        member_coords = dataset.coords["member"].values
        member_val_to_pos = {
            int(v): i for i, v in enumerate(np.atleast_1d(member_coords).reshape(-1))
        }

        if available_member_indices is not None:
            target_members = [m for m in available_member_indices if m in member_val_to_pos]
        else:
            target_members = sorted(member_val_to_pos.keys())

        format_version = manifest_storage_format(store_path)
        generation = manifest_generation(store_path)

        if format_version == "sharded_v1":
            reader = get_sharded_reader(store_path)
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

            def _stored(value: int, size: int, descending: bool) -> int:
                return (size - 1 - value) if descending else value

            lat_size = int(dataset.sizes["latitude"])
            lon_size = int(dataset.sizes["longitude"])
            lat_idx = [_stored(row_0, lat_size, lat_descending), _stored(row_1, lat_size, lat_descending)]
            lon_idx = [_stored(col_0, lon_size, lon_descending), _stored(col_1, lon_size, lon_descending)]

            return reader.interpolate_members(
                var_code,
                members=target_members,
                lead_time_hours=lead,
                lat_idx=lat_idx,
                lon_idx=lon_idx,
                t_row=t_row,
                t_col=t_col,
                generation=generation,
            )

        if "lead_time_hours" in field.dims:
            field = field.sel(lead_time_hours=lead)

        values_legacy: list[float] = []
        for member_num in target_members:
            member_pos = member_val_to_pos[member_num]
            member_field = field.isel(member=member_pos)
            if member_field.ndim != 2:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Variable '{var_code}' is not a 2-D surface field; "
                        "vertical-level variables are not supported."
                    ),
                )
            try:
                values_legacy.append(
                    float(
                        _interpolate_neighborhood(
                            member_field,
                            grid,
                            lat_descending,
                            lon_descending,
                            latitude,
                            longitude,
                        )
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
        return values_legacy

    return gated_read_dataset_with_selector(store_path, select_and_interpolate)


def _gated_wind_member_vectors(
    store_path: str,
    lead: int,
    latitude: float,
    longitude: float,
    available_member_indices: tuple[int, ...] | None = None,
) -> tuple[list[float], list[float]]:
    """Interpolate committed ensemble member u and v vectors at a point and lead time."""
    from api.core.manifest_reader import manifest_generation, manifest_storage_format
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.core.zarr import get_sharded_reader

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

        grid, lat_descending, lon_descending = _derive_grid(dataset)
        member_coords = dataset.coords["member"].values
        member_val_to_pos = {
            int(v): i for i, v in enumerate(np.atleast_1d(member_coords).reshape(-1))
        }

        if available_member_indices is not None:
            target_members = [m for m in available_member_indices if m in member_val_to_pos]
        else:
            target_members = sorted(member_val_to_pos.keys())

        format_version = manifest_storage_format(store_path)
        generation = manifest_generation(store_path)

        if format_version == "sharded_v1":
            reader = get_sharded_reader(store_path)
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

            def _stored(value: int, size: int, descending: bool) -> int:
                return (size - 1 - value) if descending else value

            lat_size = int(dataset.sizes["latitude"])
            lon_size = int(dataset.sizes["longitude"])
            lat_idx = [_stored(row_0, lat_size, lat_descending), _stored(row_1, lat_size, lat_descending)]
            lon_idx = [_stored(col_0, lon_size, lon_descending), _stored(col_1, lon_size, lon_descending)]

            from api.core.zarr import get_member_executor

            ex = get_member_executor()

            def _read_uv(member_num: int) -> tuple[float, float]:
                u = reader.interpolate_point(
                    "wind_u_10m",
                    member=member_num,
                    lead_time_hours=lead,
                    lat_idx=lat_idx,
                    lon_idx=lon_idx,
                    t_row=t_row,
                    t_col=t_col,
                    generation=generation,
                )
                v = reader.interpolate_point(
                    "wind_v_10m",
                    member=member_num,
                    lead_time_hours=lead,
                    lat_idx=lat_idx,
                    lon_idx=lon_idx,
                    t_row=t_row,
                    t_col=t_col,
                    generation=generation,
                )
                return u, v

            pairs = list(ex.map(_read_uv, target_members))
            return [p[0] for p in pairs], [p[1] for p in pairs]

        if "lead_time_hours" in field_u.dims:
            field_u = field_u.sel(lead_time_hours=lead)
        if "lead_time_hours" in field_v.dims:
            field_v = field_v.sel(lead_time_hours=lead)

        u_vals: list[float] = []
        v_vals: list[float] = []
        for member_num in target_members:
            member_pos = member_val_to_pos[member_num]
            mf_u = field_u.isel(member=member_pos)
            mf_v = field_v.isel(member=member_pos)
            if mf_u.ndim != 2 or mf_v.ndim != 2:
                raise HTTPException(
                    status_code=500,
                    detail="Wind variables are not 2-D surface fields per member.",
                )
            try:
                u_vals.append(
                    float(
                        _interpolate_neighborhood(
                            mf_u,
                            grid,
                            lat_descending,
                            lon_descending,
                            latitude,
                            longitude,
                        )
                    )
                )
                v_vals.append(
                    float(
                        _interpolate_neighborhood(
                            mf_v,
                            grid,
                            lat_descending,
                            lon_descending,
                            latitude,
                            longitude,
                        )
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
    available_member_indices: tuple[int, ...] | None = None,
) -> tuple[list[float], list[PrecipitationPhaseState]]:
    """Extract per-member precipitation amounts and classify their phase states under the gate."""
    from api.core.manifest_reader import manifest_generation, manifest_storage_format
    from api.core.reader_gate import gated_read_dataset_with_selector
    from api.core.zarr import get_sharded_reader

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

        grid, lat_descending, lon_descending = _derive_grid(dataset)
        member_coords = dataset.coords["member"].values
        member_val_to_pos = {
            int(v): i for i, v in enumerate(np.atleast_1d(member_coords).reshape(-1))
        }

        if available_member_indices is not None:
            target_members = [m for m in available_member_indices if m in member_val_to_pos]
        else:
            target_members = sorted(member_val_to_pos.keys())

        format_version = manifest_storage_format(store_path)
        generation = manifest_generation(store_path)

        if format_version == "sharded_v1":
            reader = get_sharded_reader(store_path)
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

            def _stored(value: int, size: int, descending: bool) -> int:
                return (size - 1 - value) if descending else value

            lat_size = int(dataset.sizes["latitude"])
            lon_size = int(dataset.sizes["longitude"])
            lat_idx = [_stored(row_0, lat_size, lat_descending), _stored(row_1, lat_size, lat_descending)]
            lon_idx = [_stored(col_0, lon_size, lon_descending), _stored(col_1, lon_size, lon_descending)]

            leads_in_ds = (
                [int(v) for v in np.atleast_1d(dataset.coords["lead_time_hours"].values).reshape(-1)]
                if "lead_time_hours" in dataset.coords
                else []
            )

            from api.core.zarr import get_member_executor

            ex = get_member_executor()

            def _read_precip_member(member_num: int) -> tuple[float, PrecipitationPhaseState]:
                amt_val = float(
                    reader.interpolate_point(
                        "precipitation_amount_3h",
                        member=member_num,
                        lead_time_hours=lead,
                        lat_idx=lat_idx,
                        lon_idx=lon_idx,
                        t_row=t_row,
                        t_col=t_col,
                        generation=generation,
                    )
                )

                flags: dict[str, int] = {}
                for c in ("crain", "csnow", "cfrzr", "cicep"):
                    if c in dataset.data_vars:
                        c_val = float(
                            reader.interpolate_point(
                                c,
                                member=member_num,
                                lead_time_hours=lead,
                                lat_idx=lat_idx,
                                lon_idx=lon_idx,
                                t_row=t_row,
                                t_col=t_col,
                                generation=generation,
                            )
                        )
                        flags[c] = 1 if c_val >= 0.5 else 0

                amt_prev: float | None = None
                flags_prev: dict[str, int] | None = None

                if lead % 6 == 0 and lead > 0:
                    pred_lead = lead - 3
                    if pred_lead in leads_in_ds:
                        amt_prev = float(
                            reader.interpolate_point(
                                "precipitation_amount_3h",
                                member=member_num,
                                lead_time_hours=pred_lead,
                                lat_idx=lat_idx,
                                lon_idx=lon_idx,
                                t_row=t_row,
                                t_col=t_col,
                                generation=generation,
                            )
                        )
                        f_p = {}
                        for c in ("crain", "csnow", "cfrzr", "cicep"):
                            if c in dataset.data_vars:
                                f_p_val = float(
                                    reader.interpolate_point(
                                        c,
                                        member=member_num,
                                        lead_time_hours=pred_lead,
                                        lat_idx=lat_idx,
                                        lon_idx=lon_idx,
                                        t_row=t_row,
                                        t_col=t_col,
                                        generation=generation,
                                    )
                                )
                                f_p[c] = 1 if f_p_val >= 0.5 else 0
                        if f_p:
                            flags_prev = f_p

                state = classify_precipitation_phase(
                    amt_val,
                    flags if flags else None,
                    amount_prev=amt_prev,
                    flags_prev=flags_prev,
                )
                return amt_val, state

            pairs_s = list(ex.map(_read_precip_member, target_members))
            return [p[0] for p in pairs_s], [p[1] for p in pairs_s]

        if "lead_time_hours" in field_amt.dims:
            field_amt = field_amt.sel(lead_time_hours=lead)

        cat_fields = {}
        for c in ("crain", "csnow", "cfrzr", "cicep"):
            if c in dataset.data_vars:
                f = dataset[c]
                if "lead_time_hours" in f.dims:
                    f = f.sel(lead_time_hours=lead)
                cat_fields[c] = f

        # Predecessor fields for 6-hour reset leads (t=6, 12, 18, 24, ...)
        pred_fields_avail = False
        pred_field_amt = None
        pred_cat_fields = {}

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

        amounts: list[float] = []
        states: list[PrecipitationPhaseState] = []

        for member_num in target_members:
            member_pos = member_val_to_pos[member_num]
            amt_val = float(
                _interpolate_neighborhood(
                    field_amt.isel(member=member_pos),
                    grid,
                    lat_descending,
                    lon_descending,
                    latitude,
                    longitude,
                )
            )
            amounts.append(amt_val)

            flags_l: dict[str, int] = {}
            for c, c_field in cat_fields.items():
                c_val = float(
                    _interpolate_neighborhood(
                        c_field.isel(member=member_pos),
                        grid,
                        lat_descending,
                        lon_descending,
                        latitude,
                        longitude,
                    )
                )
                flags_l[c] = 1 if c_val >= 0.5 else 0

            amt_prev_l: float | None = None
            flags_prev_l: dict[str, int] | None = None

            if pred_fields_avail and pred_field_amt is not None:
                amt_prev_l = float(
                    _interpolate_neighborhood(
                        pred_field_amt.isel(member=member_pos),
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
                                c_p_field.isel(member=member_pos),
                                grid,
                                lat_descending,
                                lon_descending,
                                latitude,
                                longitude,
                            )
                        )
                        f_p[c] = 1 if c_p_val >= 0.5 else 0
                    flags_prev_l = f_p

            st = classify_precipitation_phase(
                amt_val,
                flags_l if flags_l else None,
                amount_prev=amt_prev_l,
                flags_prev=flags_prev_l,
            )
            states.append(st)

        return amounts, states

    return gated_read_dataset_with_selector(store_path, select_and_classify)
