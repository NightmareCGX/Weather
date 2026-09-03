"""Forecast availability discovery: what forecast data the platform can serve.

This service answers "what forecasts actually exist and are servable" by querying
the PostgreSQL catalog — the single source of truth for available models,
variables, initial times, lead times, and ensemble member coverage. Nothing here
is hard-coded: the nested ``model -> variable -> initial_time -> lead_time``
structure is built from real ``model_runs`` in ``ready``, ``processing``, or
``partial`` lifecycle states joined to ``forecast_products`` (one row per
variable x lead time), ``ensemble_member_products`` (committed member pairs),
and the ``forecast_variables`` / ``models`` catalogs. Failed runs are strictly
excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from domain.coverage import (
    compute_coverage_ratio,
    get_expected_members,
    is_lead_servable,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.models.entities import (
    EnsembleMemberProduct,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.lifecycle import filter_visible_runs
from api.schemas import (
    ForecastAvailabilityData,
    InitialTimeAvailability,
    LayerDescriptor,
    LeadAvailabilityOut,
    ModelAvailability,
    SpatialLayerLegend,
    VariableAvailability,
)
from api.services.tiles import MAX_ZOOM, MIN_ZOOM, _color_stops

#: Internal platform variables that are not exposed as public user-facing products.
INTERNAL_VARIABLES: frozenset[str] = frozenset(
    {"wind_u_10m", "wind_v_10m", "crain", "csnow", "cfrzr", "cicep"}
)

#: Valid model run lifecycle statuses eligible for availability discovery.
SERVING_ELIGIBLE_STATUSES: frozenset[str] = frozenset({"ready", "processing", "partial"})


@dataclass
class _CycleInfo:
    """Tracking structure for a single cycle run's available leads."""

    run_id: str
    status: str
    leads: set[int] = field(default_factory=set)


@dataclass
class _VariableAccumulator:
    """Accumulator of one variable's availability rows during the query.

    Attributes:
        name: Human-readable variable name.
        unit: Registered SI unit string.
        initial_times: Map of cycle time to cycle info.
    """

    name: str
    unit: str
    initial_times: dict[datetime, _CycleInfo] = field(default_factory=dict)


@dataclass
class _ModelAccumulator:
    """Accumulator of one model's availability rows during the query.

    Attributes:
        name: Human-readable model name.
        is_ensemble: Whether the model is an ensemble product.
        variables: Map of variable code to its accumulator.
    """

    name: str
    is_ensemble: bool
    variables: dict[str, _VariableAccumulator] = field(default_factory=dict)


def build_forecast_availability(db: Session) -> ForecastAvailabilityData:
    """Build the nested model/variable/initial-time/lead-time availability.

    Queries runs in ``ready``, ``processing``, and ``partial`` statuses that
    have committed ``forecast_products`` rows. Enforces the 85% member-coverage
    serving threshold for the simple ``lead_time_hours`` list while exposing rich
    per-lead coverage descriptors in ``leads``.

    Args:
        db: Database session.

    Returns:
        The availability payload, with models ordered by model id, variables
        by variable code, initial times newest-first, and lead times ascending.
    """
    stmt = (
        select(
            Model.model_id,
            Model.name,
            Model.is_ensemble,
            ForecastVariable.variable_code,
            ForecastVariable.name,
            ForecastVariable.unit,
            ModelRun.id,
            ModelRun.cycle_time,
            ModelRun.status,
            ForecastProduct.lead_time_hours,
        )
        .select_from(ModelRun)
        .join(ModelVersion, ModelRun.model_version_id == ModelVersion.id)
        .join(Model, ModelVersion.model_id == Model.model_id)
        .join(ForecastProduct, ForecastProduct.run_id == ModelRun.id)
        .join(
            ForecastVariable,
            ForecastVariable.variable_code == ForecastProduct.variable_id,
        )
        .where(ModelRun.status.in_(SERVING_ELIGIBLE_STATUSES))
    )
    rows = db.execute(filter_visible_runs(stmt)).all()

    # Pre-query committed ensemble member counts per (run_id, lead_time_hours)
    emp_rows = db.execute(
        select(
            EnsembleMemberProduct.run_id,
            EnsembleMemberProduct.lead_time_hours,
            func.count(EnsembleMemberProduct.member_index),
        ).group_by(
            EnsembleMemberProduct.run_id,
            EnsembleMemberProduct.lead_time_hours,
        )
    ).all()
    emp_counts: dict[tuple[str, int], int] = {
        (str(r_id), int(lead)): int(cnt) for r_id, lead, cnt in emp_rows
    }

    by_model: dict[str, _ModelAccumulator] = {}
    for (
        model_id,
        model_name,
        is_ensemble,
        variable_code,
        variable_name,
        variable_unit,
        run_id,
        cycle_time,
        run_status,
        lead,
    ) in rows:
        model_acc = by_model.setdefault(
            model_id,
            _ModelAccumulator(name=model_name, is_ensemble=is_ensemble),
        )
        variable_acc = model_acc.variables.setdefault(
            variable_code,
            _VariableAccumulator(name=variable_name, unit=variable_unit),
        )
        cycle_info = variable_acc.initial_times.setdefault(
            cycle_time,
            _CycleInfo(run_id=run_id, status=run_status),
        )
        cycle_info.leads.add(int(lead))

    models: list[ModelAvailability] = []
    for model_id in sorted(by_model):
        model_acc = by_model[model_id]
        expected_members = get_expected_members(
            model_id, default_if_unknown=30 if model_acc.is_ensemble else 1
        )
        variables: list[VariableAvailability] = []

        # Synthesize public wind_10m product when both wind_u_10m and wind_v_10m exist
        if "wind_u_10m" in model_acc.variables and "wind_v_10m" in model_acc.variables:
            u_acc = model_acc.variables["wind_u_10m"]
            v_acc = model_acc.variables["wind_v_10m"]
            common_cycles = set(u_acc.initial_times.keys()).intersection(v_acc.initial_times.keys())
            initial_times_wind: list[InitialTimeAvailability] = []
            for cycle_time in sorted(common_cycles, reverse=True):
                u_info = u_acc.initial_times[cycle_time]
                v_info = v_acc.initial_times[cycle_time]
                common_leads = u_info.leads.intersection(v_info.leads)
                if common_leads:
                    servable_leads: list[int] = []
                    rich_leads: list[LeadAvailabilityOut] = []
                    for lead in sorted(common_leads):
                        if model_acc.is_ensemble:
                            avail_u = emp_counts.get((u_info.run_id, lead), 0)
                            avail_v = emp_counts.get((v_info.run_id, lead), 0)
                            avail_count = min(avail_u, avail_v)
                        else:
                            avail_count = 1
                        ratio = compute_coverage_ratio(avail_count, expected_members)
                        servable = is_lead_servable(avail_count, expected_members)
                        if servable:
                            servable_leads.append(lead)
                        rich_leads.append(
                            LeadAvailabilityOut(
                                lead_time_hours=lead,
                                available_members=avail_count,
                                expected_members=expected_members,
                                coverage_ratio=ratio,
                                servable=servable,
                            )
                        )
                    initial_times_wind.append(
                        InitialTimeAvailability(
                            value=cycle_time,
                            lead_time_hours=servable_leads,
                            status=u_info.status,
                            leads=rich_leads,
                        )
                    )
            if initial_times_wind:
                stops_wind: list[list[float | str]] = [
                    [float(value), f"#{red:02x}{green:02x}{blue:02x}"]
                    for value, (red, green, blue) in _color_stops("wind_10m")
                ]
                layer_wind = LayerDescriptor(
                    tile_url_template=(
                        f"/v1/maps/{model_id}/wind_10m/surface/{{z}}/{{x}}/{{y}}.png"
                        f"?lead_time_hours={{lead_time_hours}}&initial_time={{initial_time}}"
                    ),
                    min_zoom=MIN_ZOOM,
                    max_zoom=MAX_ZOOM,
                    legend=SpatialLayerLegend(unit="km/h", stops=stops_wind),
                    vector_field_url_template=(
                        f"/v1/maps/{model_id}/wind_10m/vector-field"
                        f"?lead_time_hours={{lead_time_hours}}&initial_time={{initial_time}}"
                    ),
                )
                variables.append(
                    VariableAvailability(
                        id="wind_10m",
                        name="10-Meter Wind",
                        unit="km/h",
                        initial_times=initial_times_wind,
                        layer=layer_wind,
                    )
                )

        for variable_code in sorted(model_acc.variables):
            if variable_code in INTERNAL_VARIABLES:
                continue
            variable_acc = model_acc.variables[variable_code]
            initial_times: list[InitialTimeAvailability] = []
            for cycle_time, cycle_info in sorted(
                variable_acc.initial_times.items(),
                key=lambda item: item[0],
                reverse=True,
            ):
                servable_leads = []
                rich_leads = []
                for lead in sorted(cycle_info.leads):
                    if model_acc.is_ensemble:
                        avail_count = emp_counts.get((cycle_info.run_id, lead), 0)
                    else:
                        avail_count = 1
                    ratio = compute_coverage_ratio(avail_count, expected_members)
                    servable = is_lead_servable(avail_count, expected_members)
                    if servable:
                        servable_leads.append(lead)
                    rich_leads.append(
                        LeadAvailabilityOut(
                            lead_time_hours=lead,
                            available_members=avail_count,
                            expected_members=expected_members,
                            coverage_ratio=ratio,
                            servable=servable,
                        )
                    )
                initial_times.append(
                    InitialTimeAvailability(
                        value=cycle_time,
                        lead_time_hours=servable_leads,
                        status=cycle_info.status,
                        leads=rich_leads,
                    )
                )

            stops: list[list[float | str]] = [
                [float(value), f"#{red:02x}{green:02x}{blue:02x}"]
                for value, (red, green, blue) in _color_stops(variable_code)
            ]
            layer = LayerDescriptor(
                tile_url_template=(
                    f"/v1/maps/{model_id}/{variable_code}/surface/{{z}}/{{x}}/{{y}}.png"
                    f"?lead_time_hours={{lead_time_hours}}&initial_time={{initial_time}}"
                ),
                min_zoom=MIN_ZOOM,
                max_zoom=MAX_ZOOM,
                legend=SpatialLayerLegend(unit=variable_acc.unit, stops=stops),
            )
            variables.append(
                VariableAvailability(
                    id=variable_code,
                    name=variable_acc.name,
                    unit=variable_acc.unit,
                    initial_times=initial_times,
                    layer=layer,
                )
            )
        variables.sort(key=lambda v: v.id)
        models.append(
            ModelAvailability(
                id=model_id,
                name=model_acc.name,
                is_ensemble=model_acc.is_ensemble,
                variables=variables,
            )
        )

    return ForecastAvailabilityData(models=models)
