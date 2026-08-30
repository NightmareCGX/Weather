"""Forecast availability discovery: what forecast data the platform can serve.

This service answers "what forecasts actually exist" by querying the
PostgreSQL catalog — the single source of truth for available models,
variables, initial times, and lead times. Nothing here is hard-coded: the
nested ``model -> variable -> initial_time -> lead_time`` structure is built
from real ``model_runs`` (``status='ready'``) joined to ``forecast_products``
(one row per variable x lead time) and the ``forecast_variables`` /
``models`` catalogs.

The router stays thin (ENGINEERING_CONTRACT section 2): it validates query
parameters and serializes the structure this service builds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models.entities import (
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.schemas import (
    ForecastAvailabilityData,
    InitialTimeAvailability,
    LayerDescriptor,
    ModelAvailability,
    SpatialLayerLegend,
    VariableAvailability,
)
from api.services.tiles import MAX_ZOOM, MIN_ZOOM, _color_stops


#: Internal platform variables that are not exposed as public user-facing products.
INTERNAL_VARIABLES: frozenset[str] = frozenset({"wind_u_10m", "wind_v_10m"})


@dataclass
class _VariableAccumulator:
    """Accumulator of one variable's availability rows during the query.

    Attributes:
        name: Human-readable variable name.
        unit: Registered SI unit string.
        initial_times: Map of cycle time to the set of available lead hours.
    """

    name: str
    unit: str
    initial_times: dict[datetime, set[int]] = field(default_factory=dict)


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

    Only models with at least one ``ready`` run that has ``forecast_products``
    rows are included, so the returned structure reflects exactly what the
    serving tier can serve (a model row with no ready run or no products is
    not advertised). ``forecast_products`` is authoritative for the
    model/variable/lead combinations, so the availability always matches the
    catalog rows written by ingestion.

    Args:
        db: Database session.

    Returns:
        The availability payload, with models ordered by model id, variables
        by variable code, initial times newest-first, and lead times
        ascending.
    """
    rows = db.execute(
        select(
            Model.model_id,
            Model.name,
            Model.is_ensemble,
            ForecastVariable.variable_code,
            ForecastVariable.name,
            ForecastVariable.unit,
            ModelRun.cycle_time,
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
        .where(ModelRun.status == "ready")
    ).all()

    by_model: dict[str, _ModelAccumulator] = {}
    for (
        model_id,
        model_name,
        is_ensemble,
        variable_code,
        variable_name,
        variable_unit,
        cycle_time,
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
        variable_acc.initial_times.setdefault(cycle_time, set()).add(lead)

    models: list[ModelAvailability] = []
    for model_id in sorted(by_model):
        model_acc = by_model[model_id]
        variables: list[VariableAvailability] = []
        # Synthesize public wind_10m product when both wind_u_10m and wind_v_10m exist
        if "wind_u_10m" in model_acc.variables and "wind_v_10m" in model_acc.variables:
            u_acc = model_acc.variables["wind_u_10m"]
            v_acc = model_acc.variables["wind_v_10m"]
            common_cycles = set(u_acc.initial_times.keys()).intersection(v_acc.initial_times.keys())
            initial_times_wind: list[InitialTimeAvailability] = []
            for cycle_time in sorted(common_cycles, reverse=True):
                common_leads = u_acc.initial_times[cycle_time].intersection(v_acc.initial_times[cycle_time])
                if common_leads:
                    initial_times_wind.append(
                        InitialTimeAvailability(
                            value=cycle_time,
                            lead_time_hours=sorted(common_leads),
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
            initial_times = [
                InitialTimeAvailability(
                    value=cycle_time,
                    lead_time_hours=sorted(leads),
                )
                for cycle_time, leads in sorted(
                    variable_acc.initial_times.items(),
                    key=lambda item: item[0],
                    reverse=True,
                )
            ]
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
