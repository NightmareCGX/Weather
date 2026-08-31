"""Pydantic response schemas for the API service.

These models reproduce the response envelope and resource shapes defined in
``docs/API.md`` exactly. The API layer serializes ORM rows into these schemas;
no weather calculations or database access live here.
"""

from datetime import date, datetime, timezone
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, field_serializer, model_serializer


def format_datetime_utc(value: datetime) -> str:
    """Format a datetime as an ISO 8601 UTC string with a ``Z`` suffix.

    API.md section 2.6 requires absolute timestamps as ISO 8601 UTC strings
    (e.g. ``2026-07-21T00:00:00Z``). Pydantic's default serialization emits a
    ``+00:00`` offset, so datetimes are normalized here.

    Args:
        value: The datetime to format.

    Returns:
        The ISO 8601 UTC string with a trailing ``Z``.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


class CenterOut(BaseModel):
    """A forecast center resource (API.md section 1.1)."""

    id: str
    object: Literal["center"] = "center"
    name: str
    country: str


class ModelOut(BaseModel):
    """A weather model resource (API.md section 1.2)."""

    id: str
    object: Literal["model"] = "model"
    name: str
    center_id: str
    is_ensemble: bool
    resolution_km: float


class RunOut(BaseModel):
    """A model run resource (API.md section 1.3)."""

    id: str
    object: Literal["run"] = "run"
    model_id: str
    cycle_time: datetime
    status: str

    @field_serializer("cycle_time")
    def _serialize_cycle_time(self, value: datetime) -> str:
        return format_datetime_utc(value)


class VariableOut(BaseModel):
    """A forecast variable resource (API.md section 1.4)."""

    id: str
    object: Literal["variable"] = "variable"
    name: str
    unit: str


class GridOut(BaseModel):
    """A forecast grid resource (API.md section 1.5)."""

    id: str
    object: Literal["grid"] = "grid"
    name: str
    resolution_km: float


T = TypeVar("T", bound=BaseModel)


class ListEnvelope(BaseModel, Generic[T]):
    """The universal list response envelope (API.md section 2.3)."""

    object: Literal["list"] = "list"
    data: list[T]
    has_more: bool = False
    next_cursor: str | None = None


class SpatialLayerLegend(BaseModel):
    """The legend of a spatial layer (API.md section 4.1)."""

    unit: str
    stops: list[list[float | str]]


class LayerDescriptor(BaseModel):
    """The spatial layer descriptor for rendering map tiles and legends."""

    tile_url_template: str
    min_zoom: int
    max_zoom: int
    legend: SpatialLayerLegend
    vector_field_url_template: str | None = None


class InitialTimeAvailability(BaseModel):
    """One available forecast initialization (cycle time) of a variable.

    Attributes:
        value: The run's ``cycle_time`` in ISO 8601 UTC. ``valid_time`` for a
            given lead is derived as ``value + lead_time_hours``
            (DATABASE.md section 1).
        lead_time_hours: The forecast offset hours available for this
            model/variable/initial time, ascending.
    """

    value: datetime
    lead_time_hours: list[int]

    @field_serializer("value")
    def _serialize_value(self, value: datetime) -> str:
        return format_datetime_utc(value)


class VariableAvailability(BaseModel):
    """A forecast variable and the initial times available for it.

    Attributes:
        id: The ``forecast_variables.variable_code`` (e.g.
            ``temperature_2m``).
        name: Human-readable variable name.
        unit: The registered SI unit string (e.g. ``°C``).
        initial_times: The initial times with data for this
            model/variable, newest first.
        layer: Authoritative map layer descriptor for rendering tiles and
            legends without an extra metadata roundtrip.
    """

    id: str
    name: str
    unit: str
    initial_times: list[InitialTimeAvailability]
    layer: LayerDescriptor | None = None


class ModelAvailability(BaseModel):
    """A forecast model and the variables available for it.

    Attributes:
        id: The ``models.model_id`` (e.g. ``gfs``).
        name: Human-readable model name.
        is_ensemble: Whether the model is an ensemble product.
        variables: The variables with ready forecast data for this model,
            ordered by variable code.
    """

    id: str
    name: str
    is_ensemble: bool
    variables: list[VariableAvailability]


class ForecastAvailabilityData(BaseModel):
    """The payload of the forecast availability endpoint.

    Models are ordered by model id; only models that have at least one
    ``ready`` run with forecast product rows are included, so the list
    reflects exactly what the platform can serve.
    """

    models: list[ModelAvailability]


class ForecastAvailabilityEnvelope(BaseModel):
    """The forecast availability response envelope.

    ``object`` is ``forecast_availability`` (a new resource shape that is a
    non-breaking addition to the v1 surface per API.md section 1.3).
    """

    object: Literal["forecast_availability"] = "forecast_availability"
    data: ForecastAvailabilityData
    has_more: bool = False
    next_cursor: str | None = None


class ErrorDetail(BaseModel):
    """The RFC 7807-style error body (API.md section 2.4)."""

    code: str
    type: str
    message: str
    param: str | None = None
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    """The wrapper around :class:`ErrorDetail` returned on failure."""

    error: ErrorDetail


class ForecastLocationOut(BaseModel):
    """The resolved location of a point forecast (API.md section 2.1)."""

    latitude: float
    longitude: float
    elevation_m: float | None = None
    resolved_via: str


class ForecastSeries(BaseModel):
    """A single forecast entry indexed by lead time (API.md section 2.1).

    Requested forecast variables are attached as additional top-level keys
    (e.g. ``temperature_2m``), matching the dynamic keyed shape shown in
    API.md. Variable keys are captured through ``extra="allow"``.

    ``cycle_time`` is the source forecast run's cycle/reference time. The point
    forecast is a **cross-cycle** deterministic time series: entries may come
    from different cycles (the minimum-lead record for each valid_time), so the
    source cycle is exposed to make the provenance unambiguous. It is additive
    and non-breaking.
    """

    lead_time_hours: int
    valid_time: datetime
    cycle_time: datetime | None = None

    model_config = ConfigDict(extra="allow")

    @field_serializer("valid_time")
    def _serialize_valid_time(self, value: datetime) -> str:
        return format_datetime_utc(value)

    @field_serializer("cycle_time")
    def _serialize_cycle_time(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return format_datetime_utc(value)


class PointForecastData(BaseModel):
    """The payload of a point forecast (API.md section 2.1).

    ``generated_at`` is the forecast dataset generation time -- the selected
    model run's ``cycle_time`` -- so that identical forecast data produces
    identical payloads and deterministic cache behavior.
    """

    location: ForecastLocationOut
    generated_at: datetime
    model: str
    forecasts: list[ForecastSeries]

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return format_datetime_utc(value)


class PointForecastEnvelope(BaseModel):
    """The point forecast response envelope (API.md sections 2.1 and 2.3).

    Only the documented single-model ``point_forecast`` envelope is
    implemented in Milestone 9. The multi-model response format is not
    defined by the approved design documents and is therefore a recorded
    specification gap; multi-model requests are rejected (see the
    ``/v1/points`` router).
    """

    object: Literal["point_forecast"] = "point_forecast"
    data: PointForecastData
    has_more: bool = False
    next_cursor: str | None = None


class SearchResultOut(BaseModel):
    """A location search result (API.md section 6.1).

    The ``object`` field distinguishes the source table: ``city``,
    ``ski_resort``, or ``station``. Type-specific fields (``region``,
    ``country``, ``elevation_m``) are optional because not every source
    table defines them.
    """

    id: str
    object: str
    name: str
    region: str | None = None
    country: str | None = None
    elevation_m: float | None = None
    latitude: float
    longitude: float
    place_id: str | None = None


class ProbabilityLocation(BaseModel):
    """The location of a probability forecast (API.md section 3.1)."""

    latitude: float
    longitude: float


class ProbabilityForecastData(BaseModel):
    """The payload of a probability forecast (API.md section 3.1).

    ``threshold_max`` is the upper bound of the ``between`` operator. It is
    serialized only when the operator is ``between``, so ``gt``/``lt``
    payloads contain exactly the documented fields and never a ``null``
    ``threshold_max``.
    """

    location: ProbabilityLocation
    variable: str
    threshold: float
    operator: str
    lead_time_hours: int
    probability: float
    confidence_interval_95: list[float]
    threshold_max: float | None = None
    direction_sector: str | None = None
    phase: str | None = None

    @model_serializer
    def _serialize_threshold_max(self) -> dict[str, object]:
        """Omit ``threshold_max`` unless the operator is ``between``."""
        payload: dict[str, object] = {
            "location": self.location,
            "variable": self.variable,
            "threshold": self.threshold,
            "operator": self.operator,
            "lead_time_hours": self.lead_time_hours,
            "probability": self.probability,
            "confidence_interval_95": self.confidence_interval_95,
        }
        if self.operator == "between" and self.threshold_max is not None:
            payload["threshold_max"] = self.threshold_max
        if self.direction_sector is not None:
            payload["direction_sector"] = self.direction_sector
        if self.phase is not None:
            payload["phase"] = self.phase
        return payload


class ProbabilityForecastEnvelope(BaseModel):
    """The probability forecast response envelope (API.md sections 3.1 and 2.3)."""

    object: Literal["probability_forecast"] = "probability_forecast"
    data: ProbabilityForecastData
    has_more: bool = False
    next_cursor: str | None = None


class SpatialLayerData(BaseModel):
    """The payload of a spatial layer (API.md section 4.1).

    ``tile_url_template`` is a self-referential template for a future tile
    endpoint; map tile generation is out of scope for Milestone 10.
    """

    tile_url_template: str
    min_zoom: int
    max_zoom: int
    lead_time_hours: int
    legend: SpatialLayerLegend
    vector_field_url_template: str | None = None


class SpatialLayerEnvelope(BaseModel):
    """The spatial layer response envelope (API.md sections 4.1 and 2.3)."""

    object: Literal["spatial_layer"] = "spatial_layer"
    data: SpatialLayerData
    has_more: bool = False
    next_cursor: str | None = None


class EnsembleStatistics(BaseModel):
    """Ensemble dispersion statistics (API.md section 5.1)."""

    mean: float | None = None
    median: float | None = None
    spread: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None


class EnsemblePDF(BaseModel):
    """Ensemble probability density function estimate."""

    x: list[float]
    density: list[float]


class ConsensusVectorOut(BaseModel):
    """Ensemble consensus vector flow metrics for wind products."""

    speed: float
    direction: float | None = None
    cardinal: str
    coherence: float


class WindRoseSectorOut(BaseModel):
    """One 45-degree directional sector in an ensemble Wind Rose."""

    sector: str
    count: int
    probability: float
    bins: dict[str, float]


class WindRoseOut(BaseModel):
    """8-sector Wind Rose representing speed x direction ensemble distribution."""

    calm_percentage: float
    calm_count: int
    sectors: list[WindRoseSectorOut]


class EnsembleStatisticsData(BaseModel):
    """The payload of ensemble statistics (API.md section 5.1).

    ``members`` carries the raw ensemble-member forecast values (in dataset
    ``member``-coordinate order) for the requested model, location, variable,
    and lead time. ``pdf`` carries the canonical 1-D Gaussian Kernel Density
    Estimate over the ensemble members (or ``null`` when variation across
    members is degenerate / std = 0). Both are opt-in fields returned only
    when the request sets ``include_members=true``; they are omitted on
    statistics-only responses (API.md section 5.1, additive opt-in extension).
    """

    model: str
    lead_time_hours: int
    member_count: int
    statistics: EnsembleStatistics
    members: list[float] | None = None
    pdf: EnsemblePDF | None = None
    consensus_vector: ConsensusVectorOut | None = None
    wind_rose: WindRoseOut | None = None
    phase_support: dict[str, float] | None = None
    transition_frequency: dict[str, float] | None = None
    valid_member_count: int | None = None
    unlimited_probability: float | None = None
    finite_member_count: int | None = None
    unlimited_member_count: int | None = None

    @model_serializer
    def _serialize_distribution_fields(self) -> dict[str, object]:
        """Omit ``members`` and ``pdf`` unless include_members=true."""
        payload: dict[str, object] = {
            "model": self.model,
            "lead_time_hours": self.lead_time_hours,
            "member_count": self.member_count,
            "statistics": self.statistics,
        }
        if self.members is not None:
            payload["members"] = self.members
            payload["pdf"] = self.pdf
        if self.consensus_vector is not None:
            payload["consensus_vector"] = self.consensus_vector
        if self.wind_rose is not None:
            payload["wind_rose"] = self.wind_rose
        if self.phase_support is not None:
            payload["phase_support"] = self.phase_support
        if self.transition_frequency is not None:
            payload["transition_frequency"] = self.transition_frequency
        if self.valid_member_count is not None:
            payload["valid_member_count"] = self.valid_member_count
        if self.unlimited_probability is not None:
            payload["unlimited_probability"] = self.unlimited_probability
        if self.finite_member_count is not None:
            payload["finite_member_count"] = self.finite_member_count
        if self.unlimited_member_count is not None:
            payload["unlimited_member_count"] = self.unlimited_member_count
        return payload


class EnsembleStatisticsEnvelope(BaseModel):
    """The ensemble statistics response envelope (API.md sections 5.1 and 2.3)."""

    object: Literal["ensemble_statistics"] = "ensemble_statistics"
    data: EnsembleStatisticsData
    has_more: bool = False
    next_cursor: str | None = None


class VerificationPeriod(BaseModel):
    """The verification date window (API.md section 7.1).

    ``start`` and ``end`` are inclusive calendar dates (the window spans
    ``[start 00:00:00Z, end + 1 day)``).
    """

    start: date
    end: date


class VerificationReportData(BaseModel):
    """The payload of a verification report (API.md section 7.1).

    ``metrics`` maps a flat per-variable key (``{variable}_rmse``,
    ``{variable}_bias``, ``{variable}_mae``) to its value. A variable with no
    valid forecast/observation pairs in the window contributes no keys, so
    ``metrics`` may be empty.
    """

    model: str
    period: VerificationPeriod
    metrics: dict[str, float]


class VerificationReportEnvelope(BaseModel):
    """The verification report response envelope (API.md sections 7.1 and 2.3)."""

    object: Literal["verification_report"] = "verification_report"
    data: VerificationReportData
    has_more: bool = False
    next_cursor: str | None = None


class HealthCheckData(BaseModel):
    """The payload of a system health check (API.md section 8.1).

    ``status`` is ``healthy`` when every dependency is connected and
    ``degraded`` otherwise. ``version`` is the API contract version.
    """

    status: str
    version: str
    database: str
    redis: str
    object_storage: str


class HealthCheckEnvelope(BaseModel):
    """The health check response envelope (API.md sections 8.1 and 2.3)."""

    object: Literal["health_check"] = "health_check"
    data: HealthCheckData
    has_more: bool = False
    next_cursor: str | None = None
