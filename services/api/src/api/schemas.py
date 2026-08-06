"""Pydantic response schemas for the API service.

These models reproduce the response envelope and resource shapes defined in
``docs/API.md`` exactly. The API layer serializes ORM rows into these schemas;
no weather calculations or database access live here.
"""

from datetime import datetime, timezone
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
    """

    lead_time_hours: int
    valid_time: datetime

    model_config = ConfigDict(extra="allow")

    @field_serializer("valid_time")
    def _serialize_valid_time(self, value: datetime) -> str:
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
    operator: Literal["gt", "lt", "between"]
    lead_time_hours: int
    probability: float
    confidence_interval_95: list[float]
    threshold_max: float | None = None

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
        return payload


class ProbabilityForecastEnvelope(BaseModel):
    """The probability forecast response envelope (API.md sections 3.1 and 2.3)."""

    object: Literal["probability_forecast"] = "probability_forecast"
    data: ProbabilityForecastData
    has_more: bool = False
    next_cursor: str | None = None


class SpatialLayerLegend(BaseModel):
    """The legend of a spatial layer (API.md section 4.1)."""

    unit: str
    stops: list[list[float | str]]


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


class SpatialLayerEnvelope(BaseModel):
    """The spatial layer response envelope (API.md sections 4.1 and 2.3)."""

    object: Literal["spatial_layer"] = "spatial_layer"
    data: SpatialLayerData
    has_more: bool = False
    next_cursor: str | None = None


class EnsembleStatistics(BaseModel):
    """Ensemble dispersion statistics (API.md section 5.1)."""

    mean: float
    median: float
    spread: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


class EnsembleStatisticsData(BaseModel):
    """The payload of ensemble statistics (API.md section 5.1)."""

    model: str
    lead_time_hours: int
    member_count: int
    statistics: EnsembleStatistics


class EnsembleStatisticsEnvelope(BaseModel):
    """The ensemble statistics response envelope (API.md sections 5.1 and 2.3)."""

    object: Literal["ensemble_statistics"] = "ensemble_statistics"
    data: EnsembleStatisticsData
    has_more: bool = False
    next_cursor: str | None = None
