"""Pydantic response schemas for the API service.

These models reproduce the response envelope and resource shapes defined in
``docs/API.md`` exactly. The API layer serializes ORM rows into these schemas;
no weather calculations or database access live here.
"""

from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, field_serializer


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
