"""Spatial layer (map tile metadata) endpoint (API.md section 4.1).

Returns tile URL templates and legend configuration for weather map
visualization. This is a metadata-only endpoint: it validates that the model
and variable exist in the catalog and returns a self-referential tile template
plus a legend. Map tile generation/serving is out of scope for Milestone 10
(API.md section 4.1 returns metadata, not tiles). The router is thin
(ENGINEERING_CONTRACT section 2) and follows the catalog endpoint pattern --
no weather calculations or Redis caching (the response is static metadata).
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.models.entities import ForecastVariable, Model
from api.schemas import (
    SpatialLayerData,
    SpatialLayerEnvelope,
    SpatialLayerLegend,
)

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for spatial layer metadata (API.md 4.1: 60 minutes).
CACHE_CONTROL_MAPS = "public, max-age=3600"

#: Minimum Web-Mercator zoom level of the tile template.
MIN_ZOOM = 0
#: Maximum Web-Mercator zoom level of the tile template.
MAX_ZOOM = 9

#: Default legend color ramp matching the API.md section 4.1 example. The
#: approved design does not define a color scheme; this fixed ramp is a
#: recorded stand-in until one is approved.
DEFAULT_LEGEND_STOPS: list[list[float | str]] = [
    [-40, "#0000ff"],
    [0, "#00ff00"],
    [40, "#ff0000"],
]


@router.get(
    "/maps",
    response_model=SpatialLayerEnvelope,
    summary="Get map tile metadata",
)
def get_spatial_layer(
    response: Response,
    model: Annotated[str, Query(description="A model identifier.")],
    variable: Annotated[str, Query(description="A forecast variable code.")],
    level: Annotated[
        Literal["surface"],
        Query(description="The vertical level; only 'surface' is supported."),
    ],
    lead_time_hours: Annotated[
        int, Query(ge=0, description="Forecast offset hours from cycle time.")
    ],
    db: Session = DB,
) -> SpatialLayerEnvelope:
    """Return the tile template and legend for a weather map layer.

    The model and variable must exist in the catalog; any ``level`` other than
    ``surface`` is rejected by the ``Literal`` validation.
    """
    _require_model(db, model)
    unit = _require_variable_unit(db, variable)

    tile_url_template = (
        f"/v1/maps/{model}/{variable}/{level}/"
        f"{{z}}/{{x}}/{{y}}.png?lead_time_hours={lead_time_hours}"
    )
    data = SpatialLayerData(
        tile_url_template=tile_url_template,
        min_zoom=MIN_ZOOM,
        max_zoom=MAX_ZOOM,
        lead_time_hours=lead_time_hours,
        legend=SpatialLayerLegend(unit=unit, stops=DEFAULT_LEGEND_STOPS),
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_MAPS
    return SpatialLayerEnvelope(data=data)


def _require_model(db: Session, model: str) -> None:
    """Raise 404 if the model identifier is not in the catalog."""
    found = db.execute(
        select(Model.model_id).where(Model.model_id == model)
    ).scalar_one_or_none()
    if found is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{model}' was not found."
        )


def _require_variable_unit(db: Session, variable: str) -> str:
    """Return the variable's registered unit, raising 404 if unknown."""
    unit = db.execute(
        select(ForecastVariable.unit).where(
            ForecastVariable.variable_code == variable
        )
    ).scalar_one_or_none()
    if unit is None:
        raise HTTPException(
            status_code=404, detail=f"Variable '{variable}' was not found."
        )
    return unit
