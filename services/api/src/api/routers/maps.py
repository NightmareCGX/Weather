"""Spatial layer (map) endpoints: metadata and raster tiles (API.md 4.1).

Two endpoints are provided:

* ``GET /v1/maps`` — tile URL template and legend configuration for a weather
  map layer. This is the documented metadata endpoint.
* ``GET /v1/maps/{model}/{variable}/{level}/{z}/{x}/{y}.png`` — the actual
  raster tile image, served from the run's Zarr store. This is the tile
  endpoint the metadata template points at; it renders real forecast values
  (no fake data) via :mod:`api.services.tiles`.

The metadata endpoint validates that the requested combination actually
exists in the catalog (a ready run with a matching ``forecast_products`` row
for the model/variable/level/lead/initial-time), so a client can never build a
template for data that does not exist. The legend is per-variable: the tile
renderer's color ramp is reported so the legend always matches the tiles.

The router is thin (ENGINEERING_CONTRACT section 2): it validates parameters,
calls the availability/tile services, and serializes responses. No weather
calculations live in the handler.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import Response as StarletteResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.models.entities import ForecastVariable, Model
from api.schemas import (
    SpatialLayerData,
    SpatialLayerEnvelope,
    SpatialLayerLegend,
)
from api.services.tiles import (
    MAX_ZOOM,
    MIN_ZOOM,
    _color_stops,
    render_tile_png,
)

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for spatial layer metadata (API.md 4.1: 60 minutes).
CACHE_CONTROL_MAPS = "public, max-age=3600"
#: Cache policy for rendered tile images (short, so new runs appear promptly).
CACHE_CONTROL_TILE = "public, max-age=300"


def _legend_stops(variable: str) -> list[list[float | str]]:
    """Build the legend stops for a variable, matching the tile color ramp.

    The first stop's color is the "no data / below range" color used by the
    tile renderer for the low end, and the last stop is the high end. The
    stops are returned as the ``[value, color]`` pairs the legend renders.
    """
    return [
        [float(value), f"#{red:02x}{green:02x}{blue:02x}"]
        for value, (red, green, blue) in _color_stops(variable)
    ]


def _legend_unit(db: Session, variable: str) -> str:
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
    initial_time: Annotated[
        str | None,
        Query(
            description=(
                "Optional ISO 8601 UTC cycle time pinning the model run "
                "(e.g. 2026-08-13T00:00:00Z)."
            )
        ),
    ] = None,
    db: Session = DB,
) -> SpatialLayerEnvelope:
    """Return the tile template and legend for a weather map layer.

    The model and variable must exist in the catalog, and a ready run with a
    matching forecast product must be available for the requested
    model/variable/level/lead/initial-time combination; otherwise 404 is
    raised so the client never builds a template for data that does not exist.
    """
    _require_model(db, model)
    unit = _legend_unit(db, variable)
    _require_available(db, model, variable, level, lead_time_hours, initial_time)


    template_path = (
        f"/v1/maps/{model}/{variable}/{level}/{{z}}/{{x}}/{{y}}.png"
        f"?lead_time_hours={lead_time_hours}"
    )
    if initial_time is not None:
        template_path += f"&initial_time={initial_time}"

    data = SpatialLayerData(
        tile_url_template=template_path,
        min_zoom=MIN_ZOOM,
        max_zoom=MAX_ZOOM,
        lead_time_hours=lead_time_hours,
        legend=SpatialLayerLegend(unit=unit, stops=_legend_stops(variable)),
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_MAPS
    return SpatialLayerEnvelope(data=data)


@router.get(
    "/maps/{model}/{variable}/{level}/{z}/{x}/{y}.png",
    response_class=StarletteResponse,
    summary="Render a forecast raster tile",
)
def get_map_tile(
    model: str,
    variable: str,
    level: str,
    z: int,
    x: int,
    y: int,
    lead_time_hours: Annotated[
        int, Query(ge=0, description="Forecast offset hours from cycle time.")
    ],
    initial_time: Annotated[
        str | None,
        Query(
            description=(
                "Optional ISO 8601 UTC cycle time pinning the model run."
            )
        ),
    ] = None,
    db: Session = DB,
) -> StarletteResponse:
    """Render a 256x256 PNG tile of the forecast field for the selection.

    The tile reads genuine values from the run's Zarr store for the requested
    model/variable/initial time/lead time and colors them by the variable's
    ramp. Tiles are served with a short cache policy so newly ingested runs
    become visible promptly.
    """
    try:
        png = render_tile_png(
            db,
            model=model,
            variable=variable,
            level=level,
            zoom=z,
            x=x,
            y=y,
            lead_time_hours=lead_time_hours,
            initial_time=initial_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response = StarletteResponse(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": CACHE_CONTROL_TILE},
    )
    return response


def _require_model(db: Session, model: str) -> None:
    """Raise 404 if the model identifier is not in the catalog."""
    found = db.execute(
        select(Model.model_id).where(Model.model_id == model)
    ).scalar_one_or_none()
    if found is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{model}' was not found."
        )


def _require_available(
    db: Session,
    model: str,
    variable: str,
    level: str,
    lead_time_hours: int,
    initial_time: str | None,
) -> None:
    """Raise 404 if no ready run + product matches the selection.

    This makes the metadata endpoint honest: it only returns a template for
    combinations that actually exist in the catalog. Availability is derived
    entirely from real ``model_runs``/``forecast_products`` rows (a DB-only
    check; no Zarr store is opened for a metadata request).
    """
    from api.services.tiles import check_available

    check_available(
        db,
        model=model,
        variable=variable,
        level=level,
        lead_time_hours=lead_time_hours,
        initial_time=initial_time,
    )
