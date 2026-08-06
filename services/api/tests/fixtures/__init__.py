"""Test fixtures for the API service.

Provides helpers to generate deterministic, tiny Zarr forecast datasets on
disk for the point-forecast integration tests, and to build PostGIS location
records (cities, ski resorts, stations) for the search and point-resolution
tests. The Zarr stores are written to a temporary directory at test time by
:func:`build_forecast_dataset` + the ``ingestion.core.zarr_writer`` module;
no binary fixtures are committed. See ``README.md`` for details.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from ingestion.core.zarr_writer import write_dataset

#: Fixture grid geometry (a regular 0.25 degree grid that fits the
#: deterministic dataset generated below).
LAT_START = 38.0
LAT_STEP = 0.25
LAT_ROWS = 4
LON_START = -107.0
LON_STEP = 0.25
LON_COLS = 4
#: Lead-time coordinate values used by the fixture datasets.
LEAD_TIMES = [0, 6, 12, 18]

#: Latitude values of the fixture grid.
LATITUDES = [LAT_START + i * LAT_STEP for i in range(LAT_ROWS)]
#: Longitude values of the fixture grid.
LONGITUDES = [LON_START + j * LON_STEP for j in range(LON_COLS)]


def build_forecast_dataset() -> xr.Dataset:
    """Build a deterministic forecast dataset with surface variables.

    The dataset contains ``temperature_2m`` and ``precipitation_rate`` data
    variables over the ``lead_time_hours``, ``latitude``, and ``longitude``
    dimensions. Values follow analytic fields so expected interpolation
    results are exact:

    * ``temperature_2m(lat, lon, lead) = 10 + 10*(lat - LAT_START)
      + 10*(lon - LON_START) + 0.5*lead``
    * ``precipitation_rate(lead) = 0.5*lead``

    Returns:
        The deterministic dataset.
    """
    lat = np.asarray(LATITUDES, dtype=float)
    lon = np.asarray(LONGITUDES, dtype=float)
    lead = np.asarray(LEAD_TIMES, dtype=float)
    lead_grid, lat_grid, lon_grid = np.meshgrid(lead, lat, lon, indexing="ij")

    temperature = (
        10.0
        + 10.0 * (lat_grid - LAT_START)
        + 10.0 * (lon_grid - LON_START)
        + 0.5 * lead_grid
    )
    precipitation = 0.5 * lead_grid

    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                temperature,
            ),
            "precipitation_rate": (
                ("lead_time_hours", "latitude", "longitude"),
                precipitation,
            ),
        },
        coords={
            "lead_time_hours": LEAD_TIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )


def write_forecast_zarr(store: str) -> str:
    """Write the deterministic fixture dataset to a local Zarr store.

    Args:
        store: Local directory path for the Zarr store.

    Returns:
        The store path (for reporting).
    """
    return write_dataset(build_forecast_dataset(), store)


def temperature_at(latitude: float, longitude: float, lead: int) -> float:
    """Expected metric temperature value at a point and lead time."""
    return (
        10.0
        + 10.0 * (latitude - LAT_START)
        + 10.0 * (longitude - LON_START)
        + 0.5 * lead
    )


def precipitation_at(lead: int) -> float:
    """Expected metric precipitation value at a lead time."""
    return 0.5 * lead


#: Number of ensemble members in the ensemble fixture dataset.
MEMBER_COUNT = 5
#: Member coordinate values of the ensemble fixture dataset.
MEMBER_INDICES = list(range(MEMBER_COUNT))


def build_ensemble_dataset() -> xr.Dataset:
    """Build a deterministic ensemble dataset with surface variables.

    The dataset mirrors :func:`build_forecast_dataset` but adds a ``member``
    dimension so ensemble statistics and exceedance probabilities can be
    exercised. Values follow analytic fields so expected statistics are exact:

    * ``temperature_2m(member, lat, lon, lead) = 10 + 10*(lat - LAT_START)
      + 10*(lon - LON_START) + 0.5*lead + 2*member``
    * ``precipitation_rate(member, lead) = 0.5*lead + 1.0*member``

    Returns:
        The deterministic ensemble dataset with dimensions
        ``(member, lead_time_hours, latitude, longitude)``.
    """
    member = np.asarray(MEMBER_INDICES, dtype=float)
    lat = np.asarray(LATITUDES, dtype=float)
    lon = np.asarray(LONGITUDES, dtype=float)
    lead = np.asarray(LEAD_TIMES, dtype=float)
    member_grid, lead_grid, lat_grid, lon_grid = np.meshgrid(
        member, lead, lat, lon, indexing="ij"
    )

    temperature = (
        10.0
        + 10.0 * (lat_grid - LAT_START)
        + 10.0 * (lon_grid - LON_START)
        + 0.5 * lead_grid
        + 2.0 * member_grid
    )
    precipitation = 0.5 * lead_grid + 1.0 * member_grid

    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                temperature,
            ),
            "precipitation_rate": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                precipitation,
            ),
        },
        coords={
            "member": MEMBER_INDICES,
            "lead_time_hours": LEAD_TIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )


def write_ensemble_zarr(store: str) -> str:
    """Write the deterministic ensemble fixture dataset to a local Zarr store.

    Args:
        store: Local directory path for the Zarr store.

    Returns:
        The store path (for reporting).
    """
    return write_dataset(build_ensemble_dataset(), store)


def ensemble_temperature_at(
    member: int, latitude: float, longitude: float, lead: int
) -> float:
    """Expected metric ensemble temperature at a member, point, and lead time."""
    return (
        10.0
        + 10.0 * (latitude - LAT_START)
        + 10.0 * (longitude - LON_START)
        + 0.5 * lead
        + 2.0 * member
    )


def ensemble_precipitation_at(member: int, lead: int) -> float:
    """Expected metric ensemble precipitation at a member and lead time."""
    return 0.5 * lead + 1.0 * member
