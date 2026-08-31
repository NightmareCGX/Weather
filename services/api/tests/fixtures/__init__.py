"""Test fixtures for the API service.

Provides helpers to generate deterministic, tiny Zarr forecast datasets on
disk for the point-forecast integration tests, and to build PostGIS location
records (cities, ski resorts, stations) for the search and point-resolution
tests. The Zarr stores are written to a temporary directory at test time by
:func:`build_forecast_dataset` + the test-only ``tests._zarr_writer`` module
(an API-local Zarr writer so the API service never imports the ingestion
package); no binary fixtures are committed. See ``README.md`` for details.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from tests._zarr_writer import (
    write_dataset,
)  # test-only Zarr writer (no ingestion import)

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
    precip_amount_3h = np.where(lead_grid == 0, np.nan, 0.4 * lead_grid)
    crain = np.where((lead_grid > 0) & (temperature > 0), 1, 0).astype(np.uint8)
    csnow = np.where((lead_grid > 0) & (temperature <= 0), 1, 0).astype(np.uint8)
    cfrzr = np.zeros_like(temperature, dtype=np.uint8)
    cicep = np.zeros_like(temperature, dtype=np.uint8)
    cloud_cover_3h = np.where(lead_grid == 0, np.nan, 20.0 + 2.0 * lead_grid)
    cloud_ceiling = np.where(lead_grid == 18, 20000.0, 1000.0 + 50.0 * lead_grid)

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
            "precipitation_amount_3h": (
                ("lead_time_hours", "latitude", "longitude"),
                precip_amount_3h,
            ),
            "crain": (
                ("lead_time_hours", "latitude", "longitude"),
                crain,
            ),
            "csnow": (
                ("lead_time_hours", "latitude", "longitude"),
                csnow,
            ),
            "cfrzr": (
                ("lead_time_hours", "latitude", "longitude"),
                cfrzr,
            ),
            "cicep": (
                ("lead_time_hours", "latitude", "longitude"),
                cicep,
            ),
            "cloud_cover_3h": (
                ("lead_time_hours", "latitude", "longitude"),
                cloud_cover_3h,
            ),
            "cloud_ceiling": (
                ("lead_time_hours", "latitude", "longitude"),
                cloud_ceiling,
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


def _longitudes_0_360() -> np.ndarray:
    """A compact 0-360 longitude axis (0, 20, ..., 340) covering the western hemisphere.

    The axis starts at 0 (valid WGS84), so the derived :class:`RegularGrid`
    origin passes validation, and spans past 180 so serving aligns
    western-hemisphere queries into the ``[0, 360]`` convention.
    """
    return np.arange(0.0, 360.0, 20.0)


def _build_0_360_dataset(*, with_member: bool) -> xr.Dataset:
    """Build a deterministic 0-360 longitude-convention dataset.

    Mirrors :func:`build_forecast_dataset` / :func:`build_ensemble_dataset`
    but stores longitude in the native GFS ``[0, 360]`` convention. The
    temperature field uses ``lon360 - (LON_START + 360)`` which equals the
    standard ``lon - LON_START`` for equivalent geographic points, so expected
    values computed with :func:`temperature_at` /
    :func:`ensemble_temperature_at` are exact for western-hemisphere queries.

    Args:
        with_member: Whether to add an ensemble ``member`` dimension.

    Returns:
        The deterministic dataset with longitude coordinates in ``[0, 360]``.
    """
    lon_0_360 = _longitudes_0_360()
    lat = np.asarray(LATITUDES, dtype=float)
    lead = np.asarray(LEAD_TIMES, dtype=float)

    if with_member:
        member = np.asarray(MEMBER_INDICES, dtype=float)
        member_grid, lead_grid, lat_grid, lon_grid = np.meshgrid(
            member, lead, lat, lon_0_360, indexing="ij"
        )
        temperature = (
            10.0
            + 10.0 * (lat_grid - LAT_START)
            + 10.0 * (lon_grid - (LON_START + 360.0))
            + 0.5 * lead_grid
            + 2.0 * member_grid
        )
        precipitation = 0.5 * lead_grid + 1.0 * member_grid
        precip_amount_3h = np.where(lead_grid == 0, np.nan, 0.4 * lead_grid + 0.2 * member_grid)
        crain = np.where((lead_grid > 0) & (temperature > 0), 1, 0).astype(np.uint8)
        csnow = np.where((lead_grid > 0) & (temperature <= 0), 1, 0).astype(np.uint8)
        cfrzr = np.zeros_like(temperature, dtype=np.uint8)
        cicep = np.zeros_like(temperature, dtype=np.uint8)
        cloud_cover_3h = np.where(lead_grid == 0, np.nan, 20.0 + 2.0 * lead_grid + 1.0 * member_grid)
        cloud_ceiling = np.where(member_grid >= 20, 20000.0, 1000.0 + 50.0 * lead_grid + 20.0 * member_grid)
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
                "precipitation_amount_3h": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    precip_amount_3h,
                ),
                "crain": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    crain,
                ),
                "csnow": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    csnow,
                ),
                "cfrzr": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    cfrzr,
                ),
                "cicep": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    cicep,
                ),
                "cloud_cover_3h": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    cloud_cover_3h,
                ),
                "cloud_ceiling": (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    cloud_ceiling,
                ),
            },
            coords={
                "member": MEMBER_INDICES,
                "lead_time_hours": LEAD_TIMES,
                "latitude": LATITUDES,
                "longitude": lon_0_360,
            },
        )

    lead_grid, lat_grid, lon_grid = np.meshgrid(lead, lat, lon_0_360, indexing="ij")
    temperature = (
        10.0
        + 10.0 * (lat_grid - LAT_START)
        + 10.0 * (lon_grid - (LON_START + 360.0))
        + 0.5 * lead_grid
    )
    precipitation = 0.5 * lead_grid
    precip_amount_3h = np.where(lead_grid == 0, np.nan, 0.4 * lead_grid)
    crain = np.where((lead_grid > 0) & (temperature > 0), 1, 0).astype(np.uint8)
    csnow = np.where((lead_grid > 0) & (temperature <= 0), 1, 0).astype(np.uint8)
    cfrzr = np.zeros_like(temperature, dtype=np.uint8)
    cicep = np.zeros_like(temperature, dtype=np.uint8)
    cloud_cover_3h = np.where(lead_grid == 0, np.nan, 20.0 + 2.0 * lead_grid)
    cloud_ceiling = np.where(lead_grid == 18, 20000.0, 1000.0 + 50.0 * lead_grid)
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
            "precipitation_amount_3h": (
                ("lead_time_hours", "latitude", "longitude"),
                precip_amount_3h,
            ),
            "crain": (
                ("lead_time_hours", "latitude", "longitude"),
                crain,
            ),
            "csnow": (
                ("lead_time_hours", "latitude", "longitude"),
                csnow,
            ),
            "cfrzr": (
                ("lead_time_hours", "latitude", "longitude"),
                cfrzr,
            ),
            "cicep": (
                ("lead_time_hours", "latitude", "longitude"),
                cicep,
            ),
            "cloud_cover_3h": (
                ("lead_time_hours", "latitude", "longitude"),
                cloud_cover_3h,
            ),
            "cloud_ceiling": (
                ("lead_time_hours", "latitude", "longitude"),
                cloud_ceiling,
            ),
        },
        coords={
            "lead_time_hours": LEAD_TIMES,
            "latitude": LATITUDES,
            "longitude": lon_0_360,
        },
    )


def build_forecast_dataset_0_360() -> xr.Dataset:
    """Build a deterministic 0-360 longitude-convention forecast dataset."""
    return _build_0_360_dataset(with_member=False)


def write_forecast_zarr_0_360(store: str) -> str:
    """Write the deterministic 0-360 fixture dataset to a local Zarr store.

    Args:
        store: Local directory path for the Zarr store.

    Returns:
        The store path (for reporting).
    """
    return write_dataset(build_forecast_dataset_0_360(), store)


def build_ensemble_dataset_0_360() -> xr.Dataset:
    """Build a deterministic 0-360 longitude-convention ensemble dataset."""
    return _build_0_360_dataset(with_member=True)


def write_ensemble_zarr_0_360(store: str) -> str:
    """Write the deterministic 0-360 ensemble fixture dataset to a Zarr store.

    Args:
        store: Local directory path for the Zarr store.

    Returns:
        The store path (for reporting).
    """
    return write_dataset(build_ensemble_dataset_0_360(), store)


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

#: The canonical set of GEFS variables the fixture GEFS store carries. This is
#: the **contract** for the table-driven GEFS surface regression tests: every
#: variable here must be servable across Map / Hourly Forecast (points) /
#: Ensemble Statistics. A future GEFS variable addition must extend this set
#: and thereby fail CI if one serving surface is forgotten.
GEFS_FIXTURE_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "precipitation_rate",
    "precipitation_amount_3h",
    "cloud_cover_3h",
    "cloud_ceiling",
)

#: The canonical set of GFS (deterministic) variables the fixture GFS store
#: carries. Added for the deterministic non-regression matrix.
GFS_FIXTURE_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "precipitation_rate",
    "precipitation_amount_3h",
    "cloud_cover_3h",
    "cloud_ceiling",
)


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
    precip_amount_3h = np.where(lead_grid == 0, np.nan, 0.4 * lead_grid + 0.2 * member_grid)
    crain = np.where((lead_grid > 0) & (temperature > 0), 1, 0).astype(np.uint8)
    csnow = np.where((lead_grid > 0) & (temperature <= 0), 1, 0).astype(np.uint8)
    cfrzr = np.zeros_like(temperature, dtype=np.uint8)
    cicep = np.zeros_like(temperature, dtype=np.uint8)
    cloud_cover_3h = np.where(lead_grid == 0, np.nan, 20.0 + 2.0 * lead_grid + 1.0 * member_grid)
    cloud_ceiling = np.where(member_grid >= 4, 20000.0, 1000.0 + 50.0 * lead_grid + 20.0 * member_grid)

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
            "precipitation_amount_3h": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                precip_amount_3h,
            ),
            "crain": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                crain,
            ),
            "csnow": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                csnow,
            ),
            "cfrzr": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                cfrzr,
            ),
            "cicep": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                cicep,
            ),
            "cloud_cover_3h": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                cloud_cover_3h,
            ),
            "cloud_ceiling": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                cloud_ceiling,
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


def ensemble_precipitation_amount_at(member: int, lead: int) -> float:
    """Expected metric ensemble 3h precipitation amount at a member and lead time."""
    return 0.4 * lead + 0.2 * member


def precipitation_amount_at(lead: int) -> float:
    """Expected metric deterministic 3h precipitation amount at a lead time."""
    return 0.4 * lead
