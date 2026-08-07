"""Unit tests for GRIB2 parsing and normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ingestion.providers.noaa.parser import GribParsingError, normalize, parse_grib2

EXPECTED_LAT = np.array([44.0, 43.0, 42.0, 41.0, 0.0])


def test_parse_grib2_decodes_normalized_dataset(grib_fixture: Path) -> None:
    """The committed fixture decodes and is normalized to the convention."""
    ds = parse_grib2(grib_fixture)

    # Single forecast step is exposed as lead_time_hours.
    assert ds.lead_time_hours.values == 6

    # Coordinates retained.
    assert "time" in ds.coords
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords

    # Latitude axis matches the fixture's north-to-south grid.
    assert np.allclose(ds.latitude.values, EXPECTED_LAT)
    assert ds.latitude.values[0] > ds.latitude.values[-1]

    # The raw GRIB step / valid_time coords are dropped.
    assert "step" not in ds.coords
    assert "valid_time" not in ds.coords

    # The temperature data variable is present and 2-D on the grid.
    assert "t" in ds.data_vars
    assert ds.t.dims == ("latitude", "longitude")
    assert ds.t.shape == (5, 10)
    assert ds.t.dtype == np.float32

    # Grid values land in the expected physical range.
    assert np.allclose(ds.t.values, np.linspace(280.0, 300.0, 50).reshape(5, 10))


def test_normalize_rejects_missing_step() -> None:
    """A dataset without a step coordinate cannot be normalized."""
    ds = xr.Dataset({"t": (("latitude",), [1.0])})
    with pytest.raises(GribParsingError, match="no 'step' coordinate"):
        normalize(ds)


def test_normalize_renames_number_dimension_to_member() -> None:
    """A GEFS dataset decoded with the ``number`` dimension is normalized to
    the platform's ``member`` convention (cfgrib exposes GEFS as ``number``)."""
    import numpy as np

    ds = xr.Dataset(
        {"t": (("number", "latitude", "longitude"), np.ones((2, 2, 2)))},
        coords={
            "number": [0, 1],
            "step": np.array([6 * 3600 * 10**9], dtype="timedelta64[ns]"),
            "time": np.array(["2026-07-21T00:00:00"], dtype="datetime64[ns]"),
            "valid_time": np.array(["2026-07-21T06:00:00"], dtype="datetime64[ns]"),
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
    )
    normalized = normalize(ds)
    assert "member" in normalized.dims
    assert "number" not in normalized.dims
    assert "member" in normalized.coords
    assert "number" not in normalized.coords
    # step/valid_time are dropped and lead_time_hours is set.
    assert "step" not in normalized.coords
    assert "valid_time" not in normalized.coords
    assert int(normalized["lead_time_hours"].values) == 6


def test_parse_grib2_raises_on_corrupt_file(tmp_path: Path) -> None:
    """A corrupt file raises GribParsingError."""
    corrupt = tmp_path / "corrupt.grib2"
    corrupt.write_bytes(b"this is not a grib file")

    with pytest.raises(GribParsingError, match="Failed to decode"):
        parse_grib2(corrupt)
