"""Unit tests for GRIB2 parsing and normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ingestion.providers.noaa.parser import GribParsingError, normalize, parse_grib2

EXPECTED_LAT = np.array([40.0, 39.0, 38.0, 37.0, 36.0])
#: Longitudes decode in the GRIB native 0..360 convention (250..259 = -110..-101).
EXPECTED_LON = np.array([250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 256.0, 257.0, 258.0, 259.0])


def test_parse_grib2_decodes_normalized_dataset(grib_fixture: Path) -> None:
    """The committed fixture decodes and is normalized to the convention."""
    ds = parse_grib2(grib_fixture)

    # Single forecast step is exposed as lead_time_hours.
    assert ds.lead_time_hours.values == 6

    # Coordinates retained.
    assert "time" in ds.coords
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords

    # Latitude axis matches the fixture's uniform north-to-south grid.
    assert np.allclose(ds.latitude.values, EXPECTED_LAT)
    assert ds.latitude.values[0] > ds.latitude.values[-1]
    # The axis must be uniformly spaced so the serving tier can derive a
    # RegularGrid (a non-uniform axis is rejected as HTTP 500).
    assert np.allclose(np.diff(ds.latitude.values), ds.latitude.values[1] - ds.latitude.values[0])

    # Longitude decodes in the GRIB native 0..360 convention, uniformly spaced.
    assert np.allclose(ds.longitude.values, EXPECTED_LON)
    assert np.allclose(np.diff(ds.longitude.values), ds.longitude.values[1] - ds.longitude.values[0])

    # The raw GRIB step / valid_time coords are dropped.
    assert "step" not in ds.coords
    assert "valid_time" not in ds.coords

    # The temperature data variable is present and 2-D on the grid. The
    # fixture decodes to the cfgrib-emitted name ``t2m`` (the GRIB
    # ``cfVarName`` for the ``2t``/``heightAboveGround``/2 field), which is
    # what the pipeline's ``DEFAULT_VARIABLES`` mapping matches on.
    assert "t2m" in ds.data_vars
    assert ds.t2m.dims == ("latitude", "longitude")
    assert ds.t2m.shape == (5, 10)
    assert ds.t2m.dtype == np.float32

    # Grid values land in the expected physical range.
    assert np.allclose(ds.t2m.values, np.linspace(280.0, 300.0, 50).reshape(5, 10))


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


#: Path to the committed multi-typeOfLevel regression fixture.
MULTI_FIXTURE = Path(__file__).parent / "fixtures" / "gfs_multi_typeoflevel.grib2"


def test_parse_grib2_multi_typeoflevel_succeeds() -> None:
    """A realistic multi-typeOfLevel file no longer raises DatasetBuildError.

    Regression for the production failure: the parser used to open the whole
    GRIB2 file as one unfiltered cfgrib dataset, which raised
    ``DatasetBuildError: multiple values for unique key 'typeOfLevel'`` on
    real operational GFS ``pgrb2`` files (which mix ``surface``,
    ``heightAboveGround``, ``isobaricInhPa``, ``meanSea``, etc.). The fixture
    reproduces that structure.
    """
    ds = parse_grib2(MULTI_FIXTURE)

    # The two platform-required surface fields are selected and merged.
    assert set(ds.data_vars) == {"t2m", "prate"}
    assert ds.t2m.dims == ("latitude", "longitude")
    assert ds.prate.dims == ("latitude", "longitude")

    # lead_time_hours is preserved through the merge.
    assert ds.lead_time_hours.values == 6

    # step / valid_time are dropped per the normalized contract.
    assert "step" not in ds.coords
    assert "valid_time" not in ds.coords


def test_parse_grib2_temperature_emits_t2m() -> None:
    """The 2 m temperature field decodes to the cfgrib ``t2m`` variable."""
    ds = parse_grib2(MULTI_FIXTURE)
    assert "t2m" in ds.data_vars
    assert ds.t2m.attrs["units"] == "K"
    # Representative value: 280.0 K selected from the fixture.
    assert float(ds.t2m.values.flat[0]) == pytest.approx(280.0)


def test_parse_grib2_precipitation_rate_selects_instant() -> None:
    """The parser deterministically selects the instantaneous prate field.

    The fixture contains both an instant (0.0003 kg m-2 s-1) and an avg
    (0.0001 kg m-2 s-1) ``prate`` message. A broad ``shortName=prate``
    selection is ambiguous; the parser must pick the instant one.
    """
    ds = parse_grib2(MULTI_FIXTURE)
    assert "prate" in ds.data_vars
    assert ds.prate.attrs["units"] == "kg m**-2 s**-1"
    assert float(ds.prate.values.flat[0]) == pytest.approx(0.0003)
    # The avg prate message (0.0001) must NOT be selected.
    assert not np.any(np.isclose(ds.prate.values, 0.0001))


def test_parse_grib2_ignores_unrelated_levels() -> None:
    """Unrelated fields/levels in the file are not part of the result.

    The fixture also carries a ``t`` field at ``isobaricInhPa``/850 and a
    ``prmsl`` field at ``meanSea``. Only the platform-required surface fields
    are returned.
    """
    ds = parse_grib2(MULTI_FIXTURE)
    assert "t" not in ds.data_vars  # the isobaric 850 hPa temperature
    assert "prmsl" not in ds.data_vars
    assert "isobaricInhPa" not in ds.coords
    assert "meanSea" not in ds.coords


def test_parse_grib2_merged_integrity() -> None:
    """The merged dataset has compatible coordinates and no duplicate dims."""
    ds = parse_grib2(MULTI_FIXTURE)

    assert dict(ds.sizes) == {"latitude": 5, "longitude": 10}
    assert list(ds.dims) == ["latitude", "longitude"]

    # Both fields share the identical grid.
    for name in ("t2m", "prate"):
        assert ds[name].dims == ("latitude", "longitude")
        assert ds[name].shape == (5, 10)

    # Latitude/longitude are uniformly spaced (serving tier requires this).
    assert np.allclose(
        np.diff(ds.latitude.values), ds.latitude.values[1] - ds.latitude.values[0]
    )
    assert np.allclose(
        np.diff(ds.longitude.values),
        ds.longitude.values[1] - ds.longitude.values[0],
    )

    # Transient scalar GRIB level coordinates are stripped during normalization.
    assert "heightAboveGround" not in ds.coords
    assert "surface" not in ds.coords
    assert "isobaricInhPa" not in ds.coords
    assert "meanSea" not in ds.coords

    # Platform coordinates are preserved.
    assert "time" in ds.coords
    assert "lead_time_hours" in ds.coords
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords


def test_parse_grib2_empty_file_no_fields_raises(tmp_path: Path) -> None:
    """A file containing none of the required surface fields raises."""
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    path = tmp_path / "only_levels.grib2"
    with path.open("wb") as f:
        msg = codes_grib_new_from_samples("GRIB2")
        codes_set(msg, "dataDate", 20260812)
        codes_set(msg, "dataTime", 12)
        codes_set(msg, "stepType", "instant")
        codes_set(msg, "stepRange", "6")
        codes_set(msg, "stepUnits", "h")
        codes_set(msg, "paramId", 130)
        codes_set(msg, "shortName", "t")
        codes_set(msg, "typeOfLevel", "isobaricInhPa")
        codes_set(msg, "level", 850)
        codes_set(msg, "gridType", "regular_ll")
        codes_set(msg, "Ni", 2)
        codes_set(msg, "Nj", 2)
        codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
        codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
        codes_set(msg, "latitudeOfLastGridPointInDegrees", 38.0)
        codes_set(msg, "longitudeOfLastGridPointInDegrees", 252.0)
        codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
        codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
        codes_set_values(msg, np.full((2, 2), 250.0, dtype=np.float32).ravel())
        codes_write(msg, f)
        codes_release(msg)

    with pytest.raises(GribParsingError, match="none of the required"):
        parse_grib2(path)


def test_parse_grib2_gefs_preserves_member_dimension(tmp_path: Path) -> None:
    """Field-selective parsing preserves the GEFS ensemble dimension.

    A GEFS ``pgrb2`` file carries every ensemble member in one file, exposed
    by cfgrib as a ``number`` dimension. The parser must preserve that
    dimension through selection and normalization (renamed to ``member``),
    not collapse or drop it.
    """
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    def _gefs_t2m(f, member_number: int, value: float) -> None:
        msg = codes_grib_new_from_samples("GRIB2")
        codes_set(msg, "dataDate", 20260812)
        codes_set(msg, "dataTime", 12)
        codes_set(msg, "stepType", "instant")
        codes_set(msg, "stepRange", "6")
        codes_set(msg, "stepUnits", "h")
        codes_set(msg, "paramId", 167)
        codes_set(msg, "shortName", "2t")
        codes_set(msg, "typeOfLevel", "heightAboveGround")
        codes_set(msg, "level", 2)
        codes_set(msg, "productDefinitionTemplateNumber", 1)
        codes_set(msg, "perturbationNumber", member_number)
        codes_set(msg, "numberOfForecastsInEnsemble", 2)
        codes_set(msg, "typeOfEnsembleForecast", 3)
        codes_set(msg, "gridType", "regular_ll")
        codes_set(msg, "Ni", 10)
        codes_set(msg, "Nj", 5)
        codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
        codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
        codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
        codes_set(msg, "longitudeOfLastGridPointInDegrees", 259.0)
        codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
        codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
        codes_set_values(msg, np.full((5, 10), value, dtype=np.float32).ravel())
        codes_write(msg, f)
        codes_release(msg)

    path = tmp_path / "gefs_t2m.grib2"
    with path.open("wb") as f:
        _gefs_t2m(f, 0, 280.0)
        _gefs_t2m(f, 1, 281.0)

    ds = parse_grib2(path)

    # The ensemble dimension is preserved and renamed to the platform member
    # convention.
    assert "member" in ds.dims
    assert "number" not in ds.dims
    assert list(ds.coords["member"].values) == [0, 1]
    assert ds.t2m.dims == ("member", "latitude", "longitude")
    assert ds.t2m.shape == (2, 5, 10)
    assert ds.lead_time_hours.values == 6


def test_parse_grib2_selective_vs_full_equivalence(tmp_path: Path) -> None:
    """A selective GRIB artifact with only required messages decodes identically to full file."""
    from eccodes import (
        codes_get,
        codes_get_message,
        codes_grib_new_from_file,
        codes_release,
    )

    # Decode full file
    ds_full = parse_grib2(MULTI_FIXTURE)

    # Extract only required messages (2t and instantaneous prate)
    selected_messages: list[bytes] = []
    with MULTI_FIXTURE.open("rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            sn = codes_get(gid, "shortName")
            tol = codes_get(gid, "typeOfLevel")
            st = codes_get(gid, "stepType")
            if (sn == "2t" and tol == "heightAboveGround") or (
                sn == "prate" and tol == "surface" and st == "instant"
            ):
                selected_messages.append(codes_get_message(gid))
            codes_release(gid)

    assert len(selected_messages) == 2

    # Write selective GRIB artifact
    sel_path = tmp_path / "selective.grib2"
    with sel_path.open("wb") as f:
        for msg in selected_messages:
            f.write(msg)

    # Decode selective artifact
    ds_sel = parse_grib2(sel_path)

    # Verify 100% equivalence of data variables and coordinates
    assert set(ds_full.data_vars) == set(ds_sel.data_vars)
    assert set(ds_full.coords) == set(ds_sel.coords)
    assert ds_full.lead_time_hours.values == ds_sel.lead_time_hours.values

    for var in ds_full.data_vars:
        assert ds_full[var].dims == ds_sel[var].dims
        assert ds_full[var].shape == ds_sel[var].shape
        assert np.array_equal(ds_full[var].values, ds_sel[var].values)

    for coord in ds_full.coords:
        assert np.array_equal(ds_full[coord].values, ds_sel[coord].values)


def test_parse_grib2_gfs_multi_height_succeeds(tmp_path: Path) -> None:
    """GFS decode with multiple heightAboveGround levels (2m + 10m) merges without conflict.

    Regression test: Prior to stripping transient GRIB level coordinates during
    normalization, combining 2 m fields (2t, 2r -> heightAboveGround=2) and 10 m
    fields (10u, 10v -> heightAboveGround=10) caused xr.merge to raise MergeError.
    """
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    def _write_msg(f, sn: str, tol: str, lvl: int, st: str = "instant", val: float = 1.0) -> None:
        msg = codes_grib_new_from_samples("GRIB2")
        codes_set(msg, "dataDate", 20260829)
        codes_set(msg, "dataTime", 1800)
        codes_set(msg, "stepType", st)
        codes_set(msg, "stepRange", "6")
        codes_set(msg, "stepUnits", "h")
        codes_set(msg, "shortName", sn)
        codes_set(msg, "typeOfLevel", tol)
        codes_set(msg, "level", lvl)
        codes_set(msg, "gridType", "regular_ll")
        codes_set(msg, "Ni", 10)
        codes_set(msg, "Nj", 5)
        codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
        codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
        codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
        codes_set(msg, "longitudeOfLastGridPointInDegrees", 259.0)
        codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
        codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
        codes_set_values(msg, np.full((5, 10), val, dtype=np.float32).ravel())
        codes_write(msg, f)
        codes_release(msg)

    path = tmp_path / "gfs_multi_height.grib2"
    with path.open("wb") as f:
        # 2m fields (heightAboveGround = 2)
        _write_msg(f, "2t", "heightAboveGround", 2, "instant", 280.0)
        _write_msg(f, "2r", "heightAboveGround", 2, "instant", 75.0)
        # 10m fields (heightAboveGround = 10)
        _write_msg(f, "10u", "heightAboveGround", 10, "instant", 5.0)
        _write_msg(f, "10v", "heightAboveGround", 10, "instant", -3.0)
        # Surface fields (surface = 0)
        _write_msg(f, "gust", "surface", 0, "instant", 12.0)
        _write_msg(f, "vis", "surface", 0, "instant", 10000.0)
        _write_msg(f, "sde", "surface", 0, "instant", 0.05)
        _write_msg(f, "tp", "surface", 0, "accum", 2.5)
        _write_msg(f, "crain", "surface", 0, "avg", 1.0)
        _write_msg(f, "csnow", "surface", 0, "avg", 0.0)
        _write_msg(f, "cfrzr", "surface", 0, "avg", 0.0)
        _write_msg(f, "cicep", "surface", 0, "avg", 0.0)

    ds = parse_grib2(path)

    expected_vars = {
        "t2m",
        "r2",
        "u10",
        "v10",
        "gust",
        "vis",
        "sde",
        "tp",
        "crain",
        "csnow",
        "cfrzr",
        "cicep",
    }
    assert expected_vars.issubset(set(ds.data_vars))

    # All variables share the 2D grid
    for var in expected_vars:
        assert ds[var].dims == ("latitude", "longitude")
        assert ds[var].shape == (5, 10)

    # Lead time is preserved
    assert ds.lead_time_hours.values == 6

    # Transient scalar GRIB coordinates are absent
    for scalar_coord in (
        "heightAboveGround",
        "surface",
        "atmosphere",
        "cloudCeiling",
        "entireAtmosphere",
        "meanSea",
    ):
        assert scalar_coord not in ds.coords

    # Required platform coordinates are present
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "time" in ds.coords
    assert "lead_time_hours" in ds.coords


def test_parse_grib2_gefs_multi_height_preserves_member_dimension(tmp_path: Path) -> None:
    """GEFS multi-height decode preserves ensemble member dimension without level coordinate conflicts."""
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    def _write_gefs_msg(
        f, sn: str, tol: str, lvl: int, member_num: int, st: str = "instant", val: float = 1.0
    ) -> None:
        msg = codes_grib_new_from_samples("GRIB2")
        codes_set(msg, "dataDate", 20260829)
        codes_set(msg, "dataTime", 1800)
        codes_set(msg, "stepType", st)
        codes_set(msg, "stepRange", "6")
        codes_set(msg, "stepUnits", "h")
        codes_set(msg, "shortName", sn)
        codes_set(msg, "typeOfLevel", tol)
        codes_set(msg, "level", lvl)
        codes_set(msg, "productDefinitionTemplateNumber", 1)
        codes_set(msg, "perturbationNumber", member_num)
        codes_set(msg, "numberOfForecastsInEnsemble", 30)
        codes_set(msg, "typeOfEnsembleForecast", 3)
        codes_set(msg, "gridType", "regular_ll")
        codes_set(msg, "Ni", 10)
        codes_set(msg, "Nj", 5)
        codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
        codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
        codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
        codes_set(msg, "longitudeOfLastGridPointInDegrees", 259.0)
        codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
        codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
        codes_set_values(msg, np.full((5, 10), val, dtype=np.float32).ravel())
        codes_write(msg, f)
        codes_release(msg)

    path = tmp_path / "gefs_multi_height.grib2"
    with path.open("wb") as f:
        for member in (1, 2):
            _write_gefs_msg(f, "2t", "heightAboveGround", 2, member, "instant", 280.0 + member)
            _write_gefs_msg(f, "2r", "heightAboveGround", 2, member, "instant", 70.0 + member)
            _write_gefs_msg(f, "10u", "heightAboveGround", 10, member, "instant", 4.0 + member)
            _write_gefs_msg(f, "10v", "heightAboveGround", 10, member, "instant", -2.0 + member)
            _write_gefs_msg(f, "gust", "surface", 0, member, "instant", 10.0 + member)
            _write_gefs_msg(f, "vis", "surface", 0, member, "instant", 9000.0)
            _write_gefs_msg(f, "sde", "surface", 0, member, "instant", 0.0)

    ds = parse_grib2(path)

    expected_vars = {"t2m", "r2", "u10", "v10", "gust", "vis", "sde"}
    assert expected_vars.issubset(set(ds.data_vars))

    # Ensemble dimension is preserved
    assert "member" in ds.dims
    assert list(ds.coords["member"].values) == [1, 2]

    for var in expected_vars:
        assert ds[var].dims == ("member", "latitude", "longitude")
        assert ds[var].shape == (2, 5, 10)

    # Transient scalar GRIB coordinates are absent
    for scalar_coord in ("heightAboveGround", "surface", "atmosphere", "cloudCeiling"):
        assert scalar_coord not in ds.coords

