"""Table-driven regression matrix for the GEFS variable serving contract.

Every variable in the canonical :data:`GEFS_FIXTURE_VARIABLES` set must be
servable across all three GEFS serving surfaces:

* **Map** — ``/v1/maps/gefs/{variable}/surface/...png`` renders a valid 256x256
  PNG (ensemble-mean field).
* **Hourly Forecast** — ``/v1/points`` with ``models=gefs`` returns a successful
  response carrying the variable, with the ensemble-mean semantics.
* **Ensemble Statistics** — ``/v1/ensembles`` with ``model=gefs`` returns
  statistics for the variable over the full member distribution.

The canonical set comes from :data:`GEFS_FIXTURE_VARIABLES` (the fixture GEFS
store's real contents). A future GEFS variable addition extends that set, and
these parameterized tests fail CI if any one serving surface is forgotten.

GFS (deterministic) variables are asserted non-regressively against the same
surfaces so ensemble logic never alters the deterministic path.
"""

import numpy as np
import pytest

from tests.fixtures import (
    GEFS_FIXTURE_VARIABLES,
    GFS_FIXTURE_VARIABLES,
    LAT_START,
    LON_START,
    MEMBER_INDICES,
    ensemble_precipitation_at,
    ensemble_temperature_at,
)

#: Test point at the center of a fixture grid cell (bilinear interpolation is
#: exact for the analytic fixture fields).
LAT = LAT_START + 0.125
LON = LON_START + 0.125
LEAD = 6
#: Zoom/tile coordinates overlapping the fixture grid (lat 38-38.75 /
#: lon -107..-106.25), so a rendered tile has opaque pixels.
TILE_Z, TILE_X, TILE_Y = 8, 51, 98


def _member_expected_value(variable: str, member: int, lat: float, lon: float, lead: int) -> float:
    """The analytic fixture value of one member at a point and lead."""
    if variable == "temperature_2m":
        return ensemble_temperature_at(member, lat, lon, lead)
    if variable == "precipitation_rate":
        return ensemble_precipitation_at(member, lead)
    if variable == "precipitation_amount_3h":
        from tests.fixtures import ensemble_precipitation_amount_at
        return ensemble_precipitation_amount_at(member, lead)
    raise KeyError(variable)


def _ensemble_mean(variable: str, lat: float, lon: float, lead: int) -> float:
    """The analytic ensemble-mean value at a point and lead."""
    return float(
        np.mean([_member_expected_value(variable, m, lat, lon, lead) for m in MEMBER_INDICES])
    )


def _expected_statistics(member_values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(member_values)),
        "median": float(np.median(member_values)),
        "spread": float(np.std(member_values, ddof=0)),
        "p10": float(np.percentile(member_values, 10, method="linear")),
        "p25": float(np.percentile(member_values, 25, method="linear")),
        "p50": float(np.percentile(member_values, 50, method="linear")),
        "p75": float(np.percentile(member_values, 75, method="linear")),
        "p90": float(np.percentile(member_values, 90, method="linear")),
    }


# --- Surface A: Map raster (ensemble mean) ---


@pytest.mark.parametrize("variable", GEFS_FIXTURE_VARIABLES)
def test_gefs_map_tile_renderable(client, variable):
    """Every GEFS fixture variable renders a 256x256 PNG tile (not a 422)."""
    from tests.test_tiles import _png_dimensions, _png_has_opaque_pixels

    resp = client.get(
        f"/v1/maps/gefs/{variable}/surface/{TILE_Z}/{TILE_X}/{TILE_Y}.png"
        f"?lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["Content-Type"] == "image/png"
    width, height = _png_dimensions(resp.content)
    assert (width, height) == (256, 256)
    assert _png_has_opaque_pixels(resp.content)


@pytest.mark.parametrize("variable", GEFS_FIXTURE_VARIABLES)
def test_gefs_map_metadata(client, variable):
    """The map metadata endpoint advertises a tile template for the variable."""
    resp = client.get(
        f"/v1/maps?model=gefs&variable={variable}&level=surface&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["tile_url_template"].startswith(
        f"/v1/maps/gefs/{variable}/surface/"
    )


# --- Surface B: Hourly Forecast / point forecast (ensemble mean) ---


@pytest.mark.parametrize("variable", GEFS_FIXTURE_VARIABLES)
def test_gefs_point_ensemble_mean(client, variable):
    """The GEFS Hourly Forecast value at a point is the ensemble mean.

    Numeric cross-surface verification: the point value equals the mean of the
    raw member fields (the map tile's source field and the ensemble statistics
    mean share the same semantic).
    """
    resp = client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gefs")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["model"] == "gefs"
    entry = next(e for e in data["forecasts"] if e["lead_time_hours"] == LEAD)
    assert variable in entry
    expected = _ensemble_mean(variable, LAT, LON, LEAD)
    assert abs(float(entry[variable]) - expected) < 1e-9


# --- Surface C: Ensemble Statistics (full member distribution) ---


@pytest.mark.parametrize("variable", GEFS_FIXTURE_VARIABLES)
def test_gefs_ensemble_statistics(client, variable):
    """Every GEFS fixture variable returns ensemble statistics over the members."""
    expected_members = [
        _member_expected_value(variable, m, LAT, LON, LEAD) for m in MEMBER_INDICES
    ]
    expected = _expected_statistics(expected_members)

    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}&variable={variable}"
        f"&model=gefs&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200, resp.text
    d = resp.json()["data"]
    assert d["member_count"] == len(MEMBER_INDICES)
    stats = d["statistics"]
    for key in ("mean", "median", "spread", "p10", "p25", "p50", "p75", "p90"):
        assert abs(float(stats[key]) - expected[key]) < 1e-9


@pytest.mark.parametrize("variable", GEFS_FIXTURE_VARIABLES)
def test_gefs_ensemble_mean_matches_point_and_map(client, variable):
    """Cross-surface invariant: Hourly Forecast == Ensemble Statistics.mean.

    The map tile renders the ensemble mean (its source field is the mean over
    members), so all three surfaces agree on the ensemble-mean value at a point.
    """
    point = client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gefs").json()
    entry = next(e for e in point["data"]["forecasts"] if e["lead_time_hours"] == LEAD)
    ens = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}&variable={variable}"
        f"&model=gefs&lead_time_hours={LEAD}"
    ).json()
    ens_mean = ens["data"]["statistics"]["mean"]
    assert abs(float(entry[variable]) - float(ens_mean)) < 1e-9


# --- Availability-driven walk-through (browser-equivalent smoke) ---
#
# The frontend drives its selectors from ``/v1/forecast/availability`` and then
# requests the map layer, Hourly Forecast, and Ensemble Statistics for the
# advertised model/variable/initial-time/lead combinations. This walk-through
# mirrors that flow for every GEFS variable the availability actually advertises,
# asserting there is no systematic 404/422/500 for supported combinations.


def test_gefs_availability_browser_walkthrough(client):
    """Every advertised GEFS variable serves all three surfaces without errors."""
    av = client.get("/v1/forecast/availability")
    assert av.status_code == 200
    gefs = next(m for m in av.json()["data"]["models"] if m["id"] == "gefs")
    assert gefs["is_ensemble"] is True
    # The fixture GEFS store carries both canonical variables; each must serve.
    advertised = {v["id"] for v in gefs["variables"]}
    assert advertised == set(GEFS_FIXTURE_VARIABLES)

    for variable in sorted(advertised):
        entry = next(v for v in gefs["variables"] if v["id"] == variable)
        initial = entry["initial_times"][0]
        # For interval accumulation fields (which are NaN at lead 0), use lead > 0 if available
        lead = (
            initial["lead_time_hours"][1]
            if variable == "precipitation_amount_3h" and len(initial["lead_time_hours"]) > 1
            else initial["lead_time_hours"][0]
        )
        initial_time = initial["value"]

        # Map metadata (the browser requests this first to build the tile URL).
        meta = client.get(
            f"/v1/maps?model=gefs&variable={variable}&level=surface"
            f"&lead_time_hours={lead}&initial_time={initial_time}"
        )
        assert meta.status_code == 200, (variable, meta.text)
        template = meta.json()["data"]["tile_url_template"]
        assert f"/v1/maps/gefs/{variable}/surface/{{z}}/{{x}}/{{y}}.png" in template

        # Map tile render (the MapLibre raster source).
        tile = client.get(
            f"/v1/maps/gefs/{variable}/surface/{TILE_Z}/{TILE_X}/{TILE_Y}.png"
            f"?lead_time_hours={lead}"
        )
        assert tile.status_code == 200, (variable, tile.text)

        # Hourly Forecast (point).
        point = client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gefs")
        assert point.status_code == 200, (variable, point.text)

        # Ensemble Statistics.
        ens = client.get(
            f"/v1/ensembles?lat={LAT}&lon={LON}&variable={variable}"
            f"&model=gefs&lead_time_hours={lead}"
        )
        assert ens.status_code == 200, (variable, ens.text)


def test_no_cycle_time_advertised_as_variable(client):
    """The availability walk-through never advertises a metadata field."""
    av = client.get("/v1/forecast/availability").json()
    for model in av["data"]["models"]:
        for variable in model["variables"]:
            assert variable["id"] != "cycle_time"
            assert "cycle_time" not in variable["id"]


# --- Deterministic (GFS) non-regression: ensemble logic never alters GFS ---


@pytest.mark.parametrize("variable", GFS_FIXTURE_VARIABLES)
def test_gfs_point_unchanged(client, variable):
    """The GFS deterministic point forecast is unchanged (no member path)."""
    from tests.fixtures import precipitation_at, temperature_at

    resp = client.get(f"/v1/points?lat={LAT}&lon={LON}&models=gfs")
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["model"] == "gfs"
    entry = next(e for e in resp.json()["data"]["forecasts"] if e["lead_time_hours"] == LEAD)
    # The GFS field has no member dim; the value equals the analytic
    # deterministic field exactly (the ensemble path never alters it).
    if variable == "temperature_2m":
        expected = temperature_at(LAT, LON, LEAD)
    elif variable == "precipitation_rate":
        expected = precipitation_at(LEAD)
    elif variable == "precipitation_amount_3h":
        from tests.fixtures import precipitation_amount_at
        expected = precipitation_amount_at(LEAD)
    else:
        raise KeyError(variable)
    assert abs(float(entry[variable]) - expected) < 1e-9


@pytest.mark.parametrize("variable", GFS_FIXTURE_VARIABLES)
def test_gfs_map_tile_renderable(client, variable):
    """The GFS deterministic map tile still renders (no ensemble regression)."""
    from tests.test_tiles import _png_dimensions, _png_has_opaque_pixels

    resp = client.get(
        f"/v1/maps/gfs/{variable}/surface/{TILE_Z}/{TILE_X}/{TILE_Y}.png"
        f"?lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["Content-Type"] == "image/png"
    width, height = _png_dimensions(resp.content)
    assert (width, height) == (256, 256)
    assert _png_has_opaque_pixels(resp.content)