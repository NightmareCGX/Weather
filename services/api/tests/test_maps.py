"""Contract and integration tests for the Milestone 10 /v1/maps endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the documented ``spatial_layer`` envelope, the self-referential tile URL
template, the legend unit/ramp, and error envelopes. When PostgreSQL is
unreachable they skip, following the existing convention.
"""


def _assert_envelope(body):
    assert body["object"] == "spatial_layer"
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_maps_contract(client):
    resp = client.get(
        "/v1/maps?model=gfs&variable=temperature_2m"
        "&level=surface&lead_time_hours=12"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    assert resp.headers["Cache-Control"] == "public, max-age=3600"

    data = body["data"]
    assert data["tile_url_template"] == (
        "/v1/maps/gfs/temperature_2m/surface/"
        "{z}/{x}/{y}.png?lead_time_hours=12"
    )
    assert data["min_zoom"] == 0
    assert data["max_zoom"] == 9
    assert data["lead_time_hours"] == 12
    assert data["legend"]["unit"] == "°C"
    assert data["legend"]["stops"] == [[-40, "#0000ff"], [0, "#00ff00"], [40, "#ff0000"]]


def test_maps_precipitation_unit(client):
    resp = client.get(
        "/v1/maps?model=gfs&variable=precipitation_rate"
        "&level=surface&lead_time_hours=6"
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["legend"]["unit"] == "mm/h"


def test_maps_ensemble_model_accepted(client):
    # A valid ensemble model identifier is accepted for a map layer.
    resp = client.get(
        "/v1/maps?model=gefs&variable=temperature_2m"
        "&level=surface&lead_time_hours=6"
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["tile_url_template"].startswith("/v1/maps/gefs/")


def test_maps_unknown_model_404(client):
    resp = client.get(
        "/v1/maps?model=nope&variable=temperature_2m"
        "&level=surface&lead_time_hours=6"
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["request_id"] == resp.headers["X-Request-Id"]


def test_maps_unknown_variable_404(client):
    resp = client.get(
        "/v1/maps?model=gfs&variable=wind_speed"
        "&level=surface&lead_time_hours=6"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_maps_unknown_level_422(client):
    resp = client.get(
        "/v1/maps?model=gfs&variable=temperature_2m"
        "&level=500hPa&lead_time_hours=6"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_maps_missing_required_422(client):
    resp = client.get("/v1/maps?model=gfs&variable=temperature_2m&level=surface")
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_maps_negative_lead_time_422(client):
    resp = client.get(
        "/v1/maps?model=gfs&variable=temperature_2m"
        "&level=surface&lead_time_hours=-1"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"
