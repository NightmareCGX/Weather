"""Contract and integration tests for the Milestone 10 /v1/ensembles endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the documented ``ensemble_statistics`` envelope, exact statistics over the
deterministic ensemble fixture, and error envelopes. When PostgreSQL is
unreachable they skip, following the existing convention.
"""

import math

import pytest

from tests.fixtures import LAT_START, LON_START

#: Test point at the center of a fixture grid cell (the analytic fields are
#: linear, so bilinear interpolation reproduces the analytic values exactly).
LAT = LAT_START + 0.125  # 38.125
LON = LON_START + 0.125  # -106.875
LEAD = 6


def _assert_envelope(body):
    assert body["object"] == "ensemble_statistics"
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_ensembles_temperature_contract(client):
    # temperature_2m(member) = 15.5 + 2*member -> [15.5, 17.5, 19.5, 21.5, 23.5].
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        f"&variable=temperature_2m&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    assert resp.headers["Cache-Control"] == "public, max-age=1800"

    data = body["data"]
    assert data["model"] == "gefs"
    assert data["lead_time_hours"] == LEAD
    assert data["member_count"] == 5

    stats = data["statistics"]
    assert stats["mean"] == pytest.approx(19.5)
    assert stats["median"] == pytest.approx(19.5)
    assert stats["spread"] == pytest.approx(math.sqrt(8.0))
    assert stats["p10"] == pytest.approx(16.3)
    assert stats["p25"] == pytest.approx(17.5)
    assert stats["p50"] == pytest.approx(19.5)
    assert stats["p75"] == pytest.approx(21.5)
    assert stats["p90"] == pytest.approx(22.7)


def test_ensembles_precipitation_contract(client):
    # precipitation_rate(member) = 3 + member -> [3, 4, 5, 6, 7].
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["model"] == "gefs"
    assert data["member_count"] == 5

    stats = data["statistics"]
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["median"] == pytest.approx(5.0)
    assert stats["spread"] == pytest.approx(math.sqrt(2.0))
    assert stats["p10"] == pytest.approx(3.4)
    assert stats["p25"] == pytest.approx(4.0)
    assert stats["p50"] == pytest.approx(5.0)
    assert stats["p75"] == pytest.approx(6.0)
    assert stats["p90"] == pytest.approx(6.6)


def test_ensembles_deterministic(client):
    url = (
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        f"&variable=temperature_2m&lead_time_hours={LEAD}"
    )
    resp1 = client.get(url)
    resp2 = client.get(url)
    assert resp1.status_code == 200 and resp2.status_code == 200
    assert resp1.json() == resp2.json()


def test_ensembles_unknown_variable_404(client):
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=wind_speed"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["request_id"] == resp.headers["X-Request-Id"]


def test_ensembles_unknown_model_404(client):
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}&model=nope"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_ensembles_non_ensemble_model_422(client):
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}&model=gfs"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_ensembles_lead_not_available_404(client):
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m&lead_time_hours=24"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_ensembles_missing_required_422(client):
    resp = client.get(f"/v1/ensembles?lat={LAT}&lon={LON}")
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_ensembles_bad_coordinates_422(client):
    resp = client.get(
        "/v1/ensembles?lat=99&lon=0"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"
