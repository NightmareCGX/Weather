"""Contract and integration tests for the Milestone 10 /v1/ensembles endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the documented ``ensemble_statistics`` envelope, exact statistics over the
deterministic ensemble fixture, and error envelopes. When PostgreSQL is
unreachable they skip, following the existing convention.
"""

import numpy as np
import pytest

from tests.fixtures import (
    LAT_START,
    LON_START,
    MEMBER_COUNT,
    MEMBER_INDICES,
    ensemble_precipitation_at,
    ensemble_temperature_at,
)

#: Test point at the center of a fixture grid cell (the analytic fields are
#: linear, so bilinear interpolation reproduces the analytic values exactly).
LAT = LAT_START + 0.125  # 38.125
LON = LON_START + 0.125  # -106.875
LEAD = 6


def _assert_envelope(body):
    assert body["object"] == "ensemble_statistics"
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def _expected_statistics(member_values: list[float]) -> dict[str, float]:
    """Expected ensemble statistics derived from the analytic member values.

    Mirrors the domain convention: mean/median via NumPy, population spread
    (``ddof=0``), and linear percentiles (P10/P25/P50/P75/P90).
    """
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


def test_ensembles_temperature_contract(client):
    # temperature_2m(member) = 15.5 + 2*member -> [15.5, 17.5, 19.5, 21.5, 23.5].
    expected_members = [
        ensemble_temperature_at(member, LAT, LON, LEAD) for member in MEMBER_INDICES
    ]
    expected = _expected_statistics(expected_members)

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
    assert data["member_count"] == MEMBER_COUNT

    stats = data["statistics"]
    assert stats["mean"] == pytest.approx(expected["mean"])
    assert stats["median"] == pytest.approx(expected["median"])
    assert stats["spread"] == pytest.approx(expected["spread"])
    assert stats["p10"] == pytest.approx(expected["p10"])
    assert stats["p25"] == pytest.approx(expected["p25"])
    assert stats["p50"] == pytest.approx(expected["p50"])
    assert stats["p75"] == pytest.approx(expected["p75"])
    assert stats["p90"] == pytest.approx(expected["p90"])


def test_ensembles_precipitation_contract(client):
    # precipitation_rate(member) = 3 + member -> [3, 4, 5, 6, 7].
    expected_members = [
        ensemble_precipitation_at(member, LEAD) for member in MEMBER_INDICES
    ]
    expected = _expected_statistics(expected_members)

    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["model"] == "gefs"
    assert data["member_count"] == MEMBER_COUNT

    stats = data["statistics"]
    assert stats["mean"] == pytest.approx(expected["mean"])
    assert stats["median"] == pytest.approx(expected["median"])
    assert stats["spread"] == pytest.approx(expected["spread"])
    assert stats["p10"] == pytest.approx(expected["p10"])
    assert stats["p25"] == pytest.approx(expected["p25"])
    assert stats["p50"] == pytest.approx(expected["p50"])
    assert stats["p75"] == pytest.approx(expected["p75"])
    assert stats["p90"] == pytest.approx(expected["p90"])


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
