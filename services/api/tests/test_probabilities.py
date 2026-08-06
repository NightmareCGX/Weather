"""Contract and integration tests for the Milestone 10 /v1/probabilities endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the documented ``probability_forecast`` envelope, the empirical exceedance
probability, the deterministic Wilson 95% confidence interval, the
``between``/``threshold_max`` validation, and error envelopes. When
PostgreSQL is unreachable they skip, following the existing convention.
"""

import math

import pytest

from tests.fixtures import (
    LAT_START,
    LON_START,
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
    assert body["object"] == "probability_forecast"
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def _wilson(probability, sample_size):
    """Independent re-derivation of the Wilson score interval for asserts.

    Mirrors ``domain.ensemble.probability_confidence_interval`` from first
    principles so the test does not rely on the implementation it checks.
    """
    z = 1.959963984540054
    z_squared = z * z
    n = float(sample_size)
    denominator = 1.0 + z_squared / n
    center = (probability + z_squared / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(probability * (1.0 - probability) / n + z_squared / (4.0 * n * n))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _precipitation_members():
    """precipitation_rate member values at the test point and lead time."""
    return [ensemble_precipitation_at(member, LEAD) for member in MEMBER_INDICES]


def _temperature_members():
    """temperature_2m member values at the test point and lead time."""
    return [ensemble_temperature_at(member, LAT, LON, LEAD) for member in MEMBER_INDICES]


def _empirical_probability_above(members, threshold):
    """Expected strict-exceedance probability derived from the member values."""
    return sum(1.0 for value in members if value > threshold) / len(members)


def test_probability_gt_contract(client):
    # precipitation_rate members at lead 6 are [3, 4, 5, 6, 7]; strict >4
    # matches {5, 6, 7} -> p = 0.6.
    members = _precipitation_members()
    expected_probability = _empirical_probability_above(members, 4.0)

    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&threshold=4&operator=gt"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    assert resp.headers["Cache-Control"] == "public, max-age=3600"

    data = body["data"]
    assert data["variable"] == "precipitation_rate"
    assert data["threshold"] == 4.0
    assert data["operator"] == "gt"
    assert data["lead_time_hours"] == LEAD
    assert data["probability"] == pytest.approx(expected_probability)
    assert data["location"]["latitude"] == pytest.approx(LAT)
    assert data["location"]["longitude"] == pytest.approx(LON)

    lower, upper = data["confidence_interval_95"]
    assert isinstance(lower, float) and isinstance(upper, float)
    assert 0.0 <= lower <= data["probability"] <= upper <= 1.0
    expected_lower, expected_upper = _wilson(expected_probability, len(members))
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)
    # The response must match the API.md gt/lt shape exactly (no threshold_max).
    assert "threshold_max" not in data


def test_probability_lt_contract(client):
    # Strict <6 matches {3, 4, 5} -> p = 0.6.
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&threshold=6&operator=lt"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["operator"] == "lt"
    assert data["probability"] == pytest.approx(0.6)
    assert "threshold_max" not in data


def test_probability_between_contract(client):
    # Inclusive [4, 6] matches {4, 5, 6} -> p = 0.6.
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&threshold=4&operator=between"
        f"&threshold_max=6&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["operator"] == "between"
    assert data["probability"] == pytest.approx(0.6)
    assert data["threshold_max"] == 6.0


def test_probability_temperature_contract(client):
    # temperature_2m members at lead 6 are [15.5, 17.5, 19.5, 21.5, 23.5];
    # strict >20 matches {21.5, 23.5} -> p = 0.4.
    expected_probability = _empirical_probability_above(_temperature_members(), 20.0)
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=temperature_2m&threshold=20&operator=gt"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["probability"] == pytest.approx(expected_probability)


def test_probability_deterministic(client):
    url = (
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&threshold=4&operator=gt"
        f"&lead_time_hours={LEAD}"
    )
    resp1 = client.get(url)
    resp2 = client.get(url)
    assert resp1.status_code == 200 and resp2.status_code == 200
    assert resp1.json() == resp2.json()


def test_probability_between_missing_threshold_max_422(client):
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&threshold=4&operator=between"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "threshold_max" in body["error"]["message"]
    assert body["error"]["request_id"] == resp.headers["X-Request-Id"]


def test_probability_threshold_max_with_gt_422(client):
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&threshold=4&operator=gt"
        f"&threshold_max=6&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_probability_bad_coordinates_422(client):
    resp = client.get(
        "/v1/probabilities?lat=99&lon=0"
        "&variable=precipitation_rate&threshold=4&operator=gt"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_probability_unknown_variable_404(client):
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        "&variable=wind_speed&threshold=4&operator=gt"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["request_id"] == resp.headers["X-Request-Id"]


def test_probability_unknown_model_404(client):
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        "&variable=precipitation_rate&threshold=4&operator=gt"
        f"&lead_time_hours={LEAD}&model=nope"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_probability_non_ensemble_model_422(client):
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        "&variable=precipitation_rate&threshold=4&operator=gt"
        f"&lead_time_hours={LEAD}&model=gfs"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_probability_lead_not_available_404(client):
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        "&variable=precipitation_rate&threshold=4&operator=gt"
        "&lead_time_hours=24"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_probability_missing_required_422(client):
    resp = client.get(f"/v1/probabilities?lat={LAT}&lon={LON}&variable=precipitation_rate")
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_probability_between_inverted_thresholds_422(client):
    # threshold_max below threshold is rejected by the domain math (422).
    resp = client.get(
        f"/v1/probabilities?lat={LAT}&lon={LON}"
        f"&variable=precipitation_rate&threshold=6&operator=between"
        f"&threshold_max=4&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"
