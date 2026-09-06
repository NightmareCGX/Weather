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
    assert resp.headers["Cache-Control"] == "no-cache"

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


def test_ensembles_default_omits_members(client):
    """The default request returns the existing statistics contract only."""
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "members" not in data
    assert "pdf" not in data
    assert data["member_count"] == MEMBER_COUNT
    assert set(data["statistics"]) == {
        "mean",
        "median",
        "spread",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
    }


def test_ensembles_explicit_false_omits_members(client):
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}&include_members=false"
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "members" not in data
    assert "pdf" not in data
    assert data["member_count"] == MEMBER_COUNT


def test_ensembles_include_members_returns_genuine_values(client):
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}&include_members=true"
    )
    assert resp.status_code == 200
    data = resp.json()["data"]

    expected_members = [
        ensemble_temperature_at(member, LAT, LON, LEAD) for member in MEMBER_INDICES
    ]
    assert data["members"] == pytest.approx(expected_members)
    assert data["member_count"] == len(data["members"])
    assert "pdf" in data
    assert data["pdf"] is not None
    assert len(data["pdf"]["x"]) == 100
    assert len(data["pdf"]["density"]) == 100
    assert data["pdf"]["x"][0] < min(expected_members)
    assert data["pdf"]["x"][-1] > max(expected_members)


def test_ensembles_members_match_statistics(client):
    """Statistics are computed from the exact member array returned."""
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}&include_members=true"
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    members = data["members"]

    expected = _expected_statistics(members)
    stats = data["statistics"]
    assert data["member_count"] == len(members)
    assert stats["mean"] == pytest.approx(expected["mean"])
    assert stats["median"] == pytest.approx(expected["median"])
    assert stats["spread"] == pytest.approx(expected["spread"])
    assert stats["p10"] == pytest.approx(expected["p10"])
    assert stats["p25"] == pytest.approx(expected["p25"])
    assert stats["p50"] == pytest.approx(expected["p50"])
    assert stats["p75"] == pytest.approx(expected["p75"])
    assert stats["p90"] == pytest.approx(expected["p90"])


def test_ensembles_cache_separates_members_flag(client):
    """include_members=true and false must not collide in the cache.

    A statistics-only cached response (no members) must never satisfy a
    distribution request, and vice versa. Two requests that differ only in the
    flag must produce distinct payloads (members present vs. absent) even when
    served from the cache on the second call.
    """
    stats_url = (
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}"
    )
    dist_url = stats_url + "&include_members=true"

    resp_stats = client.get(stats_url)
    resp_dist = client.get(dist_url)
    assert resp_stats.status_code == 200
    assert resp_dist.status_code == 200

    stats_data = resp_stats.json()["data"]
    dist_data = resp_dist.json()["data"]
    assert "members" not in stats_data
    assert "members" in dist_data

    # Second call on each URL must be served (cache hit) and still be distinct.
    resp_stats2 = client.get(stats_url)
    resp_dist2 = client.get(dist_url)
    assert "members" not in resp_stats2.json()["data"]
    assert "members" in resp_dist2.json()["data"]
    assert resp_stats2.json()["data"]["statistics"] == stats_data["statistics"]
    assert resp_dist2.json()["data"]["statistics"] == dist_data["statistics"]


def test_ensembles_include_members_degenerate_returns_pdf_null(monkeypatch, client):
    """When member values have zero variance, pdf is returned as null."""
    from api.services import ensemble_data

    monkeypatch.setattr(
        ensemble_data,
        "_gated_member_values",
        lambda *args, **kwargs: [20.0, 20.0, 20.0, 20.0, 20.0],
    )

    resp = client.get(
        f"/v1/ensembles?lat={LAT + 0.05}&lon={LON + 0.05}"
        "&variable=temperature_2m"
        f"&lead_time_hours={LEAD}&include_members=true"
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["members"] == [20.0, 20.0, 20.0, 20.0, 20.0]
    assert data["statistics"]["spread"] == 0.0
    assert "pdf" in data
    assert data["pdf"] is None


def test_ensembles_precipitation_amount_3h_and_phase_support(client):
    """Ensemble statistics for precipitation_amount_3h includes phase_support."""
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        f"&variable=precipitation_amount_3h&lead_time_hours={LEAD}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["model"] == "gefs"
    assert data["member_count"] == MEMBER_COUNT

    # Phase support is present and sums to 1.0
    assert "phase_support" in data
    ps = data["phase_support"]
    assert "rain" in ps
    assert "snow" in ps
    assert "dry" in ps
    assert "freezing_rain" in ps
    assert "ice_pellets" in ps
    assert "unknown" in ps
    assert sum(ps.values()) == pytest.approx(1.0, abs=1e-4)

    # Transition frequency is present
    assert "transition_frequency" in data


def test_ensembles_valid_time_resolution(client):
    """Under Lifecycle V2, /v1/ensembles serves by valid_time."""
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        "&valid_time=2026-07-21T06:00:00Z"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["model"] == "gefs"
    assert data["valid_time"] == "2026-07-21T06:00:00Z"
    assert data["source_cycle"] == "2026-07-21T00:00:00Z"
    assert data["lead_time_hours"] == 6


def test_ensembles_conflicting_valid_time_and_initial_time_rejected(client):
    """Passing both valid_time and initial_time returns 422."""
    resp = client.get(
        f"/v1/ensembles?lat={LAT}&lon={LON}"
        "&variable=temperature_2m"
        "&valid_time=2026-07-21T06:00:00Z"
        "&initial_time=2026-07-21T00:00:00Z"
    )
    assert resp.status_code == 422
    assert "Provide either valid_time or initial_time" in resp.json()["error"]["message"]
