"""Contract and integration tests for the /v1/verifications endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the response envelope, metric aggregation (pool-all over matching forecast
products), error handling, and headers defined in ``docs/API.md`` Domain 7.
When PostgreSQL is unreachable they skip, following the existing suite
convention.
"""

import math
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from api.models.entities import ForecastProduct, ModelRun, VerificationObservation

#: The single-pair window: only the conftest observations land here.
SINGLE_PAIR_WINDOW = "2026-07-21"
#: The multi-candidate window: only the fixture-seeded observation lands here.
MULTI_CANDIDATE_WINDOW = "2026-07-28"
#: A window with no observations (empty verification result).
EMPTY_WINDOW = "2026-08-01"


def _assert_envelope(body: dict) -> None:
    """Assert the universal verification_report envelope shape."""
    assert body["object"] == "verification_report"
    assert body["has_more"] is False
    assert body["next_cursor"] is None


@pytest.fixture(scope="module")
def multi_cycle_seed(migrated_db, seed_data, tmp_zarr_stores):
    """Add a second ready gfs cycle so one observation pairs with two products.

    Two ready runs (2026-07-28T00:00Z with lead 12 and 2026-07-28T06:00Z with
    lead 6) both verify the same valid time 2026-07-28T12:00Z, exercising the
    pool-all aggregation across multiple cycles and lead times.
    """
    with Session(migrated_db) as session:
        session.add_all(
            [
                ModelRun(
                    id="run_2026072800_gfs",
                    model_version_id="version_gfs_v1",
                    cycle_time=datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc),
                    status="ready",
                    zarr_store_path=tmp_zarr_stores["gfs"],
                ),
                ModelRun(
                    id="run_2026072806_gfs",
                    model_version_id="version_gfs_v1",
                    cycle_time=datetime(2026, 7, 28, 6, 0, tzinfo=timezone.utc),
                    status="ready",
                    zarr_store_path=tmp_zarr_stores["gfs"],
                ),
            ]
        )
        session.add_all(
            [
                ForecastProduct(
                    id="product_gfs_20260728_00z_temperature_2m_12",
                    run_id="run_2026072800_gfs",
                    variable_id="temperature_2m",
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=12,
                ),
                ForecastProduct(
                    id="product_gfs_20260728_06z_temperature_2m_6",
                    run_id="run_2026072806_gfs",
                    variable_id="temperature_2m",
                    grid_id="global_025deg",
                    product_type="surface",
                    lead_time_hours=6,
                ),
            ]
        )
        session.add(
            VerificationObservation(
                id="obs_20260728_12z_temperature_2m",
                station_id="KASE",
                valid_time=datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
                variable_code="temperature_2m",
                observed_value=18.0,
            )
        )
        session.commit()
    yield


def test_verification_single_pair_metrics(client):
    """A window with one observation per variable yields exact metrics.

    Conftest seeds KASE temperature at 2026-07-21T06:00Z (00Z+6h forecast value
    17.0, observed 20.0) and precipitation at 2026-07-21T18:00Z (00Z+18h
    forecast value 9.0, observed 12.0); each error is -3.0.
    """
    resp = client.get(
        f"/v1/verifications?model=gfs&start_date={SINGLE_PAIR_WINDOW}"
        f"&end_date={SINGLE_PAIR_WINDOW}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["model"] == "gfs"
    assert data["period"] == {"start": SINGLE_PAIR_WINDOW, "end": SINGLE_PAIR_WINDOW}
    assert data["metrics"]["temperature_2m_bias"] == pytest.approx(-3.0)
    assert data["metrics"]["temperature_2m_mae"] == pytest.approx(3.0)
    assert data["metrics"]["temperature_2m_rmse"] == pytest.approx(3.0)
    assert data["metrics"]["precipitation_rate_bias"] == pytest.approx(-3.0)
    assert data["metrics"]["precipitation_rate_mae"] == pytest.approx(3.0)
    assert data["metrics"]["precipitation_rate_rmse"] == pytest.approx(3.0)


def test_verification_multicandidate_pooling(client, multi_cycle_seed):
    """Two products verifying the same valid time are both pooled.

    The observation at 2026-07-28T12:00Z pairs with 00Z+12h (forecast 20.0) and
    06Z+6h (forecast 17.0); observed is 18.0, so errors are [2.0, -1.0]:
    bias 0.5, MAE 1.5, RMSE sqrt(2.5).
    """
    resp = client.get(
        f"/v1/verifications?model=gfs&start_date={MULTI_CANDIDATE_WINDOW}"
        f"&end_date={MULTI_CANDIDATE_WINDOW}"
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["metrics"]["temperature_2m_bias"] == pytest.approx(0.5)
    assert data["metrics"]["temperature_2m_mae"] == pytest.approx(1.5)
    assert data["metrics"]["temperature_2m_rmse"] == pytest.approx(math.sqrt(2.5))


def test_verification_empty_result(client):
    """A valid request with no observations returns 200 and empty metrics."""
    resp = client.get(
        f"/v1/verifications?model=gfs&start_date={EMPTY_WINDOW}&end_date={EMPTY_WINDOW}"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    assert body["data"]["metrics"] == {}


def test_verification_unknown_model_404(client):
    resp = client.get(
        f"/v1/verifications?model=nope&start_date={SINGLE_PAIR_WINDOW}"
        f"&end_date={SINGLE_PAIR_WINDOW}"
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "invalid_request_error"
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["request_id"] == resp.headers["X-Request-Id"]


def test_verification_inverted_dates_400(client):
    resp = client.get(
        "/v1/verifications?model=gfs&start_date=2026-07-22&end_date=2026-07-21"
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "invalid_request_error"
    assert body["error"]["type"] == "invalid_request_error"


def test_verification_missing_params_422(client):
    resp = client.get(
        f"/v1/verifications?start_date={SINGLE_PAIR_WINDOW}&end_date={SINGLE_PAIR_WINDOW}"
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["param"] == "model"


def test_verification_cache_control(client):
    resp = client.get(
        f"/v1/verifications?model=gfs&start_date={SINGLE_PAIR_WINDOW}"
        f"&end_date={SINGLE_PAIR_WINDOW}"
    )
    assert resp.headers["Cache-Control"] == "no-cache"


def test_verification_request_id_header(client):
    resp = client.get(
        f"/v1/verifications?model=gfs&start_date={SINGLE_PAIR_WINDOW}"
        f"&end_date={SINGLE_PAIR_WINDOW}"
    )
    assert "X-Request-Id" in resp.headers
    assert resp.headers["X-Request-Id"].startswith("req_")
