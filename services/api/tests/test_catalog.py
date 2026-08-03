"""Contract and integration tests for the Milestone 8 catalog endpoints.

These tests run against a real PostgreSQL instance via TestClient and verify
the response envelope, field mapping, filters, pagination, and headers defined
in ``docs/API.md`` Domain 1. When PostgreSQL is unreachable they skip, following
the existing ``test_migrations.py`` convention.
"""


def test_centers_contract(client):
    resp = client.get("/v1/centers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    center = body["data"][0]
    assert center["object"] == "center"
    assert center["id"] == "noaa"
    assert center["name"] == "National Oceanic and Atmospheric Administration"
    assert center["country"] == "USA"
    assert set(center.keys()) == {"id", "object", "name", "country"}


def test_models_contract_and_filters(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    models = {model["id"]: model for model in body["data"]}
    assert set(models) == {"gfs", "gefs"}
    for model in body["data"]:
        assert model["object"] == "model"
        assert model["center_id"] == "noaa"
        assert model["resolution_km"] == 25.0
    assert models["gfs"]["is_ensemble"] is False
    assert models["gefs"]["is_ensemble"] is True

    resp = client.get("/v1/models?center_id=noaa")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2

    resp = client.get("/v1/models?is_ensemble=true")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert [model["id"] for model in data] == ["gefs"]

    resp = client.get("/v1/models?is_ensemble=false")
    assert [model["id"] for model in resp.json()["data"]] == ["gfs"]

    # A filter that matches nothing yields an empty list envelope.
    resp = client.get("/v1/models?center_id=missing")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_runs_contract_and_model_id_resolution(client):
    resp = client.get("/v1/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    runs = body["data"]
    assert len(runs) == 3
    for run in runs:
        assert run["object"] == "run"
        assert run["status"] in {"ready", "processing"}
        assert run["model_id"] in {"gfs", "gefs"}

    by_id = {run["id"]: run for run in runs}
    gfs_run = by_id["run_2026072100_gfs"]
    assert gfs_run["model_id"] == "gfs"
    assert gfs_run["cycle_time"] == "2026-07-21T00:00:00Z"
    assert gfs_run["status"] == "ready"
    assert by_id["run_2026072100_gefs"]["model_id"] == "gefs"

    resp = client.get("/v1/runs?model_id=gfs")
    assert resp.status_code == 200
    assert {run["id"] for run in resp.json()["data"]} == {
        "run_2026072100_gfs",
        "run_2026072112_gfs",
    }

    resp = client.get("/v1/runs?status=ready")
    assert {run["id"] for run in resp.json()["data"]} == {
        "run_2026072100_gfs",
        "run_2026072100_gefs",
    }

    resp = client.get("/v1/runs?model_id=gfs&status=processing")
    data = resp.json()["data"]
    assert [run["id"] for run in data] == ["run_2026072112_gfs"]


def test_variables_contract(client):
    resp = client.get("/v1/variables")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    variables = {variable["id"]: variable for variable in body["data"]}
    assert variables["temperature_2m"]["name"] == "2-Meter Temperature"
    assert variables["temperature_2m"]["unit"] == "°C"
    assert variables["precipitation_rate"]["unit"] == "mm/h"
    for variable in body["data"]:
        assert variable["object"] == "variable"


def test_grids_contract(client):
    resp = client.get("/v1/grids")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    grids = {grid["id"]: grid for grid in body["data"]}
    assert grids["global_025deg"]["resolution_km"] == 25.0
    assert grids["downscaled_3km"]["resolution_km"] == 3.0
    for grid in body["data"]:
        assert grid["object"] == "grid"


def test_pagination_forward_and_has_more(client):
    # Models order by model_id ascending: gefs, gfs.
    resp = client.get("/v1/models?limit=1")
    assert resp.status_code == 200
    body = resp.json()
    assert [model["id"] for model in body["data"]] == ["gefs"]
    assert body["has_more"] is True
    assert body["next_cursor"] == "gefs"

    resp = client.get("/v1/models?limit=1&starting_after=gefs")
    assert resp.status_code == 200
    body = resp.json()
    assert [model["id"] for model in body["data"]] == ["gfs"]
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_backward(client):
    resp = client.get("/v1/models?limit=1&ending_before=gfs")
    assert resp.status_code == 200
    body = resp.json()
    assert [model["id"] for model in body["data"]] == ["gefs"]
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_backward_has_more(client):
    # Runs ordered by id ascending: run_2026072100_gefs, run_2026072100_gfs,
    # run_2026072112_gfs. One page before the last run returns the previous
    # run with more still ahead, so has_more is True and next_cursor allows
    # continuing further back.
    resp = client.get("/v1/runs?limit=1&ending_before=run_2026072112_gfs")
    assert resp.status_code == 200
    body = resp.json()
    assert [run["id"] for run in body["data"]] == ["run_2026072100_gfs"]
    assert body["has_more"] is True
    assert body["next_cursor"] == "run_2026072100_gfs"


def test_cache_control_headers(client):
    assert client.get("/v1/centers").headers["Cache-Control"] == "public, max-age=86400"
    assert client.get("/v1/models").headers["Cache-Control"] == "public, max-age=86400"
    assert client.get("/v1/variables").headers["Cache-Control"] == "public, max-age=86400"
    assert client.get("/v1/grids").headers["Cache-Control"] == "public, max-age=86400"
    assert client.get("/v1/runs").headers["Cache-Control"] == "public, max-age=300"


def test_request_id_header_present(client):
    for path in ("/v1/centers", "/v1/models", "/v1/runs", "/v1/variables", "/v1/grids"):
        resp = client.get(path)
        assert "X-Request-Id" in resp.headers
        assert resp.headers["X-Request-Id"].startswith("req_")


def test_validation_error_envelope(client):
    resp = client.get("/v1/models?limit=500")
    assert resp.status_code == 422
    body = resp.json()
    error = body["error"]
    assert error["code"] == "invalid_request_error"
    assert error["type"] == "validation_error"
    assert error["param"] == "limit"
    assert error["request_id"] == resp.headers["X-Request-Id"]


def test_mutually_exclusive_cursors_rejected(client):
    resp = client.get("/v1/models?starting_after=gfs&ending_before=gefs")
    assert resp.status_code == 422
    assert "error" in resp.json()


def test_not_found_error_envelope(client):
    resp = client.get("/v1/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "invalid_request_error"
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["request_id"] == resp.headers["X-Request-Id"]
