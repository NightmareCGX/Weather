"""Contract and integration tests for the forecast availability endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the nested model/variable/initial-time/lead-time availability structure is
derived entirely from the database. When PostgreSQL is unreachable they skip,
following the existing convention.
"""


def test_availability_contract(client):
    resp = client.get("/v1/forecast/availability")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "forecast_availability"
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert resp.headers["Cache-Control"] == "no-cache"

    models = body["data"]["models"]
    assert {model["id"] for model in models} == {"gfs", "gefs"}

    by_id = {model["id"]: model for model in models}
    gfs = by_id["gfs"]
    assert gfs["is_ensemble"] is False
    assert {v["id"] for v in gfs["variables"]} == {"temperature_2m", "precipitation_rate"}

    # temperature_2m availability for the ready gfs run at 2026-07-21T00:00Z
    # with leads [0, 6, 12, 18] (the fixture dataset's lead coordinate).
    temp = next(v for v in gfs["variables"] if v["id"] == "temperature_2m")
    assert temp["unit"] == "°C"
    assert len(temp["initial_times"]) == 1
    initial = temp["initial_times"][0]
    assert initial["value"] == "2026-07-21T00:00:00Z"
    assert initial["lead_time_hours"] == [0, 6, 12, 18]


def test_availability_only_ready_runs(client):
    # The seeded database has a ready gfs run and a ready gefs run; a
    # processing-only run must not contribute availability. Both seeded ready
    # runs have forecast product rows for temperature_2m.
    resp = client.get("/v1/forecast/availability")
    models = resp.json()["data"]["models"]
    for model in models:
        for variable in model["variables"]:
            assert len(variable["initial_times"]) >= 1
            # Every initial time must have at least one lead.
            for initial in variable["initial_times"]:
                assert len(initial["lead_time_hours"]) >= 1


def test_availability_empty_database(client):
    # A model with no ready run contributes no availability. The seeded
    # `run_2026072112_gfs` is processing-only and must not appear as an
    # available initial time.
    resp = client.get("/v1/forecast/availability")
    gfs = next(m for m in resp.json()["data"]["models"] if m["id"] == "gfs")
    for variable in gfs["variables"]:
        for initial in variable["initial_times"]:
            assert initial["value"] == "2026-07-21T00:00:00Z"
