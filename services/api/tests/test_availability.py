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
    assert {v["id"] for v in gfs["variables"]} == {
        "temperature_2m",
        "precipitation_rate",
        "precipitation_amount_3h",
        "cloud_cover_3h",
        "cloud_ceiling",
    }
    assert "wind_u_10m" not in {v["id"] for v in gfs["variables"]}
    assert "wind_v_10m" not in {v["id"] for v in gfs["variables"]}
    assert "crain" not in {v["id"] for v in gfs["variables"]}
    assert "csnow" not in {v["id"] for v in gfs["variables"]}

    # temperature_2m availability for the ready gfs run at 2026-07-21T00:00Z
    # with leads [0, 6, 12, 18] (the fixture dataset's lead coordinate).
    temp = next(v for v in gfs["variables"] if v["id"] == "temperature_2m")
    assert temp["unit"] == "°C"
    assert len(temp["initial_times"]) == 1
    initial = temp["initial_times"][0]
    assert initial["value"] == "2026-07-21T00:00:00Z"
    assert initial["lead_time_hours"] == [0, 6, 12, 18]

    # Authoritative map layer descriptor for zero-waterfall synchronous transition.
    layer = temp["layer"]
    assert layer is not None
    assert layer["tile_url_template"] == (
        "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}"
    )
    assert layer["min_zoom"] == 0
    assert layer["max_zoom"] == 9
    assert layer["legend"]["unit"] == "°C"
    assert len(layer["legend"]["stops"]) > 0


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


def test_availability_exposes_issuing_center_and_cadence(client):
    """Each model carries its issuing center and authoritative cycle cadence.

    These are additive fields consumed by the header status badge to group
    models per forecast center and to tell an old-but-healthy cycle apart from
    a stalled feed. The center is resolved from the ``models`` row already
    joined by the availability query, so no extra roundtrip is incurred.
    """
    resp = client.get("/v1/forecast/availability")
    by_id = {m["id"]: m for m in resp.json()["data"]["models"]}

    for model in by_id.values():
        assert model["center_id"] == "noaa"
        assert model["center_name"] == "National Oceanic and Atmospheric Administration"
        # Both seeded models are registered in domain.cadence as 6-hourly.
        assert model["cycle_cadence_hours"] == 6


def test_availability_etag_revalidation_returns_304(client):
    """A matching If-None-Match yields a bodyless 304 with the same headers.

    The frontend re-polls this ~444KB payload every 60s, so revalidation is
    what keeps that poll cheap. The tag must survive ``generated_at`` changing
    on every request -- an ETag over the whole payload would never match.
    """
    first = client.get("/v1/forecast/availability")
    assert first.status_code == 200
    etag = first.headers["ETag"]
    assert etag
    assert first.headers["Cache-Control"] == "no-cache"

    second = client.get("/v1/forecast/availability", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["ETag"] == etag
    assert second.headers["Cache-Control"] == "no-cache"


def test_availability_etag_differs_for_stale_if_none_match(client):
    """A non-matching If-None-Match still returns the full payload."""
    resp = client.get(
        "/v1/forecast/availability", headers={"If-None-Match": '"not-the-current-tag"'}
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["models"]
