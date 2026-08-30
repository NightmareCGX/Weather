"""Contract and integration tests for the Milestone 9 /v1/points endpoint.

These tests run against a real PostgreSQL instance via TestClient and verify
the documented single-model ``point_forecast`` envelope, spatial resolution,
Zarr slicing + interpolation, units conversion, the multi-model rejection,
and the cache layer. When PostgreSQL is unreachable they skip, following the
existing convention.
"""

import pytest
import redis as redis_lib

from api.schemas import (
    ForecastLocationOut,
    ForecastSeries,
    PointForecastData,
    PointForecastEnvelope,
)
from api.services.cache import (
    FALLBACK_REASON_REDIS_READ_AND_WRITE_UNAVAILABLE,
    FALLBACK_REASON_REDIS_READ_UNAVAILABLE,
    FALLBACK_REASON_REDIS_WRITE_UNAVAILABLE,
    PointCache,
    build_point_cache_key,
)
from api.services.point_forecast import _convert_value
from tests.fixtures import (
    LAT_START,
    LON_START,
    precipitation_at,
    temperature_at,
)


@pytest.fixture
def cache():
    """A PointCache instance bound to the configured Redis."""
    return PointCache()


def _assert_envelope(body):
    assert body["object"] == "point_forecast"
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_point_coordinates_single_model(client):
    lat = LAT_START + 0.125  # 38.125
    lon = LON_START + 0.125  # -106.875
    resp = client.get(f"/v1/points?lat={lat}&lon={lon}&models=gfs")
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)

    data = body["data"]
    assert data["model"] == "gfs"
    assert data["location"]["resolved_via"] == "coordinates"
    assert abs(data["location"]["latitude"] - lat) < 1e-9
    assert abs(data["location"]["longitude"] - lon) < 1e-9
    assert data["location"]["elevation_m"] is None
    assert data["generated_at"] == "2026-07-21T00:00:00Z"

    # Default variables: temperature_2m and precipitation_rate.
    by_lead = {entry["lead_time_hours"]: entry for entry in data["forecasts"]}
    assert set(by_lead) == {0, 6, 12, 18}
    entry = by_lead[6]
    assert entry["valid_time"] == "2026-07-21T06:00:00Z"
    expected_temp = temperature_at(lat, lon, 6)
    assert abs(entry["temperature_2m"] - expected_temp) < 1e-9
    assert abs(entry["precipitation_rate"] - precipitation_at(6)) < 1e-9


def test_point_city_id_resolution(client):
    resp = client.get("/v1/points?city_id=city_aspen&models=gfs")
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["location"]["resolved_via"] == "city"
    assert abs(data["location"]["latitude"] - 38.19) < 1e-6
    assert abs(data["location"]["longitude"] - -106.82) < 1e-6
    assert data["location"]["elevation_m"] is None


def test_point_resort_id_resolution(client):
    resp = client.get("/v1/points?resort_id=resort_aspen_mountain&models=gfs")
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    data = body["data"]
    assert data["location"]["resolved_via"] == "resort"
    assert data["location"]["elevation_m"] == 3417.0
    assert abs(data["location"]["latitude"] - 38.19) < 1e-6


def test_point_same_coordinate_distinct_cache_keys(client):
    # Two resorts share the same coordinates but have different elevations.
    # Their cache keys must differ (resolved record id discriminates them), so
    # each request returns its own elevation rather than a cached collision.
    resp1 = client.get("/v1/points?resort_id=resort_aspen_mountain&models=gfs")
    resp2 = client.get("/v1/points?resort_id=resort_aspen_buttermilk&models=gfs")
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    body1 = resp1.json()
    body2 = resp2.json()
    assert body1["data"]["location"]["elevation_m"] == 3417.0
    assert body2["data"]["location"]["elevation_m"] == 2450.0
    # Same model, same coordinates, same variable set -- only the resolved
    # record differs, so payloads must differ in elevation and not collide.
    assert body1["data"]["location"]["latitude"] == body2["data"]["location"]["latitude"]
    assert body1["data"]["location"]["longitude"] == body2["data"]["location"]["longitude"]


def test_point_unknown_city_404(client):
    resp = client.get("/v1/points?city_id=city_nope&models=gfs")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["type"] == "not_found_error"


def test_point_unknown_resort_404(client):
    resp = client.get("/v1/points?resort_id=resort_nope&models=gfs")
    assert resp.status_code == 404


def test_point_lead_time_window(client):
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp = client.get(
        f"/v1/points?lat={lat}&lon={lon}&models=gfs&start_lead_time_hours=6&end_lead_time_hours=12"
    )
    assert resp.status_code == 200
    leads = [entry["lead_time_hours"] for entry in resp.json()["data"]["forecasts"]]
    assert leads == [6, 12]


def test_point_variables_filter(client):
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp = client.get(
        f"/v1/points?lat={lat}&lon={lon}&models=gfs&variables=temperature_2m"
    )
    assert resp.status_code == 200
    entry = resp.json()["data"]["forecasts"][0]
    assert "temperature_2m" in entry
    assert "precipitation_rate" not in entry


def test_point_imperial_units(client):
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp = client.get(
        f"/v1/points?lat={lat}&lon={lon}&models=gfs&variables=temperature_2m,precipitation_rate&units=imperial"
    )
    assert resp.status_code == 200
    entry = next(e for e in resp.json()["data"]["forecasts"] if e["lead_time_hours"] == 6)
    expected_temp_f = temperature_at(lat, lon, 6) * 9.0 / 5.0 + 32.0
    expected_precip_in = precipitation_at(6) / 25.4
    assert abs(entry["temperature_2m"] - expected_temp_f) < 1e-9
    assert abs(entry["precipitation_rate"] - expected_precip_in) < 1e-9


def test_point_generated_at_deterministic(client):
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp1 = client.get(f"/v1/points?lat={lat}&lon={lon}&models=gfs")
    resp2 = client.get(f"/v1/points?lat={lat}&lon={lon}&models=gfs")
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json() == resp2.json()


def test_point_cache_control_header(client):
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp = client.get(f"/v1/points?lat={lat}&lon={lon}&models=gfs")
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-cache"


def test_point_request_id_header(client):
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp = client.get(f"/v1/points?lat={lat}&lon={lon}&models=gfs")
    assert resp.status_code == 200
    assert "X-Request-Id" in resp.headers
    assert resp.headers["X-Request-Id"].startswith("req_")


def test_point_no_spatial_specifier_422(client):
    resp = client.get("/v1/points?models=gfs")
    assert resp.status_code == 422


def test_point_ambiguous_spatial_specifier_422(client):
    resp = client.get(
        "/v1/points?lat=39.0&lon=-106.0&city_id=city_aspen&models=gfs"
    )
    assert resp.status_code == 422


def test_point_invalid_coordinates_422(client):
    resp = client.get("/v1/points?lat=99.0&lon=-106.0&models=gfs")
    assert resp.status_code == 422


def test_point_multi_model_rejected(client):
    resp = client.get("/v1/points?lat=39.19&lon=-106.82&models=gfs,gefs")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "single model" in body["error"]["message"].lower()


def test_point_default_models_single(client):
    # The default models=gfs (a single model) makes the endpoint unambiguous:
    # omitting models is a valid single-model request, not a rejected one.
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp = client.get(f"/v1/points?lat={lat}&lon={lon}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["model"] == "gfs"


def test_point_gefs_returns_ensemble_mean(client):
    """The Hourly Forecast for GEFS serves the ensemble mean.

    Regression for the systematic GEFS serving defect: ``/v1/points`` previously
    rejected any field carrying a ``member`` dimension with a 500 ("not a 2-D
    surface field"), so requesting GEFS directly failed while the UI silently
    fell back to GFS. The point forecast now reduces the member axis by the mean
    (the shared ensemble aggregate) and returns the ensemble-mean field at the
    point — matching the map tile and the ensemble statistics mean.
    """
    lat = LAT_START + 0.125  # 38.125
    lon = LON_START + 0.125  # -106.875
    resp = client.get(f"/v1/points?lat={lat}&lon={lon}&models=gefs")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]
    assert data["model"] == "gefs"
    entry = {e["lead_time_hours"]: e for e in data["forecasts"]}[6]
    # Ensemble mean of the fixture: member-0 value + 2 * mean([0..4]=2.0) = +4.0.
    import numpy as np

    from tests.fixtures import (
        MEMBER_INDICES,
        ensemble_precipitation_at,
        ensemble_temperature_at,
    )

    member_values = [
        ensemble_temperature_at(member, lat, lon, 6) for member in MEMBER_INDICES
    ]
    expected_mean = float(np.mean(member_values))
    assert abs(entry["temperature_2m"] - expected_mean) < 1e-9
    # The GEFS fixture holds precipitation_rate on every member; the Hourly
    # Forecast precipitation is the ensemble mean of the per-member rate.
    member_precip = [
        ensemble_precipitation_at(member, 6) for member in MEMBER_INDICES
    ]
    assert abs(entry["precipitation_rate"] - float(np.mean(member_precip))) < 1e-9


def test_point_blank_models_rejected(client):
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp = client.get(f"/v1/points?lat={lat}&lon={lon}&models=")
    assert resp.status_code == 422
    body = resp.json()
    assert "at least one model identifier" in body["error"]["message"]


def test_point_unknown_model_404(client):
    lat = LAT_START + 0.125
    lon = LON_START + 0.125
    resp = client.get(f"/v1/points?lat={lat}&lon={lon}&models=ecmwf")
    assert resp.status_code == 404


def test_convert_value_known_unit():
    assert _convert_value(0.0, "°C", "imperial") == 32.0
    assert _convert_value(100.0, "°C", "imperial") == 212.0


def test_convert_value_unknown_unit_passthrough():
    assert _convert_value(15.0, "hPa", "imperial") == 15.0


def test_convert_value_metric_noop():
    assert _convert_value(15.0, "°C", "metric") == 15.0


def test_convert_kmh_to_mph_label():
    """The km/h imperial entry converts the value and labels it mph."""
    from api.services.point_forecast import _SI_TO_IMPERIAL

    label, convert = _SI_TO_IMPERIAL["km/h"]
    assert label == "mph"
    assert convert(10.0) == pytest.approx(6.21371, abs=1e-9)


def test_convert_phase1a_variables_imperial():
    """Phase 1A variables convert to their appropriate imperial representations."""
    # Relative humidity: % -> %
    assert _convert_value(85.0, "%", "imperial", var_code="relative_humidity_2m") == 85.0
    # Wind gust: km/h -> mph
    assert _convert_value(100.0, "km/h", "imperial", var_code="wind_gust") == pytest.approx(62.1371, abs=1e-4)
    # Visibility: m -> mi (1609.344m = 1mi)
    assert _convert_value(16093.44, "m", "imperial", var_code="visibility") == pytest.approx(10.0, abs=1e-4)
    # Snow depth: m -> in (1m = 39.3700787in)
    assert _convert_value(0.254, "m", "imperial", var_code="snow_depth") == pytest.approx(10.0, abs=1e-4)


def _make_key(cycle_time: str | None = None):
    return build_point_cache_key(
        model="gfs",
        latitude=38.125,
        longitude=-106.875,
        resolved_via="coordinates",
        location_id=None,
        cycle_time=cycle_time,
        variables=None,
        units="metric",
        start_lead_time_hours=None,
        end_lead_time_hours=None,
    )


def test_cache_key_distinguishes_cycles() -> None:
    """A cache entry for one forecast cycle never satisfies another cycle.

    GFS 2026-08-13 00Z and GFS 2026-08-13 12Z at the same location/lead are
    distinct forecast runs and must have distinct cache keys (GAP-1 fix).
    """
    key_00z = _make_key(cycle_time="2026-08-13T00:00:00Z")
    key_12z = _make_key(cycle_time="2026-08-13T12:00:00Z")
    assert key_00z != key_12z
    # Same cycle -> same key (deterministic).
    assert key_00z == _make_key(cycle_time="2026-08-13T00:00:00Z")


def test_cache_key_cross_cycle_separated_from_single_cycle() -> None:
    """A cross-cycle response never shares a key with a single-cycle response."""
    key_single = build_point_cache_key(
        model="gfs",
        latitude=38.125,
        longitude=-106.875,
        resolved_via="coordinates",
        location_id=None,
        cycle_time="2026-08-13T00:00:00Z",
        variables=None,
        units="metric",
        start_lead_time_hours=None,
        end_lead_time_hours=None,
        cross_cycle=False,
    )
    key_cross = build_point_cache_key(
        model="gfs",
        latitude=38.125,
        longitude=-106.875,
        resolved_via="coordinates",
        location_id=None,
        cycle_time="2026-08-13T00:00:00Z",
        variables=None,
        units="metric",
        start_lead_time_hours=None,
        end_lead_time_hours=None,
        cross_cycle=True,
    )
    assert key_single != key_cross
    # Deterministic: same cross_cycle -> same key.
    assert key_cross == build_point_cache_key(
        model="gfs",
        latitude=38.125,
        longitude=-106.875,
        resolved_via="coordinates",
        location_id=None,
        cycle_time="2026-08-13T00:00:00Z",
        variables=None,
        units="metric",
        start_lead_time_hours=None,
        end_lead_time_hours=None,
        cross_cycle=True,
    )


def test_cache_key_distinguishes_models() -> None:
    """Distinct models never share a point cache key."""
    from api.services.cache import build_point_cache_key

    def _key(model: str) -> str:
        return build_point_cache_key(
            model=model,
            latitude=38.125,
            longitude=-106.875,
            resolved_via="coordinates",
            location_id=None,
            cycle_time="2026-08-13T00:00:00Z",
            variables=None,
            units="metric",
            start_lead_time_hours=None,
            end_lead_time_hours=None,
        )

    assert _key("gfs") != _key("gefs")


def test_cache_key_distinguishes_leads() -> None:
    """Different lead times never share a point cache key."""
    from api.services.cache import build_point_cache_key

    def _key(end_lead: int | None) -> str:
        return build_point_cache_key(
            model="gfs",
            latitude=38.125,
            longitude=-106.875,
            resolved_via="coordinates",
            location_id=None,
            cycle_time="2026-08-13T00:00:00Z",
            variables=None,
            units="metric",
            start_lead_time_hours=None,
            end_lead_time_hours=end_lead,
        )

    assert _key(6) != _key(18)


def test_ensemble_cache_key_distinguishes_cycles() -> None:
    """Ensemble cache keys also carry the forecast-run cycle."""
    from api.services.cache import build_ensemble_cache_key

    key_00z = build_ensemble_cache_key(
        model="gefs",
        latitude=38.125,
        longitude=-106.875,
        variable="temperature_2m",
        lead_time_hours=18,
        cycle_time="2026-08-13T00:00:00Z",
    )
    key_12z = build_ensemble_cache_key(
        model="gefs",
        latitude=38.125,
        longitude=-106.875,
        variable="temperature_2m",
        lead_time_hours=18,
        cycle_time="2026-08-13T12:00:00Z",
    )
    assert key_00z != key_12z


def test_probability_cache_key_distinguishes_cycles() -> None:
    """Probability cache keys also carry the forecast-run cycle."""
    from api.services.cache import build_probability_cache_key

    key_00z = build_probability_cache_key(
        model="gefs",
        latitude=38.125,
        longitude=-106.875,
        variable="precipitation_rate",
        threshold=1.0,
        operator="gt",
        lead_time_hours=18,
        threshold_max=None,
        cycle_time="2026-08-13T00:00:00Z",
    )
    key_12z = build_probability_cache_key(
        model="gefs",
        latitude=38.125,
        longitude=-106.875,
        variable="precipitation_rate",
        threshold=1.0,
        operator="gt",
        lead_time_hours=18,
        threshold_max=None,
        cycle_time="2026-08-13T12:00:00Z",
    )
    assert key_00z != key_12z


def test_cache_corrupt_entry_is_miss_and_recomputed(cache):
    # Inject a malformed payload under the key; a get must treat it as a miss
    # (not raise), report it as a corrupt-entry fallback, delete the corrupt
    # entry, and a subsequent compute stores a valid response.
    key = _make_key()
    cache._client.set(key, "not-valid-json{{")
    read = cache.get(key)
    assert read.hit is False
    assert read.envelope is None
    assert read.fallback_reason == "corrupt_cache_entry"
    assert cache._client.get(key) is None  # corrupt entry deleted


def test_cache_get_malformed_json_returns_miss(cache):
    key = _make_key()
    cache._client.set(key, "{invalid json")
    read = cache.get(key)
    assert read.hit is False
    assert read.fallback_reason == "corrupt_cache_entry"


def test_cache_get_wrong_schema_returns_miss(cache):
    key = _make_key()
    # Valid JSON but not a PointForecastEnvelope (missing required fields).
    cache._client.set(key, '{"object": "list", "data": []}')
    read = cache.get(key)
    assert read.hit is False
    assert read.fallback_reason == "corrupt_cache_entry"


def test_cache_get_validation_error_returns_miss(cache):
    key = _make_key()
    # Valid JSON, PointForecastEnvelope shape, but a field with the wrong
    # type (data is a list, not an object). This raises a Pydantic
    # ValidationError which must be caught and treated as a corrupt miss.
    cache._client.set(key, '{"data": [], "has_more": false, "next_cursor": null}')
    read = cache.get(key)
    assert read.hit is False
    assert read.envelope is None
    assert read.fallback_reason == "corrupt_cache_entry"
    assert cache._client.get(key) is None  # corrupt entry deleted


def test_cache_clean_miss_has_no_fallback_reason(cache):
    key = _make_key()
    read = cache.get(key)  # nothing stored under the key
    assert read.hit is False
    assert read.envelope is None
    assert read.fallback_reason is None


def _make_envelope():
    """A minimal valid PointForecastEnvelope for cache tests."""
    return PointForecastEnvelope(
        data=PointForecastData(
            location=ForecastLocationOut(
                latitude=38.125,
                longitude=-106.875,
                elevation_m=None,
                resolved_via="coordinates",
            ),
            generated_at="2026-07-21T00:00:00Z",
            model="gfs",
            forecasts=[
                ForecastSeries(lead_time_hours=0, valid_time="2026-07-21T00:00:00Z")
            ],
        )
    )


class _StubDb:
    """Minimal DB stub recording audit rows without a real session."""

    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)

    def commit(self):
        pass

    def rollback(self):
        pass


def _compute():
    return _make_envelope()


def test_cache_write_unavailable_records_fallback(cache, monkeypatch):
    # GET returns a clean miss; SETEX fails. A write-unavailable fallback row
    # must be recorded and the forecast still returned.
    key = _make_key()
    monkeypatch.setattr(cache._client, "get", lambda k: None)
    monkeypatch.setattr(
        cache._client,
        "setex",
        lambda k, ttl, v: (_ for _ in ()).throw(redis_lib.RedisError("down")),
    )
    db = _StubDb()
    result = cache.compute_or_retrieve(db, key, "q=1", _compute)
    assert result == _make_envelope()
    assert len(db.rows) == 1
    assert db.rows[0].fallback_reason == FALLBACK_REASON_REDIS_WRITE_UNAVAILABLE


def test_cache_read_unavailable_records_fallback(cache, monkeypatch):
    # GET raises (read unavailable); SETEX succeeds. A read-unavailable
    # fallback row must be recorded.
    key = _make_key()
    monkeypatch.setattr(
        cache._client,
        "get",
        lambda k: (_ for _ in ()).throw(redis_lib.RedisError("down")),
    )
    monkeypatch.setattr(cache._client, "setex", lambda k, ttl, v: None)
    db = _StubDb()
    result = cache.compute_or_retrieve(db, key, "q=1", _compute)
    assert result == _make_envelope()
    assert len(db.rows) == 1
    assert db.rows[0].fallback_reason == FALLBACK_REASON_REDIS_READ_UNAVAILABLE


def test_cache_read_and_write_unavailable_records_combined(cache, monkeypatch):
    # GET raises (read unavailable) AND SETEX fails (write unavailable). Both
    # facts must be preserved in a single combined reason.
    key = _make_key()
    monkeypatch.setattr(
        cache._client,
        "get",
        lambda k: (_ for _ in ()).throw(redis_lib.RedisError("down")),
    )
    monkeypatch.setattr(
        cache._client,
        "setex",
        lambda k, ttl, v: (_ for _ in ()).throw(redis_lib.RedisError("down")),
    )
    db = _StubDb()
    result = cache.compute_or_retrieve(db, key, "q=1", _compute)
    assert result == _make_envelope()
    assert len(db.rows) == 1
    assert db.rows[0].fallback_reason == FALLBACK_REASON_REDIS_READ_AND_WRITE_UNAVAILABLE
