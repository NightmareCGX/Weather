"""R11 cache-key normalization regression tests (Lifecycle V3 §4 R11 / §11-11).

The single-variable endpoints bind their cache identity to the resolver's
actual valid_time, not the raw request string, so equivalent ISO 8601
spellings of the same instant (``...Z`` vs ``...+00:00``) resolve to one
source and must share one cache entry instead of fragmenting it.
"""

from urllib.parse import quote

from api.core.config import settings

#: A valid_time served by the seeded fixture catalog (gfs/gefs 00Z run + 6h).
VT = "2026-07-21T06:00:00Z"
#: The same instant spelled with an explicit UTC offset.
VT_ALT = "2026-07-21T06:00:00+00:00"


def _probability_keys(client_redis):
    """Return the set of probability cache keys currently in Redis."""
    return set(client_redis.scan_iter("probability:*"))


def _ensemble_keys(client_redis):
    return set(client_redis.scan_iter("ensemble:*"))


def _redis_client_or_skip():
    import redis as redis_lib

    try:
        client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
    except redis_lib.RedisError:
        import pytest

        pytest.skip("Redis not reachable; cache-key sharing cannot be observed")
    return client


def test_probability_valid_time_spelling_shares_cache_entry(client):
    """Two spellings of one valid_time must map to a single cache entry."""

    r = _redis_client_or_skip()
    url = (
        "/v1/probabilities?lat=38.19&lon=-106.82&variable=temperature_2m"
        "&threshold=10&operator=gt&model=gefs&valid_time="
    )
    before = _probability_keys(r)
    resp1 = client.get(url + quote(VT))
    assert resp1.status_code == 200
    after_first = _probability_keys(r)
    assert after_first - before, "first request should create a probability cache entry"

    resp2 = client.get(url + quote(VT_ALT))
    assert resp2.status_code == 200
    assert resp2.json() == resp1.json()
    after_second = _probability_keys(r)
    assert after_second == after_first, (
        "equivalent valid_time spellings must share one cache entry, "
        f"new keys appeared: {after_second - after_first}"
    )
    # Clean up the entries created by this test so the Redis delta logic of
    # sibling tests is not affected by TTL overlap.
    for key in after_first - before:
        r.delete(key)


def test_ensemble_valid_time_spelling_shares_cache_entry(client):
    """Same-key sharing contract for /v1/ensembles."""
    r = _redis_client_or_skip()
    url = (
        "/v1/ensembles?lat=38.0&lon=-107.0&model=gefs"
        "&variable=temperature_2m&valid_time="
    )
    before = _ensemble_keys(r)
    resp1 = client.get(url + quote(VT))
    assert resp1.status_code == 200
    after_first = _ensemble_keys(r)
    assert after_first - before, "first request should create an ensemble cache entry"

    resp2 = client.get(url + quote(VT_ALT))
    assert resp2.status_code == 200
    assert resp2.json() == resp1.json()
    after_second = _ensemble_keys(r)
    assert after_second == after_first, (
        "equivalent valid_time spellings must share one cache entry, "
        f"new keys appeared: {after_second - after_first}"
    )
    for key in after_first - before:
        r.delete(key)


def test_maps_template_binds_resolved_valid_time(client):
    """/v1/maps template URLs embed the normalized resolved valid_time (R11)."""
    resp = client.get(
        "/v1/maps?model=gfs&variable=temperature_2m"
        f"&level=surface&valid_time={quote(VT_ALT)}"
    )
    assert resp.status_code == 200
    template = resp.json()["data"]["tile_url_template"]
    assert "valid_time=2026-07-21T06:00:00Z" in template, (
        f"template must embed the normalized valid_time, got: {template}"
    )
