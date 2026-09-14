"""Contract and integration tests for the vector field endpoint and caching.

Tests:
* GET /v1/maps/{model}/wind_10m/vector-field
* GFS deterministic canonical U/V
* GEFS ensemble consensus mean vector
* Metadata and URL template integration
* Server-side derived cache and generation isolation
* Error cases (404 missing model/lead, 422 validation error)
"""

from typing import Any

import numpy as np
import pytest
import xarray as xr

from domain.models.wind import (
    VECTOR_FIELD_MAGIC,
    decode_vector_field_int16,
)

from api.services import vector_field
from api.services.vector_field import (
    _select_and_encode_vector_field,
    _vector_cache,
    _vector_cache_get,
    _vector_cache_key,
    _vector_cache_set,
)


@pytest.fixture(autouse=True)
def _isolate_from_shared_redis(monkeypatch):
    """Keep cache tests on the process-local layer: no writes to a live Redis.

    A local/CI Redis would otherwise receive setex writes and leak entries
    across tests via the shared L2 read path.
    """
    monkeypatch.setattr(vector_field, "_get_redis_client", lambda: None)


def test_select_and_encode_vector_field_gfs_deterministic():
    """GFS deterministic vector field extracts exact U and V values with stride."""
    # 4x4 grid at 0.25 deg: lat 38..38.75, lon -107..-106.25
    u_vals = np.array([
        [10.0, 12.0, 14.0, 16.0],
        [18.0, 20.0, 22.0, 24.0],
        [26.0, 28.0, 30.0, 32.0],
        [34.0, 36.0, 38.0, 40.0],
    ], dtype=np.float32)
    v_vals = np.array([
        [-5.0, -6.0, -7.0, -8.0],
        [-9.0, -10.0, -11.0, -12.0],
        [-13.0, -14.0, -15.0, -16.0],
        [-17.0, -18.0, -19.0, -20.0],
    ], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "wind_u_10m": (("lead_time_hours", "latitude", "longitude"), u_vals[np.newaxis, ...]),
            "wind_v_10m": (("lead_time_hours", "latitude", "longitude"), v_vals[np.newaxis, ...]),
        },
        coords={
            "lead_time_hours": [6],
            "latitude": [38.75, 38.5, 38.25, 38.0],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )

    # Stride 2 decimation -> 2x2 grid
    payload = _select_and_encode_vector_field(ds, lead=6, stride=2)
    assert payload[:4] == VECTOR_FIELD_MAGIC

    u_dec, v_dec, meta = decode_vector_field_int16(payload)
    assert meta.lat_count == 2
    assert meta.lon_count == 2
    assert meta.lat_start == 38.75
    assert meta.lat_step == -0.5
    assert meta.lon_start == -107.0
    assert meta.lon_step == 0.5

    # Subsampled corners: (0,0), (0,2), (2,0), (2,2)
    expected_u = np.array([[10.0, 14.0], [26.0, 30.0]], dtype=np.float32)
    expected_v = np.array([[-5.0, -7.0], [-13.0, -15.0]], dtype=np.float32)
    np.testing.assert_allclose(u_dec, expected_u, atol=0.0051)
    np.testing.assert_allclose(v_dec, expected_v, atol=0.0051)


def test_select_and_encode_vector_field_gefs_consensus_mean():
    """GEFS consensus vector field computes (mean(u_i), mean(v_i)) across members."""
    # 2 members: member 0 (u=10, v=-10), member 1 (u=20, v=0)
    # Expected consensus mean vector: mean_u = 15.0, mean_v = -5.0
    u_data = np.zeros((2, 1, 4, 4), dtype=np.float32)
    u_data[0] = 10.0
    u_data[1] = 20.0

    v_data = np.zeros((2, 1, 4, 4), dtype=np.float32)
    v_data[0] = -10.0
    v_data[1] = 0.0

    ds = xr.Dataset(
        data_vars={
            "wind_u_10m": (("member", "lead_time_hours", "latitude", "longitude"), u_data),
            "wind_v_10m": (("member", "lead_time_hours", "latitude", "longitude"), v_data),
        },
        coords={
            "member": [1, 2],
            "lead_time_hours": [6],
            "latitude": [38.75, 38.5, 38.25, 38.0],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )

    payload = _select_and_encode_vector_field(ds, lead=6, stride=2)
    u_dec, v_dec, meta = decode_vector_field_int16(payload)

    assert meta.lat_count == 2
    assert meta.lon_count == 2
    np.testing.assert_allclose(u_dec, 15.0, atol=0.0051)
    np.testing.assert_allclose(v_dec, -5.0, atol=0.0051)


def test_select_and_encode_vector_field_missing_component_raises():
    ds_bad = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), np.zeros((1, 2, 2)))},
        coords={"lead_time_hours": [6], "latitude": [38.0, 39.0], "longitude": [-107.0, -106.0]},
    )
    with pytest.raises(ValueError, match="Variables 'wind_u_10m' and 'wind_v_10m' must be in the dataset"):
        _select_and_encode_vector_field(ds_bad, lead=6)


class _RecordingShardedReader:
    """Minimal sharded-reader test double recording read_window calls."""

    def __init__(self, has_mean: bool, member_u: dict[int, float], member_v: dict[int, float]):
        self.has_mean = has_mean
        self.member_u = member_u
        self.member_v = member_v
        self.calls: list[dict[str, object]] = []

    def has_mean_shard(self, variable: str, lead: int, *, generation: str | None = None) -> bool:
        return self.has_mean

    def read_window(
        self,
        variable: str,
        *,
        member: int | None,
        lead_time_hours: int,
        lat_min: int,
        lat_max: int,
        lon_min: int,
        lon_max: int,
        generation: str | None = None,
        is_mean: bool = False,
    ) -> np.ndarray[Any, Any]:
        self.calls.append({"variable": variable, "member": member, "is_mean": is_mean})
        window = np.full((lat_max - lat_min + 1, lon_max - lon_min + 1), np.nan, dtype=np.float32)
        if is_mean:
            window[:] = 15.0 if variable == "wind_u_10m" else -5.0
        else:
            assert member is not None
            window[:] = self.member_u[member] if variable == "wind_u_10m" else self.member_v[member]
        return window


@pytest.fixture()
def _sharded_store(monkeypatch):
    """Route the selector onto the sharded_v1 path with an injectable reader."""
    from api.core import manifest_reader as manifest_module
    from api.core import zarr as zarr_module

    holder: dict[str, object] = {}

    monkeypatch.setattr(manifest_module, "manifest_storage_format", lambda store_path: "sharded_v1")
    monkeypatch.setattr(manifest_module, "manifest_generation", lambda store_path: "gen-test")
    monkeypatch.setattr(
        zarr_module,
        "get_sharded_reader",
        lambda store_path: holder["reader"],
    )
    return holder


def _gefs_sharded_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "wind_u_10m": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((2, 1, 4, 4), dtype=np.float32)),
            "wind_v_10m": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((2, 1, 4, 4), dtype=np.float32)),
        },
        coords={
            "member": [1, 2],
            "lead_time_hours": [6],
            "latitude": [38.75, 38.5, 38.25, 38.0],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )


def test_select_and_encode_gefs_reads_official_mean_shard(_sharded_store):
    """Sharded GEFS vector field reads the 2 official mean shards directly."""
    reader = _RecordingShardedReader(
        has_mean=True, member_u={1: 10.0, 2: 20.0}, member_v={1: -10.0, 2: 0.0}
    )
    _sharded_store["reader"] = reader

    payload = _select_and_encode_vector_field(_gefs_sharded_dataset(), lead=6, stride=2, store_path="s3://bucket/store")
    u_dec, v_dec, meta = decode_vector_field_int16(payload)

    assert meta.lat_count == 2
    assert meta.lon_count == 2
    np.testing.assert_allclose(u_dec, 15.0, atol=0.0051)
    np.testing.assert_allclose(v_dec, -5.0, atol=0.0051)

    # Exactly one full-globe read per component from the official mean shard —
    # no per-member reads.
    assert reader.calls == [
        {"variable": "wind_u_10m", "member": None, "is_mean": True},
        {"variable": "wind_v_10m", "member": None, "is_mean": True},
    ]


def test_select_and_encode_gefs_falls_back_to_members_without_mean_shard(_sharded_store):
    """Without official mean shards the member-wise consensus mean is computed."""
    reader = _RecordingShardedReader(
        has_mean=False, member_u={1: 10.0, 2: 20.0}, member_v={1: -10.0, 2: 0.0}
    )
    _sharded_store["reader"] = reader

    payload = _select_and_encode_vector_field(_gefs_sharded_dataset(), lead=6, stride=2, store_path="s3://bucket/store")
    u_dec, v_dec, meta = decode_vector_field_int16(payload)

    assert meta.lat_count == 2
    assert meta.lon_count == 2
    np.testing.assert_allclose(u_dec, 15.0, atol=0.0051)
    np.testing.assert_allclose(v_dec, -5.0, atol=0.0051)

    assert len(reader.calls) == 4
    assert all(call["is_mean"] is False for call in reader.calls)
    assert {call["member"] for call in reader.calls} == {1, 2}


def test_vector_field_cache_serves_warm_hit():
    """Vector field cache returns cached bytes on warm request and isolates leads/cycles."""
    _vector_cache.clear()
    k1 = _vector_cache_key("gfs", "wind_10m", 6, None, "gen1", 2)
    k2 = _vector_cache_key("gfs", "wind_10m", 12, None, "gen1", 2)
    k_gen2 = _vector_cache_key("gfs", "wind_10m", 6, None, "gen2", 2)

    assert k1 != k2
    assert k1 != k_gen2
    assert _vector_cache_get(k1) is None

    _vector_cache_set(k1, b"CACHED_PAYLOAD")
    assert _vector_cache_get(k1) == b"CACHED_PAYLOAD"
    assert _vector_cache_get(k2) is None
    assert _vector_cache_get(k_gen2) is None
    _vector_cache.clear()


def test_vector_field_unknown_model_404(client):
    resp = client.get("/v1/maps/nonexistent/wind_10m/vector-field?lead_time_hours=6")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_vector_field_unavailable_lead_404(client):
    resp = client.get("/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=999")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_vector_field_negative_lead_422(client):
    resp = client.get("/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=-6")
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_maps_metadata_template_includes_vector_field_for_wind(client):
    """Maps metadata includes vector_field_url_template for wind_10m and None for others."""
    resp_temp = client.get("/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=6")
    assert resp_temp.status_code == 200
    assert resp_temp.json()["data"]["vector_field_url_template"] is None
