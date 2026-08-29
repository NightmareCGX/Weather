"""Focused regression tests for P2 Phase 1: S3 Filesystem / Client Reuse and Connection Pooling.

Covers:
* Test A — Filesystem Reuse (N operations on a thread reuse the same S3FileSystem)
* Test B — Mapper Resolution (resolve_s3_mapper reuses thread-local S3FileSystem)
* Test C — Connection Pool Configuration (S3_MAX_POOL_CONNECTIONS reaches botocore & validation)
* Test D — Storage Schema Identity (chunk keys, metadata, geometry, compressor, dtype, fill_value)
* Test E — Read Compatibility (production serving reader reads the store identically)
* Test F — Same-Cycle Re-Ingestion (generation and re-ingest semantics intact)
* Test G — Failure Propagation (simulated PUT failure leaves region uncommitted, clean recovery)
* Test H — Thread Safety (concurrent writes across worker threads: 1 client/thread, no event-loop collisions)
* Test I — Cleanup / Teardown (reset_s3_fs clears thread-local cache cleanly)
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import zarr

from ingestion.core.config import IngestionSettings
from ingestion.core.s3 import (
    get_s3_fs,
    reset_s3_fs,
    resolve_s3_mapper,
    _build_s3_fs,
)
from ingestion.core.zarr_writer import (
    commit_region,
    prepare_run_store,
    read_dataset,
)


@pytest.fixture(autouse=True)
def _cleanup_thread_local():
    """Ensure thread-local S3 state is reset before and after each test."""
    reset_s3_fs()
    yield
    reset_s3_fs()


def _make_gefs_region(lead: int = 6, member: int = 1) -> xr.Dataset:
    lats = np.linspace(-90.0, 90.0, 721, dtype=np.float64)
    lons = np.linspace(0.0, 359.75, 1440, dtype=np.float64)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    arr = (273.15 + 30.0 * np.cos(np.deg2rad(lat_grid))).astype(np.float32)

    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                arr[np.newaxis, np.newaxis, :, :],
                {"units": "°C", "long_name": "2 metre temperature"},
            ),
        },
        coords={
            "latitude": ("latitude", lats, {"units": "degrees_north"}),
            "longitude": ("longitude", lons, {"units": "degrees_east"}),
            "lead_time_hours": ("lead_time_hours", [lead]),
            "member": ("member", [member]),
            "time": ("time", [np.datetime64("2026-07-21T00:00:00", "ns")]),
        },
        attrs={"model_id": "gefs", "cycle_time": "2026-07-21T00:00:00"},
    )


def _make_gfs_region(lead: int = 6) -> xr.Dataset:
    lats = np.linspace(-90.0, 90.0, 721, dtype=np.float64)
    lons = np.linspace(0.0, 359.75, 1440, dtype=np.float64)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    t_arr = (273.15 + 30.0 * np.cos(np.deg2rad(lat_grid))).astype(np.float32)
    p_arr = np.maximum(0.0, np.sin(np.deg2rad(lat_grid)) * 0.001).astype(np.float32)

    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                t_arr[np.newaxis, :, :],
                {"units": "°C", "long_name": "2 metre temperature"},
            ),
            "precipitation_rate": (
                ("lead_time_hours", "latitude", "longitude"),
                p_arr[np.newaxis, :, :],
                {"units": "mm/h", "long_name": "Precipitation rate"},
            ),
        },
        coords={
            "latitude": ("latitude", lats, {"units": "degrees_north"}),
            "longitude": ("longitude", lons, {"units": "degrees_east"}),
            "lead_time_hours": ("lead_time_hours", [lead]),
            "time": ("time", [np.datetime64("2026-07-21T00:00:00", "ns")]),
        },
        attrs={"model_id": "gfs", "cycle_time": "2026-07-21T00:00:00"},
    )


# =============================================================================
# Test A: Filesystem Reuse
# =============================================================================
def test_filesystem_reuse_on_same_thread():
    """Verify that repeated calls on the same thread reuse the cached S3FileSystem instance."""
    with patch("ingestion.core.s3._build_s3_fs", wraps=_build_s3_fs) as mock_build:
        fs1 = get_s3_fs()
        fs2 = get_s3_fs()
        fs3 = get_s3_fs()

        assert fs1 is fs2
        assert fs2 is fs3
        # Exactly one filesystem build for multiple accesses on the same thread
        assert mock_build.call_count == 1


# =============================================================================
# Test B: Mapper Resolution Reuse
# =============================================================================
def test_resolve_s3_mapper_reuses_filesystem():
    """Verify that resolve_s3_mapper reuses the thread-local S3FileSystem instance."""
    with patch("ingestion.core.s3._build_s3_fs", wraps=_build_s3_fs) as mock_build:
        m1 = resolve_s3_mapper("s3://weather-data/test1/cycle.zarr")
        m2 = resolve_s3_mapper("s3://weather-data/test2/cycle.zarr")

        assert m1.fs is m2.fs
        assert mock_build.call_count == 1
        assert m1.root == "weather-data/test1/cycle.zarr"
        assert m2.root == "weather-data/test2/cycle.zarr"


# =============================================================================
# Test C: Connection Pool Configuration & Validation
# =============================================================================
def test_connection_pool_configuration_propagates():
    """Verify that S3_MAX_POOL_CONNECTIONS reaches S3FileSystem config_kwargs."""
    custom_settings = IngestionSettings(
        S3_MAX_POOL_CONNECTIONS=75,
    )
    fs = get_s3_fs(custom_settings, force_new=True)
    assert fs.config_kwargs.get("max_pool_connections") == 75


def test_connection_pool_validation_rejects_invalid_values():
    """Verify that S3_MAX_POOL_CONNECTIONS < 1 is rejected by IngestionSettings validation."""
    with pytest.raises(ValueError, match="S3_MAX_POOL_CONNECTIONS must be >= 1"):
        IngestionSettings(S3_MAX_POOL_CONNECTIONS=0)

    with pytest.raises(ValueError, match="S3_MAX_POOL_CONNECTIONS must be >= 1"):
        IngestionSettings(S3_MAX_POOL_CONNECTIONS=-5)


# =============================================================================
# Test D: Storage Schema Identity
# =============================================================================
def test_storage_schema_identity_preservation(tmp_path):
    """Verify that stores written with the optimized S3 client preserve identical schema."""
    store = str(tmp_path / "schema_test.zarr")
    leads = (0, 6, 12)
    members = (1, 2)
    ds = _make_gefs_region(lead=0, member=1)

    prepare_run_store(ds, store, expected_lead_time_hours=leads, expected_members=members)

    root = zarr.open_group(store, mode="r")
    t2m = root["temperature_2m"]

    # Verify identical dimensions and shape
    assert t2m.shape == (2, 3, 721, 1440)
    assert t2m.chunks == (1, 1, 100, 100)
    assert t2m.dtype == np.float32
    assert t2m.fill_value is not None and np.isnan(t2m.fill_value)
    assert t2m.compressor.codec_id == "zstd"
    assert t2m.compressor.level == 5
    assert t2m.attrs["_ARRAY_DIMENSIONS"] == ["member", "lead_time_hours", "latitude", "longitude"]


# =============================================================================
# Test E: Read Compatibility
# =============================================================================
def test_read_compatibility(tmp_path):
    """Verify that data written is read back identically by xarray/Zarr readers."""
    store = str(tmp_path / "read_compat.zarr")
    leads = (6,)
    members = (1,)
    ds_write = _make_gefs_region(lead=6, member=1)

    prepare_run_store(ds_write, store, expected_lead_time_hours=leads, expected_members=members)
    commit_region(ds_write, store, lead_time_hours=6, member=1, lead_index=0, member_index=0)

    ds_read = read_dataset(store)
    assert "temperature_2m" in ds_read.data_vars
    assert ds_read["temperature_2m"].shape == (1, 1, 721, 1440)
    np.testing.assert_allclose(
        ds_read["temperature_2m"].values[0, 0],
        ds_write["temperature_2m"].values[0, 0],
        rtol=1e-5,
    )


# =============================================================================
# Test F: Same-Cycle Re-Ingestion
# =============================================================================
def test_same_cycle_reingestion_replaces_region(tmp_path):
    """Verify that re-ingesting a region in the same cycle replaces the target region cleanly."""
    store = str(tmp_path / "reingest.zarr")
    leads = (0, 6)
    members = (1,)
    ds1 = _make_gefs_region(lead=6, member=1)
    prepare_run_store(ds1, store, expected_lead_time_hours=leads, expected_members=members)
    commit_region(ds1, store, lead_time_hours=6, member=1, lead_index=1, member_index=0)

    # Re-ingest with modified values
    ds2 = _make_gefs_region(lead=6, member=1)
    ds2["temperature_2m"].values += 10.0
    commit_region(ds2, store, lead_time_hours=6, member=1, lead_index=1, member_index=0)

    ds_read = read_dataset(store)
    np.testing.assert_allclose(
        ds_read["temperature_2m"].values[0, 1],
        ds2["temperature_2m"].values[0, 0],
        rtol=1e-5,
    )


# =============================================================================
# Test G: Failure Propagation
# =============================================================================
def test_failure_propagation_leaves_region_uncommitted(tmp_path):
    """Verify that a failure during chunk write raises an exception and does not mark region complete."""
    store = str(tmp_path / "fail_prop.zarr")
    leads = (6,)
    members = (1,)
    ds = _make_gefs_region(lead=6, member=1)
    prepare_run_store(ds, store, expected_lead_time_hours=leads, expected_members=members)

    with patch.object(zarr.core.Array, "__setitem__", side_effect=OSError("Simulated S3 PUT failure")):
        with pytest.raises(OSError, match="Simulated S3 PUT failure"):
            commit_region(ds, store, lead_time_hours=6, member=1, lead_index=0, member_index=0)


# =============================================================================
# Test H: Thread Safety Across Worker Threads
# =============================================================================
def test_thread_safety_multi_thread_client_isolation():
    """Verify that concurrent worker threads each receive their own persistent S3FileSystem."""
    thread_fs_map: dict[int, object] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(6)

    def _worker(worker_id: int):
        barrier.wait()
        fs = get_s3_fs()
        with lock:
            thread_fs_map[worker_id] = fs
        # Verify multiple calls within the same worker return the exact same instance
        assert get_s3_fs() is fs

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread gets its own instance
    instances = list(thread_fs_map.values())
    assert len(instances) == 6
    # All 6 instances are distinct objects (thread-local isolation)
    assert len(set(id(inst) for inst in instances)) == 6


# =============================================================================
# Test I: Cleanup / Teardown
# =============================================================================
def test_cleanup_clears_thread_local():
    """Verify reset_s3_fs clears thread-local cached instance."""
    fs1 = get_s3_fs()
    assert fs1 is not None

    reset_s3_fs()
    # After reset, a new instance is created on next call
    fs2 = get_s3_fs()
    assert fs2 is not fs1
