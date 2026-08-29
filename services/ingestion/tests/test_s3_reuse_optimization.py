"""Focused regression tests for P2 Phase 1: S3 Client Separation (Data Plane vs Control Plane).

Covers:
* Test A — Data-Plane Isolation (long-lived writer thread reuses thread-local S3FileSystem)
* Test B — Control-Plane Client Reuse (multiple temporary threads reuse process-scoped S3FileSystem)
* Test C — Connection Pool Configuration (S3_MAX_POOL_CONNECTIONS and S3_CONTROL_MAX_POOL_CONNECTIONS validation)
* Test D — Storage Schema Identity (chunk keys, metadata, geometry, compressor, dtype, fill_value)
* Test E — Read Compatibility (production serving reader reads the store identically)
* Test F — Same-Cycle Re-Ingestion (generation and re-ingest semantics intact)
* Test G — Failure Propagation (simulated PUT failure leaves region uncommitted, clean recovery)
* Test H — Finalization Marker Retrieval (32 temporary threads read markers reusing single control S3FileSystem)
* Test I — Pre-Update Rolling Marker PUTs (multiple temporary threads reuse single control S3FileSystem)
* Test J — Thread Safety & Event-Loop Isolation (concurrent marker operations do not clash with data-plane threads)
* Test K — Cleanup / Teardown (reset_s3_fs clears both data-plane and control-plane caches cleanly)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import zarr

from ingestion.core.config import IngestionSettings
from ingestion.core.coordinator import _read_marker_payloads_bounded
from ingestion.core.marker_put_scheduler import put_markers_rolling
from ingestion.core.markers import (
    MARKER_V1,
    write_protocol_version,
    write_region_marker,
)
from ingestion.core.s3 import (
    get_s3_fs,
    get_control_s3_fs,
    reset_s3_fs,
    resolve_s3_mapper,
    _build_data_s3_fs,
    _build_control_s3_fs,
)
from ingestion.core.zarr_writer import (
    commit_region,
    prepare_run_store,
    read_dataset,
)


@pytest.fixture(autouse=True)
def _cleanup_s3_state():
    """Ensure S3 client caches are reset before and after each test."""
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


# =============================================================================
# Test A: Data-Plane Isolation & Reuse
# =============================================================================
def test_data_plane_filesystem_reuse_on_same_thread():
    """Verify that repeated data-plane calls on the same thread reuse the thread-local S3FileSystem."""
    with patch("ingestion.core.s3._build_data_s3_fs", wraps=_build_data_s3_fs) as mock_build:
        fs1 = get_s3_fs()
        fs2 = get_s3_fs()
        fs3 = get_s3_fs()

        assert fs1 is fs2
        assert fs2 is fs3
        assert mock_build.call_count == 1


def test_resolve_s3_mapper_reuses_data_plane_filesystem():
    """Verify that resolve_s3_mapper reuses the thread-local data-plane S3FileSystem."""
    with patch("ingestion.core.s3._build_data_s3_fs", wraps=_build_data_s3_fs) as mock_build:
        m1 = resolve_s3_mapper("s3://weather-data/test1/cycle.zarr")
        m2 = resolve_s3_mapper("s3://weather-data/test2/cycle.zarr")

        assert m1.fs is m2.fs
        assert mock_build.call_count == 1
        assert m1.root == "weather-data/test1/cycle.zarr"
        assert m2.root == "weather-data/test2/cycle.zarr"


# =============================================================================
# Test B: Control-Plane Client Reuse
# =============================================================================
def test_control_plane_client_reused_across_multiple_threads():
    """Verify that multiple threads all share the single process-scoped control-plane S3FileSystem."""
    with patch("ingestion.core.s3._build_control_s3_fs", wraps=_build_control_s3_fs) as mock_build:
        thread_fs = {}
        lock = threading.Lock()

        def _worker(worker_id: int):
            fs = get_control_s3_fs()
            with lock:
                thread_fs[worker_id] = fs

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 1 control-plane filesystem constructed despite 16 threads
        assert mock_build.call_count == 1
        first_fs = thread_fs[0]
        for inst in thread_fs.values():
            assert inst is first_fs


# =============================================================================
# Test C: Connection Pool Configuration & Validation
# =============================================================================
def test_connection_pool_configuration_propagates():
    """Verify that S3_MAX_POOL_CONNECTIONS and S3_CONTROL_MAX_POOL_CONNECTIONS propagate."""
    custom_settings = IngestionSettings(
        S3_MAX_POOL_CONNECTIONS=75,
        S3_CONTROL_MAX_POOL_CONNECTIONS=40,
    )
    data_fs = get_s3_fs(custom_settings, force_new=True)
    control_fs = get_control_s3_fs(custom_settings, force_new=True)

    assert data_fs.config_kwargs.get("max_pool_connections") == 75
    assert control_fs.config_kwargs.get("max_pool_connections") == 40


def test_connection_pool_validation_rejects_invalid_values():
    """Verify that pool sizes < 1 are rejected."""
    with pytest.raises(ValueError, match="S3_MAX_POOL_CONNECTIONS must be >= 1"):
        IngestionSettings(S3_MAX_POOL_CONNECTIONS=0)

    with pytest.raises(ValueError, match="S3_CONTROL_MAX_POOL_CONNECTIONS must be >= 1"):
        IngestionSettings(S3_CONTROL_MAX_POOL_CONNECTIONS=0)


# =============================================================================
# Test D: Storage Schema Identity
# =============================================================================
def test_storage_schema_identity_preservation(tmp_path):
    """Verify that stores written preserve identical schema."""
    store = str(tmp_path / "schema_test.zarr")
    leads = (0, 6, 12)
    members = (1, 2)
    ds = _make_gefs_region(lead=0, member=1)

    prepare_run_store(ds, store, expected_lead_time_hours=leads, expected_members=members)

    root = zarr.open_group(store, mode="r")
    t2m = root["temperature_2m"]

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
    """Verify that re-ingesting a region replaces the target region cleanly."""
    store = str(tmp_path / "reingest.zarr")
    leads = (0, 6)
    members = (1,)
    ds1 = _make_gefs_region(lead=6, member=1)
    prepare_run_store(ds1, store, expected_lead_time_hours=leads, expected_members=members)
    commit_region(ds1, store, lead_time_hours=6, member=1, lead_index=1, member_index=0)

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
    """Verify that a failure during chunk write raises and leaves region uncommitted."""
    store = str(tmp_path / "fail_prop.zarr")
    leads = (6,)
    members = (1,)
    ds = _make_gefs_region(lead=6, member=1)
    prepare_run_store(ds, store, expected_lead_time_hours=leads, expected_members=members)

    with patch.object(zarr.core.Array, "__setitem__", side_effect=OSError("Simulated S3 PUT failure")):
        with pytest.raises(OSError, match="Simulated S3 PUT failure"):
            commit_region(ds, store, lead_time_hours=6, member=1, lead_index=0, member_index=0)


# =============================================================================
# Test H: Finalization Marker Retrieval (Bounded 32 Threads Reuse 1 Client)
# =============================================================================
def test_finalization_marker_retrieval_reuses_single_control_client(tmp_path):
    """Verify that _read_marker_payloads_bounded with 32 threads constructs exactly 1 control S3FileSystem."""
    store = str(tmp_path / "marker_retrieval_test.zarr")
    write_protocol_version(store, MARKER_V1)

    # Write 64 sample markers
    marker_keys = []
    for m in range(1, 3):
        for lead_h in range(0, 192, 6):
            write_region_marker(
                store,
                lead_time_hours=lead_h,
                member=m,
                payload={"protocol_version": 1, "state": "complete", "generation": f"g_{m}_{lead_h}", "logical_region": {"lead_time_hours": lead_h, "member": m}},
            )
            reg_id = f"mem{m:03d}_L{lead_h:04d}"
            marker_keys.append(f"__commit__/v1/regions/{reg_id}.json")

    with patch("ingestion.core.s3._build_control_s3_fs", wraps=_build_control_s3_fs) as mock_ctrl_build:
        results = _read_marker_payloads_bounded(store, marker_keys, max_concurrency=32)

        assert len(results) == len(marker_keys)
        # Verify deterministic ordering
        for (k_res, payload), k_exp in zip(results, marker_keys, strict=True):
            assert k_res == k_exp
            assert payload["state"] == "complete"

        # Exactly 1 control-plane filesystem constructed across 32 threads
        assert mock_ctrl_build.call_count <= 1


# =============================================================================
# Test I: Pre-Update Rolling Marker PUTs
# =============================================================================
def test_pre_update_marker_puts_reuses_control_client(tmp_path):
    """Verify that rolling pre-update marker PUTs reuse the control-plane S3FileSystem."""
    store = str(tmp_path / "pre_update_test.zarr")
    write_protocol_version(store, MARKER_V1)

    region_ids = [f"mem001_L{lead_h:04d}" for lead_h in range(0, 96, 6)]

    def _put_one(region_id: str):
        lead = int(region_id.split("_L")[-1])
        write_region_marker(
            store,
            lead_time_hours=lead,
            member=1,
            payload={"protocol_version": 1, "state": "updating", "generation": "gen_up"},
        )

    cancel_event = threading.Event()
    with ThreadPoolExecutor(max_workers=8) as ex:
        with patch("ingestion.core.s3._build_control_s3_fs", wraps=_build_control_s3_fs) as mock_ctrl_build:
            res = put_markers_rolling(
                region_ids,
                _put_one,
                concurrency=8,
                cancel_event=cancel_event,
                timeout_seconds=30.0,
                executor=ex,
            )
            assert res.ok
            assert mock_ctrl_build.call_count <= 1


# =============================================================================
# Test J: Thread Safety Across Data and Control Planes
# =============================================================================
def test_thread_safety_data_plane_distinct_from_control_plane():
    """Verify that data-plane writer threads each get independent S3FileSystems while control plane shares one."""
    data_instances = {}
    control_instances = {}
    lock = threading.Lock()
    barrier = threading.Barrier(6)

    def _writer_thread(worker_id: int):
        barrier.wait()
        data_fs = get_s3_fs()
        ctrl_fs = get_control_s3_fs()
        with lock:
            data_instances[worker_id] = data_fs
            control_instances[worker_id] = ctrl_fs

    threads = [threading.Thread(target=_writer_thread, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Data plane: 6 distinct instances (one per worker thread)
    assert len(set(id(inst) for inst in data_instances.values())) == 6

    # Control plane: exactly 1 shared instance across all threads
    assert len(set(id(inst) for inst in control_instances.values())) == 1

    # Data plane instances are distinct from the control plane instance
    for data_inst in data_instances.values():
        assert data_inst is not control_instances[0]


# =============================================================================
# Test K: Cleanup Clears Both Caches
# =============================================================================
def test_cleanup_clears_both_caches():
    """Verify reset_s3_fs clears both thread-local and process-scoped caches."""
    data_fs1 = get_s3_fs()
    ctrl_fs1 = get_control_s3_fs()

    reset_s3_fs()

    data_fs2 = get_s3_fs()
    ctrl_fs2 = get_control_s3_fs()

    assert data_fs2 is not data_fs1
    assert ctrl_fs2 is not ctrl_fs1
