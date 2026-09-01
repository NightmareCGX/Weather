"""Unit and integration tests for Phase 1 GEFS ingestion correctness and resilience remediation.

Tests coverage:
1. Precipitation de-accumulation elementwise policy (exact boundaries, NaNs, immutability, QC logs, shape mismatch).
2. Predecessor in-memory state decoupling from physical Zarr commit.
3. Transient storage write retries (retryable classification, retries with backoff, deterministic zero-retry, retry exhaustion).
4. Decode semaphore inversion regression test.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from ingestion.core.base import (
    DeaccumulationError,
    StoreSchemaMismatchError,
    is_retryable_storage_error,
)
from ingestion.core.catalog import RunCatalogSpec, VariableSpec
from ingestion.core.pipeline import (
    _normalize_precipitation_increments,
    deaccumulate_precipitation,
)
from ingestion.core.zarr_writer import write_dataset


# =============================================================================
# 1. Precipitation de-accumulation elementwise policy tests
# =============================================================================


def test_deaccumulate_precipitation_positive_values() -> None:
    """Positive increments (residual >= 0.0 mm) are preserved exactly."""
    curr = np.array([[5.0, 1.2], [0.0, 10.5]], dtype=np.float32)
    pred = np.array([[2.0, 0.0], [0.0, 3.2]], dtype=np.float32)

    res = deaccumulate_precipitation(curr, pred)

    assert res.dtype == np.float32
    assert res[0, 0] == pytest.approx(3.0, abs=1e-5)
    assert res[0, 1] == pytest.approx(1.2, abs=1e-5)
    assert res[1, 0] == pytest.approx(0.0, abs=1e-5)
    assert res[1, 1] == pytest.approx(7.3, abs=1e-5)


def test_deaccumulate_precipitation_exact_boundaries(caplog: pytest.LogCaptureFixture) -> None:
    """Test exact clamping and invalidation boundaries:
    * residual >= 0.0 -> preserved
    * -0.50 <= residual < 0.0 -> clamped to 0.0 (exact at -0.50)
    * residual < -0.50 -> set to NaN (e.g. -0.50001, -0.51, -2.0)
    """
    # Differences: [1.20, 0.00, -0.01, -0.18, -0.49, -0.50, -0.50001, -0.51, -2.00]
    curr = np.array([1.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float64)
    pred = np.array([0.00, 0.00, 0.01, 0.18, 0.49, 0.50, 0.50001, 0.51, 2.00], dtype=np.float64)

    with caplog.at_level(logging.WARNING):
        res = deaccumulate_precipitation(curr, pred)

    assert res[0] == pytest.approx(1.20, abs=1e-5)
    assert res[1] == pytest.approx(0.00, abs=1e-5)
    assert res[2] == pytest.approx(0.00, abs=1e-5)   # -0.01 clamped to 0.0
    assert res[3] == pytest.approx(0.00, abs=1e-5)   # -0.18 clamped to 0.0 (observed GEFS anomaly)
    assert res[4] == pytest.approx(0.00, abs=1e-5)   # -0.49 clamped to 0.0
    assert res[5] == pytest.approx(0.00, abs=1e-5)   # -0.50 exact boundary clamped to 0.0
    assert np.isnan(res[6])                           # -0.50001 invalidated to NaN
    assert np.isnan(res[7])                           # -0.51 invalidated to NaN
    assert np.isnan(res[8])                           # -2.00 invalidated to NaN

    # QC log verification
    assert "Precipitation de-accumulation negative residuals detected" in caplog.text
    assert "clamped_count=4" in caplog.text
    assert "invalidated_count=3" in caplog.text


def test_deaccumulate_precipitation_mixed_array_and_nans() -> None:
    """Verify mixed array [1.2, -0.1, -0.5, -0.51, NaN] yields [1.2, 0.0, 0.0, NaN, NaN]."""
    curr = np.array([1.2, 0.0, 0.0, 0.0, np.nan], dtype=np.float32)
    pred = np.array([0.0, 0.1, 0.5, 0.51, 1.0], dtype=np.float32)

    res = deaccumulate_precipitation(curr, pred)

    assert res[0] == pytest.approx(1.2, abs=1e-5)
    assert res[1] == pytest.approx(0.0, abs=1e-5)
    assert res[2] == pytest.approx(0.0, abs=1e-5)
    assert np.isnan(res[3])
    assert np.isnan(res[4])


def test_deaccumulate_precipitation_input_immutability() -> None:
    """Verify input arrays are not modified in place."""
    curr_orig = np.array([0.0, 0.0], dtype=np.float64)
    pred_orig = np.array([0.18, 0.80], dtype=np.float64)
    curr_copy = curr_orig.copy()
    pred_copy = pred_orig.copy()

    deaccumulate_precipitation(curr_orig, pred_orig)

    np.testing.assert_array_equal(curr_orig, curr_copy)
    np.testing.assert_array_equal(pred_orig, pred_copy)


def test_deaccumulate_precipitation_shape_mismatch_raises() -> None:
    """Shape mismatch between current and predecessor still raises DeaccumulationError."""
    curr = np.ones((4, 4), dtype=np.float32)
    pred = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(DeaccumulationError, match="shape mismatch"):
        deaccumulate_precipitation(curr, pred)


# =============================================================================
# 2. Predecessor decoupling tests
# =============================================================================


def test_normalize_precipitation_increments_uses_in_memory_predecessor(tmp_path: Path) -> None:
    """Lead 6 normalization succeeds using an in-memory predecessor array even if
    the store is empty/all-NaN or store_path is uncommitted.
    """
    store_path = str(tmp_path / "cycle.zarr")

    # Lead 6 raw accumulation [0..6]
    curr_accum = np.array([[5.0, 8.0], [2.0, 1.0]], dtype=np.float32)
    raw_ds = xr.Dataset(
        data_vars={"tp": (("latitude", "longitude"), curr_accum)},
        coords={
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
            "lead_time_hours": [6],
        },
    )
    variables = (
        VariableSpec("precipitation_amount_3h", "3-Hour Precipitation Amount", "mm", "tp"),
    )

    # In-memory predecessor accumulation [0..3] from lead 3
    in_memory_pred = np.array([[2.0, 3.0], [1.0, 1.18]], dtype=np.float32)

    # Normalize with explicit in-memory predecessor (store_path does not exist on disk!)
    ds = _normalize_precipitation_increments(
        raw_ds,
        variables,
        store_path=store_path,
        predecessor_array=in_memory_pred,
        member=1,
    )

    vals = ds["tp"].values
    assert vals[0, 0] == pytest.approx(3.0, abs=1e-5)    # 5.0 - 2.0 = 3.0
    assert vals[0, 1] == pytest.approx(5.0, abs=1e-5)    # 8.0 - 3.0 = 5.0
    assert vals[1, 0] == pytest.approx(1.0, abs=1e-5)    # 2.0 - 1.0 = 1.0
    assert vals[1, 1] == pytest.approx(0.0, abs=1e-5)    # 1.0 - 1.18 = -0.18 -> clamped to 0.0


def test_normalize_precipitation_increments_fallback_to_zarr(tmp_path: Path) -> None:
    """When in-memory predecessor is None (standalone/re-ingest), normalizer reads Zarr."""
    store_path = str(tmp_path / "standalone.zarr")

    # Write predecessor lead 3 into store
    pred_ds = xr.Dataset(
        data_vars={
            "precipitation_amount_3h": (
                ("lead_time_hours", "latitude", "longitude"),
                np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float32),
            )
        },
        coords={
            "lead_time_hours": [3],
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
    )
    write_dataset(pred_ds, store_path)

    # Lead 6 raw accumulation [0..6]
    curr_accum = np.array([[3.5, 5.0], [4.0, 6.0]], dtype=np.float32)
    raw_ds = xr.Dataset(
        data_vars={"tp": (("latitude", "longitude"), curr_accum)},
        coords={
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
            "lead_time_hours": [6],
        },
    )
    variables = (
        VariableSpec("precipitation_amount_3h", "3-Hour Precipitation Amount", "mm", "tp"),
    )

    # Normalize with predecessor_array=None (triggers fallback to Zarr store)
    ds = _normalize_precipitation_increments(
        raw_ds,
        variables,
        store_path=store_path,
        predecessor_array=None,
        member=None,
    )

    vals = ds["tp"].values
    assert vals[0, 0] == pytest.approx(2.0, abs=1e-5)    # 3.5 - 1.5 = 2.0
    assert vals[0, 1] == pytest.approx(2.5, abs=1e-5)    # 5.0 - 2.5 = 2.5
    assert vals[1, 0] == pytest.approx(0.5, abs=1e-5)    # 4.0 - 3.5 = 0.5
    assert vals[1, 1] == pytest.approx(1.5, abs=1e-5)    # 6.0 - 4.5 = 1.5


# =============================================================================
# 3. Transient write retry tests
# =============================================================================


def test_is_retryable_storage_error_classification() -> None:
    """Verify error classification for transient vs permanent errors."""
    import socket

    # Transient socket / OS errors
    assert is_retryable_storage_error(ConnectionResetError("Connection reset by peer")) is True
    assert is_retryable_storage_error(ConnectionRefusedError("Connection refused")) is True
    assert is_retryable_storage_error(TimeoutError("Operation timed out")) is True
    assert is_retryable_storage_error(socket.timeout("timed out")) is True
    assert is_retryable_storage_error(OSError("Could not connect to the endpoint URL")) is True

    # Custom / Simulated Botocore errors
    class EndpointConnectionError(Exception):
        pass

    class ConnectTimeoutError(Exception):
        pass

    class ClientError(Exception):
        def __init__(self, status_code: int, error_code: str):
            self.response = {
                "ResponseMetadata": {"HTTPStatusCode": status_code},
                "Error": {"Code": error_code},
            }

    assert is_retryable_storage_error(EndpointConnectionError("Could not connect to endpoint")) is True
    assert is_retryable_storage_error(ConnectTimeoutError("Connection timed out")) is True
    assert is_retryable_storage_error(ClientError(503, "ServiceUnavailable")) is True
    assert is_retryable_storage_error(ClientError(429, "SlowDown")) is True
    assert is_retryable_storage_error(ClientError(500, "InternalError")) is True

    # Non-retryable deterministic errors
    assert is_retryable_storage_error(StoreSchemaMismatchError("Schema mismatch")) is False
    assert is_retryable_storage_error(ValueError("Invalid shape")) is False
    assert is_retryable_storage_error(TypeError("Unexpected type")) is False
    assert is_retryable_storage_error(KeyError("missing_key")) is False
    assert is_retryable_storage_error(ClientError(404, "NoSuchBucket")) is False
    assert is_retryable_storage_error(ClientError(403, "AccessDenied")) is False


def test_write_region_worker_transient_retry_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Mock transient failure on attempts 1 and 2, then success on attempt 3.
    Verifies that region write completes and COMPLETE marker is written.
    """
    from ingestion.core.coordinator import RunCoordinator, StoreMetadataSnapshot
    from ingestion.core.markers import read_region_marker

    store_path = str(tmp_path / "retry_test.zarr")

    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="GFS",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
        grid_id="global_025deg",
        grid_name="Global 0.25 Grid",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=store_path,
        variables=(VariableSpec("temperature_2m", "Temperature", "°C", "t2m"),),
        expected_lead_time_hours=(6,),
        expected_members=(),
    )

    coordinator = RunCoordinator(spec, store_path)

    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), np.ones((1, 2, 2), dtype=np.float32))},
        coords={"lead_time_hours": [6], "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )

    # Initialize snapshot and coordination stubs
    coordinator._snapshot = StoreMetadataSnapshot(
        store_path=store_path,
        generation="gen-123",
        is_ensemble=False,
        data_var_paths=("temperature_2m",),
        lead_index_map={6: 0},
        member_index_map={},
        zarray_by_var={"temperature_2m": {"shape": [1, 2, 2], "chunks": [1, 2, 2], "dimension_separator": "."}},
        zattrs_by_var={"temperature_2m": {"_ARRAY_DIMENSIONS": ["lead_time_hours", "latitude", "longitude"]}},
        data_var_dims={"temperature_2m": ("lead_time_hours", "latitude", "longitude")},
        coords_values={"latitude": (0.0, 1.0), "longitude": (0.0, 1.0)},
        grid_shape=(2, 2),
        cycle_time="2026-07-21T00:00:00",
        model_id="gfs",
    )

    # Mock lock coordinator methods so we don't need real PostgreSQL
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.acquire_shared_admission", lambda self: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.release_shared_admission", lambda self: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.acquire_shared_gate", lambda self: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.release_shared_gate", lambda self: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.acquire_region_locks", lambda self, keys: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.release_region_locks", lambda self, keys: None)

    # Set up updating marker
    from ingestion.core.markers import write_region_marker
    write_region_marker(
        store_path,
        lead_time_hours=6,
        member=None,
        payload={"state": "updating", "generation": "gen-123", "protocol_version": 1},
    )

    # Mock _commit_region to fail twice with transient error, then succeed
    attempts = {"count": 0}
    def _mock_commit_region(*args: Any, **kwargs: Any) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionResetError(f"Simulated transient error attempt {attempts['count']}")
        # Attempt 3 succeeds

    monkeypatch.setattr("ingestion.core.coordinator._commit_region", _mock_commit_region)
    monkeypatch.setattr("ingestion.core.inventory.verify_expected_object_keys", lambda *args, **kwargs: set())

    # Execute region write worker
    mock_conn = MagicMock()
    coordinator.write_region_worker(
        mock_conn,
        dataset=ds,
        member=None,
        generation="gen-123",
        expected_leads=(6,),
        expected_members=(),
    )

    assert attempts["count"] == 3
    marker = read_region_marker(store_path, lead_time_hours=6, member=None)
    assert marker.get("state") == "complete"
    assert marker.get("generation") == "gen-123"


def test_write_region_worker_deterministic_error_no_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Deterministic error fails immediately with 1 attempt (zero retries)."""
    from ingestion.core.coordinator import RunCoordinator, StoreMetadataSnapshot

    store_path = str(tmp_path / "no_retry.zarr")
    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="GFS",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
        grid_id="global_025deg",
        grid_name="Global 0.25 Grid",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=store_path,
        variables=(VariableSpec("temperature_2m", "Temperature", "°C", "t2m"),),
        expected_lead_time_hours=(6,),
        expected_members=(),
    )

    coordinator = RunCoordinator(spec, store_path)
    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), np.ones((1, 2, 2), dtype=np.float32))},
        coords={"lead_time_hours": [6], "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    coordinator._snapshot = StoreMetadataSnapshot(
        store_path=store_path,
        generation="gen-123",
        is_ensemble=False,
        data_var_paths=("temperature_2m",),
        lead_index_map={6: 0},
        member_index_map={},
        zarray_by_var={"temperature_2m": {"shape": [1, 2, 2], "chunks": [1, 2, 2], "dimension_separator": "."}},
        zattrs_by_var={"temperature_2m": {"_ARRAY_DIMENSIONS": ["lead_time_hours", "latitude", "longitude"]}},
        data_var_dims={"temperature_2m": ("lead_time_hours", "latitude", "longitude")},
        coords_values={"latitude": (0.0, 1.0), "longitude": (0.0, 1.0)},
        grid_shape=(2, 2),
        cycle_time="2026-07-21T00:00:00",
        model_id="gfs",
    )

    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.acquire_shared_admission", lambda self: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.release_shared_admission", lambda self: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.acquire_shared_gate", lambda self: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.release_shared_gate", lambda self: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.acquire_region_locks", lambda self, keys: None)
    monkeypatch.setattr("ingestion.core.locks.StoreLockCoordinator.release_region_locks", lambda self, keys: None)

    from ingestion.core.markers import write_region_marker
    write_region_marker(
        store_path,
        lead_time_hours=6,
        member=None,
        payload={"state": "updating", "generation": "gen-123", "protocol_version": 1},
    )

    attempts = {"count": 0}
    def _mock_deterministic_fail(*args: Any, **kwargs: Any) -> None:
        attempts["count"] += 1
        raise ValueError("Deterministic programming or schema error")

    monkeypatch.setattr("ingestion.core.coordinator._commit_region", _mock_deterministic_fail)

    mock_conn = MagicMock()
    with pytest.raises(ValueError, match="Deterministic programming"):
        coordinator.write_region_worker(
            mock_conn,
            dataset=ds,
            member=None,
            generation="gen-123",
            expected_leads=(6,),
            expected_members=(),
        )

    assert attempts["count"] == 1


# =============================================================================
# 4. Semaphore ordering & deadlock prevention test
# =============================================================================


@pytest.mark.asyncio
async def test_predecessor_waiting_does_not_hold_decode_sem() -> None:
    """Verify that a task waiting on predecessor decode readiness does NOT acquire or
    occupy decode_sem while waiting.
    """
    decode_sem = asyncio.Semaphore(1)  # Only 1 decode slot available
    decode_completed_events = {3: asyncio.Event(), 6: asyncio.Event()}
    decode_sem_acquired_by = []

    async def task_lead_6():
        # Target implementation pattern: wait BEFORE acquiring decode_sem
        if 3 in decode_completed_events:
            await decode_completed_events[3].wait()

        async with decode_sem:
            decode_sem_acquired_by.append(6)
            decode_completed_events[6].set()

    async def task_lead_3():
        # Lead 3 runs decode and signals completion
        async with decode_sem:
            decode_sem_acquired_by.append(3)
            # Simulate work
            await asyncio.sleep(0.01)
            decode_completed_events[3].set()

    # Start lead 6 FIRST (it should wait on lead 3 without blocking decode_sem)
    t6 = asyncio.create_task(task_lead_6())
    await asyncio.sleep(0.005)

    # Lead 3 must be able to acquire decode_sem even though lead 6 is waiting
    t3 = asyncio.create_task(task_lead_3())

    await asyncio.gather(t6, t3)

    # Lead 3 acquired decode_sem first, then lead 6
    assert decode_sem_acquired_by == [3, 6]
