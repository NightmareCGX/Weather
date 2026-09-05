"""Deterministic tests for Object-Store Bootstrap & Early-Failure S3 Cleanup.

Validates:
1. Preflight bucket validation fails fast with MissingBucketError when configured bucket is absent.
2. RealtimeScheduler and CLI refuse to dispatch waves without configured bucket.
3. Early failures in _run_wave (initialize_run_store, download, decode, write, finalize)
   guarantee that close_wave_data_s3_fs() executes and leaves _active_data_filesystems empty.
4. No unclosed aiohttp/aiobotocore client sessions on early failures.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import xarray as xr
import numpy as np
from sqlalchemy import create_engine

from ingestion.core.config import IngestionSettings
from ingestion.core.s3 import (
    MissingBucketError,
    _active_data_filesystems,
    get_s3_fs,
    verify_object_store_preflight,
)
from ingestion.core.wave_runner import RunCatalogSpec, RunSpec, VariableSpec, _run_wave
from ingestion.realtime.scheduler import RealtimeScheduler


def _make_dummy_dataset() -> xr.Dataset:
    lat = np.array([38.0, 38.25])
    lon = np.array([-107.0, -106.75])
    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), np.full((1, 2, 2), 20.0, dtype=np.float32)),
        },
        coords={
            "lead_time_hours": [0],
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture(autouse=True)
def cleanup_s3_state(monkeypatch):
    """Ensure S3 filesystem state is clean and database is mocked."""
    from ingestion.core.s3 import shutdown_s3_fs
    from ingestion.core.db import CatalogBase
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(test_engine)
    monkeypatch.setattr("ingestion.core.wave_runner._catalog_session_factory", lambda: test_engine)
    sm = sessionmaker(bind=test_engine)
    monkeypatch.setattr("ingestion.core.wave_runner._catalog_session", lambda: sm())

    shutdown_s3_fs()
    yield
    shutdown_s3_fs()


def test_missing_bucket_preflight_fails_cleanly(monkeypatch):
    """Preflight check raises MissingBucketError with actionable diagnostic when bucket is absent."""
    custom_cfg = IngestionSettings(
        MINIO_BUCKET_NAME="nonexistent-test-bucket-" + uuid.uuid4().hex[:8],
    )

    with pytest.raises(MissingBucketError, match="does not exist"):
        verify_object_store_preflight(custom_cfg)


def test_realtime_scheduler_refuses_startup_on_missing_bucket(monkeypatch):
    """RealtimeScheduler.run() aborts with MissingBucketError before polling when bucket is absent."""
    custom_cfg = IngestionSettings(
        MINIO_BUCKET_NAME="missing-realtime-bucket-" + uuid.uuid4().hex[:8],
        REALTIME_ENABLED=True,
    )
    scheduler = RealtimeScheduler(
        conn_settings=custom_cfg,
        leadership=None,
    )

    with pytest.raises(MissingBucketError, match="does not exist"):
        scheduler.run(once=True, dry_run=False)


@pytest.mark.asyncio
async def test_run_wave_early_initialize_failure_cleans_up_s3_filesystems(tmp_path, monkeypatch):
    """When initialize_run_store fails (e.g. S3 error), _active_data_filesystems must be cleanly drained."""
    spec = RunSpec(
        model="gfs",
        cycle_date=date(2026, 9, 4),
        cycle_hour=0,
        target_lead_time_hours=(0,),
        members=(),
    )
    catalog_spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="Global Forecast System",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        grid_id="global_025deg",
        grid_name="Global 0.25 Grid",
        grid_resolution_km=25.0,
        cycle_time=datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc),
        variables=[VariableSpec(code="temperature_2m", name="2m Temp", unit="C")],
        expected_lead_time_hours=[0],
        expected_members=[],
    )
    args = SimpleNamespace(
        download_dir=str(tmp_path),
        keep_downloads=False,
        no_progress=True,
        lock_timeout=1.0,
        concurrency=1,
    )

    # Mock NOAAConnector download to succeed with dummy dataset
    async def fake_download(self, *a, **kw):
        dest = a[4]
        Path(dest).touch()

    monkeypatch.setattr("ingestion.providers.noaa.connector.NOAAConnector.download", fake_download)
    monkeypatch.setattr("ingestion.core.wave_runner._decode_and_normalize", lambda *a, **kw: _make_dummy_dataset())

    # Simulate S3 initialization failure in initialize_run_store
    def failing_initialize(*a, **kw):
        # Register a data-plane filesystem before raising
        get_s3_fs()
        assert len(_active_data_filesystems) > 0
        raise RuntimeError("Simulated S3 PutObject failure in initialize_run_store")

    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.initialize_run_store", failing_initialize)

    failures: list[str] = []
    with pytest.raises(RuntimeError, match="Simulated S3 PutObject failure"):
        await _run_wave(
            spec=spec,
            args=args,
            catalog_spec=catalog_spec,
            store_path="s3://weather-data/test/cycle.zarr",
            concurrency=1,
            failures=failures,
        )

    # Assert that despite early failure in step 3, _active_data_filesystems was completely closed and drained
    assert len(_active_data_filesystems) == 0, f"Leaked S3 filesystems after early wave failure: {_active_data_filesystems}"


@pytest.mark.asyncio
async def test_run_wave_download_decode_failure_cleans_up_s3_filesystems(tmp_path, monkeypatch):
    """When download/decode fails, _active_data_filesystems must be cleanly drained."""
    spec = RunSpec(
        model="gfs",
        cycle_date=date(2026, 9, 4),
        cycle_hour=0,
        target_lead_time_hours=(0,),
        members=(),
    )
    catalog_spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="Global Forecast System",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        grid_id="global_025deg",
        grid_name="Global 0.25 Grid",
        grid_resolution_km=25.0,
        cycle_time=datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc),
        variables=[VariableSpec(code="temperature_2m", name="2m Temp", unit="C")],
        expected_lead_time_hours=[0],
        expected_members=[],
    )
    args = SimpleNamespace(
        download_dir=str(tmp_path),
        keep_downloads=False,
        no_progress=True,
        lock_timeout=1.0,
        concurrency=1,
    )

    async def failing_download(self, *a, **kw):
        raise ConnectionResetError("Simulated upstream download abort")

    monkeypatch.setattr("ingestion.providers.noaa.connector.NOAAConnector.download", failing_download)

    failures: list[str] = []
    with pytest.raises(ConnectionResetError, match="Simulated upstream download abort"):
        await _run_wave(
            spec=spec,
            args=args,
            catalog_spec=catalog_spec,
            store_path="s3://weather-data/test/cycle.zarr",
            concurrency=1,
            failures=failures,
        )

    assert len(_active_data_filesystems) == 0, "Leaked S3 filesystems after download failure!"


@pytest.mark.asyncio
async def test_run_wave_write_stage_failure_cleans_up_s3_filesystems(tmp_path, monkeypatch):
    """When a region write fails, _active_data_filesystems must be cleanly drained."""
    spec = RunSpec(
        model="gfs",
        cycle_date=date(2026, 9, 4),
        cycle_hour=0,
        target_lead_time_hours=(0,),
        members=(),
    )
    catalog_spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="Global Forecast System",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        grid_id="global_025deg",
        grid_name="Global 0.25 Grid",
        grid_resolution_km=25.0,
        cycle_time=datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc),
        variables=[VariableSpec(code="temperature_2m", name="2m Temp", unit="C")],
        expected_lead_time_hours=[0],
        expected_members=[],
    )
    args = SimpleNamespace(
        download_dir=str(tmp_path),
        keep_downloads=False,
        no_progress=True,
        lock_timeout=1.0,
        concurrency=1,
    )

    async def fake_download(self, *a, **kw):
        dest = a[4]
        Path(dest).touch()

    import concurrent.futures
    from ingestion.core.decode_worker import DecodePool
    from ingestion.core.db import CatalogBase

    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(test_engine)
    monkeypatch.setattr("ingestion.core.wave_runner._catalog_session_factory", lambda: test_engine)

    def mock_submit(self, path):
        fut = concurrent.futures.Future()
        fut.set_result(_make_dummy_dataset())
        return fut

    monkeypatch.setattr(DecodePool, "submit", mock_submit)
    monkeypatch.setattr("ingestion.providers.noaa.connector.NOAAConnector.download", fake_download)
    monkeypatch.setattr("ingestion.core.wave_runner._decode_and_normalize", lambda *a, **kw: _make_dummy_dataset())
    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.initialize_run_store", lambda *a, **kw: None)
    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.pre_update_wave", lambda *a, **kw: None)
    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.publish_settled_lead", lambda *a, **kw: None)

    def failing_write(*a, **kw):
        get_s3_fs()
        raise IOError("Simulated S3 chunk PUT network partition")

    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.write_region_worker", failing_write)

    # Mock finalization
    from ingestion.core.coordinator import FinalizeResult
    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.finalize_run", lambda *a, **kw: FinalizeResult(status="partial", committed_regions={}))

    failures: list[str] = []
    status = await _run_wave(
        spec=spec,
        args=args,
        catalog_spec=catalog_spec,
        store_path="s3://weather-data/test/cycle.zarr",
        concurrency=1,
        failures=failures,
    )

    assert len(_active_data_filesystems) == 0, f"Leaked S3 filesystems after write failure: {_active_data_filesystems}"
    assert status == "partial"


@pytest.mark.asyncio
async def test_run_wave_finalize_failure_cleans_up_s3_filesystems(tmp_path, monkeypatch):
    """When finalization raises an unexpected error, _active_data_filesystems must be cleanly drained."""
    spec = RunSpec(
        model="gfs",
        cycle_date=date(2026, 9, 4),
        cycle_hour=0,
        target_lead_time_hours=(0,),
        members=(),
    )
    catalog_spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="Global Forecast System",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        grid_id="global_025deg",
        grid_name="Global 0.25 Grid",
        grid_resolution_km=25.0,
        cycle_time=datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc),
        variables=[VariableSpec(code="temperature_2m", name="2m Temp", unit="C")],
        expected_lead_time_hours=[0],
        expected_members=[],
    )
    args = SimpleNamespace(
        download_dir=str(tmp_path),
        keep_downloads=False,
        no_progress=True,
        lock_timeout=1.0,
        concurrency=1,
    )

    async def fake_download(self, *a, **kw):
        dest = a[4]
        Path(dest).touch()

    import concurrent.futures
    from ingestion.core.decode_worker import DecodePool
    from ingestion.core.db import CatalogBase

    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(test_engine)
    monkeypatch.setattr("ingestion.core.wave_runner._catalog_session_factory", lambda: test_engine)

    def mock_submit(self, path):
        fut = concurrent.futures.Future()
        fut.set_result(_make_dummy_dataset())
        return fut

    monkeypatch.setattr(DecodePool, "submit", mock_submit)
    monkeypatch.setattr("ingestion.providers.noaa.connector.NOAAConnector.download", fake_download)
    monkeypatch.setattr("ingestion.core.wave_runner._decode_and_normalize", lambda *a, **kw: _make_dummy_dataset())
    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.initialize_run_store", lambda *a, **kw: None)
    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.pre_update_wave", lambda *a, **kw: None)
    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.publish_settled_lead", lambda *a, **kw: None)
    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.write_region_worker", lambda *a, **kw: get_s3_fs())

    def failing_finalize(*a, **kw):
        get_s3_fs()
        raise RuntimeError("Simulated manifest finalization write crash")

    monkeypatch.setattr("ingestion.core.coordinator.RunCoordinator.finalize_run", failing_finalize)

    failures: list[str] = []
    with pytest.raises(RuntimeError, match="Simulated manifest finalization write crash"):
        await _run_wave(
            spec=spec,
            args=args,
            catalog_spec=catalog_spec,
            store_path="s3://weather-data/test/cycle.zarr",
            concurrency=1,
            failures=failures,
        )

    assert len(_active_data_filesystems) == 0, f"Leaked S3 filesystems after finalization crash: {_active_data_filesystems}"


def test_realtime_preflight_runs_strictly_once_at_startup_not_per_poll(monkeypatch):
    """Verify preflight is executed exactly once at scheduler startup and never per-poll or per-wave."""
    preflight_call_count = 0

    def counting_preflight(conn_settings=None):
        nonlocal preflight_call_count
        preflight_call_count += 1

    monkeypatch.setattr("ingestion.core.s3.verify_object_store_preflight", counting_preflight)

    cfg = IngestionSettings(
        MINIO_BUCKET_NAME="weather-data",
        REALTIME_ENABLED=True,
    )
    scheduler = RealtimeScheduler(
        conn_settings=cfg,
        leadership=None,
    )

    poll_count = 0

    def mock_poll_once(*a, **kw):
        nonlocal poll_count
        poll_count += 1
        from ingestion.realtime.scheduler import PollOutcome
        if poll_count >= 5:
            scheduler.request_stop()
        return PollOutcome(kind="idle")

    monkeypatch.setattr(scheduler, "poll_once", mock_poll_once)
    monkeypatch.setattr(scheduler, "_sleep_fn", lambda *a: None)

    exit_code = scheduler.run(once=False, dry_run=False)
    assert exit_code == 0
    assert poll_count >= 5, "Scheduler did not execute multiple poll iterations"
    assert preflight_call_count == 1, (
        f"Preflight was called {preflight_call_count} times; expected strictly 1 at startup (not per poll)!"
    )


