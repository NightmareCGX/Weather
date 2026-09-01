"""Tests for Phase 2: Finalizer Scalability and Drain Observability.

Covers:
1. Commit-marker contract tests (success, write failure, retry exhaustion, generation check, partial run)
2. No-full-scan regression test for normal finalize_run()
3. Manifest & catalog reconciliation correctness
4. Pipeline drain accounting tests (all success, write failure, decode failure, download failure, mixed)
5. Finalization subphase timing milestones & reporting
6. Full physical-store integrity audit capability
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import xarray as xr

from ingestion.core.catalog import (
    CatalogBase,
    CenterRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    RunCatalogSpec,
    VariableSpec,
)
from ingestion.core.coordinator import (
    RunCoordinator,
    WaveRegion,
    _new_generation,
)
from ingestion.core.inventory import (
    audit_store_integrity,
    expected_write_set_fingerprint,
    region_expected_object_keys,
)
from ingestion.core.markers import (
    MARKER_V1,
    read_manifest,
    read_region_marker,
    write_protocol_version,
    write_region_marker,
)
from ingestion.core.observability import (
    PipelineProgressTracker,
)
from ingestion.core.pipeline import _commit_region
from ingestion.core.zarr_writer import prepare_run_store

CYCLE = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)
TEMP_VAR = VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m")
PRECIP_VAR = VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h", "prate")


class _NoopLockCoordinator:
    """Stub advisory-lock coordinator for SQLite tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def acquire_shared_gate(self) -> None:
        pass

    def release_shared_gate(self) -> None:
        pass

    def acquire_exclusive_gate(self) -> None:
        pass

    def release_exclusive_gate(self) -> None:
        pass

    def acquire_admission(self) -> None:
        pass

    def release_admission(self) -> None:
        pass

    def acquire_shared_admission(self) -> None:
        pass

    def release_shared_admission(self) -> None:
        pass

    def acquire_region_locks(self, region_ids: list[str]) -> None:
        pass

    def release_region_locks(self, region_ids: list[str]) -> None:
        pass

    def release_all(self) -> None:
        pass

    def close_connection(self) -> None:
        pass


@pytest.fixture
def catalog_engine(tmp_path: Path):
    """An in-memory SQLite catalog engine."""
    eng = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def stub_advisory_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub PostgreSQL advisory locks so coordinator tests run against SQLite."""
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator",
        _NoopLockCoordinator,
    )


def _spec(
    *,
    is_ensemble: bool = False,
    expected_leads: tuple[int, ...] = (6,),
    expected_members: tuple[int, ...] = (),
    store: str = "/tmp/test.zarr",
    model: str = "gfs",
) -> RunCatalogSpec:
    return RunCatalogSpec(
        center_id="noaa",
        center_name="National Oceanic and Atmospheric Administration",
        center_country="USA",
        model_id=model,
        model_name="Global Forecast System" if model == "gfs" else "GEFS",
        is_ensemble=is_ensemble,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=CYCLE,
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree Grid",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=store,
        variables=(TEMP_VAR, PRECIP_VAR),
        expected_lead_time_hours=expected_leads,
        expected_members=expected_members,
    )


def _make_dataset(lead: int, member: int | None = None) -> xr.Dataset:
    coords: dict[str, Any] = {
        "lead_time_hours": np.array([lead], dtype=np.int32),
        "time": np.datetime64("2026-07-22T00:00:00", "ns"),
        "latitude": np.array([-90.0, 0.0, 90.0], dtype=np.float32),
        "longitude": np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
    }
    dims = ("lead_time_hours", "latitude", "longitude")
    shape = (1, 3, 4)
    if member is not None:
        coords["member"] = np.array([member], dtype=np.int32)
        dims = ("member", "lead_time_hours", "latitude", "longitude")
        shape = (1, 1, 3, 4)
    data = np.full(shape, float(lead) + (member or 0.0), dtype=np.float32)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (dims, data),
            "precipitation_rate": (dims, data),
        },
        coords=coords,
    )


def _init_mock_store(
    store_dir: Path,
    leads: tuple[int, ...],
    members: tuple[int, ...] = (),
) -> str:
    """Set up a real Zarr store using prepare_run_store and initialize protocol version."""
    store_path = str(store_dir)
    seed = _make_dataset(leads[0], member=members[0] if members else None)
    prepare_run_store(
        seed,
        store_path,
        expected_lead_time_hours=leads,
        expected_members=members,
    )
    write_protocol_version(store_path, MARKER_V1)
    return store_path


def _seed_db_run(engine, spec: RunCatalogSpec, status: str = "partial", run_id: str = "run_test") -> str:
    """Create initial DB records for a run."""
    with Session(engine) as db:
        if db.query(CenterRecord).filter_by(center_id=spec.center_id).first() is None:
            db.add(
                CenterRecord(
                    id=f"center_{spec.center_id}",
                    center_id=spec.center_id,
                    name=spec.center_name,
                    country=spec.center_country,
                )
            )
        if db.query(ModelRecord).filter_by(model_id=spec.model_id).first() is None:
            db.add(
                ModelRecord(
                    id=f"model_{spec.model_id}",
                    center_id=spec.center_id,
                    model_id=spec.model_id,
                    name=spec.model_name,
                    is_ensemble=spec.is_ensemble,
                    resolution_km=spec.resolution_km,
                )
            )
        version_id = f"version_{spec.model_id}_{spec.version_string}"
        if db.query(ModelVersionRecord).filter_by(id=version_id).first() is None:
            db.add(
                ModelVersionRecord(
                    id=version_id,
                    model_id=spec.model_id,
                    version_string=spec.version_string,
                )
            )
        run_record = ModelRunRecord(
            id=run_id,
            model_version_id=version_id,
            cycle_time=spec.cycle_time,
            status=status,
            zarr_store_path=spec.zarr_store_path,
        )
        db.add(run_record)
        db.commit()
    return run_id


# =============================================================================
# 1. Commit-Marker Contract Tests
# =============================================================================


def test_marker_contract_successful_write_produces_complete_marker(
    catalog_engine, tmp_path: Path
) -> None:
    """Invariant 1: Writer produces a valid COMPLETE marker only after all data writes succeed."""
    leads = (0, 6)
    store_path = _init_mock_store(tmp_path / "contract_success.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        # Pre-update wave to create UPDATING markers
        gen_0 = _new_generation()
        gen_6 = _new_generation()
        regions = [
            WaveRegion(lead_time_hours=0, member=None, generation=gen_0),
            WaveRegion(lead_time_hours=6, member=None, generation=gen_6),
        ]
        import concurrent.futures
        import threading
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        coordinator.pre_update_wave(
            conn,
            regions=regions,
            run_id=run_id,
            is_same_cycle=False,
            executor=executor,
            cancel_event=threading.Event(),
        )

        # Write region 0
        ds0 = _make_dataset(0)
        coordinator.write_region_worker(
            conn,
            dataset=ds0,
            member=None,
            generation=gen_0,
            expected_leads=leads,
            expected_members=(),
        )

        # Check marker for region 0
        m0 = read_region_marker(store_path, lead_time_hours=0, member=None)
        assert m0.get("state") == "complete"
        assert m0.get("generation") == gen_0
        assert len(m0.get("required_materialized_object_keys")) > 0
        assert m0.get("logical_region") == {"lead_time_hours": 0}

        # Check marker for region 6 (still updating)
        m6 = read_region_marker(store_path, lead_time_hours=6, member=None)
        assert m6.get("state") == "updating"

        # Finalize run
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
        )
        assert res.status == "partial"
        assert "det_L0000" in res.committed_regions
        assert "det_L0006" not in res.committed_regions

        # Write region 6
        ds6 = _make_dataset(6)
        coordinator.write_region_worker(
            conn,
            dataset=ds6,
            member=None,
            generation=gen_6,
            expected_leads=leads,
            expected_members=(),
        )

        res2 = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
        )
        assert res2.status == "ready"
        assert "det_L0000" in res2.committed_regions
        assert "det_L0006" in res2.committed_regions
    finally:
        conn.close()
        executor.shutdown(wait=True)


def test_marker_contract_write_exception_prevents_complete_marker(
    catalog_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 2: Any exception during chunk writes prevents COMPLETE marker creation."""
    leads = (0, 6)
    store_path = _init_mock_store(tmp_path / "contract_failure.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        gen = _new_generation()
        regions = [WaveRegion(lead_time_hours=6, member=None, generation=gen)]
        import concurrent.futures
        import threading
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        coordinator.pre_update_wave(
            conn,
            regions=regions,
            run_id=run_id,
            is_same_cycle=False,
            executor=executor,
            cancel_event=threading.Event(),
        )

        # Inject simulated failure during _commit_region
        def _failing_commit(*args, **kwargs):
            raise IOError("Simulated disk write failure")

        monkeypatch.setattr("ingestion.core.coordinator._commit_region", _failing_commit)

        ds = _make_dataset(6)
        with pytest.raises(IOError, match="Simulated disk write failure"):
            coordinator.write_region_worker(
                conn,
                dataset=ds,
                member=None,
                generation=gen,
                expected_leads=leads,
                expected_members=(),
            )

        # Marker must NOT be complete
        marker = read_region_marker(store_path, lead_time_hours=6, member=None)
        assert marker.get("state") == "updating"

        # Finalizer rejects uncommitted region (0 of 2 committed -> processing, not ready)
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
        )
        assert res.status == "processing"
        assert "det_L0006" not in res.committed_regions

        # If lead 0 succeeds while lead 6 fails, run becomes partial (1 of 2 committed)
        # Restore original _commit_region for lead 0
        monkeypatch.setattr("ingestion.core.coordinator._commit_region", _commit_region)
        gen_0 = _new_generation()
        coordinator.pre_update_wave(
            conn,
            regions=[WaveRegion(lead_time_hours=0, member=None, generation=gen_0)],
            run_id=run_id,
            is_same_cycle=False,
            executor=executor,
            cancel_event=threading.Event(),
        )
        coordinator.write_region_worker(
            conn,
            dataset=_make_dataset(0),
            member=None,
            generation=gen_0,
            expected_leads=leads,
            expected_members=(),
        )
        res_partial = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
        )
        assert res_partial.status == "partial"
        assert "det_L0000" in res_partial.committed_regions
        assert "det_L0006" not in res_partial.committed_regions
    finally:
        conn.close()
        executor.shutdown(wait=True)


def test_marker_contract_retry_exhaustion_no_complete_marker(
    catalog_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 3: Retry exhaustion during transient write errors cannot produce a COMPLETE marker."""
    leads = (0, 6)
    store_path = _init_mock_store(tmp_path / "retry_exhaust.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        gen = _new_generation()
        regions = [WaveRegion(lead_time_hours=6, member=None, generation=gen)]
        import concurrent.futures
        import threading
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        coordinator.pre_update_wave(
            conn,
            regions=regions,
            run_id=run_id,
            is_same_cycle=False,
            executor=executor,
            cancel_event=threading.Event(),
        )

        # Force retryable ConnectionResetError on every attempt
        attempt_count = 0

        def _retryable_error(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise ConnectionResetError("Transient network drop")

        monkeypatch.setattr("ingestion.core.coordinator._commit_region", _retryable_error)

        ds = _make_dataset(6)
        with pytest.raises(ConnectionResetError):
            coordinator.write_region_worker(
                conn,
                dataset=ds,
                member=None,
                generation=gen,
                expected_leads=leads,
                expected_members=(),
            )

        assert attempt_count == 3  # All 3 attempts exhausted
        marker = read_region_marker(store_path, lead_time_hours=6, member=None)
        assert marker.get("state") == "updating"
    finally:
        conn.close()
        executor.shutdown(wait=True)


def test_marker_contract_generation_mismatch_prevents_write(
    catalog_engine, tmp_path: Path
) -> None:
    """Invariant 4: Generation mismatch aborts region worker with zero data writes."""
    leads = (0, 6)
    store_path = _init_mock_store(tmp_path / "gen_mismatch.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    _seed_db_run(catalog_engine, spec, status="processing")

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        # Write marker with gen_A
        write_region_marker(
            store_path,
            lead_time_hours=6,
            member=None,
            payload={
                "protocol_version": 1,
                "state": "updating",
                "generation": "gen_A",
                "logical_region": {"lead_time_hours": 6},
                "expected_write_set_fingerprint": "",
                "required_materialized_object_keys": [],
                "intentionally_omitted_fill_chunks": [],
            },
        )

        # Worker runs with wrong generation gen_B
        from ingestion.core.base import StoreSchemaMismatchError
        ds = _make_dataset(6)
        with pytest.raises(StoreSchemaMismatchError, match="is not owned by generation gen_B"):
            coordinator.write_region_worker(
                conn,
                dataset=ds,
                member=None,
                generation="gen_B",
                expected_leads=leads,
                expected_members=(),
            )

        # Marker remains updating with gen_A
        m = read_region_marker(store_path, lead_time_hours=6, member=None)
        assert m.get("generation") == "gen_A"
        assert m.get("state") == "updating"
    finally:
        conn.close()


# =============================================================================
# 2. No-Full-Scan Regression Test & Manifest / Catalog Reconcile
# =============================================================================


def test_normal_finalize_run_does_not_invoke_full_object_scan(
    catalog_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression Test: Prove normal finalize_run() does NOT invoke build_object_inventory."""
    members = tuple(range(1, 11))
    leads = (0, 6, 12, 18, 24, 30)
    store_path = _init_mock_store(tmp_path / "gefs_fast_final.zarr", leads, members=members)
    spec = _spec(
        is_ensemble=True,
        expected_leads=leads,
        expected_members=members,
        store=store_path,
        model="gefs",
    )
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    # Populate COMPLETE markers for all 60 regions
    for m in members:
        for i, lead in enumerate(leads):
            ds = _make_dataset(lead, member=m)
            _commit_region(
                ds,
                store_path,
                member=m,
                expected_lead_time_hours=leads,
                expected_members=members,
            )
            exp_keys = region_expected_object_keys(
                store_path,
                member=m,
                lead_index=i,
                data_var_paths=["temperature_2m", "precipitation_rate"],
            )
            write_region_marker(
                store_path,
                lead_time_hours=lead,
                member=m,
                payload={
                    "protocol_version": 1,
                    "state": "complete",
                    "generation": f"gen_m{m}_l{lead}",
                    "logical_region": {"lead_time_hours": lead, "member": m},
                    "expected_write_set_fingerprint": expected_write_set_fingerprint(exp_keys, []),
                    "required_materialized_object_keys": exp_keys,
                    "intentionally_omitted_fill_chunks": [],
                },
            )

    # Patch build_object_inventory to fail test if invoked
    import ingestion.core.inventory as INV

    def _forbidden_scan(*args, **kwargs):
        raise AssertionError("Normal finalize_run must NOT call build_object_inventory!")

    monkeypatch.setattr(INV, "build_object_inventory", _forbidden_scan)

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=members,
            marker_concurrency=16,
        )
    finally:
        conn.close()

    assert res.status == "ready"
    assert len(res.committed_regions) == 60

    # Manifest and catalog state verified
    manifest = read_manifest(store_path)
    assert manifest is not None
    assert manifest.get("store_protocol_mode") == MARKER_V1
    assert manifest.get("run_identity", {}).get("is_ensemble") is True

    with Session(catalog_engine) as db:
        run = db.get(ModelRunRecord, run_id)
        assert run is not None
        assert run.status == "ready"


# =============================================================================
# 3. Pipeline Drain Accounting Tests
# =============================================================================


def test_drain_accounting_all_success() -> None:
    """Drain accounting: all items succeed -> downloads, decodes, writes all drain cleanly."""
    tracker = PipelineProgressTracker(model="gfs", cycle_str="2026-07-21 00:00Z", total_items=3)

    # Seed
    tracker.on_download_start(member=None, lead=0, is_seed=True)
    tracker.on_download_complete(member=None, lead=0, duration_ms=10.0)
    tracker.on_decode_start(member=None, lead=0)
    tracker.on_decode_complete(member=None, lead=0, duration_ms=10.0)

    # Item 2 & 3 downloads
    tracker.on_download_start(member=None, lead=6, is_seed=False)
    tracker.on_download_complete(member=None, lead=6, duration_ms=10.0)
    tracker.on_download_start(member=None, lead=12, is_seed=False)
    tracker.on_download_complete(member=None, lead=12, duration_ms=10.0)

    assert tracker.timeline.get("downloads_drained") is not None
    assert tracker.timeline.get("decodes_drained") is None
    assert tracker.timeline.get("writes_drained") is None

    # Decodes
    tracker.on_decode_start(member=None, lead=6)
    tracker.on_decode_complete(member=None, lead=6, duration_ms=10.0)
    tracker.on_decode_start(member=None, lead=12)
    tracker.on_decode_complete(member=None, lead=12, duration_ms=10.0)

    assert tracker.timeline.get("decodes_drained") is not None
    assert tracker.timeline.get("writes_drained") is None

    # Writes
    for lead in (0, 6, 12):
        tracker.on_write_start(member=None, lead=lead)
        tracker.on_write_complete(member=None, lead=lead, duration_ms=10.0)

    assert tracker.timeline.get("writes_drained") is not None


def test_drain_accounting_write_failure() -> None:
    """Drain accounting: 1 write failure -> writes_drained is accurately recorded."""
    tracker = PipelineProgressTracker(model="gfs", cycle_str="2026-07-21 00:00Z", total_items=2)

    for lead in (0, 6):
        tracker.on_download_start(member=None, lead=lead, is_seed=(lead == 0))
        tracker.on_download_complete(member=None, lead=lead, duration_ms=10.0)
        tracker.on_decode_start(member=None, lead=lead)
        tracker.on_decode_complete(member=None, lead=lead, duration_ms=10.0)

    # Lead 0 succeeds, Lead 6 fails
    tracker.on_write_start(member=None, lead=0)
    tracker.on_write_complete(member=None, lead=0, duration_ms=10.0)
    assert tracker.timeline.get("writes_drained") is None

    tracker.on_write_start(member=None, lead=6)
    tracker.on_write_failed(member=None, lead=6, duration_ms=10.0)

    assert tracker.timeline.get("writes_drained") is not None
    assert tracker.counters.write_done == 1
    assert tracker.counters.write_failed == 1


def test_drain_accounting_decode_failure_before_write() -> None:
    """Drain accounting: 1 decode failure before write -> writes_drained is NOT blocked."""
    tracker = PipelineProgressTracker(model="gfs", cycle_str="2026-07-21 00:00Z", total_items=2)

    for lead in (0, 6):
        tracker.on_download_start(member=None, lead=lead, is_seed=(lead == 0))
        tracker.on_download_complete(member=None, lead=lead, duration_ms=10.0)

    # Lead 0 decode succeeds & writes
    tracker.on_decode_start(member=None, lead=0)
    tracker.on_decode_complete(member=None, lead=0, duration_ms=10.0)
    tracker.on_write_start(member=None, lead=0)
    tracker.on_write_complete(member=None, lead=0, duration_ms=10.0)

    assert tracker.timeline.get("decodes_drained") is None
    assert tracker.timeline.get("writes_drained") is None

    # Lead 6 decode FAILS (never reaches write)
    tracker.on_decode_start(member=None, lead=6)
    tracker.on_decode_failed(member=None, lead=6, duration_ms=10.0)

    assert tracker.timeline.get("decodes_drained") is not None
    assert tracker.timeline.get("writes_drained") is not None
    assert tracker.counters.decode_failed == 1
    assert tracker.counters.write_done == 1
    assert tracker.counters.write_failed == 0


def test_drain_accounting_download_failure_before_decode() -> None:
    """Drain accounting: 1 download failure before decode -> decodes_drained & writes_drained recorded."""
    tracker = PipelineProgressTracker(model="gfs", cycle_str="2026-07-21 00:00Z", total_items=2)

    # Lead 0 download & decode & write succeed
    tracker.on_download_start(member=None, lead=0, is_seed=True)
    tracker.on_download_complete(member=None, lead=0, duration_ms=10.0)
    tracker.on_decode_start(member=None, lead=0)
    tracker.on_decode_complete(member=None, lead=0, duration_ms=10.0)
    tracker.on_write_start(member=None, lead=0)
    tracker.on_write_complete(member=None, lead=0, duration_ms=10.0)

    # Lead 6 download FAILS (never reaches decode or write)
    tracker.on_download_start(member=None, lead=6, is_seed=False)
    tracker.on_download_failed(member=None, lead=6, duration_ms=10.0)

    assert tracker.timeline.get("downloads_drained") is not None
    assert tracker.timeline.get("decodes_drained") is not None
    assert tracker.timeline.get("writes_drained") is not None
    assert tracker.counters.download_failed == 1
    assert tracker.counters.write_done == 1


def test_drain_accounting_mixed_failures() -> None:
    """Drain accounting: mixed download, decode, write failures and successes settle all milestones."""
    tracker = PipelineProgressTracker(model="gefs", cycle_str="2026-07-21 00:00Z", total_items=4)

    # Item 1: Download failed
    tracker.on_download_start(member=1, lead=6, is_seed=False)
    tracker.on_download_failed(member=1, lead=6, duration_ms=10.0)

    # Item 2: Decode failed
    tracker.on_download_start(member=2, lead=6, is_seed=False)
    tracker.on_download_complete(member=2, lead=6, duration_ms=10.0)
    tracker.on_decode_start(member=2, lead=6)
    tracker.on_decode_failed(member=2, lead=6, duration_ms=10.0)

    # Item 3: Write failed
    tracker.on_download_start(member=3, lead=6, is_seed=False)
    tracker.on_download_complete(member=3, lead=6, duration_ms=10.0)
    tracker.on_decode_start(member=3, lead=6)
    tracker.on_decode_complete(member=3, lead=6, duration_ms=10.0)
    tracker.on_write_start(member=3, lead=6)
    tracker.on_write_failed(member=3, lead=6, duration_ms=10.0)

    # Item 4: All success
    tracker.on_download_start(member=4, lead=6, is_seed=True)
    tracker.on_download_complete(member=4, lead=6, duration_ms=10.0)
    tracker.on_decode_start(member=4, lead=6)
    tracker.on_decode_complete(member=4, lead=6, duration_ms=10.0)
    tracker.on_write_start(member=4, lead=6)
    tracker.on_write_complete(member=4, lead=6, duration_ms=10.0)

    assert tracker.timeline.get("downloads_drained") is not None
    assert tracker.timeline.get("decodes_drained") is not None
    assert tracker.timeline.get("writes_drained") is not None
    assert tracker.counters.download_failed == 1
    assert tracker.counters.decode_failed == 1
    assert tracker.counters.write_failed == 1
    assert tracker.counters.write_done == 1


# =============================================================================
# 4. Finalization Subphase Timing Milestones & Observability
# =============================================================================


def test_finalization_timing_milestones_emitted_in_order(
    catalog_engine, tmp_path: Path
) -> None:
    """Finalization records granular timing milestones in strict monotonic order."""
    leads = (0, 6)
    store_path = _init_mock_store(tmp_path / "timing_test.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    for i, lead in enumerate(leads):
        ds = _make_dataset(lead)
        _commit_region(ds, store_path, member=None, expected_lead_time_hours=leads, expected_members=())
        exp_keys = region_expected_object_keys(
            store_path, member=None, lead_index=i, data_var_paths=["temperature_2m", "precipitation_rate"]
        )
        write_region_marker(
            store_path,
            lead_time_hours=lead,
            member=None,
            payload={
                "protocol_version": 1,
                "state": "complete",
                "generation": f"gen_{lead}",
                "logical_region": {"lead_time_hours": lead},
                "expected_write_set_fingerprint": expected_write_set_fingerprint(exp_keys, []),
                "required_materialized_object_keys": exp_keys,
                "intentionally_omitted_fill_chunks": [],
            },
        )

    tracker = PipelineProgressTracker(model="gfs", cycle_str="2026-07-21 00:00Z", total_items=2)
    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
            observer=tracker,
        )
    finally:
        conn.close()

    assert res.status == "ready"

    # Verify all milestones exist
    t_start = tracker.timeline.get("finalize_start")
    t_ml_start = tracker.timeline.get("marker_listing_start")
    t_ml_end = tracker.timeline.get("marker_listing_complete")
    t_mrv_start = tracker.timeline.get("marker_read_validation_start")
    t_mrv_end = tracker.timeline.get("marker_read_validation_complete")
    t_mw_start = tracker.timeline.get("manifest_write_start")
    t_mw_end = tracker.timeline.get("manifest_write_complete")
    t_cr_start = tracker.timeline.get("catalog_reconcile_start")
    t_cr_end = tracker.timeline.get("catalog_reconcile_complete")
    t_end = tracker.timeline.get("finalize_complete")

    assert t_start is not None
    assert t_ml_start is not None
    assert t_ml_end is not None
    assert t_mrv_start is not None
    assert t_mrv_end is not None
    assert t_mw_start is not None
    assert t_mw_end is not None
    assert t_cr_start is not None
    assert t_cr_end is not None
    assert t_end is not None

    # Verify monotonic ordering
    assert t_start <= t_ml_start <= t_ml_end
    assert t_ml_end <= t_mrv_start <= t_mrv_end
    assert t_mrv_end <= t_mw_start <= t_mw_end
    assert t_mw_end <= t_cr_start <= t_cr_end
    assert t_cr_end <= t_end

    # Check report format includes finalization breakdown
    report = tracker.timeline.format_report(model="gfs", cycle_str="2026-07-21 00:00Z", total_items=2)
    assert "Finalization Breakdown:" in report
    assert "Marker Listing" in report
    assert "Marker Read & Validation" in report
    assert "Manifest Write" in report
    assert "Catalog Reconciliation" in report
    assert "finalize_complete" in report


# =============================================================================
# 5. Full Physical-Store Integrity Audit Capability
# =============================================================================


def test_audit_store_integrity_detects_corrupted_marker_and_missing_object(
    tmp_path: Path
) -> None:
    """Store integrity audit detects corrupted fingerprints and missing physical objects."""
    store_dir = tmp_path / "audit_corrupt.zarr"
    store = str(store_dir)
    coords = {
        "lead_time_hours": np.array([0, 6, 12], dtype=np.int32),
        "time": np.datetime64("2026-07-22T00:00:00", "ns"),
        "latitude": np.array([-90.0, 0.0, 90.0], dtype=np.float32),
        "longitude": np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
    }
    dims = ("lead_time_hours", "latitude", "longitude")
    data = np.zeros((1, 3, 4), dtype=np.float32)
    seed = xr.Dataset(
        data_vars={"temperature_2m": (dims, data)},
        coords={k: v[:1] if k == "lead_time_hours" else v for k, v in coords.items()},
    )
    prepare_run_store(seed, store, expected_lead_time_hours=(0, 6, 12), expected_members=())
    write_protocol_version(store, MARKER_V1)

    import os
    def _write_chunk(store_p: str, k: str) -> None:
        full = os.path.join(store_p, *k.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(b"chunk")

    # Lead 0: 100% valid
    exp0 = region_expected_object_keys(store, member=None, lead_index=0, data_var_paths=["temperature_2m"])
    for k in exp0:
        _write_chunk(store, k)
    write_region_marker(
        store,
        lead_time_hours=0,
        member=None,
        payload={
            "protocol_version": 1,
            "state": "complete",
            "generation": "gen0",
            "logical_region": {"lead_time_hours": 0},
            "expected_write_set_fingerprint": expected_write_set_fingerprint(exp0, []),
            "required_materialized_object_keys": exp0,
            "intentionally_omitted_fill_chunks": [],
        },
    )

    # Lead 6: Corrupted fingerprint
    exp6 = region_expected_object_keys(store, member=None, lead_index=1, data_var_paths=["temperature_2m"])
    for k in exp6:
        _write_chunk(store, k)
    write_region_marker(
        store,
        lead_time_hours=6,
        member=None,
        payload={
            "protocol_version": 1,
            "state": "complete",
            "generation": "gen6",
            "logical_region": {"lead_time_hours": 6},
            "expected_write_set_fingerprint": "corrupted_bad_fingerprint_hash",
            "required_materialized_object_keys": exp6,
            "intentionally_omitted_fill_chunks": [],
        },
    )

    # Lead 12: Missing physical object
    exp12 = region_expected_object_keys(store, member=None, lead_index=2, data_var_paths=["temperature_2m"])
    write_region_marker(
        store,
        lead_time_hours=12,
        member=None,
        payload={
            "protocol_version": 1,
            "state": "complete",
            "generation": "gen12",
            "logical_region": {"lead_time_hours": 12},
            "expected_write_set_fingerprint": expected_write_set_fingerprint(exp12, []),
            "required_materialized_object_keys": exp12,
            "intentionally_omitted_fill_chunks": [],
        },
    )

    report = audit_store_integrity(store, array_paths=["temperature_2m"])
    assert not report.is_valid
    assert report.total_markers == 3
    assert report.valid_markers == 1
    assert report.invalid_markers == 2
    assert "det_L0006" in report.errors_by_region
    assert "det_L0012" in report.errors_by_region
    assert any("fingerprint" in err for err in report.errors_by_region.values())
    assert any("missing" in err for err in report.errors_by_region.values())
