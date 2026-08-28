"""Regression tests for P1-B: Finalization Latency Optimization.

Covers:
* Test A — Concurrent Marker Retrieval Actually Occurs (bounded, >1 concurrent)
* Test B — Every Marker Read Exactly Once
* Test C — Deterministic Result under Scrambled Completion
* Test D — Single Marker Read Failure (halts finalization, clean resource teardown, no false READY)
* Test E — Invalid Marker Payload (structural rejection, treats as uncommitted)
* Test F — Missing Physical Object (full-store physical inventory rejection preserved)
* Test G — Full Physical Object Inventory Still Runs Exactly Once
* Test H — Full GFS Store Finalization (37 leads)
* Test I — Representative Multi-Member GEFS Store Finalization
* Test J — Cleanup Path Efficiency, Selectivity, and Safety
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import xarray as xr

from ingestion.cli import _cleanup_source, _cleanup_sources
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
    _read_marker_payload,
    _read_marker_payloads_bounded,
)
from ingestion.core.inventory import (
    expected_write_set_fingerprint,
    region_expected_object_keys,
)
from ingestion.core.markers import (
    MARKER_V1,
    MarkerError,
    marker_body,
    read_manifest,
    write_protocol_version,
    write_region_marker,
)
from ingestion.core.pipeline import _commit_region
from ingestion.core.zarr_writer import prepare_run_store

CYCLE = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)
TEMP_VAR = VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m")
PRECIP_VAR = VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h", "prate")


class _NoopStoreLockCoordinator:
    """No-op advisory-lock coordinator for SQLite tests."""

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
        _NoopStoreLockCoordinator,
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


def _populate_mock_region(
    store_path: str,
    lead: int,
    member: int | None,
    lead_index: int,
    expected_leads: tuple[int, ...],
    expected_members: tuple[int, ...] = (),
    generation: str = "gen_test",
    state: str = "complete",
    corrupt_evidence: bool = False,
    omit_chunk: bool = False,
) -> None:
    """Populate chunks and a COMPLETE marker for a region."""
    ds = _make_dataset(lead, member=member)
    if not omit_chunk:
        _commit_region(
            ds,
            store_path,
            member=member,
            expected_lead_time_hours=expected_leads,
            expected_members=expected_members,
        )

    expected_keys = region_expected_object_keys(
        store_path,
        member=member,
        lead_index=lead_index,
        data_var_paths=["temperature_2m", "precipitation_rate"],
    )

    fp = expected_write_set_fingerprint(expected_keys, [])
    if corrupt_evidence:
        fp = "corrupted_fingerprint_00000000"

    body = marker_body(
        lead_time_hours=lead,
        member=member,
        state=state,
        generation=generation,
        expected_write_set_fingerprint=fp,
        required_materialized_object_keys=expected_keys if not corrupt_evidence else ["bogus/chunk"],
        intentionally_omitted_fill_chunks=[],
    )
    write_region_marker(
        store_path,
        lead_time_hours=lead,
        member=member,
        payload=body,
    )


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


# ---------------------------------------------------------------------------
# Test A: Concurrent Marker Retrieval Actually Occurs
# ---------------------------------------------------------------------------


def test_concurrent_marker_retrieval_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test A: Verify bounded concurrency (> 1 and <= max_concurrency) during marker retrieval."""
    store_path = str(tmp_path / "test_a.zarr")
    keys = [f"__commit__/v1/regions/det_L{lead:04d}.json" for lead in range(24)]

    # Write markers
    for lead in range(24):
        write_region_marker(
            store_path,
            lead_time_hours=lead,
            member=None,
            payload={"state": "complete", "generation": "g1"},
        )

    lock = threading.Lock()
    active_count = 0
    max_active = 0
    read_calls = 0

    orig_read = _read_marker_payload

    def _mock_read(store: str, key: str) -> dict[str, object]:
        nonlocal active_count, max_active, read_calls
        with lock:
            active_count += 1
            read_calls += 1
            if active_count > max_active:
                max_active = active_count
        try:
            time.sleep(0.02)
            return orig_read(store, key)
        finally:
            with lock:
                active_count -= 1

    monkeypatch.setattr("ingestion.core.coordinator._read_marker_payload", _mock_read)

    # 1. Test bound = 6
    max_active = 0
    read_calls = 0
    results_6 = _read_marker_payloads_bounded(store_path, keys, max_concurrency=6)
    assert len(results_6) == 24
    assert max_active > 1, f"Expected concurrency > 1, got {max_active}"
    assert max_active <= 6, f"Expected max concurrency <= 6, got {max_active}"
    assert read_calls == 24

    # 2. Test bound = 12
    max_active = 0
    read_calls = 0
    results_12 = _read_marker_payloads_bounded(store_path, keys, max_concurrency=12)
    assert len(results_12) == 24
    assert max_active > 1, f"Expected concurrency > 1, got {max_active}"
    assert max_active <= 12, f"Expected max concurrency <= 12, got {max_active}"
    assert read_calls == 24


# ---------------------------------------------------------------------------
# Test B: Every Marker Read Exactly Once
# ---------------------------------------------------------------------------


def test_every_marker_read_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test B: For N markers, exactly N reads occur with zero skips or duplicates."""
    store_path = str(tmp_path / "test_b.zarr")
    n = 50
    keys = [f"__commit__/v1/regions/det_L{lead:04d}.json" for lead in range(n)]

    for lead in range(n):
        write_region_marker(
            store_path,
            lead_time_hours=lead,
            member=None,
            payload={"state": "complete", "generation": f"gen_{lead}"},
        )

    lock = threading.Lock()
    read_keys: list[str] = []

    orig_read = _read_marker_payload

    def _tracking_read(store: str, key: str) -> dict[str, object]:
        with lock:
            read_keys.append(key)
        return orig_read(store, key)

    monkeypatch.setattr("ingestion.core.coordinator._read_marker_payload", _tracking_read)

    results = _read_marker_payloads_bounded(store_path, keys, max_concurrency=16)

    assert len(results) == n
    assert len(read_keys) == n
    assert len(set(read_keys)) == n  # No duplicates
    assert sorted(read_keys) == sorted(keys)  # All covered


# ---------------------------------------------------------------------------
# Test C: Deterministic Result under Scrambled Completion
# ---------------------------------------------------------------------------


def test_deterministic_result_under_scrambled_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test C: Regardless of thread completion timing/order, results and ordering are deterministic."""
    store_path = str(tmp_path / "test_c.zarr")
    n = 20
    keys = [f"__commit__/v1/regions/det_L{lead:04d}.json" for lead in range(n)]

    for lead in range(n):
        write_region_marker(
            store_path,
            lead_time_hours=lead,
            member=None,
            payload={"state": "complete", "generation": f"gen_{lead:02d}"},
        )

    orig_read = _read_marker_payload

    # Sleep longer for earlier leads so later leads finish first (reverse completion)
    def _scrambled_read(store: str, key: str) -> dict[str, object]:
        region_id = key.rsplit("/", 1)[-1].removesuffix(".json")
        lead = int(region_id[len("det_L") :])
        delay = (n - lead) * 0.005  # lead 0 sleeps 0.1s, lead 19 sleeps 0.005s
        time.sleep(delay)
        return orig_read(store, key)

    monkeypatch.setattr("ingestion.core.coordinator._read_marker_payload", _scrambled_read)

    results = _read_marker_payloads_bounded(store_path, keys, max_concurrency=8)

    # Returned tuples must match the exact original sorted order of keys
    assert len(results) == n
    for i, (k, payload) in enumerate(results):
        assert k == keys[i]
        assert payload.get("generation") == f"gen_{i:02d}"


# ---------------------------------------------------------------------------
# Test D: Single Marker Read Failure
# ---------------------------------------------------------------------------


def test_single_marker_read_failure_halts_finalization(
    catalog_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test D: If one marker GET fails, finalization fails immediately, does NOT set READY, and cleans up."""
    leads = (0, 6, 12, 18)
    store_path = _init_mock_store(tmp_path / "test_d.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    run_id = _seed_db_run(catalog_engine, spec, status="partial")

    for i, lead in enumerate(leads):
        _populate_mock_region(
            store_path,
            lead=lead,
            member=None,
            lead_index=i,
            expected_leads=leads,
            expected_members=(),
        )

    # Make reading lead 12 fail
    orig_read = _read_marker_payload

    def _failing_read(store: str, key: str) -> dict[str, object]:
        if "det_L0012" in key:
            raise MarkerError("Simulated remote S3 read failure for marker det_L0012")
        return orig_read(store, key)

    monkeypatch.setattr("ingestion.core.coordinator._read_marker_payload", _failing_read)

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        with pytest.raises(MarkerError, match="Simulated remote S3 read failure"):
            coordinator.finalize_run(
                conn,
                run_id=run_id,
                spec=spec,
                expected_leads=leads,
                expected_members=(),
                marker_concurrency=4,
            )
    finally:
        conn.close()

    # Verify DB status is NOT ready
    with Session(catalog_engine) as db:
        run = db.get(ModelRunRecord, run_id)
        assert run is not None
        assert run.status != "ready"
        assert run.status == "partial"

    # Verify no manifest was written
    assert read_manifest(store_path) is None


# ---------------------------------------------------------------------------
# Test E: Invalid Marker Payload
# ---------------------------------------------------------------------------


def test_invalid_marker_payload_treated_as_uncommitted(
    catalog_engine, tmp_path: Path
) -> None:
    """Test E: Syntactically readable but structurally invalid marker evidence is rejected."""
    leads = (0, 6, 12)
    store_path = _init_mock_store(tmp_path / "test_e.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    # Lead 0 and 6 valid; lead 12 has corrupted evidence fingerprint
    _populate_mock_region(
        store_path,
        lead=0,
        member=None,
        lead_index=0,
        expected_leads=leads,
        expected_members=(),
    )
    _populate_mock_region(
        store_path,
        lead=6,
        member=None,
        lead_index=1,
        expected_leads=leads,
        expected_members=(),
    )
    _populate_mock_region(
        store_path,
        lead=12,
        member=None,
        lead_index=2,
        expected_leads=leads,
        expected_members=(),
        corrupt_evidence=True,
    )

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
        )
    finally:
        conn.close()

    # Region 12 rejected -> run status remains partial
    assert res.status == "partial"
    assert "det_L0000" in res.committed_regions
    assert "det_L0006" in res.committed_regions
    assert "det_L0012" not in res.committed_regions

    with Session(catalog_engine) as db:
        run = db.get(ModelRunRecord, run_id)
        assert run is not None
        assert run.status == "partial"


# ---------------------------------------------------------------------------
# Test F: Missing Physical Object
# ---------------------------------------------------------------------------


def test_missing_physical_object_rejected_by_inventory(
    catalog_engine, tmp_path: Path
) -> None:
    """Test F: Missing required physical chunk object rejects finalization (preserves correctness)."""
    leads = (0, 6)
    store_path = _init_mock_store(tmp_path / "test_f.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    _populate_mock_region(
        store_path,
        lead=0,
        member=None,
        lead_index=0,
        expected_leads=leads,
        expected_members=(),
    )
    # Lead 6 has COMPLETE marker written, but chunk files are physically missing
    _populate_mock_region(
        store_path,
        lead=6,
        member=None,
        lead_index=1,
        expected_leads=leads,
        expected_members=(),
        omit_chunk=True,
    )

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
        )
    finally:
        conn.close()

    assert res.status == "partial"
    assert "det_L0000" in res.committed_regions
    assert "det_L0006" not in res.committed_regions


# ---------------------------------------------------------------------------
# Test G: Full Physical Object Inventory Still Runs Exactly Once
# ---------------------------------------------------------------------------


def test_full_inventory_still_runs_once(
    catalog_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test G: Verify build_object_inventory runs once and is not replaced by per-marker HEADs."""
    leads = (0, 6, 12, 18, 24)
    store_path = _init_mock_store(tmp_path / "test_g.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path)
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    for i, lead in enumerate(leads):
        _populate_mock_region(
            store_path,
            lead=lead,
            member=None,
            lead_index=i,
            expected_leads=leads,
            expected_members=(),
        )

    inventory_call_count = 0
    import ingestion.core.inventory as INV

    orig_build_inventory = INV.build_object_inventory

    def _counting_build_inventory(store: str, var_paths: list[str]) -> set[str]:
        nonlocal inventory_call_count
        inventory_call_count += 1
        return orig_build_inventory(store, var_paths)

    monkeypatch.setattr(INV, "build_object_inventory", _counting_build_inventory)

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
        )
    finally:
        conn.close()

    assert inventory_call_count == 1, f"Expected exactly 1 inventory call, got {inventory_call_count}"
    assert res.status == "ready"
    assert len(res.committed_regions) == len(leads)


# ---------------------------------------------------------------------------
# Test H: GFS Finalization
# ---------------------------------------------------------------------------


def test_gfs_finalization_with_bounded_retrieval(
    catalog_engine, tmp_path: Path
) -> None:
    """Test H: Full GFS 37-lead finalization with bounded concurrent marker retrieval."""
    leads = tuple(range(0, 37 * 6, 6))  # 37 leads
    assert len(leads) == 37
    store_path = _init_mock_store(tmp_path / "gfs_37.zarr", leads)
    spec = _spec(expected_leads=leads, store=store_path, model="gfs")
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    for i, lead in enumerate(leads):
        _populate_mock_region(
            store_path,
            lead=lead,
            member=None,
            lead_index=i,
            expected_leads=leads,
            expected_members=(),
            generation="gfs_gen_1",
        )

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=(),
            marker_concurrency=16,
        )
    finally:
        conn.close()

    assert res.status == "ready"
    assert len(res.committed_regions) == 37
    for lead in leads:
        assert f"det_L{lead:04d}" in res.committed_regions

    with Session(catalog_engine) as db:
        run = db.get(ModelRunRecord, run_id)
        assert run is not None
        assert run.status == "ready"

    manifest = read_manifest(store_path)
    assert manifest is not None
    assert manifest.get("store_protocol_mode") == MARKER_V1


# ---------------------------------------------------------------------------
# Test I: GEFS Finalization
# ---------------------------------------------------------------------------


def test_gefs_finalization_with_bounded_retrieval(
    catalog_engine, tmp_path: Path
) -> None:
    """Test I: Scaled multi-member GEFS finalization (e.g. 10 members x 6 leads = 60 regions)."""
    members = tuple(range(1, 11))  # 10 members
    leads = (0, 6, 12, 18, 24, 30)  # 6 leads
    store_path = _init_mock_store(
        tmp_path / "gefs_60.zarr", leads, members=members
    )
    spec = _spec(
        is_ensemble=True,
        expected_leads=leads,
        expected_members=members,
        store=store_path,
        model="gefs",
    )
    run_id = _seed_db_run(catalog_engine, spec, status="processing")

    for m in members:
        for i, lead in enumerate(leads):
            _populate_mock_region(
                store_path,
                lead=lead,
                member=m,
                lead_index=i,
                expected_leads=leads,
                expected_members=members,
                generation=f"gefs_gen_{m}",
            )

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        res = coordinator.finalize_run(
            conn,
            run_id=run_id,
            spec=spec,
            expected_leads=leads,
            expected_members=members,
            marker_concurrency=32,
        )
    finally:
        conn.close()

    assert res.status == "ready"
    assert len(res.committed_regions) == 60  # 10 * 6
    for m in members:
        for lead in leads:
            assert f"mem{m:03d}_L{lead:04d}" in res.committed_regions

    with Session(catalog_engine) as db:
        run = db.get(ModelRunRecord, run_id)
        assert run is not None
        assert run.status == "ready"


# ---------------------------------------------------------------------------
# Test J: Cleanup Path Efficiency, Selectivity, and Safety
# ---------------------------------------------------------------------------


def test_cleanup_sources_selectivity_and_efficiency(tmp_path: Path) -> None:
    """Test J: Verify _cleanup_sources removes only committed primary and .idx files in a single pass."""
    staging_dir = tmp_path / "staging_test"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create 5 committed files with direct and hash index files
    committed_dests: list[Path] = []
    for i in range(5):
        grib = staging_dir / f"gfs.20260721.t00z.pgrb2.0p25.f{i*6:03d}.grib2"
        grib.write_text("grib data", encoding="utf-8")
        idx_direct = staging_dir / f"{grib.name}.idx"
        idx_direct.write_text("idx direct", encoding="utf-8")
        idx_hash = staging_dir / f"{grib.name}.90c6b.idx"
        idx_hash.write_text("idx hash", encoding="utf-8")
        committed_dests.append(grib)

    # 2. Create 2 uncommitted files with direct and hash index files
    uncommitted_grib1 = staging_dir / "gfs.20260721.t00z.pgrb2.0p25.f036.grib2"
    uncommitted_grib1.write_text("uncommitted 1", encoding="utf-8")
    (staging_dir / f"{uncommitted_grib1.name}.idx").write_text("idx direct", encoding="utf-8")
    (staging_dir / f"{uncommitted_grib1.name}.a1b2c.idx").write_text("idx hash", encoding="utf-8")

    uncommitted_grib2 = staging_dir / "gep02.20260721.t00z.pgrb2s.0p25.f006.grib2"
    uncommitted_grib2.write_text("uncommitted 2", encoding="utf-8")
    (staging_dir / f"{uncommitted_grib2.name}.f4e3d.idx").write_text("idx hash", encoding="utf-8")

    # 3. Create unrelated files
    unrelated_txt = staging_dir / "notes.txt"
    unrelated_txt.write_text("unrelated text", encoding="utf-8")

    # Run batch cleanup
    _cleanup_sources(staging_dir, committed_dests)

    # Verify committed files and all their index files are deleted
    for dest in committed_dests:
        assert not dest.exists(), f"Committed file {dest.name} was not deleted"
        assert not (staging_dir / f"{dest.name}.idx").exists()
        assert not (staging_dir / f"{dest.name}.90c6b.idx").exists()

    # Verify uncommitted files and their index files SURVIVE
    assert uncommitted_grib1.exists()
    assert (staging_dir / f"{uncommitted_grib1.name}.idx").exists()
    assert (staging_dir / f"{uncommitted_grib1.name}.a1b2c.idx").exists()

    assert uncommitted_grib2.exists()
    assert (staging_dir / f"{uncommitted_grib2.name}.f4e3d.idx").exists()

    # Verify unrelated file SURVIVES
    assert unrelated_txt.exists()


def test_cleanup_sources_idempotent_and_safe_on_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test J part 2: Verify cleanup is idempotent on missing paths and safe against PermissionError."""
    staging_dir = tmp_path / "staging_errors"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Missing file: should cleanly return with no exception
    missing_file = staging_dir / "non_existent.grib2"
    _cleanup_sources(staging_dir, [missing_file])
    _cleanup_source(missing_file)

    # 2. PermissionError during unlink: should be caught and logged as warning
    grib = staging_dir / "locked.grib2"
    grib.write_text("locked", encoding="utf-8")

    orig_unlink = Path.unlink

    def _locked_unlink(self, missing_ok=False):
        if self.name == "locked.grib2":
            raise PermissionError("File locked by external antivirus scanner")
        return orig_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _locked_unlink)

    # Must not raise
    _cleanup_sources(staging_dir, [grib])
