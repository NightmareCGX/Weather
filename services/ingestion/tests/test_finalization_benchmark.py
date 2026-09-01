"""Scalability benchmark for GEFS 1110-region finalization (Phase 2).

Demonstrates:
1. Normal realtime finalize_run() scales as O(regions), executing on ~1110 markers
   instead of scanning ~1,864,800 physical Zarr data chunk objects.
2. Verified absence of recursive S3/Zarr chunk inventory calls during normal finalization.
3. Subphase timing breakdown for marker listing, marker validation, manifest write,
   and catalog reconciliation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
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
from ingestion.core.coordinator import RunCoordinator
from ingestion.core.inventory import (
    expected_write_set_fingerprint,
    region_expected_object_keys,
)
from ingestion.core.markers import (
    MARKER_V1,
    read_manifest,
    write_protocol_version,
    write_region_marker,
)
from ingestion.core.observability import PipelineProgressTracker
from ingestion.core.zarr_writer import prepare_run_store

CYCLE = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)

GEFS_VARIABLES = (
    VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),
    VariableSpec("precipitation_amount_3h", "3-Hour Precipitation Amount", "mm", "tp"),
    VariableSpec("relative_humidity_2m", "2-Meter Relative Humidity", "%", "r2"),
    VariableSpec("wind_u_10m", "10-Meter U Wind Component", "m/s", "u10"),
    VariableSpec("wind_v_10m", "10-Meter V Wind Component", "m/s", "v10"),
    VariableSpec("wind_gust", "Wind Gust", "km/h", "gust"),
    VariableSpec("visibility", "Visibility", "m", "vis"),
    VariableSpec("snow_depth", "Snow Depth", "m", "sde"),
    VariableSpec("cloud_cover_3h", "3-Hour Cloud Cover", "%", "tcc"),
    VariableSpec("cloud_ceiling", "Cloud Ceiling Height", "m", "gh"),
    VariableSpec("crain", "Categorical Rain Flag", "flag", "crain"),
    VariableSpec("csnow", "Categorical Snow Flag", "flag", "csnow"),
    VariableSpec("cfrzr", "Categorical Freezing Rain Flag", "flag", "cfrzr"),
    VariableSpec("cicep", "Categorical Ice Pellets Flag", "flag", "cicep"),
)


class _NoopLockCoordinator:
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
def catalog_engine():
    eng = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def stub_advisory_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator",
        _NoopLockCoordinator,
    )


def test_gefs_1110_region_finalization_benchmark(
    catalog_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Benchmark full GEFS 1110-region finalization (30 members x 37 leads x 14 variables)."""
    members = tuple(range(1, 31))  # 30 members
    leads = tuple(range(0, 37 * 6, 6))  # 37 leads (0, 6, ..., 216)
    total_regions = len(members) * len(leads)
    assert total_regions == 1110

    store_dir = tmp_path / "gefs_1110_bench.zarr"
    store_path = str(store_dir)

    # Initialize store with 14 variables and full dimensions
    dims = ("member", "lead_time_hours", "latitude", "longitude")
    seed_data = np.zeros((1, 1, 3, 4), dtype=np.float32)
    seed_coords = {
        "member": np.array([1], dtype=np.int32),
        "lead_time_hours": np.array([0], dtype=np.int32),
        "time": np.datetime64("2026-07-21T00:00:00", "ns"),
        "latitude": np.array([-90.0, 0.0, 90.0], dtype=np.float32),
        "longitude": np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
    }
    seed_ds = xr.Dataset(
        data_vars={v.code: (dims, seed_data) for v in GEFS_VARIABLES},
        coords=seed_coords,
    )
    prepare_run_store(
        seed_ds,
        store_path,
        expected_lead_time_hours=leads,
        expected_members=members,
    )
    write_protocol_version(store_path, MARKER_V1)

    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="National Oceanic and Atmospheric Administration",
        center_country="USA",
        model_id="gefs",
        model_name="GEFS",
        is_ensemble=True,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=CYCLE,
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree Grid",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=store_path,
        variables=GEFS_VARIABLES,
        expected_lead_time_hours=leads,
        expected_members=members,
    )

    # Seed catalog
    with Session(catalog_engine) as db:
        db.add(CenterRecord(id="center_noaa", center_id="noaa", name="NOAA", country="USA"))
        db.add(ModelRecord(id="model_gefs", center_id="noaa", model_id="gefs", name="GEFS", is_ensemble=True, resolution_km=25.0))
        db.add(ModelVersionRecord(id="version_gefs_v1.0", model_id="gefs", version_string="v1.0"))
        run_record = ModelRunRecord(
            id="run_gefs_1110",
            model_version_id="version_gefs_v1.0",
            cycle_time=CYCLE,
            status="processing",
            zarr_store_path=store_path,
        )
        db.add(run_record)
        db.commit()

    var_paths = [v.code for v in GEFS_VARIABLES]
    zarray_cache: dict[str, dict[str, Any]] = {}
    zattrs_cache: dict[str, dict[str, Any]] = {}
    member_index_cache = {m: idx for idx, m in enumerate(members)}

    # Pre-populate all 1110 COMPLETE region markers
    print(f"\nPopulating {total_regions} region markers for GEFS benchmark...")
    for m in members:
        for i, lead in enumerate(leads):
            exp_keys = region_expected_object_keys(
                store_path,
                member=m,
                lead_index=i,
                data_var_paths=var_paths,
                zarray_cache=zarray_cache,
                zattrs_cache=zattrs_cache,
                member_index_cache=member_index_cache,
            )
            write_region_marker(
                store_path,
                lead_time_hours=lead,
                member=m,
                payload={
                    "protocol_version": 1,
                    "state": "complete",
                    "generation": f"gen_m{m:02d}_L{lead:03d}",
                    "logical_region": {"lead_time_hours": lead, "member": m},
                    "expected_write_set_fingerprint": expected_write_set_fingerprint(exp_keys, []),
                    "required_materialized_object_keys": exp_keys,
                    "intentionally_omitted_fill_chunks": [],
                },
            )

    # Assert build_object_inventory is never called
    import ingestion.core.inventory as INV

    def _forbidden_build_inventory(*args, **kwargs):
        raise AssertionError("Normal finalize_run must NOT invoke build_object_inventory!")

    monkeypatch.setattr(INV, "build_object_inventory", _forbidden_build_inventory)

    tracker = PipelineProgressTracker(
        model="gefs",
        cycle_str="2026-07-21 00:00Z",
        total_items=total_regions,
    )

    coordinator = RunCoordinator(spec, store_path)
    conn = catalog_engine.connect()
    try:
        t_start = time.perf_counter()
        res = coordinator.finalize_run(
            conn,
            run_id="run_gefs_1110",
            spec=spec,
            expected_leads=leads,
            expected_members=members,
            observer=tracker,
            marker_concurrency=32,
        )
        t_elapsed = time.perf_counter() - t_start
    finally:
        conn.close()

    assert res.status == "ready"
    assert len(res.committed_regions) == total_regions

    # Verify manifest was written with valid status
    manifest = read_manifest(store_path)
    assert manifest is not None
    assert manifest.get("store_protocol_mode") == MARKER_V1
    assert manifest.get("generation") is not None

    with Session(catalog_engine) as db:
        run = db.get(ModelRunRecord, "run_gefs_1110")
        assert run is not None
        assert run.status == "ready"

    report = tracker.timeline.format_report(
        model="gefs",
        cycle_str="2026-07-21 00:00Z",
        total_items=total_regions,
    )
    print("\n" + "=" * 80)
    print("      GEFS 1110-REGION FINALIZATION SCALABILITY BENCHMARK REPORT")
    print("=" * 80)
    print(f"Total Regions: {total_regions} (30 members x 37 leads x 14 variables)")
    print("Theoretical Zarr Physical Chunks: ~1,864,800 chunks")
    print("Physical S3 Object Inventory Calls: 0 (Eliminated in Phase 2)")
    print(f"Total Finalization Elapsed Time: {t_elapsed:.3f}s")
    print("-" * 80)
    print(report)
    print("=" * 80)
