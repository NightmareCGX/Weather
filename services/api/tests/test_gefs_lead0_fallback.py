"""Comprehensive regression tests for GEFS official geavg lead-0 fallback (Lifecycle V2).

Verifies:
A. Newest cycle lead 0 instantaneous variable (temperature_2m) comes from newest cycle lead 0.
B. Existing fallback interval variable at lead 0 comes from previous cycle positive lead with SAME valid_time.
C. GEFS version: fallback value comes from previous-cycle official geavg mean shard (shard.mean_Lxxxx.shard).
D. Previous cycle members exist but geavg does not: do NOT runtime-average members as substitute for official mean.
E. No eligible previous positive lead exists: preserve the pre-existing unavailable/null behavior.
F. All forecast entries remain present (complete lead horizon of 81 leads).
G. Map and right-side forecast resolve the same fallback source/value for the same target valid_time.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.models.entities import (
    Base,
    EnsembleMember,
    EnsembleMemberProduct,
    ForecastCenter,
    ForecastCycleLifecycle,
    ForecastGrid,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.point_forecast import (
    ResolvedLocation,
    build_point_forecast,
)
from api.services.tiles import (
    resolve_tile_read_context,
)
from tests.test_sharded_reader import _build_test_shard


@pytest.fixture(autouse=True)
def _no_reader_pool(monkeypatch):
    """Force the direct bounded read path for standalone SQLite tests."""
    try:
        import api.main as main
        if hasattr(main, "reader_pool"):
            monkeypatch.setattr(main, "reader_pool", None)
        if hasattr(main, "reader_lifecycle"):
            monkeypatch.setattr(main, "reader_lifecycle", None)
    except ImportError:
        pass
    yield


LATITUDES = [90.0 - i * 0.25 for i in range(721)]
LONGITUDES = [0.0 + j * 0.25 for j in range(1440)]
GEFS_81_LEADS = [i * 3 for i in range(81)]  # 0, 3, 6, ..., 240


def _create_sharded_gefs_store(
    store_dir: Path,
    *,
    generation: str,
    leads: list[int],
    mean_shards: dict[str, dict[int, float]],
    member_shards: dict[str, dict[tuple[int, int], float]] | None = None,
) -> None:
    """Helper to create a valid sharded_v1 Zarr store with official mean and/or member shards."""
    store_dir.mkdir(parents=True, exist_ok=True)

    all_vars = set(mean_shards.keys())
    if member_shards:
        all_vars.update(member_shards.keys())

    ds = xr.Dataset(
        data_vars={
            v: (("lead_time_hours", "latitude", "longitude"), np.zeros((len(leads), 721, 1440), dtype=np.float32))
            for v in all_vars
        },
        coords={
            "lead_time_hours": leads,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
            "member": list(range(1, 31)),
        },
    )
    ds.to_zarr(str(store_dir), mode="w", consolidated=True, zarr_format=2)

    manifest_dir = store_dir / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({
            "manifest_schema_version": 1,
            "generation": generation,
            "storage_format_version": "sharded_v1",
        })
    )

    # Write official mean shards
    for v_code, lead_vals in mean_shards.items():
        v_dir = store_dir / v_code
        v_dir.mkdir(parents=True, exist_ok=True)
        for lead, val in lead_vals.items():
            shard_path = v_dir / f"shard.mean_L{lead:04d}.shard"
            shard_path.write_bytes(_build_test_shard(val_offset=val))

    # Write member shards if specified
    if member_shards:
        for v_code, mem_lead_vals in member_shards.items():
            v_dir = store_dir / v_code
            v_dir.mkdir(parents=True, exist_ok=True)
            for (m, lead), val in mem_lead_vals.items():
                shard_path = v_dir / f"shard.mem{m:03d}_L{lead:04d}.shard"
                shard_path.write_bytes(_build_test_shard(val_offset=val))


def _init_gefs_db(engine, c_18z: datetime, c_00z: datetime, store_18z: str, store_00z: str) -> None:
    tables = [
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ForecastGrid.__table__,
        ForecastVariable.__table__,
        ModelRun.__table__,
        ForecastProduct.__table__,
        EnsembleMember.__table__,
        EnsembleMemberProduct.__table__,
        ForecastCycleLifecycle.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)

    with Session(engine) as session:
        session.add(ForecastCenter(id="noaa", center_id="noaa", name="NOAA", country="US"))
        session.add(Model(id="gefs", model_id="gefs", name="GEFS", center_id="noaa", is_ensemble=True, resolution_km=25.0))
        session.add(ModelVersion(id="v_gefs", model_id="gefs", version_string="v1.0"))
        session.add(ForecastGrid(id="g_glob", grid_code="global_025deg", name="Global", resolution_km=25.0))
        for v, u in [
            ("temperature_2m", "°C"),
            ("precipitation_amount_3h", "mm"),
            ("cloud_cover_3h", "%"),
            ("crain", "flag"),
            ("csnow", "flag"),
        ]:
            session.add(ForecastVariable(id=f"v_{v}", variable_code=v, name=v, unit=u))
        session.add(ModelRun(id="r_18z", model_version_id="v_gefs", cycle_time=c_18z, status="ready", zarr_store_path=store_18z))
        session.add(ModelRun(id="r_00z", model_version_id="v_gefs", cycle_time=c_00z, status="ready", zarr_store_path=store_00z))

        # Products for 18Z (lead 6)
        for v in ["temperature_2m", "precipitation_amount_3h", "cloud_cover_3h", "crain", "csnow"]:
            session.add(ForecastProduct(id=f"p_18_{v}", run_id="r_18z", variable_id=v, grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=6))

        # Products for 00Z (lead 0)
        session.add(ForecastProduct(id="p_00_t2m", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=0))
        session.commit()


def test_gefs_lead0_fallback_contract_point_forecast(tmp_path: Path):
    """Test Requirements A, B, C, F for GEFS:
    A. Newest cycle lead 0 instantaneous variable (temperature_2m) comes from 00Z lead 0.
    B. Existing fallback interval variable (precipitation_amount_3h, cloud_cover_3h) comes from 18Z lead 6.
    C. GEFS official geavg mean shard (shard.mean_L0006.shard) is the data source.
    F. Companion variable crain and precipitation_type come from 18Z lead 6.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/gefs_test.db")
    c_18z = datetime(2026, 9, 6, 18, 0, tzinfo=timezone.utc)
    c_00z = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)

    # 18Z: lead 6 has temp=10.0, precip=7.5, cloud=80.0, crain=1.0, csnow=0.0
    store_18z = tmp_path / "gefs_18z.zarr"
    _create_sharded_gefs_store(
        store_18z,
        generation="gen_18z",
        leads=[6],
        mean_shards={
            "temperature_2m": {6: 10.0},
            "precipitation_amount_3h": {6: 7.5},
            "cloud_cover_3h": {6: 80.0},
            "crain": {6: 1.0},
            "csnow": {6: -100.0},
        },
    )

    # 00Z: lead 0 has temp=15.0 (only instantaneous)
    store_00z = tmp_path / "gefs_00z.zarr"
    _create_sharded_gefs_store(
        store_00z,
        generation="gen_00z",
        leads=[0],
        mean_shards={
            "temperature_2m": {0: 15.0},
        },
    )

    _init_gefs_db(engine, c_18z, c_00z, str(store_18z), str(store_00z))

    loc = ResolvedLocation(latitude=38.0, longitude=-107.0, elevation_m=None, resolved_via="coords")
    with Session(engine) as session:
        data = build_point_forecast(
            session,
            location=loc,
            model="gefs",
            variables=["temperature_2m", "precipitation_amount_3h", "cloud_cover_3h", "crain"],
            units="metric",
            start_lead_time_hours=0,
            end_lead_time_hours=0,
        )

    assert len(data.forecasts) == 1
    f0 = data.forecasts[0]
    assert f0.lead_time_hours == 0
    assert f0.valid_time == c_00z
    assert f0.cycle_time == c_00z

    # A: Instantaneous temperature_2m comes from 00Z lead 0 (val_offset 15.0 + chunk 40 = 55.0)
    assert getattr(f0, "temperature_2m") == pytest.approx(55.0, abs=1e-2)

    # B & C: Interval variables come from 18Z lead 6 official mean shard
    # precip: val_offset 7.5 + chunk 40 = 47.5
    assert getattr(f0, "precipitation_amount_3h") == pytest.approx(47.5, abs=1e-2)
    # cloud: val_offset 80.0 + chunk 40 = 120.0
    assert getattr(f0, "cloud_cover_3h") == pytest.approx(120.0, abs=1e-2)

    # Companion variable crain and phase classification come from 18Z lead 6
    assert getattr(f0, "precipitation_type") == "rain"
    assert getattr(f0, "crain") == pytest.approx(41.0, abs=1e-2)


def test_gefs_lead0_fallback_map_tile_and_provenance(tmp_path: Path):
    """Test Requirement G: Map tiles and right-side forecast resolve the same fallback source.
    - Map tile for precipitation_amount_3h at valid_time 00Z resolves to 18Z lead 6.
    - Map tile for temperature_2m at valid_time 00Z resolves to 00Z lead 0.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/gefs_map_test.db")
    c_18z = datetime(2026, 9, 6, 18, 0, tzinfo=timezone.utc)
    c_00z = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)

    store_18z = tmp_path / "gefs_18z.zarr"
    _create_sharded_gefs_store(
        store_18z,
        generation="gen_18z",
        leads=[6],
        mean_shards={
            "temperature_2m": {6: 10.0},
            "precipitation_amount_3h": {6: 7.5},
            "cloud_cover_3h": {6: 80.0},
        },
    )

    store_00z = tmp_path / "gefs_00z.zarr"
    _create_sharded_gefs_store(
        store_00z,
        generation="gen_00z",
        leads=[0],
        mean_shards={
            "temperature_2m": {0: 15.0},
        },
    )

    _init_gefs_db(engine, c_18z, c_00z, str(store_18z), str(store_00z))

    with Session(engine) as session:
        # Context for precipitation_amount_3h at valid_time = 00Z
        ctx_tp = resolve_tile_read_context(
            session,
            model="gefs",
            variable="precipitation_amount_3h",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            valid_time="2026-09-07T00:00:00Z",
        )
        assert ctx_tp.lead_time_hours == 6
        assert ctx_tp.initial_time == "2026-09-06T18:00:00Z"
        assert ctx_tp.store_path == str(store_18z)

        # Context for temperature_2m at valid_time = 00Z
        ctx_t2m = resolve_tile_read_context(
            session,
            model="gefs",
            variable="temperature_2m",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            valid_time="2026-09-07T00:00:00Z",
        )
        assert ctx_t2m.lead_time_hours == 0
        assert ctx_t2m.initial_time == "2026-09-07T00:00:00Z"
        assert ctx_t2m.store_path == str(store_00z)


def test_gefs_lead0_no_runtime_member_reduction_when_mean_missing(tmp_path: Path):
    """Test Requirement D: When previous cycle has 30 members but NO geavg shard,
    do NOT compute runtime mean of members. Point forecast gracefully emits null.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/gefs_nomember_test.db")
    c_18z = datetime(2026, 9, 6, 18, 0, tzinfo=timezone.utc)
    c_00z = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)

    # 18Z has member shards (30 members) for lead 6, but NO shard.mean_L0006.shard!
    store_18z = tmp_path / "gefs_18z.zarr"
    mem_shards = {
        "precipitation_amount_3h": {(m, 6): float(m) for m in range(1, 31)},
        "temperature_2m": {(m, 6): 10.0 for m in range(1, 31)},
    }
    _create_sharded_gefs_store(
        store_18z,
        generation="gen_18z",
        leads=[6],
        mean_shards={},  # NO official mean shard!
        member_shards=mem_shards,
    )

    store_00z = tmp_path / "gefs_00z.zarr"
    _create_sharded_gefs_store(
        store_00z,
        generation="gen_00z",
        leads=[0],
        mean_shards={
            "temperature_2m": {0: 15.0},
        },
    )

    _init_gefs_db(engine, c_18z, c_00z, str(store_18z), str(store_00z))

    loc = ResolvedLocation(latitude=38.0, longitude=-107.0, elevation_m=None, resolved_via="coords")
    with Session(engine) as session:
        data = build_point_forecast(
            session,
            location=loc,
            model="gefs",
            variables=["temperature_2m", "precipitation_amount_3h"],
            units="metric",
            start_lead_time_hours=0,
            end_lead_time_hours=0,
        )

    assert len(data.forecasts) == 1
    f0 = data.forecasts[0]
    # Temperature stays on 00Z lead 0
    assert getattr(f0, "temperature_2m") == pytest.approx(55.0, abs=1e-2)
    # Precipitation is null because official geavg mean is missing (no member averaging)
    assert getattr(f0, "precipitation_amount_3h") is None


def test_gefs_lead0_missing_fallback_graceful_null(tmp_path: Path):
    """Test Requirement E: When only 00Z lead 0 exists (no previous cycle),
    temperature_2m is returned from 00Z lead 0, interval variables are null,
    and the series has complete entries.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/gefs_solo.db")
    c_00z = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)

    store_00z = tmp_path / "gefs_00z.zarr"
    _create_sharded_gefs_store(
        store_00z,
        generation="gen_00z",
        leads=[0],
        mean_shards={
            "temperature_2m": {0: 18.0},
        },
    )

    tables = [
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ForecastGrid.__table__,
        ForecastVariable.__table__,
        ModelRun.__table__,
        ForecastProduct.__table__,
        EnsembleMember.__table__,
        EnsembleMemberProduct.__table__,
        ForecastCycleLifecycle.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)

    with Session(engine) as session:
        session.add(ForecastCenter(id="noaa", center_id="noaa", name="NOAA", country="US"))
        session.add(Model(id="gefs", model_id="gefs", name="GEFS", center_id="noaa", is_ensemble=True, resolution_km=25.0))
        session.add(ModelVersion(id="v_gefs", model_id="gefs", version_string="v1.0"))
        session.add(ForecastGrid(id="g_glob", grid_code="global_025deg", name="Global", resolution_km=25.0))
        for v, u in [
            ("temperature_2m", "°C"),
            ("precipitation_amount_3h", "mm"),
            ("cloud_cover_3h", "%"),
        ]:
            session.add(ForecastVariable(id=f"v_{v}", variable_code=v, name=v, unit=u))
        session.add(ModelRun(id="r_00z", model_version_id="v_gefs", cycle_time=c_00z, status="ready", zarr_store_path=str(store_00z)))
        session.add(ForecastProduct(id="p_00_t2m", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=0))
        session.commit()

        loc = ResolvedLocation(latitude=38.0, longitude=-107.0, elevation_m=None, resolved_via="coords")
        data = build_point_forecast(
            session,
            location=loc,
            model="gefs",
            variables=["temperature_2m", "precipitation_amount_3h", "cloud_cover_3h"],
            units="metric",
            start_lead_time_hours=0,
            end_lead_time_hours=0,
        )

    assert len(data.forecasts) == 1
    f0 = data.forecasts[0]
    assert getattr(f0, "temperature_2m") == pytest.approx(58.0, abs=1e-2)
    assert getattr(f0, "precipitation_amount_3h") is None
    assert getattr(f0, "cloud_cover_3h") is None
    assert getattr(f0, "precipitation_type") == "none"


def test_gefs_81_leads_preserved_with_lead0_fallback(tmp_path: Path):
    """Test Requirement F: Complete 81-lead series is preserved when lead 0 has fallback."""
    engine = create_engine(f"sqlite:///{tmp_path}/gefs_81_test.db")
    c_18z = datetime(2026, 9, 6, 18, 0, tzinfo=timezone.utc)
    c_00z = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)

    # 18Z: lead 6 has precip
    store_18z = tmp_path / "gefs_18z.zarr"
    _create_sharded_gefs_store(
        store_18z,
        generation="gen_18z",
        leads=[6],
        mean_shards={
            "temperature_2m": {6: 10.0},
            "precipitation_amount_3h": {6: 7.5},
            "cloud_cover_3h": {6: 80.0},
            "crain": {6: 1.0},
            "csnow": {6: -100.0},
        },
    )

    # 00Z: has all 81 leads!
    store_00z = tmp_path / "gefs_00z.zarr"
    t_shards = {lead: 20.0 for lead in GEFS_81_LEADS}
    p_shards = {lead: 3.0 for lead in GEFS_81_LEADS if lead > 0}
    c_shards = {lead: 50.0 for lead in GEFS_81_LEADS if lead > 0}
    cr_shards = {lead: 1.0 for lead in GEFS_81_LEADS if lead > 0}
    _create_sharded_gefs_store(
        store_00z,
        generation="gen_00z",
        leads=GEFS_81_LEADS,
        mean_shards={
            "temperature_2m": t_shards,
            "precipitation_amount_3h": p_shards,
            "cloud_cover_3h": c_shards,
            "crain": cr_shards,
        },
    )

    tables = [
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ForecastGrid.__table__,
        ForecastVariable.__table__,
        ModelRun.__table__,
        ForecastProduct.__table__,
        EnsembleMember.__table__,
        EnsembleMemberProduct.__table__,
        ForecastCycleLifecycle.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)

    with Session(engine) as session:
        session.add(ForecastCenter(id="noaa", center_id="noaa", name="NOAA", country="US"))
        session.add(Model(id="gefs", model_id="gefs", name="GEFS", center_id="noaa", is_ensemble=True, resolution_km=25.0))
        session.add(ModelVersion(id="v_gefs", model_id="gefs", version_string="v1.0"))
        session.add(ForecastGrid(id="g_glob", grid_code="global_025deg", name="Global", resolution_km=25.0))
        for v, u in [
            ("temperature_2m", "°C"),
            ("precipitation_amount_3h", "mm"),
            ("cloud_cover_3h", "%"),
            ("crain", "flag"),
        ]:
            session.add(ForecastVariable(id=f"v_{v}", variable_code=v, name=v, unit=u))
        session.add(ModelRun(id="r_18z", model_version_id="v_gefs", cycle_time=c_18z, status="ready", zarr_store_path=str(store_18z)))
        session.add(ModelRun(id="r_00z", model_version_id="v_gefs", cycle_time=c_00z, status="ready", zarr_store_path=str(store_00z)))
        # Products for 18Z (lead 6)
        for v in ["temperature_2m", "precipitation_amount_3h", "cloud_cover_3h", "crain"]:
            session.add(ForecastProduct(id=f"p_18_{v}", run_id="r_18z", variable_id=v, grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=6))

        # Products for 00Z (all 81 leads)
        for lead in GEFS_81_LEADS:
            session.add(ForecastProduct(id=f"p_00_t2m_{lead}", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=lead))
            if lead > 0:
                for v in ["precipitation_amount_3h", "cloud_cover_3h", "crain"]:
                    session.add(ForecastProduct(id=f"p_00_{v}_{lead}", run_id="r_00z", variable_id=v, grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=lead))
        session.commit()

        loc = ResolvedLocation(latitude=38.0, longitude=-107.0, elevation_m=None, resolved_via="coords")
        data = build_point_forecast(
            session,
            location=loc,
            model="gefs",
            variables=["temperature_2m", "precipitation_amount_3h", "cloud_cover_3h", "crain"],
            units="metric",
            start_lead_time_hours=None,
            end_lead_time_hours=None,
        )

    assert len(data.forecasts) == 81
    leads_returned = [f.lead_time_hours for f in data.forecasts]
    assert leads_returned == GEFS_81_LEADS

    # Lead 0:
    f0 = data.forecasts[0]
    assert f0.lead_time_hours == 0
    # Temperature from 00Z lead 0 (20.0 + 40 = 60.0)
    assert getattr(f0, "temperature_2m") == pytest.approx(60.0, abs=1e-2)
    # Precip from 18Z lead 6 (7.5 + 40 = 47.5)
    assert getattr(f0, "precipitation_amount_3h") == pytest.approx(47.5, abs=1e-2)
    # Cloud from 18Z lead 6 (80.0 + 40 = 120.0)
    assert getattr(f0, "cloud_cover_3h") == pytest.approx(120.0, abs=1e-2)
    # Crain from 18Z lead 6 (1.0 + 40 = 41.0)
    assert getattr(f0, "crain") == pytest.approx(41.0, abs=1e-2)

    # Lead 6: comes from 00Z lead 6
    f6 = data.forecasts[2]
    assert f6.lead_time_hours == 6
    assert getattr(f6, "temperature_2m") == pytest.approx(60.0, abs=1e-2)
    assert getattr(f6, "precipitation_amount_3h") == pytest.approx(43.0, abs=1e-2)
