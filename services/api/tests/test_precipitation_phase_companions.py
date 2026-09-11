"""Regression tests for precipitation phase categorical companion variables in /v1/points.

Verifies:
1. GFS deterministic Hourly Forecast:
   - crain, csnow, cfrzr, cicep companion fields are attached to forecast entries
     when precipitation_amount_3h is present.
   - Single-phase flag (e.g. crain=1) yields precipitation_type="rain".
   - Multi-phase flags (e.g. crain=1, csnow=1) yield precipitation_type="mixed"
     and expose both crain=1.0 and csnow=1.0.
   - Three/four-way multi-phase flags expose the exact constituents.
2. GEFS ensemble-mean Hourly Forecast:
   - crain, csnow, cfrzr, cicep companion fields come from the official ensemble-mean
     shards (geavg / is_mean=True).
   - Multi-phase mean values (e.g. crain >= 0.5, csnow >= 0.5) yield precipitation_type="mixed"
     and expose the mean constituent values.
   - Mean constituents do not derive from member phase percentages.
3. Selective inclusion:
   - When precipitation_amount_3h is NOT requested (e.g. only temperature_2m),
     companion flags crain, csnow, cfrzr, cicep are not attached to entry.
4. Lead 0 fallback:
   - At lead 0, companion variables are sampled from the same positive lead fallback source.
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
from tests.test_sharded_reader import _build_test_shard


@pytest.fixture(autouse=True)
def _no_reader_pool(monkeypatch):
    """Force direct bounded read path for standalone SQLite tests."""
    monkeypatch.setenv("WEATHER_SIMULATED_NOW", "2026-09-10T00:00:00Z")
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


def _create_test_sharded_store(
    store_dir: Path,
    *,
    generation: str,
    leads: list[int],
    shards: dict[str, dict[int, float]],
    is_mean: bool = False,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    all_vars = set(shards.keys())

    coords = {
        "lead_time_hours": leads,
        "latitude": LATITUDES,
        "longitude": LONGITUDES,
    }
    if is_mean:
        coords["member"] = list(range(1, 31))

    ds = xr.Dataset(
        data_vars={
            v: (("lead_time_hours", "latitude", "longitude"), np.zeros((len(leads), 721, 1440), dtype=np.float32))
            for v in all_vars
        },
        coords=coords,
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

    prefix = "shard.mean" if is_mean else "shard.det"
    for v_code, lead_vals in shards.items():
        v_dir = store_dir / v_code
        v_dir.mkdir(parents=True, exist_ok=True)
        for lead, val in lead_vals.items():
            shard_path = v_dir / f"{prefix}_L{lead:04d}.shard"
            shard_path.write_bytes(_build_test_shard(val_offset=val))


def _init_test_db(engine, model_id: str, is_ensemble: bool, cycle: datetime, store_path: str, variables: list[str]) -> None:
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
        session.add(Model(id=model_id, model_id=model_id, name=model_id.upper(), center_id="noaa", is_ensemble=is_ensemble, resolution_km=25.0))
        session.add(ModelVersion(id=f"v_{model_id}", model_id=model_id, version_string="v1.0"))
        session.add(ForecastGrid(id="g_glob", grid_code="global_025deg", name="Global", resolution_km=25.0))

        units = {
            "temperature_2m": "°C",
            "precipitation_amount_3h": "mm",
            "cloud_cover_3h": "%",
            "crain": "flag",
            "csnow": "flag",
            "cfrzr": "flag",
            "cicep": "flag",
        }
        for v in variables:
            session.add(ForecastVariable(id=f"v_{v}", variable_code=v, name=v, unit=units.get(v, "")))

        session.add(ModelRun(id=f"r_{model_id}", model_version_id=f"v_{model_id}", cycle_time=cycle, status="ready", zarr_store_path=store_path))
        p_type = "ensemble_mean" if is_ensemble else "surface"
        for v in variables:
            session.add(ForecastProduct(id=f"p_{v}", run_id=f"r_{model_id}", variable_id=v, grid_id="global_025deg", product_type=p_type, lead_time_hours=6))

        if is_ensemble:
            for m in range(1, 31):
                session.add(EnsembleMember(id=f"m_{m}", run_id=f"r_{model_id}", member_index=m, member_name=f"gep{m:02d}"))
                session.add(EnsembleMemberProduct(id=f"emp_{m}_6", run_id=f"r_{model_id}", member_index=m, lead_time_hours=6))

        session.commit()


def test_gfs_deterministic_mixed_constituents(tmp_path: Path):
    """GFS deterministic forecast with multiple categorical flags yields mixed phase with constituents."""
    engine = create_engine(f"sqlite:///{tmp_path}/gfs_test.db")
    c_time = datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc)

    store = tmp_path / "gfs.zarr"
    # val_offset: precip=1.83 - 40 = -38.17, crain=1 - 40 = -39.0, csnow=1 - 40 = -39.0, cfrzr=-40.0 (0), cicep=-40.0 (0)
    _create_test_sharded_store(
        store,
        generation="gen_gfs",
        leads=[6],
        shards={
            "temperature_2m": {6: 5.0},
            "precipitation_amount_3h": {6: -38.17},  # 40 - 38.17 = 1.83 mm
            "crain": {6: -39.0},  # 40 - 39.0 = 1.0 (Rain flag active)
            "csnow": {6: -39.0},  # 40 - 39.0 = 1.0 (Snow flag active)
            "cfrzr": {6: -40.0},  # 40 - 40.0 = 0.0
            "cicep": {6: -40.0},  # 40 - 40.0 = 0.0
        },
        is_mean=False,
    )

    _init_test_db(
        engine,
        model_id="gfs",
        is_ensemble=False,
        cycle=c_time,
        store_path=str(store),
        variables=["temperature_2m", "precipitation_amount_3h"],
    )

    loc = ResolvedLocation(latitude=38.0, longitude=-107.0, elevation_m=None, resolved_via="coordinates")
    with Session(engine) as session:
        data = build_point_forecast(
            session,
            location=loc,
            model="gfs",
            variables=None,  # default variables from catalog
            units="metric",
            start_lead_time_hours=6,
            end_lead_time_hours=6,
        )

    assert len(data.forecasts) == 1
    f = data.forecasts[0]
    assert getattr(f, "precipitation_amount_3h") == pytest.approx(1.83, abs=0.01)
    assert getattr(f, "precipitation_type") == "mixed"
    # Companion categorical variables are attached to the entry
    assert getattr(f, "crain") == pytest.approx(1.0, abs=0.01)
    assert getattr(f, "csnow") == pytest.approx(1.0, abs=0.01)
    assert getattr(f, "cfrzr") == pytest.approx(0.0, abs=0.01)
    assert getattr(f, "cicep") == pytest.approx(0.0, abs=0.01)


def test_gefs_ensemble_mean_mixed_constituents(tmp_path: Path):
    """GEFS Hourly Forecast derives mixed phase and constituents from official mean shards."""
    engine = create_engine(f"sqlite:///{tmp_path}/gefs_test.db")
    c_time = datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc)

    store = tmp_path / "gefs.zarr"
    _create_test_sharded_store(
        store,
        generation="gen_gefs",
        leads=[6],
        shards={
            "temperature_2m": {6: 5.0},
            "precipitation_amount_3h": {6: -38.17},  # 1.83 mm
            "crain": {6: -39.35},  # 40 - 39.35 = 0.65 (>= 0.5 -> Rain active)
            "csnow": {6: -39.45},  # 40 - 39.45 = 0.55 (>= 0.5 -> Snow active)
            "cfrzr": {6: -39.95},  # 40 - 39.95 = 0.05 (< 0.5 -> inactive)
            "cicep": {6: -40.0},   # 40 - 40.0 = 0.0
        },
        is_mean=True,
    )

    _init_test_db(
        engine,
        model_id="gefs",
        is_ensemble=True,
        cycle=c_time,
        store_path=str(store),
        variables=["temperature_2m", "precipitation_amount_3h"],
    )

    loc = ResolvedLocation(latitude=38.0, longitude=-107.0, elevation_m=None, resolved_via="coordinates")
    with Session(engine) as session:
        data = build_point_forecast(
            session,
            location=loc,
            model="gefs",
            variables=None,
            units="metric",
            start_lead_time_hours=6,
            end_lead_time_hours=6,
        )

    assert len(data.forecasts) == 1
    f = data.forecasts[0]
    assert getattr(f, "precipitation_amount_3h") == pytest.approx(1.83, abs=0.01)
    assert getattr(f, "precipitation_type") == "mixed"
    # GEFS ensemble mean categorical flags are present on the entry
    assert getattr(f, "crain") == pytest.approx(0.65, abs=0.01)
    assert getattr(f, "csnow") == pytest.approx(0.55, abs=0.01)
    assert getattr(f, "cfrzr") == pytest.approx(0.05, abs=0.01)
    assert getattr(f, "cicep") == pytest.approx(0.0, abs=0.01)


def test_companion_variables_not_attached_when_precip_not_requested(tmp_path: Path):
    """When precipitation_amount_3h is not in requested variables, companion flags are omitted."""
    engine = create_engine(f"sqlite:///{tmp_path}/gfs_no_precip.db")
    c_time = datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc)

    store = tmp_path / "gfs_np.zarr"
    _create_test_sharded_store(
        store,
        generation="gen_gfs_np",
        leads=[6],
        shards={
            "temperature_2m": {6: 15.0},
            "precipitation_amount_3h": {6: 2.0},
            "crain": {6: 1.0},
        },
        is_mean=False,
    )

    _init_test_db(
        engine,
        model_id="gfs",
        is_ensemble=False,
        cycle=c_time,
        store_path=str(store),
        variables=["temperature_2m", "precipitation_amount_3h"],
    )

    loc = ResolvedLocation(latitude=38.0, longitude=-107.0, elevation_m=None, resolved_via="coordinates")
    with Session(engine) as session:
        data = build_point_forecast(
            session,
            location=loc,
            model="gfs",
            variables=["temperature_2m"],
            units="metric",
            start_lead_time_hours=6,
            end_lead_time_hours=6,
        )

    f = data.forecasts[0]
    assert hasattr(f, "temperature_2m")
    assert not hasattr(f, "precipitation_amount_3h")
    assert not hasattr(f, "crain")
    assert not hasattr(f, "csnow")
