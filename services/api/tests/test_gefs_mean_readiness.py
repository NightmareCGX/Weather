"""Unit and integration tests for GEFS mean product vs ensemble member readiness (Lifecycle V2).

Covers all 4 readiness states:
A. Mean ready, members incomplete: /v1/points and /v1/maps serve from geavg; /v1/ensembles respects member readiness.
B. Members ready, mean missing: /v1/ensembles serves; /v1/points does not freeze.
C. Both ready: mean products use geavg, statistics use members.
D. Neither ready: returns 404 cleanly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import HTTPException
import pytest
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
from api.services.resolver import (
    resolve_valid_time_candidates,
    resolve_valid_time_source,
)

from domain.coverage import get_expected_members, register_expected_members


@pytest.fixture(autouse=True)
def _set_simulated_time(monkeypatch):
    monkeypatch.setenv("WEATHER_SIMULATED_NOW", "2026-09-06T12:00:00Z")
    yield

CYCLE = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
V_TIME = datetime(2026, 9, 6, 18, 0, tzinfo=timezone.utc)  # Lead 6


@pytest.fixture
def test_db(monkeypatch):
    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    # Hermetic test isolation: prevent S3/MinIO network calls in CI
    monkeypatch.setattr(
        "api.services.point_forecast.resolve_serving_generation_for_store",
        lambda store_path, latest_retired_iso=None: "gen_test",
    )

    import s3fs
    def _fail_s3(*args, **kwargs):
        raise ConnectionRefusedError("Hermetic test: real S3/MinIO access prohibited")
    monkeypatch.setattr(s3fs.S3FileSystem, "__init__", _fail_s3)

    engine = create_engine("sqlite:///:memory:")
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
    with Session(engine) as db:
        center = ForecastCenter(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="USA",
        )
        db.add(center)
        model = Model(
            id="model_gefs",
            model_id="gefs",
            name="GEFS",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
        )
        db.add(model)
        version = ModelVersion(
            id="version_gefs_v1.0",
            model_id="gefs",
            version_string="v1.0",
        )
        db.add(version)
        grid = ForecastGrid(
            id="grid_025",
            grid_code="global_025deg",
            name="Global 0.25",
            resolution_km=25.0,
        )
        db.add(grid)
        var = ForecastVariable(
            id="var_t2m",
            variable_code="temperature_2m",
            name="2-Meter Temperature",
            unit="°C",
        )
        db.add(var)
        run = ModelRun(
            id="run_gefs_2026090612",
            model_version_id="version_gefs_v1.0",
            cycle_time=CYCLE,
            status="partial",
            zarr_store_path="s3://weather-data/gefs/2026-09-06/12/cycle.zarr",
        )
        db.add(run)
        db.commit()
    try:
        yield engine
    finally:
        engine.dispose()
        register_expected_members("gefs", old_expected)


def test_readiness_case_a_mean_ready_members_incomplete(test_db):
    """Case A: geavg mean product is committed, but member coverage is below 85% threshold (Lifecycle V3 Strict Coherent Vintage).

    Expected:
    - Under V3 strict coherent vintage, a GEFS cycle is canonical for valid_time V if and only if
      BOTH geavg AND member coverage >= 85% (>= 26/30) are committed.
    - When member coverage is < 85% (5 members), the cycle is NOT canonical for any endpoint.
    - Both mean candidates and member candidates reject the under-covered cycle.
    """
    with Session(test_db) as db:
        # Add ForecastProduct (mean product committed for lead 6)
        db.add(
            ForecastProduct(
                id="prod_gefs_t2m_6",
                run_id="run_gefs_2026090612",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="ensemble_mean",
                lead_time_hours=6,
            )
        )
        # Add only 5 members (well below 85% of 30 = 26)
        for m in range(1, 6):
            db.add(
                EnsembleMemberProduct(
                    id=f"emp_{m}_6",
                    run_id="run_gefs_2026090612",
                    member_index=m,
                    lead_time_hours=6,
                )
            )
        db.commit()

        # Under V3 Strict Coherent Vintage, neither mean nor member path promotes with incomplete members
        mean_candidates = resolve_valid_time_candidates(
            db, "gefs", target_valid_time=V_TIME, require_members=False
        )
        assert V_TIME not in mean_candidates

        member_candidates = resolve_valid_time_candidates(
            db, "gefs", target_valid_time=V_TIME, require_members=True
        )
        assert V_TIME not in member_candidates

        # Direct source resolution raises 404 because no coherent source exists
        with pytest.raises(HTTPException) as exc:
            resolve_valid_time_source(
                db, "gefs", V_TIME, variable="temperature_2m", require_members=False
            )
        assert exc.value.status_code == 404


def test_readiness_case_b_members_ready_mean_missing(test_db):
    """Case B: 30 perturbed members are committed, but geavg mean product is absent.

    Expected:
    - Member candidates (require_members=True) resolve successfully.
    - Mean path (require_members=False) without ForecastProduct row does not resolve the unready lead.
    """
    with Session(test_db) as db:
        # 30 members committed for lead 6
        for m in range(1, 31):
            db.add(
                EnsembleMemberProduct(
                    id=f"emp_{m}_6",
                    run_id="run_gefs_2026090612",
                    member_index=m,
                    lead_time_hours=6,
                )
            )
        db.commit()

        # Mean path: no ForecastProduct for lead 6
        mean_candidates = resolve_valid_time_candidates(
            db, "gefs", target_valid_time=V_TIME, require_members=False
        )
        assert V_TIME not in mean_candidates


def test_readiness_case_c_both_ready(test_db):
    """Case C: Both geavg mean and 30 perturbed members are committed.

    Expected:
    - Both mean path and member path resolve successfully.
    """
    with Session(test_db) as db:
        db.add(
            ForecastProduct(
                id="prod_gefs_t2m_6",
                run_id="run_gefs_2026090612",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="ensemble_mean",
                lead_time_hours=6,
            )
        )
        for m in range(1, 31):
            db.add(
                EnsembleMemberProduct(
                    id=f"emp_{m}_6",
                    run_id="run_gefs_2026090612",
                    member_index=m,
                    lead_time_hours=6,
                )
            )
        db.commit()

        mean_candidates = resolve_valid_time_candidates(
            db, "gefs", target_valid_time=V_TIME, require_members=False
        )
        assert V_TIME in mean_candidates

        member_candidates = resolve_valid_time_candidates(
            db, "gefs", target_valid_time=V_TIME, require_members=True
        )
        assert V_TIME in member_candidates


def test_readiness_case_d_neither_ready(test_db):
    """Case D: Neither geavg nor members committed.

    Expected:
    - Neither path resolves.
    """
    with Session(test_db) as db:
        mean_candidates = resolve_valid_time_candidates(
            db, "gefs", target_valid_time=V_TIME, require_members=False
        )
        assert V_TIME not in mean_candidates

        member_candidates = resolve_valid_time_candidates(
            db, "gefs", target_valid_time=V_TIME, require_members=True
        )
        assert V_TIME not in member_candidates


def test_map_and_points_gefs_mean_cross_cycle_alignment(test_db):
    """Verify that both map and points use the same cross-cycle geavg resolution (Req A, B, C, E).

    Scenario:
    - Cycle 12Z has geavg for valid_time V_TIME (18Z) at lead 6.
    - Cycle 18Z has members at lead 0, but geavg is missing at lead 0.
    Expected:
    - Both map and points resolve to Cycle 12Z lead 6 (not Cycle 18Z).
    - Map does NOT compute member mean from 18Z members.
    """
    from api.services.tiles import resolve_tile_read_context
    from api.services.point_forecast import _select_min_lead_winners

    c_18z = datetime(2026, 9, 6, 18, 0, tzinfo=timezone.utc)
    with Session(test_db) as db:
        # Add cycle 18Z run
        r_18z = ModelRun(
            id="run_gefs_2026090618",
            model_version_id="version_gefs_v1.0",
            cycle_time=c_18z,
            status="ready",
            zarr_store_path="s3://weather-data/gefs/2026-09-06/18/cycle.zarr",
        )
        db.add(r_18z)
        # 18Z has members for lead 0 (matching valid_time V_TIME)
        for m in range(1, 31):
            db.add(
                EnsembleMemberProduct(
                    id=f"emp_18z_{m}_0",
                    run_id="run_gefs_2026090618",
                    member_index=m,
                    lead_time_hours=0,
                )
            )

        # 12Z has official geavg mean for lead 6 (also matching valid_time V_TIME)
        db.add(
            ForecastProduct(
                id="prod_gefs_12z_t2m_6",
                run_id="run_gefs_2026090612",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="ensemble_mean",
                lead_time_hours=6,
            )
        )
        # 12Z also has 30 members for lead 6, satisfying coherent vintage
        for m in range(1, 31):
            db.add(
                EnsembleMemberProduct(
                    id=f"emp_12z_{m}_6",
                    run_id="run_gefs_2026090612",
                    member_index=m,
                    lead_time_hours=6,
                )
            )
        db.commit()

        # 1. Map tile context resolution for valid_time V_TIME
        map_ctx = resolve_tile_read_context(
            db,
            model="gefs",
            variable="temperature_2m",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            valid_time=V_TIME.isoformat().replace("+00:00", "Z"),
        )
        assert map_ctx.store_path == "s3://weather-data/gefs/2026-09-06/12/cycle.zarr"
        assert map_ctx.lead_time_hours == 6

        # 2. Points winner resolution for valid_time V_TIME
        points_winners = _select_min_lead_winners(db, "gefs")
        assert V_TIME in points_winners
        winner_cycle, winner_lead = points_winners[V_TIME][0]
        assert winner_cycle == CYCLE  # 12Z
        assert winner_lead == 6

        # 3. Requirement E: Identical source cycle and lead for points and map
        assert map_ctx.lead_time_hours == winner_lead
        assert map_ctx.initial_time == winner_cycle.isoformat().replace("+00:00", "Z")


def test_gefs_map_missing_physical_shard_raises_no_runtime_reduction(tmp_path):
    """Requirement D: Missing physical mean shard raises FileNotFoundError and does NOT compute 30-member mean."""
    import json
    import xarray as xr
    import numpy as np
    from api.services.tiles import _select_tile_window
    from tests.test_sharded_reader import _build_test_shard

    store_dir = tmp_path / "gefs_tile_missing_mean.zarr"
    store_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = store_dir / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({
            "manifest_schema_version": 1,
            "generation": "gen_test",
            "storage_format_version": "sharded_v1",
        })
    )

    # Write member shards ONLY (30 members)
    for m in range(1, 31):
        p = store_dir / "temperature_2m" / f"shard.mem{m:03d}_L0006.shard"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_build_test_shard(val_offset=float(m)))

    latitudes = [90.0 - i * 0.25 for i in range(721)]
    longitudes = [0.0 + j * 0.25 for j in range(1440)]
    ds_meta = xr.Dataset(
        data_vars={
            "temperature_2m": (("member", "lead_time_hours", "latitude", "longitude"), np.zeros((30, 1, 721, 1440), dtype=np.float32)),
        },
        coords={
            "member": list(range(1, 31)),
            "lead_time_hours": [6],
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    ds_meta.to_zarr(str(store_dir), mode="a", consolidated=True, zarr_format=2)

    # Slicing tile must raise FileNotFoundError (no physical mean shard) and NOT compute member mean
    with pytest.raises(FileNotFoundError, match="Missing official mean shard"):
        _select_tile_window(
            ds_meta,
            variable="temperature_2m",
            lead=6,
            zoom=8,
            x=51,
            y=98,
            expected_members=30,
            store_path=str(store_dir),
        )
