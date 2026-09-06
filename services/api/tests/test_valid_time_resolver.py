"""Integration tests for the shared ValidTimeResolver and valid-time serving endpoints (Lifecycle V2)."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.core.database import Base
from api.models.entities import (
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
from api.services.resolver import resolve_valid_time_source


def _dt(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def resolver_test_db(tmp_path):
    """Create an isolated test database with schema and seed metadata for resolver testing."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_resolver.db",
        connect_args={"check_same_thread": False},
    )
    tables_to_create = [
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ModelRun.__table__,
        EnsembleMember.__table__,
        EnsembleMemberProduct.__table__,
        ForecastVariable.__table__,
        ForecastGrid.__table__,
        ForecastProduct.__table__,
        ForecastCycleLifecycle.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables_to_create)

    with Session(engine) as session:
        center = ForecastCenter(
            id="center_noaa",
            center_id="noaa",
            name="NOAA",
            country="US",
            created_at=_dt(2026, 1, 1, 0),
        )
        gfs = Model(
            id="model_gfs",
            model_id="gfs",
            name="GFS",
            center_id="noaa",
            is_ensemble=False,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        gefs = Model(
            id="model_gefs",
            model_id="gefs",
            name="GEFS",
            center_id="noaa",
            is_ensemble=True,
            resolution_km=25.0,
            created_at=_dt(2026, 1, 1, 0),
        )
        v_gfs = ModelVersion(
            id="version_gfs_v1.0",
            model_id="gfs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        v_gefs = ModelVersion(
            id="version_gefs_v1.0",
            model_id="gefs",
            version_string="v1.0",
            created_at=_dt(2026, 1, 1, 0),
        )
        grid = ForecastGrid(
            id="grid_global",
            grid_code="global_025deg",
            name="Global 0.25",
            resolution_km=25.0,
        )
        var_t2m = ForecastVariable(
            id="var_t2m",
            variable_code="temperature_2m",
            name="2m Temperature",
            unit="°C",
        )
        var_wind = ForecastVariable(
            id="var_wind",
            variable_code="wind_10m",
            name="10m Wind",
            unit="km/h",
        )
        session.add_all([center, gfs, gefs, v_gfs, v_gefs, grid, var_t2m, var_wind])
        session.commit()

    yield engine
    engine.dispose()


def test_resolver_newest_committed_cycle_wins(resolver_test_db):
    """Verify that when multiple cycles cover the same valid time, the newest committed cycle wins."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    # Valid time = 2026-09-02 12:00:00Z:
    # Covered by:
    # 00Z + 12h = 12Z
    # 06Z + 6h  = 12Z
    # 06Z is newer -> 06Z + 6h MUST WIN!
    with Session(resolver_test_db) as session:
        r_00z = ModelRun(
            id="run_gfs_00z",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_00z,
            status="ready",
            zarr_store_path="/stores/gfs/00z",
            created_at=c_00z,
        )
        r_06z = ModelRun(
            id="run_gfs_06z",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_06z,
            status="ready",
            zarr_store_path="/stores/gfs/06z",
            created_at=c_06z,
        )
        session.add_all([r_00z, r_06z])

        p_00z_12 = ForecastProduct(
            id="prod_00z_12",
            run_id="run_gfs_00z",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=12,
        )
        p_06z_06 = ForecastProduct(
            id="prod_06z_06",
            run_id="run_gfs_06z",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=6,
        )
        session.add_all([p_00z_12, p_06z_06])
        session.commit()

        target_v = _dt(2026, 9, 2, 12)
        source = resolve_valid_time_source(session, "gfs", target_v, variable="temperature_2m")

        assert source.valid_time == target_v
        assert source.cycle_time == c_06z
        assert source.lead_time_hours == 6
        assert source.store_path == "/stores/gfs/06z"
        assert source.run_id == "run_gfs_06z"


def test_resolver_fallback_to_older_cycle_when_newer_lead_uncommitted(resolver_test_db):
    """Verify that when 06Z has not yet committed a lead for valid_time V,
    the resolver falls back to the newest older cycle that covers V."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    # Target valid time: 2026-09-02 18:00:00Z
    # 00Z covers with lead 18h
    # 06Z has only lead 0h and 6h committed (lead 12h is not yet committed in 06Z)
    with Session(resolver_test_db) as session:
        r_00z = ModelRun(
            id="run_gfs_fallback_00z",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_00z,
            status="ready",
            zarr_store_path="/stores/gfs/00z",
            created_at=c_00z,
        )
        r_06z = ModelRun(
            id="run_gfs_fallback_06z",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_06z,
            status="partial",
            zarr_store_path="/stores/gfs/06z",
            created_at=c_06z,
        )
        session.add_all([r_00z, r_06z])

        p_00z_18 = ForecastProduct(
            id="prod_00z_18",
            run_id="run_gfs_fallback_00z",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=18,
        )
        p_06z_06 = ForecastProduct(
            id="prod_06z_06_only",
            run_id="run_gfs_fallback_06z",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=6,
        )
        session.add_all([p_00z_18, p_06z_06])
        session.commit()

        target_v = _dt(2026, 9, 2, 18)
        source = resolve_valid_time_source(session, "gfs", target_v, variable="temperature_2m")

        # Must fall back to 00Z + 18h!
        assert source.valid_time == target_v
        assert source.cycle_time == c_00z
        assert source.lead_time_hours == 18
        assert source.store_path == "/stores/gfs/00z"


def test_resolver_gefs_member_coverage_threshold(resolver_test_db):
    """Verify Principle 10: GEFS serveability rule is respected.
    If 06Z +6h has only 10/30 members (< 85%), 00Z +12h with 30/30 members must win.
    Once 06Z reaches 26/30 members (>= 85%), 06Z promotes!"""
    from domain.coverage import get_expected_members, register_expected_members

    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    try:
        c_00z = _dt(2026, 9, 2, 0)
        c_06z = _dt(2026, 9, 2, 6)
        target_v = _dt(2026, 9, 2, 12)

        with Session(resolver_test_db) as session:
            r_00z = ModelRun(
                id="run_gefs_00z",
                model_version_id="version_gefs_v1.0",
                cycle_time=c_00z,
                status="ready",
                zarr_store_path="/stores/gefs/00z",
                created_at=c_00z,
            )
            r_06z = ModelRun(
                id="run_gefs_06z",
                model_version_id="version_gefs_v1.0",
                cycle_time=c_06z,
                status="partial",
                zarr_store_path="/stores/gefs/06z",
                created_at=c_06z,
            )
            session.add_all([r_00z, r_06z])

            # 00Z has lead 12h with full 30 members
            p_00z = ForecastProduct(
                id="prod_gefs_00z_12",
                run_id="run_gefs_00z",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=12,
            )
            session.add(p_00z)
            for m in range(1, 31):
                session.add(
                    EnsembleMemberProduct(
                        id=f"emp_00z_12_{m}",
                        run_id="run_gefs_00z",
                        member_index=m,
                        lead_time_hours=12,
                    )
                )

            # 06Z has lead 6h with only 10 members (< 85% of 30 = 25.5)
            p_06z = ForecastProduct(
                id="prod_gefs_06z_06",
                run_id="run_gefs_06z",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=6,
            )
            session.add(p_06z)
            for m in range(1, 11):
                session.add(
                    EnsembleMemberProduct(
                        id=f"emp_06z_06_{m}",
                        run_id="run_gefs_06z",
                        member_index=m,
                        lead_time_hours=6,
                    )
                )
            session.commit()

            # Step 1: 06Z is below threshold -> 00Z MUST WIN
            source1 = resolve_valid_time_source(session, "gefs", target_v, variable="temperature_2m")
            assert source1.cycle_time == c_00z
            assert source1.lead_time_hours == 12

            # Step 2: Ingest 16 more members for 06Z lead 6h (total 26/30 = 86.7% >= 85%)
            for m in range(11, 27):
                session.add(
                    EnsembleMemberProduct(
                        id=f"emp_06z_06_{m}",
                        run_id="run_gefs_06z",
                        member_index=m,
                        lead_time_hours=6,
                    )
                )
            session.commit()

            # Step 3: Now 06Z reaches serveability threshold -> 06Z PROMOTES!
            source2 = resolve_valid_time_source(session, "gefs", target_v, variable="temperature_2m")
            assert source2.cycle_time == c_06z
            assert source2.lead_time_hours == 6
    finally:
        register_expected_members("gefs", old_expected)


def test_resolver_excludes_retired_and_deleted_cycles(resolver_test_db):
    """Verify that cycles marked retired or deleted in forecast_cycle_lifecycle are excluded."""
    c_retired = _dt(2026, 9, 1, 0)
    target_v = _dt(2026, 9, 1, 6)

    with Session(resolver_test_db) as session:
        r = ModelRun(
            id="run_gfs_retired",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_retired,
            status="ready",
            zarr_store_path="/stores/gfs/retired",
            created_at=c_retired,
        )
        p = ForecastProduct(
            id="prod_gfs_retired_06",
            run_id="run_gfs_retired",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=6,
        )
        lc = ForecastCycleLifecycle(
            model_id="gfs",
            cycle_time=c_retired,
            retired_at=_dt(2026, 9, 1, 12),
        )
        session.add_all([r, p, lc])
        session.commit()

        # Attempting to resolve target_v must raise 404 because c_retired is retired
        with pytest.raises(HTTPException) as exc:
            resolve_valid_time_source(session, "gfs", target_v, variable="temperature_2m")
        assert exc.value.status_code == 404
