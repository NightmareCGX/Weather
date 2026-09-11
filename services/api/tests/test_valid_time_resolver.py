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


@pytest.fixture(autouse=True)
def _set_simulated_time(monkeypatch):
    monkeypatch.setenv("WEATHER_SIMULATED_NOW", "2026-09-01T00:00:00Z")
    yield


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
        var_tp = ForecastVariable(
            id="var_tp",
            variable_code="precipitation_amount_3h",
            name="3h Precipitation",
            unit="mm",
        )
        var_tcc = ForecastVariable(
            id="var_tcc",
            variable_code="cloud_cover_3h",
            name="3h Cloud Cover",
            unit="%",
        )
        session.add_all([center, gfs, gefs, v_gfs, v_gefs, grid, var_t2m, var_wind, var_tp, var_tcc])
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


def test_resolver_retired_at_does_not_exclude_canonical_representation(resolver_test_db):
    """Verify V3 architectural decoupling: legacy retired_at does not exclude an otherwise valid representation."""
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

        # In V3, retired_at is legacy V2 whole-cycle logical state and does NOT exclude the representation
        source = resolve_valid_time_source(session, "gfs", target_v, variable="temperature_2m")
        assert source.valid_time == target_v
        assert source.cycle_time == c_retired
        assert source.lead_time_hours == 6


def test_resolver_physical_deletion_fences_exclude_source(resolver_test_db):
    """Verify that physical safety fences (deletion_started_at or deleted_at) strictly exclude the source."""
    c_fenced = _dt(2026, 9, 1, 0)
    target_v = _dt(2026, 9, 1, 6)

    with Session(resolver_test_db) as session:
        r = ModelRun(
            id="run_gfs_fenced",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_fenced,
            status="ready",
            zarr_store_path="/stores/gfs/fenced",
            created_at=c_fenced,
        )
        p = ForecastProduct(
            id="prod_gfs_fenced_06",
            run_id="run_gfs_fenced",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=6,
        )
        lc = ForecastCycleLifecycle(
            model_id="gfs",
            cycle_time=c_fenced,
            deletion_started_at=_dt(2026, 9, 1, 12),
        )
        session.add_all([r, p, lc])
        session.commit()

        # deletion_started_at is an authoritative physical safety fence -> raises 404
        with pytest.raises(HTTPException) as exc:
            resolve_valid_time_source(session, "gfs", target_v, variable="temperature_2m")
        assert exc.value.status_code == 404


def test_resolver_instantaneous_variable_at_lead0_wins(resolver_test_db):
    """Instantaneous variables (temperature_2m) at lead 0 use the newest cycle."""
    c_prev = _dt(2026, 9, 4, 18)
    c_curr = _dt(2026, 9, 5, 0)
    target_v = _dt(2026, 9, 5, 0)

    with Session(resolver_test_db) as session:
        r_prev = ModelRun(
            id="run_gfs_prev_inst",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_prev,
            status="ready",
            zarr_store_path="/stores/gfs/18z",
            created_at=c_prev,
        )
        r_curr = ModelRun(
            id="run_gfs_curr_inst",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_curr,
            status="ready",
            zarr_store_path="/stores/gfs/00z",
            created_at=c_curr,
        )
        session.add_all([r_prev, r_curr])

        p_prev = ForecastProduct(
            id="prod_gfs_prev_t2m_06",
            run_id="run_gfs_prev_inst",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=6,
        )
        p_curr = ForecastProduct(
            id="prod_gfs_curr_t2m_00",
            run_id="run_gfs_curr_inst",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        session.add_all([p_prev, p_curr])
        session.commit()

        source = resolve_valid_time_source(session, "gfs", target_v, variable="temperature_2m")
        assert source.valid_time == target_v
        assert source.cycle_time == c_curr
        assert source.lead_time_hours == 0
        assert source.run_id == "run_gfs_curr_inst"


def test_resolver_precipitation_amount_3h_at_lead0_falls_back_to_previous_positive_lead(
    resolver_test_db,
):
    """precipitation_amount_3h at lead 0 falls back to the previous cycle's positive lead."""
    c_prev = _dt(2026, 9, 4, 18)
    c_curr = _dt(2026, 9, 5, 0)
    target_v = _dt(2026, 9, 5, 0)

    with Session(resolver_test_db) as session:
        r_prev = ModelRun(
            id="run_gfs_prev_tp",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_prev,
            status="ready",
            zarr_store_path="/stores/gfs/18z",
            created_at=c_prev,
        )
        r_curr = ModelRun(
            id="run_gfs_curr_tp",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_curr,
            status="ready",
            zarr_store_path="/stores/gfs/00z",
            created_at=c_curr,
        )
        session.add_all([r_prev, r_curr])

        p_prev = ForecastProduct(
            id="prod_gfs_prev_tp_06",
            run_id="run_gfs_prev_tp",
            variable_id="precipitation_amount_3h",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=6,
        )
        p_curr = ForecastProduct(
            id="prod_gfs_curr_tp_00",
            run_id="run_gfs_curr_tp",
            variable_id="precipitation_amount_3h",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        session.add_all([p_prev, p_curr])
        session.commit()

        source = resolve_valid_time_source(
            session, "gfs", target_v, variable="precipitation_amount_3h"
        )
        assert source.valid_time == target_v
        assert source.cycle_time == c_prev
        assert source.lead_time_hours == 6
        assert source.run_id == "run_gfs_prev_tp"


def test_resolver_cloud_cover_3h_at_lead0_falls_back_to_previous_positive_lead(
    resolver_test_db,
):
    """cloud_cover_3h at lead 0 falls back to the previous cycle's positive lead."""
    c_prev = _dt(2026, 9, 4, 18)
    c_curr = _dt(2026, 9, 5, 0)
    target_v = _dt(2026, 9, 5, 0)

    with Session(resolver_test_db) as session:
        r_prev = ModelRun(
            id="run_gfs_prev_tcc",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_prev,
            status="ready",
            zarr_store_path="/stores/gfs/18z",
            created_at=c_prev,
        )
        r_curr = ModelRun(
            id="run_gfs_curr_tcc",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_curr,
            status="ready",
            zarr_store_path="/stores/gfs/00z",
            created_at=c_curr,
        )
        session.add_all([r_prev, r_curr])

        p_prev = ForecastProduct(
            id="prod_gfs_prev_tcc_06",
            run_id="run_gfs_prev_tcc",
            variable_id="cloud_cover_3h",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=6,
        )
        p_curr = ForecastProduct(
            id="prod_gfs_curr_tcc_00",
            run_id="run_gfs_curr_tcc",
            variable_id="cloud_cover_3h",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        session.add_all([p_prev, p_curr])
        session.commit()

        source = resolve_valid_time_source(
            session, "gfs", target_v, variable="cloud_cover_3h"
        )
        assert source.valid_time == target_v
        assert source.cycle_time == c_prev
        assert source.lead_time_hours == 6
        assert source.run_id == "run_gfs_prev_tcc"


def test_resolver_interval_variable_at_positive_lead_stays_on_newest_cycle(
    resolver_test_db,
):
    """Interval variables at positive lead (e.g. +3) stay on the newest cycle without fallback."""
    c_prev = _dt(2026, 9, 4, 18)
    c_curr = _dt(2026, 9, 5, 0)
    target_v = _dt(2026, 9, 5, 3)

    with Session(resolver_test_db) as session:
        r_prev = ModelRun(
            id="run_gfs_prev_pos",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_prev,
            status="ready",
            zarr_store_path="/stores/gfs/18z",
            created_at=c_prev,
        )
        r_curr = ModelRun(
            id="run_gfs_curr_pos",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_curr,
            status="ready",
            zarr_store_path="/stores/gfs/00z",
            created_at=c_curr,
        )
        session.add_all([r_prev, r_curr])

        # 00Z + 3h = 03Z (newest cycle, positive lead)
        # 18Z + 9h = 03Z (older cycle)
        p_curr = ForecastProduct(
            id="prod_gfs_curr_tp_03",
            run_id="run_gfs_curr_pos",
            variable_id="precipitation_amount_3h",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=3,
        )
        p_prev = ForecastProduct(
            id="prod_gfs_prev_tp_09",
            run_id="run_gfs_prev_pos",
            variable_id="precipitation_amount_3h",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=9,
        )
        session.add_all([p_curr, p_prev])
        session.commit()

        source = resolve_valid_time_source(
            session, "gfs", target_v, variable="precipitation_amount_3h"
        )
        assert source.valid_time == target_v
        assert source.cycle_time == c_curr
        assert source.lead_time_hours == 3
        assert source.run_id == "run_gfs_curr_pos"


def test_resolver_interval_variable_at_lead0_missing_fallback_raises_404(
    resolver_test_db,
):
    """When lead 0 is the only candidate for an interval variable, raise 404 rather than returning lead 0."""
    c_curr = _dt(2026, 9, 5, 0)
    target_v = _dt(2026, 9, 5, 0)

    with Session(resolver_test_db) as session:
        r_curr = ModelRun(
            id="run_gfs_solo_00z",
            model_version_id="version_gfs_v1.0",
            cycle_time=c_curr,
            status="ready",
            zarr_store_path="/stores/gfs/00z",
            created_at=c_curr,
        )
        session.add(r_curr)

        p_t2m = ForecastProduct(
            id="prod_gfs_solo_t2m_00",
            run_id="run_gfs_solo_00z",
            variable_id="temperature_2m",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        p_tp = ForecastProduct(
            id="prod_gfs_solo_tp_00",
            run_id="run_gfs_solo_00z",
            variable_id="precipitation_amount_3h",
            grid_id="global_025deg",
            product_type="surface",
            lead_time_hours=0,
        )
        session.add_all([p_t2m, p_tp])
        session.commit()

        # Instantaneous succeeds
        source_inst = resolve_valid_time_source(
            session, "gfs", target_v, variable="temperature_2m"
        )
        assert source_inst.cycle_time == c_curr
        assert source_inst.lead_time_hours == 0

        # Interval variable has no positive-lead fallback -> must raise 404
        with pytest.raises(HTTPException) as exc:
            resolve_valid_time_source(
                session, "gfs", target_v, variable="precipitation_amount_3h"
            )
        assert exc.value.status_code == 404


def test_resolver_gefs_interval_fallback_respects_member_coverage(
    resolver_test_db,
):
    """Under-covered GEFS fallback (<85% members) is rejected, selecting older covered candidate."""
    from domain.coverage import get_expected_members, register_expected_members

    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    try:
        c_12z = _dt(2026, 9, 4, 12)
        c_18z = _dt(2026, 9, 4, 18)
        c_00z = _dt(2026, 9, 5, 0)
        target_v = _dt(2026, 9, 5, 0)

        with Session(resolver_test_db) as session:
            r_12z = ModelRun(
                id="run_gefs_12z_cov",
                model_version_id="version_gefs_v1.0",
                cycle_time=c_12z,
                status="ready",
                zarr_store_path="/stores/gefs/12z",
                created_at=c_12z,
            )
            r_18z = ModelRun(
                id="run_gefs_18z_cov",
                model_version_id="version_gefs_v1.0",
                cycle_time=c_18z,
                status="ready",
                zarr_store_path="/stores/gefs/18z",
                created_at=c_18z,
            )
            r_00z = ModelRun(
                id="run_gefs_00z_cov",
                model_version_id="version_gefs_v1.0",
                cycle_time=c_00z,
                status="ready",
                zarr_store_path="/stores/gefs/00z",
                created_at=c_00z,
            )
            session.add_all([r_12z, r_18z, r_00z])

            # 00Z has lead 0h with 30 members
            p_00z = ForecastProduct(
                id="prod_gefs_00z_tp_00",
                run_id="run_gefs_00z_cov",
                variable_id="precipitation_amount_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
            session.add(p_00z)
            for m in range(1, 31):
                session.add(
                    EnsembleMemberProduct(
                        id=f"emp_cov_00z_{m}",
                        run_id="run_gefs_00z_cov",
                        member_index=m,
                        lead_time_hours=0,
                    )
                )

            # 18Z has lead 6h with ONLY 10 members (<85% of 30)
            p_18z = ForecastProduct(
                id="prod_gefs_18z_tp_06",
                run_id="run_gefs_18z_cov",
                variable_id="precipitation_amount_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=6,
            )
            session.add(p_18z)
            for m in range(1, 11):
                session.add(
                    EnsembleMemberProduct(
                        id=f"emp_cov_18z_{m}",
                        run_id="run_gefs_18z_cov",
                        member_index=m,
                        lead_time_hours=6,
                    )
                )

            # 12Z has lead 12h with 30 members (>=85%)
            p_12z = ForecastProduct(
                id="prod_gefs_12z_tp_12",
                run_id="run_gefs_12z_cov",
                variable_id="precipitation_amount_3h",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=12,
            )
            session.add(p_12z)
            for m in range(1, 31):
                session.add(
                    EnsembleMemberProduct(
                        id=f"emp_cov_12z_{m}",
                        run_id="run_gefs_12z_cov",
                        member_index=m,
                        lead_time_hours=12,
                    )
                )
            session.commit()

            # 18Z + 6h is under-covered -> must skip and select 12Z + 12h!
            source = resolve_valid_time_source(
                session, "gefs", target_v, variable="precipitation_amount_3h"
            )
            assert source.cycle_time == c_12z
            assert source.lead_time_hours == 12
            assert source.run_id == "run_gefs_12z_cov"
    finally:
        register_expected_members("gefs", old_expected)
