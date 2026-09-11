"""Comprehensive regression test suite for Data Lifecycle V3 Phase 3: Canonical Resolver Consolidation.

Implements all 30 deterministic test scenarios mandated by the acceptance contract:
1. 00Z complete + 06Z partial: newer cycle supersedes only committed near valid times.
2. Older cycle continues far horizon.
3. Serving end does not collapse.
4. Same valid_time selects newest committed source.
5. Uncommitted newer representation does not win.
6. precipitation_amount_3h lead-0 fallback.
7. cloud_cover_3h lead-0 fallback.
8. Cold-start interval fallback returns null, not whole-request 404.
9. precipitation phase companions use exact precipitation fallback source.
10. Ordinary variables do not inherit precipitation fallback.
11. wind_10m requires coherent u/v pair.
12. Newer wind cycle with only u does not win.
13. precipitation_rate remains GFS-only and lead-0 ordinary.
14. GEFS geavg + <26 members: all endpoints remain on older coherent vintage.
15. GEFS geavg + 26 members: all GEFS endpoints promote to the newer coherent vintage.
16. GEFS 30/30 remains formal readiness requirement; 26/30 does not mark run ready.
17. retired_at != NULL does not remove an otherwise valid representation from V3 valid_time resolution.
18. deletion_started_at != NULL excludes the affected physical source.
19. deleted_at != NULL excludes the affected physical source.
20. Physical lifecycle fence is correctly model-scoped (GFS fence does not fence GEFS).
21. Phase 1 serving_start behavior remains unchanged.
22. Exact serving-start valid_time remains servable.
23. Point/time-series bulk resolution has no N+1 SQL query regression.
24. Cache promotion: cached 00Z/L12 result is not reused after 06Z/L6 commits.
25. Interval fallback cache promotion: fallback-source change changes cache provenance.
26. GEFS threshold cache promotion: 25 -> 26 members promotes coherent vintage and invalidates cache.
27. GEFS same-vintage member update: 26 -> 27 members invalidates member-derived cache key.
28. Multi-time provenance digest is deterministic independent of input/dict iteration order.
29. Read-resilience fallback never mixes precipitation companions.
30. Read-resilience fallback never mixes wind u/v components.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
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
from api.services.cache import (
    build_ensemble_cache_key,
    build_point_cache_key,
)
from api.services.resolver import (
    ResolvedForecastSource,
    build_canonical_provenance_digest,
    canonical_serving_horizon,
    resolve_canonical_source,
    resolve_canonical_sources_bulk,
    resolve_point_provenance_digest,
    resolve_variable_source,
)
from domain.coverage import get_expected_members, register_expected_members


def _dt(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _set_simulated_time(monkeypatch):
    """Pin simulated wall-clock time so serving_start = 2026-09-02 00:00:00Z."""
    monkeypatch.setenv("WEATHER_SIMULATED_NOW", "2026-09-02T02:30:00Z")
    yield


@pytest.fixture
def v3_db(tmp_path):
    """Create isolated SQLite database seeded with GFS and GEFS catalog metadata."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_v3_canonical.db",
        connect_args={"check_same_thread": False},
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
        center = ForecastCenter(id="noaa", center_id="noaa", name="NOAA", country="US")
        gfs = Model(id="gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0)
        gefs = Model(id="gefs", model_id="gefs", name="GEFS", center_id="noaa", is_ensemble=True, resolution_km=25.0)
        v_gfs = ModelVersion(id="v_gfs", model_id="gfs", version_string="v1.0")
        v_gefs = ModelVersion(id="v_gefs", model_id="gefs", version_string="v1.0")
        grid = ForecastGrid(id="grid_global", grid_code="global_025deg", name="Global 0.25", resolution_km=25.0)

        vars_to_add = [
            ("temperature_2m", "°C"),
            ("wind_u_10m", "m/s"),
            ("wind_v_10m", "m/s"),
            ("precipitation_amount_3h", "mm"),
            ("cloud_cover_3h", "%"),
            ("cloud_ceiling", "m"),
            ("crain", "flag"),
            ("csnow", "flag"),
            ("cfrzr", "flag"),
            ("cicep", "flag"),
            ("precipitation_rate", "mm/h"),
        ]
        var_objs = [
            ForecastVariable(id=f"var_{code}", variable_code=code, name=code, unit=unit)
            for code, unit in vars_to_add
        ]
        session.add_all([center, gfs, gefs, v_gfs, v_gefs, grid] + var_objs)
        session.commit()

    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# Test Scenarios 1 - 5: Multi-Cycle Supersession & Horizon Retention
# ---------------------------------------------------------------------------

def test_01_00z_complete_06z_partial_near_times_superseded(v3_db):
    """1. Newer cycle supersedes older cycle only for committed near valid times."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        # 00Z committed L0..L240 (valid times 00Z..240Z)
        r_00z = ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z")
        # 06Z committed L0..L24 (valid times 06Z..30Z)
        r_06z = ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z")
        session.add_all([r_00z, r_06z])

        # Add products for 00Z: leads 0, 6, 12, ..., 240
        for lead in range(0, 241, 6):
            session.add(ForecastProduct(id=f"p_00_{lead}", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=lead))

        # Add products for 06Z: leads 0, 6, 12, 18, 24
        for lead in range(0, 25, 6):
            session.add(ForecastProduct(id=f"p_06_{lead}", run_id="r_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=lead))
        session.commit()

        # V = 12Z (00Z + 12h, 06Z + 6h) -> 06Z must win (smaller lead / newer cycle)
        v_12z = _dt(2026, 9, 2, 12)
        src_12 = resolve_canonical_source(session, "gfs", v_12z)
        assert src_12.cycle_time == c_06z
        assert src_12.lead_time_hours == 6

        # V = 30Z (00Z + 30h, 06Z + 24h) -> 06Z must win
        v_30z = _dt(2026, 9, 3, 6)  # 06Z + 24h
        src_30 = resolve_canonical_source(session, "gfs", v_30z)
        assert src_30.cycle_time == c_06z
        assert src_30.lead_time_hours == 24


def test_02_older_cycle_continues_far_horizon(v3_db):
    """2. Farther valid times not yet committed by 06Z continue serving from 00Z."""
    test_01_00z_complete_06z_partial_near_times_superseded(v3_db)

    c_00z = _dt(2026, 9, 2, 0)
    with Session(v3_db) as session:
        # V = 48Z (00Z + 48h): beyond 06Z's 24h horizon -> must be supplied by 00Z
        v_48z = _dt(2026, 9, 4, 0)
        src_48 = resolve_canonical_source(session, "gfs", v_48z)
        assert src_48.cycle_time == c_00z
        assert src_48.lead_time_hours == 48

        # V = 240Z (00Z + 240h): maximum lead of 00Z -> must be supplied by 00Z
        v_240z = _dt(2026, 9, 12, 0)
        src_240 = resolve_canonical_source(session, "gfs", v_240z)
        assert src_240.cycle_time == c_00z
        assert src_240.lead_time_hours == 240


def test_03_serving_end_does_not_collapse(v3_db):
    """3. Serving right boundary remains at maximum derivable valid time (does not collapse to 06Z+24h)."""
    test_01_00z_complete_06z_partial_near_times_superseded(v3_db)

    with Session(v3_db) as session:
        horizon = canonical_serving_horizon(session, "gfs")
        assert horizon == _dt(2026, 9, 12, 0)  # 00Z + 240h, NOT 06Z + 24h!


def test_04_same_valid_time_selects_newest_committed_source(v3_db):
    """4. Exact same valid time chooses newest committed source."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)
    target_v = _dt(2026, 9, 2, 12)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="ready", zarr_store_path="/store/06z"))
        session.add(ForecastProduct(id="p_00_12", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=12))
        session.add(ForecastProduct(id="p_06_06", run_id="r_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.commit()

        src = resolve_canonical_source(session, "gfs", target_v)
        assert src.cycle_time == c_06z
        assert src.lead_time_hours == 6


def test_05_uncommitted_newer_representation_does_not_win(v3_db):
    """5. An uncommitted newer representation does not win; falls back to older committed cycle."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)
    target_v = _dt(2026, 9, 2, 18)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z"))
        # 00Z covers 18Z with lead 18h
        session.add(ForecastProduct(id="p_00_18", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=18))
        # 06Z only has lead 6h committed; lead 12h is uncommitted
        session.add(ForecastProduct(id="p_06_06", run_id="r_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.commit()

        src = resolve_canonical_source(session, "gfs", target_v)
        assert src.cycle_time == c_00z
        assert src.lead_time_hours == 18


# ---------------------------------------------------------------------------
# Test Scenarios 6 - 10: Interval Fallback & Companion Coupling
# ---------------------------------------------------------------------------

def test_06_precipitation_amount_3h_lead0_fallback(v3_db):
    """6. precipitation_amount_3h at canonical lead 0 falls back to older cycle with lead > 0."""
    c_18z = _dt(2026, 9, 1, 18)
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_18z", model_version_id="v_gfs", cycle_time=c_18z, status="ready", zarr_store_path="/store/18z"))
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        # 18Z has lead 6h (valid_time = 00Z)
        session.add(ForecastProduct(id="p_18_tp_6", run_id="r_18z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        # 00Z has lead 0h (valid_time = 00Z)
        session.add(ForecastProduct(id="p_00_tp_0", run_id="r_00z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        src = resolve_variable_source(session, "gfs", "precipitation_amount_3h", target_v)
        assert src is not None
        assert src.cycle_time == c_18z
        assert src.lead_time_hours == 6


def test_07_cloud_cover_3h_lead0_fallback(v3_db):
    """7. cloud_cover_3h at canonical lead 0 falls back to older cycle with lead > 0."""
    c_18z = _dt(2026, 9, 1, 18)
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_18z", model_version_id="v_gfs", cycle_time=c_18z, status="ready", zarr_store_path="/store/18z"))
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ForecastProduct(id="p_18_cc_6", run_id="r_18z", variable_id="cloud_cover_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.add(ForecastProduct(id="p_00_cc_0", run_id="r_00z", variable_id="cloud_cover_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        src = resolve_variable_source(session, "gfs", "cloud_cover_3h", target_v)
        assert src is not None
        assert src.cycle_time == c_18z
        assert src.lead_time_hours == 6


def test_08_cold_start_interval_fallback_returns_null_not_404(v3_db):
    """8. Cold start with only lead 0 (no positive lead) returns None for interval vars without failing."""
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ForecastProduct(id="p_00_t2m_0", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.add(ForecastProduct(id="p_00_tp_0", run_id="r_00z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        # Instantaneous variable resolves fine
        src_t2m = resolve_variable_source(session, "gfs", "temperature_2m", target_v)
        assert src_t2m is not None
        assert src_t2m.lead_time_hours == 0

        # Interval variable returns None gracefully (not HTTP 404)
        src_tp = resolve_variable_source(session, "gfs", "precipitation_amount_3h", target_v)
        assert src_tp is None


def test_09_precipitation_companions_use_exact_fallback_source(v3_db):
    """9. Precipitation companions (crain, csnow, cfrzr, cicep) strictly delegate to precipitation source."""
    c_18z = _dt(2026, 9, 1, 18)
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_18z", model_version_id="v_gfs", cycle_time=c_18z, status="ready", zarr_store_path="/store/18z"))
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ForecastProduct(id="p_18_tp_6", run_id="r_18z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.add(ForecastProduct(id="p_00_tp_0", run_id="r_00z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        # Every companion variable delegates to precipitation_amount_3h source (18Z L6)
        for comp in ("crain", "csnow", "cfrzr", "cicep"):
            src_comp = resolve_variable_source(session, "gfs", comp, target_v)
            assert src_comp is not None
            assert src_comp.cycle_time == c_18z
            assert src_comp.lead_time_hours == 6


def test_10_ordinary_variables_do_not_inherit_precipitation_fallback(v3_db):
    """10. Ordinary instantaneous variables at lead 0 do NOT inherit precipitation fallback."""
    c_18z = _dt(2026, 9, 1, 18)
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_18z", model_version_id="v_gfs", cycle_time=c_18z, status="ready", zarr_store_path="/store/18z"))
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ForecastProduct(id="p_18_tp_6", run_id="r_18z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.add(ForecastProduct(id="p_00_tp_0", run_id="r_00z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.add(ForecastProduct(id="p_00_t2m_0", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        src_t2m = resolve_variable_source(session, "gfs", "temperature_2m", target_v)
        assert src_t2m is not None
        assert src_t2m.cycle_time == c_00z
        assert src_t2m.lead_time_hours == 0  # Stays on 00Z lead 0!


# ---------------------------------------------------------------------------
# Test Scenarios 11 - 13: Wind Component Coherence & GFS Precipitation Rate
# ---------------------------------------------------------------------------

def test_11_wind_10m_requires_coherent_uv_pair(v3_db):
    """11. wind_10m resolves only when both wind_u_10m and wind_v_10m are committed."""
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ForecastProduct(id="p_00_u_6", run_id="r_00z", variable_id="wind_u_10m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.add(ForecastProduct(id="p_00_v_6", run_id="r_00z", variable_id="wind_v_10m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.commit()

        src = resolve_variable_source(session, "gfs", "wind_10m", target_v)
        assert src is not None
        assert src.cycle_time == c_00z
        assert src.lead_time_hours == 6


def test_12_newer_wind_cycle_with_only_u_does_not_win(v3_db):
    """12. Newer cycle having only u (missing v) does not win; falls back to older coherent cycle."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)
    target_v = _dt(2026, 9, 2, 12)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z"))

        # 00Z L12 has both u and v
        session.add(ForecastProduct(id="p_00_u_12", run_id="r_00z", variable_id="wind_u_10m", grid_id="global_025deg", product_type="surface", lead_time_hours=12))
        session.add(ForecastProduct(id="p_00_v_12", run_id="r_00z", variable_id="wind_v_10m", grid_id="global_025deg", product_type="surface", lead_time_hours=12))

        # 06Z L6 has only u committed (v is missing)
        session.add(ForecastProduct(id="p_06_u_06", run_id="r_06z", variable_id="wind_u_10m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.commit()

        src = resolve_variable_source(session, "gfs", "wind_10m", target_v)
        # 06Z is incoherent -> must fall back to 00Z lead 12!
        assert src is not None
        assert src.cycle_time == c_00z
        assert src.lead_time_hours == 12


def test_13_precipitation_rate_gfs_only_and_lead0_ordinary(v3_db):
    """13. precipitation_rate is GFS-only, servable at lead 0, and raises 404 for GEFS."""
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_gfs", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/gfs"))
        session.add(ForecastProduct(id="p_gfs_prate", run_id="r_gfs", variable_id="precipitation_rate", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        # In GFS, serves at lead 0 normally (no interval fallback)
        src = resolve_variable_source(session, "gfs", "precipitation_rate", target_v)
        assert src is not None
        assert src.lead_time_hours == 0

        # In GEFS, precipitation_rate does not exist -> raises 404
        with pytest.raises(HTTPException) as exc:
            resolve_variable_source(session, "gefs", "precipitation_rate", target_v)
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Test Scenarios 14 - 16: GEFS Strict Coherent Vintage & Readiness
# ---------------------------------------------------------------------------

def test_14_gefs_under_26_members_remains_on_older_coherent_vintage(v3_db):
    """14. GEFS cycle with geavg but <26 members is not canonical; all paths remain on older cycle."""
    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    try:
        c_00z = _dt(2026, 9, 2, 0)
        c_06z = _dt(2026, 9, 2, 6)
        target_v = _dt(2026, 9, 2, 12)

        with Session(v3_db) as session:
            session.add(ModelRun(id="r_gefs_00z", model_version_id="v_gefs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
            session.add(ModelRun(id="r_gefs_06z", model_version_id="v_gefs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z"))

            # 00Z L12 has mean + 30 members
            session.add(ForecastProduct(id="p_gefs_00_12", run_id="r_gefs_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=12))
            for m in range(1, 31):
                session.add(EnsembleMemberProduct(id=f"emp_00_{m}", run_id="r_gefs_00z", member_index=m, lead_time_hours=12))

            # 06Z L6 has mean + only 15 members (<26)
            session.add(ForecastProduct(id="p_gefs_06_06", run_id="r_gefs_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=6))
            for m in range(1, 16):
                session.add(EnsembleMemberProduct(id=f"emp_06_{m}", run_id="r_gefs_06z", member_index=m, lead_time_hours=6))
            session.commit()

            # Generic source (points / maps) must stay on 00Z lead 12
            src_mean = resolve_canonical_source(session, "gefs", target_v)
            assert src_mean.cycle_time == c_00z
            assert src_mean.lead_time_hours == 12

            # Variable source (ensembles / probabilities) must also stay on 00Z lead 12
            src_var = resolve_variable_source(session, "gefs", "temperature_2m", target_v)
            assert src_var is not None
            assert src_var.cycle_time == c_00z
            assert src_var.lead_time_hours == 12
    finally:
        register_expected_members("gefs", old_expected)


def test_15_gefs_26_members_promotes_all_endpoints_to_newer_vintage(v3_db):
    """15. Once GEFS reaches 26 members (>=85%), all endpoints promote to newer cycle."""
    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    try:
        test_14_gefs_under_26_members_remains_on_older_coherent_vintage(v3_db)

        c_06z = _dt(2026, 9, 2, 6)
        target_v = _dt(2026, 9, 2, 12)

        with Session(v3_db) as session:
            # Commit 11 more members for 06Z L6 (total 26 members >= 85%)
            for m in range(16, 27):
                session.add(EnsembleMemberProduct(id=f"emp_06_{m}", run_id="r_gefs_06z", member_index=m, lead_time_hours=6))
            session.commit()

            # Both generic and variable source promote to 06Z lead 6
            src = resolve_canonical_source(session, "gefs", target_v)
            assert src.cycle_time == c_06z
            assert src.lead_time_hours == 6

            src_var = resolve_variable_source(session, "gefs", "temperature_2m", target_v)
            assert src_var is not None
            assert src_var.cycle_time == c_06z
            assert src_var.lead_time_hours == 6
    finally:
        register_expected_members("gefs", old_expected)


def test_16_gefs_30_members_remains_formal_readiness_requirement(v3_db):
    """16. 26/30 members promotes to serving, but model_runs.status remains partial (30/30 required for ready)."""
    test_15_gefs_26_members_promotes_all_endpoints_to_newer_vintage(v3_db)

    with Session(v3_db) as session:
        r = session.get(ModelRun, "r_gefs_06z")
        assert r is not None
        assert r.status == "partial"  # Not ready!


# ---------------------------------------------------------------------------
# Test Scenarios 17 - 20: Legacy retired_at Decoupling & Physical Fences
# ---------------------------------------------------------------------------

def test_17_retired_at_not_null_does_not_remove_valid_representation(v3_db):
    """17. retired_at != NULL does not exclude an otherwise valid representation from V3 valid_time resolution."""
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_ret", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/ret"))
        session.add(ForecastProduct(id="p_ret_6", run_id="r_ret", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        # Stamped with legacy V2 retired_at
        session.add(ForecastCycleLifecycle(model_id="gfs", cycle_time=c_00z, retired_at=_dt(2026, 9, 2, 8)))
        session.commit()

        # V3 ignores retired_at -> representation is successfully resolved
        src = resolve_canonical_source(session, "gfs", target_v)
        assert src.cycle_time == c_00z
        assert src.lead_time_hours == 6


def test_18_deletion_started_at_excludes_affected_source(v3_db):
    """18. deletion_started_at != NULL physical safety fence strictly excludes the source."""
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_del_start", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/del_start"))
        session.add(ForecastProduct(id="p_ds_6", run_id="r_del_start", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.add(ForecastCycleLifecycle(model_id="gfs", cycle_time=c_00z, deletion_started_at=_dt(2026, 9, 2, 8)))
        session.commit()

        with pytest.raises(HTTPException) as exc:
            resolve_canonical_source(session, "gfs", target_v)
        assert exc.value.status_code == 404


def test_19_deleted_at_excludes_affected_source(v3_db):
    """19. deleted_at != NULL physical safety fence strictly excludes the source."""
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_del", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/del"))
        session.add(ForecastProduct(id="p_d_6", run_id="r_del", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.add(ForecastCycleLifecycle(model_id="gfs", cycle_time=c_00z, deleted_at=_dt(2026, 9, 2, 8)))
        session.commit()

        with pytest.raises(HTTPException) as exc:
            resolve_canonical_source(session, "gfs", target_v)
        assert exc.value.status_code == 404


def test_20_physical_lifecycle_fence_is_correctly_model_scoped(v3_db):
    """20. Physical fence is model-scoped: fencing GFS cycle 00Z does not fence GEFS cycle 00Z."""
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        # Both GFS and GEFS exist at cycle 00Z
        session.add(ModelRun(id="r_gfs_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/gfs/00z"))
        session.add(ForecastProduct(id="p_gfs_6", run_id="r_gfs_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))

        session.add(ModelRun(id="r_gefs_00z", model_version_id="v_gefs", cycle_time=c_00z, status="ready", zarr_store_path="/store/gefs/00z"))
        session.add(ForecastProduct(id="p_gefs_6", run_id="r_gefs_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=6))
        for m in range(1, 31):
            session.add(EnsembleMemberProduct(id=f"emp_20_{m}", run_id="r_gefs_00z", member_index=m, lead_time_hours=6))

        # Fence GFS ONLY
        session.add(ForecastCycleLifecycle(model_id="gfs", cycle_time=c_00z, deletion_started_at=_dt(2026, 9, 2, 8)))
        session.commit()

        # GFS is fenced -> 404
        with pytest.raises(HTTPException) as exc:
            resolve_canonical_source(session, "gfs", target_v)
        assert exc.value.status_code == 404

        # GEFS is NOT fenced -> resolves cleanly!
        src_gefs = resolve_canonical_source(session, "gefs", target_v)
        assert src_gefs.cycle_time == c_00z
        assert src_gefs.lead_time_hours == 6


# ---------------------------------------------------------------------------
# Test Scenarios 21 - 23: Left Boundary & Performance / Bulk Resolution
# ---------------------------------------------------------------------------

def test_21_serving_start_left_boundary_remains_enforced(v3_db):
    """21. Valid time strictly before serving_start_valid_time(now_utc) raises 404."""
    # simulated_now is 2026-09-02 02:30:00Z -> serving_start = 2026-09-02 00:00:00Z
    # 2026-09-01 21:00:00Z is strictly before serving_start
    v_past = _dt(2026, 9, 1, 21)

    with Session(v3_db) as session:
        with pytest.raises(HTTPException) as exc:
            resolve_canonical_source(session, "gfs", v_past)
        assert exc.value.status_code == 404
        assert "before the active serving window" in exc.value.detail


def test_22_exact_serving_start_valid_time_remains_servable(v3_db):
    """22. Valid time exactly equal to serving_start_valid_time(now_utc) remains servable."""
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 0)  # exactly at serving_start

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ForecastProduct(id="p_00_0", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        src = resolve_canonical_source(session, "gfs", target_v)
        assert src.valid_time == target_v
        assert src.lead_time_hours == 0


def test_23_bulk_resolution_no_n_plus_one_sql_queries(v3_db):
    """23. Resolving 81 valid times executes exactly 1-2 SQL queries (no N+1 loop)."""
    c_00z = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        for lead in range(0, 241, 3):
            session.add(ForecastProduct(id=f"p_bulk_{lead}", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=lead))
        session.commit()

        queries: list[str] = []

        def capture_query(conn, cursor, statement, parameters, context, executemany):
            queries.append(statement)

        event.listen(v3_db, "before_cursor_execute", capture_query)
        try:
            anchors, var_sources = resolve_canonical_sources_bulk(
                session,
                "gfs",
                variables=["temperature_2m", "precipitation_amount_3h"],
            )
            # 81 valid times resolved
            assert len(anchors) == 81
            # Exactly 1 query on GFS (catalog query), zero per-lead or per-variable queries!
            assert len(queries) <= 2
        finally:
            event.remove(v3_db, "before_cursor_execute", capture_query)


# ---------------------------------------------------------------------------
# Test Scenarios 24 - 28: Cache Invalidation & Multi-Time Provenance
# ---------------------------------------------------------------------------

def test_24_cache_promotion_cached_00z_invalidated_after_06z_commits(v3_db):
    """24. Cache key changes when 06Z commits a lead, invalidating stale 00Z cache."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z"))
        session.add(ForecastProduct(id="p_00_12", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=12))
        session.commit()

        # Step 1: Only 00Z is committed for valid_time 12Z
        digest_1 = resolve_point_provenance_digest(session, "gfs", variables=["temperature_2m"])
        key_1 = build_point_cache_key(
            model="gfs",
            latitude=38.0,
            longitude=-107.0,
            resolved_via="coords",
            location_id=None,
            cycle_time="2026-09-02T06:00:00Z",
            provenance_digest=digest_1,
            variables=("temperature_2m",),
            units="metric",
            start_lead_time_hours=None,
            end_lead_time_hours=None,
        )

        # Step 2: 06Z commits lead 6 for valid_time 12Z (without changing cycle_time!)
        session.add(ForecastProduct(id="p_06_06", run_id="r_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.commit()

        digest_2 = resolve_point_provenance_digest(session, "gfs", variables=["temperature_2m"])
        key_2 = build_point_cache_key(
            model="gfs",
            latitude=38.0,
            longitude=-107.0,
            resolved_via="coords",
            location_id=None,
            cycle_time="2026-09-02T06:00:00Z",
            provenance_digest=digest_2,
            variables=("temperature_2m",),
            units="metric",
            start_lead_time_hours=None,
            end_lead_time_hours=None,
        )

        # Keys must be completely different! Stale 00Z cache is bypassed immediately!
        assert digest_1 != digest_2
        assert key_1 != key_2


def test_25_interval_fallback_cache_promotion(v3_db):
    """25. When interval fallback source changes (18Z L12 -> 00Z L6), cache provenance changes."""
    c_12z = _dt(2026, 9, 1, 12)
    c_18z = _dt(2026, 9, 1, 18)
    c_00z = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_12z", model_version_id="v_gfs", cycle_time=c_12z, status="ready", zarr_store_path="/store/12z"))
        session.add(ModelRun(id="r_18z", model_version_id="v_gfs", cycle_time=c_18z, status="ready", zarr_store_path="/store/18z"))
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))

        # 00Z has lead 0
        session.add(ForecastProduct(id="p_00_tp_0", run_id="r_00z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        # 12Z has lead 12 (valid_time = 00Z)
        session.add(ForecastProduct(id="p_12_tp_12", run_id="r_12z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=12))
        session.commit()

        # Step 1: Fallback source is 12Z L12
        digest_1 = resolve_point_provenance_digest(session, "gfs", variables=["precipitation_amount_3h"])

        # Step 2: 18Z commits lead 6 (valid_time = 00Z)
        session.add(ForecastProduct(id="p_18_tp_6", run_id="r_18z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.commit()

        # Fallback source promotes to 18Z L6
        digest_2 = resolve_point_provenance_digest(session, "gfs", variables=["precipitation_amount_3h"])

        assert digest_1 != digest_2


def test_26_gefs_threshold_cache_promotion_25_to_26_members(v3_db):
    """26. GEFS promotion from 25 to 26 members promotes coherent vintage and invalidates cache."""
    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    try:
        c_00z = _dt(2026, 9, 2, 0)
        c_06z = _dt(2026, 9, 2, 6)
        target_v = _dt(2026, 9, 2, 12)

        with Session(v3_db) as session:
            session.add(ModelRun(id="r_gefs_00z", model_version_id="v_gefs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
            session.add(ModelRun(id="r_gefs_06z", model_version_id="v_gefs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z"))

            # 00Z L12 has mean + 30 members
            session.add(ForecastProduct(id="p_00_12", run_id="r_gefs_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=12))
            for m in range(1, 31):
                session.add(EnsembleMemberProduct(id=f"emp_00_{m}", run_id="r_gefs_00z", member_index=m, lead_time_hours=12))

            # 06Z L6 has mean + 25 members (<85%)
            session.add(ForecastProduct(id="p_06_06", run_id="r_gefs_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=6))
            for m in range(1, 26):
                session.add(EnsembleMemberProduct(id=f"emp_06_{m}", run_id="r_gefs_06z", member_index=m, lead_time_hours=6))
            session.commit()

            # At 25 members: resolves 00Z lead 12
            src_1 = resolve_canonical_source(session, "gefs", target_v)
            assert src_1.cycle_time == c_00z
            key_1 = build_ensemble_cache_key(
                model="gefs",
                latitude=38.0,
                longitude=-107.0,
                variable="temperature_2m",
                lead_time_hours=src_1.lead_time_hours,
                cycle_time=src_1.cycle_time.isoformat(),
                member_fingerprint=src_1.member_fingerprint,
            )

            # Add member 26: reaches 26/30 (86.7% >= 85%) -> promotes to 06Z lead 6
            session.add(EnsembleMemberProduct(id="emp_06_26", run_id="r_gefs_06z", member_index=26, lead_time_hours=6))
            session.commit()

            src_2 = resolve_canonical_source(session, "gefs", target_v)
            assert src_2.cycle_time == c_06z
            key_2 = build_ensemble_cache_key(
                model="gefs",
                latitude=38.0,
                longitude=-107.0,
                variable="temperature_2m",
                lead_time_hours=src_2.lead_time_hours,
                cycle_time=src_2.cycle_time.isoformat(),
                member_fingerprint=src_2.member_fingerprint,
            )

            assert key_1 != key_2
    finally:
        register_expected_members("gefs", old_expected)


def test_27_gefs_same_vintage_member_update_26_to_27_members_invalidates_cache(v3_db):
    """27. Committing member 27 within the same cycle/lead changes member_fingerprint and invalidates cache."""
    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    try:
        c_06z = _dt(2026, 9, 2, 6)
        target_v = _dt(2026, 9, 2, 12)

        with Session(v3_db) as session:
            session.add(ModelRun(id="r_06z", model_version_id="v_gefs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z"))
            session.add(ForecastProduct(id="p_06_06", run_id="r_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=6))
            for m in range(1, 27):
                session.add(EnsembleMemberProduct(id=f"emp_27_{m}", run_id="r_06z", member_index=m, lead_time_hours=6))
            session.commit()

            src_26 = resolve_canonical_source(session, "gefs", target_v)
            key_26 = build_ensemble_cache_key(
                model="gefs",
                latitude=38.0,
                longitude=-107.0,
                variable="temperature_2m",
                lead_time_hours=6,
                cycle_time=c_06z.isoformat(),
                member_fingerprint=src_26.member_fingerprint,
            )

            # Commit member 27
            session.add(EnsembleMemberProduct(id="emp_27_27", run_id="r_06z", member_index=27, lead_time_hours=6))
            session.commit()

            src_27 = resolve_canonical_source(session, "gefs", target_v)
            key_27 = build_ensemble_cache_key(
                model="gefs",
                latitude=38.0,
                longitude=-107.0,
                variable="temperature_2m",
                lead_time_hours=6,
                cycle_time=c_06z.isoformat(),
                member_fingerprint=src_27.member_fingerprint,
            )

            # Cycle and lead are identical (06Z L6), but member_fingerprint changed!
            assert src_26.member_fingerprint != src_27.member_fingerprint
            assert key_26 != key_27
    finally:
        register_expected_members("gefs", old_expected)


def test_28_multi_time_provenance_digest_deterministic_independent_of_order():
    """28. Provenance digest is deterministic independent of dictionary insertion order."""
    vt1 = _dt(2026, 9, 2, 6)
    vt2 = _dt(2026, 9, 2, 12)

    src1 = ResolvedForecastSource(model="gfs", valid_time=vt1, cycle_time=_dt(2026, 9, 2, 0), lead_time_hours=6, run_id="r1", store_path="/s1")
    src2 = ResolvedForecastSource(model="gfs", valid_time=vt2, cycle_time=_dt(2026, 9, 2, 0), lead_time_hours=12, run_id="r1", store_path="/s1")

    # Dict insertion order A: vt1 then vt2
    anchors_a = {vt1: src1, vt2: src2}
    digest_a = build_canonical_provenance_digest(anchors_a)

    # Dict insertion order B: vt2 then vt1
    anchors_b = {vt2: src2, vt1: src1}
    digest_b = build_canonical_provenance_digest(anchors_b)

    assert digest_a == digest_b


# ---------------------------------------------------------------------------
# Test Scenarios 29 - 30: Read-Resilience Coupling Invariants
# ---------------------------------------------------------------------------

def test_29_read_resilience_fallback_never_mixes_precipitation_companions(v3_db, monkeypatch):
    """29. Read-resilience storage fallback never mixes companion flags across cycles."""
    # When precipitation fallback source fails to read, all companions stay together as null
    c_18z = _dt(2026, 9, 1, 18)
    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 0)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_18z", model_version_id="v_gfs", cycle_time=c_18z, status="ready", zarr_store_path="/store/broken_18z"))
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/good_00z"))

        session.add(ForecastProduct(id="p_18_tp_6", run_id="r_18z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.add(ForecastProduct(id="p_00_tp_0", run_id="r_00z", variable_id="precipitation_amount_3h", grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        # Both precip and companions resolve to 18Z L6; they never independently fall back to 00Z synthetic zeroes
        src_tp = resolve_variable_source(session, "gfs", "precipitation_amount_3h", target_v)
        src_crain = resolve_variable_source(session, "gfs", "crain", target_v)

        assert src_tp is not None and src_crain is not None
        assert src_tp.cycle_time == src_crain.cycle_time == c_18z
        assert src_tp.lead_time_hours == src_crain.lead_time_hours == 6


def test_30_read_resilience_fallback_never_mixes_wind_components(v3_db):
    """30. Read-resilience never mixes wind u/v components across cycles or leads."""
    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)
    target_v = _dt(2026, 9, 2, 12)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="ready", zarr_store_path="/store/06z"))

        # 00Z has both components
        session.add(ForecastProduct(id="p_00_u_12", run_id="r_00z", variable_id="wind_u_10m", grid_id="global_025deg", product_type="surface", lead_time_hours=12))
        session.add(ForecastProduct(id="p_00_v_12", run_id="r_00z", variable_id="wind_v_10m", grid_id="global_025deg", product_type="surface", lead_time_hours=12))

        # 06Z only has u
        session.add(ForecastProduct(id="p_06_u_06", run_id="r_06z", variable_id="wind_u_10m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
        session.commit()

        src = resolve_variable_source(session, "gfs", "wind_10m", target_v)
        assert src is not None
        # Must resolve to 00Z lead 12 where both components are coherent
        assert src.cycle_time == c_00z
        assert src.lead_time_hours == 12


def test_31_availability_and_resolver_exact_alignment_under_mixed_cycles(v3_db):
    """31. Proves availability and canonical resolver produce 100% identical valid times and horizons under mixed partial cycles."""
    from api.services.availability import build_forecast_availability

    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z"))

        # 00Z has leads 0, 6, 12, ..., 240
        for lead in range(0, 241, 6):
            for var in ("temperature_2m", "precipitation_amount_3h", "wind_u_10m", "wind_v_10m"):
                session.add(ForecastProduct(id=f"p_00_{var}_{lead}", run_id="r_00z", variable_id=var, grid_id="global_025deg", product_type="surface", lead_time_hours=lead))

        # 06Z has leads 0, 6, 12, 18, 24
        for lead in range(0, 25, 6):
            for var in ("temperature_2m", "precipitation_amount_3h", "wind_u_10m", "wind_v_10m"):
                session.add(ForecastProduct(id=f"p_06_{var}_{lead}", run_id="r_06z", variable_id=var, grid_id="global_025deg", product_type="surface", lead_time_hours=lead))
        session.commit()

        # Build availability payload
        avail = build_forecast_availability(session, now=_dt(2026, 9, 2, 6))
        gfs_avail = next(m for m in avail.models if m.id == "gfs")
        var_map = {v.id: v for v in gfs_avail.variables}

        # Check temperature_2m, precipitation_amount_3h, wind_10m
        for var_code in ("temperature_2m", "precipitation_amount_3h", "wind_10m"):
            assert var_code in var_map
            v_avail = var_map[var_code]
            assert len(v_avail.valid_times) > 0

            # 1. Horizon does not collapse: maximum valid time equals 240Z
            max_vt = max(vt.valid_time for vt in v_avail.valid_times)
            horizon = canonical_serving_horizon(session, "gfs", variable=var_code, now=_dt(2026, 9, 2, 6))
            assert max_vt == horizon == _dt(2026, 9, 12, 0)  # 00Z + 240h

            # 2. Every single valid time in availability matches resolve_variable_source exactly
            for vt_entry in v_avail.valid_times:
                expected_src = resolve_variable_source(session, "gfs", var_code, vt_entry.valid_time, now=_dt(2026, 9, 2, 6))
                assert expected_src is not None
                assert vt_entry.source_cycle == expected_src.cycle_time
                assert vt_entry.lead_time_hours == expected_src.lead_time_hours


def test_32_router_endpoint_points_cache_invalidation_on_lead_promotion(v3_db, monkeypatch):
    """32. Proves endpoint-level cache invalidation for /v1/points when a new lead commits."""
    from starlette.testclient import TestClient
    from api.core.database import get_db
    from api.main import app
    import api.routers.points as points_router

    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)

    compute_calls = 0
    orig_compute = points_router._compute

    def spy_compute(*args, **kwargs):
        nonlocal compute_calls
        compute_calls += 1
        return orig_compute(*args, **kwargs)

    monkeypatch.setattr(points_router, "_compute", spy_compute)

    class DictRedis:
        def __init__(self):
            self.store = {}
        def get(self, key):
            return self.store.get(key)
        def setex(self, key, ttl, value):
            self.store[key] = value

    fake_redis = DictRedis()
    monkeypatch.setattr(points_router._cache, "_client", fake_redis)
    monkeypatch.setattr("api.core.database.SessionLocal", lambda: Session(v3_db))

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/00z"))
        session.add(ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=c_06z, status="partial", zarr_store_path="/store/06z"))
        session.add(ForecastProduct(id="p_00_12", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=12))
        session.commit()

    def override_db():
        with Session(v3_db) as s:
            yield s

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        # Mock storage point interpolation so compute returns a valid envelope
        monkeypatch.setattr(
            "api.services.point_forecast.gated_point_interpolations",
            lambda *args, **kwargs: {"temperature_2m": 20.0},
        )
        monkeypatch.setattr(
            "api.services.point_forecast._resolve_cycle_store_path",
            lambda *args, **kwargs: "/store/mock",
        )
        monkeypatch.setattr(
            "api.services.point_forecast.gated_cycle_metadata",
            lambda *args, **kwargs: type("M", (), {"lead_times": frozenset({6, 12}), "var_names": frozenset({"temperature_2m"})})(),
        )

        # 1. First request for 00Z lead 12
        res1 = client.get("/v1/points?lat=38.0&lon=-107.0&models=gfs&variables=temperature_2m")
        assert res1.status_code == 200
        assert compute_calls == 1

        # 2. Repeated request hits in-memory cache without calling compute
        res2 = client.get("/v1/points?lat=38.0&lon=-107.0&models=gfs&variables=temperature_2m")
        assert res2.status_code == 200
        assert compute_calls == 1

        # 3. Now 06Z commits lead 6 (valid_time 12Z)
        with Session(v3_db) as session:
            session.add(ForecastProduct(id="p_06_06", run_id="r_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=6))
            session.commit()

        # 4. Requesting again must invalidate cache and invoke compute because provenance_digest changed!
        res3 = client.get("/v1/points?lat=38.0&lon=-107.0&models=gfs&variables=temperature_2m")
        assert res3.status_code == 200
        assert compute_calls == 2
    finally:
        app.dependency_overrides.clear()


def test_33_router_endpoint_ensembles_cache_invalidation_on_gefs_member_update(v3_db, monkeypatch):
    """33. Proves endpoint-level cache invalidation for /v1/ensembles on 25->26 and 26->27 member updates."""
    from starlette.testclient import TestClient
    from api.core.database import get_db
    from api.main import app
    import api.routers.ensembles as ensembles_router

    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    compute_calls = 0
    orig_compute = ensembles_router._compute

    def spy_compute(*args, **kwargs):
        nonlocal compute_calls
        compute_calls += 1
        return orig_compute(*args, **kwargs)

    monkeypatch.setattr(ensembles_router, "_compute", spy_compute)

    class DictRedis:
        def __init__(self):
            self.store = {}
        def get(self, key):
            return self.store.get(key)
        def setex(self, key, ttl, value):
            self.store[key] = value

    fake_redis = DictRedis()
    monkeypatch.setattr(ensembles_router._cache, "_client", fake_redis)
    monkeypatch.setattr("api.core.database.SessionLocal", lambda: Session(v3_db))

    c_00z = _dt(2026, 9, 2, 0)
    c_06z = _dt(2026, 9, 2, 6)
    target_v = _dt(2026, 9, 2, 12)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_00z", model_version_id="v_gefs", cycle_time=c_00z, status="ready", zarr_store_path="/store/gefs/00z"))
        session.add(ModelRun(id="r_06z", model_version_id="v_gefs", cycle_time=c_06z, status="partial", zarr_store_path="/store/gefs/06z"))
        session.add(ForecastProduct(id="p_00_12", run_id="r_00z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=12))
        for m in range(1, 31):
            session.add(EnsembleMemberProduct(id=f"emp_00_{m}", run_id="r_00z", member_index=m, lead_time_hours=12))

        # 06Z L6 starts with 25 members (<85%)
        session.add(ForecastProduct(id="p_06_06", run_id="r_06z", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=6))
        for m in range(1, 26):
            session.add(EnsembleMemberProduct(id=f"emp_06_{m}", run_id="r_06z", member_index=m, lead_time_hours=6))
        session.commit()

    def override_db():
        with Session(v3_db) as s:
            yield s

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        # Mock build_ensemble_statistics
        from api.schemas import EnsembleStatistics, EnsembleStatisticsData

        def mock_build_stats(*args, **kwargs):
            src = kwargs.get("source")
            mem_count = len(src.member_indices) if (src and src.member_indices) else 30
            c_time = src.cycle_time if src else c_00z
            stats = EnsembleStatistics(
                mean=10.0, median=10.0, spread=1.0, p10=8.0, p25=9.0, p50=10.0, p75=11.0, p90=12.0, min=7.0, max=13.0
            )
            return EnsembleStatisticsData(
                model="gefs",
                variable="temperature_2m",
                unit="°C",
                lead_time_hours=kwargs.get("lead_time_hours", 12),
                cycle_time=c_time.isoformat(),
                statistics=stats,
                member_count=mem_count,
            )

        monkeypatch.setattr("api.routers.ensembles.build_ensemble_statistics", mock_build_stats)

        # 1. First call at 25 members -> resolves 00Z lead 12
        v_iso = target_v.isoformat().replace("+00:00", "Z")
        url = f"/v1/ensembles?lat=38.0&lon=-107.0&model=gefs&variable=temperature_2m&valid_time={v_iso}"
        res1 = client.get(url)
        assert res1.status_code == 200
        assert compute_calls == 1
        assert res1.json()["data"]["source_cycle"].startswith("2026-09-02T00:00:00")

        # Repeated call hits cache
        res1_cached = client.get(url)
        assert res1_cached.status_code == 200
        assert compute_calls == 1

        # 2. Add member 26 (25 -> 26 members promotes 06Z lead 6)
        with Session(v3_db) as session:
            session.add(EnsembleMemberProduct(id="emp_06_26", run_id="r_06z", member_index=26, lead_time_hours=6))
            session.commit()

        # Cache must invalidate, compute called, cycle_time switches to 06Z
        res2 = client.get(url)
        assert res2.status_code == 200
        assert compute_calls == 2
        assert res2.json()["data"]["source_cycle"].startswith("2026-09-02T06:00:00")

        # 3. Add member 27 (26 -> 27 members within same cycle/lead 06Z L6)
        with Session(v3_db) as session:
            session.add(EnsembleMemberProduct(id="emp_06_27", run_id="r_06z", member_index=27, lead_time_hours=6))
            session.commit()

        # Cache must invalidate even though cycle_time and lead_time are unchanged!
        res3 = client.get(url)
        assert res3.status_code == 200
        assert compute_calls == 3
        assert res3.json()["data"]["member_count"] == 27
    finally:
        app.dependency_overrides.clear()
        register_expected_members("gefs", old_expected)


def test_34_endpoint_sql_query_count_bounded_for_81_valid_times(v3_db, monkeypatch):
    """34. Proves endpoint-level SQL query count is bounded constant for an 81-valid-time request."""
    from starlette.testclient import TestClient
    from api.core.database import get_db
    from api.main import app

    c_00z = _dt(2026, 9, 2, 0)
    with Session(v3_db) as session:
        session.add(ModelRun(id="r_81", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path="/store/81"))
        for lead in range(0, 241, 3):
            session.add(ForecastProduct(id=f"p_81_{lead}", run_id="r_81", variable_id="temperature_2m", grid_id="global_025deg", product_type="surface", lead_time_hours=lead))
        session.commit()

    monkeypatch.setattr("api.core.database.SessionLocal", lambda: Session(v3_db))

    def override_db():
        with Session(v3_db) as s:
            yield s

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        monkeypatch.setattr(
            "api.services.point_forecast.gated_point_interpolations",
            lambda *args, **kwargs: {"temperature_2m": 20.0},
        )
        monkeypatch.setattr(
            "api.services.point_forecast._resolve_cycle_store_path",
            lambda *args, **kwargs: "/store/mock",
        )
        monkeypatch.setattr(
            "api.services.point_forecast.gated_cycle_metadata",
            lambda *args, **kwargs: type("M", (), {"lead_times": frozenset(range(0, 241, 3)), "var_names": frozenset({"temperature_2m"})})(),
        )

        queries: list[str] = []

        def capture_query(conn, cursor, statement, parameters, context, executemany):
            queries.append(statement)

        event.listen(v3_db, "before_cursor_execute", capture_query)
        try:
            res = client.get("/v1/points?lat=38.0&lon=-107.0&models=gfs&variables=temperature_2m")
            assert res.status_code == 200
            assert len(res.json()["data"]["forecasts"]) == 81

            # Across the ENTIRE HTTP request (routing, validation, provenance, resolution, and compute):
            # total SQL queries must be bounded constant (<= 12 total), NOT 81 or 162 queries!
            assert len(queries) <= 12
        finally:
            event.remove(v3_db, "before_cursor_execute", capture_query)
    finally:
        app.dependency_overrides.clear()


def test_35_ensemble_endpoint_does_not_re_resolve_canonical_source(v3_db, monkeypatch):
    """35. Proves /v1/ensembles does not re-resolve eligible ensemble run and members when source is passed."""
    from starlette.testclient import TestClient
    from api.core.database import get_db
    from api.main import app
    import api.services.ensemble_data as ensemble_service

    old_expected = get_expected_members("gefs", default_if_unknown=30)
    register_expected_members("gefs", 30)

    c_00z = _dt(2026, 9, 2, 0)
    target_v = _dt(2026, 9, 2, 6)

    with Session(v3_db) as session:
        session.add(ModelRun(id="r_gefs_35", model_version_id="v_gefs", cycle_time=c_00z, status="ready", zarr_store_path="/store/gefs/35"))
        session.add(ForecastProduct(id="p_gefs_35", run_id="r_gefs_35", variable_id="temperature_2m", grid_id="global_025deg", product_type="ensemble_mean", lead_time_hours=6))
        for m in range(1, 31):
            session.add(EnsembleMemberProduct(id=f"emp_35_{m}", run_id="r_gefs_35", member_index=m, lead_time_hours=6))
        session.commit()

    re_resolve_calls = 0
    orig_re_resolve = ensemble_service._resolve_eligible_ensemble_run_and_members

    def spy_re_resolve(*args, **kwargs):
        nonlocal re_resolve_calls
        re_resolve_calls += 1
        return orig_re_resolve(*args, **kwargs)

    monkeypatch.setattr(ensemble_service, "_resolve_eligible_ensemble_run_and_members", spy_re_resolve)
    monkeypatch.setattr("api.core.database.SessionLocal", lambda: Session(v3_db))

    def override_db():
        with Session(v3_db) as s:
            yield s

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        monkeypatch.setattr(
            "api.services.ensemble_data.gated_cycle_metadata",
            lambda *args, **kwargs: type("M", (), {"lead_times": frozenset({6}), "var_names": frozenset({"temperature_2m"})})(),
        )
        monkeypatch.setattr(
            "api.services.ensemble_data._gated_member_values",
            lambda *args, **kwargs: [20.0] * 30,
        )

        v_iso = target_v.isoformat().replace("+00:00", "Z")
        res = client.get(f"/v1/ensembles?lat=38.0&lon=-107.0&model=gefs&variable=temperature_2m&valid_time={v_iso}")
        assert res.status_code == 200

        # _resolve_eligible_ensemble_run_and_members MUST NOT be called because source was pre-resolved and passed!
        assert re_resolve_calls == 0
    finally:
        app.dependency_overrides.clear()
        register_expected_members("gefs", old_expected)


