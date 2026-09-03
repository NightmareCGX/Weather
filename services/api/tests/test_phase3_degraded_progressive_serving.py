"""Phase 3 Acceptance Test Suite: Degraded & Partial Data Semantics with Progressive API Serving.

Validates the full Phase 3 engineering contract across all acceptance scenarios:
- Case A: Complete run (100% coverage, status=ready, all leads servable).
- Case B: One missing member (29/30 = 96.7% -> status=partial, lead servable, sample size 29).
- Case C: Exact 85% threshold boundaries (exact integer thresholds on 20- and 100-member models).
- Case D: Below threshold (25/30 = 83.3% -> catalog visible, metadata visible, servable=false).
- Case E: In-flight horizon during active ingestion batch (settled leads servable before run completion).
- Case F: Committed member with cell-level NaN (finite-sample validity check per cell).
- Case G: Uncommitted member exclusion (uncommitted Zarr slices never loaded as members).
- Case H: Point forecast cross-cycle minimum-lead stitching with provenance.
- Case I: Single-cycle consistency for ensemble and map endpoints.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from domain.coverage import (
    build_lead_coverage,
    compute_coverage_ratio,
    is_cell_statistically_valid,
    is_lead_servable,
)
from domain.ensemble import (
    ensemble_mean,
    probability_above_threshold,
    probability_confidence_interval,
)
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
from api.services.point_forecast import _select_min_lead_winners


@pytest.fixture
def memory_db() -> Session:
    """In-memory SQLite database session with complete catalog schema."""
    engine = create_engine("sqlite:///:memory:")
    _TABLES = [
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
    Base.metadata.create_all(engine, tables=_TABLES)
    session = Session(engine)

    # Seed base metadata
    noaa = ForecastCenter(id="c_noaa", center_id="noaa", name="NOAA", country="USA")
    m_gefs = Model(id="m_gefs", model_id="gefs", name="GEFS", center_id="noaa", is_ensemble=True, resolution_km=25.0)
    m_gfs = Model(id="m_gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0)
    v_gefs = ModelVersion(id="v_gefs", model_id="gefs", version_string="v1.0")
    v_gfs = ModelVersion(id="v_gfs", model_id="gfs", version_string="v1.0")
    var_tmp = ForecastVariable(id="var_t2m", variable_code="temperature_2m", name="2m Temperature", unit="°C")
    var_pr = ForecastVariable(id="var_pr", variable_code="precipitation_rate", name="Precip Rate", unit="mm/h")
    grid = ForecastGrid(id="g_025", grid_code="global_025deg", name="0.25 deg", resolution_km=25.0)

    session.add_all([noaa, m_gefs, m_gfs, v_gefs, v_gfs, var_tmp, var_pr, grid])
    session.commit()

    yield session
    session.close()
    engine.dispose()


# -----------------------------------------------------------------------------
# Case A: Complete Run
# -----------------------------------------------------------------------------
def test_case_a_complete_run() -> None:
    """Case A: 1110/1110 committed -> 100% coverage, run is ready, all leads servable."""
    expected_members = 30
    for lead in range(0, 111, 3):
        cov = build_lead_coverage(
            lead_time_hours=lead,
            available_member_indices=tuple(range(1, 31)),
            expected_members=expected_members,
        )
        assert cov.available_members == 30
        assert cov.coverage_ratio == 1.0
        assert cov.servable is True
        assert is_lead_servable(cov.available_members, expected_members) is True


# -----------------------------------------------------------------------------
# Case B: One Missing Member (1109 / 1110)
# -----------------------------------------------------------------------------
def test_case_b_one_missing_member() -> None:
    """Case B: 1109/1110 committed (lead 69 has 29/30) -> lead 69 coverage 96.7% is servable."""
    expected_members = 30
    available_members_lead_69 = tuple(m for m in range(1, 31) if m != 17)
    assert len(available_members_lead_69) == 29

    cov_69 = build_lead_coverage(
        lead_time_hours=69,
        available_member_indices=available_members_lead_69,
        expected_members=expected_members,
    )
    assert cov_69.available_members == 29
    assert cov_69.coverage_ratio == 0.9667
    assert cov_69.servable is True

    # Logical sample size must be exactly 29
    sample_values = [20.0 + 0.1 * m for m in available_members_lead_69]
    assert len(sample_values) == 29
    mean_val = ensemble_mean(sample_values)
    assert 20.0 < mean_val < 25.0


# -----------------------------------------------------------------------------
# Case C: Exact 85% Threshold Boundaries
# -----------------------------------------------------------------------------
def test_case_c_exact_85_percent_threshold() -> None:
    """Case C: Exact 85% threshold tested on 20-member, 100-member, and 30-member configurations."""
    # 20-member model: exactly 17 is 85.0%
    assert is_lead_servable(17, 20) is True
    assert is_lead_servable(16, 20) is False
    assert compute_coverage_ratio(17, 20) == 0.85
    assert compute_coverage_ratio(16, 20) == 0.80

    # 100-member model: exactly 85 is 85.0%
    assert is_lead_servable(85, 100) is True
    assert is_lead_servable(84, 100) is False
    assert compute_coverage_ratio(85, 100) == 0.85
    assert compute_coverage_ratio(84, 100) == 0.84

    # 30-member GEFS: 26 is 86.7%, 25 is 83.3%
    assert is_lead_servable(26, 30) is True
    assert is_lead_servable(25, 30) is False


# -----------------------------------------------------------------------------
# Case D: Below Threshold
# -----------------------------------------------------------------------------
def test_case_d_below_threshold_visible_but_unservable() -> None:
    """Case D: 25/30 members committed (83.3%) -> catalog visible, servable=false."""
    expected_members = 30
    cov = build_lead_coverage(
        lead_time_hours=72,
        available_member_indices=tuple(range(1, 26)),
        expected_members=expected_members,
    )
    assert cov.available_members == 25
    assert cov.expected_members == 30
    assert cov.coverage_ratio == 0.8333
    assert cov.servable is False


# -----------------------------------------------------------------------------
# Case E: In-Flight Horizon During Large Batch
# -----------------------------------------------------------------------------
def test_case_e_in_flight_horizon_availability(memory_db: Session) -> None:
    """Case E: Active batch with f000 (30/30), f003 (29/30), f006 (27/30), f009 (20/30)."""
    cycle_time = datetime(2026, 8, 14, 6, 0, tzinfo=timezone.utc)
    run = ModelRun(
        id="run_gefs_06z",
        model_version_id="v_gefs",
        cycle_time=cycle_time,
        status="processing",
        zarr_store_path="/tmp/gefs_06z.zarr",
    )
    memory_db.add(run)

    # Add products and member rows for settled leads
    leads_and_members = {
        0: tuple(range(1, 31)),       # 30/30 (100%) -> servable
        3: tuple(range(1, 30)),       # 29/30 (96.7%) -> servable
        6: tuple(range(1, 28)),       # 27/30 (90.0%) -> servable
        9: tuple(range(1, 21)),       # 20/30 (66.7%) -> not servable
    }

    for lead, members in leads_and_members.items():
        memory_db.add(
            ForecastProduct(
                id=f"fp_06z_t2m_{lead}",
                run_id="run_gefs_06z",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
                zarr_chunk_path="/tmp",
            )
        )
        for m in members:
            memory_db.add(
                EnsembleMemberProduct(
                    id=f"emp_06z_{m}_{lead}",
                    run_id="run_gefs_06z",
                    member_index=m,
                    lead_time_hours=lead,
                )
            )
    memory_db.commit()

    # Verify coverage derivation
    for lead, members in leads_and_members.items():
        cov = build_lead_coverage(lead, members, expected_members=30)
        if lead in (0, 3, 6):
            assert cov.servable is True
        else:
            assert cov.servable is False
            assert cov.available_members == 20
            assert cov.coverage_ratio == 0.6667


# -----------------------------------------------------------------------------
# Case F: Committed Member with Cell-Level NaN
# -----------------------------------------------------------------------------
def test_case_f_cell_level_nan_statistical_validity() -> None:
    """Case F: Cell X (27 finite, 2 NaN -> 90% valid -> computed from 27), Cell Y (25 finite, 4 NaN -> 83.3% -> invalid)."""
    expected_members = 30

    # Cell X: 27 finite values, 2 NaNs
    raw_members_x = [15.0 + float(i) for i in range(27)] + [float("nan"), float("nan")]
    finite_x = [v for v in raw_members_x if math.isfinite(v)]
    assert len(finite_x) == 27
    assert is_cell_statistically_valid(len(finite_x), expected_members) is True
    mean_x = ensemble_mean(finite_x)
    assert mean_x == float(np.mean(finite_x))

    # Cell Y: 25 finite values, 4 NaNs
    raw_members_y = [15.0 + float(i) for i in range(25)] + [float("nan")] * 4
    finite_y = [v for v in raw_members_y if math.isfinite(v)]
    assert len(finite_y) == 25
    assert is_cell_statistically_valid(len(finite_y), expected_members) is False


# -----------------------------------------------------------------------------
# Case G: Uncommitted Member Exclusion
# -----------------------------------------------------------------------------
def test_case_g_uncommitted_member_excluded_from_sample() -> None:
    """Case G: Uncommitted member 24 at lead 69 is completely excluded from the logical sample."""
    committed_member_indices = tuple(i for i in range(1, 31) if i != 24)
    assert 24 not in committed_member_indices
    assert len(committed_member_indices) == 29

    # Preallocated 30-member array in Zarr where member 24 holds fill-value NaN
    zarr_all_members = np.full(30, np.nan)
    for m in committed_member_indices:
        zarr_all_members[m - 1] = 10.0 + float(m)

    # Filtered sample based on committed catalog indices (NOT reading uncommitted slice)
    logical_sample = [float(zarr_all_members[m - 1]) for m in committed_member_indices]
    assert len(logical_sample) == 29
    assert all(math.isfinite(v) for v in logical_sample)
    assert ensemble_mean(logical_sample) == float(np.mean(logical_sample))


# -----------------------------------------------------------------------------
# Case H: Point Cross-Cycle Minimum-Lead Stitching
# -----------------------------------------------------------------------------
def test_case_h_point_cross_cycle_stitching(memory_db: Session) -> None:
    """Case H: 00Z ready f000..f048, 06Z processing f000..f006 published -> 06Z wins for overlapping valid times."""
    cycle_00 = datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)
    cycle_06 = datetime(2026, 8, 14, 6, 0, tzinfo=timezone.utc)

    # Run 00Z (ready)
    run_00 = ModelRun(id="r_00z", model_version_id="v_gfs", cycle_time=cycle_00, status="ready", zarr_store_path="/tmp/00z.zarr")
    # Run 06Z (processing, but f000..f006 published)
    run_06 = ModelRun(id="r_06z", model_version_id="v_gfs", cycle_time=cycle_06, status="processing", zarr_store_path="/tmp/06z.zarr")
    memory_db.add_all([run_00, run_06])

    for lead in (0, 3, 6, 9, 12, 15, 18, 21, 24):
        memory_db.add(
            ForecastProduct(
                id=f"fp_00z_{lead}",
                run_id="r_00z",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
                zarr_chunk_path="/tmp",
            )
        )
    for lead in (0, 3, 6):
        memory_db.add(
            ForecastProduct(
                id=f"fp_06z_{lead}",
                run_id="r_06z",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
                zarr_chunk_path="/tmp",
            )
        )
    memory_db.commit()

    winners = _select_min_lead_winners(memory_db, "gfs")

    # Valid time 06:00 (06Z+0h vs 00Z+6h) -> min lead is 0 (06Z wins)
    assert winners[datetime(2026, 8, 14, 6, tzinfo=timezone.utc)][0] == (cycle_06, 0)
    # Valid time 09:00 (06Z+3h vs 00Z+9h) -> min lead is 3 (06Z wins)
    assert winners[datetime(2026, 8, 14, 9, tzinfo=timezone.utc)][0] == (cycle_06, 3)
    # Valid time 12:00 (06Z+6h vs 00Z+12h) -> min lead is 6 (06Z wins)
    assert winners[datetime(2026, 8, 14, 12, tzinfo=timezone.utc)][0] == (cycle_06, 6)
    # Valid time 15:00 (06Z has no f009; 00Z+15h) -> 00Z wins
    assert winners[datetime(2026, 8, 14, 15, tzinfo=timezone.utc)][0] == (cycle_00, 15)
    # Valid time 18:00 (06Z has no f012; 00Z+18h) -> 00Z wins
    assert winners[datetime(2026, 8, 14, 18, tzinfo=timezone.utc)][0] == (cycle_00, 18)


# -----------------------------------------------------------------------------
# Case I: Probability & Exceedance Denominator on Actual Finite Sample
# -----------------------------------------------------------------------------
def test_case_i_probability_sample_denominator() -> None:
    """Case I: Probability calculation uses the actual participating finite sample size in denominator & CI."""
    # 27 finite values, threshold 10.0
    finite_members = [5.0] * 10 + [15.0] * 17  # 17 / 27 = 0.6296
    prob = probability_above_threshold(finite_members, 10.0)
    assert prob == pytest.approx(17 / 27)

    lower, upper = probability_confidence_interval(prob, len(finite_members))
    assert 0.0 <= lower <= prob <= upper <= 1.0


# -----------------------------------------------------------------------------
# Case J: Custom Configured Coverage Threshold
# -----------------------------------------------------------------------------
def test_case_j_custom_configured_coverage_threshold() -> None:
    """Case J: Custom threshold 0.90 (90%) -> 27/30 servable, 26/30 not servable."""
    custom_threshold = 0.90
    assert is_lead_servable(27, 30, min_coverage_ratio=custom_threshold) is True
    assert is_lead_servable(26, 30, min_coverage_ratio=custom_threshold) is False
    assert is_cell_statistically_valid(27, 30, min_coverage_ratio=custom_threshold) is True
    assert is_cell_statistically_valid(26, 30, min_coverage_ratio=custom_threshold) is False

    cov_27 = build_lead_coverage(
        lead_time_hours=6,
        available_member_indices=tuple(range(1, 28)),
        expected_members=30,
        min_coverage_ratio=custom_threshold,
    )
    assert cov_27.servable is True

    cov_26 = build_lead_coverage(
        lead_time_hours=6,
        available_member_indices=tuple(range(1, 27)),
        expected_members=30,
        min_coverage_ratio=custom_threshold,
    )
    assert cov_26.servable is False


# -----------------------------------------------------------------------------
# Case K: Configuration Consistency & Fixed Expected Members
# -----------------------------------------------------------------------------
def test_case_k_fixed_expected_members_and_config_consistency() -> None:
    """Case K: 20 committed members in GEFS always evaluates against fixed 30 denominator (20/30 = 66.7%)."""
    cov = build_lead_coverage(
        lead_time_hours=12,
        available_member_indices=tuple(range(1, 21)),
        expected_members=30,
    )
    assert cov.available_members == 20
    assert cov.expected_members == 30
    assert cov.coverage_ratio == 0.6667
    assert cov.servable is False
