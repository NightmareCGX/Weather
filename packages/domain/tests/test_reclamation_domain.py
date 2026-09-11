"""Unit tests for pure reclamation domain logic (packages/domain/src/domain/reclamation.py).

Enforces 100% test coverage across all branches, helpers, and invariants.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from domain.reclamation import (
    IngestionRegionIdentity,
    PhysicalShardTarget,
    _ensure_utc,
    can_delete_region_marker,
    get_expected_region_variables,
    get_predecessor_lead,
    is_predecessor_dependent_lead,
    is_predecessor_variable,
    make_region_marker_physical_key,
    make_region_marker_relative_key,
    make_shard_physical_key,
    make_shard_relative_key,
    normalize_member_index,
    register_expected_region_variables,
)


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def test_ensure_utc():
    naive = datetime(2026, 9, 10, 6, 0)
    aware = _dt(2026, 9, 10, 6, 0)
    assert _ensure_utc(naive) == aware
    assert _ensure_utc(aware) == aware


def test_normalize_member_index():
    assert normalize_member_index("det") == 0
    assert normalize_member_index("DET", 5) == 0
    assert normalize_member_index("mean") == -1
    assert normalize_member_index("MEAN", 10) == -1
    assert normalize_member_index("mem", 5) == 5
    assert normalize_member_index("MEM", None) == 1
    with pytest.raises(ValueError, match="Unknown target_kind"):
        normalize_member_index("invalid")


def test_physical_shard_target_invariants():
    pst = PhysicalShardTarget(
        run_id="run_1",
        model_id="GFS",
        cycle_time=datetime(2026, 9, 10, 6),
        lead_time_hours=0,
        variable_code="temperature_2m",
        target_kind="det",
        member_index=99,  # should normalize to 0 for det
        valid_time=datetime(2026, 9, 10, 6),
        store_path="s3://store",
        physical_key="temperature_2m/shard.det_L0000.shard",
    )
    assert pst.model_id == "gfs"
    assert pst.target_kind == "det"
    assert pst.member_index == 0
    assert pst.cycle_time.tzinfo == timezone.utc
    assert pst.valid_time.tzinfo == timezone.utc
    assert pst.target_tuple == ("run_1", 0, "temperature_2m", "det", 0)

    with pytest.raises(ValueError, match="Invalid target_kind"):
        PhysicalShardTarget(
            run_id="run_1",
            model_id="gfs",
            cycle_time=_dt(2026, 9, 10, 6),
            lead_time_hours=0,
            variable_code="temperature_2m",
            target_kind="unknown_kind",
            member_index=0,
            valid_time=_dt(2026, 9, 10, 6),
            store_path="s3://store",
            physical_key="key",
        )


def test_ingestion_region_identity():
    reg_det = IngestionRegionIdentity(
        run_id="run_1",
        lead_time_hours=6,
        target_kind="DET",
        member_index=15,
    )
    assert reg_det.target_kind == "det"
    assert reg_det.member_index == 0

    reg_mem = IngestionRegionIdentity(
        run_id="run_1",
        lead_time_hours=6,
        target_kind="mem",
        member_index=15,
    )
    assert reg_mem.member_index == 15


def test_make_shard_keys():
    # Deterministic
    rel_det = make_shard_relative_key("temperature_2m", "det", 0)
    assert rel_det == "temperature_2m/shard.det_L0000.shard"
    full_det = make_shard_physical_key("s3://store/", "temperature_2m", "det", 0)
    assert full_det == "s3://store/temperature_2m/shard.det_L0000.shard"

    # Mean
    rel_mean = make_shard_relative_key("temperature_2m", "mean", 6)
    assert rel_mean == "temperature_2m/shard.mean_L0006.shard"
    full_mean = make_shard_physical_key("s3://store", "temperature_2m", "mean", 6)
    assert full_mean == "s3://store/temperature_2m/shard.mean_L0006.shard"

    # Member
    rel_mem = make_shard_relative_key("temperature_2m", "mem", 12, member_index=3)
    assert rel_mem == "temperature_2m/shard.mem003_L0012.shard"
    full_mem = make_shard_physical_key("s3://store", "temperature_2m", "mem", 12, member_index=3)
    assert full_mem == "s3://store/temperature_2m/shard.mem003_L0012.shard"

    with pytest.raises(ValueError, match="Unknown target_kind"):
        make_shard_relative_key("var", "invalid", 0)


def test_make_region_marker_keys():
    # Deterministic
    rel_det = make_region_marker_relative_key("det", 0)
    assert rel_det == "__commit__/v1/regions/det_L0000.json"
    full_det = make_region_marker_physical_key("s3://store/", "det", 0)
    assert full_det == "s3://store/__commit__/v1/regions/det_L0000.json"

    # Mean
    rel_mean = make_region_marker_relative_key("mean", 6)
    assert rel_mean == "__commit__/v1/regions/mean_L0006.json"
    full_mean = make_region_marker_physical_key("s3://store", "mean", 6)
    assert full_mean == "s3://store/__commit__/v1/regions/mean_L0006.json"

    # Member
    rel_mem = make_region_marker_relative_key("mem", 12, member_index=7)
    assert rel_mem == "__commit__/v1/regions/mem007_L0012.json"
    full_mem = make_region_marker_physical_key("s3://store", "mem", 12, member_index=7)
    assert full_mem == "s3://store/__commit__/v1/regions/mem007_L0012.json"

    with pytest.raises(ValueError, match="Unknown target_kind"):
        make_region_marker_relative_key("invalid", 0)


def test_predecessor_rules():
    # Reset leads (L > 0 and L % 6 == 0): 6, 12, 18, 24, 30, 36, ...
    assert is_predecessor_dependent_lead(6) is True
    assert is_predecessor_dependent_lead(24) is True
    assert is_predecessor_dependent_lead(30) is True

    # Non-reset leads
    assert is_predecessor_dependent_lead(0) is False
    assert is_predecessor_dependent_lead(3) is False
    assert is_predecessor_dependent_lead(9) is False
    assert is_predecessor_dependent_lead(21) is False

    # Predecessor leads
    assert get_predecessor_lead(6) == 3
    assert get_predecessor_lead(24) == 21
    assert get_predecessor_lead(30) == 27
    assert get_predecessor_lead(36) == 33

    with pytest.raises(ValueError, match="not a 6-hour reset lead"):
        get_predecessor_lead(3)

    # Predecessor variables
    assert is_predecessor_variable("precipitation_amount_3h") is True
    assert is_predecessor_variable("cloud_cover_3h") is True
    assert is_predecessor_variable("temperature_2m") is False
    assert is_predecessor_variable(None) is False


def test_can_delete_region_marker():
    expected = {"var_a", "var_b", "var_c"}

    # Not all deleted
    assert can_delete_region_marker(expected, {"var_a", "var_b"}) is False

    # Exactly all deleted
    assert can_delete_region_marker(expected, {"var_a", "var_b", "var_c"}) is True

    # Superset deleted
    assert can_delete_region_marker(expected, {"var_a", "var_b", "var_c", "var_d"}) is True

    # Empty expected
    assert can_delete_region_marker(set(), {"var_a"}) is False


def test_authoritative_expected_region_variables():
    # GFS deterministic has 15 variables
    gfs_det = get_expected_region_variables("gfs", "det")
    assert len(gfs_det) == 15
    assert "precipitation_rate" in gfs_det

    # GEFS mean has 14 variables (no precipitation_rate)
    gefs_mean = get_expected_region_variables("gefs", "mean")
    assert len(gefs_mean) == 14
    assert "precipitation_rate" not in gefs_mean

    # GEFS member has 14 variables
    gefs_mem = get_expected_region_variables("gefs", "mem")
    assert len(gefs_mem) == 14
    assert "precipitation_rate" not in gefs_mem

    # Unknown model without fallback raises ValueError
    with pytest.raises(ValueError, match="No authoritative variable schema"):
        get_expected_region_variables("unknown_model", "det")

    # Unknown model with fallback returns fallback
    fallback = {"custom_var"}
    res_fallback = get_expected_region_variables("unknown_model", "det", fallback=fallback)
    assert res_fallback == frozenset(fallback)

    # Register custom model schema
    register_expected_region_variables("custom_model", "det", ["var1", "var2"])
    assert get_expected_region_variables("custom_model", "det") == frozenset({"var1", "var2"})

    # Version-specific registration and lookup
    register_expected_region_variables("gfs", "det", ["temperature_2m"], version_string="v2.0")
    gfs_v2 = get_expected_region_variables("gfs", "det", version_string="v2.0")
    assert gfs_v2 == frozenset({"temperature_2m"})
    # v1.0 remains 15 variables!
    assert len(get_expected_region_variables("gfs", "det", version_string="v1.0")) == 15


