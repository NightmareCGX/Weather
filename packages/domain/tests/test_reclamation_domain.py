"""Unit tests for pure reclamation domain logic (packages/domain/src/domain/reclamation.py).

Enforces 100% test coverage across all branches, helpers, and invariants.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from domain.reclamation import (
    SHARD_SUFFIX,
    TARGET_KIND_AGG,
    VALID_TARGET_KINDS,
    IngestionRegionIdentity,
    PhysicalShardTarget,
    _ensure_utc,
    can_delete_region_marker,
    get_expected_region_variables,
    get_predecessor_lead,
    is_predecessor_dependent_lead,
    is_predecessor_variable,
    is_shard_filename,
    make_aggregate_relative_key,
    make_region_marker_physical_key,
    make_region_marker_relative_key,
    make_shard_filename,
    make_shard_physical_key,
    make_shard_relative_key,
    normalize_member_index,
    parse_shard_filename,
    register_expected_region_variables,
    target_kind_for,
)


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=UTC)


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
    # A container is not a member, so its index is the normalized 0 -- the same convention ``det``
    # uses, and the reason the schema's member_index check admits exactly that pair.
    assert normalize_member_index("agg") == 0
    assert normalize_member_index("AGG", 7) == 0
    with pytest.raises(ValueError, match="Unknown target_kind"):
        normalize_member_index("invalid")


def test_the_aggregate_kind_is_a_deletion_unit_and_never_a_member_shard():
    """``agg`` has a key builder and a kind, and the member grammar must keep refusing its key.

    Store-layout detection, the ingestion reader's reassembly and the API reader all identify
    member shards by the ``shard.<kind>_L####.shard`` grammar. A key that parsed as one would have
    a reader try to reassemble statistic planes as a member field -- a full field of plausible
    numbers rather than a failure -- so the three facts below have to hold together.
    """
    assert TARGET_KIND_AGG in VALID_TARGET_KINDS
    key = make_aggregate_relative_key("temperature_2m", 6)
    assert key == "temperature_2m/shard.agg_L0006.shard"
    # The same key through the generic builder, which is what the GC planner calls.
    assert make_shard_relative_key("temperature_2m", TARGET_KIND_AGG, 6) == key
    assert make_shard_physical_key("s3://store/", "temperature_2m", TARGET_KIND_AGG, 6) == (
        f"s3://store/{key}"
    )
    # And the member grammar still refuses it.
    assert not is_shard_filename(key)
    with pytest.raises(ValueError, match="Unrecognized shard filename"):
        parse_shard_filename(key)


def test_the_aggregate_kind_has_no_region_marker():
    """A marker records an acquired member region; an aggregate is computed, not acquired."""
    with pytest.raises(ValueError, match="no region marker"):
        make_region_marker_relative_key(TARGET_KIND_AGG, 6)


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
    assert pst.cycle_time.tzinfo == UTC
    assert pst.valid_time.tzinfo == UTC
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


def test_target_kind_for_maps_the_acquisition_convention():
    assert target_kind_for(None) == ("det", 0)
    assert target_kind_for(None, is_mean=True) == ("mean", -1)
    assert target_kind_for(3) == ("mem", 3)
    # is_mean wins over a supplied member, matching how the writer treats them
    assert target_kind_for(3, is_mean=True) == ("mean", -1)


def test_make_shard_filename_agrees_with_make_shard_relative_key():
    """The acquisition-convention builder and the target-kind builder must not diverge."""
    cases = (
        (None, 0, False, "temperature_2m", "det", 0),
        (None, 6, True, "temperature_2m", "mean", -1),
        (3, 12, False, "temperature_2m", "mem", 3),
    )
    for member, lead, is_mean, variable, kind, member_index in cases:
        assert make_shard_filename(
            variable, member=member, lead_time_hours=lead, is_mean=is_mean
        ) == make_shard_relative_key(variable, kind, lead, member_index)

    assert (
        make_shard_filename("temperature_2m", member=None, lead_time_hours=0)
        == "temperature_2m/shard.det_L0000.shard"
    )
    assert (
        make_shard_filename("temperature_2m", member=None, lead_time_hours=6, is_mean=True)
        == "temperature_2m/shard.mean_L0006.shard"
    )
    assert (
        make_shard_filename("temperature_2m", member=3, lead_time_hours=6)
        == "temperature_2m/shard.mem003_L0006.shard"
    )


def test_parse_shard_filename_round_trips_every_kind():
    """parse_shard_filename must invert make_shard_filename for every region shape."""
    for member, lead, is_mean in (
        (None, 0, False),
        (None, 6, True),
        (1, 6, False),
        (30, 240, False),
    ):
        key = make_shard_filename(
            "temperature_2m", member=member, lead_time_hours=lead, is_mean=is_mean
        )
        assert parse_shard_filename(key) == (member, lead, is_mean)
        # a bare filename and a prefixed key must parse identically
        assert parse_shard_filename(key.rsplit("/", 1)[-1]) == (member, lead, is_mean)
        assert is_shard_filename(key)


def test_parse_shard_filename_rejects_non_shard_names():
    for bad in (
        "temperature_2m/.zarray",
        "temperature_2m/0.0.0.0",
        "manifest.json",
        "temperature_2m/shard.agg_L0006.shard",  # agg is not a recognized kind yet
        "shard.det_L0006.shard",  # no variable prefix is fine, but...
    ):
        # the last entry IS a valid shard name (prefix is optional); assert per-case
        if bad == "shard.det_L0006.shard":
            assert is_shard_filename(bad)
            continue
        assert not is_shard_filename(bad)

    with pytest.raises(ValueError, match="Not a shard filename"):
        parse_shard_filename("temperature_2m/.zarray")

    with pytest.raises(ValueError, match="Unrecognized shard filename"):
        parse_shard_filename("temperature_2m/shard.weird_L0006.shard")


def test_parse_shard_filename_rejects_malformed_member_name():
    """A truncated member name must not be silently parsed as a different region."""
    assert not is_shard_filename("temperature_2m/shard.mem001.shard")
    assert not is_shard_filename("temperature_2m/shard.mem_L0006.shard")

    with pytest.raises(ValueError, match="Unrecognized shard filename"):
        parse_shard_filename("temperature_2m/shard.mem001.shard")


def test_shard_suffix_constant_is_the_store_layout_marker():
    """Store layout detection keys on this suffix; it must match the emitted keys."""
    assert SHARD_SUFFIX == ".shard"
    for member, is_mean in ((None, False), (None, True), (7, False)):
        key = make_shard_filename(
            "temperature_2m", member=member, lead_time_hours=0, is_mean=is_mean
        )
        assert key.endswith(SHARD_SUFFIX)


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

    # Predecessor leads (defaults: W=3, R=6 — the historical GFS/GEFS configuration)
    assert get_predecessor_lead(6) == 3
    assert get_predecessor_lead(24) == 21
    assert get_predecessor_lead(30) == 27
    assert get_predecessor_lead(36) == 33

    with pytest.raises(ValueError, match="not a reset lead for R=6"):
        get_predecessor_lead(3)

    # (W, R) generalization (architecture doc I19): predecessor is L - W and the
    # reset-lead gate is L % R == 0, both from per-variable metadata.
    assert is_predecessor_dependent_lead(12, reset_period_hours=12) is True
    assert is_predecessor_dependent_lead(6, reset_period_hours=12) is False
    assert get_predecessor_lead(12, interval_width_hours=3, reset_period_hours=6) == 9
    assert get_predecessor_lead(12, interval_width_hours=3, reset_period_hours=12) == 9
    assert get_predecessor_lead(12, interval_width_hours=12, reset_period_hours=12) == 0
    with pytest.raises(ValueError, match="must not exceed reset period"):
        get_predecessor_lead(12, interval_width_hours=6, reset_period_hours=3)
    with pytest.raises(ValueError, match="must be positive"):
        get_predecessor_lead(12, interval_width_hours=0, reset_period_hours=6)

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


