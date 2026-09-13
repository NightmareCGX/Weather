"""Unit tests for domain.temporal interval classification and lead-0 fallback."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from domain.temporal import (
    INTERVAL_LEAD0_FALLBACK_VARIABLES,
    PRECIPITATION_COMPANION_VARIABLES,
    is_precipitation_companion,
    requires_lead0_display_fallback,
    serving_start_valid_time,
)


def test_interval_lead0_fallback_variables_set() -> None:
    assert {
        "precipitation_amount_3h",
        "cloud_cover_3h",
    } == INTERVAL_LEAD0_FALLBACK_VARIABLES


def test_precipitation_companion_variables_set() -> None:
    assert {
        "crain",
        "csnow",
        "cfrzr",
        "cicep",
    } == PRECIPITATION_COMPANION_VARIABLES


def test_requires_lead0_display_fallback() -> None:
    assert requires_lead0_display_fallback("precipitation_amount_3h") is True
    assert requires_lead0_display_fallback("cloud_cover_3h") is True
    assert requires_lead0_display_fallback("  PRECIPITATION_AMOUNT_3H  ") is True
    assert requires_lead0_display_fallback("temperature_2m") is False
    assert requires_lead0_display_fallback("wind_10m") is False
    assert requires_lead0_display_fallback("precipitation_rate") is False
    assert requires_lead0_display_fallback(None) is False
    assert requires_lead0_display_fallback("") is False


def test_is_precipitation_companion() -> None:
    assert is_precipitation_companion("crain") is True
    assert is_precipitation_companion("csnow") is True
    assert is_precipitation_companion("cfrzr") is True
    assert is_precipitation_companion("cicep") is True
    assert is_precipitation_companion("  CRAIN  ") is True
    assert is_precipitation_companion("precipitation_amount_3h") is False
    assert is_precipitation_companion("temperature_2m") is False
    assert is_precipitation_companion(None) is False
    assert is_precipitation_companion("") is False


def test_serving_start_valid_time_exact_cadence_boundaries() -> None:
    # 05:59:59Z -> 03:00:00Z
    dt_055959 = datetime(2026, 9, 10, 5, 59, 59, tzinfo=UTC)
    expected_03 = datetime(2026, 9, 10, 3, 0, 0, tzinfo=UTC)
    assert serving_start_valid_time(dt_055959) == expected_03

    # 06:00:00Z -> 06:00:00Z (exact boundary advances immediately)
    dt_060000 = datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC)
    expected_06 = datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC)
    assert serving_start_valid_time(dt_060000) == expected_06

    # 06:00:01Z -> 06:00:00Z
    dt_060001 = datetime(2026, 9, 10, 6, 0, 1, tzinfo=UTC)
    assert serving_start_valid_time(dt_060001) == expected_06

    # 07:00:00Z -> 06:00:00Z
    dt_070000 = datetime(2026, 9, 10, 7, 0, 0, tzinfo=UTC)
    assert serving_start_valid_time(dt_070000) == expected_06

    # 08:59:59Z -> 06:00:00Z
    dt_085959 = datetime(2026, 9, 10, 8, 59, 59, tzinfo=UTC)
    assert serving_start_valid_time(dt_085959) == expected_06

    # 09:00:00Z -> 09:00:00Z (exact boundary advances immediately)
    dt_090000 = datetime(2026, 9, 10, 9, 0, 0, tzinfo=UTC)
    expected_09 = datetime(2026, 9, 10, 9, 0, 0, tzinfo=UTC)
    assert serving_start_valid_time(dt_090000) == expected_09

    # 09:00:01Z -> 09:00:00Z
    dt_090001 = datetime(2026, 9, 10, 9, 0, 1, tzinfo=UTC)
    assert serving_start_valid_time(dt_090001) == expected_09


def test_serving_start_valid_time_day_boundaries_and_cadence() -> None:
    # 00:00:00Z -> 00:00:00Z
    dt_000000 = datetime(2026, 9, 10, 0, 0, 0, tzinfo=UTC)
    assert serving_start_valid_time(dt_000000) == dt_000000

    # 23:59:59Z -> 21:00:00Z
    dt_235959 = datetime(2026, 9, 10, 23, 59, 59, tzinfo=UTC)
    expected_21 = datetime(2026, 9, 10, 21, 0, 0, tzinfo=UTC)
    assert serving_start_valid_time(dt_235959) == expected_21

    # Custom cadence (e.g. 6 hours)
    dt_1130 = datetime(2026, 9, 10, 11, 30, 0, tzinfo=UTC)
    expected_06 = datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC)
    assert serving_start_valid_time(dt_1130, cadence_hours=6) == expected_06

    dt_1200 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=UTC)
    assert serving_start_valid_time(dt_1200, cadence_hours=6) == dt_1200


def test_serving_start_valid_time_timezone_awareness() -> None:
    # Non-UTC timezone (UTC+2): 08:00:00 at UTC+2 is 06:00:00 UTC -> 06:00:00 UTC
    tz_plus_2 = timezone(timedelta(hours=2))
    dt_tz = datetime(2026, 9, 10, 8, 0, 0, tzinfo=tz_plus_2)
    res = serving_start_valid_time(dt_tz)
    assert res == datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC)
    assert res.tzinfo == UTC


def test_serving_start_valid_time_validation_errors() -> None:
    # Naive datetime
    naive_dt = datetime(2026, 9, 10, 6, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        serving_start_valid_time(naive_dt)

    # Invalid cadence
    aware_dt = datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="strictly positive"):
        serving_start_valid_time(aware_dt, cadence_hours=0)
    with pytest.raises(ValueError, match="strictly positive"):
        serving_start_valid_time(aware_dt, cadence_hours=-3)


def test_variable_temporal_metadata_registry() -> None:
    """I19: interval width and reset period are independent per-variable metadata."""
    from domain.temporal import (
        VARIABLE_TEMPORAL_METADATA,
        VariableTemporalMetadata,
        get_variable_temporal_metadata,
    )

    precip = get_variable_temporal_metadata("precipitation_amount_3h")
    cloud = get_variable_temporal_metadata("cloud_cover_3h")
    assert precip == VariableTemporalMetadata(
        interval_width_hours=3, reset_period_hours=6
    )
    assert cloud == VariableTemporalMetadata(
        interval_width_hours=3, reset_period_hours=6
    )
    # Case/whitespace normalization
    assert get_variable_temporal_metadata("  Cloud_Cover_3H ") == cloud
    # Unknown / non-interval variables are rejected loudly
    with pytest.raises(ValueError, match="No temporal metadata registered"):
        get_variable_temporal_metadata("temperature_2m")
    with pytest.raises(ValueError, match="must not be None"):
        get_variable_temporal_metadata(None)  # type: ignore[arg-type]
    assert "precipitation_amount_3h" in VARIABLE_TEMPORAL_METADATA


def test_is_valid_time_protected_boundary_semantics() -> None:
    """I3/I20: protected = {vt >= serving_start(now)}; boundary exit is permanent."""
    from domain.temporal import is_valid_time_protected

    # 07:00Z -> serving_start 06Z: 06Z protected, 03Z exited
    now = datetime(2026, 9, 10, 7, 0, 0, tzinfo=UTC)
    assert is_valid_time_protected(datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC), now)
    assert not is_valid_time_protected(
        datetime(2026, 9, 10, 3, 0, 0, tzinfo=UTC), now
    )

    # Exact cadence boundary advance: at 09:00Z the anchor is 09Z, so 06Z and 03Z
    # have both exited the window. The boundary is monotonically non-decreasing —
    # once a valid_time exits, it never re-enters (permanent exit).
    now_9 = datetime(2026, 9, 10, 9, 0, 0, tzinfo=UTC)
    assert is_valid_time_protected(datetime(2026, 9, 10, 9, 0, 0, tzinfo=UTC), now_9)
    assert not is_valid_time_protected(
        datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC), now_9
    )
    assert not is_valid_time_protected(
        datetime(2026, 9, 10, 3, 0, 0, tzinfo=UTC), now_9
    )

    # Future valid times are protected
    assert is_valid_time_protected(
        datetime(2026, 9, 11, 0, 0, 0, tzinfo=UTC), now
    )

    # Naive valid_time is rejected
    with pytest.raises(ValueError, match="timezone-aware"):
        is_valid_time_protected(datetime(2026, 9, 10, 6, 0, 0), now)


def test_is_valid_time_on_horizon_grid() -> None:
    """I20: grid membership is derived from the model's canonical horizon."""
    from domain.horizon import (
        MODEL_CANONICAL_HORIZONS,
        register_canonical_lead_horizon,
    )
    from domain.temporal import is_valid_time_on_horizon_grid

    # Canonical gfs registry: 3h grid -> 06:00Z aligned; 05:00Z / 06:30Z not.
    assert is_valid_time_on_horizon_grid(
        datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC), model_id="gfs"
    )
    assert not is_valid_time_on_horizon_grid(
        datetime(2026, 9, 10, 5, 0, 0, tzinfo=UTC), model_id="gfs"
    )
    assert not is_valid_time_on_horizon_grid(
        datetime(2026, 9, 10, 6, 30, 0, tzinfo=UTC), model_id="gfs"
    )

    # Unknown model is rejected loudly.
    with pytest.raises(ValueError, match="Unknown model"):
        is_valid_time_on_horizon_grid(
            datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC), model_id="nope"
        )

    # Naive valid_time is rejected.
    with pytest.raises(ValueError, match="timezone-aware"):
        is_valid_time_on_horizon_grid(datetime(2026, 9, 10, 6, 0, 0))

    # Explicit cadence override (no model).
    assert is_valid_time_on_horizon_grid(
        datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC), cadence_hours=6
    )
    assert not is_valid_time_on_horizon_grid(
        datetime(2026, 9, 10, 3, 0, 0, tzinfo=UTC), cadence_hours=6
    )

    # Default canonical cadence (no model, no override).
    assert is_valid_time_on_horizon_grid(
        datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC)
    )
    assert not is_valid_time_on_horizon_grid(
        datetime(2026, 9, 10, 4, 0, 0, tzinfo=UTC)
    )

    # Single-lead horizon falls back to the canonical cadence.
    register_canonical_lead_horizon("test_single_lead", (0,))
    try:
        assert is_valid_time_on_horizon_grid(
            datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC),
            model_id="test_single_lead",
        )
    finally:
        MODEL_CANONICAL_HORIZONS.pop("test_single_lead", None)


def test_is_valid_time_protected_model_grid_membership() -> None:
    """I20: with a model, protection = boundary AND horizon-grid membership."""
    from domain.horizon import (
        MODEL_CANONICAL_HORIZONS,
        register_canonical_lead_horizon,
    )
    from domain.temporal import is_valid_time_protected

    # Hypothetical 6h-grid-only model (leads 0/6/12): 03:00Z is off-grid.
    register_canonical_lead_horizon("test_model_6h", (0, 6, 12))
    try:
        # now = 05:00Z -> serving_start 03Z, so 03:00Z passes the boundary ...
        now = datetime(2026, 9, 10, 5, 0, 0, tzinfo=UTC)
        # ... but is rejected by the model's horizon grid.
        assert not is_valid_time_protected(
            datetime(2026, 9, 10, 3, 0, 0, tzinfo=UTC),
            now,
            model_id="test_model_6h",
        )
        # 06:00Z is inside the window and on the grid -> protected.
        assert is_valid_time_protected(
            datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC),
            now,
            model_id="test_model_6h",
        )
        # Unknown model propagates the registry error.
        with pytest.raises(ValueError, match="Unknown model"):
            is_valid_time_protected(
                datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC),
                now,
                model_id="nope",
            )
    finally:
        MODEL_CANONICAL_HORIZONS.pop("test_model_6h", None)

    # Without a model, legacy boundary-only semantics are preserved: 03:00Z
    # passes the same boundary even though no model grid was consulted.
    assert is_valid_time_protected(
        datetime(2026, 9, 10, 3, 0, 0, tzinfo=UTC), now
    )


def test_protected_valid_times_enumeration() -> None:
    """I20: the enumerable protected window is [serving_start, +max_lead]."""
    from domain.horizon import (
        MODEL_CANONICAL_HORIZONS,
        MODEL_VERSION_HORIZONS,
        register_canonical_lead_horizon,
    )
    from domain.temporal import protected_valid_times

    now = datetime(2026, 9, 10, 7, 0, 0, tzinfo=UTC)
    vts = protected_valid_times("gfs", now)
    # 07:00Z -> serving_start 06Z; gfs max lead 240h at 3h cadence -> 81 instants.
    assert vts[0] == datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC)
    assert len(vts) == 81
    assert vts[-1] == datetime(2026, 9, 20, 6, 0, 0, tzinfo=UTC)
    assert all((b - a) == timedelta(hours=3) for a, b in zip(vts, vts[1:], strict=False))

    # Version-scoped horizon selection. NOTE: register_canonical_lead_horizon
    # also overwrites the model-level horizon, so the finally block must
    # restore both tables or every test file running after this one sees a
    # corrupted gfs registry.
    register_canonical_lead_horizon("gfs", (0,), version_string="v9.9")
    try:
        vts_v = protected_valid_times("gfs", now, version_string="v9.9")
        assert vts_v == (datetime(2026, 9, 10, 6, 0, 0, tzinfo=UTC),)
    finally:
        MODEL_VERSION_HORIZONS.pop(("gfs", "v9.9"), None)
        MODEL_CANONICAL_HORIZONS["gfs"] = tuple(range(0, 241, 3))


def test_model_serving_start_valid_time_per_model_grid() -> None:
    """I20: the boundary is taken on the model's own horizon cadence.

    A model whose registered canonical horizon carries a cadence other than
    the canonical 3h must get its own grid-aligned boundary — a bare
    global-cadence floor would silently diverge from the per-model
    ``is_valid_time_protected`` membership test.
    """
    from domain.horizon import MODEL_CANONICAL_HORIZONS, register_canonical_lead_horizon
    from domain.temporal import (
        is_valid_time_protected,
        model_serving_start_valid_time,
    )

    # Canonical model: 3h cadence -> floor 07:30Z and 08:00Z to 06:00Z.
    now_0730 = datetime(2026, 9, 10, 7, 30, 0, tzinfo=UTC)
    now_0800 = datetime(2026, 9, 10, 8, 0, 0, tzinfo=UTC)
    assert model_serving_start_valid_time("gfs", now_0730) == datetime(
        2026, 9, 10, 6, 0, 0, tzinfo=UTC
    )
    assert model_serving_start_valid_time("gfs", now_0800) == datetime(
        2026, 9, 10, 6, 0, 0, tzinfo=UTC
    )

    # 6h-cadence model: 08:00Z is NOT a boundary instant on its grid.
    register_canonical_lead_horizon("test_model_6h", (0, 6, 12, 18, 24))
    try:
        assert model_serving_start_valid_time("test_model_6h", now_0730) == datetime(
            2026, 9, 10, 6, 0, 0, tzinfo=UTC
        )
        assert model_serving_start_valid_time("test_model_6h", now_0800) == datetime(
            2026, 9, 10, 6, 0, 0, tzinfo=UTC
        )
        now_0900 = datetime(2026, 9, 10, 9, 0, 0, tzinfo=UTC)
        assert model_serving_start_valid_time("test_model_6h", now_0900) == datetime(
            2026, 9, 10, 6, 0, 0, tzinfo=UTC
        )
        now_1200 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=UTC)
        assert model_serving_start_valid_time("test_model_6h", now_1200) == datetime(
            2026, 9, 10, 12, 0, 0, tzinfo=UTC
        )
        # Boundary agrees with the per-model membership test on grid instants.
        vt = datetime(2026, 9, 10, 12, 0, 0, tzinfo=UTC)
        boundary = model_serving_start_valid_time("test_model_6h", now_0900)
        assert is_valid_time_protected(vt, now_0900, model_id="test_model_6h") == (
            vt >= boundary
        )
    finally:
        MODEL_CANONICAL_HORIZONS.pop("test_model_6h", None)

    # Unknown model fails loudly (same registry as canonical_lead_time_hours).
    with pytest.raises(ValueError):
        model_serving_start_valid_time("unregistered_model", now_0730)
