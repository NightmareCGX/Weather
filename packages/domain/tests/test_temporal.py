"""Unit tests for domain.temporal interval classification and lead-0 fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    dt_055959 = datetime(2026, 9, 10, 5, 59, 59, tzinfo=timezone.utc)
    expected_03 = datetime(2026, 9, 10, 3, 0, 0, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_055959) == expected_03

    # 06:00:00Z -> 06:00:00Z (exact boundary advances immediately)
    dt_060000 = datetime(2026, 9, 10, 6, 0, 0, tzinfo=timezone.utc)
    expected_06 = datetime(2026, 9, 10, 6, 0, 0, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_060000) == expected_06

    # 06:00:01Z -> 06:00:00Z
    dt_060001 = datetime(2026, 9, 10, 6, 0, 1, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_060001) == expected_06

    # 07:00:00Z -> 06:00:00Z
    dt_070000 = datetime(2026, 9, 10, 7, 0, 0, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_070000) == expected_06

    # 08:59:59Z -> 06:00:00Z
    dt_085959 = datetime(2026, 9, 10, 8, 59, 59, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_085959) == expected_06

    # 09:00:00Z -> 09:00:00Z (exact boundary advances immediately)
    dt_090000 = datetime(2026, 9, 10, 9, 0, 0, tzinfo=timezone.utc)
    expected_09 = datetime(2026, 9, 10, 9, 0, 0, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_090000) == expected_09

    # 09:00:01Z -> 09:00:00Z
    dt_090001 = datetime(2026, 9, 10, 9, 0, 1, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_090001) == expected_09


def test_serving_start_valid_time_day_boundaries_and_cadence() -> None:
    # 00:00:00Z -> 00:00:00Z
    dt_000000 = datetime(2026, 9, 10, 0, 0, 0, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_000000) == dt_000000

    # 23:59:59Z -> 21:00:00Z
    dt_235959 = datetime(2026, 9, 10, 23, 59, 59, tzinfo=timezone.utc)
    expected_21 = datetime(2026, 9, 10, 21, 0, 0, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_235959) == expected_21

    # Custom cadence (e.g. 6 hours)
    dt_1130 = datetime(2026, 9, 10, 11, 30, 0, tzinfo=timezone.utc)
    expected_06 = datetime(2026, 9, 10, 6, 0, 0, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_1130, cadence_hours=6) == expected_06

    dt_1200 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    assert serving_start_valid_time(dt_1200, cadence_hours=6) == dt_1200


def test_serving_start_valid_time_timezone_awareness() -> None:
    # Non-UTC timezone (UTC+2): 08:00:00 at UTC+2 is 06:00:00 UTC -> 06:00:00 UTC
    tz_plus_2 = timezone(timedelta(hours=2))
    dt_tz = datetime(2026, 9, 10, 8, 0, 0, tzinfo=tz_plus_2)
    res = serving_start_valid_time(dt_tz)
    assert res == datetime(2026, 9, 10, 6, 0, 0, tzinfo=timezone.utc)
    assert res.tzinfo == timezone.utc


def test_serving_start_valid_time_validation_errors() -> None:
    # Naive datetime
    naive_dt = datetime(2026, 9, 10, 6, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        serving_start_valid_time(naive_dt)

    # Invalid cadence
    aware_dt = datetime(2026, 9, 10, 6, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="strictly positive"):
        serving_start_valid_time(aware_dt, cadence_hours=0)
    with pytest.raises(ValueError, match="strictly positive"):
        serving_start_valid_time(aware_dt, cadence_hours=-3)
