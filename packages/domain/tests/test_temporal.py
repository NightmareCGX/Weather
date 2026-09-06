"""Unit tests for domain.temporal interval classification and lead-0 fallback."""

from __future__ import annotations

from domain.temporal import (
    INTERVAL_LEAD0_FALLBACK_VARIABLES,
    PRECIPITATION_COMPANION_VARIABLES,
    is_precipitation_companion,
    requires_lead0_display_fallback,
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
