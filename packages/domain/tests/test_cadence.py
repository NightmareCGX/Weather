"""Unit tests for domain.cadence contract."""

from __future__ import annotations

from datetime import timedelta

import pytest
from domain.cadence import (
    DEFAULT_CYCLE_CADENCE_HOURS,
    MODEL_CYCLE_CADENCE_HOURS,
    canonical_cycle_cadence,
    canonical_cycle_cadence_hours,
    register_canonical_cycle_cadence,
)


def test_default_models_registered() -> None:
    """Verify default models GFS and GEFS are registered with 6-hour cadence."""
    assert DEFAULT_CYCLE_CADENCE_HOURS == 6
    assert MODEL_CYCLE_CADENCE_HOURS["gfs"] == 6
    assert MODEL_CYCLE_CADENCE_HOURS["gefs"] == 6
    assert canonical_cycle_cadence_hours("gfs") == 6
    assert canonical_cycle_cadence_hours("gefs") == 6
    assert canonical_cycle_cadence("gfs") == timedelta(hours=6)
    assert canonical_cycle_cadence("gefs") == timedelta(hours=6)


def test_case_and_whitespace_insensitivity() -> None:
    """Verify model identifiers are normalized (lowercased, stripped)."""
    assert canonical_cycle_cadence_hours("  GFS  ") == 6
    assert canonical_cycle_cadence("  GeFs  ") == timedelta(hours=6)


def test_unknown_model_raises_or_defaults() -> None:
    """Verify unknown model raises ValueError unless default_if_unknown is provided."""
    with pytest.raises(ValueError, match="Unknown model identifier 'unknown_model'"):
        canonical_cycle_cadence_hours("unknown_model")

    with pytest.raises(ValueError, match="Unknown model identifier 'unknown_model'"):
        canonical_cycle_cadence("unknown_model")

    # With default provided
    assert canonical_cycle_cadence_hours("unknown_model", default_if_unknown=3) == 3
    assert canonical_cycle_cadence(
        "unknown_model", default_if_unknown=timedelta(hours=3)
    ) == timedelta(hours=3)

    # Invalid default raises ValueError
    with pytest.raises(ValueError, match="Cadence must be positive"):
        canonical_cycle_cadence_hours("unknown_model", default_if_unknown=0)


def test_register_canonical_cycle_cadence() -> None:
    """Verify registration of a new/future model cadence (e.g. 3h or 1h product)."""
    register_canonical_cycle_cadence("future_3h", 3)
    assert canonical_cycle_cadence_hours("future_3h") == 3
    assert canonical_cycle_cadence("future_3h") == timedelta(hours=3)

    register_canonical_cycle_cadence("hrrr_1h", 1)
    assert canonical_cycle_cadence_hours("hrrr_1h") == 1
    assert canonical_cycle_cadence("hrrr_1h") == timedelta(hours=1)


def test_register_invalid_cadence_raises() -> None:
    """Verify invalid (<= 0) cadence values are rejected."""
    with pytest.raises(ValueError, match="must be strictly positive"):
        register_canonical_cycle_cadence("bad_model", 0)

    with pytest.raises(ValueError, match="must be strictly positive"):
        register_canonical_cycle_cadence("bad_model", -6)
