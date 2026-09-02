"""Tests for the canonical forecast-lead horizon contract."""

from __future__ import annotations

import pytest
from domain.horizon import (
    CANONICAL_LEAD_CADENCE_HOURS,
    CANONICAL_MAX_LEAD_HOURS,
    MODEL_CANONICAL_HORIZONS,
    canonical_lead_time_hours,
    register_canonical_lead_horizon,
)


@pytest.fixture()
def _restore_horizons():
    """Preserve and restore the horizon registry around each test."""
    saved = dict(MODEL_CANONICAL_HORIZONS)
    yield
    MODEL_CANONICAL_HORIZONS.clear()
    MODEL_CANONICAL_HORIZONS.update(saved)


def test_canonical_horizon_is_81_leads_0_to_240_at_3h() -> None:
    for model in ("gfs", "gefs"):
        leads = canonical_lead_time_hours(model)
        assert len(leads) == 81
        assert leads[0] == 0
        assert leads[-1] == CANONICAL_MAX_LEAD_HOURS == 240
        assert all(
            b - a == CANONICAL_LEAD_CADENCE_HOURS == 3 for a, b in zip(leads, leads[1:])
        )
        assert leads == tuple(range(0, 241, 3))


def test_canonical_horizon_registry_covers_both_contract_models() -> None:
    assert set(MODEL_CANONICAL_HORIZONS) == {"gfs", "gefs"}
    assert MODEL_CANONICAL_HORIZONS["gfs"] == MODEL_CANONICAL_HORIZONS["gefs"]


def test_unknown_model_raises_without_default() -> None:
    with pytest.raises(ValueError, match="Unknown model identifier"):
        canonical_lead_time_hours("ecmwf")


def test_unknown_model_with_default_returns_default() -> None:
    fallback = (0, 6, 12)
    assert canonical_lead_time_hours("ecmwf", default_if_unknown=fallback) == fallback


def test_model_id_normalization() -> None:
    assert canonical_lead_time_hours(" GFS ") == canonical_lead_time_hours("gfs")


def test_register_reduced_horizon_override(_restore_horizons) -> None:
    reduced = (0, 3, 6, 9)
    register_canonical_lead_horizon("gfs", reduced)
    assert canonical_lead_time_hours("gfs") == reduced
    # Other models are untouched.
    assert canonical_lead_time_hours("gefs") != reduced


def test_register_rejects_empty_sequence(_restore_horizons) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        register_canonical_lead_horizon("gfs", ())


def test_register_rejects_negative_leads(_restore_horizons) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        register_canonical_lead_horizon("gfs", (0, 3, -6))


def test_register_rejects_non_increasing_sequence(_restore_horizons) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        register_canonical_lead_horizon("gfs", (0, 3, 3, 6))
    with pytest.raises(ValueError, match="strictly increasing"):
        register_canonical_lead_horizon("gfs", (6, 3, 0))


def test_register_normalizes_model_id(_restore_horizons) -> None:
    register_canonical_lead_horizon(" GEFS ", (0, 12))
    assert canonical_lead_time_hours("gefs") == (0, 12)
