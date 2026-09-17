"""Tests for the domain model-registration audit.

The audit exists to separate two failure modes that look identical from the
outside: a missing canonical horizon, which raises on the ingestion and
reclamation paths, and a missing cadence or member count, which degrades
silently to a default. These tests pin that distinction, because collapsing it
would either hide the silent gaps or turn them into needless hard failures.
"""

from __future__ import annotations

from domain.coverage import register_expected_members
from domain.horizon import register_canonical_lead_horizon
from domain.model_registration import audit_model_registration


def test_registered_models_report_clean():
    audit = audit_model_registration(["gfs", "gefs"])

    assert audit.is_clean
    assert not audit.has_fatal
    assert audit.audited_models == ()
    assert audit.describe() == "all audited models are registered"


def test_empty_input_is_clean():
    audit = audit_model_registration([])

    assert audit.is_clean
    assert audit.audited_models == ()


def test_unknown_model_is_missing_from_every_registry():
    audit = audit_model_registration(["aigfs"])

    assert not audit.is_clean
    assert audit.missing_horizon == ("aigfs",)
    assert audit.missing_cadence == ("aigfs",)
    assert audit.missing_expected_members == ("aigfs",)


def test_missing_horizon_is_fatal_but_missing_cadence_alone_is_not():
    # A model with a horizon and member count but no cadence degrades silently:
    # ingestion works, only lag and staleness are computed against a guess.
    register_canonical_lead_horizon("cadence_gap_model", (0, 3, 6))
    register_expected_members("cadence_gap_model", 1)

    audit = audit_model_registration(["cadence_gap_model"])

    assert not audit.has_fatal
    assert audit.missing_horizon == ()
    assert audit.missing_cadence == ("cadence_gap_model",)
    assert audit.missing_expected_members == ()


def test_missing_horizon_alone_is_fatal():
    register_expected_members("horizon_gap_model", 1)

    audit = audit_model_registration(["horizon_gap_model"])

    assert audit.has_fatal
    assert audit.missing_horizon == ("horizon_gap_model",)


def test_ids_are_normalized_and_deduplicated():
    # The registries normalize with lower().strip(); the audit must query them
    # on the same footing, and must not report a model twice.
    audit = audit_model_registration(["  AIGFS  ", "aigfs", "AiGfS"])

    assert audit.missing_horizon == ("aigfs",)
    assert audit.audited_models == ("aigfs",)


def test_registered_models_are_not_reported_even_alongside_unknown_ones():
    audit = audit_model_registration(["gfs", "gefs", "aifs"])

    assert audit.missing_horizon == ("aifs",)
    assert "gfs" not in audit.audited_models
    assert "gefs" not in audit.audited_models


def test_describe_names_the_models_and_flags_the_fatal_group():
    audit = audit_model_registration(["aifs"])
    description = audit.describe()

    assert "aifs" in description
    assert "FATAL" in description
    # Each remedy is per-model, so the report must name the registry to fix.
    assert "horizon" in description
    assert "cadence" in description
    assert "members" in description


def test_audited_models_is_the_union_of_all_gaps():
    register_canonical_lead_horizon("partial_model", (0, 3))
    register_expected_members("partial_model", 1)
    # Cadence is still unregistered for partial_model; nothing is registered for
    # the second model.

    audit = audit_model_registration(["partial_model", "totally_unknown"])

    assert audit.audited_models == ("partial_model", "totally_unknown")
