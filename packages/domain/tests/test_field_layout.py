"""The stored field layout: one authority for order, scales, roles and missing-member policy.

A layout disagreement between the writer and the reader is not an error at either end -- it is a
mean decoded with a bin's scale, or a bin read from a level's index. These tests pin the facts a
reader depends on, and the refusals that keep a mis-specified variable from being written at all.
"""

from __future__ import annotations

import pytest
from domain.aggregate import KIND_MEAN_STD_BINS, KIND_QUANTILE_FUNCTION
from domain.field_layout import (
    MISSING_SKIP_AT_COVERAGE,
    ROLE_BIN,
    ROLE_LEVEL,
    ROLE_MEAN,
    ROLE_STD,
    FieldLayout,
    FieldLayoutError,
    aggregate_fields_for,
)
from domain.variable_class import APPROVED_BIN_COUNT, APPROVED_LEVEL_COUNT


def test_near_gaussian_layout_is_mean_std_then_bins() -> None:
    layout = aggregate_fields_for("temperature_2m")
    assert layout.kind == KIND_MEAN_STD_BINS
    assert layout.n_fields == 2 + APPROVED_BIN_COUNT
    assert layout.roles[:2] == (ROLE_MEAN, ROLE_STD)
    assert layout.roles[2:] == (ROLE_BIN,) * APPROVED_BIN_COUNT
    assert layout.index_of_role(ROLE_MEAN) == 0
    assert layout.index_of_role(ROLE_STD) == 1
    assert layout.indices_of_role(ROLE_BIN)[0] == 2


def test_a_role_that_more_than_one_field_claims_is_refused_rather_than_guessed() -> None:
    """Asking for "the mean" must not be handed the first of several.

    A bin role legitimately repeats, so this uses a role that must be singular. The point is
    that the refusal is explicit: a reader addressing ``index_of_role(ROLE_MEAN)`` on a layout
    that somehow carries two means gets an error rather than a statistic from the wrong field.
    """
    ambiguous = FieldLayout(
        variable="x",
        kind=KIND_MEAN_STD_BINS,
        field_names=("a", "b"),
        field_scales=(0.01, 0.01),
        roles=(ROLE_MEAN, ROLE_MEAN),
    )
    with pytest.raises(FieldLayoutError, match="2 fields carry role 'mean'"):
        ambiguous.index_of_role(ROLE_MEAN)
    # A repeated role is not itself invalid -- the bins repeat by design.
    asserted = aggregate_fields_for("temperature_2m")
    assert len(asserted.indices_of_role(ROLE_BIN)) > 1
    assert asserted.index_of_role(ROLE_MEAN) == 0


def test_quantile_layout_is_one_level_per_field() -> None:
    layout = aggregate_fields_for("precipitation_amount_3h")
    assert layout.kind == KIND_QUANTILE_FUNCTION
    assert layout.n_fields == APPROVED_LEVEL_COUNT
    assert layout.roles == (ROLE_LEVEL,) * APPROVED_LEVEL_COUNT
    assert layout.index_of_role(ROLE_MEAN) is None


def test_every_field_carries_its_own_scale_because_one_container_holds_several() -> None:
    """The descriptor's single scale field cannot describe a multi-field aggregate.

    A 314 K mean quantised at the bins' 0.001 step would need 314000, which overflows int16.
    The layout is what makes a per-field step possible, and the values are asserted rather than
    described so a spec change cannot silently move a mean onto a probability's step.
    """
    layout = aggregate_fields_for("temperature_2m")
    mean_scale = layout.field_scales[layout.index_of_role(ROLE_MEAN) or 0]
    bin_scale = layout.field_scales[layout.indices_of_role(ROLE_BIN)[0]]
    assert bin_scale < mean_scale, (bin_scale, mean_scale)
    assert 314.0 / mean_scale < 32767, "a mean of 314 K must fit int16 at its own scale"
    assert 314.0 / bin_scale > 32767, "...and must NOT fit at the bins' finer scale"


def test_a_flag_has_no_field_vector() -> None:
    """A 0/1 flag's approved representation is a per-cell fraction, not an ``AggregateSpec``."""
    with pytest.raises(FieldLayoutError, match="no aggregate spec"):
        aggregate_fields_for("crain")


def test_an_unclassified_variable_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(FieldLayoutError, match="no approved aggregate encoding"):
        aggregate_fields_for("mystery_variable")


def test_layouts_are_stable_across_calls() -> None:
    """Called on both tiers, so it must be a pure function of the variable."""
    assert aggregate_fields_for("wind_u_10m") == aggregate_fields_for("wind_u_10m")
    assert aggregate_fields_for(" temperature_2m ") == aggregate_fields_for("temperature_2m")


def test_layout_rejects_a_role_count_mismatch() -> None:
    with pytest.raises(FieldLayoutError, match="disagree in length"):
        FieldLayout(
            variable="x",
            kind=KIND_MEAN_STD_BINS,
            field_names=("a", "b"),
            field_scales=(0.01,),
            roles=(ROLE_MEAN, ROLE_STD),
        )


def test_layout_rejects_an_empty_field_set_and_an_unknown_policy() -> None:
    with pytest.raises(FieldLayoutError, match="at least one field"):
        FieldLayout(
            variable="x", kind=KIND_MEAN_STD_BINS, field_names=(), field_scales=(), roles=()
        )
    with pytest.raises(FieldLayoutError, match="unknown missing-member policy"):
        FieldLayout(
            variable="x",
            kind=KIND_MEAN_STD_BINS,
            field_names=("a",),
            field_scales=(0.01,),
            roles=(ROLE_MEAN,),
            missing_policy="invented",
        )


def test_missing_member_policy_is_uniform_and_matches_the_serving_paths() -> None:
    """Every class skips non-finite members at coverage, which is what the API already does.

    The member reader filters to finite members for *every* variable -- `ensemble_data.py`'s
    generic branch as much as the censored ones -- and then applies the per-cell coverage rule.
    An aggregate that instead propagated NaN would be stricter than the path it replaces, and
    would refuse cells the store can answer.
    """
    for variable in ("temperature_2m", "precipitation_amount_3h", "cloud_ceiling"):
        assert aggregate_fields_for(variable).missing_policy == MISSING_SKIP_AT_COVERAGE
