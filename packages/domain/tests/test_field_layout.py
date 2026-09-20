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
    ROLE_FRACTION,
    ROLE_LEVEL,
    ROLE_MEAN,
    ROLE_MEMBER_COUNT,
    ROLE_PHASE_CURRENT,
    ROLE_PHASE_PREVIOUS,
    ROLE_STD,
    ROLE_TRANSITION,
    FieldLayout,
    FieldLayoutError,
    aggregate_fields_for,
    group_field_names,
    group_field_scales,
)
from domain.variable_class import APPROVED_BIN_COUNT, APPROVED_LEVEL_COUNT


def test_the_member_count_leads_every_layout() -> None:
    """Field 0 answers "is this cell aggregated, and over how many members" before any statistic.

    Both questions come before a mean or a percentile means anything, so the count is at a
    fixed index rather than appended after a variable-length field list.
    """
    for variable in ("temperature_2m", "precipitation_amount_3h", "visibility"):
        layout = aggregate_fields_for(variable)
        assert layout.roles[0] == ROLE_MEMBER_COUNT
        assert layout.field_names[0] == "MEMBER_COUNT"
        assert layout.index_of_role(ROLE_MEMBER_COUNT) == 0
        # One member per step, so the count survives the fixed-point round trip exactly.
        assert layout.field_scales[0] == 1.0


def test_near_gaussian_layout_is_count_then_mean_std_then_bins() -> None:
    layout = aggregate_fields_for("temperature_2m")
    assert layout.kind == KIND_MEAN_STD_BINS
    assert layout.n_fields == 1 + 2 + APPROVED_BIN_COUNT
    assert layout.roles[1:3] == (ROLE_MEAN, ROLE_STD)
    assert layout.roles[3:] == (ROLE_BIN,) * APPROVED_BIN_COUNT
    assert layout.index_of_role(ROLE_MEAN) == 1
    assert layout.index_of_role(ROLE_STD) == 2
    assert layout.indices_of_role(ROLE_BIN)[0] == 3


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
    assert asserted.index_of_role(ROLE_MEAN) == 1


def test_quantile_layout_is_one_level_per_field_then_its_groups() -> None:
    """A variable's distribution fields come first, then whatever groups its products need.

    Precipitation carries the amount's distribution and then its phase and transition groups,
    because its products are functions of the per-member phase rather than of the distribution.
    """
    layout = aggregate_fields_for("precipitation_amount_3h")
    assert layout.kind == KIND_QUANTILE_FUNCTION
    assert layout.groups == ("phase", "transition")
    assert layout.roles[1 : 1 + APPROVED_LEVEL_COUNT] == (ROLE_LEVEL,) * APPROVED_LEVEL_COUNT
    assert layout.n_fields == 1 + APPROVED_LEVEL_COUNT + 12 + 20
    assert layout.index_of_role(ROLE_MEAN) is None
    # The two phase groups are distinguishable: a consumer asking for the current interval's
    # phases must not be handed the predecessor's.
    assert len(layout.indices_of_role(ROLE_PHASE_CURRENT)) == 6
    assert len(layout.indices_of_role(ROLE_PHASE_PREVIOUS)) == 6
    assert len(layout.indices_of_role(ROLE_TRANSITION)) == 20


def test_a_variable_distribution_is_locatable_next_to_its_groups() -> None:
    """Group offsets are derived from the group sizes, so a group can be read without counting."""
    layout = aggregate_fields_for("precipitation_amount_3h")
    phase = layout.group_slice("phase")
    transition = layout.group_slice("transition")
    assert phase.stop == transition.start
    assert transition.stop == layout.n_fields
    assert layout.roles[phase.start] == ROLE_PHASE_CURRENT
    assert layout.roles[transition.start] == ROLE_TRANSITION
    # A group the variable does not carry is a specification mismatch, not a missing value.
    with pytest.raises(FieldLayoutError, match="carries no 'rose' fields"):
        layout.group_slice("rose")


def test_a_flag_is_its_fraction() -> None:
    """A 0/1 flag's shape carries nothing, so its fraction *is* its representation.

    This is the one case where a container has no distribution fields at all: reading it means
    reading the fraction, and the member count says how many members that fraction is over.
    """
    layout = aggregate_fields_for("crain")
    assert layout.kind == ""
    assert layout.groups == ("fraction",)
    assert layout.n_fields == 2
    assert layout.roles == (ROLE_MEMBER_COUNT, ROLE_FRACTION)
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


def test_the_special_variables_carry_the_groups_their_products_need() -> None:
    """Every one of the four has a container, which is what makes the reclamation rule uniform.

    A variable with no container would make "the aggregate exists" unusable as the condition for
    reclaiming its member shards, and would need a hand-written exception in the planner.
    """
    # wind_10m has no distribution fields: its speed histogram is its rose summed over sectors.
    assert aggregate_fields_for("wind_10m").groups == ("rose",)
    assert aggregate_fields_for("wind_10m").n_fields == 1 + 64 + 4
    assert aggregate_fields_for("wind_10m").kind == ""
    # The cloud variables carry their distribution (19 levels), the counts taken before
    # summarising, and the statistics computed over the finite subset.
    for variable in ("cloud_ceiling", "cloud_cover_3h"):
        layout = aggregate_fields_for(variable)
        assert layout.groups == ("censoring", "conditional"), variable
        assert layout.n_fields == 1 + 19 + 3 + 7, variable
        assert layout.group_slice("censoring").stop == layout.group_slice("conditional").start
    for flag in ("crain", "csnow", "cfrzr", "cicep"):
        assert aggregate_fields_for(flag).groups == ("fraction",)
        assert aggregate_fields_for(flag).n_fields == 2


def test_group_metadata_is_available_without_a_layout() -> None:
    """A caller sizing or labelling a group should not have to build a variable's layout."""
    assert len(group_field_names("rose")) == 68
    assert len(group_field_names("phase")) == 12
    assert len(group_field_names("transition")) == 20
    assert len(group_field_scales("censoring")) == 3
    assert set(group_field_scales("censoring")) == {1.0}
    with pytest.raises(FieldLayoutError, match="unknown field group"):
        group_field_names("invented")


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


def test_a_specless_variable_with_no_group_is_refused() -> None:
    """Class D and S carry no spec, so their container must come from a registered group.

    Registering such a variable without a group would otherwise produce a layout with only the
    member count, which no reader could interpret as anything.
    """
    from domain.variable_class import (
        CLASS_PRODUCT_FIELDS,
        VariableEncoding,
        register_encoding,
    )

    variable = "specless_without_a_group"
    register_encoding(
        VariableEncoding(variable=variable, variable_class=CLASS_PRODUCT_FIELDS, spec=None)
    )
    with pytest.raises(FieldLayoutError, match="no fraction group registered"):
        aggregate_fields_for(variable)


def test_an_unknown_group_role_is_refused_rather_than_defaulted() -> None:
    """A group added to the map without a role would decode with fields nobody can address."""
    from domain.field_layout import _group_roles

    with pytest.raises(FieldLayoutError, match="unknown field group"):
        _group_roles("invented")


def test_the_distribution_slice_does_not_depend_on_how_many_groups_follow() -> None:
    """Reading the distribution by position is what keeps a group addition from shifting it.

    A role lookup cannot serve this: ``quantile_function`` has nineteen fields carrying
    ``ROLE_LEVEL``, so "the field with that role" has nineteen answers. The slice is derived from
    the group sizes instead.
    """
    # temperature has no groups, precipitation two, a flag none and no distribution either.
    temp = aggregate_fields_for("temperature_2m").distribution_slice
    assert (temp.start, temp.stop) == (1, 35)
    precip = aggregate_fields_for("precipitation_amount_3h").distribution_slice
    assert (precip.start, precip.stop) == (1, 20)
    assert precip.stop == aggregate_fields_for("precipitation_amount_3h").group_slice("phase").start
    # A flag's distribution is empty: its fraction is its representation.
    flag = aggregate_fields_for("crain").distribution_slice
    assert flag.start == flag.stop
    # ...and so is a product-fields variable's, whose products are its rose.
    wind = aggregate_fields_for("wind_10m").distribution_slice
    assert wind.start == wind.stop


def test_a_group_declares_which_member_variables_it_reads() -> None:
    """The inputs are part of the group's definition, so the writer and the planner agree.

    A publication has to know every member set a container depends on before it can build one;
    for most variables that is the variable itself, but the phase and transition fields are
    functions of the four flags and the rose is a function of the two wind components.
    """
    from domain.field_layout import group_inputs, needs_predecessor, required_member_variables

    assert group_inputs("rose") == (("wind_u_10m", "wind_v_10m"), False)
    assert group_inputs("phase")[0] == ("crain", "csnow", "cfrzr", "cicep")
    assert group_inputs("phase")[1] is True
    with pytest.raises(FieldLayoutError, match="unknown field group"):
        group_inputs("invented")

    # The variable's own members are always a further input.
    assert required_member_variables("temperature_2m") == ("temperature_2m",)
    assert required_member_variables("wind_10m") == (
        "wind_10m",
        "wind_u_10m",
        "wind_v_10m",
    )
    assert required_member_variables("precipitation_amount_3h") == (
        "precipitation_amount_3h",
        "crain",
        "csnow",
        "cfrzr",
        "cicep",
    )
    assert required_member_variables("crain") == ("crain",)

    # Only the precipitation groups read the predecessor interval.
    assert needs_predecessor("precipitation_amount_3h") is True
    assert needs_predecessor("temperature_2m") is False
    assert needs_predecessor("cloud_ceiling") is False
