"""Tests for the variable → aggregate-encoding classification (domain/variable_class.py).

Getting a variable's class wrong is the most expensive mistake this area can make: the
near-Gaussian encoding stores a plausible-looking field that simply cannot answer a heavy-
tailed variable's thresholds, and nothing downstream would notice. So the mapping is asserted
against the platform's own authoritative variable list rather than against a copy.
"""

from __future__ import annotations

import pytest
from domain.aggregate import KIND_MEAN_STD_BINS, KIND_QUANTILE_FUNCTION
from domain.reclamation import (
    DEFAULT_EXPECTED_REGION_VARIABLES,
    TARGET_KIND_DET,
    TARGET_KIND_MEAN,
    TARGET_KIND_MEM,
    get_expected_region_variables,
)
from domain.variable_class import (
    APPROVED_BIN_COUNT,
    APPROVED_LEVEL_COUNT,
    CLASS_BOUNDED,
    CLASS_FLAG,
    CLASS_NEAR_GAUSSIAN,
    CLASS_PRODUCT_FIELDS,
    CLASS_ZERO_INFLATED,
    DERIVED_VARIABLES,
    REGISTERED_VARIABLES,
    SPECLESS_CLASSES,
    VALID_CLASSES,
    VariableClassError,
    VariableEncoding,
    encoding_for,
    is_registered,
    register_encoding,
    spec_for,
    unclassified,
)

#: Unregistering is not offered, so a test that registers must use a variable the platform
#: does not store, and re-registering it repeatedly is harmless.
SCRATCH_VARIABLE = "scratch_variable_for_tests"


def test_registry_covers_every_stored_variable() -> None:
    """The platform's authoritative variable list and this registry must agree.

    A variable added to the platform without a classification would otherwise fall through
    to whatever default the writer happened to use.
    """
    for model_id, kind in (
        ("gfs", TARGET_KIND_DET),
        ("gefs", TARGET_KIND_MEAN),
        ("gefs", TARGET_KIND_MEM),
    ):
        expected = get_expected_region_variables(model_id, kind)
        assert unclassified(expected) == frozenset(), (model_id, kind)


def test_registry_has_no_variable_the_platform_does_not_store() -> None:
    """Goes the other way too: a stale entry would mask a renamed variable.

    Reads the pristine shipped snapshot rather than the live registry: the registry is
    mutable, and an unrelated test that registers a version-specific variable schema also
    overwrites the version-less key, so a getter-based assertion would depend on test
    execution order.
    """
    stored: set[str] = set()
    for model_id, kind in (
        ("gfs", TARGET_KIND_DET),
        ("gefs", TARGET_KIND_MEAN),
        ("gefs", TARGET_KIND_MEM),
    ):
        stored |= set(DEFAULT_EXPECTED_REGION_VARIABLES[(model_id, "v1.0", kind)])
    # A derived variable is classified without being stored: wind_10m is built at serve time
    # from its two components, so it has a container but no member shards of its own.
    assert frozenset(stored) >= (REGISTERED_VARIABLES - DERIVED_VARIABLES)
    for variable in DERIVED_VARIABLES:
        assert variable not in stored, variable


def test_approved_counts_are_the_measured_ones() -> None:
    """32 bins and 19 levels are the measured optima, not round numbers."""
    assert APPROVED_BIN_COUNT == 32
    assert APPROVED_LEVEL_COUNT == 19
    for variable in ("temperature_2m", "wind_u_10m", "wind_v_10m"):
        assert spec_for(variable).n_bins == APPROVED_BIN_COUNT
    for variable in ("wind_gust", "cloud_cover_3h", "visibility"):
        assert len(spec_for(variable).levels) == APPROVED_LEVEL_COUNT


def test_near_gaussian_variables_use_the_bin_encoding() -> None:
    for variable in ("temperature_2m", "wind_u_10m", "wind_v_10m"):
        encoding = encoding_for(variable)
        assert encoding.variable_class == CLASS_NEAR_GAUSSIAN
        assert encoding.spec is not None
        assert encoding.spec.kind == KIND_MEAN_STD_BINS
        assert encoding.field_count == 2 + APPROVED_BIN_COUNT


def test_zero_inflated_variables_use_the_quantile_encoding() -> None:
    """The bin encoding's bounded support cannot represent these variables' tails."""
    for variable in (
        "precipitation_rate",
        "precipitation_amount_3h",
        "snow_depth",
        "wind_gust",
    ):
        encoding = encoding_for(variable)
        assert encoding.variable_class == CLASS_ZERO_INFLATED
        assert encoding.spec is not None
        assert encoding.spec.kind == KIND_QUANTILE_FUNCTION


def test_bounded_variables_share_the_quantile_encoding_deliberately() -> None:
    """C is treated as B because a censored endpoint mass is *not* the binding error source.

    A dedicated two-part model for the endpoint mass was built and measured: it was identical
    to the bins on two variables and worse on the other two. Recording that here keeps a
    future reader from re-deriving the model on the assumption it was never tried.
    """
    bounded = (
        "relative_humidity_2m",
        "cloud_cover_3h",
        "cloud_ceiling",
        "visibility",
    )
    for variable in bounded:
        encoding = encoding_for(variable)
        assert encoding.variable_class == CLASS_BOUNDED
        assert encoding.spec is not None
        assert encoding.spec.kind == KIND_QUANTILE_FUNCTION
        assert encoding.field_count == APPROVED_LEVEL_COUNT


def test_flag_variables_carry_no_spec() -> None:
    """A shape over two values carries nothing, so flags store a fraction instead.

    ``field_count`` counts *distribution* fields, and a flag has none: its fraction is a
    product-defined group, which is ``domain.field_layout``'s business rather than the
    distribution's.
    """
    for variable in ("crain", "csnow", "cfrzr", "cicep"):
        encoding = encoding_for(variable)
        assert encoding.variable_class == CLASS_FLAG
        assert encoding.is_flag
        assert encoding.spec is None
        assert not encoding.has_distribution
        assert encoding.field_count == 0


def test_a_derived_variable_carries_no_spec_and_is_not_stored() -> None:
    """``wind_10m`` has a container but no distribution spec and no member shards of its own.

    Everything its products show is a function of each member's ``(u, v)`` pair, and its speed
    histogram is its rose summed over sectors, so there is no distribution for a spec to order.
    """
    encoding = encoding_for("wind_10m")
    assert encoding.variable_class == CLASS_PRODUCT_FIELDS
    assert encoding.spec is None
    assert not encoding.has_distribution
    assert encoding.field_count == 0
    assert frozenset({"wind_10m"}) == DERIVED_VARIABLES


def test_spec_for_rejects_a_flag() -> None:
    with pytest.raises(VariableClassError, match="flag"):
        spec_for("crain")


def test_unknown_variable_is_an_error_not_a_default() -> None:
    """No fallback: a default would store the wrong encoding silently."""
    with pytest.raises(VariableClassError, match="no approved aggregate encoding"):
        encoding_for("not_a_variable")
    assert not is_registered("not_a_variable")
    assert unclassified({"temperature_2m", "not_a_variable"}) == frozenset({"not_a_variable"})


def test_variable_lookup_tolerates_surrounding_whitespace() -> None:
    assert encoding_for("  temperature_2m  ") is encoding_for("temperature_2m")


def test_class_identifiers_are_the_documented_letters() -> None:
    assert {
        CLASS_NEAR_GAUSSIAN,
        CLASS_ZERO_INFLATED,
        CLASS_BOUNDED,
        CLASS_FLAG,
        CLASS_PRODUCT_FIELDS,
    } == VALID_CLASSES
    assert (
        CLASS_NEAR_GAUSSIAN,
        CLASS_ZERO_INFLATED,
        CLASS_BOUNDED,
        CLASS_FLAG,
        CLASS_PRODUCT_FIELDS,
    ) == (
        "A",
        "B",
        "C",
        "D",
        "S",
    )
    # The classes with no distribution spec, asserted by name so adding a fourth has to be
    # deliberate rather than incidental.
    assert frozenset({CLASS_FLAG, CLASS_PRODUCT_FIELDS}) == SPECLESS_CLASSES


def test_encoding_invariants_are_enforced() -> None:
    """A flag carrying a spec, or a non-flag without one, means a mis-specified registration."""
    with pytest.raises(VariableClassError, match="unknown variable class"):
        VariableEncoding(variable="x", variable_class="Z", spec=None)
    with pytest.raises(VariableClassError, match="must not carry an aggregate spec"):
        VariableEncoding(
            variable="x", variable_class=CLASS_FLAG, spec=spec_for("temperature_2m")
        )
    with pytest.raises(VariableClassError, match="requires an aggregate spec"):
        VariableEncoding(variable="x", variable_class=CLASS_NEAR_GAUSSIAN, spec=None)


def test_register_encoding_adds_a_variable_without_unregistering_anything() -> None:
    register_encoding(
        VariableEncoding(
            variable=SCRATCH_VARIABLE,
            variable_class=CLASS_NEAR_GAUSSIAN,
            spec=spec_for("temperature_2m"),
        )
    )
    assert is_registered(SCRATCH_VARIABLE)
    assert unclassified({SCRATCH_VARIABLE}) == frozenset()
    # the approved set is untouched
    assert "temperature_2m" in REGISTERED_VARIABLES or is_registered("temperature_2m")
