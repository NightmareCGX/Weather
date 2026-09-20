"""Unit tests for domain.supersession: the member-reclamation predicate.

The predicate decides whether thirty member shards may be deleted because one aggregate container
replaces them. Every branch is exercised here, and so is the fail-closed direction each of them
takes, because the cost of a wrong "yes" is a cycle whose members exist nowhere.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from domain.supersession import (
    REASON_AUTHORIZED,
    REASON_DISABLED,
    REASON_EVIDENCE_MISSING,
    REASON_NOT_SUPERSEDABLE,
    REASON_PREDECESSOR_WINDOW,
    REASON_READER_EVIDENCE_MISSING,
    SUPERSEDABLE_VARIABLES,
    SUPERSESSION_ENV_VAR,
    decide_member_reclamation,
    is_supersedable,
    is_supersession_enabled,
    predecessor_interval_dependent_lead,
    reset_supersession_enabled,
    same_lead_reader_variables,
    set_supersession_enabled,
)
from domain.temporal import VARIABLE_TEMPORAL_METADATA, VariableTemporalMetadata
from domain.variable_class import DERIVED_VARIABLES, REGISTERED_VARIABLES


@pytest.fixture(autouse=True)
def clean_supersession_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(SUPERSESSION_ENV_VAR, raising=False)
    reset_supersession_enabled()
    yield
    monkeypatch.delenv(SUPERSESSION_ENV_VAR, raising=False)
    reset_supersession_enabled()


def test_the_supersedable_set_is_derived_from_the_classification() -> None:
    """The set is not a second list of "converted variables" kept in step by hand."""
    assert SUPERSEDABLE_VARIABLES == REGISTERED_VARIABLES - DERIVED_VARIABLES
    # The derived variable has no member shards of its own, so it can never be superseded.
    assert "wind_10m" not in SUPERSEDABLE_VARIABLES
    assert not is_supersedable("wind_10m")
    assert is_supersedable("temperature_2m")
    # An unclassified variable is retained: no container exists that could answer for it.
    assert not is_supersedable("not_a_variable")
    assert not is_supersedable(None)


def test_the_switch_defaults_to_off_and_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SUPERSESSION_ENV_VAR, raising=False)
    reset_supersession_enabled()
    assert is_supersession_enabled() is False

    for raw in ("1", "true", "TRUE", " yes ", "on"):
        monkeypatch.setenv(SUPERSESSION_ENV_VAR, raw)
        reset_supersession_enabled()
        assert is_supersession_enabled() is True, raw

    for raw in ("0", "false", "no", "off", ""):
        monkeypatch.setenv(SUPERSESSION_ENV_VAR, raw)
        reset_supersession_enabled()
        assert is_supersession_enabled() is False, raw

    set_supersession_enabled(True)
    assert is_supersession_enabled() is True
    reset_supersession_enabled()
    assert is_supersession_enabled() is False


def test_a_predecessor_variable_reports_the_lead_that_reads_it() -> None:
    """``(W, R)`` come from the registry: W=3 and R=6 make lead 3 the predecessor of lead 6."""
    assert predecessor_interval_dependent_lead("precipitation_amount_3h", 3) == 6
    assert predecessor_interval_dependent_lead("cloud_cover_3h", 3) == 6
    # A lead that is not a predecessor of any reset lead has no dependent.
    assert predecessor_interval_dependent_lead("precipitation_amount_3h", 4) is None
    # A non-interval variable carries no predecessor semantics at all.
    assert predecessor_interval_dependent_lead("temperature_2m", 3) is None
    # Lead 0 is the analysis and is nobody's predecessor.
    assert predecessor_interval_dependent_lead("precipitation_amount_3h", 0) is None
    # An unregistered variable is answered rather than raised: the caller is a bulk pass over
    # catalog rows, and one unknown code must not stop the pass.
    assert predecessor_interval_dependent_lead("not_a_variable", 3) is None


def test_a_variable_with_no_interval_semantics_is_not_a_predecessor_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two guards that the registered pair cannot reach, both answered rather than raised.

    The first is reachable only by a variable that is in ``PREDECESSOR_VARIABLES`` without
    temporal metadata; the second only by ``W >= R``, which the registered pair (W=3, R=6) does
    not have. Both are registered here rather than deleted, because the metadata is a registry a
    new interval variable enters -- and a variable that enters it without metadata, or with
    ``W == R``, must not silently acquire a predecessor window it does not have.
    """
    monkeypatch.delitem(
        VARIABLE_TEMPORAL_METADATA, "precipitation_amount_3h", raising=False
    )
    assert predecessor_interval_dependent_lead("precipitation_amount_3h", 3) is None

    monkeypatch.setitem(
        VARIABLE_TEMPORAL_METADATA,
        "precipitation_amount_3h",
        VariableTemporalMetadata(interval_width_hours=6, reset_period_hours=6),
    )
    assert predecessor_interval_dependent_lead("precipitation_amount_3h", 3) is None

    monkeypatch.setitem(
        VARIABLE_TEMPORAL_METADATA,
        "precipitation_amount_3h",
        VariableTemporalMetadata(interval_width_hours=0, reset_period_hours=6),
    )
    assert predecessor_interval_dependent_lead("precipitation_amount_3h", 3) is None


def test_the_predicate_fails_closed_on_every_dependency() -> None:
    """Each condition is checked before the switch, and each retains the members."""
    open_predecessor = decide_member_reclamation(
        "precipitation_amount_3h",
        3,
        aggregate_evidence_ok=True,
        committed_leads=(3,),
        enabled=True,
    )
    assert open_predecessor.authorized is False
    assert open_predecessor.reason == REASON_PREDECESSOR_WINDOW

    missing_evidence = decide_member_reclamation(
        "temperature_2m",
        6,
        aggregate_evidence_ok=False,
        committed_leads=(6,),
        enabled=True,
    )
    assert missing_evidence.authorized is False
    assert missing_evidence.reason == REASON_EVIDENCE_MISSING

    not_supersedable = decide_member_reclamation(
        "wind_10m",
        6,
        aggregate_evidence_ok=True,
        committed_leads=(6,),
        enabled=True,
    )
    assert not_supersedable.authorized is False
    assert not_supersedable.reason == REASON_NOT_SUPERSEDABLE


def test_the_evidence_and_the_window_outrank_the_switch() -> None:
    """With the switch off the reason still names the *next* obstacle, not "disabled" blindly.

    A deployment bringing the path up reads these reasons to see what a flip of the switch would
    actually do: a target blocked by its own predecessor window would stay blocked.
    """
    blocked_by_window = decide_member_reclamation(
        "cloud_cover_3h",
        3,
        aggregate_evidence_ok=True,
        committed_leads=(3,),
        enabled=False,
    )
    assert blocked_by_window.reason == REASON_PREDECESSOR_WINDOW

    blocked_by_evidence = decide_member_reclamation(
        "temperature_2m",
        6,
        aggregate_evidence_ok=False,
        committed_leads=(6,),
        enabled=False,
    )
    assert blocked_by_evidence.reason == REASON_EVIDENCE_MISSING

    only_the_switch = decide_member_reclamation(
        "temperature_2m",
        6,
        aggregate_evidence_ok=True,
        committed_leads=(6,),
        enabled=False,
    )
    assert only_the_switch.authorized is False
    assert only_the_switch.reason == REASON_DISABLED


def test_everything_in_order_authorizes_the_deletion() -> None:
    decision = decide_member_reclamation(
        "temperature_2m",
        6,
        aggregate_evidence_ok=True,
        committed_leads=(3, 6),
        enabled=True,
    )
    assert decision.authorized is True
    assert decision.reason == REASON_AUTHORIZED
    assert decision.variable == "temperature_2m"
    assert decision.lead_time_hours == 6


def test_a_predecessor_variable_is_authorized_once_its_dependent_lead_is_committed() -> None:
    """The window closes when the lead that reads the interval has its members committed."""
    decision = decide_member_reclamation(
        "precipitation_amount_3h",
        3,
        aggregate_evidence_ok=True,
        committed_leads=(3, 6),
        enabled=True,
    )
    assert decision.authorized is True
    assert decision.reason == REASON_AUTHORIZED


def test_a_dependent_lead_beyond_the_horizon_closes_the_window() -> None:
    """The last lead a cycle fills has no successor that reads it, so nothing waits for it.

    This is the "or L is the last lead depending on it" half of the rule: without it a
    predecessor variable's members at the final reset lead of every cycle would be retained
    forever, waiting for a lead the run does not have.
    """
    assert predecessor_interval_dependent_lead(
        "precipitation_amount_3h", 237, max_lead=240
    ) == 240
    assert predecessor_interval_dependent_lead(
        "precipitation_amount_3h", 237, max_lead=237
    ) is None
    assert (
        decide_member_reclamation(
            "precipitation_amount_3h",
            237,
            aggregate_evidence_ok=True,
            committed_leads=(237,),
            max_lead=237,
            enabled=True,
        ).authorized
        is True
    )
    assert (
        decide_member_reclamation(
            "precipitation_amount_3h",
            237,
            aggregate_evidence_ok=True,
            committed_leads=(237,),
            max_lead=240,
            enabled=True,
        ).reason
        == REASON_PREDECESSOR_WINDOW
    )


def test_the_same_lead_readers_are_derived_from_the_layout_declaration() -> None:
    """A container is not a function of one variable's members, and the set says which reads which.

    ``field_layout.required_member_variables`` is the declaration the writer itself consults, so
    the reader relation is derived from it rather than listed. The four flags and the two wind
    components are the cases where deleting one variable's members would break another's
    container.
    """
    assert same_lead_reader_variables("wind_u_10m") == frozenset({"wind_10m"})
    assert same_lead_reader_variables("wind_v_10m") == frozenset({"wind_10m"})
    assert same_lead_reader_variables("crain") == frozenset(
        {"precipitation_amount_3h", "precipitation_rate"}
    )
    # The variable itself is excluded: its own container reading its own members is what
    # ``aggregate_evidence_ok`` already checks.
    assert "temperature_2m" not in same_lead_reader_variables("temperature_2m")
    assert same_lead_reader_variables("temperature_2m") == frozenset()
    # A derived variable is nobody's input, since it has no members of its own.
    assert same_lead_reader_variables("wind_10m") == frozenset()


def test_a_reader_without_its_own_container_retains_the_members_it_reads() -> None:
    """The live case: a ``wind_10m`` publication refreshes before ``wind_u_10m`` has a container.

    Reclaiming ``wind_u_10m``'s members then would leave the rose with no input, and every later
    patch of ``wind_10m`` would refuse to build -- so the members are retained until the reader
    that consumes them has its own aggregate.
    """
    pending_reader = decide_member_reclamation(
        "wind_u_10m",
        6,
        aggregate_evidence_ok=True,
        committed_leads=(6,),
        reader_variables=("wind_10m",),
        reader_evidence_ok=False,
        enabled=True,
    )
    assert pending_reader.authorized is False
    assert pending_reader.reason == REASON_READER_EVIDENCE_MISSING

    ready_reader = decide_member_reclamation(
        "wind_u_10m",
        6,
        aggregate_evidence_ok=True,
        committed_leads=(6,),
        reader_variables=("wind_10m",),
        reader_evidence_ok=True,
        enabled=True,
    )
    assert ready_reader.authorized is True

    # With no readers the flag is not consulted at all, so a caller that does not compute the set
    # is not silently blocked by a default.
    no_readers = decide_member_reclamation(
        "temperature_2m",
        6,
        aggregate_evidence_ok=True,
        committed_leads=(6,),
        reader_evidence_ok=False,
        enabled=True,
    )
    assert no_readers.authorized is True
    """``enabled=None`` reads the module's switch, so a caller need not pass it."""
    assert (
        decide_member_reclamation(
            "temperature_2m", 6, aggregate_evidence_ok=True, committed_leads=(6,)
        ).reason
        == REASON_DISABLED
    )
    set_supersession_enabled(True)
    assert (
        decide_member_reclamation(
            "temperature_2m", 6, aggregate_evidence_ok=True, committed_leads=(6,)
        ).authorized
        is True
    )


def test_the_reason_is_carried_on_the_decision() -> None:
    """A frozen value, so a caller can attach the reason to the row it declined to delete."""
    decision = decide_member_reclamation(
        "temperature_2m", 6, aggregate_evidence_ok=True, committed_leads=(6,), enabled=True
    )
    with pytest.raises(FrozenInstanceError):
        decision.authorized = False  # type: ignore[misc]
