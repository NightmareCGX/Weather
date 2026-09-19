"""The settlement policy: when a lead publishes, and when it patches.

The policy is timing behaviour, so it is tested as timing behaviour -- with a supplied clock
rather than by sleeping. The four arrival shapes the approved policy distinguishes are each
pinned here, because they are what the publication cadence consists of:

* members arrive slowly enough that the set completes inside the window: one publication,
  and it is the complete one;
* the set reaches the serving floor and then stalls: one quiescent publication, then nothing
  while it stays stalled (an unchanged set must not keep flushing the caches);
* a member arrives after a quiescent publication: one patch, carrying the new count;
* the wave finishes with members still missing: a final publication that records what exists.

Coverage, not item completion, gates a quiescent publication, and the gate is the same
``domain.coverage`` authority the serving tier applies -- publishing below it would advertise
a lead the API refuses to serve, so that is pinned too.
"""

from __future__ import annotations

import pytest
from domain.coverage import set_min_coverage_ratio
from ingestion.core.settlement import (
    PUBLISH_COMPLETE,
    PUBLISH_FINAL,
    PUBLISH_QUIESCENT,
    LeadSettlementBook,
    settlement_quiet_seconds,
)


@pytest.fixture(autouse=True)
def _restore_coverage_ratio():
    """These tests read the serving floor, so leave it as they found it."""
    from domain.coverage import get_min_coverage_ratio, reset_min_coverage_ratio

    before = get_min_coverage_ratio()
    yield
    reset_min_coverage_ratio()
    assert get_min_coverage_ratio() == before


def _book(
    expected_members: tuple[int, ...] = tuple(range(1, 31)),
    *,
    quiescence: float = 600.0,
) -> LeadSettlementBook:
    return LeadSettlementBook(
        expected_members=expected_members, start_at=0.0, quiescence_seconds=quiescence
    )


def _commit(book: LeadSettlementBook, lead: int, members: range | tuple[int, ...], at: float) -> None:
    for member in members:
        book.note_member_committed(lead, member, at=at)


# ---------------------------------------------------------------------------
# Nothing is due before there is something to publish
# ---------------------------------------------------------------------------


def test_nothing_is_due_for_an_untouched_lead() -> None:
    book = _book()
    assert book.due(now=10_000.0) == []


def test_a_lead_below_the_serving_floor_is_not_published() -> None:
    """85% is the floor the API serves at; publishing under it advertises an unservable lead.

    The floor is a ratio, not a count: 25 of 30 members is 83.3%, below it, while 26 is 86.7%.
    """
    book = _book()
    _commit(book, 6, range(1, 26), at=0.0)
    assert book.due(now=10_000.0) == []

    # The member that crosses the floor is what makes the set publishable.
    book.note_member_committed(6, 26, at=0.0)
    due = book.due(now=10_000.0)
    assert [d.member_count for d in due] == [26]
    assert due[0].reason == PUBLISH_QUIESCENT


def test_an_unmoved_window_holds_a_publishable_set_back() -> None:
    """The window runs from the last arrival, so a set below it waits."""
    book = _book()
    _commit(book, 6, range(1, 28), at=0.0)
    assert book.due(now=599.0) == []
    assert [d.reason for d in book.due(now=600.0)] == [PUBLISH_QUIESCENT]


def test_a_new_member_restarts_the_window() -> None:
    book = _book()
    _commit(book, 6, range(1, 28), at=0.0)
    assert book.due(now=600.0) != []

    book.note_member_committed(6, 28, at=600.0)
    assert book.due(now=900.0) == []
    assert [d.member_count for d in book.due(now=1200.0)] == [28]


def test_a_repeated_report_of_the_same_member_is_not_an_arrival() -> None:
    """A retry that re-reports a member carries no new data and must not hold the window open."""
    book = _book()
    _commit(book, 6, range(1, 28), at=0.0)
    assert book.note_member_committed(6, 3, at=590.0) is False
    assert [d.reason for d in book.due(now=600.0)] == [PUBLISH_QUIESCENT]


# ---------------------------------------------------------------------------
# The four arrival shapes
# ---------------------------------------------------------------------------


def test_a_set_that_completes_inside_the_window_publishes_once_and_complete() -> None:
    book = _book()
    _commit(book, 6, range(1, 31), at=0.0)
    due = book.due(now=5.0)
    assert len(due) == 1
    assert due[0].reason == PUBLISH_COMPLETE
    assert due[0].member_count == 30
    assert due[0].is_final

    book.note_published(due[0])
    assert book.due(now=10_000.0) == []


def test_a_stalled_set_publishes_once_and_then_stays_quiet() -> None:
    """The point of comparing against the last publication: no periodic cache flush."""
    book = _book()
    _commit(book, 6, range(1, 28), at=0.0)
    first = book.due(now=600.0)
    assert [d.reason for d in first] == [PUBLISH_QUIESCENT]
    book.note_published(first[0])

    # Hours pass with no new member. Nothing is due, however long the stall.
    for now in (601.0, 5000.0, 50_000.0):
        assert book.due(now=now) == []


def test_a_member_arriving_after_a_patch_produces_exactly_one_more_publication() -> None:
    book = _book()
    _commit(book, 6, range(1, 28), at=0.0)
    first = book.due(now=600.0)
    book.note_published(first[0])

    book.note_member_committed(6, 28, at=700.0)
    assert book.due(now=1200.0) == []
    patch = book.due(now=1300.0)
    assert [d.member_count for d in patch] == [28]
    assert patch[0].reason == PUBLISH_QUIESCENT
    book.note_published(patch[0])

    # and then quiet again until the next member
    assert book.due(now=99_999.0) == []


def test_a_settled_wave_publishes_whatever_is_committed() -> None:
    """The termination signal: the wave knows its target set, so it knows it is done."""
    book = _book()
    _commit(book, 6, range(1, 20), at=0.0)
    assert book.due(now=100.0) == []  # below the floor, still growing

    book.note_lead_settled(6)
    due = book.due(now=100.0)
    assert len(due) == 1
    assert due[0].reason == PUBLISH_FINAL
    assert due[0].member_count == 19
    assert due[0].is_final

    book.note_published(due[0])
    assert book.due(now=99_999.0) == []


def test_a_settled_wave_with_nothing_committed_publishes_an_empty_count() -> None:
    """A lead whose every member failed is still a fact the catalog should record."""
    book = _book()
    book.note_lead_settled(6)
    due = book.due(now=0.0)
    assert [d.member_count for d in due] == [0]
    assert due[0].reason == PUBLISH_FINAL


def test_the_complete_set_wins_over_the_final_one() -> None:
    """Both reasons apply when a fully-committed lead's wave ends; complete is the truer one."""
    book = _book()
    _commit(book, 6, range(1, 31), at=0.0)
    book.note_lead_settled(6)
    due = book.due(now=0.0)
    assert [d.reason for d in due] == [PUBLISH_COMPLETE]


def test_a_failed_publication_is_retried_rather_than_assumed_done() -> None:
    """The book records publications, not decisions: nothing is called done until it is."""
    book = _book()
    _commit(book, 6, range(1, 28), at=0.0)  # 27 members, above the 26-member floor
    assert book.due(now=600.0) != []
    # no note_published: the caller's publication failed, because a publication takes the
    # store's exclusive gate and can be fenced or lose a lock race
    assert [d.member_count for d in book.due(now=700.0)] == [27]


# ---------------------------------------------------------------------------
# Order and cadence
# ---------------------------------------------------------------------------


def test_due_is_ascending_by_lead() -> None:
    book = _book()
    for lead in (12, 0, 6):
        _commit(book, lead, range(1, 31), at=0.0)
    assert [d.lead_time_hours for d in book.due(now=0.0)] == [0, 6, 12]


def test_the_tick_is_a_fraction_of_the_window() -> None:
    """Polled far more often than the window is wasted work; far less is a late publication."""
    assert _book(quiescence=600.0).tick_seconds() == 60.0
    assert _book(quiescence=40.0).tick_seconds() == 10.0
    # a very long window is capped, and a very short one still ticks at a sane rate
    assert _book(quiescence=100_000.0).tick_seconds() == 60.0
    assert _book(quiescence=2.0).tick_seconds() == 1.0


def test_a_one_member_contract_completes_on_its_single_member() -> None:
    book = _book(expected_members=(1,))
    book.note_member_committed(6, 1, at=0.0)
    assert [d.reason for d in book.due(now=0.0)] == [PUBLISH_COMPLETE]


def test_a_single_member_contract_has_no_partial_state() -> None:
    book = _book(expected_members=(1,))
    assert book.quiescent_publication_applies is False
    assert book.expected_member_count == 1


def test_a_thirty_member_contract_does_have_partial_states() -> None:
    book = _book(expected_members=tuple(range(1, 31)))
    assert book.quiescent_publication_applies is True
    assert book.expected_member_count == 30


def test_reporting_uses_the_publication_recorded_not_the_decision_taken() -> None:
    book = _book()
    assert book.committed_members(6) == frozenset()
    assert book.published_member_count(6) is None
    assert book.is_settled(6) is False

    _commit(book, 6, (1, 2), at=0.0)
    book.note_lead_settled(6)
    self_decision = book.due(now=0.0)[0]
    book.note_published(self_decision)
    assert book.committed_members(6) == frozenset({1, 2})
    assert book.published_member_count(6) == 2
    assert book.is_settled(6) is True


def test_an_explicit_expected_count_may_exceed_the_enumerated_set() -> None:
    """A wave may ingest a subset while the contract's count stays the whole set."""
    book = LeadSettlementBook(
        expected_members=(1, 2, 3),
        start_at=0.0,
        quiescence_seconds=10.0,
        expected_member_count=30,
    )
    _commit(book, 6, (1, 2, 3), at=0.0)
    # 3/30 is far below the floor, so nothing is published even though every *enumerated*
    # member is committed: the contract, not this wave, decides completeness.
    assert book.due(now=1000.0) == []


def test_invalid_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="positive expected member count"):
        LeadSettlementBook(expected_members=(), start_at=0.0)
    with pytest.raises(ValueError, match="quiescence_seconds must be positive"):
        LeadSettlementBook(expected_members=(1, 2), start_at=0.0, quiescence_seconds=0.0)


def test_reporting_reflects_what_was_recorded() -> None:
    book = _book()
    assert book.committed_members(6) == frozenset()
    assert book.published_member_count(6) is None
    assert book.is_settled(6) is False

    _commit(book, 6, (1, 2), at=0.0)
    book.note_published(book.due(now=100_000.0)[0]) if book.due(now=100_000.0) else None
    book.note_lead_settled(6)
    assert book.committed_members(6) == frozenset({1, 2})
    assert book.is_settled(6) is True


def test_the_floor_follows_the_configured_coverage_ratio() -> None:
    """The gate is the serving tier's authority, not a second hard-coded number."""
    set_min_coverage_ratio(0.50)
    book = _book()
    _commit(book, 6, range(1, 16), at=0.0)  # 15/30, below the default floor
    assert [d.member_count for d in book.due(now=600.0)] == [15]


def test_quiet_window_defaults_to_the_approved_ten_minutes() -> None:
    assert settlement_quiet_seconds() == 600.0
