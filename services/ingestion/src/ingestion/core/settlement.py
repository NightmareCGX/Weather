"""When a lead's aggregate is published, and how often it is patched.

The approved policy
-------------------
A lead publishes as soon as its whole expected member set is committed. If the set stalls
short of that -- an upstream member is late, or an item failed and is being retried -- a first
version publishes anyway once member coverage reaches the serving floor (85%) and no new
member has arrived for a quiet window (10 minutes by default). Any new member restarts the
window, and the next publication carries it. The lead keeps being patched that way until one
of two things ends it: the full set arrives, or the wave settles the lead.

This module is that policy and nothing else: it holds no locks, touches no store, and takes
the current time as an argument. The caller owns the effects. That split is what makes the
timing behaviour testable without sleeping -- the alternative, a policy that reads a clock and
writes a store, can only be tested by waiting, which is how timing bugs survive.

Two decisions worth stating outright
------------------------------------
**A window expiry publishes only when the committed set has changed since the last
publication.** Re-publishing an unchanged set would produce byte-identical content while
bumping the manifest generation, and a generation bump invalidates the Redis response cache
and the tile LRU for the whole store. On a lead that stalls for hours that is a periodic
self-inflicted cache flush, during the one period when no content is changing to justify it.
The observable behaviour is otherwise identical: the window still restarts when a member
arrives, so a patch always carries new data.

**Coverage, not item completion, decides a quiescent publication.** The wave's per-lead item
bookkeeping counts members whose download or decode *failed* as settled, which is right for
"no more work is coming" and wrong for "these members exist". Only committed members are
counted here; the floor comes from ``domain.coverage``, the same authority the serving tier
uses, so a lead cannot be published at a coverage the API would refuse to serve.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from domain.coverage import is_lead_servable

#: Default quiet window before a partial member set is published. Approved at 10 minutes: long
#: enough that the common case (a member minutes behind its siblings) publishes once when the
#: set is complete rather than twice, short enough that a stalled lead is not held back.
DEFAULT_QUIESCENCE_SECONDS: float = 600.0

#: Why a publication is due.
PUBLISH_QUIESCENT: str = "quiescent"
PUBLISH_COMPLETE: str = "complete"
PUBLISH_FINAL: str = "final"

#: Every reason, in the order a caller should prefer them when several apply.
PUBLISH_REASONS: tuple[str, ...] = (PUBLISH_COMPLETE, PUBLISH_FINAL, PUBLISH_QUIESCENT)


@dataclass(frozen=True)
class PublicationDecision:
    """One lead's aggregate is due for publication.

    Attributes:
        lead_time_hours: The forecast lead.
        member_count: Committed members the aggregate will be computed from.
        reason: One of :data:`PUBLISH_REASONS`.
    """

    lead_time_hours: int
    member_count: int
    reason: str

    @property
    def is_final(self) -> bool:
        """Whether no further members can arrive for this lead.

        The aggregate is still written for a partial set, but the caller may treat this as the
        last publication it will make for the lead.
        """
        return self.reason in (PUBLISH_COMPLETE, PUBLISH_FINAL)


@dataclass
class _LeadState:
    """Settlement state for one lead. Mutable, and private to the book."""

    committed: set[int] = field(default_factory=set)
    #: Monotonic time of the last arrival that changed ``committed``; the quiet window runs
    #: from here. Seeded with the wave's start so a lead that never receives a member has a
    #: finite window instead of ``None``.
    last_change_at: float = 0.0
    #: ``None`` until something is published; then the member count of the last publication.
    published_member_count: int | None = None
    #: Whether the wave has finished every item it had for this lead.
    settled: bool = False


class LeadSettlementBook:
    """The settlement state of every lead in one wave.

    Not thread-safe: the wave's event loop is the only caller. The publication itself runs on
    an executor, and the book is updated on the loop thread before and after it, which is what
    keeps the ordering "aggregate the set that was decided on" observable here.

    Args:
        expected_members: The complete member set the lead must reach. An empty or
            single-member set has no partial state to publish: it is either nothing or
            everything, so quiescent publication is not applicable.
        start_at: The monotonic time the wave started, used as the initial window start.
        quiescence_seconds: Quiet window before a partial set is published.
        expected_member_count: Overrides ``len(expected_members)`` when the caller knows the
            contract's count without enumerating it.
    """

    def __init__(
        self,
        *,
        expected_members: Iterable[int],
        start_at: float,
        quiescence_seconds: float = DEFAULT_QUIESCENCE_SECONDS,
        expected_member_count: int | None = None,
    ) -> None:
        members = tuple(int(m) for m in expected_members)
        self._expected_members = frozenset(members)
        self._expected_count = (
            int(expected_member_count) if expected_member_count is not None else len(members)
        )
        self._quiescence_seconds = float(quiescence_seconds)
        self._start_at = float(start_at)
        self._leads: dict[int, _LeadState] = {}
        if self._expected_count <= 0:
            raise ValueError("settlement requires a positive expected member count")
        if self._quiescence_seconds <= 0.0:
            raise ValueError(
                f"quiescence_seconds must be positive, got {self._quiescence_seconds}"
            )

    # -- reporting -----------------------------------------------------------------

    @property
    def expected_member_count(self) -> int:
        """Members a lead must reach to be complete."""
        return self._expected_count

    @property
    def quiescence_seconds(self) -> float:
        """The quiet window this book was configured with."""
        return self._quiescence_seconds

    @property
    def quiescent_publication_applies(self) -> bool:
        """Whether a partial set can reach the serving floor at all.

        False for a one-member contract -- there is no coverage between nothing and everything
        -- and False for any contract whose floor rounds up to completeness, where the
        complete-set rule already covers the only publishable state.
        """
        return self._expected_count > 1 and is_lead_servable(
            self._expected_count - 1, self._expected_count
        )

    def committed_members(self, lead_time_hours: int) -> frozenset[int]:
        """Members committed for a lead, as far as the book has been told."""
        state = self._leads.get(int(lead_time_hours))
        return frozenset() if state is None else frozenset(state.committed)

    def published_member_count(self, lead_time_hours: int) -> int | None:
        """Members the last publication for a lead carried, or ``None`` if never published."""
        state = self._leads.get(int(lead_time_hours))
        return None if state is None else state.published_member_count

    def is_settled(self, lead_time_hours: int) -> bool:
        """Whether the wave has finished every item it had for a lead."""
        state = self._leads.get(int(lead_time_hours))
        return bool(state is not None and state.settled)

    # -- events --------------------------------------------------------------------

    def _state_for(self, lead_time_hours: int) -> _LeadState:
        lead = int(lead_time_hours)
        state = self._leads.get(lead)
        if state is None:
            state = _LeadState(
                last_change_at=self._start_at,
            )
            self._leads[lead] = state
        return state

    def note_member_committed(
        self, lead_time_hours: int, member: int, *, at: float
    ) -> bool:
        """Record that a member is durably committed for a lead.

        Returns:
            Whether this changed the committed set. A repeated report of the same member --
            a retry that succeeded after a first report, or a resumed wave re-observing a
            commit -- is not a new arrival and must not restart the quiet window, or a lead
            could be held below the window by reports that carry no new data.
        """
        state = self._state_for(lead_time_hours)
        before = len(state.committed)
        state.committed.add(int(member))
        if len(state.committed) == before:
            return False
        state.last_change_at = float(at)
        return True

    def note_lead_settled(self, lead_time_hours: int) -> None:
        """Record that the wave has no more items for a lead.

        This is the "no further member can arrive" signal the policy terminates on. It is fed
        from the wave's own item bookkeeping rather than from the database, because the wave
        knows its target set (``spec.members``) and therefore knows when it is done, whereas a
        database would have to be polled and could not tell "not yet" from "never".
        """
        self._state_for(lead_time_hours).settled = True

    def note_published(self, decision: PublicationDecision) -> None:
        """Record that a decision was carried out.

        Called after a *successful* publication. A failed one leaves the state untouched, so
        the next evaluation retries it rather than treating it as done.
        """
        state = self._state_for(decision.lead_time_hours)
        state.published_member_count = int(decision.member_count)

    # -- evaluation ----------------------------------------------------------------

    def decision_for(self, lead_time_hours: int, *, now: float) -> PublicationDecision | None:
        """What, if anything, is due for one lead right now."""
        lead = int(lead_time_hours)
        state = self._leads.get(lead)
        if state is None:
            return None
        count = len(state.committed)

        # The whole set is in. Nothing better can be computed, so publish immediately rather
        # than making the readers wait out a window that has nothing to wait for. Checked
        # before settlement so a lead that completed *and* whose wave ended is recorded as
        # complete rather than as a partial set that happened to reach the contract.
        if count and count >= self._expected_count:
            if state.published_member_count != count:
                return PublicationDecision(lead, count, PUBLISH_COMPLETE)
            return None

        # The wave has nothing left for this lead: publish whatever is committed, once. This is
        # also the path that records a genuinely partial lead in the catalog, so it is not
        # gated on the coverage floor.
        if state.settled:
            if state.published_member_count != count:
                return PublicationDecision(lead, count, PUBLISH_FINAL)
            return None

        # A partial set is publishable only at serving coverage, which is the same floor the
        # API applies -- publishing below it would advertise a lead the serving tier refuses.
        if not count or not is_lead_servable(count, self._expected_count):
            return None
        if (now - state.last_change_at) < self._quiescence_seconds:
            return None
        # An unchanged set would produce byte-identical content, and the generation bump that
        # accompanies a publication would flush every cache in front of it for no new data.
        if state.published_member_count == count:
            return None
        return PublicationDecision(lead, count, PUBLISH_QUIESCENT)

    def due(self, *, now: float) -> list[PublicationDecision]:
        """Every lead with a publication due, in ascending lead order.

        Ascending order so a caller that publishes serially works on the leads a reader is
        most likely to ask for first.
        """
        decisions = [
            decision
            for lead in sorted(self._leads)
            if (decision := self.decision_for(lead, now=now)) is not None
        ]
        return decisions

    def tick_seconds(self, *, ceiling: float = 60.0) -> float:
        """How often to evaluate the policy.

        A quarter of the window, clamped: polling far more often than the window would spend
        the wave's event loop on a comparison that cannot change, and polling less often than
        the window would let a publication drift past its deadline by a large fraction of it.
        """
        return max(1.0, min(ceiling, self._quiescence_seconds / 4.0))


def settlement_quiet_seconds() -> float:
    """The configured quiet window, or the module default."""
    try:
        from ingestion.core.config import settings

        configured = getattr(settings, "ENSEMBLE_SETTLEMENT_QUIET_SECONDS", None)
        if configured is None:
            return DEFAULT_QUIESCENCE_SECONDS
        seconds = float(configured)
        return seconds if seconds > 0.0 else DEFAULT_QUIESCENCE_SECONDS
    except Exception:  # noqa: BLE001 - an unreadable config must not change the policy
        return DEFAULT_QUIESCENCE_SECONDS


__all__ = [
    "DEFAULT_QUIESCENCE_SECONDS",
    "PUBLISH_COMPLETE",
    "PUBLISH_FINAL",
    "PUBLISH_QUIESCENT",
    "PUBLISH_REASONS",
    "LeadSettlementBook",
    "PublicationDecision",
    "settlement_quiet_seconds",
]
