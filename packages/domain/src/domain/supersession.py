"""Whether a variable's member shards may be deleted because its aggregate replaces them.

The aggregate encoding exists to **replace** thirty member shards with one statistic container,
and that replacement is the whole source of its measured 1.85x footprint reduction. Until now the
deletion half was missing: the aggregate was written beside the members and nothing removed them,
so switching the path on cost *more* storage rather than less
(``docs/investigations/numeric-encoding`` §19.8).

This module is the policy that decides when the second half may happen. It answers two questions
and no others:

**Which variables have a replacement at all.** Derived from :mod:`domain.variable_class`'s
registry -- the one authority on what has an approved encoding -- rather than a second list of
"converted variables" that would have to be kept in step with it. A derived variable
(``wind_10m``) has no member shards of its own and so is excluded by construction.

**Whether every dependency outlives the deletion.** Two holds apply, and both are read from
registered metadata rather than assumed:

* the aggregate container for the same ``(variable, lead)`` must exist and be readable, because it
  is what a reader gets instead;
* a predecessor-interval variable's members are read by the container of a *later* lead, so they
  stay until that lead's own members are committed. Without this the phase groups of that later
  lead -- and every patch of them -- would have no predecessor to classify against, and the
  classifier reads an absent predecessor as a definite dry one.

**The switch is off by default, and this module owns it.** Deletion is irreversible from the
store's own contents: the members are the only place those planes exist once the staging area is
released, so a deletion that turns out wrong is repaired by re-ingesting the cycle. That is a
product decision, and until it is taken the predicate below still evaluates -- so a deployment can
observe what it *would* do -- while no member is removed.

See ``docs/investigations/numeric-encoding/IMPLEMENTATION.md`` §19.15 for the wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from domain.field_layout import required_member_variables
from domain.reclamation import is_predecessor_variable
from domain.temporal import get_variable_temporal_metadata
from domain.variable_class import DERIVED_VARIABLES, REGISTERED_VARIABLES

#: Environment variable that turns member reclamation on.
SUPERSESSION_ENV_VAR: Final[str] = "ENSEMBLE_AGGREGATE_SUPERSESSION_ENABLED"

#: Variables whose members a stored aggregate can replace: every classified variable that the
#: platform stores, minus the ones it derives at serve time.
#:
#: Derived from the classification rather than listed, so adding a variable to
#: :mod:`domain.variable_class` cannot leave it silently un-reclaimable, and a variable removed
#: from it cannot leave a stale entry that would delete members nothing can answer for.
SUPERSEDABLE_VARIABLES: Final[frozenset[str]] = frozenset(
    REGISTERED_VARIABLES - DERIVED_VARIABLES
)

#: Reason strings, spelled once so a caller logging them and a test asserting on them agree.
REASON_DISABLED: Final[str] = "supersession_disabled"
REASON_NOT_SUPERSEDABLE: Final[str] = "variable_not_supersedable"
REASON_EVIDENCE_MISSING: Final[str] = "aggregate_evidence_missing"
REASON_READER_EVIDENCE_MISSING: Final[str] = "reader_aggregate_evidence_missing"
REASON_PREDECESSOR_WINDOW: Final[str] = "predecessor_interval_window_open"
REASON_AUTHORIZED: Final[str] = "aggregate_supersedes_members"


def same_lead_reader_variables(variable: str) -> frozenset[str]:
    """Variables whose aggregate container reads ``variable``'s members at the same lead.

    A container is not a function of one variable's members: ``wind_10m``'s rose reads both wind
    components, and ``precipitation_amount_3h``'s phase and transition groups read the four
    categorical flags. The relation is derived from
    :func:`domain.field_layout.required_member_variables` -- the declaration the writer itself
    consults -- rather than listed, so a group that acquires a new input cannot leave the reader
    it depends on reclaimable.

    The variable's own entry is excluded: a variable's container reading its own members is what
    :func:`decide_member_reclamation`'s ``aggregate_evidence_ok`` already checks.
    """
    name = variable.strip()
    readers: set[str] = set()
    for candidate in REGISTERED_VARIABLES:
        if candidate == name:
            continue
        if name in required_member_variables(candidate):
            readers.add(candidate)
    return frozenset(readers)


def _init_enabled_from_env() -> bool:
    raw = os.getenv(SUPERSESSION_ENV_VAR)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_is_enabled: bool = _init_enabled_from_env()


def is_supersession_enabled() -> bool:
    """Whether member reclamation by supersession is turned on."""
    return _is_enabled


def set_supersession_enabled(enabled: bool) -> None:
    """Turn member reclamation on or off (deployment configuration and tests)."""
    global _is_enabled
    _is_enabled = bool(enabled)


def reset_supersession_enabled() -> None:
    """Reset to the environment-configured value, or off when unset."""
    global _is_enabled
    _is_enabled = _init_enabled_from_env()


def is_supersedable(variable: str | None) -> bool:
    """Whether a variable's members have a stored replacement at all."""
    if variable is None:
        return False
    return variable.strip() in SUPERSEDABLE_VARIABLES


def predecessor_interval_dependent_lead(
    variable: str, lead_time_hours: int, *, max_lead: int | None = None
) -> int | None:
    """The lead whose container reads ``lead_time_hours``'s members as its predecessor.

    A reset lead ``L`` is differenced against the samples at ``L - W``, so lead ``L - W``'s
    members are an input to lead ``L``'s container. The caller asks this for a *predecessor*
    lead and gets back the lead that depends on it.

    ``(W, R)`` come from :mod:`domain.temporal`'s registry: the historically coincident
    ``W == R / 2`` is a configuration fact, never a formula, so ``L + W`` is only a dependent
    lead when it is itself a reset lead.

    Args:
        variable: Variable code.
        lead_time_hours: The lead whose members would be reclaimed.
        max_lead: The run's authoritative maximum lead. A dependent lead beyond it has no
            container at all, so the members have no reader to wait for. ``None`` means the
            caller does not know the run's extent, and the dependent lead is reported as-is.

    Returns:
        The dependent lead, or ``None`` when no lead's container reads these members -- either
        because the variable carries no interval semantics, or because ``lead + W`` is not a
        reset lead and so reads its own sample directly.
    """
    if not is_predecessor_variable(variable):
        return None
    try:
        metadata = get_variable_temporal_metadata(variable)
    except ValueError:
        return None
    interval = metadata.interval_width_hours
    reset_period = metadata.reset_period_hours
    if interval <= 0 or reset_period <= 0 or interval >= reset_period:
        # W >= R: the reset-lead sample already covers the interval and no predecessor
        # dependency exists. The same guard the reclamation planner applies.
        return None
    dependent = lead_time_hours + interval
    if dependent <= 0 or dependent % reset_period:
        return None
    if max_lead is not None and dependent > max_lead:
        return None
    return dependent


@dataclass(frozen=True)
class SupersessionDecision:
    """Whether one ``(variable, lead)``'s member units may be reclaimed, and why.

    The reason is carried rather than logged at the decision point so the caller can attach it to
    the target it declined to delete -- the same way the reclamation worker records why a
    counterfactually-necessary target was reverted.

    Attributes:
        variable: Variable code the decision is about.
        lead_time_hours: Lead the decision is about.
        authorized: Whether the member units may be reclaimed.
        reason: Why the decision took the value it did.
        evidence_ok: Whether a readable container for this ``(variable, lead)`` exists, separately
            from the decision. A caller needs it on its own: the container is a reclamation unit
            whether or not the switch let the members beside it go, and once they are gone it is
            the only representation of the variable -- so it cannot be recovered from the reason.
    """

    variable: str
    lead_time_hours: int
    authorized: bool
    reason: str
    evidence_ok: bool = False


def decide_member_reclamation(
    variable: str,
    lead_time_hours: int,
    *,
    aggregate_evidence_ok: bool,
    committed_leads: frozenset[int] | set[int] | tuple[int, ...],
    reader_variables: frozenset[str] | set[str] | tuple[str, ...] = (),
    reader_evidence_ok: bool = True,
    max_lead: int | None = None,
    enabled: bool | None = None,
) -> SupersessionDecision:
    """Whether a variable's member units at one lead may be reclaimed.

    Every condition fails closed: an unknown variable, a missing aggregate, an open predecessor
    window, or a reader whose own container is not written all mean "keep the members". The
    members are the only copy, so the cost of retaining them is bytes and the cost of removing
    them wrongly is a cycle that has to be re-ingested.

    Args:
        variable: Variable code.
        lead_time_hours: The lead whose member units are candidates.
        aggregate_evidence_ok: Whether the ``(variable, lead)`` aggregate container exists, is
            decodable, and matches this variable's declared field layout. Supplied by the caller
            because deciding it requires reading the store, which this module does not do.
        committed_leads: Leads whose members are committed for this run, as the caller knows them
            from catalog evidence.
        reader_variables: The variables whose own containers read ``variable``'s members at this
            same lead, from :func:`same_lead_reader_variables`. The caller supplies them rather
            than this module computing them, so the set can be narrowed to the readers actually
            being published for the lead without the decision changing shape.
        reader_evidence_ok: Whether every one of those readers has its own aggregate container.
            ``True`` when there are none. A reader that has not built yet would find this
            variable's members gone and refuse to publish at all -- and the live case is real:
            a ``wind_10m`` publication refreshes before ``wind_u_10m``'s first container exists.
        max_lead: The run's authoritative maximum lead, for the predecessor window's boundary --
            a dependent lead beyond it has no container that will ever read these members.
        enabled: Override the configured switch. ``None`` reads it from the module.

    Returns:
        The decision, authorized or not, with the reason it took that value.
    """
    active = is_supersession_enabled() if enabled is None else bool(enabled)
    evidence = bool(aggregate_evidence_ok)
    if not is_supersedable(variable):
        return SupersessionDecision(
            variable, lead_time_hours, False, REASON_NOT_SUPERSEDABLE, evidence
        )
    dependent = predecessor_interval_dependent_lead(
        variable, lead_time_hours, max_lead=max_lead
    )
    if dependent is not None and dependent not in set(committed_leads):
        # The lead that reads these members as its predecessor interval has not been committed,
        # so its container is still to be built and still needs them. Checked before the evidence
        # and before the switch: it is a dependency of the data, not a policy of the deployment,
        # and the caller reports it as the reason the members were retained.
        return SupersessionDecision(
            variable, lead_time_hours, False, REASON_PREDECESSOR_WINDOW, evidence
        )
    if reader_variables and not reader_evidence_ok:
        return SupersessionDecision(
            variable, lead_time_hours, False, REASON_READER_EVIDENCE_MISSING, evidence
        )
    if not evidence:
        return SupersessionDecision(
            variable, lead_time_hours, False, REASON_EVIDENCE_MISSING, False
        )
    if not active:
        # Evaluated last, so a deployment with the switch off can still be told, per target, that
        # the only thing standing between it and a deletion is the switch itself.
        return SupersessionDecision(
            variable, lead_time_hours, False, REASON_DISABLED, True
        )
    return SupersessionDecision(variable, lead_time_hours, True, REASON_AUTHORIZED, True)


__all__ = [
    "REASON_AUTHORIZED",
    "REASON_DISABLED",
    "REASON_EVIDENCE_MISSING",
    "REASON_NOT_SUPERSEDABLE",
    "REASON_PREDECESSOR_WINDOW",
    "REASON_READER_EVIDENCE_MISSING",
    "SUPERSEDABLE_VARIABLES",
    "SUPERSESSION_ENV_VAR",
    "SupersessionDecision",
    "decide_member_reclamation",
    "is_supersedable",
    "is_supersession_enabled",
    "predecessor_interval_dependent_lead",
    "reset_supersession_enabled",
    "same_lead_reader_variables",
    "set_supersession_enabled",
]
