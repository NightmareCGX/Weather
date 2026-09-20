"""Which ``(run, lead, variable)`` member sets a stored aggregate has replaced.

The planner and the worker both have to answer this, and they have to answer it identically: the
planner decides which member units to enqueue, the worker re-decides before deleting. If the two
disagreed, one pass would delete members the other expected to find, and the store would end up
with neither representation. So the evidence reads and the policy call live here once, and both
callers supply only what they know from the catalog.

What "replaced" means
---------------------
A member set is superseded when a reader could get everything it served from the container
instead: the container is present, decodable, and laid out for that variable
(:mod:`ingestion.gc.aggregate_evidence`), every other variable whose container reads these members
at the same lead has its own container too, the predecessor window has closed, and the deployment
switch is on. :func:`domain.supersession.decide_member_reclamation` is the policy; this module is
the store-facing half that feeds it, plus one cache so a cycle does not re-read the same tail.

The cache is per call and per store, not process-wide. A generation bump replaces containers under
a store, and a stale answer here is exactly the answer that deletes the wrong bytes; one pass over
one model's runs is short enough that re-reading per pass costs the measurement says it does
(0.04 ms per probe locally) and removes the question.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from domain.supersession import (
    SupersessionDecision,
    decide_member_reclamation,
    is_supersession_enabled,
    same_lead_reader_variables,
)

from ingestion.gc.aggregate_evidence import probe_aggregate

logger = logging.getLogger(__name__)

#: One reclamation unit's identity, as the callers key their structures.
Group = tuple[str, int, str]


class AggregateEvidenceCache:
    """Per-pass memo of aggregate readings, keyed by store, variable and lead.

    A cycle asks about the same ``(store, variable, lead)`` more than once: once for the variable's
    own container and once for each reader's. The cache collapses those to one probe each.
    """

    def __init__(self) -> None:
        self._readable: dict[tuple[str, str, int], bool] = {}

    def readable(self, store_path: str, variable: str, lead_time_hours: int) -> bool:
        """Whether a readable aggregate container for this triple exists."""
        key = (store_path, variable, lead_time_hours)
        cached = self._readable.get(key)
        if cached is None:
            cached = probe_aggregate(store_path, variable, lead_time_hours).ok
            self._readable[key] = cached
        return cached


def decide_supersession(
    groups: Iterable[Group],
    *,
    store_path_by_run: Mapping[str, str],
    committed_leads_by_run: Mapping[str, set[int]],
    max_lead_by_run: Mapping[str, int],
    cache: AggregateEvidenceCache | None = None,
    enabled: bool | None = None,
) -> dict[Group, SupersessionDecision]:
    """Decide, per ``(run, lead, variable)``, whether the aggregate replaces the members.

    Args:
        groups: The triples to decide. A caller passes the ones whose members it is considering --
            the planner its canonically-held member sets, the worker the groups its claimed targets
            belong to.
        store_path_by_run: Where each run's objects live. A model has one store per cycle, so this
            cannot be a single path.
        committed_leads_by_run: Leads with committed members, per run, for the predecessor window.
        max_lead_by_run: The run's authoritative maximum lead, for the same window's boundary.
        cache: Reuse a cache across calls within one pass. ``None`` builds a fresh one.
        enabled: Override the switch; ``None`` reads it from the module.

    Returns:
        One decision per requested triple. A triple naming a run with no known store is left out
        rather than decided: without a store there is no evidence to read, and the caller's own
        defaults apply.
    """
    active = is_supersession_enabled() if enabled is None else bool(enabled)
    evidence = cache if cache is not None else AggregateEvidenceCache()
    out: dict[Group, SupersessionDecision] = {}
    for run_id, lead_time_hours, variable in groups:
        store_path = store_path_by_run.get(run_id)
        if not store_path:
            continue
        readers = same_lead_reader_variables(variable)
        out[(run_id, lead_time_hours, variable)] = decide_member_reclamation(
            variable,
            lead_time_hours,
            aggregate_evidence_ok=evidence.readable(store_path, variable, lead_time_hours),
            committed_leads=committed_leads_by_run.get(run_id, set()),
            reader_variables=readers,
            reader_evidence_ok=all(
                evidence.readable(store_path, reader, lead_time_hours) for reader in readers
            ),
            max_lead=max_lead_by_run.get(run_id),
            enabled=active,
        )
    return out


__all__ = ["AggregateEvidenceCache", "Group", "decide_supersession"]
