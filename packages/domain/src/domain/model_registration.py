"""Registration audit for platform models across the domain contracts.

A model can exist in the catalog and be served without appearing in every
domain registry. That gap matters because the registries fail in two very
different ways, and only one of them is loud:

**Fatal — the model's canonical horizon** (:mod:`domain.horizon`).
``canonical_lead_time_hours`` raises for an unregistered model when no caller
default is supplied, and the production call sites indeed supply none:
ingestion builds its run spec with ``canonical_lead_time_hours(spec.model)``,
and the reclamation planner derives per-model serving boundaries and protection
sets with ``model_serving_start_valid_time`` / ``model_max_lead_hours`` /
``is_valid_time_protected``, none of which take a default. So an unregistered
horizon means ingestion cannot write the model at all, and — worse — the
planner raises *inside* its per-model loop, where a single bad model would
otherwise abort reclamation planning for every model.

**Degraded — cycle cadence and expected members** (:mod:`domain.cadence`,
:mod:`domain.coverage`).
Every production call site passes a ``default_if_unknown``, so an unregistered
model does not crash: it silently assumes a 6-hour cadence and a default member
count. Nothing surfaces the assumption, yet it distorts exactly the derived
numbers operators and the status badge read — ingestion lag is measured in
cadence multiples, and every coverage ratio and servability decision is taken
against the member count.

The audit therefore reports the two groups separately. Callers decide the
response: the writer may refuse to start on a fatal gap, while a read tier that
merely *serves* an unregistered model should keep running and say so loudly
rather than turn a configuration gap into an outage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from domain.cadence import is_cycle_cadence_registered
from domain.coverage import is_expected_members_registered
from domain.horizon import is_canonical_lead_horizon_registered

__all__ = ["ModelRegistrationAudit", "audit_model_registration"]


@dataclass(frozen=True)
class ModelRegistrationAudit:
    """Which models are missing from which domain registries.

    Attributes:
        missing_horizon: Models with no canonical lead horizon. Fatal: their
            ingestion and reclamation paths raise.
        missing_cadence: Models with no explicit cycle cadence. Degrades
            silently to the 6-hour default.
        missing_expected_members: Models with no explicit expected-member
            count. Degrades silently to the caller's default.
    """

    missing_horizon: tuple[str, ...] = ()
    missing_cadence: tuple[str, ...] = ()
    missing_expected_members: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """True when every audited model is registered in every registry."""
        return not (
            self.missing_horizon
            or self.missing_cadence
            or self.missing_expected_members
        )

    @property
    def has_fatal(self) -> bool:
        """True when the audit found a gap that raises on the ingestion path."""
        return bool(self.missing_horizon)

    @property
    def audited_models(self) -> tuple[str, ...]:
        """Every model named in the report, sorted and de-duplicated."""
        return tuple(
            sorted(
                set(self.missing_horizon)
                | set(self.missing_cadence)
                | set(self.missing_expected_members)
            )
        )

    def describe(self) -> str:
        """Render a single-line operator-facing summary.

        Names the offending models rather than only the registries, because the
        fix is per-model: each gap is closed by registering that model's horizon
        / cadence / member count, not by changing code.
        """
        if self.is_clean:
            return "all audited models are registered"
        parts: list[str] = []
        if self.missing_horizon:
            parts.append(
                "canonical horizon (FATAL: ingestion and reclamation raise) "
                f"for {', '.join(self.missing_horizon)}"
            )
        if self.missing_cadence:
            parts.append(
                "cycle cadence (assumes "
                f"6h) for {', '.join(self.missing_cadence)}"
            )
        if self.missing_expected_members:
            parts.append(
                "expected members (assumes caller default) "
                f"for {', '.join(self.missing_expected_members)}"
            )
        return "unregistered model contracts: " + "; ".join(parts)


def audit_model_registration(model_ids: Iterable[str]) -> ModelRegistrationAudit:
    """Report which of ``model_ids`` are missing from each domain registry.

    Pure and side-effect-free: it reads the registries and nothing else, so both
    the API and the ingestion CLI can call it at startup without touching the
    database or the object store.

    Ids are normalized (lower-cased, stripped) and de-duplicated before the
    check, matching the registries' own normalization, so a caller may pass
    catalog values verbatim.
    """
    normalized = sorted({model_id.lower().strip() for model_id in model_ids if model_id})
    return ModelRegistrationAudit(
        missing_horizon=tuple(
            m for m in normalized if not is_canonical_lead_horizon_registered(m)
        ),
        missing_cadence=tuple(
            m for m in normalized if not is_cycle_cadence_registered(m)
        ),
        missing_expected_members=tuple(
            m for m in normalized if not is_expected_members_registered(m)
        ),
    )
