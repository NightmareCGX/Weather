"""Registration audit of catalog models against the domain model contracts.

A model row can exist in the catalog — and be served — while missing from one or
more of the domain registries that describe it (its canonical lead horizon, its
cycle cadence, its expected member count). Those registries fail in two very
different ways (see :mod:`domain.model_registration`): an unregistered horizon
raises on the ingestion and reclamation paths, while an unregistered cadence or
member count degrades silently to a default that quietly distorts every number
derived from it.

The serving tier is the one place that can see *every* catalog model at once, so
it is where a model that slipped through registration becomes visible at all.
This module answers that question for a request-scoped session; the caller
decides how loudly to report it.
"""

from __future__ import annotations

from domain.model_registration import ModelRegistrationAudit, audit_model_registration
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models.entities import Model

__all__ = ["audit_catalog_model_registration"]


def audit_catalog_model_registration(db: Session) -> ModelRegistrationAudit:
    """Audit every ``models.model_id`` in the catalog against the registries.

    Reads only the model catalog (one bounded distinct-id query) and the
    in-process registries, so it is safe to call from a startup hook or a
    diagnostics endpoint.

    Args:
        db: Database session.

    Returns:
        The audit report. An empty catalog yields a clean report.
    """
    model_ids = db.execute(select(Model.model_id).distinct()).scalars().all()
    return audit_model_registration(model_ids)
