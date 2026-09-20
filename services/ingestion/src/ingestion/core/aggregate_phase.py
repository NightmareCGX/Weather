"""Turning a published cycle into its aggregate form (the pass behind the phase-in switch).

Scope, deliberately narrow
--------------------------
This module wires the aggregate encodings into the write path, but only when
``ENSEMBLE_STAGING_ENABLED`` is set. With the switch off it is inert: nothing is staged, no
aggregate object is written, and a cycle's bytes are exactly what they were before. That is
what makes the phase-in safe to land before the reader exists.

Two properties of where it writes
---------------------------------
**Staging is written during the region write, not after it.** Staging must happen exactly
where the member shard is committed, inside the same retry loop, so that "a committed member
has a staged member" holds without a second failure domain. Staging after the loop would let
a member be durably committed while its staged copy is lost to a burst of transient errors,
and the aggregate pass would then silently aggregate a partial set.

**The aggregate object is a shadow of the member it replaces, and the committed write set
says so.** The aggregate is written to the real key (``<variable>/shard.agg_L####.shard``) --
it is the object a reader will eventually serve -- but the region's COMPLETE marker keeps
listing the member shards, because the write that just succeeded wrote member shards. The two
already coexist today: ``validate_marker_evidence`` checks the marker's declared set as a
subset of the expected set, and the expectation is *derived* rather than enumerated, so a
declared member set stays valid while an aggregate object sits alongside it. The reader of
record therefore remains the member shard; the aggregate is unread extra bytes until the read
path is switched over.

This is also why the aggregate is written **after** the marker exists rather than before: the
marker is the durability record for the member shards, and the aggregate must not exist
without it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import xarray as xr
from domain.variable_class import VariableClassError, spec_for

from ingestion.core.aggregate_staging import (
    StagingError,
    aggregate_lead_all_variables,
    aggregate_staged_lead,
    stage_region,
)
from ingestion.core.store_io import StoreRef

logger = logging.getLogger(__name__)


class AggregatePhaseError(RuntimeError):
    """Raised when the aggregate phase cannot run for a region it was asked to process."""


@dataclass(frozen=True)
class AggregatePhaseResult:
    """What one aggregate-phase invocation did.

    Attributes:
        staged_keys: Staging objects written, empty when the phase is disabled.
        aggregates: ``(variable, aggregate_key, member_count)`` per variable aggregated.
        skipped_variables: Variables left unaggregated, with the reason.
    """

    staged_keys: tuple[str, ...] = ()
    aggregates: tuple[tuple[str, str, int], ...] = ()
    skipped_variables: tuple[tuple[str, str], ...] = ()


def staging_enabled() -> bool:
    """Whether the aggregate phase is switched on for this environment."""
    try:
        from ingestion.core.config import settings

        return bool(getattr(settings, "ENSEMBLE_STAGING_ENABLED", False))
    except Exception:  # noqa: BLE001 - an unreadable config must not enable the phase
        return False


def staging_version() -> str:
    """Configured staging namespace version, or the module default."""
    try:
        from ingestion.core.config import settings

        return str(getattr(settings, "ENSEMBLE_STAGING_VERSION", "")) or "v1"
    except Exception:  # noqa: BLE001
        return "v1"


def stage_member_region(
    dataset: xr.Dataset,
    store: StoreRef,
    *,
    member: int | None,
    lead_time_hours: int,
    is_mean: bool = False,
) -> tuple[str, ...]:
    """Stage one member region's variables, if the phase is enabled.

    Only perturbed members are staged. The deterministic and mean products have no member
    axis to aggregate over, so staging them would cost bytes that nothing would ever combine.

    Args:
        dataset: The normalized single-lead (optionally single-member) dataset.
        store: Store root.
        member: Upstream member identity, or ``None`` for a deterministic/mean region.
        lead_time_hours: Forecast lead.
        is_mean: Whether the region is the official ensemble mean product.

    Returns:
        Staging keys written; empty when the phase is off or the region has no member axis.
    """
    if not staging_enabled():
        return ()
    if member is None or is_mean:
        return ()
    try:
        return tuple(
            stage_region(
                dataset, store, member=int(member), lead_time_hours=lead_time_hours
            )
        )
    except StagingError as exc:
        # Staging is failed loudly by the caller's retry loop: a member committed without a
        # staged copy would be aggregated as part of a partial set, which is invisible.
        raise AggregatePhaseError(
            f"cannot stage member {member} lead {lead_time_hours}h: {exc}"
        ) from exc


def aggregate_lead(
    store: StoreRef,
    lead_time_hours: int,
    *,
    variables: Sequence[str] | None = None,
    grid_lat: int | None = None,
    grid_lon: int | None = None,
    chunk_lat: int = 100,
    chunk_lon: int = 100,
    drop_staging: bool = True,
    expected_members: int | None = None,
    wave_leads: Sequence[int] | None = None,
    published_leads: Sequence[int] | None = None,
) -> AggregatePhaseResult:
    """Aggregate every staged variable of one lead, if the phase is enabled.

    A variable whose aggregate fails does not abort the pass. The member shards remain the
    reader of record, so a failed aggregate is a missing optimisation rather than a serving
    outage; the failure is reported in ``skipped_variables`` and logged at warning level.

    Args:
        store: Store root.
        lead_time_hours: Forecast lead to aggregate.
        variables: Variables to consider; defaults to every staged variable.
        grid_lat: Grid latitude extent; defaults to the configured platform grid.
        grid_lon: Grid longitude extent; defaults to the configured platform grid.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.
        drop_staging: Remove each variable's staging objects once the whole pass has written its
            containers. **Pass ``False`` for a partial publication**: the staging area is the
            only place the already-committed members' planes exist, and the next patch has to be
            computed from all of them, not from the ones that arrived since. The default is
            ``True`` for the caller that publishes the last version of a lead.
        expected_members: The contract's member count, which the per-cell coverage floor is
            measured against and which a partial publication is refused below.
        wave_leads: The leads this wave is filling, so a predecessor that has not landed yet is
            refused rather than encoded as absent.
        published_leads: The leads of this wave already published, whose staging is gone. A
            predecessor among them is finished rather than late.
    """
    if not staging_enabled():
        return AggregatePhaseResult()
    try:
        results = aggregate_lead_all_variables(
            store,
            lead_time_hours,
            variables=variables,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            chunk_lat=chunk_lat,
            chunk_lon=chunk_lon,
            drop_staging=drop_staging,
            expected_members=expected_members,
            wave_leads=wave_leads,
            published_leads=published_leads,
        )
    except StagingError as exc:
        logger.warning(
            "aggregate phase found nothing to do for lead %d: %s", lead_time_hours, exc
        )
        return AggregatePhaseResult()

    aggregates: list[tuple[str, str, int]] = []
    for variable, key, count in results:
        aggregates.append((variable, key, count))
        logger.info(
            "aggregate written: %s from %d members (lead %d)",
            key,
            count,
            lead_time_hours,
        )
    return AggregatePhaseResult(aggregates=tuple(aggregates))


def aggregate_variable_lead(
    store: StoreRef,
    variable: str,
    lead_time_hours: int,
    *,
    grid_lat: int,
    grid_lon: int,
    chunk_lat: int = 100,
    chunk_lon: int = 100,
    drop_staging: bool = True,
    expected_members: int | None = None,
    wave_leads: Sequence[int] | None = None,
    published_leads: Sequence[int] | None = None,
) -> tuple[str, int]:
    """Aggregate one variable's staged members for one lead.

    Unlike :func:`aggregate_lead` this does not consult the enable switch: it is the explicit
    form, for a caller that has already decided to aggregate (a repair, a backfill, or a
    test). The switch gates the implicit pipeline hook only.

    Raises:
        AggregatePhaseError: for an unclassified variable or a staging failure.
    """
    try:
        spec_for(variable)
    except VariableClassError as exc:
        raise AggregatePhaseError(str(exc)) from exc
    try:
        return aggregate_staged_lead(
            store,
            variable,
            lead_time_hours,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            chunk_lat=chunk_lat,
            chunk_lon=chunk_lon,
            drop_staging=drop_staging,
            expected_members=expected_members,
            wave_leads=wave_leads,
            published_leads=published_leads,
        )
    except StagingError as exc:
        raise AggregatePhaseError(str(exc)) from exc


__all__ = [
    "AggregatePhaseError",
    "AggregatePhaseResult",
    "aggregate_lead",
    "aggregate_variable_lead",
    "stage_member_region",
    "staging_enabled",
    "staging_version",
]
