"""Planner-side tests for aggregate supersession (member reclamation).

The planner is where a member unit stops being retained, and where the container that replaces it
enters the physical inventory. These tests pin both halves: the members go only when a readable
container answers for them, and the container then follows those members' fate -- held while the
canonical resolution still selects the variable, reclaimable when it stops -- so the "replacement"
is never an object nothing reclaims and never a variable the cycle is still serving.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import numpy as np
import pytest
from domain.field_layout import aggregate_fields_for
from domain.supersession import (
    SUPERSESSION_ENV_VAR,
    reset_supersession_enabled,
    set_supersession_enabled,
)
from ingestion.core.aggregate_writer import (
    AggregateShardLayout,
    aggregate_store_relative_key,
    encode_aggregate_shard,
)
from ingestion.core.catalog import (
    CenterRecord,
    EnsembleMemberProductRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    ReclamationQueueRecord,
)
from ingestion.core.db import CatalogBase
from ingestion.gc.planner import plan_reclamation_pass
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

NOW = datetime(2026, 9, 2, 12, 30, tzinfo=UTC)
CYCLE = datetime(2026, 9, 2, 6, tzinfo=UTC)
AGES_AGO = datetime(2027, 1, 1, tzinfo=UTC)
LEAD = 6
MEMBERS = 30


@pytest.fixture(autouse=True)
def clean_switch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(SUPERSESSION_ENV_VAR, raising=False)
    reset_supersession_enabled()
    yield
    monkeypatch.delenv(SUPERSESSION_ENV_VAR, raising=False)
    reset_supersession_enabled()


@pytest.fixture
def catalog(tmp_path):
    """A catalog with one GEFS cycle whose lead 6 has all 30 members and two products."""
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    store = tmp_path / "store"
    store.mkdir()
    with Session(engine) as session:
        session.add(
            CenterRecord(
                id="center_noaa",
                center_id="noaa",
                name="NOAA",
                country="US",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        session.add(
            ModelRecord(
                id="model_gefs",
                model_id="gefs",
                name="GEFS",
                center_id="noaa",
                is_ensemble=True,
                resolution_km=25.0,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        session.add(
            ModelVersionRecord(
                id="version_gefs_v1.0",
                model_id="gefs",
                version_string="v1.0",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        run = ModelRunRecord(
            id="run_gefs",
            model_version_id="version_gefs_v1.0",
            cycle_time=CYCLE,
            status="processing",
            created_at=CYCLE,
            zarr_store_path=str(store),
        )
        session.add(run)
        session.commit()
        run_id = str(run.id)
        session.add(
            ProductRecord(
                id=f"prod_{LEAD}_temperature_2m",
                run_id=run_id,
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=LEAD,
                zarr_chunk_path="/fake",
            )
        )
        for member in range(1, MEMBERS + 1):
            session.add(
                EnsembleMemberProductRecord(
                    id=f"emp_{run_id}_{member}_{LEAD}",
                    run_id=run_id,
                    member_index=member,
                    lead_time_hours=LEAD,
                )
            )
        session.commit()
    yield engine, str(store), run_id
    engine.dispose()


def _add_product(engine, run_id: str, variable: str, lead: int = LEAD) -> None:
    with Session(engine) as session:
        session.add(
            ProductRecord(
                id=f"prod_{lead}_{variable}",
                run_id=run_id,
                variable_id=variable,
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
                zarr_chunk_path="/fake",
            )
        )
        session.commit()


def _write_aggregate(store: str, variable: str, lead: int) -> None:
    """Write a real container for ``(variable, lead)`` with the production encoder."""
    layout = aggregate_fields_for(variable)
    grid_lat, grid_lon, chunk = 20, 20, 10
    rng = np.random.default_rng(11)
    planes = [
        rng.normal(0.5, 0.1, (grid_lat, grid_lon)).astype(np.float32)
        for _ in range(layout.n_fields)
    ]
    container = encode_aggregate_shard(
        planes,
        AggregateShardLayout(
            n_fields=layout.n_fields,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            chunk_lat=chunk,
            chunk_lon=chunk,
        ),
        member_count=MEMBERS,
        field_scales=layout.field_scales,
    )
    key = aggregate_store_relative_key(variable, lead)
    full = os.path.join(store, *key.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(container)


def _plan(engine, now: datetime = NOW):
    with Session(engine) as session:
        return plan_reclamation_pass(session, models=("gefs",), dry_run=True, now=now)


def _enqueued(result) -> set[tuple[str, str, int]]:
    return {(t.target_kind, t.variable_code, t.member_index) for t in result.would_enqueue}


def _member_units(result) -> set[tuple[str, int]]:
    return {
        (t.variable_code, t.member_index)
        for t in result.would_enqueue
        if t.target_kind == "mem"
    }


def _aggregate_units(result) -> set[tuple[str, int]]:
    return {
        (t.variable_code, t.lead_time_hours)
        for t in result.would_enqueue
        if t.target_kind == "agg"
    }


def _enqueue(engine, result, now: datetime = NOW) -> None:
    """Write the rows a real enqueue would write, for tests about repeated passes."""
    with Session(engine) as session:
        seen: set[tuple[str, int, str, str, int]] = set()
        for target in result.would_enqueue:
            if target.target_tuple in seen:
                continue
            seen.add(target.target_tuple)
            session.add(
                ReclamationQueueRecord(
                    id=f"rec_{len(seen)}_{target.target_kind}",
                    run_id=target.run_id,
                    model_id=target.model_id,
                    cycle_time=target.cycle_time,
                    lead_time_hours=target.lead_time_hours,
                    variable_code=target.variable_code,
                    target_kind=target.target_kind,
                    member_index=target.member_index,
                    valid_time=target.valid_time,
                    store_path=target.store_path,
                    physical_key=target.physical_key,
                    status="queued",
                    attempt_count=0,
                    created_at=now,
                    updated_at=now,
                )
            )
        session.commit()


# -- the switch is off -------------------------------------------------------


def test_with_the_switch_off_no_member_is_released(catalog):
    """The default. The container is readable, so every guard but the switch says yes."""
    engine, store, _run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)

    result = _plan(engine)
    assert result.superseded_member_units == 0
    assert _member_units(result) == set()
    # The container is still tracked: it is a physical unit of a variable the cycle serves, and
    # the pass holds it where the members it replaces stand.
    assert _aggregate_units(result) == set()
    # The 30 member units are still held by the canonical resolution; the container is held
    # alongside them.
    assert result.active_held_shards == MEMBERS + 1


def test_a_container_whose_members_left_the_window_is_reclaimed_with_the_switch_off(catalog):
    """Reclaiming a container is not releasing members, so the switch does not govern it.

    Without this the container would outlive every member it replaced -- an object nothing ever
    reclaims -- and the "replacement" would be a permanent second copy rather than a replacement.
    """
    engine, store, _run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)

    aged = _plan(engine, now=AGES_AGO)
    assert aged.superseded_member_units == 0
    assert aged.active_held_shards == 0
    assert _aggregate_units(aged) == {("temperature_2m", LEAD)}
    assert aged.aggregate_units_enqueued == 1
    # The members went with their container: the same window, the same decision.
    assert _member_units(aged) == {("temperature_2m", m) for m in range(1, MEMBERS + 1)}


# -- the switch is on --------------------------------------------------------


def test_with_a_readable_aggregate_the_members_are_superseded(catalog):
    """The whole point: the members are released and the container stands in their place.

    The container is **held**, not enqueued: it is the only representation of the variable left
    once the members go, and the canonical resolution has no notion of a container, so putting it
    on the reclamation queue while the cycle still serves the variable would delete it outright.
    """
    engine, store, _run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)
    set_supersession_enabled(True)

    result = _plan(engine)
    assert result.superseded_member_units == MEMBERS
    assert _member_units(result) == {("temperature_2m", m) for m in range(1, MEMBERS + 1)}
    assert result.aggregate_units_enqueued == 0
    assert _aggregate_units(result) == set()
    # The inventory is the 30 member shards plus the container; the container is the held one.
    assert result.total_committed_shards == MEMBERS + 1
    assert result.active_held_shards == 1


def test_the_container_leaves_with_the_members_it_replaced(catalog):
    """Both ends of "takes their place": held while served, reclaimable when not.

    With the switch on and the lead outside the serving window, the members are reclaimable
    because the canon stopped selecting them -- not because they were superseded -- and the
    container goes with them.
    """
    engine, store, _run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)
    set_supersession_enabled(True)

    aged = _plan(engine, now=AGES_AGO)
    assert aged.superseded_member_units == 0  # nothing was held to release
    assert aged.active_held_shards == 0
    assert _aggregate_units(aged) == {("temperature_2m", LEAD)}
    assert _member_units(aged) == {("temperature_2m", m) for m in range(1, MEMBERS + 1)}


def test_without_a_readable_aggregate_the_members_are_retained(catalog):
    """No container, no replacement: the members are held and there is nothing to hand over to."""
    engine, _store, _run_id = catalog
    set_supersession_enabled(True)

    result = _plan(engine)
    assert result.superseded_member_units == 0
    assert result.aggregate_units_enqueued == 0
    assert _aggregate_units(result) == set()
    assert _member_units(result) == set()
    # Nothing was superseded, so the members are held exactly as before this mechanism existed.
    assert result.active_held_shards == MEMBERS


def test_an_unreadable_container_is_not_a_replacement(catalog):
    """A container at the key that does not parse is worse than none: it is not evidence."""
    engine, store, _run_id = catalog
    key = aggregate_store_relative_key("temperature_2m", LEAD)
    full = os.path.join(store, *key.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(b"\x00" * 4096)  # long enough to look like an object, not a container
    set_supersession_enabled(True)

    result = _plan(engine)
    assert result.superseded_member_units == 0
    assert _member_units(result) == set()
    assert _aggregate_units(result) == set()


def test_a_repeated_pass_proposes_nothing_new(catalog):
    """Idempotence: the second pass sees the first pass's rows and proposes nothing.

    A pass runs on a schedule, so this is the ordinary state rather than an edge case -- and the
    container's row is the one that matters, because a second row for one object would be a
    duplicate deletion unit.
    """
    engine, store, _run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)
    set_supersession_enabled(True)

    first = _plan(engine, now=AGES_AGO)
    assert first.aggregate_units_enqueued == 1
    _enqueue(engine, first, now=AGES_AGO)

    second = _plan(engine, now=AGES_AGO)
    assert second.would_enqueue == ()
    assert second.aggregate_units_enqueued == 0
    assert _enqueued(second) == set()


# -- readers -----------------------------------------------------------------


def test_a_reader_variable_is_not_superseded_until_its_reader_has_a_container(catalog):
    """``wind_u_10m``'s members are read by ``wind_10m``'s rose, so they wait for it.

    Without the reader check, releasing the components' members would leave the rose with no input
    and every later ``wind_10m`` patch refusing to build. ``wind_10m`` is derived and has no
    members of its own, so only its own container can stand in for them.
    """
    engine, store, run_id = catalog
    for variable in ("wind_u_10m", "wind_v_10m"):
        _add_product(engine, run_id, variable)

    # The components' containers exist, but wind_10m's does not.
    _write_aggregate(store, "wind_u_10m", LEAD)
    _write_aggregate(store, "wind_v_10m", LEAD)
    set_supersession_enabled(True)

    result = _plan(engine)
    wind_members = {
        (variable, member)
        for variable in ("wind_u_10m", "wind_v_10m")
        for member in range(1, MEMBERS + 1)
    }
    assert {pair for pair in _member_units(result) if pair[0].startswith("wind_")} == set(), (
        "the components' members were released without the reader that consumes them"
    )

    # Now give wind_10m its own container: the pair becomes releasable, and the reader's
    # container is tracked alongside theirs.
    _write_aggregate(store, "wind_10m", LEAD)
    result = _plan(engine)
    assert {pair for pair in _member_units(result) if pair[0].startswith("wind_")} == wind_members
    assert result.superseded_member_units == len(wind_members)
    # The derived variable has no members of its own, so its container is the only representation
    # of it -- it is held, not enqueued, exactly like the components' containers.
    assert _aggregate_units(result) == set()
    # Held: the three containers, plus ``temperature_2m``'s member set, which has no container in
    # this test and so is retained as it always was.
    assert result.active_held_shards == 3 + MEMBERS


# -- reporting ---------------------------------------------------------------


def test_the_passes_counts_describe_what_moved(catalog):
    """The figures a deployment watches while it brings the path up."""
    engine, store, _run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)

    off = _plan(engine)
    assert off.superseded_member_units == 0
    assert off.total_committed_shards == MEMBERS + 1
    assert off.active_held_shards == MEMBERS + 1
    assert off.reclaimable_shards == 0

    set_supersession_enabled(True)
    on = _plan(engine)
    assert on.superseded_member_units == MEMBERS
    assert on.aggregate_units_enqueued == 0
    # The 30 member units moved from held to reclaimable; the container moved into the held slot.
    assert on.reclaimable_shards - off.reclaimable_shards == MEMBERS
    assert on.active_held_shards == 1
    assert on.total_committed_shards == off.total_committed_shards


def test_the_evidence_is_read_once_per_container(catalog, monkeypatch):
    """A cycle's variables share their inputs, so the probe behind them is cached per pass.

    The ask is cheap and the answer costs a store read: without the cache the four flags ask
    about the same precipitation container once each, and the wind pair asks about ``wind_10m``
    alongside them.
    """
    from ingestion.gc import supersession as supersession_module

    probes: list[tuple[str, str, int]] = []
    real_probe = supersession_module.probe_aggregate

    def _counting_probe(store_path: str, variable: str, lead_time_hours: int):
        probes.append((store_path, variable, lead_time_hours))
        return real_probe(store_path, variable, lead_time_hours)

    monkeypatch.setattr(supersession_module, "probe_aggregate", _counting_probe)

    engine, store, _run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)
    cache = supersession_module.AggregateEvidenceCache()
    decisions = supersession_module.decide_supersession(
        [("run_gefs", LEAD, "temperature_2m"), ("run_gefs", LEAD, "temperature_2m")],
        store_path_by_run={"run_gefs": store},
        committed_leads_by_run={"run_gefs": {LEAD}},
        max_lead_by_run={"run_gefs": 240},
        cache=cache,
        enabled=False,
    )
    assert len(decisions) == 1
    assert probes.count((store, "temperature_2m", LEAD)) == 1
    # A later question within the same pass is answered from the cache, not the store.
    assert cache.readable(store, "temperature_2m", LEAD) is True
    assert probes.count((store, "temperature_2m", LEAD)) == 1


def test_a_run_with_no_known_store_is_left_undecided(catalog):
    """Without a store there is no evidence to read, so no decision is invented for the group."""
    from ingestion.gc.supersession import decide_supersession

    decisions = decide_supersession(
        [("run_gefs", LEAD, "temperature_2m")],
        store_path_by_run={},
        committed_leads_by_run={"run_gefs": {LEAD}},
        max_lead_by_run={"run_gefs": 240},
        enabled=True,
    )
    assert decisions == {}


def test_a_retired_cycle_takes_its_containers_with_it(catalog):
    """A claimed cycle's units are all reclaimable, so nothing would hold a container back.

    The tombstone path enumerates catalog units, and a container is not one -- so without this a
    reclaimed container would be an object nothing ever reclaims, left behind by the cycle that
    owned it.
    """
    from ingestion.core.catalog import ForecastCycleLifecycleRecord

    engine, store, run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)
    with Session(engine) as session:
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gefs",
                cycle_time=CYCLE,
                deletion_started_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
                created_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
                updated_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
            )
        )
        session.commit()

    result = _plan(engine)
    assert _aggregate_units(result) == {("temperature_2m", LEAD)}
    # The member units of the retired cycle are reclaimable too, which is the pre-existing path.
    assert _member_units(result) == {("temperature_2m", m) for m in range(1, MEMBERS + 1)}

    # A second pass proposes nothing new.
    _enqueue(engine, result)
    assert _plan(engine).would_enqueue == ()
