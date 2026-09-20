"""Worker-side tests for the aggregate-supersession override.

The worker is the last gate before a member shard is physically removed, and supersession is the
only authority that can override a *hold*: the counterfactual revalidation says "the canon needs
this unit", and a readable container says "and yet the data is still served". These tests pin
both directions of that override, the switch that governs it, and the counters an operator reads.
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
from ingestion.gc.worker import run_reclamation_worker_pass
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

NOW = datetime(2026, 9, 2, 12, 30, tzinfo=UTC)
CYCLE = datetime(2026, 9, 2, 6, tzinfo=UTC)
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
    """One GEFS cycle, lead 6 fully committed, with a real store directory."""
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
        session.add(
            ProductRecord(
                id=f"prod_geavg_{LEAD}_temperature_2m",
                run_id=run_id,
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="ensemble_mean",
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


def _write_aggregate(store: str, variable: str, lead: int) -> None:
    layout = aggregate_fields_for(variable)
    grid_lat, grid_lon, chunk = 20, 20, 10
    rng = np.random.default_rng(5)
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


def _member_rows(engine) -> list[ReclamationQueueRecord]:
    with Session(engine) as session:
        return list(
            session.execute(
                select(ReclamationQueueRecord).where(
                    ReclamationQueueRecord.target_kind == "mem"
                )
            ).scalars()
        )


def _member_statuses(engine) -> dict[int, str]:
    return {
        int(row.member_index): str(row.status)
        for row in _member_rows(engine)
    }


def test_the_planner_releases_members_and_the_worker_deletes_them(catalog):
    """The two halves together, on a store whose container is readable.

    The planner takes the member units out of the held set, and the worker -- seeing the same
    evidence -- deletes them rather than restoring the hold the counterfactual revalidation
    recomputes. This is the replacement happening.
    """
    engine, store, run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)
    set_supersession_enabled(True)

    with Session(engine) as session:
        plan = plan_reclamation_pass(session, models=("gefs",), dry_run=False, now=NOW)
    member_units = [t for t in plan.would_enqueue if t.target_kind == "mem"]
    assert len(member_units) == MEMBERS

    with Session(engine) as session:
        result = run_reclamation_worker_pass(session, delete_enabled=True, now=NOW)
    assert result.supersession_released_count == MEMBERS
    assert _member_statuses(engine) == {m: "deleted" for m in range(1, MEMBERS + 1)}
    # The physical objects are gone from the store as well, not only recorded as gone.
    assert not any(
        name.startswith("shard.mem")
        for name in os.listdir(os.path.join(store, "temperature_2m"))
    )


def test_with_the_switch_off_no_member_row_is_written(catalog):
    """The switch is upstream of the queue: with it off the worker is never asked."""
    engine, store, _run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)

    with Session(engine) as session:
        plan = plan_reclamation_pass(session, models=("gefs",), dry_run=False, now=NOW)
        result = run_reclamation_worker_pass(session, delete_enabled=True, now=NOW)

    assert [t for t in plan.would_enqueue if t.target_kind == "mem"] == []
    assert result.supersession_released_count == 0
    assert _member_rows(engine) == []


def test_the_override_deletes_a_unit_the_counterfactual_fence_holds(catalog):
    """The override on its own: a row enqueued while the container existed, deleted under it.

    The unit is one the counterfactual revalidation *holds* -- the canon still selects the run and
    lead, so ``necessary_tuples`` contains it -- which is exactly the case the override exists
    for. Every other deletion path leaves such a unit alone.
    """
    engine, store, run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)
    set_supersession_enabled(True)
    _enqueue_members(engine, run_id, store, "queued")

    with Session(engine) as session:
        result = run_reclamation_worker_pass(session, delete_enabled=True, now=NOW)

    assert result.supersession_released_count == MEMBERS
    assert result.revalidated_held_count == 0
    assert _member_statuses(engine) == {m: "deleted" for m in range(1, MEMBERS + 1)}


def test_a_lost_container_restores_the_hold(catalog):
    """The same row, with the container gone: the hold stands and the row goes back to queued.

    This is the direction that keeps the platform correct when the two representations drift --
    the aggregate was deleted or never written, the members are the only copy, and the
    counterfactual revalidation's answer is the one that survives.
    """
    engine, store, run_id = catalog
    set_supersession_enabled(True)
    _enqueue_members(engine, run_id, store, "queued")

    with Session(engine) as session:
        result = run_reclamation_worker_pass(session, delete_enabled=True, now=NOW)

    assert result.supersession_released_count == 0
    assert result.revalidated_held_count == MEMBERS
    assert _member_statuses(engine) == {m: "queued" for m in range(1, MEMBERS + 1)}
    # Each row records the hold that stopped it, which is the record an operator traces a
    # retained unit by.
    assert all(
        str(row.last_error).startswith("counterfactual_revalidation_abort: ")
        for row in _member_rows(engine)
    )


def test_a_lost_container_restores_the_hold_even_with_the_switch_on(catalog):
    """The evidence is what authorizes, not the switch: with it on but no container, nothing goes.

    A store whose aggregate was written and then reclaimed, or one where the variable's container
    was never built, is in exactly this state -- and the members have to stay, because a switch
    cannot substitute for the bytes a reader would need.
    """
    engine, store, run_id = catalog
    _write_aggregate(store, "temperature_2m", LEAD)
    set_supersession_enabled(True)
    _enqueue_members(engine, run_id, store, "queued")
    # The container disappears between the enqueue and the deletion.
    key = aggregate_store_relative_key("temperature_2m", LEAD)
    os.remove(os.path.join(store, *key.split("/")))

    with Session(engine) as session:
        result = run_reclamation_worker_pass(session, delete_enabled=True, now=NOW)

    assert result.supersession_released_count == 0
    assert result.revalidated_held_count == MEMBERS
    assert _member_statuses(engine) == {m: "queued" for m in range(1, MEMBERS + 1)}


def test_the_override_never_subtracts_a_deletion_path(catalog):
    """A store with no aggregate at all behaves exactly as it did before the override.

    The override only ever *adds* a way to delete a unit the counterfactual fence holds; it never
    requires a container for a unit that was already unnecessary. Requiring one would refuse every
    store written before the aggregate path existed, which is the whole platform's backlog. Here
    the switch is off, there is no container, and every unit the canon holds goes back to queued --
    the pre-existing answer, unchanged.
    """
    engine, store, run_id = catalog
    _enqueue_members(engine, run_id, store, "queued")

    with Session(engine) as session:
        result = run_reclamation_worker_pass(session, delete_enabled=True, now=NOW)

    assert result.supersession_released_count == 0
    assert result.revalidated_held_count == MEMBERS
    assert _member_statuses(engine) == {m: "queued" for m in range(1, MEMBERS + 1)}


def _enqueue_members(
    engine, run_id: str, store: str, status: str
) -> None:
    """Write the member rows a planner pass would have written, without running the planner.

    The worker is what is under test, and the planner's own enqueue depends on the same evidence:
    writing the rows directly isolates the worker's decision from the planner's.
    """
    from domain.reclamation import TARGET_KIND_MEM

    with Session(engine) as session:
        for member in range(1, MEMBERS + 1):
            session.add(
                ReclamationQueueRecord(
                    id=f"rec_mem_{member}",
                    run_id=run_id,
                    model_id="gefs",
                    cycle_time=CYCLE,
                    lead_time_hours=LEAD,
                    variable_code="temperature_2m",
                    target_kind=TARGET_KIND_MEM,
                    member_index=member,
                    valid_time=CYCLE,
                    store_path=store,
                    physical_key=f"temperature_2m/shard.mem{member:03d}_L{LEAD:04d}.shard",
                    status=status,
                    attempt_count=0,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        session.commit()
