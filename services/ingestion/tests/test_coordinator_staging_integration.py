"""The region write's staging hook, exercised through the coordinator.

Two properties are asserted here, both through the real coordinator entry points rather than
through the phase module directly:

* The hook is inert unless enabled, and it stages inside the same retry loop as the member
  commit. A member committed without a staged copy would later be aggregated as part of a
  partial set, and nothing would notice.
* The hook **stages and never aggregates**. An aggregate is a function of the whole member
  set, so aggregating at a member commit would publish a statistic computed from one member
  and, because the pass drops what it consumed, destroy the staging every later member needs.
"""

from __future__ import annotations

import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from ingestion.core import aggregate_phase, aggregate_staging as staging
from ingestion.core.catalog import CatalogBase, RunCatalogSpec, VariableSpec, record_run
from ingestion.core.zarr_writer import encode_region_sharded_v1

VARIABLE = "temperature_2m"
LEAD = 6
GRID_LAT, GRID_LON = 128, 160
CYCLE = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)

#: The region-write regression's grid: one full inner chunk in each direction, so the member
#: shards, the staging shards and the aggregate pass all agree on the geometry without any
#: of them being reconfigured. A smaller grid would make the pre-fix behaviour fail loudly on
#: a chunk-shape mismatch instead of publishing a wrong statistic, which would hide the
#: defect the regression is for.
REGRESSION_GRID = 100


class _NoopLockCoordinator:
    """Advisory locks without PostgreSQL, for a SQLite/local-store test."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def acquire_shared_gate(self) -> None: ...
    def release_shared_gate(self) -> None: ...
    def acquire_exclusive_gate(self) -> None: ...
    def release_exclusive_gate(self) -> None: ...
    def acquire_admission(self) -> None: ...
    def release_admission(self) -> None: ...
    def acquire_shared_admission(self) -> None: ...
    def release_shared_admission(self) -> None: ...
    def acquire_region_locks(self, region_ids: object) -> None: ...
    def release_region_locks(self, region_ids: object) -> None: ...
    def release_all(self) -> None: ...
    def close_connection(self) -> None: ...


def _dataset(seed: int = 0, member: int = 1) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    plane = rng.normal(280.0, 8.0, (GRID_LAT, GRID_LON)).astype(np.float32)
    return xr.Dataset(
        {
            VARIABLE: (
                ("member", "lead_time_hours", "latitude", "longitude"),
                plane[None, None],
            )
        }
    )


def _stage_many(store: str, n: int) -> None:
    for member in range(1, n + 1):
        dataset = _dataset(seed=member, member=member)
        blob = encode_region_sharded_v1(
            dataset, member=member, lead_time_hours=LEAD
        )[0][1]
        relative = staging.staging_relative_key(VARIABLE, member, LEAD)
        full = os.path.join(store, *relative.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(blob)


def test_disabled_phase_writes_nothing(tmp_path) -> None:
    assert aggregate_phase.staging_enabled() is False
    assert (
        aggregate_phase.stage_member_region(
            _dataset(), str(tmp_path), member=1, lead_time_hours=LEAD
        )
        == ()
    )
    assert staging.staged_objects_by_variable(str(tmp_path)) == {}


def test_phase_stages_then_aggregates_through_the_public_entry_points(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two phase entry points, called directly.

    Settlement calls these in this order once a lead's member set has stopped growing: every
    member is already staged, then one aggregate is computed from all of them. The aggregate
    is written to its real key, so a reader of record that has not been switched over
    continues to serve the member shards.
    """
    store = str(tmp_path)
    monkeypatch.setattr(aggregate_phase, "staging_enabled", lambda: True)
    monkeypatch.setattr(
        "ingestion.core.config.settings.ENSEMBLE_STAGING_ENABLED", True, raising=False
    )

    staged = aggregate_phase.stage_member_region(
        _dataset(seed=1), store, member=1, lead_time_hours=LEAD
    )
    assert staged == (staging.staging_relative_key(VARIABLE, 1, LEAD),)

    result = aggregate_phase.aggregate_lead(
        store, LEAD, grid_lat=GRID_LAT, grid_lon=GRID_LON
    )
    assert result.aggregates == (
        (VARIABLE, f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard", 1),
    )
    assert os.path.isfile(
        os.path.join(store, VARIABLE, f"shard.agg_L{LEAD:04d}.shard")
    )
    # staging is dropped for the aggregated lead
    assert staging.staged_objects_by_variable(store) == {}


def test_aggregate_object_is_invisible_to_the_member_shard_grammar(tmp_path) -> None:
    """The aggregate key must not be mistaken for a member shard by key-driven discovery.

    Store layout detection, the ingestion reader's reassembly and the API reader all identify
    member shards by the filename grammar; if the aggregate key parsed as one, a reader would
    try to reassemble statistic planes as a member field.
    """
    from domain.reclamation import is_shard_filename

    key = f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"
    assert not is_shard_filename(key)
    assert is_shard_filename(f"{VARIABLE}/shard.mem001_L{LEAD:04d}.shard")


def test_marker_evidence_still_holds_with_an_aggregate_present(tmp_path) -> None:
    """The declared write set is the member shards, and the aggregate is extra bytes.

    This is the coexistence argument for the phase-in: the marker lists what the member write
    produced, and validation checks that set as a subset of the derived expectation, so an
    aggregate object alongside it does not invalidate the evidence.
    """
    from ingestion.core.inventory import (
        build_object_inventory,
        region_expected_object_keys,
        validate_marker_evidence,
    )
    from ingestion.core.inventory import expected_write_set_fingerprint

    store = str(tmp_path)
    data_vars = [VARIABLE]
    dataset = xr.Dataset(
        {
            VARIABLE: (
                ("lead_time_hours", "latitude", "longitude"),
                _dataset(seed=2)[VARIABLE].values[0],
            )
        }
    )
    blob = encode_region_sharded_v1(dataset, member=None, lead_time_hours=LEAD)[0][1]
    relative = f"{VARIABLE}/shard.det_L{LEAD:04d}.shard"
    full = os.path.join(store, *relative.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(blob)

    expected = region_expected_object_keys(
        store,
        member=None,
        lead_index=0,
        lead_time_hours=LEAD,
        format_version="sharded_v1",
        data_var_paths=data_vars,
    )
    # the aggregate object sits next to the member shard and is simply not in the set
    agg_key = f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"
    agg_full = os.path.join(store, *agg_key.split("/"))
    with open(agg_full, "wb") as handle:
        handle.write(b"not read by anything yet")

    existing = build_object_inventory(store, data_vars)
    required = [k for k in expected if k in existing]
    omitted = [k for k in expected if k not in existing]
    validate_marker_evidence(
        store,
        marker_required_materialized=required,
        marker_omitted=omitted,
        actual_expected_keys=expected,
        marker_expected_fingerprint=expected_write_set_fingerprint(required, omitted),
        existing_objects=existing,
    )


# ---------------------------------------------------------------------------
# The region write stages every member and aggregates none of them
# ---------------------------------------------------------------------------


def _ensemble_spec() -> RunCatalogSpec:
    return RunCatalogSpec(
        center_id="noaa",
        center_name="National Oceanic and Atmospheric Administration",
        center_country="USA",
        model_id="gefs",
        model_name="GEFS",
        is_ensemble=True,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=CYCLE,
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree Grid",
        grid_resolution_km=25.0,
        product_type="surface",
        variables=(VariableSpec(VARIABLE, "2-Meter Temperature", "K"),),
        expected_lead_time_hours=(LEAD,),
        expected_members=(1, 2),
    )


def _member_dataset(member: int, size: int = REGRESSION_GRID) -> xr.Dataset:
    """A member plane identifiable by value: member ``m`` is filled with ``280 + m``."""
    plane = np.full((size, size), 280.0 + member, dtype=np.float32)
    return xr.Dataset(
        {
            VARIABLE: (
                ("member", "lead_time_hours", "latitude", "longitude"),
                plane[None, None],
            )
        },
        coords={
            "member": [member],
            "lead_time_hours": [LEAD],
            "latitude": np.arange(size, dtype=float),
            "longitude": np.arange(size, dtype=float),
        },
        attrs={"cycle_time": CYCLE.isoformat(), "model_id": "gefs"},
    )


def _aggregate_mean(store: str) -> float:
    """The MEAN field's first cell in the lead's aggregate object."""
    from ingestion.core.aggregate_writer import (
        decode_aggregate_chunk,
        layout_from_descriptor,
    )
    from domain.shard_format import split_v2_tail

    key = f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"
    with open(os.path.join(store, *key.split("/")), "rb") as handle:
        blob = handle.read()
    num_chunks, index_byte_size, _magic = struct.unpack("<III", blob[-12:])
    _index, descriptor = split_v2_tail(
        blob[-(index_byte_size + 40 + 12) :], num_chunks
    )
    layout = layout_from_descriptor(descriptor)
    assert layout.n_fields == 34  # A class: MEAN + STD + 32 bins
    return float(decode_aggregate_chunk(blob, 0)[0, 0])


def test_region_write_stages_every_member_and_writes_no_aggregate(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: a commit must never aggregate the partial member set it just wrote.

    An earlier wiring called the aggregate pass after each member's COMPLETE marker. Because
    the pass consumes and drops the staging it aggregated, that both published a statistic
    computed from a single member -- for two members of 281 and 282 it wrote MEAN = 281, then
    282, never 281.5 -- and destroyed the staging every later member needed, so the damage was
    unrecoverable. This drives the real coordinator for two members and asserts the staging of
    both survives and no aggregate object exists; the aggregate is settlement's job.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    import ingestion.core.coordinator as CO
    from ingestion.cli import _synthetic_spec_dataset
    from ingestion.core.coordinator import RunCoordinator, WaveRegion

    store = str(tmp_path / "staging_hook.zarr")
    spec = _ensemble_spec()
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)

    monkeypatch.setattr(CO, "StoreLockCoordinator", _NoopLockCoordinator)
    monkeypatch.setattr(aggregate_phase, "staging_enabled", lambda: True)

    coordinator = RunCoordinator(spec, store, timeout_seconds=2.0)
    conn = engine.connect()
    try:
        with Session(bind=conn) as catalog_session:
            run = record_run(
                catalog_session, spec, _synthetic_spec_dataset(spec), committed_state=None
            )
            run_id = str(run.id)

        coordinator.initialize_run_store(
            conn,
            seed_dataset=_member_dataset(1),
            expected_leads=(LEAD,),
            expected_members=(1, 2),
            run_id=run_id,
            is_same_cycle=True,
        )

        for member in (1, 2):
            generation = f"gen-{member}"
            coordinator.pre_update_wave(
                conn,
                regions=[
                    WaveRegion(
                        lead_time_hours=LEAD, member=member, generation=generation
                    )
                ],
                run_id=run_id,
                is_same_cycle=True,
                executor=ThreadPoolExecutor(1),
                cancel_event=threading.Event(),
            )
            coordinator.write_region_worker(
                conn,
                dataset=_member_dataset(member),
                member=member,
                generation=generation,
                expected_leads=(LEAD,),
                expected_members=(1, 2),
            )

            staged = staging.staged_members_for_lead(store, VARIABLE, LEAD)
            assert staged == list(range(1, member + 1)), (
                f"after member {member} the staged set is {staged}; a commit that dropped "
                f"staging would have made the next member's aggregate silently partial"
            )
            aggregate = os.path.join(store, VARIABLE, f"shard.agg_L{LEAD:04d}.shard")
            assert not os.path.isfile(aggregate), (
                f"the region write published {aggregate} from {member} member(s)"
            )
    finally:
        conn.close()
        engine.dispose()

    # Settlement is the only thing that may aggregate, and it produces the whole-set answer.
    result = aggregate_phase.aggregate_lead(
        store, LEAD, grid_lat=REGRESSION_GRID, grid_lon=REGRESSION_GRID
    )
    assert result.aggregates == ((VARIABLE, f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard", 2),)
    assert _aggregate_mean(store) == 281.5
    assert staging.staged_objects_by_variable(store) == {}


def test_a_patch_publication_keeps_the_staging_the_next_patch_needs(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial publication must not discard the members it will have to re-aggregate.

    The staging area is the only place a committed member's plane is kept. A publication that
    discarded it would make the next patch a statistic of the members that arrived *since* --
    for 27 members followed by 3 more it wrote MEAN over the last 3, not over all 30 -- and
    there would be nothing left to repair it from.
    """
    import struct as _struct

    from ingestion.core.aggregate_writer import (
        decode_aggregate_chunk,
        layout_from_descriptor,
    )
    from domain.shard_format import split_v2_tail

    store = str(tmp_path)
    monkeypatch.setattr(aggregate_phase, "staging_enabled", lambda: True)

    def _stage(member: int) -> None:
        plane = np.full((REGRESSION_GRID, REGRESSION_GRID), 280.0 + member, dtype=np.float32)
        dataset = xr.Dataset(
            {
                VARIABLE: (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    plane[None, None],
                )
            }
        )
        aggregate_phase.stage_member_region(
            dataset, store, member=member, lead_time_hours=LEAD
        )

    def _mean() -> float:
        key = f"{VARIABLE}/shard.agg_L{LEAD:04d}.shard"
        with open(os.path.join(store, *key.split("/")), "rb") as handle:
            blob = handle.read()
        num_chunks, index_byte_size, _magic = _struct.unpack("<III", blob[-12:])
        _index, descriptor = split_v2_tail(
            blob[-(index_byte_size + 40 + 12) :], num_chunks
        )
        layout = layout_from_descriptor(descriptor)
        assert layout.n_fields == 34
        return float(decode_aggregate_chunk(blob, 0)[0, 0])

    for member in (1, 2, 3):
        _stage(member)
    # A quiescent patch: partial, so the staging survives.
    first = aggregate_phase.aggregate_lead(
        store, LEAD, grid_lat=REGRESSION_GRID, grid_lon=REGRESSION_GRID, drop_staging=False
    )
    assert first.aggregates[0][2] == 3
    assert _mean() == 282.0
    assert len(staging.staged_members_for_lead(store, VARIABLE, LEAD)) == 3

    # A later member, then the publication that ends the lead's growth.
    _stage(4)
    second = aggregate_phase.aggregate_lead(
        store, LEAD, grid_lat=REGRESSION_GRID, grid_lon=REGRESSION_GRID, drop_staging=True
    )
    assert second.aggregates[0][2] == 4, "the patch aggregated only the new member"
    assert _mean() == 282.5
    assert staging.staged_objects_by_variable(store) == {}


def test_publication_releases_staging_only_when_it_is_the_leads_last_version(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator must pass the publication's finality through to the aggregate pass.

    ``aggregate_lead`` defaults to dropping the staging it consumed, which is right for a
    lead's last publication and wrong for a patch: the next patch has to be computed from
    every committed member, and their planes exist only in the staging area. This asserts the
    wiring rather than the parameter, because a correct default is exactly what makes the
    mistake invisible -- the patch would aggregate only the members that arrived since the
    previous one, and nothing downstream could tell.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    import ingestion.core.coordinator as CO
    import ingestion.core.coordinator as coordinator_module
    from ingestion.cli import _synthetic_spec_dataset
    from ingestion.core.coordinator import RunCoordinator

    calls: list[dict[str, object]] = []

    def _spy_aggregate_lead(store, lead, **kwargs):  # noqa: ANN001, ANN003
        calls.append({"lead": lead, **kwargs})
        return aggregate_phase.AggregatePhaseResult()

    monkeypatch.setattr(coordinator_module, "aggregate_lead", _spy_aggregate_lead)
    monkeypatch.setattr(CO, "StoreLockCoordinator", _NoopLockCoordinator)

    store = str(tmp_path / "publish.zarr")
    spec = _ensemble_spec()
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    coordinator = RunCoordinator(spec, store, timeout_seconds=2.0)
    conn = engine.connect()
    try:
        with Session(bind=conn) as catalog_session:
            run = record_run(
                catalog_session, spec, _synthetic_spec_dataset(spec), committed_state=None
            )
            run_id = str(run.id)

        coordinator.publish_settled_lead(
            conn,
            run_id=run_id,
            spec=spec,
            lead_time_hours=LEAD,
            expected_members=spec.expected_members,
            aggregate_variables=(VARIABLE,),
            aggregate_is_final=False,
        )
        coordinator.publish_settled_lead(
            conn,
            run_id=run_id,
            spec=spec,
            lead_time_hours=LEAD,
            expected_members=spec.expected_members,
            aggregate_variables=(VARIABLE,),
            aggregate_is_final=True,
        )
    finally:
        conn.close()
        engine.dispose()

    assert [call["drop_staging"] for call in calls] == [False, True]
    assert all(call["variables"] == (VARIABLE,) for call in calls)
