"""The stored distribution read at a point, with no member values anywhere in the request.

The reader half of the stored-only mode. It goes through the reader gate -- the same SHARED store
gate every forecast read takes -- so these tests need the seeded catalog and a running lifespan,
which the module-scoped ``client`` fixture provides. What they assert is the property the mode
exists for: a container that no member read can reach still yields a drawable distribution, on a
grid it states about itself.
"""

from __future__ import annotations

import itertools
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr
from api.core.aggregate_reader import aggregate_shard_key
from api.services.aggregate_serving import (
    stored_distribution_at_point,
    stored_histogram_at_point,
)
from api.services.dual_source import member_bin_count
from domain.aggregate import compute_aggregate, finite_member_count
from domain.field_layout import aggregate_fields_for
from domain.supersession import reset_supersession_enabled, set_supersession_enabled
from domain.variable_class import spec_for
from api.models.entities import (
    EnsembleMemberProduct,
    ForecastCenter,
    ForecastProduct,
    Model,
    ModelRun,
    ModelVersion,
    ReclamationQueue,
)
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from tests._zarr_writer import write_dataset
from tests.test_aggregate_serving import _encode

GRID_LAT, GRID_LON = 128, 160
LEAD = 6
#: Model identity the seeded run is registered under. **GEFS, not GFS**, because the resolver rule
#: these tests exercise is the *ensemble* member-coverage floor: a deterministic model never reaches
#: it, so a deterministic registration could not distinguish a working waiver from an absent one.
MODEL_ID = "gefs"
#: Members the registration declares. The contract's count, so the coverage floor is the real one.
REGISTERED_MEMBERS = 30


#: Cycle time the registered runs count up from, **and it is late in the day on purpose.**
#:
#: Two reasons, and the first one bit during development:
#:
#: * ``(model_version_id, cycle_time)`` is unique on ``model_runs``, so each registration needs a
#:   cycle of its own -- the gate looks a store up by path, which means two test stores cannot
#:   share one run row.
#: * the cycle has to be **newer than anything the shared fixture catalog provides**, or the
#:   resolver's newest-cycle-wins rule hands the valid time back to a seeded run with a complete
#:   member set, and a test of the coverage floor silently tests the fixture instead.
#:
#: At 18Z with the six-hour lead the tests use, the valid time is the next day's 00Z -- which the
#: fixture GEFS run (cycle 00Z, leads 0/6/12/18) cannot reach at all. So a registered run is the
#: only candidate there, and whatever the resolver decides for it is observable.
_REGISTERED_CYCLE = datetime(2026, 7, 21, 18, tzinfo=timezone.utc)
_registration_count = itertools.count()


def _register_store(
    engine, store: str, variable: str = "temperature_2m"
) -> datetime:
    """Give a test store a catalog run, because the reader gate revalidates against it.

    The gate's revalidation is a catalog read: it looks the store path up in ``model_runs`` and
    refuses to hand back a dataset when no run in a readable status owns it. That is the mechanism
    which stops a reader from serving a store the platform has stopped serving, so a test that
    wants to read through the gate has to own a row -- the same thing every other gated suite does.

    Returns:
        The cycle time the run was registered at, which the run's valid times are derived from.
        Handed back rather than recomputed because the cycle is allocated from a module counter --
        two stores cannot share a run row, since the gate looks a store up by path.
    """
    version_id = f"version_{MODEL_ID}_stored_dist"
    with Session(engine) as session:
        if not session.get(ForecastCenter, "center_noaa"):
            session.add(
                ForecastCenter(
                    id="center_noaa", center_id="noaa", name="NOAA", country="USA"
                )
            )
            session.flush()
        if not session.get(Model, f"model_{MODEL_ID}"):
            session.add(
                Model(
                    id=f"model_{MODEL_ID}",
                    model_id=MODEL_ID,
                    name="GFS",
                    center_id="noaa",
                    is_ensemble=False,
                    resolution_km=25.0,
                )
            )
            session.flush()
        if not session.get(ModelVersion, version_id):
            session.add(
                ModelVersion(
                    id=version_id,
                    model_id=MODEL_ID,
                    version_string="stored-dist",
                )
            )
            session.flush()
        index = next(_registration_count)
        cycle = _REGISTERED_CYCLE + timedelta(hours=index)
        run_id = f"run_stored_dist_{index:03d}"
        session.add(
            ModelRun(
                id=run_id,
                model_version_id=version_id,
                cycle_time=cycle,
                status="ready",
                zarr_store_path=store,
                created_at=cycle,
            )
        )
        session.add(
            ForecastProduct(
                id=f"product_{run_id}_{variable}",
                run_id=run_id,
                variable_id=variable,
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=LEAD,
                zarr_chunk_path=store,
            )
        )
        # The committed member rows, so the coverage floor has something to count. Without them the
        # resolver's legacy fallback assumes a complete set, and a test of the floor would be
        # testing the fallback instead.
        for member in range(1, REGISTERED_MEMBERS + 1):
            session.add(
                EnsembleMemberProduct(
                    id=f"emp_{run_id}_{member}",
                    run_id=run_id,
                    member_index=member,
                    lead_time_hours=LEAD,
                )
            )
        session.commit()
    return cycle


def _point_store(tmp_path, variable: str, members: np.ndarray) -> str:
    """A store carrying both the Zarr dataset and a real container for ``variable``.

    **The dataset goes first.** The container's key is ``<variable>/shard.agg_L####.shard``, so it
    lives inside that variable's Zarr array directory -- and writing the dataset afterwards
    replaces the directory and takes the container with it. That is also the production order (a
    cycle store exists before any publication writes into it), so the reverse would be exercising
    a layout no publish produces.
    """
    store = str(tmp_path)
    dataset = xr.Dataset(
        data_vars={
            variable: (
                ("lead_time_hours", "latitude", "longitude"),
                np.zeros((1, GRID_LAT, GRID_LON), dtype=np.float32),
            )
        },
        coords={
            "lead_time_hours": [LEAD],
            "latitude": np.linspace(-49.5, 49.5, GRID_LAT, dtype=np.float32),
            "longitude": np.linspace(-99.5, 99.5, GRID_LON, dtype=np.float32),
        },
    )
    write_dataset(dataset, store)

    layout = aggregate_fields_for(variable)
    fields: list[np.ndarray] = [finite_member_count(members)]
    if layout.distribution_slice.stop > layout.distribution_slice.start:
        fields.extend(compute_aggregate(members, spec_for(variable)))
    stacked = np.concatenate([field[None] if field.ndim == 2 else field for field in fields])
    assert stacked.shape[0] == layout.n_fields
    blob = _encode(stacked, field_scales=layout.field_scales, member_count=members.shape[0])
    full = os.path.join(store, *aggregate_shard_key(variable, LEAD).split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(blob)
    return store


def _members(variable: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if variable == "temperature_2m":
        return rng.normal(280.0, 8.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    return rng.gamma(2.0, 3.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)


def test_the_stored_distribution_needs_no_member_values(client, migrated_db, tmp_path) -> None:
    """The mode a fully converted store draws from: its own grid, its own counts.

    This is what the comparison mode cannot do. That one takes its grid from the member values
    because they are what is being replaced -- correct while they exist, and impossible once they
    do not. The chart drawn from the stored fields is then the answer rather than a second opinion
    about one, so it has to be drawable without them.
    """
    members = _members("temperature_2m", 21)
    store = _point_store(tmp_path, "temperature_2m", members)
    _register_store(migrated_db, store)

    resolved = stored_distribution_at_point(
        "temperature_2m",
        store_path=store,
        lead_time_hours=LEAD,
        latitude=0.0,
        longitude=0.0,
    )
    assert resolved is not None
    edges, counts = resolved
    # The stored-only line is the *bars* a converted store draws, and the bars are binned at the
    # client's own Sturges count for this member set -- so the container states its grid at the
    # same resolution, and the chart does not change resolution when the members go.
    expected_bins = member_bin_count(30)
    assert len(edges) == expected_bins + 1
    assert len(counts) == expected_bins
    assert sum(counts) == 30
    assert all(edges[index] < edges[index + 1] for index in range(expected_bins))
    # The grid brackets the values **at the cell that was read**, which is what the bin encoding's
    # support is built around: ``mean +- sigma_range * std`` of that cell, not of the field. A
    # global sample minimum over 128x160 cells sits many standard deviations out by construction --
    # that is the tail the encoding deliberately does not cover -- so comparing against it would
    # assert the wrong thing about the wrong quantity.
    cell = members[:, 0, 0]
    assert edges[0] <= float(np.min(cell))
    assert edges[-1] >= float(np.max(cell))


def test_the_two_grid_modes_read_the_same_distribution(client, migrated_db, tmp_path) -> None:
    """The comparison mode and the stored-only mode differ in the grid, not in the distribution.

    Stated as a test because it is the property that lets the stored-only line take over from the
    comparison's dashed one without the chart changing what it shows: the same CDF, sampled at
    different edges.
    """
    members = _members("wind_gust", 22)
    store = _point_store(tmp_path, "wind_gust", members)
    _register_store(migrated_db, store)

    own = stored_distribution_at_point(
        "wind_gust",
        store_path=store,
        lead_time_hours=LEAD,
        latitude=0.0,
        longitude=0.0,
    )
    assert own is not None
    edges, counts = own

    on_own_grid = stored_histogram_at_point(
        "wind_gust",
        store_path=store,
        lead_time_hours=LEAD,
        latitude=0.0,
        longitude=0.0,
        edges=edges,
    )
    assert on_own_grid == counts


def test_a_store_with_no_container_reports_no_stored_distribution(client, migrated_db, tmp_path) -> None:
    """Absence is reported, not drawn: a flat line would say the distribution is empty.

    The store is readable and carries the variable -- only the container is missing -- so this is
    the "an aggregate was never written here" case rather than an unreadable store.
    """
    store = str(tmp_path)
    write_dataset(
        xr.Dataset(
            data_vars={
                "temperature_2m": (
                    ("lead_time_hours", "latitude", "longitude"),
                    np.zeros((1, GRID_LAT, GRID_LON), dtype=np.float32),
                )
            },
            coords={
                "lead_time_hours": [LEAD],
                "latitude": np.linspace(-49.5, 49.5, GRID_LAT, dtype=np.float32),
                "longitude": np.linspace(-99.5, 99.5, GRID_LON, dtype=np.float32),
            },
        ),
        store,
    )
    _register_store(migrated_db, store)

    assert (
        stored_distribution_at_point(
            "temperature_2m",
            store_path=store,
            lead_time_hours=LEAD,
            latitude=0.0,
            longitude=0.0,
        )
        is None
    )


def test_an_off_grid_point_reports_no_stored_distribution(client, migrated_db, tmp_path) -> None:
    """The gate's own coordinate derivation refuses a point the container does not cover."""
    members = _members("temperature_2m", 23)
    store = _point_store(tmp_path, "temperature_2m", members)
    _register_store(migrated_db, store)

    assert (
        stored_distribution_at_point(
            "temperature_2m",
            store_path=store,
            lead_time_hours=LEAD,
            latitude=89.9,
            longitude=179.9,
        )
        is None
    )


def test_an_unclassified_variable_reports_no_stored_distribution(client, migrated_db, tmp_path) -> None:
    """No encoding means no distribution to read, whatever the store holds."""
    members = _members("temperature_2m", 24)
    store = _point_store(tmp_path, "temperature_2m", members)
    _register_store(migrated_db, store)

    assert (
        stored_distribution_at_point(
            "not_a_variable",
            store_path=store,
            lead_time_hours=LEAD,
            latitude=0.0,
            longitude=0.0,
        )
        is None
    )
    # And a lead the store has no container for is the same answer.
    assert (
        stored_distribution_at_point(
            "temperature_2m",
            store_path=store,
            lead_time_hours=LEAD + 3,
            latitude=0.0,
            longitude=0.0,
        )
        is None
    )


def test_an_empty_edge_list_is_refused_before_the_read(client) -> None:
    """``stored_histogram_at_point`` with no grid is a caller error, answered without I/O."""
    assert (
        stored_histogram_at_point(
            "temperature_2m",
            store_path="/nonexistent",
            lead_time_hours=LEAD,
            latitude=0.0,
            longitude=0.0,
            edges=[],
        )
        is None
    )


def test_the_response_adapter_translates_the_reader_payload(monkeypatch) -> None:
    """The adapter from the reader's ``(edges, counts)`` to the response's histogram shape."""
    import api.services.aggregate_serving as serving
    from api.schemas import EnsembleHistogram
    from api.services.ensemble_data import _stored_only_histogram

    monkeypatch.setattr(
        serving,
        "stored_distribution_at_point",
        lambda *args, **kwargs: ([0.0, 1.0, 2.0], [3, 4]),
    )
    payload = _stored_only_histogram(
        variable="temperature_2m",
        store_path="/store",
        lead_time_hours=LEAD,
        latitude=0.0,
        longitude=0.0,
    )
    assert isinstance(payload, EnsembleHistogram)
    assert payload.edges == [0.0, 1.0, 2.0]
    assert payload.counts == [3, 4]

    # A store that cannot answer produces no line rather than an empty one.
    monkeypatch.setattr(serving, "stored_distribution_at_point", lambda *args, **kwargs: None)
    assert (
        _stored_only_histogram(
            variable="temperature_2m",
            store_path="/store",
            lead_time_hours=LEAD,
            latitude=0.0,
            longitude=0.0,
        )
        is None
    )


@pytest.mark.parametrize("variable", ["temperature_2m", "wind_gust"])
def test_the_grid_is_a_function_of_the_encodings_own_information(variable: str) -> None:
    """Neither encoding needs anything but its own fields to state a range.

    Kept as a unit assertion beside the gated ones because it is the claim the whole mode rests on:
    the quantile encoding reads its outermost levels, the bin encoding its mean and spread, and
    neither asks the members.
    """
    from api.services.dual_source import stored_edges

    members = _members(variable, 25)
    spec = spec_for(variable)
    fields = compute_aggregate(members, spec, expected_members=30)[:, 0, 0]
    edges = stored_edges(fields, spec, bins=10)
    assert len(edges) == 11
    assert all(edges[i] < edges[i + 1] for i in range(10))


# ---------------------------------------------------------------------------
# The valid-time resolution path, once the members are gone
# ---------------------------------------------------------------------------


def test_the_valid_time_path_still_resolves_a_reclaimed_variable(
    client, migrated_db, tmp_path
) -> None:
    """The whole point of the resolver change: the timeline URL keeps working.

    Every endpoint that takes ``valid_time`` resolves through the canonical resolver, and that
    resolver's candidate rule is a member-coverage floor. The floor counts committed member rows
    minus the ones the platform deleted -- so a store whose members were reclaimed by *design*
    reads as an outage, and every one of those endpoints would 404 at exactly the moment the
    members were released. The floor is now waived for a candidate whose caller-named variables are
    all represented by a readable container.
    """
    from api.services.resolver import resolve_valid_time_source

    members = _members("temperature_2m", 51)
    store = _point_store(tmp_path, "temperature_2m", members)
    cycle = _register_store(migrated_db, store)

    target = cycle + timedelta(hours=LEAD)
    with Session(migrated_db) as session:
        run_id = session.execute(
            select(ModelRun.id).where(ModelRun.zarr_store_path == store)
        ).scalar_one()
        # The members reach 'deleted' exactly as the reclamation worker leaves them, and the
        # container is the only representation left. All but one is deleted: a run with *no*
        # surviving member rows is assumed complete by the resolver's legacy fallback, which would
        # hide the coverage failure instead of exercising it.
        for member in range(1, REGISTERED_MEMBERS):
            session.add(
                ReclamationQueue(
                    id=f"probe_rec_{run_id}_{member}",
                    run_id=run_id,
                    model_id=MODEL_ID,
                    cycle_time=cycle,
                    lead_time_hours=LEAD,
                    variable_code="temperature_2m",
                    target_kind="mem",
                    member_index=member,
                    valid_time=target,
                    store_path=store,
                    physical_key=f"temperature_2m/shard.mem{member:03d}_L{LEAD:04d}.shard",
                    status="deleted",
                    attempt_count=1,
                    created_at=cycle,
                    updated_at=cycle,
                )
            )
        session.commit()

    # With the reclamation switch off the coverage floor still applies, so the release never
    # happens and this is the pre-existing answer.
    reset_supersession_enabled()
    with Session(migrated_db) as session:
        with pytest.raises(HTTPException):
            resolve_valid_time_source(
                session, MODEL_ID, target, variable="temperature_2m", require_members=True
            )

    # With it on, the container answers for the variable and the resolution succeeds.
    set_supersession_enabled(True)
    try:
        with Session(migrated_db) as session:
            source = resolve_valid_time_source(
                session, MODEL_ID, target, variable="temperature_2m", require_members=True
            )
        assert source.store_path == store
        assert source.lead_time_hours == LEAD
    finally:
        reset_supersession_enabled()


def test_the_floor_is_unchanged_for_a_variable_with_no_container(
    client, migrated_db, tmp_path
) -> None:
    """A reclaimed store is not a licence to serve something nobody can read.

    The waiver is per variable and per lead and needs readable bytes. Here the members are deleted
    and there is no container, so the resolution fails -- which is also the state of every store
    written before the aggregate path existed.
    """
    from api.services.resolver import resolve_valid_time_source

    store = str(tmp_path)
    write_dataset(
        xr.Dataset(
            data_vars={
                "temperature_2m": (
                    ("lead_time_hours", "latitude", "longitude"),
                    np.zeros((1, GRID_LAT, GRID_LON), dtype=np.float32),
                )
            },
            coords={
                "lead_time_hours": [LEAD],
                "latitude": np.linspace(-49.5, 49.5, GRID_LAT, dtype=np.float32),
                "longitude": np.linspace(-99.5, 99.5, GRID_LON, dtype=np.float32),
            },
        ),
        store,
    )
    cycle = _register_store(migrated_db, store)
    target = cycle + timedelta(hours=LEAD)

    with Session(migrated_db) as session:
        run_id = session.execute(
            select(ModelRun.id).where(ModelRun.zarr_store_path == store)
        ).scalar_one()
        for member in range(1, REGISTERED_MEMBERS):
            session.add(
                ReclamationQueue(
                    id=f"probe_noc_{run_id}_{member}",
                    run_id=run_id,
                    model_id=MODEL_ID,
                    cycle_time=cycle,
                    lead_time_hours=LEAD,
                    variable_code="temperature_2m",
                    target_kind="mem",
                    member_index=member,
                    valid_time=target,
                    store_path=store,
                    physical_key=f"temperature_2m/shard.mem{member:03d}_L{LEAD:04d}.shard",
                    status="deleted",
                    attempt_count=1,
                    created_at=cycle,
                    updated_at=cycle,
                )
            )
        session.commit()

    set_supersession_enabled(True)
    try:
        with Session(migrated_db) as session:
            with pytest.raises(HTTPException):
                resolve_valid_time_source(
                    session, MODEL_ID, target, variable="temperature_2m", require_members=True
                )
    finally:
        reset_supersession_enabled()


def test_asking_for_members_of_a_reclaimed_variable_is_refused(
    client, migrated_db, tmp_path
) -> None:
    """A request for raw members gets an error, not an empty list beside an unread distribution.

    ``members: []`` with null statistics reads as "no data for this lead" while the stored
    distribution sits in the same response unread. That is worse than a refusal: it looks like a
    gap in the data rather than a request for something this store no longer keeps.
    """
    from api.services.ensemble_data import build_ensemble_statistics

    members = _members("temperature_2m", 52)
    store = _point_store(tmp_path, "temperature_2m", members)
    cycle = _register_store(migrated_db, store)
    _reclaim_most_members(migrated_db, store, cycle)

    set_supersession_enabled(True)
    try:
        with Session(migrated_db) as session:
            with pytest.raises(HTTPException) as refused:
                build_ensemble_statistics(
                    session,
                    latitude=0.0,
                    longitude=0.0,
                    variable="temperature_2m",
                    model=MODEL_ID,
                    lead_time_hours=LEAD,
                    include_members=True,
                    initial_time=cycle.isoformat(),
                )
        assert refused.value.status_code == 422
        assert "include_members" in str(refused.value.detail)

        # And the statistics-only request in the same state is served, from the container.
        with Session(migrated_db) as session:
            served = build_ensemble_statistics(
                session,
                latitude=0.0,
                longitude=0.0,
                variable="temperature_2m",
                model=MODEL_ID,
                lead_time_hours=LEAD,
                initial_time=cycle.isoformat(),
            )
        assert served.members is None
        assert served.histogram_stored is not None
        assert served.statistics.mean is not None
    finally:
        reset_supersession_enabled()


def _reclaim_most_members(engine, store: str, cycle: datetime) -> None:
    """Mark every member but one deleted, as the reclamation worker leaves them.

    One survivor rather than none: a run with *no* surviving member rows is assumed complete by
    the resolver's legacy fallback, which would hide the coverage failure these tests exercise.
    """
    with Session(engine) as session:
        run_id = session.execute(
            select(ModelRun.id).where(ModelRun.zarr_store_path == store)
        ).scalar_one()
        for member in range(1, REGISTERED_MEMBERS):
            session.add(
                ReclamationQueue(
                    id=f"probe_inc_{run_id}_{member}",
                    run_id=run_id,
                    model_id=MODEL_ID,
                    cycle_time=cycle,
                    lead_time_hours=LEAD,
                    variable_code="temperature_2m",
                    target_kind="mem",
                    member_index=member,
                    valid_time=cycle + timedelta(hours=LEAD),
                    store_path=store,
                    physical_key=f"temperature_2m/shard.mem{member:03d}_L{LEAD:04d}.shard",
                    status="deleted",
                    attempt_count=1,
                    created_at=cycle,
                    updated_at=cycle,
                )
            )
        session.commit()


# ---------------------------------------------------------------------------
# The point window: a request between nodes is not the cell that contains it
# ---------------------------------------------------------------------------


def _linspace_store(tmp_path, variable: str, members: np.ndarray) -> str:
    """A store whose axes start at a known node, so a midpoint can be addressed exactly.

    ``_point_store``'s axes are ``linspace(-49.5, 49.5, 128)`` and ``linspace(-99.5, 99.5, 160)``,
    which puts 0.0 on a node of both. Latitude 0.375 and longitude 0.5 are then half a cell away
    in each direction -- the fractional position a point request actually asks about, and the one
    the member path interpolates for.
    """
    return _point_store(tmp_path, variable, members)


def test_a_point_between_nodes_is_blended_not_truncated(client, migrated_db, tmp_path) -> None:
    """The stored answer at a fractional point must be the window's, not the containing cell's.

    The member path interpolates the 2x2 neighbourhood (``_interpolate_neighborhood``); the stored
    path used to read the one cell containing the point. Measured on the real 09-23 00Z GEFS store,
    that made ``temperature_2m``'s mean 0.45 K from the interpolated answer against 0.40 K of
    split-half noise -- over the §9 gate -- and the user-visible symptom was a stored distribution
    whose shape differed from the member one.

    This pins the mechanism locally: the point below sits halfway between two nodes whose stored
    means differ, so a blended answer lands between them and a truncated one on a node.
    """
    from api.core.aggregate_reader import AggregateShardReader
    from api.services.aggregate_serving import point_window
    from api.services.point_forecast import _derive_grid

    # A gradient along latitude, so neighbouring nodes differ by much more than the encoding's step.
    base = np.linspace(0.0, 20.0, GRID_LAT, dtype=np.float32)[None, :, None]
    members = np.broadcast_to(base, (30, GRID_LAT, GRID_LON)).copy()
    store = _point_store(tmp_path, "temperature_2m", members)
    _register_store(migrated_db, store)

    import xarray as xr

    dataset = xr.open_zarr(store)
    grid, _latd, _lond = _derive_grid(dataset)
    assert grid.rows >= 2 and grid.cols >= 2
    # Half a cell above the grid's first latitude node.
    node_lat = float(dataset.coords["latitude"].values[0])
    step = float(dataset.coords["latitude"].values[1] - node_lat)
    midpoint_lat = node_lat + step / 2.0
    node_lon = float(dataset.coords["longitude"].values[0])

    at_node = point_window(
        dataset, "temperature_2m", store_path=store, lead_time_hours=LEAD,
        latitude=node_lat, longitude=node_lon,
    )
    at_mid = point_window(
        dataset, "temperature_2m", store_path=store, lead_time_hours=LEAD,
        latitude=midpoint_lat, longitude=node_lon,
    )
    dataset.close()
    assert at_node is not None and at_mid is not None
    mid_mean = float(at_mid[0][1])

    # The two latitudes' own stored means, read straight from their cells.
    reader = AggregateShardReader(store)
    geometry = reader.open("temperature_2m", LEAD, generation=None)
    assert geometry is not None
    first = reader.read_location(
        "temperature_2m", lead_time_hours=LEAD, chunk_row=0, chunk_col=0
    )
    assert first is not None
    cell_low = float(first[1, 0, 0])
    cell_high = float(first[1, 1, 0])
    # The midpoint is strictly inside the pair, so it is a blend rather than either cell.
    assert min(cell_low, cell_high) < mid_mean < max(cell_low, cell_high)
    assert mid_mean != pytest.approx(cell_low, abs=1e-6)
    assert mid_mean != pytest.approx(cell_high, abs=1e-6)
