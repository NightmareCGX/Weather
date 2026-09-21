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
from domain.aggregate import compute_aggregate, finite_member_count
from domain.field_layout import aggregate_fields_for
from domain.variable_class import spec_for
from api.models.entities import (
    ForecastCenter,
    ForecastProduct,
    Model,
    ModelRun,
    ModelVersion,
)
from sqlalchemy.orm import Session
from tests._zarr_writer import write_dataset
from tests.test_aggregate_serving import _encode

GRID_LAT, GRID_LON = 128, 160
LEAD = 6
#: Model identity the seeded run is registered under. One model is enough: nothing here depends on
#: the model, only on a catalog row whose ``zarr_store_path`` is the store being read.
MODEL_ID = "gfs"


#: Cycle time the registered runs count up from. Far from the seeded cycles, and paired with a
#: version string of this module's own: ``(model_version_id, cycle_time)`` is unique on
#: ``model_runs``, so each registration needs a cycle of its own -- the gate looks a store up by
#: path, which means two test stores cannot share one run row.
_REGISTERED_CYCLE = datetime(2026, 7, 21, tzinfo=timezone.utc)
_registration_count = itertools.count()


def _register_store(engine, store: str, variable: str = "temperature_2m") -> None:
    """Give a test store a catalog run, because the reader gate revalidates against it.

    The gate's revalidation is a catalog read: it looks the store path up in ``model_runs`` and
    refuses to hand back a dataset when no run in a readable status owns it. That is the mechanism
    which stops a reader from serving a store the platform has stopped serving, so a test that
    wants to read through the gate has to own a row -- the same thing every other gated suite does.
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
        session.commit()


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
    assert len(edges) == 11
    assert len(counts) == 10
    assert sum(counts) == 30
    assert all(edges[index] < edges[index + 1] for index in range(10))
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
