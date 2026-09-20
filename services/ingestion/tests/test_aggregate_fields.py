"""The container builder: which member sets a variable's fields are a function of.

The property this suite pins is not the arithmetic -- ``domain.product_fields`` owns that and
tests it against the serving paths' own references -- but the *reading*: which staged member sets
a container needs, and what happens when one of them is missing. A container published from a
subset of its inputs is a statistic of the wrong member set, and nothing downstream could tell, so
every missing input has to be a refusal.

The fixture stages a small store by hand rather than through the coordinator, so a test can leave
exactly one input out.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr
from domain.field_layout import aggregate_fields_for
from ingestion.core.aggregate_fields import (
    AggregateBuildError,
    MemberReader,
    build_container_fields,
)
from ingestion.core.aggregate_staging import staging_relative_key
from ingestion.core.zarr_writer import encode_region_sharded_v1

LEAD = 6
PREDECESSOR_LEAD = LEAD - 3
GRID = 8
MEMBERS = (1, 2, 3, 4, 5)

#: The variables one fixture store carries: every input the groups read.
STORE_VARIABLES = (
    "temperature_2m",
    "precipitation_amount_3h",
    "crain",
    "csnow",
    "cfrzr",
    "cicep",
    "cloud_ceiling",
    "cloud_cover_3h",
    "wind_u_10m",
    "wind_v_10m",
)


def _plane(variable: str, member: int, rng: np.random.Generator) -> np.ndarray:
    """A plausible member plane for a variable, distinct enough to tell the groups apart."""
    if variable == "temperature_2m":
        return np.full((GRID, GRID), 280.0 + member, dtype=np.float32)
    if variable in ("crain", "csnow", "cfrzr", "cicep"):
        return (rng.random((GRID, GRID)) < 0.5).astype(np.float32)
    if variable == "precipitation_amount_3h":
        return np.abs(rng.normal(1.0, 2.0, (GRID, GRID))).astype(np.float32)
    if variable == "cloud_ceiling":
        return np.where(
            rng.random((GRID, GRID)) < 0.4, 20.0, rng.uniform(0.2, 12.0, (GRID, GRID))
        ).astype(np.float32)
    if variable == "cloud_cover_3h":
        return rng.uniform(0.0, 100.0, (GRID, GRID)).astype(np.float32)
    return rng.normal(3.0, 4.0, (GRID, GRID)).astype(np.float32)


def _stage(store: str, variable: str, member: int, lead: int, plane: np.ndarray) -> None:
    dataset = xr.Dataset(
        {variable: (("member", "lead_time_hours", "latitude", "longitude"), plane[None, None])}
    )
    blob = encode_region_sharded_v1(dataset, member=member, lead_time_hours=lead)[0][1]
    relative = staging_relative_key(variable, member, lead)
    full = os.path.join(store, *relative.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(blob)


@pytest.fixture()
def store(tmp_path) -> str:
    """A staging area carrying every input, for both the lead and its predecessor."""
    path = str(tmp_path / "cycle.zarr")
    rng = np.random.default_rng(0)
    for member in MEMBERS:
        for lead in (LEAD, PREDECESSOR_LEAD):
            for variable in STORE_VARIABLES:
                _stage(path, variable, member, lead, _plane(variable, member, rng))
    return path


def _reader(store: str) -> MemberReader:
    return MemberReader(store, grid_lat=GRID, grid_lon=GRID, chunk_lat=GRID, chunk_lon=GRID)


def test_every_variable_produces_the_field_vector_its_layout_declares(store: str) -> None:
    """The container's descriptor describes the fields the builder produces, field for field."""
    reader = _reader(store)
    for variable in STORE_VARIABLES:
        fields, count = build_container_fields(
            reader, variable, LEAD, expected_members=30
        )
        layout = aggregate_fields_for(variable)
        assert len(fields) == layout.n_fields, variable
        assert count == len(MEMBERS), variable
        for plane in fields:
            assert plane.shape == (GRID, GRID), variable


def test_the_distribution_is_the_variables_own_aggregate(store: str) -> None:
    """The fields after the count and before the groups are ``compute_aggregate``'s output."""
    from domain.aggregate import compute_aggregate
    from domain.variable_class import spec_for

    reader = _reader(store)
    variable = "temperature_2m"
    fields, _count = build_container_fields(reader, variable, LEAD, expected_members=30)
    members, stack = reader.stack(variable, LEAD)

    assert np.array_equal(fields[0], np.isfinite(stack).sum(axis=0).astype(np.float32))
    expected = compute_aggregate(stack, spec_for(variable), expected_members=len(members))
    for offset, plane in enumerate(expected):
        assert np.array_equal(fields[1 + offset], plane, equal_nan=True), offset


def test_a_derived_variable_takes_its_member_set_from_its_inputs(store: str) -> None:
    """``wind_10m`` is stored as no member shards; its products describe its components' members.

    This is also why the variable has a container at all: without one, a reclamation rule phrased
    as "the aggregate exists" would need a hand-written exception for the wind pair, which is a
    fifth of a cycle's member bytes.
    """
    reader = _reader(store)
    assert reader.members("wind_10m", LEAD) == []
    fields, count = build_container_fields(reader, "wind_10m", LEAD, expected_members=30)
    assert count == len(MEMBERS)
    assert len(fields) == aggregate_fields_for("wind_10m").n_fields
    # The rose's bucket edges are the group's last nine planes, and they are ordered.
    edges = fields[-9:]
    assert (np.diff(np.array([plane[0, 0] for plane in edges])) >= 0).all()


def test_the_rose_is_refused_when_a_component_is_missing(store: str) -> None:
    """One component is not a vector field: a rose needs the same members' u and v."""
    import shutil

    shutil.rmtree(os.path.join(store, "__staging__", "v1", "wind_v_10m"))
    with pytest.raises(AggregateBuildError, match="has no staged member"):
        build_container_fields(_reader(store), "wind_10m", LEAD, expected_members=30)


def test_the_rose_is_refused_when_the_components_disagree_on_members(store: str) -> None:
    """A u stack and a v stack of different members would pair the wrong vectors."""
    path = os.path.join(store, "__staging__", "v1", "wind_v_10m")
    os.remove(os.path.join(path, f"mem{MEMBERS[-1]:03d}_L{LEAD:04d}.shard"))
    with pytest.raises(AggregateBuildError, match="both components of the same members"):
        build_container_fields(_reader(store), "wind_10m", LEAD, expected_members=30)


def test_precipitation_is_refused_without_each_flag(store: str) -> None:
    """The phases are a function of all four flags, so one missing flag is not a phase."""
    import shutil

    for flag in ("crain", "csnow", "cfrzr", "cicep"):
        shutil.rmtree(os.path.join(store, "__staging__", "v1", flag))
        with pytest.raises(AggregateBuildError, match="not staged at lead"):
            build_container_fields(
                _reader(store), "precipitation_amount_3h", LEAD, expected_members=30
            )
        # Restore for the next iteration by re-staging that flag.
        rng = np.random.default_rng(1)
        for member in MEMBERS:
            for lead in (LEAD, PREDECESSOR_LEAD):
                _stage(store, flag, member, lead, _plane(flag, member, rng))


def test_precipitation_is_refused_when_a_flag_describes_other_members(store: str) -> None:
    """Phases computed for a different member set than the amount are not this lead's phases."""
    path = os.path.join(store, "__staging__", "v1", "crain")
    os.remove(os.path.join(path, f"mem{MEMBERS[-1]:03d}_L{LEAD:04d}.shard"))
    with pytest.raises(AggregateBuildError, match="cannot be computed for a different member set"):
        build_container_fields(
            _reader(store), "precipitation_amount_3h", LEAD, expected_members=30
        )


def test_a_predecessor_with_a_different_member_set_is_refused(store: str) -> None:
    """A transition between different member sets is not a transition."""
    for variable in ("precipitation_amount_3h", "crain"):
        path = os.path.join(store, "__staging__", "v1", variable)
        os.remove(os.path.join(path, f"mem{MEMBERS[-1]:03d}_L{PREDECESSOR_LEAD:04d}.shard"))
    # The member sets are compared first, because a mismatched predecessor is the more specific
    # fault: it says the two intervals describe different ensembles.
    with pytest.raises(AggregateBuildError, match="is not a transition"):
        build_container_fields(
            _reader(store), "precipitation_amount_3h", LEAD, expected_members=30
        )


def test_a_lead_without_a_predecessor_produces_absent_previous_planes(store: str) -> None:
    """Lead 3 is not a reset lead, so it has no predecessor -- and that is not an error.

    The phase group's second six fields are NaN rather than zero: a zero would claim every
    member's predecessor was dry, which is a different statement from "there was none".
    """
    variable = "precipitation_amount_3h"
    for member in MEMBERS:
        _stage(store, variable, member, 3, _plane(variable, member, np.random.default_rng(member)))
        for flag in ("crain", "csnow", "cfrzr", "cicep"):
            _stage(store, flag, member, 3, _plane(flag, member, np.random.default_rng(member)))
    fields, _count = build_container_fields(
        _reader(store), variable, 3, expected_members=30
    )
    # Field 0 is the count, then the 19 distribution levels, then the twelve phase planes.
    previous = fields[1 + 19 + 6 : 1 + 19 + 12]
    assert np.isnan(previous).all()


def test_a_predecessor_the_wave_is_still_filling_is_refused_rather_than_absent(store: str) -> None:
    """The distinction the transition group's correctness rests on.

    A lead's predecessor is a different work item of the same wave, so at a lead's first
    publication the predecessor may simply be later rather than absent. The classifier reports
    ``persistent_rain`` for an absent predecessor and ``dry_to_rain`` for a dry one, so encoding
    the absent reading is a definite but wrong answer -- and the next patch would silently
    correct it, which is exactly the kind of error nobody notices.
    """
    predecessor_path = os.path.join(
        store, "__staging__", "v1", "precipitation_amount_3h"
    )
    for member in MEMBERS:
        os.remove(
            os.path.join(predecessor_path, f"mem{member:03d}_L{PREDECESSOR_LEAD:04d}.shard")
        )
    with pytest.raises(AggregateBuildError, match="would be encoded as a definite one"):
        build_container_fields(
            _reader(store),
            "precipitation_amount_3h",
            LEAD,
            expected_members=30,
            wave_leads=(LEAD, PREDECESSOR_LEAD),
        )
    # The same store without the wave's target set is the "cannot tell" case: the builder reads
    # the predecessor when present and accepts its absence, which is what a repair caller needs.
    fields, _count = build_container_fields(
        _reader(store), "precipitation_amount_3h", LEAD, expected_members=30
    )
    assert len(fields) == aggregate_fields_for("precipitation_amount_3h").n_fields


def test_a_predecessor_outside_the_wave_is_accepted_as_absent(store: str) -> None:
    """A lead whose predecessor predates this wave has none, and that is a definite answer."""
    import shutil

    shutil.rmtree(os.path.join(store, "__staging__", "v1", "precipitation_amount_3h"))
    for member in MEMBERS:
        _stage(store, "precipitation_amount_3h", member, LEAD, _plane("precipitation_amount_3h", member, np.random.default_rng(member)))
    fields, _count = build_container_fields(
        _reader(store),
        "precipitation_amount_3h",
        LEAD,
        expected_members=30,
        wave_leads=(LEAD,),
    )
    layout = aggregate_fields_for("precipitation_amount_3h")
    previous = fields[layout.group_slice("phase")][6:]
    assert np.isnan(previous).all()


def test_cloud_censoring_needs_only_the_variable_itself(store: str) -> None:
    """A cloud variable's groups read nothing else, so they survive the flags being absent."""
    import shutil

    for flag in ("crain", "csnow", "cfrzr", "cicep"):
        shutil.rmtree(os.path.join(store, "__staging__", "v1", flag))
    fields, count = build_container_fields(
        _reader(store), "cloud_ceiling", LEAD, expected_members=30
    )
    assert count == len(MEMBERS)
    assert len(fields) == aggregate_fields_for("cloud_ceiling").n_fields


def test_an_unclassified_variable_is_refused(store: str) -> None:
    with pytest.raises(AggregateBuildError, match="no approved aggregate encoding"):
        build_container_fields(_reader(store), "mystery", LEAD, expected_members=30)


def test_a_flag_fraction_is_the_per_cell_share_of_set_members(store: str) -> None:
    """A flag's representation is its fraction, over the members that have a value at all."""
    reader = _reader(store)
    fields, _count = build_container_fields(reader, "crain", LEAD, expected_members=30)
    _members, stack = reader.stack("crain", LEAD)
    assert fields[0].shape == (GRID, GRID)
    assert fields[1] == pytest.approx((stack >= 0.5).mean(axis=0), abs=1e-6)
    assert ((fields[1] >= 0.0) & (fields[1] <= 1.0)).all()


def test_the_reader_lists_each_variable_once(store: str) -> None:
    """A member-major walk asks for the same keys thirty times; re-listing would be thirty LISTs."""
    reader = _reader(store)
    calls: list[str] = []
    original = reader.keys

    def counted(variable: str):
        calls.append(variable)
        return original(variable)

    reader.keys = counted  # type: ignore[method-assign]
    reader.plane("temperature_2m", LEAD, MEMBERS[0])
    reader.plane("temperature_2m", LEAD, MEMBERS[1])
    reader.members("temperature_2m", LEAD)
    assert calls.count("temperature_2m") >= 3  # the method is called, the listing happens once
    assert set(reader._keys) == {"temperature_2m"}
