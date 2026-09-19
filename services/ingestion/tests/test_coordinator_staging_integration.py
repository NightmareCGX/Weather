"""The region write's staging hook, exercised through the coordinator.

The hook is inert unless enabled, and this asserts both states. The enabled state matters
because of where it sits: staging runs inside the same retry loop as the member commit, and
the aggregate pass runs only after the COMPLETE marker exists. A staging write outside that
loop, or an aggregate before the marker, would let a partial member set be aggregated as if
it were complete.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr
from ingestion.core import aggregate_phase, aggregate_staging as staging
from ingestion.core.zarr_writer import encode_region_sharded_v1

VARIABLE = "temperature_2m"
LEAD = 6
GRID_LAT, GRID_LON = 128, 160


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


def test_enabled_phase_stages_and_aggregates_through_the_public_entry_points(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two hooks together, in the order the region write calls them.

    Order is the property under test: stage first, then (once the member is durable)
    aggregate. The aggregate is written to its real key, so a reader of record that has not
    been switched over continues to serve the member shards.
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
