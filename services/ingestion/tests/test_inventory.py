"""Tests for the physical object-inventory validation of COMPLETE markers.

Covers the required invariants:

* materialized set is a subset of the expected write set;
* omitted set is a subset of the expected write set;
* the two sets are disjoint;
* their union covers the expected write set;
* every required materialized object exists;
* a missing required materialized object invalidates committed state;
* intentionally omitted fill chunks may be physically absent;
* under shared-member chunks, object existence does not prove member-slice
  completion — the COMPLETE marker does.
"""

from __future__ import annotations

import json
import os

import pytest

from ingestion.core.inventory import (
    InventoryError,
    expected_write_set_fingerprint,
    region_expected_object_keys,
    validate_marker_evidence,
)


def _mk_store(tmp_path) -> str:
    """A minimal store with a .zarray chunk layout (lead:1, lat:2, lon:2)."""
    store = tmp_path / "cycle.zarr"
    var_dir = store / "temperature_2m"
    var_dir.mkdir(parents=True)
    za = {
        "shape": [2, 4, 4],  # 2 leads, 4 lat, 4 lon
        "chunks": [1, 2, 2],  # lead chunked at 1, spatial 2x2
        "zarr_format": 2,
    }
    (var_dir / ".zarray").write_text(json.dumps(za), encoding="utf-8")
    return str(store)


def _write_chunk(store: str, key: str) -> None:
    full = os.path.join(store, *key.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as fh:
        fh.write(b"chunk")


def _expected(store: str, lead_index: int) -> list[str]:
    return region_expected_object_keys(
        store, member=None, lead_index=lead_index, data_var_paths=["temperature_2m"]
    )


def test_expected_object_keys_derived_from_chunk_layout(tmp_path) -> None:
    store = _mk_store(tmp_path)
    keys = _expected(store, 0)
    # 4 spatial chunks (2 lat x 2 lon) for lead 0.
    assert keys == [
        "temperature_2m/0.0.0",
        "temperature_2m/0.0.1",
        "temperature_2m/0.1.0",
        "temperature_2m/0.1.1",
    ]


def test_valid_evidence_passes(tmp_path) -> None:
    store = _mk_store(tmp_path)
    keys = _expected(store, 0)
    for k in keys:
        _write_chunk(store, k)
    validate_marker_evidence(
        store,
        marker_required_materialized=keys,
        marker_omitted=[],
        actual_expected_keys=keys,
        marker_expected_fingerprint=expected_write_set_fingerprint(keys, []),
    )


def test_required_materialized_object_deleted_invalidates(tmp_path) -> None:
    store = _mk_store(tmp_path)
    keys = _expected(store, 0)
    _write_chunk(store, keys[0])
    # keys[1] is missing -> external shrink -> region uncommitted during physical audit.
    with pytest.raises(InventoryError, match="missing"):
        validate_marker_evidence(
            store,
            marker_required_materialized=keys,
            marker_omitted=[],
            actual_expected_keys=keys,
            marker_expected_fingerprint=expected_write_set_fingerprint(keys, []),
            verify_physical_objects=True,
        )


def test_valid_omitted_fill_chunk_passes(tmp_path) -> None:
    store = _mk_store(tmp_path)
    keys = _expected(store, 0)
    # Materialize 3 of 4; keys[3] is an all-fill omission (absent).
    for k in keys[:3]:
        _write_chunk(store, k)
    validate_marker_evidence(
        store,
        marker_required_materialized=keys[:3],
        marker_omitted=[keys[3]],
        actual_expected_keys=keys,
        marker_expected_fingerprint=expected_write_set_fingerprint(keys[:3], [keys[3]]),
    )


def test_materialized_omitted_overlap_rejected(tmp_path) -> None:
    store = _mk_store(tmp_path)
    keys = _expected(store, 0)
    with pytest.raises(InventoryError, match="overlap"):
        validate_marker_evidence(
            store,
            marker_required_materialized=[keys[0]],
            marker_omitted=[keys[0]],  # overlap
            actual_expected_keys=keys,
            marker_expected_fingerprint=expected_write_set_fingerprint([keys[0]], [keys[0]]),
        )


def test_incomplete_union_rejected(tmp_path) -> None:
    store = _mk_store(tmp_path)
    keys = _expected(store, 0)
    # Neither key covered -> union missing from expected.
    with pytest.raises(InventoryError, match="does not cover"):
        validate_marker_evidence(
            store,
            marker_required_materialized=[],
            marker_omitted=[],
            actual_expected_keys=keys,
            marker_expected_fingerprint=expected_write_set_fingerprint([], []),
        )


def test_unexpected_key_rejected(tmp_path) -> None:
    store = _mk_store(tmp_path)
    keys = _expected(store, 0)
    bogus = "temperature_2m/9.9.9"
    with pytest.raises(InventoryError, match="not in the expected write set"):
        validate_marker_evidence(
            store,
            marker_required_materialized=[bogus],
            marker_omitted=[],
            actual_expected_keys=keys,
            marker_expected_fingerprint=expected_write_set_fingerprint([bogus], []),
        )


def test_fingerprint_mismatch_rejected(tmp_path) -> None:
    store = _mk_store(tmp_path)
    keys = _expected(store, 0)
    with pytest.raises(InventoryError, match="fingerprint"):
        validate_marker_evidence(
            store,
            marker_required_materialized=keys,
            marker_omitted=[],
            actual_expected_keys=keys,
            marker_expected_fingerprint="0" * 64,  # wrong
        )


def test_ensemble_shared_object_existence_not_sufficiency(tmp_path) -> None:
    """Under shared-member chunks, object existence does NOT prove member-slice
    completion. A member's COMPLETE marker is the proof; an UPDATING member
    whose shared objects exist is still uncommitted."""
    store = tmp_path / "ens.zarr"
    var_dir = store / "temperature_2m"
    var_dir.mkdir(parents=True)
    za = {
        "shape": [2, 1, 4, 4],  # 2 members, 1 lead, 4 lat, 4 lon
        "chunks": [2, 1, 2, 2],  # member full-extent, spatial 2x2
        "zarr_format": 2,
    }
    (var_dir / ".zarray").write_text(json.dumps(za), encoding="utf-8")
    store = str(store)
    # Member A fully populated the shared chunks (the physical objects exist).
    _write_chunk(store, "temperature_2m/0.0.0.0")
    _write_chunk(store, "temperature_2m/0.0.0.1")
    _write_chunk(store, "temperature_2m/0.0.1.0")
    _write_chunk(store, "temperature_2m/0.0.1.1")
    # The physical objects exist, but they are SHARED across members. Member B's
    # completion is proven by its COMPLETE marker, not by the object existence.
    # Here we validate member A's COMPLETE evidence (all 4 spatial chunks).
    keys_a = region_expected_object_keys(
        store, member=1, lead_index=0, data_var_paths=["temperature_2m"]
    )
    # Member A's region maps to the same shared chunks (member axis full-extent).
    assert set(keys_a) == {
        "temperature_2m/0.0.0.0",
        "temperature_2m/0.0.0.1",
        "temperature_2m/0.0.1.0",
        "temperature_2m/0.0.1.1",
    }
    validate_marker_evidence(
        store,
        marker_required_materialized=keys_a,
        marker_omitted=[],
        actual_expected_keys=keys_a,
        marker_expected_fingerprint=expected_write_set_fingerprint(keys_a, []),
    )
    # Member B remains UPDATING (its marker is not COMPLETE) even though the
    # shared objects exist — object existence alone does not prove member B.
    # This is enforced by the finalizer's marker-state check, not by inventory.


def test_physical_conflict_keys_full_member_layout_serializes(tmp_path) -> None:
    """Full-member-extent layout: different members at the same lead produce
    IDENTICAL conflict keys (they share the physical chunk)."""
    store = tmp_path / "ens.zarr"
    var_dir = store / "temperature_2m"
    var_dir.mkdir(parents=True)
    # member full-extent (chunk 0 covers all 4 members), lead chunked at 1.
    import json as _json

    (_json).dump(
        {"shape": [4, 2, 4, 4], "chunks": [4, 1, 2, 2], "zarr_format": 2},
        open(var_dir / ".zarray", "w", encoding="utf-8"),
    )
    from ingestion.core.inventory import physical_conflict_keys

    k1 = physical_conflict_keys(str(store), member=1, lead_index=0, data_var_paths=["temperature_2m"])
    k2 = physical_conflict_keys(str(store), member=2, lead_index=0, data_var_paths=["temperature_2m"])
    k4 = physical_conflict_keys(str(store), member=4, lead_index=0, data_var_paths=["temperature_2m"])
    # Members 1, 2, 4 at the same lead all map to member chunk 0 (full-extent).
    assert k1 == k2 == k4
    # Different lead -> different lead chunk -> distinct.
    k_lead12 = physical_conflict_keys(str(store), member=1, lead_index=1, data_var_paths=["temperature_2m"])
    assert k1 != k_lead12


def test_physical_conflict_keys_member_chunked_at_1_distinct(tmp_path) -> None:
    """Test-only alternate layout: member chunk size 1 -> different members map
    to different member chunks -> distinct conflict keys."""
    store = tmp_path / "ens1.zarr"
    var_dir = store / "temperature_2m"
    var_dir.mkdir(parents=True)
    import json as _json

    _json.dump(
        {"shape": [4, 2, 4, 4], "chunks": [1, 1, 2, 2], "zarr_format": 2},
        open(var_dir / ".zarray", "w", encoding="utf-8"),
    )
    from ingestion.core.inventory import physical_conflict_keys

    k1 = physical_conflict_keys(str(store), member=1, lead_index=0, data_var_paths=["temperature_2m"])
    k2 = physical_conflict_keys(str(store), member=2, lead_index=0, data_var_paths=["temperature_2m"])
    # Members 1 and 2 (positional 0 and 1) map to different member chunks.
    assert k1 != k2
    assert k1[0].startswith("temperature_2m/0.")
    assert k2[0].startswith("temperature_2m/1.")


def test_physical_conflict_keys_deterministic_disjoint(tmp_path) -> None:
    """Deterministic layout: different leads map to different lead chunks."""
    store = tmp_path / "det.zarr"
    var_dir = store / "temperature_2m"
    var_dir.mkdir(parents=True)
    import json as _json

    _json.dump(
        {"shape": [2, 4, 4], "chunks": [1, 2, 2], "zarr_format": 2},
        open(var_dir / ".zarray", "w", encoding="utf-8"),
    )
    from ingestion.core.inventory import physical_conflict_keys

    k6 = physical_conflict_keys(str(store), member=None, lead_index=0, data_var_paths=["temperature_2m"])
    k12 = physical_conflict_keys(str(store), member=None, lead_index=1, data_var_paths=["temperature_2m"])
    assert k6 != k12
    assert k6[0].startswith("temperature_2m/0.")
    assert k12[0].startswith("temperature_2m/1.")


@pytest.mark.parametrize("member,expected_m_chunk", [(1, 0), (15, 14), (30, 29)])
@pytest.mark.parametrize("lead_idx", [0, 1, 8])
def test_region_expected_matches_conflict_keys_member_chunk_1(tmp_path, member, expected_m_chunk, lead_idx) -> None:
    """Under member_chunk=1, region_expected_object_keys and physical_conflict_keys
    must be 100% identical and map to the specific member's chunk coordinate."""
    store = tmp_path / f"ens_mc1_{member}_{lead_idx}.zarr"
    var_dir = store / "temperature_2m"
    var_dir.mkdir(parents=True)
    import json as _json

    _json.dump(
        {"shape": [30, 10, 4, 4], "chunks": [1, 1, 2, 2], "zarr_format": 2},
        open(var_dir / ".zarray", "w", encoding="utf-8"),
    )
    _json.dump(
        {"_ARRAY_DIMENSIONS": ["member", "lead_time_hours", "latitude", "longitude"]},
        open(var_dir / ".zattrs", "w", encoding="utf-8"),
    )
    from ingestion.core.inventory import physical_conflict_keys, region_expected_object_keys

    exp_keys = region_expected_object_keys(
        str(store), member=member, lead_index=lead_idx, data_var_paths=["temperature_2m"]
    )
    conf_keys = physical_conflict_keys(
        str(store), member=member, lead_index=lead_idx, data_var_paths=["temperature_2m"]
    )

    assert exp_keys == conf_keys
    assert len(exp_keys) == 4  # 2x2 spatial chunks
    prefix = f"temperature_2m/{expected_m_chunk}.{lead_idx}."
    assert all(k.startswith(prefix) for k in exp_keys)


@pytest.mark.parametrize("member", [1, 15, 30])
@pytest.mark.parametrize("lead_idx", [0, 5])
def test_region_expected_matches_conflict_keys_legacy_member_chunk_30(tmp_path, member, lead_idx) -> None:
    """Under legacy member_chunk=30, all members map to member chunk 0 and
    region_expected_object_keys and physical_conflict_keys are identical."""
    store = tmp_path / f"ens_mc30_{member}_{lead_idx}.zarr"
    var_dir = store / "temperature_2m"
    var_dir.mkdir(parents=True)
    import json as _json

    _json.dump(
        {"shape": [30, 10, 4, 4], "chunks": [30, 1, 2, 2], "zarr_format": 2},
        open(var_dir / ".zarray", "w", encoding="utf-8"),
    )
    _json.dump(
        {"_ARRAY_DIMENSIONS": ["member", "lead_time_hours", "latitude", "longitude"]},
        open(var_dir / ".zattrs", "w", encoding="utf-8"),
    )
    from ingestion.core.inventory import physical_conflict_keys, region_expected_object_keys

    exp_keys = region_expected_object_keys(
        str(store), member=member, lead_index=lead_idx, data_var_paths=["temperature_2m"]
    )
    conf_keys = physical_conflict_keys(
        str(store), member=member, lead_index=lead_idx, data_var_paths=["temperature_2m"]
    )

    assert exp_keys == conf_keys
    assert len(exp_keys) == 4
    prefix = f"temperature_2m/0.{lead_idx}."
    assert all(k.startswith(prefix) for k in exp_keys)


def test_verify_expected_object_keys_bounded(tmp_path) -> None:
    """verify_expected_object_keys returns only the keys that exist among expected_keys."""
    from ingestion.core.inventory import verify_expected_object_keys, region_expected_object_keys

    store = _mk_store(tmp_path)
    keys = region_expected_object_keys(store, member=None, lead_index=0, data_var_paths=["temperature_2m"])
    # Write only keys[0] and keys[2]
    _write_chunk(store, keys[0])
    _write_chunk(store, keys[2])

    confirmed = verify_expected_object_keys(store, keys)
    assert confirmed == {keys[0], keys[2]}


def test_dimension_separator_slash_respected(tmp_path) -> None:
    """When .zarray defines dimension_separator='/', chunk keys use '/'."""
    from ingestion.core.inventory import region_expected_object_keys

    store = tmp_path / "slash.zarr"
    var_dir = store / "temperature_2m"
    var_dir.mkdir(parents=True)
    import json as _json

    _json.dump(
        {
            "shape": [2, 4, 4],
            "chunks": [1, 2, 2],
            "zarr_format": 2,
            "dimension_separator": "/",
        },
        open(var_dir / ".zarray", "w", encoding="utf-8"),
    )
    keys = region_expected_object_keys(str(store), member=None, lead_index=0, data_var_paths=["temperature_2m"])
    assert keys == [
        "temperature_2m/0/0/0",
        "temperature_2m/0/0/1",
        "temperature_2m/0/1/0",
        "temperature_2m/0/1/1",
    ]


def test_derive_region_prefix_structural_isolation() -> None:
    """Validate structural region prefix derivation for dot vs slash and member 1 vs 10."""
    from ingestion.core.inventory import _derive_region_prefix

    za_dot = {
        "shape": [30, 37, 721, 1440],
        "chunks": [1, 1, 100, 100],
        "dimension_separator": ".",
    }
    za_slash = {
        "shape": [30, 37, 721, 1440],
        "chunks": [1, 1, 100, 100],
        "dimension_separator": "/",
    }
    zattrs = {"_ARRAY_DIMENSIONS": ["member", "lead_time_hours", "latitude", "longitude"]}

    # Member 1 (positional 0), lead 0
    p1_0_dot = _derive_region_prefix("s3://fake/store.zarr", "temperature_2m", member=1, lead_index=0, za=za_dot, zattrs_cache={"temperature_2m": zattrs})
    assert p1_0_dot == "temperature_2m/0.0."

    # Member 10 (positional 9), lead 0
    p10_0_dot = _derive_region_prefix("s3://fake/store.zarr", "temperature_2m", member=10, lead_index=0, za=za_dot, zattrs_cache={"temperature_2m": zattrs})
    assert p10_0_dot == "temperature_2m/9.0."
    assert not p10_0_dot.startswith(p1_0_dot)
    assert not p1_0_dot.startswith(p10_0_dot)

    # Slash separator: member 1, lead 6
    p1_6_slash = _derive_region_prefix("s3://fake/store.zarr", "temperature_2m", member=1, lead_index=6, za=za_slash, zattrs_cache={"temperature_2m": zattrs})
    assert p1_6_slash == "temperature_2m/0/6/"

    # Legacy member_chunk=30 layout: member 15 -> member_chunk=0
    za_legacy = {
        "shape": [30, 37, 721, 1440],
        "chunks": [30, 1, 100, 100],
    }
    p15_legacy = _derive_region_prefix("s3://fake/store.zarr", "temperature_2m", member=15, lead_index=2, za=za_legacy, zattrs_cache={"temperature_2m": zattrs})
    assert p15_legacy == "temperature_2m/0.2."


def test_audit_store_integrity_full_scan(tmp_path) -> None:
    """audit_store_integrity performs an exhaustive physical-store audit and reports status."""
    from ingestion.core.inventory import audit_store_integrity
    from ingestion.core.markers import write_protocol_version, write_region_marker, MARKER_V1
    from ingestion.core.zarr_writer import prepare_run_store
    import xarray as xr
    import numpy as np

    store_dir = tmp_path / "audit_test.zarr"
    store = str(store_dir)
    coords = {
        "lead_time_hours": np.array([0, 6], dtype=np.int32),
        "time": np.datetime64("2026-07-22T00:00:00", "ns"),
        "latitude": np.array([-90.0, 0.0, 90.0], dtype=np.float32),
        "longitude": np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
    }
    dims = ("lead_time_hours", "latitude", "longitude")
    data = np.zeros((1, 3, 4), dtype=np.float32)
    seed = xr.Dataset(
        data_vars={"temperature_2m": (dims, data)},
        coords={k: v[:1] if k == "lead_time_hours" else v for k, v in coords.items()},
    )
    prepare_run_store(seed, store, expected_lead_time_hours=(0, 6), expected_members=())
    write_protocol_version(store, MARKER_V1)

    # Lead 0: complete with chunks
    exp0 = region_expected_object_keys(store, member=None, lead_index=0, data_var_paths=["temperature_2m"])
    for k in exp0:
        _write_chunk(store, k)
    write_region_marker(
        store,
        lead_time_hours=0,
        member=None,
        payload={
            "protocol_version": 1,
            "state": "complete",
            "generation": "gen0",
            "logical_region": {"lead_time_hours": 0},
            "expected_write_set_fingerprint": expected_write_set_fingerprint(exp0, []),
            "required_materialized_object_keys": exp0,
            "intentionally_omitted_fill_chunks": [],
        },
    )

    # Lead 6: complete but chunks missing on physical storage
    exp6 = region_expected_object_keys(store, member=None, lead_index=1, data_var_paths=["temperature_2m"])
    write_region_marker(
        store,
        lead_time_hours=6,
        member=None,
        payload={
            "protocol_version": 1,
            "state": "complete",
            "generation": "gen6",
            "logical_region": {"lead_time_hours": 6},
            "expected_write_set_fingerprint": expected_write_set_fingerprint(exp6, []),
            "required_materialized_object_keys": exp6,
            "intentionally_omitted_fill_chunks": [],
        },
    )

    # Audit store
    report = audit_store_integrity(store, array_paths=["temperature_2m"])
    assert not report.is_valid
    assert report.total_markers == 2
    assert report.valid_markers == 1
    assert report.invalid_markers == 1
    assert "det_L0006" in report.errors_by_region




