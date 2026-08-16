"""Tests for the stable marker / protocol-version / manifest layer.

Covers atomic marker state transitions, version-sidecar semantics
(absent/recognized/malformed/unknown/unreadable), the serving-generation rule,
and the manifest schema. Local filesystem tests always run; MinIO tests run
when ``WEATHER_TEST_MINIO=1`` and the endpoint is reachable.
"""

from __future__ import annotations

import json
import os

import pytest

from ingestion.core.markers import (
    HYBRID,
    LEGACY,
    MARKER_V1,
    MarkerError,
    ProtocolVersionError,
    list_region_marker_keys,
    manifest_key,
    marker_body,
    marker_key,
    read_manifest,
    read_protocol_version,
    read_region_marker,
    region_evidence_fingerprint,
    write_manifest,
    write_protocol_version,
    write_region_marker,
)
from domain.locks import manifest_canonical_json, serving_state_fingerprint, sha256_hex


def _payload(state: str = "complete", gen: str = "g1", **over) -> dict[str, object]:
    base = marker_body(
        lead_time_hours=6,
        member=None,
        state=state,
        generation=gen,
        expected_write_set_fingerprint=sha256_hex("write-set"),
        required_materialized_object_keys=["temperature_2m/0.0.0"],
        intentionally_omitted_fill_chunks=["temperature_2m/0.0.1"],
    )
    base.update(over)
    return base


def test_marker_key_scoped_and_safe() -> None:
    k = marker_key("s3://b/prefix", lead_time_hours=6, member=None)
    assert k == "__commit__/v1/regions/det_L0006.json"
    assert "s3://" not in k
    k2 = marker_key("s3://b/prefix", lead_time_hours=6, member=17)
    assert k2 == "__commit__/v1/regions/mem017_L0006.json"


def test_manifest_key() -> None:
    assert manifest_key() == "__commit__/v1/manifest.json"


def test_local_marker_atomic_write_read(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    write_region_marker(store, lead_time_hours=6, member=None, payload=_payload("updating", "g1"))
    p = read_region_marker(store, lead_time_hours=6, member=None)
    assert p["state"] == "updating"
    assert p["generation"] == "g1"
    # Atomic overwrite to COMPLETE on the SAME stable key.
    write_region_marker(store, lead_time_hours=6, member=None, payload=_payload("complete", "g1"))
    p2 = read_region_marker(store, lead_time_hours=6, member=None)
    assert p2["state"] == "complete"
    assert p2["generation"] == "g1"
    # No leftover temp files.
    assert not [f for f in os.listdir(os.path.join(store, "__commit__", "v1", "regions")) if f.startswith(".sidecar-")]


def test_marker_absent_returns_absent_state(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    p = read_region_marker(store, lead_time_hours=6, member=None)
    assert p["state"] == "absent"


def test_list_region_marker_keys(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    write_region_marker(store, lead_time_hours=6, member=None, payload=_payload())
    write_region_marker(store, lead_time_hours=12, member=1, payload=_payload())
    keys = list_region_marker_keys(store)
    assert len(keys) == 2
    assert all(k.endswith(".json") for k in keys)


def test_protocol_version_absent_is_legacy(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    assert read_protocol_version(store) == LEGACY


def test_protocol_version_recognized(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    write_protocol_version(store, MARKER_V1)
    assert read_protocol_version(store) == MARKER_V1
    write_protocol_version(store, HYBRID)
    assert read_protocol_version(store) == HYBRID


def test_protocol_version_unknown_is_hard_failure(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    os.makedirs(os.path.join(store, "__commit__", "v1"), exist_ok=True)
    with open(os.path.join(store, "__commit__", "v1", "version"), "w", encoding="utf-8") as fh:
        fh.write("future_v2")
    with pytest.raises(ProtocolVersionError):
        read_protocol_version(store)


def test_protocol_version_malformed_is_hard_failure(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    os.makedirs(os.path.join(store, "__commit__", "v1"), exist_ok=True)
    with open(os.path.join(store, "__commit__", "v1", "version"), "wb") as fh:
        fh.write(b"\xff\xfe\x00")
    with pytest.raises(ProtocolVersionError):
        read_protocol_version(store)


def test_write_unknown_version_rejected(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    with pytest.raises(ProtocolVersionError):
        write_protocol_version(store, "bogus")


def test_manifest_roundtrip_and_validation(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    payload = {
        "manifest_schema_version": 1,
        "store_protocol_mode": MARKER_V1,
        "generation": "0123456789abcdef0123456789abcdef",
        "run_identity": {
            "model_version_id": "version_gfs_v1.0",
            "cycle_time": "2026-07-22T00:00:00Z",
            "is_ensemble": False,
        },
        "canonical_store_identity_hash": sha256_hex("ident"),
        "serving_state_fingerprint": sha256_hex("serving"),
        "committed_state_fingerprint": sha256_hex("committed"),
        "store_schema_fingerprint": sha256_hex("schema"),
        "region_marker_set_fingerprint": sha256_hex("markers"),
        "legacy_region_evidence_fingerprint": None,
    }
    write_manifest(store, payload)
    read = read_manifest(store)
    assert read == payload
    # Canonical serialization: sorted keys, no NaN.
    raw = manifest_canonical_json(payload)
    assert json.loads(raw) == payload


def test_manifest_missing_returns_none(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    assert read_manifest(store) is None


def test_manifest_malformed_is_hard_failure(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    os.makedirs(os.path.join(store, "__commit__", "v1"), exist_ok=True)
    with open(os.path.join(store, "__commit__", "v1", "manifest.json"), "wb") as fh:
        fh.write(b"not json")
    with pytest.raises(ProtocolVersionError):
        read_manifest(store)


def test_marker_malformed_is_marker_error(tmp_path) -> None:
    store = str(tmp_path / "cycle.zarr")
    key = marker_key(store, lead_time_hours=6, member=None)
    full = os.path.join(store, *key.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as fh:
        fh.write(b"garbage")
    with pytest.raises(MarkerError):
        read_region_marker(store, lead_time_hours=6, member=None)


def test_serving_generation_rule_same_set_new_generation() -> None:
    run = {"model_version_id": "v", "cycle_time": "2026-07-22T00:00:00Z", "is_ensemble": False}
    regions_g1 = [{
        "region": "det_L0006",
        "state": "complete",
        "generation": "g1",
        "write_set_fp": sha256_hex("w"),
        "materialized_fp": sha256_hex("m"),
        "omitted_fp": sha256_hex("o"),
    }]
    regions_g2 = [{**regions_g1[0], "generation": "g2"}]
    f1 = serving_state_fingerprint(
        store_protocol_mode=MARKER_V1, run_identity=run,
        store_schema_fingerprint="schema", region_serving_states=regions_g1,
    )
    f2 = serving_state_fingerprint(
        store_protocol_mode=MARKER_V1, run_identity=run,
        store_schema_fingerprint="schema", region_serving_states=regions_g2,
    )
    # Same committed set + new marker generation -> new fingerprint.
    assert f1 != f2
    # Pure catalog retry (identical serving state) -> same fingerprint.
    assert serving_state_fingerprint(
        store_protocol_mode=MARKER_V1, run_identity=run,
        store_schema_fingerprint="schema", region_serving_states=regions_g1,
    ) == f1


def test_region_evidence_fingerprint_deterministic() -> None:
    a = region_evidence_fingerprint("s3://b/k", ["det_L0006", "det_L0012"])
    b = region_evidence_fingerprint("s3://b/k", ["det_L0006", "det_L0012"])
    c = region_evidence_fingerprint("s3://b/k", ["det_L0012", "det_L0006"])
    assert a == b == c  # sorted inside the helper


def test_marker_body_member_none_vs_zero_distinct() -> None:
    b1 = marker_body(
        lead_time_hours=6, member=None, state="complete", generation="g",
        expected_write_set_fingerprint="w", required_materialized_object_keys=[],
        intentionally_omitted_fill_chunks=[],
    )
    b0 = marker_body(
        lead_time_hours=6, member=0, state="complete", generation="g",
        expected_write_set_fingerprint="w", required_materialized_object_keys=[],
        intentionally_omitted_fill_chunks=[],
    )
    assert b1["logical_region"] == {"lead_time_hours": 6}
    assert b0["logical_region"] == {"lead_time_hours": 6, "member": 0}
    assert b1 != b0


def test_minio_marker_atomic_put(minio_store: str) -> None:
    """MinIO/S3: a single PutObject atomically replaces the stable marker key."""
    store = minio_store
    write_region_marker(store, lead_time_hours=6, member=None, payload=_payload("updating", "g1"))
    p = read_region_marker(store, lead_time_hours=6, member=None)
    assert p["state"] == "updating"
    assert p["generation"] == "g1"
    # Atomic overwrite to COMPLETE on the same stable key.
    write_region_marker(store, lead_time_hours=6, member=None, payload=_payload("complete", "g1"))
    p2 = read_region_marker(store, lead_time_hours=6, member=None)
    assert p2["state"] == "complete"
    # Listing sees exactly one marker key (no leftover state keys).
    keys = list_region_marker_keys(store)
    assert len(keys) == 1


def test_minio_manifest_roundtrip(minio_store: str) -> None:
    store = minio_store
    payload = {
        "manifest_schema_version": 1,
        "store_protocol_mode": MARKER_V1,
        "generation": "0123456789abcdef0123456789abcdef",
        "run_identity": {
            "model_version_id": "version_gfs_v1.0",
            "cycle_time": "2026-07-22T00:00:00Z",
            "is_ensemble": False,
        },
        "canonical_store_identity_hash": sha256_hex("ident"),
        "serving_state_fingerprint": sha256_hex("serving"),
        "committed_state_fingerprint": sha256_hex("committed"),
        "store_schema_fingerprint": sha256_hex("schema"),
        "region_marker_set_fingerprint": sha256_hex("markers"),
        "legacy_region_evidence_fingerprint": None,
    }
    write_manifest(store, payload)
    assert read_manifest(store) == payload
