"""Pure unit tests for domain.locks canonical identity and key derivation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from domain.locks import (
    admission_key,
    canonical_storage_identity,
    logical_region_encoding,
    manifest_canonical_json,
    physical_conflict_identity,
    region_key,
    serving_state_fingerprint,
    sha256_hex,
    store_gate_key,
)


def test_sha256_hex_deterministic() -> None:
    assert sha256_hex("a", "b") == sha256_hex("a", "b")
    assert len(sha256_hex("a")) == 64
    assert sha256_hex("a", "b") != sha256_hex("a", "c")


def test_sha256_hex_none_distinct_from_empty() -> None:
    assert sha256_hex(None) != sha256_hex("")


def test_canonical_identity_s3_uses_config_endpoint() -> None:
    ident = canonical_storage_identity(
        "s3://weather-data/gfs/2026-07-21/00/cycle.zarr",
        endpoint="localhost:9000",
        secure=False,
    )
    assert "localhost:9000" in ident
    assert "weather-data" in ident
    assert "gfs/2026-07-21/00/cycle.zarr" in ident
    # No credentials or query in the identity.
    assert "access" not in ident
    assert "?X-Amz" not in ident


def test_canonical_identity_s3_strips_default_port() -> None:
    ident = canonical_storage_identity(
        "s3://weather-data/cycle.zarr", endpoint="s3.example.com:443", secure=True
    )
    assert "s3.example.com" in ident
    assert ":443" not in ident


def test_canonical_identity_s3_strips_http_default_port() -> None:
    ident = canonical_storage_identity(
        "s3://weather-data/cycle.zarr", endpoint="s3.example.com:80", secure=False
    )
    assert "s3.example.com" in ident
    assert ":80" not in ident


def test_canonical_identity_s3_bare_bucket() -> None:
    ident = canonical_storage_identity("s3://weather-data", endpoint="localhost:9000")
    assert "weather-data" in ident


def test_key_helpers_without_identity_kwargs() -> None:
    # Exercise the default-arg path of _resolve_identity.
    g = store_gate_key("s3://weather-data/cycle.zarr")
    r = region_key("s3://weather-data/cycle.zarr", "det_L006")
    a = admission_key("s3://weather-data/cycle.zarr")
    assert g & 0xF000000000000000 == 0x0000000000000000
    assert r & 0xF000000000000000 == 0x1000000000000000
    assert a & 0xF000000000000000 == 0x2000000000000000


def test_canonical_identity_s3_trailing_slash_normalized() -> None:
    a = canonical_storage_identity("s3://weather-data/gfs/", endpoint="e")
    b = canonical_storage_identity("s3://weather-data/gfs", endpoint="e")
    assert a == b


def test_canonical_identity_local_realpath_normcase() -> None:
    import tempfile

    d = tempfile.mkdtemp()
    path = os.path.join(d, "cycle.zarr")
    a = canonical_storage_identity(path)
    b = canonical_storage_identity(path + os.sep)
    assert a == b
    # realpath/abspath resolves symlinks and relativizes.
    assert canonical_storage_identity(path).startswith("local://")


def test_canonical_identity_local_forced_windows_normcase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Simulate the Windows ``os.name == "nt"`` branch on any host OS.

    ``locks.canonical_storage_identity`` only applies ``os.path.normcase`` to a
    local store path when ``os.name == "nt"`` (locks.py:117-118). On Linux CI
    that branch is never taken by real execution, so it stays uncovered.
    Patching the shared ``os`` module the function reads at call time forces
    the branch; a recording ``normcase`` proves the branch is entered, the
    call happens, and its normalized result is the exact value embedded in the
    returned canonical local-store identity.
    """
    calls: list[str] = []

    def fake_normcase(path: str) -> str:
        calls.append(path)
        return "C:\\DATA\\CYCLE.ZARR"

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os.path, "normcase", fake_normcase)
    ident = canonical_storage_identity(str(tmp_path / "cycle.zarr"))
    assert calls, "os.path.normcase was not invoked under the nt branch"
    assert ident == "local://C:\\DATA\\CYCLE.ZARR"


def test_canonical_identity_local_file_uri() -> None:
    ident = canonical_storage_identity("file:///tmp/x/y.zarr")
    assert ident.startswith("local://")


def test_canonical_identity_empty_raises() -> None:
    with pytest.raises(ValueError):
        canonical_storage_identity("")


def test_canonical_identity_empty_s3_raises() -> None:
    with pytest.raises(ValueError):
        canonical_storage_identity("s3://", endpoint="e")


def test_store_gate_and_region_namespaces_disjoint() -> None:
    # Any identity must never produce the same key across namespaces.
    for ident in ("a", "b", "weather-data/gfs", "x" * 100):
        g = store_gate_key(ident)
        r = region_key(ident, "det_L006")
        a = admission_key(ident)
        assert g != r != a
        # The top nibble selects the namespace.
        assert g & 0xF000000000000000 == 0x0000000000000000
        assert r & 0xF000000000000000 == 0x1000000000000000
        assert a & 0xF000000000000000 == 0x2000000000000000


def test_region_key_deterministic_and_collision_safe() -> None:
    k1 = region_key("s3://b/k", "det_L006", endpoint="e")
    k2 = region_key("s3://b/k", "det_L006", endpoint="e")
    k3 = region_key("s3://b/k", "det_L012", endpoint="e")
    assert k1 == k2
    assert k1 != k3


def test_logical_region_encoding() -> None:
    assert logical_region_encoding(lead_time_hours=6) == "det_L0006"
    assert logical_region_encoding(lead_time_hours=6, member=17) == "mem017_L0006"
    assert logical_region_encoding(lead_time_hours=0, member=0) == "mem000_L0000"


def test_physical_conflict_identity() -> None:
    assert (
        physical_conflict_identity(array_path="temperature_2m", chunk_coords=(0, 0, 1))
        == "temperature_2m/0.0.1"
    )


def test_manifest_canonical_json_sorted_utf8() -> None:
    payload = {"b": 1, "a": 2, "s": "é"}
    raw = manifest_canonical_json(payload)
    assert json.loads(raw) == {"a": 2, "b": 1, "s": "é"}
    assert raw == b'{"a":2,"b":1,"s":"\xc3\xa9"}'


def test_manifest_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        manifest_canonical_json({"x": float("nan")})


def test_serving_state_fingerprint_changes_with_generation() -> None:
    run = {"model_version_id": "v", "cycle_time": "2026-07-22T00:00:00Z", "is_ensemble": True}
    base = {
        "store_protocol_mode": "marker_v1",
        "run_identity": run,
        "store_schema_fingerprint": sha256_hex("schema"),
        "region_serving_states": [
            {
                "region": "mem017_L0006",
                "state": "complete",
                "generation": "g1",
                "write_set_fp": sha256_hex("w"),
                "materialized_fp": sha256_hex("m"),
                "omitted_fp": sha256_hex("o"),
            }
        ],
    }
    f1 = serving_state_fingerprint(**base)
    # Same committed set, new generation -> fingerprint changes.
    changed_states = [
        {**base["region_serving_states"][0], "generation": "g2"}
    ]
    f2 = serving_state_fingerprint(
        **{**base, "region_serving_states": changed_states}
    )
    assert f1 != f2
    # Identical inputs -> identical fingerprint.
    assert serving_state_fingerprint(**base) == f1


def test_serving_state_fingerprint_run_identity_order_insensitive() -> None:
    run1 = {"model_version_id": "v", "cycle_time": "t", "is_ensemble": False}
    run2 = {"is_ensemble": False, "cycle_time": "t", "model_version_id": "v"}
    a = serving_state_fingerprint(
        store_protocol_mode="marker_v1",
        run_identity=run1,
        store_schema_fingerprint="s",
        region_serving_states=[],
    )
    b = serving_state_fingerprint(
        store_protocol_mode="marker_v1",
        run_identity=run2,
        store_schema_fingerprint="s",
        region_serving_states=[],
    )
    assert a == b


def test_scheduler_leader_key_namespaced_and_stable() -> None:
    from domain.locks import (
        REALTIME_SCHEDULER_LEADER_IDENTITY,
        scheduler_leader_key,
        store_gate_key,
    )

    key = scheduler_leader_key()
    assert key == scheduler_leader_key(REALTIME_SCHEDULER_LEADER_IDENTITY)
    # Stable across calls, 64-bit, and disjoint from the store-gate namespace.
    assert 0 <= key < 2**64
    assert key & 0xF000000000000000 == 0x3000000000000000
    assert key != store_gate_key("s3://weather-data/gfs/2026-07-21/00/cycle.zarr")


def test_scheduler_leader_key_identity_sensitive() -> None:
    from domain.locks import scheduler_leader_key

    assert scheduler_leader_key("deployment-a") != scheduler_leader_key("deployment-b")
