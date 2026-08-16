"""Stable per-region completion-marker and protocol-version protocol.

The marker protocol records, per logical forecast region, whether a compliant
writer has completed writing that region. A single stable marker key per
region holds the state and generation in its **body**; the key never changes.

Marker states:

    UPDATING(generation)     declared before the first data object changes
    COMPLETE(generation)     written only after every expected data object
                             has been written (the last store-side operation)

Atomic state transitions:

    MinIO/S3:      one PutObject of the stable marker key (atomic replace)
    local filesystem: tempfile + os.replace (atomic file replacement)

The marker body is the compliant writer's attestation of the logical region
completion. The finalizer validates structural consistency (schema
fingerprint, materialized/omitted sets disjoint and covering the expected
write set, required materialized objects exist); it cannot independently prove
the fill values of omitted chunks without content evidence (out of scope).

The store-level protocol-version sidecar distinguishes:

    absent               -> legacy
    "hybrid_marker_v1"   -> per-region marker override of the legacy rule
    "marker_v1"          -> strict marker rule

Malformed, unknown, or unreadable versions are a **hard failure** (never a
silent fallback to legacy).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Mapping

import s3fs  # type: ignore[import-untyped]

from domain.locks import (
    logical_region_encoding,
    manifest_canonical_json,
    sha256_hex,
)

#: Protocol version identifiers.
LEGACY = "legacy"
HYBRID = "hybrid_marker_v1"
MARKER_V1 = "marker_v1"
RECOGNIZED_VERSIONS = frozenset({HYBRID, MARKER_V1})

#: Root of the commit sidecar namespace under a store.
_COMMIT_ROOT = "__commit__"
_VERSION_ROOT = f"{_COMMIT_ROOT}/v1/version"
_MARKER_ROOT = f"{_COMMIT_ROOT}/v1/regions"
_MANIFEST_PATH = f"{_COMMIT_ROOT}/v1/manifest.json"


class ProtocolVersionError(RuntimeError):
    """Raised when the store's protocol-version sidecar is malformed/unknown."""


class MarkerError(RuntimeError):
    """Raised for invalid marker state/structural inconsistencies."""


def version_sidecar_key() -> str:
    return _VERSION_ROOT


def marker_key(store_path: str, *, lead_time_hours: int, member: int | None) -> str:
    """Return the stable marker object key for a logical region.

    The key is scoped under the store's commit namespace and uses the
    object-key-safe logical region encoding. The store itself scopes the
    namespace, so no raw canonical store identity appears in the filename.

    Args:
        store_path: The store path/URL (used only to resolve the storage root;
            the marker key itself is a relative object key).
        lead_time_hours: The forecast lead in hours.
        member: The ensemble member identity, or ``None`` for deterministic.

    Returns:
        The marker object key (relative to the store root).
    """
    region = logical_region_encoding(lead_time_hours=lead_time_hours, member=member)
    return f"{_MARKER_ROOT}/{region}.json"


def manifest_key() -> str:
    return _MANIFEST_PATH


def _storage_backend(store_path: str) -> tuple[str, str]:
    """Resolve a store path to (backend_kind, backend_root).

    Returns ``("s3", "bucket/prefix")`` or ``("local", "<abs path>")``.
    """
    if store_path.startswith("s3://"):
        rest = store_path[len("s3://") :].strip("/")
        return "s3", rest
    path = store_path
    if path.startswith("file://"):
        path = path[len("file://") :]
    return "local", os.path.abspath(os.path.normpath(path))


def _s3_fs() -> "s3fs.S3FileSystem":
    from ingestion.core.config import settings

    scheme = "https" if settings.MINIO_SECURE else "http"
    return s3fs.S3FileSystem(
        key=settings.MINIO_ACCESS_KEY,
        secret=settings.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": f"{scheme}://{settings.MINIO_ENDPOINT}"},
    )


def _read_object(store_path: str, rel_key: str) -> bytes | None:
    """Read a sidecar object, returning None when it is absent."""
    backend, root = _storage_backend(store_path)
    if backend == "local":
        full = os.path.join(root, *rel_key.split("/"))
        try:
            with open(full, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None
    fs = _s3_fs()
    full = f"{root}/{rel_key}"
    try:
        raw = fs.cat_file(full)
        # cat_file is untyped in the s3fs stubs; narrow to bytes.
        return raw if isinstance(raw, bytes) else bytes(raw)
    except FileNotFoundError:
        return None


def _write_object_atomic(store_path: str, rel_key: str, data: bytes) -> None:
    """Write a sidecar object atomically.

    MinIO/S3: a single PutObject (atomic replace of the object key).
    Local: tempfile + os.replace (atomic file replacement).
    """
    backend, root = _storage_backend(store_path)
    if backend == "local":
        full = os.path.join(root, *rel_key.split("/"))
        parent = os.path.dirname(full)
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".sidecar-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, full)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return
    fs = _s3_fs()
    full = f"{root}/{rel_key}"
    fs.pipe_file(full, data)


def read_protocol_version(store_path: str) -> str:
    """Return the store's protocol-version mode.

    Returns:
        ``"legacy"`` (sidecar absent), ``"hybrid_marker_v1"``, or
        ``"marker_v1"``.

    Raises:
        ProtocolVersionError: If the sidecar is malformed, unknown, or
            unreadable (hard failure; never a silent legacy fallback).
    """
    raw = _read_object(store_path, _VERSION_ROOT)
    if raw is None:
        return LEGACY
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ProtocolVersionError(
            f"store protocol-version sidecar is not valid UTF-8: {store_path!r}"
        ) from exc
    if value not in RECOGNIZED_VERSIONS:
        raise ProtocolVersionError(
            f"store protocol-version sidecar has unknown value {value!r} at "
            f"{store_path!r}; refusing to fall back to legacy."
        )
    return value


def write_protocol_version(store_path: str, version: str) -> None:
    """Atomically write the store protocol-version sidecar."""
    if version not in RECOGNIZED_VERSIONS:
        raise ProtocolVersionError(f"cannot write unrecognized version {version!r}")
    _write_object_atomic(store_path, _VERSION_ROOT, version.encode("utf-8"))


def marker_body(
    *,
    lead_time_hours: int,
    member: int | None,
    state: str,
    generation: str,
    expected_write_set_fingerprint: str,
    required_materialized_object_keys: list[str],
    intentionally_omitted_fill_chunks: list[str],
) -> dict[str, object]:
    """Build a stable-marker body payload.

    Args:
        lead_time_hours: The forecast lead in hours.
        member: The ensemble member identity, or ``None`` for deterministic.
        state: ``"updating"`` or ``"complete"``.
        generation: The write-generation token.
        expected_write_set_fingerprint: sha256 of the expected write set.
        required_materialized_object_keys: Keys that MUST exist.
        intentionally_omitted_fill_chunks: Keys that are all-fill and
            deliberately absent (write_empty_chunks=False).

    Returns:
        A JSON-serializable marker payload.
    """
    logical_region = {"lead_time_hours": lead_time_hours}
    if member is not None:
        logical_region["member"] = member
    return {
        "protocol_version": 1,
        "state": state,
        "generation": generation,
        "logical_region": logical_region,
        "expected_write_set_fingerprint": expected_write_set_fingerprint,
        "required_materialized_object_keys": sorted(required_materialized_object_keys),
        "intentionally_omitted_fill_chunks": sorted(intentionally_omitted_fill_chunks),
    }


def _marker_payload(store_path: str, key: str) -> dict[str, object]:
    raw = _read_object(store_path, key)
    if raw is None:
        return {"state": "absent"}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MarkerError(f"marker {key} is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise MarkerError(f"marker {key} is not a JSON object")
    return payload


def read_region_marker(
    store_path: str,
    *,
    lead_time_hours: int,
    member: int | None,
) -> dict[str, object]:
    """Read a region's stable marker payload (``{"state": "absent"}`` if none)."""
    key = marker_key(store_path, lead_time_hours=lead_time_hours, member=member)
    return _marker_payload(store_path, key)


def write_region_marker(
    store_path: str,
    *,
    lead_time_hours: int,
    member: int | None,
    payload: Mapping[str, object],
) -> None:
    """Atomically write (overwrite) a region's stable marker object."""
    key = marker_key(store_path, lead_time_hours=lead_time_hours, member=member)
    data = manifest_canonical_json(dict(payload))
    _write_object_atomic(store_path, key, data)


def list_region_marker_keys(store_path: str) -> list[str]:
    """Return the object keys of every region marker under the commit namespace.

    Handles S3 listing pagination (a single LIST may not return all keys).
    """
    backend, root = _storage_backend(store_path)
    if backend == "local":
        base = os.path.join(root, *(_MARKER_ROOT.split("/")))
        out: list[str] = []
        if os.path.isdir(base):
            for name in os.listdir(base):
                if name.endswith(".json"):
                    out.append(f"{_MARKER_ROOT}/{name}")
        return sorted(out)
    fs = _s3_fs()
    full_prefix = f"{root}/{_MARKER_ROOT}"
    keys: list[str] = []
    # s3fs find returns all keys under a prefix (handles pagination internally).
    for item in fs.find(full_prefix):
        rel = item[len(root) + 1 :]
        keys.append(rel)
    return sorted(keys)


def read_manifest(store_path: str) -> dict[str, object] | None:
    """Read the committed manifest, returning None when absent.

    Raises:
        ProtocolVersionError: If the manifest is malformed/unreadable.
    """
    raw = _read_object(store_path, _MANIFEST_PATH)
    if raw is None:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolVersionError(f"committed manifest is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolVersionError("committed manifest is not a JSON object")
    return payload


def write_manifest(store_path: str, payload: Mapping[str, object]) -> None:
    """Atomically write the committed manifest."""
    _write_object_atomic(store_path, _MANIFEST_PATH, manifest_canonical_json(dict(payload)))


def region_evidence_fingerprint(store_path: str, region_keys: list[str]) -> str:
    """Fingerprint the persisted legacy-region evidence for a hybrid store.

    The evidence is the sorted list of marker-less legacy region identities
    (those that still use the legacy committed-state rule). The finalizer
    persists this so the serving-state fingerprint is stable and includes the
    legacy regions.
    """
    return sha256_hex("legacy-region-evidence", *sorted(region_keys))
