"""Pure canonical storage identity and advisory-lock key derivation.

This module contains **no** SQLAlchemy or PostgreSQL connection behavior. It
provides the pure deterministic functions shared by the ingestion writer and
the API serving tier so the two services cannot drift on:

* canonical storage identity normalization;
* advisory-key namespace separation;
* logical-region encoding;
* physical-conflict identity encoding;
* canonical manifest JSON serialization;
* SHA-256 fingerprint helpers.

Collisions in the advisory-key space can only cause conservative
over-serialization (two distinct physical regions sharing a key serialize
instead of running concurrently). They can never cause a gate/region
self-deadlock because the namespaces are disjoint by construction and every
process acquires locks in one global order.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Mapping


def sha256_hex(*parts: object) -> str:
    """Return the lowercase 64-character SHA-256 hex of the joined parts.

    Each part is UTF-8 encoded. ``None`` is serialized as the literal string
    ``"<null>"`` so a missing optional field and the empty string never
    collide.

    Args:
        *parts: Objects contributing to the fingerprint. Order is significant
            and must be deterministic at every call site.

    Returns:
        A 64-character lowercase hex digest.
    """
    hasher = hashlib.sha256()
    for part in parts:
        if part is None:
            hasher.update(b"<null>")
        else:
            hasher.update(str(part).encode("utf-8"))
    return hasher.hexdigest()


#: Advisory-key namespaces. The top nibble of the 64-bit key selects the
#: namespace so different namespaces are disjoint by construction.
_NS_STORE_GATE = 0x0000000000000000
_NS_REGION_CONFLICT = 0x1000000000000000
_NS_ADMISSION = 0x2000000000000000

#: Mask that keeps the low 60 bits (the effective hash payload).
_HASH_MASK = 0x0FFFFFFFFFFFFFFF


def _namespaced_key(namespace: int, identity: str) -> int:
    """Derive a 60-bit advisory key in ``namespace`` for ``identity``."""
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    base = int.from_bytes(digest, "big")
    return (base & _HASH_MASK) | namespace


def canonical_storage_identity(
    store_path: str,
    *,
    endpoint: str | None = None,
    secure: bool = False,
) -> str:
    """Normalize a store path/URL to a canonical identity string.

    The identity is derived from the **actual resolved backend** (the endpoint
    comes from configuration, not assumed to be present in ``s3://`` URLs).
    Credentials and signed query strings never enter the identity.

    Args:
        store_path: A local path, ``file://`` URL, or ``s3://`` URL.
        endpoint: The resolved storage endpoint host[:port] (from config).
            Ignored for non-S3 paths.
        secure: Whether the storage scheme is HTTPS (from config). Ignored for
            non-S3 paths.

    Returns:
        A canonical identity string.
    """
    if not store_path:
        raise ValueError("store_path must be non-empty")
    if store_path.startswith("s3://"):
        rest = store_path[len("s3://") :].strip("/")
        if not rest:
            raise ValueError("Invalid S3 store URL: empty bucket/key")
        # Split bucket from the object prefix: the first path segment is the
        # bucket; the remainder is the object key. The bucket is always present
        # in a well-formed s3:// URL (the path cannot be empty here).
        if "/" in rest:
            bucket, _, key = rest.partition("/")
        else:
            bucket, key = rest, ""
        key = key.rstrip("/")
        scheme = "https" if secure else "http"
        host = (endpoint or "").lower()
        # Strip default ports for canonicalization.
        if scheme == "https" and host.endswith(":443"):
            host = host[: -len(":443")]
        if scheme == "http" and host.endswith(":80"):
            host = host[: -len(":80")]
        return f"s3://{scheme}://{host}/{bucket}/{key}"
    path = store_path
    if path.startswith("file://"):
        path = path[len("file://") :]
    path = os.path.abspath(os.path.realpath(os.path.normpath(path)))
    if os.name == "nt":
        path = os.path.normcase(path)
    return f"local://{path}"


def store_gate_key(store_path: str, **identity_kwargs: object) -> int:
    """Return the store-gate advisory key for a canonical store identity."""
    identity = _resolve_identity(store_path, identity_kwargs)
    return _namespaced_key(_NS_STORE_GATE, identity)


def admission_key(store_path: str, **identity_kwargs: object) -> int:
    """Return the admission-turnstile advisory key for a store."""
    identity = _resolve_identity(store_path, identity_kwargs)
    return _namespaced_key(_NS_ADMISSION, identity)


def region_key(
    store_path: str,
    region_id: str,
    **identity_kwargs: object,
) -> int:
    """Return a physical-region conflict advisory key.

    Args:
        store_path: The store path/URL.
        region_id: A logical or physical region identity (see
            :func:`logical_region_encoding` and
            :func:`physical_conflict_identity`).
        **identity_kwargs: Canonical identity inputs (endpoint/secure/bucket).
    """
    identity = _resolve_identity(store_path, identity_kwargs)
    return _namespaced_key(_NS_REGION_CONFLICT, f"{identity}\x00{region_id}")


def _resolve_identity(store_path: str, kwargs: Mapping[str, object]) -> str:
    endpoint = kwargs.get("endpoint")
    endpoint_str = endpoint if endpoint is None else str(endpoint)
    return canonical_storage_identity(
        store_path,
        endpoint=endpoint_str,
        secure=bool(kwargs.get("secure")),
    )


def logical_region_encoding(
    *,
    lead_time_hours: int,
    member: int | None = None,
) -> str:
    """Return an object-key-safe encoded logical region identity.

    The encoding uses explicit member-is-None checks (never ``member or ...``)
    so member 0, if ever supported, is encoded distinctly from ``None``.

    Args:
        lead_time_hours: The forecast lead in hours.
        member: The ensemble member identity, or ``None`` for deterministic.

    Returns:
        A filesystem/S3-object-key-safe string, e.g. ``"det_L006"`` or
        ``"mem017_L006"``.
    """
    lead_token = f"L{int(lead_time_hours):04d}"
    if member is None:
        return f"det_{lead_token}"
    return f"mem{int(member):03d}_{lead_token}"


def physical_conflict_identity(
    *,
    array_path: str,
    chunk_coords: tuple[int, ...],
) -> str:
    """Encode an array's physical chunk coordinates for conflict identity.

    Args:
        array_path: The Zarr array path (e.g. ``"temperature_2m"``).
        chunk_coords: The physical chunk coordinates in the array's chunk grid.

    Returns:
        A deterministic string (e.g. ``"temperature_2m/0.0.1"``).
    """
    coords = ".".join(str(int(c)) for c in chunk_coords)
    return f"{array_path}/{coords}"


def manifest_canonical_json(payload: Mapping[str, object]) -> bytes:
    """Serialize a manifest payload to the canonical UTF-8 JSON bytes.

    Uses ``sort_keys=True`` (the single canonicalization rule), compact
    separators, UTF-8 (no ASCII escaping), and rejects NaN/Infinity.

    Args:
        payload: The manifest payload (JSON-serializable).

    Returns:
        The canonical UTF-8 JSON bytes.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def serving_state_fingerprint(
    *,
    store_protocol_mode: str,
    run_identity: Mapping[str, object],
    store_schema_fingerprint: str,
    region_serving_states: list[Mapping[str, object]],
) -> str:
    """Compute the serving-state fingerprint for cache-generation identity.

    The fingerprint includes, per marker-controlled logical region: the region
    identity, marker state, marker generation, and the write-set/omission
    fingerprints. It deliberately includes marker generation so a same-set
    same-cycle data replacement (marker generation changes while the committed
    logical region set is unchanged) produces a new fingerprint.

    Args:
        store_protocol_mode: ``"legacy"``, ``"hybrid_marker_v1"``, or
            ``"marker_v1"``.
        run_identity: A mapping of ``model_version_id``/``cycle_time``/
            ``is_ensemble`` (order-insensitive; canonicalized by key).
        store_schema_fingerprint: The store's schema fingerprint.
        region_serving_states: A list of per-region serving-state mappings,
            each with deterministic keys (sorted at the call site).

    Returns:
        A 64-character lowercase hex fingerprint.
    """
    run = json.dumps(
        {k: str(run_identity[k]) for k in sorted(run_identity)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    regions = json.dumps(
        [dict(sorted(r.items())) for r in region_serving_states],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_hex(store_protocol_mode, run, store_schema_fingerprint, regions)
