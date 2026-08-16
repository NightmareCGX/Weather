"""API-side committed-manifest reader for generation-aware cache identity.

The API reads the committed manifest (written by the ingestion EXCLUSIVE
finalizer) to obtain the serving generation used in cache keys. A generation
change (e.g. a same-set same-cycle data replacement) makes old cache entries
unreachable without cross-process LRU invalidation.
"""

from __future__ import annotations

import json
import os
from typing import Any

import s3fs  # type: ignore[import-untyped]

from domain.locks import canonical_storage_identity, sha256_hex

from api.core.config import settings

#: Committed manifest path under a store's commit namespace.
_MANIFEST_PATH = "__commit__/v1/manifest.json"


class ManifestReadError(RuntimeError):
    """Raised when the committed manifest is missing/malformed for a marker_v1 store."""


def _resolve_store_root(store_path: str) -> str:
    if store_path.startswith("s3://"):
        return store_path[len("s3://") :].strip("/")
    path = store_path
    if path.startswith("file://"):
        path = path[len("file://") :]
    return os.path.abspath(os.path.normpath(path))


def _read_manifest(store_path: str) -> dict[str, Any] | None:
    root = _resolve_store_root(store_path)
    if store_path.startswith("s3://"):
        scheme = "https" if settings.MINIO_SECURE else "http"
        fs = s3fs.S3FileSystem(
            key=settings.MINIO_ACCESS_KEY,
            secret=settings.MINIO_SECRET_KEY,
            client_kwargs={"endpoint_url": f"{scheme}://{settings.MINIO_ENDPOINT}"},
        )
        try:
            raw = fs.cat_file(f"{root}/{_MANIFEST_PATH}")
        except FileNotFoundError:
            return None
    else:
        full = os.path.join(root, *_MANIFEST_PATH.split("/"))
        try:
            with open(full, "rb") as fh:
                raw = fh.read()
        except FileNotFoundError:
            return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestReadError(f"committed manifest is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestReadError("committed manifest is not a JSON object")
    return payload


def manifest_generation(store_path: str) -> str:
    """Return the committed-manifest serving generation for cache keys.

    When no manifest exists (a legacy/hybrid store not yet finalized), derive a
    deterministic legacy token from the canonical store identity so the cache
    key is stable until the first marker-aware finalization writes a real
    generation.

    Raises:
        ManifestReadError: If a manifest exists but is malformed (fail closed).
    """
    payload = _read_manifest(store_path)
    if payload is None:
        # Legacy compatibility token: deterministic, stable until the first
        # marker-aware finalization writes a real manifest generation.
        return "legacy-" + sha256_hex(canonical_storage_identity(store_path))
    generation = payload.get("generation")
    if not isinstance(generation, str) or not generation:
        raise ManifestReadError("committed manifest has no valid generation")
    return generation
