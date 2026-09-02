"""API-side committed-manifest reader for generation-aware cache identity.

The API reads the committed manifest (written by the ingestion EXCLUSIVE
finalizer) to obtain the serving generation used in cache keys. A generation
change (e.g. a same-set same-cycle data replacement) makes old cache entries
unreachable without cross-process LRU invalidation.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import s3fs  # type: ignore[import-untyped]

from api.core.config import settings

#: Committed manifest path under a store's commit namespace.
_MANIFEST_PATH = "__commit__/v1/manifest.json"

_s3_fs_instance: s3fs.S3FileSystem | None = None
_s3_fs_lock = threading.Lock()


def _get_s3_fs() -> s3fs.S3FileSystem:
    global _s3_fs_instance
    with _s3_fs_lock:
        if _s3_fs_instance is None:
            scheme = "https" if settings.MINIO_SECURE else "http"
            _s3_fs_instance = s3fs.S3FileSystem(
                key=settings.MINIO_ACCESS_KEY,
                secret=settings.MINIO_SECRET_KEY,
                client_kwargs={"endpoint_url": f"{scheme}://{settings.MINIO_ENDPOINT}"},
                use_listings_cache=False,
            )
        return _s3_fs_instance


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
        fs = _get_s3_fs()
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


def manifest_generation(store_path: str) -> str | None:
    """Return the committed-manifest serving generation for cache keys.

    Returns:
        The trusted generation string if a valid committed manifest exists,
        or ``None`` when the manifest is confirmed absent (legacy or
        unfinalized store).

    Raises:
        ManifestReadError: If a manifest exists but is malformed/invalid (fail closed).
    """
    payload = _read_manifest(store_path)
    if payload is None:
        # Confirmed absent -> no trusted generation (caller must bypass handle cache).
        return None
    generation = payload.get("generation")
    if not isinstance(generation, str) or not generation:
        raise ManifestReadError("committed manifest has no valid generation")
    return generation


def manifest_storage_format(store_path: str) -> str:
    """Return the storage_format_version declared in the committed manifest.

    Returns:
        The format version string (e.g. 'sharded_v1'), or 'v2_unsharded' when
        the manifest is missing or does not declare a format version.
    """
    try:
        payload = _read_manifest(store_path)
        if payload is None:
            return "v2_unsharded"
        return str(payload.get("storage_format_version") or "v2_unsharded")
    except Exception:  # noqa: BLE001 - unreadable manifest -> default legacy
        return "v2_unsharded"
