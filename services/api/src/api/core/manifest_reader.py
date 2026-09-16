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
import time
from typing import Any

import s3fs  # type: ignore[import-untyped]

from api.core.config import settings

#: Committed manifest path under a store's commit namespace.
_MANIFEST_PATH = "__commit__/v1/manifest.json"

_s3_fs_instance: s3fs.S3FileSystem | None = None
_s3_fs_lock = threading.Lock()

#: In-memory TTL micro-cache for committed manifests to collapse concurrent
#: cold tile reads into a single MinIO/disk access.
#:
#: Only *positive* payloads are cached. An absent manifest is deliberately
#: re-probed on every call: the ingestion EXCLUSIVE finalizer can commit a
#: manifest at any moment, and the transition from legacy (uncached, opened
#: fresh per request) to generation-aware handle caching must be visible on the
#: very next lookup. Negative caching would introduce a staleness window on
#: that transition, so it is intentionally not done here even though a legacy
#: store therefore pays a storage round trip per lookup (see
#: tests/test_store_cache.py::test_manifest_created_after_initial_no_manifest_serving
#: and ::test_uncached_dataset_closed_after_selection).
_MANIFEST_CACHE_TTL: float = 30.0
_MANIFEST_CACHE_MAX_SIZE: int = 1024
_manifest_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_manifest_cache_lock = threading.Lock()
_manifest_flight_locks: dict[str, threading.Lock] = {}


def _clear_manifest_cache() -> None:
    """Clear the in-memory manifest TTL cache and inflight locks."""
    with _manifest_cache_lock:
        _manifest_cache.clear()
        _manifest_flight_locks.clear()


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


def _read_manifest_uncached(store_path: str) -> dict[str, Any] | None:
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


def _read_manifest(store_path: str) -> dict[str, Any] | None:
    now = time.monotonic()
    with _manifest_cache_lock:
        entry = _manifest_cache.get(store_path)
        if entry is not None and entry[0] > now:
            return entry[1]
        flight_lock = _manifest_flight_locks.setdefault(store_path, threading.Lock())

    with flight_lock:
        now = time.monotonic()
        with _manifest_cache_lock:
            entry = _manifest_cache.get(store_path)
            if entry is not None and entry[0] > now:
                return entry[1]

        try:
            payload = _read_manifest_uncached(store_path)

            now = time.monotonic()
            with _manifest_cache_lock:
                # Only positive payloads are cached: caching an absent manifest
                # would delay the transition to generation-aware caching after
                # the finalizer commits one (see the note on _MANIFEST_CACHE_TTL).
                if payload is not None:
                    if len(_manifest_cache) >= _MANIFEST_CACHE_MAX_SIZE:
                        expired_keys = [
                            k for k, (exp, _) in _manifest_cache.items() if exp <= now
                        ]
                        for k in expired_keys:
                            _manifest_cache.pop(k, None)
                        if len(_manifest_cache) >= _MANIFEST_CACHE_MAX_SIZE:
                            oldest_key = next(iter(_manifest_cache))
                            _manifest_cache.pop(oldest_key, None)

                    _manifest_cache[store_path] = (now + _MANIFEST_CACHE_TTL, payload)
            return payload
        finally:
            # Release the single-flight slot on every path, including when
            # ``_read_manifest_uncached`` raises (malformed manifest, transient
            # storage error). Leaving the entry behind leaked one Lock per
            # distinct failing store, with no bound or eviction.
            with _manifest_cache_lock:
                _manifest_flight_locks.pop(store_path, None)


def manifest_generation(store_path: str) -> str | None:
    """Return the committed-manifest serving generation for cache keys.

    Returns:
        The trusted generation string if a valid committed manifest exists,
        or ``None`` when the manifest is confirmed absent (legacy or
        unfinalized store). Absence is never cached, so a manifest committed
        later is observed by the very next lookup.

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
