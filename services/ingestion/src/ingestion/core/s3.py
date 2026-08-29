"""Reusable S3FileSystem and client provider for the ingestion service.

Maintains explicit separation between:
1. DATA PLANE: Thread-local persistent :class:`s3fs.S3FileSystem` instances
   tuned for high concurrent chunk PUT throughput (:attr:`IngestionSettings.S3_MAX_POOL_CONNECTIONS`).
   Each long-lived write worker thread owns an independent instance and connection pool.
2. CONTROL PLANE: Process-shared reusable :class:`s3fs.S3FileSystem` instance
   tuned for markers, inventory, and coordination (:attr:`IngestionSettings.S3_CONTROL_MAX_POOL_CONNECTIONS`).
   Eliminates per-thread client and background event-loop creation churn across
   temporary thread pools (e.g. pre-update marker PUTs and bounded finalization marker GETs).
"""

from __future__ import annotations

import threading
from collections.abc import MutableMapping

import s3fs  # type: ignore[import-untyped]

from ingestion.core.config import IngestionSettings, settings

_thread_local = threading.local()
_control_lock = threading.Lock()
_control_fs: s3fs.S3FileSystem | None = None
_control_fs_key: tuple[str, str, str, int] | None = None


def _endpoint_url(conn_settings: IngestionSettings) -> str:
    """Build the S3 endpoint URL from MinIO settings."""
    scheme = "https" if conn_settings.MINIO_SECURE else "http"
    return f"{scheme}://{conn_settings.MINIO_ENDPOINT}"


def get_s3_fs(
    conn_settings: IngestionSettings | None = None,
    *,
    force_new: bool = False,
) -> s3fs.S3FileSystem:
    """Return a thread-local persistent S3FileSystem for data-plane chunk writes.

    Reuses the thread-local instance when credentials, endpoint, and pool size match.
    Pass ``force_new=True`` to bypass caching and obtain a fresh instance.

    Args:
        conn_settings: Ingestion settings providing MinIO credentials and pool
            size. Defaults to the global :data:`settings`.
        force_new: If True, creates a fresh instance without updating the cache.

    Returns:
        A configured :class:`s3fs.S3FileSystem` instance.
    """
    cfg = conn_settings if conn_settings is not None else settings
    if force_new:
        return _build_data_s3_fs(cfg)

    cache_key = (
        cfg.MINIO_ACCESS_KEY,
        cfg.MINIO_SECRET_KEY,
        _endpoint_url(cfg),
        int(cfg.S3_MAX_POOL_CONNECTIONS),
    )

    cached_fs = getattr(_thread_local, "s3_fs", None)
    cached_key = getattr(_thread_local, "s3_fs_key", None)

    if cached_fs is not None and cached_key == cache_key:
        return cached_fs

    fs = _build_data_s3_fs(cfg)
    _thread_local.s3_fs = fs
    _thread_local.s3_fs_key = cache_key
    return fs


def get_control_s3_fs(
    conn_settings: IngestionSettings | None = None,
    *,
    force_new: bool = False,
) -> s3fs.S3FileSystem:
    """Return a process-shared reusable S3FileSystem for control-plane operations.

    Used by markers, inventory, protocol version, and manifest helpers. Reuses a
    single shared instance across temporary thread pools (e.g. 32-thread finalization
    marker GETs and 8-thread pre-update marker PUTs), preventing client construction churn.

    Args:
        conn_settings: Ingestion settings providing MinIO credentials and control
            pool size. Defaults to the global :data:`settings`.
        force_new: If True, creates a fresh instance without updating the cache.

    Returns:
        A configured :class:`s3fs.S3FileSystem` instance.
    """
    global _control_fs, _control_fs_key
    cfg = conn_settings if conn_settings is not None else settings
    if force_new:
        return _build_control_s3_fs(cfg)

    cache_key = (
        cfg.MINIO_ACCESS_KEY,
        cfg.MINIO_SECRET_KEY,
        _endpoint_url(cfg),
        int(cfg.S3_CONTROL_MAX_POOL_CONNECTIONS),
    )

    with _control_lock:
        if _control_fs is not None and _control_fs_key == cache_key:
            return _control_fs

        fs = _build_control_s3_fs(cfg)
        _control_fs = fs
        _control_fs_key = cache_key
        return fs


def _build_data_s3_fs(cfg: IngestionSettings) -> s3fs.S3FileSystem:
    """Construct a new data-plane S3FileSystem instance."""
    max_pool = int(cfg.S3_MAX_POOL_CONNECTIONS)
    return s3fs.S3FileSystem(
        key=cfg.MINIO_ACCESS_KEY,
        secret=cfg.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": _endpoint_url(cfg)},
        config_kwargs={"max_pool_connections": max_pool},
        use_listings_cache=False,
        skip_instance_cache=True,
    )


def _build_control_s3_fs(cfg: IngestionSettings) -> s3fs.S3FileSystem:
    """Construct a new control-plane S3FileSystem instance."""
    max_pool = int(cfg.S3_CONTROL_MAX_POOL_CONNECTIONS)
    return s3fs.S3FileSystem(
        key=cfg.MINIO_ACCESS_KEY,
        secret=cfg.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": _endpoint_url(cfg)},
        config_kwargs={"max_pool_connections": max_pool},
        use_listings_cache=False,
        skip_instance_cache=True,
    )


def reset_s3_fs() -> None:
    """Clear both data-plane and control-plane S3FileSystem caches (testing/teardown)."""
    global _control_fs, _control_fs_key
    if hasattr(_thread_local, "s3_fs"):
        del _thread_local.s3_fs
    if hasattr(_thread_local, "s3_fs_key"):
        del _thread_local.s3_fs_key
    with _control_lock:
        _control_fs = None
        _control_fs_key = None


def resolve_s3_mapper(
    path: str, conn_settings: IngestionSettings | None = None
) -> MutableMapping[str, bytes]:
    """Build an ``FSMap`` over an ``s3://`` URL using the thread-local data-plane S3FileSystem.

    Args:
        path: An ``s3://bucket/prefix`` URL.
        conn_settings: Optional IngestionSettings.

    Returns:
        An ``s3fs`` mapping object accepted by xarray.

    Raises:
        ValueError: If the bucket/prefix cannot be derived from the URL.
    """
    rest = path[len("s3://") :].strip("/")
    if not rest:
        raise ValueError(f"Invalid S3 store URL: {path!r}")

    fs = get_s3_fs(conn_settings)
    mapper: MutableMapping[str, bytes] = fs.get_mapper(rest)
    return mapper
