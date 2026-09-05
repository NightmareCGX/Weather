"""Reusable S3FileSystem and client provider for the ingestion service.

Maintains explicit separation between:
1. DATA PLANE: Thread-local persistent :class:`IngestionS3FileSystem` instances
   tuned for high concurrent chunk PUT throughput (:attr:`IngestionSettings.S3_MAX_POOL_CONNECTIONS`).
   Held in a strong process registry during active waves for deterministic per-wave teardown.
2. CONTROL PLANE: Process-shared reusable :class:`IngestionS3FileSystem` instance
   tuned for markers, inventory, and coordination (:attr:`IngestionSettings.S3_CONTROL_MAX_POOL_CONNECTIONS`).
   Eliminates per-thread client and background event-loop creation churn across
   temporary thread pools (e.g. pre-update marker PUTs and bounded finalization marker GETs).

Lifecycle & Event Loop Architecture:
Synchronous s3fs.S3FileSystem instances dispatch asynchronous S3 I/O to a dedicated
daemon background thread ('fsspecIO') running an independent asyncio event loop.
To prevent cross-loop cleanup failures (RuntimeError: Future attached to a different loop
and Task exception was never retrieved) when filesystems are closed while an external
asyncio event loop is active:
1. :class:`IngestionS3FileSystem` overrides :meth:`close_session` to execute teardown
   synchronously ON THE OWNING LOOP (``fs.loop``) via :func:`fsspec.asyn.sync`.
2. Explicit lifecycle state (``OPEN -> CLOSING -> CLOSED / CLOSE_FAILED``) is tracked per
   `aiobotocore.session.ClientCreatorContext` using an object-identity `WeakKeyDictionary`.
   Explicit close and deferred weakref finalization share this state, guaranteeing idempotent,
   at-most-once teardown.
3. Worker data-plane filesystems are held in strong registry :data:`_active_data_filesystems`
   until explicit wave teardown (:func:`close_wave_data_s3_fs`), preventing premature worker GC.
4. Process-wide control-plane filesystem is cleanly closed at command termination (:func:`shutdown_s3_fs`).
"""

from __future__ import annotations

import asyncio
import atexit
import enum
import logging
import threading
import weakref
from collections.abc import MutableMapping
from typing import Any

import fsspec.asyn  # type: ignore[import-untyped]
import s3fs  # type: ignore[import-untyped]

from ingestion.core.config import IngestionSettings, settings

logger = logging.getLogger(__name__)

_thread_local = threading.local()
_control_lock = threading.Lock()
_control_fs: IngestionS3FileSystem | None = None
_control_fs_key: tuple[str, str, str, int] | None = None

# Strong registry for active data-plane filesystems created during a wave.
# Keyed by instance id to prevent set-deduplication of distinct S3FileSystem instances with matching config.
# Guarantees that worker-thread filesystems are not prematurely collected before explicit wave close.
_active_data_filesystems: dict[int, IngestionS3FileSystem] = {}
_data_registry_lock = threading.Lock()


class LifecycleState(enum.Enum):
    """Lifecycle state of an aiobotocore S3 client creator context."""

    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CLOSE_FAILED = "CLOSE_FAILED"


class CloseState:
    """Thread-safe lifecycle token for an underlying S3 client creator context."""

    def __init__(self) -> None:
        self.state: LifecycleState = LifecycleState.OPEN
        self.lock: threading.Lock = threading.Lock()


# Maps ClientCreatorContext instance -> CloseState token using true object identity.
# Bounded and automatically garbage-collected by Python when the creator is deallocated.
_creator_states: weakref.WeakKeyDictionary[Any, CloseState] = weakref.WeakKeyDictionary()
_states_lock = threading.Lock()


def _get_or_create_close_state(s3creator: Any) -> CloseState:
    """Retrieve or allocate the shared CloseState token for an S3 client creator context."""
    with _states_lock:
        state = _creator_states.get(s3creator)
        if state is None:
            state = CloseState()
            _creator_states[s3creator] = state
        return state


def _perform_safe_creator_close(loop: Any, s3creator: Any, timeout: float = 5.0) -> bool:
    """Safely and synchronously close an aiobotocore ClientCreatorContext on its owning loop.

    Guarantees:
    - Atomically transitions OPEN/CLOSE_FAILED -> CLOSING -> CLOSED (or CLOSE_FAILED on error).
    - Teardown executes strictly on ``loop`` (fsspecIO or zarr_io loop) via coroutine execution.
    - If already in CLOSED or CLOSING, returns immediately without re-entering teardown.
    - If teardown raises or times out, state transitions to CLOSE_FAILED so it remains retryable.

    Returns:
        True if closed successfully (or was already CLOSED), False if close failed/timed out.
    """
    if s3creator is None:
        return True

    close_state = _get_or_create_close_state(s3creator)
    with close_state.lock:
        if close_state.state == LifecycleState.CLOSED:
            return True
        if close_state.state == LifecycleState.CLOSING:
            return True
        close_state.state = LifecycleState.CLOSING

    # Execute standard async context manager exit on the owning loop
    if loop is not None and not loop.is_closed():
        try:
            fsspec.asyn.sync(loop, s3creator.__aexit__, None, None, None, timeout=timeout)
            with close_state.lock:
                close_state.state = LifecycleState.CLOSED
            return True
        except Exception as exc:
            with close_state.lock:
                close_state.state = LifecycleState.CLOSE_FAILED
            logger.warning(
                "Error closing S3 client creator session on owning loop: %s; "
                "committed forecast data is safe.",
                exc,
            )
            return False

    # Loop is already closed or unavailable
    with close_state.lock:
        close_state.state = LifecycleState.CLOSE_FAILED
    return False


class IngestionS3FileSystem(s3fs.S3FileSystem):
    """Loop-safe S3FileSystem with deterministic session lifecycle and state tracking.

    Eliminates upstream s3fs cross-event-loop cleanup defects by:
    1. Overriding :meth:`close_session` to perform teardown on ``self.loop`` (fsspecIO)
       or ``self._session_loop`` (zarr_io) instead of scheduling un-awaited tasks.
    2. Tracking wave data-plane instances and anonymous Zarr clones in :data:`_active_data_filesystems`.
    3. Providing an explicit, idempotent :meth:`close` method for deterministic shutdown.
    4. Sharing :class:`CloseState` tokens between explicit close and deferred weakref finalizers.
    """

    # Ingestion manages its own caching (control-plane singleton and data-plane strong registry).
    # Disable fsspec global instance cache to prevent closed instances from being revived.
    cachable = False

    def __init__(self, *args: Any, is_data_plane: bool = True, **kwargs: Any) -> None:
        self.is_data_plane = is_data_plane
        super().__init__(*args, **kwargs)
        self._is_closed: bool = False
        self._session_loop: asyncio.AbstractEventLoop | None = None
        if self.is_data_plane:
            with _data_registry_lock:
                _active_data_filesystems[id(self)] = self

    async def set_session(self, refresh: bool = False, kwargs: dict[str, Any] = {}) -> Any:
        res = await super().set_session(refresh=refresh, kwargs=kwargs)
        try:
            self._session_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        return res

    @staticmethod
    def close_session(loop: Any, s3: Any) -> None:
        """Weakref finalizer callback invoked upon garbage collection.

        Consults the shared :class:`CloseState` token:
        - If already CLOSED via explicit :meth:`close`, exits immediately as a no-op.
        - If OPEN or CLOSE_FAILED, executes safe synchronous teardown on ``loop``.
        """
        if s3 is None or loop is None or loop.is_closed():
            return
        _perform_safe_creator_close(loop, s3, timeout=5.0)

    def close(self, timeout: float = 5.0) -> bool:
        """Explicitly and synchronously close the S3 client session on its owning event loop.

        Returns:
            True if session closed successfully (or was already closed), False on failure/timeout.
        """
        if self._is_closed:
            return True

        # ``_s3creator`` holds the aiobotocore.session.ClientCreatorContext created during set_session()
        s3creator = getattr(self, "_s3creator", None)
        loop = getattr(self, "loop", None)
        if loop is None:
            loop = getattr(self, "_session_loop", None)

        if s3creator is not None and loop is not None and not loop.is_closed():
            success = _perform_safe_creator_close(loop, s3creator, timeout=timeout)
            if not success:
                # Do NOT clear references on failure; keep creator intact for retry
                return False

        self._s3 = None
        self._s3creator = None
        self._is_closed = True
        return True


def _endpoint_url(conn_settings: IngestionSettings) -> str:
    """Build the S3 endpoint URL from MinIO settings."""
    scheme = "https" if conn_settings.MINIO_SECURE else "http"
    return f"{scheme}://{conn_settings.MINIO_ENDPOINT}"


def get_s3_fs(
    conn_settings: IngestionSettings | None = None,
    *,
    force_new: bool = False,
) -> IngestionS3FileSystem:
    """Return a thread-local persistent IngestionS3FileSystem for data-plane chunk writes.

    Reuses the thread-local instance when credentials, endpoint, and pool size match.
    Registers a strong reference in :data:`_active_data_filesystems` so the instance
    cannot be prematurely collected before explicit wave completion.

    Args:
        conn_settings: Ingestion settings providing MinIO credentials and pool
            size. Defaults to the global :data:`settings`.
        force_new: If True, creates a fresh instance without updating the cache.

    Returns:
        A configured :class:`IngestionS3FileSystem` instance.
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

    cached_fs: IngestionS3FileSystem | None = getattr(_thread_local, "s3_fs", None)
    cached_key = getattr(_thread_local, "s3_fs_key", None)

    if cached_fs is not None and cached_key == cache_key and not cached_fs._is_closed:
        return cached_fs

    fs = _build_data_s3_fs(cfg)
    _thread_local.s3_fs = fs
    _thread_local.s3_fs_key = cache_key
    return fs


def get_control_s3_fs(
    conn_settings: IngestionSettings | None = None,
    *,
    force_new: bool = False,
) -> IngestionS3FileSystem:
    """Return a process-shared reusable IngestionS3FileSystem for control-plane operations.

    Used by markers, inventory, protocol version, and manifest helpers. Reuses a
    single shared instance across temporary thread pools (e.g. 32-thread finalization
    marker GETs and 8-thread pre-update marker PUTs), preventing client construction churn.

    Args:
        conn_settings: Ingestion settings providing MinIO credentials and control
            pool size. Defaults to the global :data:`settings`.
        force_new: If True, creates a fresh instance without updating the cache.

    Returns:
        A configured :class:`IngestionS3FileSystem` instance.
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
        if _control_fs is not None and _control_fs_key == cache_key and not _control_fs._is_closed:
            return _control_fs

        fs = _build_control_s3_fs(cfg)
        _control_fs = fs
        _control_fs_key = cache_key
        return fs


def _build_data_s3_fs(cfg: IngestionSettings) -> IngestionS3FileSystem:
    """Construct a new data-plane IngestionS3FileSystem instance and register strong ownership."""
    max_pool = int(cfg.S3_MAX_POOL_CONNECTIONS)
    return IngestionS3FileSystem(
        key=cfg.MINIO_ACCESS_KEY,
        secret=cfg.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": _endpoint_url(cfg)},
        config_kwargs={"max_pool_connections": max_pool},
        use_listings_cache=False,
        skip_instance_cache=True,
        is_data_plane=True,
    )


def _build_control_s3_fs(cfg: IngestionSettings) -> IngestionS3FileSystem:
    """Construct a new control-plane IngestionS3FileSystem instance."""
    max_pool = int(cfg.S3_CONTROL_MAX_POOL_CONNECTIONS)
    return IngestionS3FileSystem(
        key=cfg.MINIO_ACCESS_KEY,
        secret=cfg.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": _endpoint_url(cfg)},
        config_kwargs={"max_pool_connections": max_pool},
        use_listings_cache=False,
        skip_instance_cache=True,
        is_data_plane=False,
    )


def close_wave_data_s3_fs() -> bool:
    """Explicitly close all worker data-plane S3 filesystems and Zarr clones at wave completion.

    Drains :data:`_active_data_filesystems` and closes each instance on its owning
    ``fsspecIO`` or ``zarr_io`` loop before the wave terminates.

    Returns:
        True if all filesystems closed successfully, False if any close failed.
    """
    with _data_registry_lock:
        filesystems = list(_active_data_filesystems.values())
        _active_data_filesystems.clear()

    all_success = True
    for fs in filesystems:
        if not fs.close():
            all_success = False

    # Clear current thread's thread-local reference if present
    if hasattr(_thread_local, "s3_fs"):
        del _thread_local.s3_fs
    if hasattr(_thread_local, "s3_fs_key"):
        del _thread_local.s3_fs_key

    return all_success


def shutdown_s3_fs() -> bool:
    """Explicit top-level command shutdown for all ingestion S3 filesystem resources.

    Closes the process-wide control-plane filesystem and any remaining data-plane
    filesystems deterministically on their owning ``fsspecIO`` loops.

    Returns:
        True if all resources closed successfully, False if any close failed.
    """
    global _control_fs, _control_fs_key
    all_success = True

    # 1. Close control-plane filesystem
    with _control_lock:
        if _control_fs is not None:
            if not _control_fs.close():
                all_success = False
            else:
                _control_fs = None
                _control_fs_key = None

    # 2. Close any remaining data-plane filesystems
    if not close_wave_data_s3_fs():
        all_success = False

    return all_success


class MissingBucketError(RuntimeError):
    """Raised when the configured object-store bucket does not exist."""


def verify_object_store_preflight(
    conn_settings: IngestionSettings | None = None,
) -> None:
    """Validate S3 endpoint reachability, credentials, and configured bucket existence.

    Preflight validation executed once at service startup before any wave dispatch.
    Fails fast with actionable diagnosis if the bucket is unprovisioned. Does NOT
    auto-create buckets.

    Raises:
        MissingBucketError: When the configured bucket does not exist.
        RuntimeError: When the S3 endpoint is unreachable or credentials are rejected.
    """
    cfg = conn_settings if conn_settings is not None else settings
    bucket = str(cfg.MINIO_BUCKET_NAME)
    fs = get_control_s3_fs(cfg)
    try:
        exists = fs.exists(bucket)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to S3 endpoint at {_endpoint_url(cfg)}: {exc}"
        ) from exc

    if not exists:
        raise MissingBucketError(
            f"Configured object-store bucket '{bucket}' does not exist on "
            f"{_endpoint_url(cfg)}. Provision the bucket before starting ingestion."
        )


def reset_s3_fs() -> None:
    """Clear both data-plane and control-plane S3FileSystem caches (testing/teardown)."""
    shutdown_s3_fs()


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


# Defensive atexit hook as a last-resort fallback for abrupt script termination
atexit.register(shutdown_s3_fs)
