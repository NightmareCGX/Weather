"""Chunked, compressed Zarr storage for normalized forecast datasets.

A single ``xarray.Dataset`` is written to a Zarr store (local directory or
an ``s3://`` bucket) with per-dimension chunking and ``Zstd`` compression,
then read back for round-trip verification.

The ``s3://`` scheme is resolved to a MinIO/S3-compatible store using
:class:`s3fs.S3FileSystem` configured from :class:`IngestionSettings`.
This is intentionally narrow: no bucket lifecycle or multi-file
management lives here.
"""

from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike
from typing import Any, Hashable, Mapping, MutableMapping

import s3fs  # type: ignore[import-untyped]
import xarray as xr
from numcodecs import Zstd  # type: ignore[import-untyped]

from ingestion.core.base import IngestionError
from ingestion.core.config import IngestionSettings, settings

#: Default chunks applied per dimension when none are provided.
DEFAULT_CHUNKS: Mapping[str, int] = {
    "time": 1,
    "lead_time_hours": 1,
    "isobaricInhPa": 1,
    "latitude": 100,
    "longitude": 100,
}


Store = str | PathLike[str] | Mapping[str, bytes]


def _resolve_store(
    store: str | PathLike[str] | Mapping[str, bytes],
) -> str | PathLike[str] | MutableMapping[str, bytes]:
    """Resolve a Zarr store target to an xarray-compatible mapping.

    Args:
        store: A local directory path, an ``s3://`` URL, or an existing
            mutable mapping (e.g. ``fsspec`` ``FSMap``).

    Returns:
        An object accepted by :func:`xarray.Dataset.to_zarr` /
        :func:`xarray.open_zarr`.

    Raises:
        ValueError: If the ``s3://`` URL cannot be parsed or the scheme is
            unsupported.
    """
    if isinstance(store, MutableMapping):
        return store
    if isinstance(store, Mapping):
        return dict(store)
    path = os.fspath(store)
    if path.startswith("s3://"):
        return _resolve_s3_store(path, settings)
    if path.startswith("file://"):
        return path[len("file://") :]
    return path


def _resolve_s3_store(
    path: str, conn_settings: IngestionSettings
) -> MutableMapping[str, bytes]:
    """Build an ``FSMap`` over an ``s3://`` URL using S3 settings.

    Args:
        path: An ``s3://bucket/prefix`` URL.
        conn_settings: Ingestion settings providing MinIO credentials.

    Returns:
        An ``s3fs`` mapping object accepted by xarray.

    Raises:
        ValueError: If the bucket/prefix cannot be derived from the URL.
    """
    rest = path[len("s3://") :].strip("/")
    if not rest:
        raise ValueError(f"Invalid S3 store URL: {path!r}")

    fs = s3fs.S3FileSystem(
        key=conn_settings.MINIO_ACCESS_KEY,
        secret=conn_settings.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": _endpoint_url(conn_settings)},
    )
    # ``get_mapper`` is untyped in the s3fs stub (import-untyped), so mypy
    # sees ``Any``. Narrowing through a ``MutableMapping[str, bytes]``-typed
    # intermediate enforces the declared return type instead of suppressing it.
    mapper: MutableMapping[str, bytes] = fs.get_mapper(rest)
    return mapper


def _endpoint_url(conn_settings: IngestionSettings) -> str:
    """Build the S3 endpoint URL from MinIO settings."""
    scheme = "https" if conn_settings.MINIO_SECURE else "http"
    return f"{scheme}://{conn_settings.MINIO_ENDPOINT}"


class ZarrWriteError(IngestionError):
    """Raised when a Zarr store cannot be written, validated, or committed.

    The atomic write path raises this when a staged store fails to validate or
    cannot be swapped into place, so a half-written store is never exposed at
    the final path.
    """


class CorruptStoreError(IngestionError):
    """Raised when an existing Zarr store cannot be opened.

    A store that physically exists but fails to open (a half-written or
    corrupt store) must fail ingestion loudly instead of being silently
    rebuilt from the current single lead, which would drop every previously
    ingested lead while the catalog still advertises them as ready (review
    finding MAJOR-2).
    """


#: Sibling-path suffix for the in-progress staging store of an atomic write.
_STAGING_SUFFIX = ".staging"
#: Sibling-path suffix a superseded store is moved to during an atomic swap.
_OLD_SUFFIX = ".old"


def _staging_path(path: str) -> str:
    """Return the sibling staging path used for an atomic write.

    The staging target (``{path}.staging``) holds the dataset being written so
    a partially-written or corrupt store is never exposed at the final
    ``path`` before it is known-good and atomically swapped in.

    Args:
        path: The final local path or ``s3://`` URL.

    Returns:
        The sibling staging path.
    """
    return f"{path}{_STAGING_SUFFIX}"


def _old_path(path: str) -> str:
    """Return the sibling ``.old`` path a superseded store is moved to.

    Args:
        path: The final local path or ``s3://`` URL.

    Returns:
        The sibling superseded path.
    """
    return f"{path}{_OLD_SUFFIX}"


def _remove_path_if_exists(path: str) -> None:
    """Remove a staging/old directory, ignoring a missing target.

    Args:
        path: The directory to remove.
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def _write_zarr(
    dataset: xr.Dataset,
    resolved: str | PathLike[str] | MutableMapping[str, bytes],
    chunks: Mapping[str, int] | None,
) -> None:
    """Write ``dataset`` to ``resolved`` with platform chunking and Zstd.

    Chunk sizes default to :data:`DEFAULT_CHUNKS` (falling back to the full
    extent of any dimension not covered) and every data variable is stored
    with ``Zstd`` compression. No dask backend is required.

    Args:
        dataset: The normalized dataset to persist.
        resolved: An xarray-compatible store target.
        chunks: Optional per-dimension chunk sizes.
    """
    defaults = chunks or DEFAULT_CHUNKS

    def _chunk_sizes(name: Hashable) -> tuple[int, ...]:
        def _size(dim: Hashable) -> int:
            size = dataset.sizes[dim]
            assert size is not None
            return size

        def _default(dim: Hashable) -> int:
            return defaults.get(str(dim), _size(dim))

        return tuple(min(_default(dim), _size(dim)) for dim in dataset[name].dims)

    encoding = {
        name: {"chunks": _chunk_sizes(name), "compressor": Zstd(level=5)}
        for name in dataset.data_vars
    }
    dataset.to_zarr(resolved, mode="w", encoding=encoding)


def _require_readable(
    resolved: str | PathLike[str] | MutableMapping[str, bytes],
) -> None:
    """Open and close a Zarr store, failing loudly if it cannot be read.

    Args:
        resolved: The store target to validate.

    Raises:
        ZarrWriteError: If the store cannot be opened.
    """
    try:
        dataset = xr.open_zarr(resolved)
    except Exception as exc:
        raise ZarrWriteError(
            f"Just-written Zarr store {resolved!r} cannot be opened: {exc}"
        ) from exc
    dataset.close()


def _swap_local(path: str, staging: str) -> None:
    """Atomically replace ``path`` with the staged ``staging`` store.

    The previous store at ``path`` is first renamed to ``{path}.old``, then
    the staged store is renamed into ``path``, then ``.old`` is removed. Both
    renames are atomic on a single filesystem, so at every intermediate point
    either the old store or the new store is present at ``path`` — never a
    half-written directory. On failure the previous store is rolled back.

    Args:
        path: The final store directory.
        staging: The fully-written staging directory to promote.
    """
    old = _old_path(path)
    old_moved = False
    try:
        if os.path.exists(path):
            _remove_path_if_exists(old)
            os.rename(path, old)
            old_moved = True
        os.rename(staging, path)
    except BaseException:
        if old_moved and os.path.exists(old) and not os.path.exists(path):
            # Roll the previous store back into place and leave ``staging``
            # for best-effort cleanup by the caller; ``path`` now holds the
            # original store or nothing at all, never a partial write.
            try:
                os.rename(old, path)
            except Exception:
                pass
        raise
    # The swap succeeded; best-effort removal of the superseded ``.old``.
    if old_moved:
        try:
            _remove_path_if_exists(old)
        except Exception:
            pass


def _write_local_atomic(
    dataset: xr.Dataset,
    path: str,
    chunks: Mapping[str, int] | None,
) -> None:
    """Write ``dataset`` to ``path`` atomically via a sibling staging store.

    The dataset is written to ``{path}.staging``, verified readable, then
    atomically swapped into ``path`` by :func:`_swap_local`. A crash on the
    write or validation leaves only a discarded ``.staging`` (removed) and
    the previous ``path`` intact.

    Args:
        path: The final local directory path.
        chunks: Optional per-dimension chunk sizes.
    """
    staging = _staging_path(path)
    _remove_path_if_exists(staging)
    try:
        _write_zarr(dataset, staging, chunks)
        _require_readable(staging)
    except BaseException:
        _remove_path_if_exists(staging)
        raise
    _swap_local(path, staging)


def _make_s3_fs() -> Any:
    """Build the S3 filesystem configured from ingestion settings.

    Returns:
        An ``s3fs.S3FileSystem`` (typed ``Any`` because s3fs is imported
        untyped).
    """
    return s3fs.S3FileSystem(
        key=settings.MINIO_ACCESS_KEY,
        secret=settings.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": _endpoint_url(settings)},
    )


def _fs_key(url: str) -> str:
    """Return the ``bucket/prefix`` filesystem key for an ``s3://`` URL.

    Args:
        url: An ``s3://bucket/prefix`` URL.

    Returns:
        The ``bucket/prefix`` key used by s3fs.

    Raises:
        ValueError: If the URL has no bucket/prefix.
    """
    rest = url[len("s3://") :].strip("/")
    if not rest:
        raise ValueError(f"Invalid S3 store URL: {url!r}")
    return rest


def _s3_remove_if_exists(fs: Any, key: str) -> None:
    """Recursively remove an S3 prefix, ignoring a missing target.

    Args:
        fs: An ``s3fs.S3FileSystem`` instance.
        key: The ``bucket/prefix`` key to remove.
    """
    if fs.exists(key):
        fs.rm(key, recursive=True)


def _swap_s3(fs: Any, url: str, staging: str) -> None:
    """Promote a staged S3 prefix to the final URL as best-effort-atomic.

    s3fs ``mv`` is a copy-then-delete, so S3 has no true directory-level
    atomic rename. The previous prefix is first moved aside to ``.old`` and
    the staged prefix renamed into place; on failure the previous prefix is
    rolled back. This narrows the failure window to the ``mv`` operations
    themselves (see :func:`_write_s3_atomic`).

    Args:
        url: The final ``s3://bucket/prefix`` URL.
        staging: The fully-written staging URL to promote.
    """
    key = _fs_key(url)
    staging_key = _fs_key(staging)
    old_key = _fs_key(_old_path(url))
    old_moved = False
    try:
        if fs.exists(key):
            _s3_remove_if_exists(fs, old_key)
            fs.mv(key, old_key, recursive=True)
            old_moved = True
        fs.mv(staging_key, key, recursive=True)
    except BaseException:
        if old_moved and fs.exists(old_key) and not fs.exists(key):
            # Roll the previous store back into place.
            try:
                fs.mv(old_key, key, recursive=True)
            except Exception:
                pass
        raise
    if old_moved:
        try:
            _s3_remove_if_exists(fs, old_key)
        except Exception:
            pass


def _write_s3_atomic(
    dataset: xr.Dataset,
    path: str,
    chunks: Mapping[str, int] | None,
) -> None:
    """Write ``dataset`` to an ``s3://`` store as best-effort-atomic.

    S3 has no directory-level atomic rename, so the staging swap for remote
    stores cannot be as strong as the local two-``os.rename`` exchange:
    ``s3fs`` ``mv`` is a copy-then-delete, so there is a brief window during
    which the store prefix is absent. To avoid ever serving a corrupt store,
    the dataset is written to a temporary sibling prefix, verified readable,
    and only then swapped into place while the previous prefix is preserved.
    If any step fails the staging prefix is removed and the previously-served
    store is left intact — the writer prefers raising over overwriting a
    known-good store with an unverified one.

    Args:
        path: The final ``s3://bucket/prefix`` URL.
        chunks: Optional per-dimension chunk sizes.

    Raises:
        ZarrWriteError: If the staged store cannot be validated.
    """
    fs = _make_s3_fs()
    staging = _staging_path(path)
    # Clear any leftover staging prefix from a previous crashed attempt.
    _s3_remove_if_exists(fs, _fs_key(staging))
    try:
        _write_zarr(dataset, _resolve_s3_store(staging, settings), chunks)
        _require_readable(_resolve_s3_store(staging, settings))
    except BaseException:
        _s3_remove_if_exists(fs, _fs_key(staging))
        raise
    _swap_s3(fs, path, staging)


def write_dataset(
    dataset: xr.Dataset,
    store: str | PathLike[str] | Mapping[str, bytes],
    *,
    chunks: Mapping[str, int] | None = None,
) -> str:
    """Write a normalized dataset to a Zarr store, atomically.

    Chunk sizes default to :data:`DEFAULT_CHUNKS` (falling back to the full
    extent of any dimension not covered) and every data variable is stored
    with ``Zstd`` compression. The dataset is first written to a sibling
    staging target and verified readable, then atomically swapped into the
    final ``store`` so a crash mid-write never exposes a half-written store
    — the previous store (if any) is preserved until the new one is known-good.

    Args:
        dataset: Normalized dataset (e.g. from
            :func:`ingestion.providers.noaa.parser.parse_grib2`).
        store: Local path, ``s3://`` URL, or a mutable mapping (a mapping
            target has no directory semantics and is written in place).
        chunks: Optional per-dimension chunk sizes.

    Returns:
        The store target as a string (for reporting).

    Raises:
        ValueError: If an ``s3://`` URL is malformed.
        ZarrWriteError: If the staged store cannot be validated.
    """
    if isinstance(store, MutableMapping):
        # A mutable mapping is written in place: the caller's mapping receives
        # the store bytes. Copying it with dict(store) would silently write
        # to a throwaway copy and drop the data (review finding MAJOR-4).
        _write_zarr(dataset, store, chunks)
        return str(store)
    if isinstance(store, Mapping):
        raise TypeError(
            "read-only Mapping store targets cannot be written in place; "
            "pass a mutable mapping (e.g. a dict) or a path/URL"
        )
    path = os.fspath(store)
    if path.startswith("file://"):
        path = path[len("file://") :]
    if path.startswith("s3://"):
        _write_s3_atomic(dataset, path, chunks)
        return path
    _write_local_atomic(dataset, path, chunks)
    return path


def read_dataset(store: str | PathLike[str] | Mapping[str, bytes]) -> xr.Dataset:
    """Read a Zarr store back into a dataset.

    The dataset is returned numpy-backed (no dask required); the chunk
    grid and compressor persisted at write time remain available via
    ``encoding``.

    Args:
        store: The same local path, ``s3://`` URL, or mapping used to
            write.

    Returns:
        The dataset read back from the Zarr store.
    """
    resolved = _resolve_store(store)
    # ``xr.open_zarr`` is overloaded and infers ``Any`` for the ``Any``-typed
    # store; the value is always a concrete ``Dataset`` at runtime, so narrow
    # through a typed intermediate to satisfy the declared return type.
    dataset: xr.Dataset = xr.open_zarr(resolved)
    return dataset


def store_exists(store: str | PathLike[str] | Mapping[str, bytes]) -> bool:
    """Return whether a readable Zarr store already exists at ``store``.

    Used by the ingestion orchestration to decide whether a write is a fresh
    ingest (the target may be written in place) or a re-ingest (the target is
    already served by a ``ready`` catalog row and must not be truncated until
    the new store is known-good). A store that exists but fails to open is
    treated as absent so it is never written over destructively (L2-M3).

    Both local and remote targets are verified by actually attempting to open
    the Zarr store — checking only that a directory exists would treat a
    corrupt or half-written store as present and reuse it.

    Args:
        store: A local path, ``s3://`` URL, or mapping.

    Returns:
        True when a store can be opened at ``store``, False otherwise.
    """
    resolved = _resolve_store(store)
    if isinstance(resolved, str) and not os.path.exists(resolved):
        return False
    try:
        dataset = xr.open_zarr(resolved)
    except Exception:
        return False
    dataset.close()
    return True



def store_status(
    store: str | PathLike[str] | Mapping[str, bytes],
) -> str:
    """Classify an existing store target as "missing", "readable", or "corrupt".

    Unlike :func:`store_exists`, this distinguishes a never-written target
    (missing) from one that exists but cannot be opened (corrupt). The
    ingestion orchestration treats a corrupt store as a hard failure — a
    half-written store must never be silently rebuilt from the current single
    lead (that would drop every previously ingested lead while the catalog
    still advertises them as ready).

    Args:
        store: A local path, ``s3://`` URL, or mapping.

    Returns:
        One of "missing", "readable", "corrupt".
    """
    resolved = _resolve_store(store)
    if isinstance(resolved, str):
        if not os.path.exists(resolved):
            return "missing"
        try:
            dataset = xr.open_zarr(resolved)
        except Exception:
            return "corrupt"
        dataset.close()
        return "readable"
    # Mutable mapping / fsspec mapping (e.g. an S3 mapper): an empty mapping
    # means the target was never written; anything else that fails to open
    # is treated as corrupt.
    if isinstance(resolved, Mapping) and len(resolved) == 0:
        return "missing"
    try:
        dataset = xr.open_zarr(resolved)
    except Exception:
        return "corrupt"
    dataset.close()
    return "readable"


@contextmanager
def store_lock(store: str | PathLike[str]) -> Iterator[None]:
    """Serialize read-merge-write cycles on a local Zarr store path.

    Multiple workers ingesting the same cycle run concurrently (one worker
    per lead) share the store. Without mutual exclusion the shared staging
    directory races (one worker removes another's in-flight staging) and the
    read-merge-write cycle suffers lost updates. ``flock`` provides
    process-level mutual exclusion keyed by a sibling ``{path}.lock`` file.

    ``s3://`` targets have no filesystem lock primitive; the caller (the
    scheduler/CLI) must serialize writers for a shared S3 store.

    Args:
        store: The local store path (or ``s3://`` URL, which yields without
            locking).
    """
    path = os.fspath(store)
    if path.startswith("s3://"):
        yield
        return
    lock_fd = open(path + ".lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
