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

import os
from os import PathLike
from typing import Hashable, Mapping, MutableMapping

import s3fs  # type: ignore[import-untyped]
import xarray as xr
from numcodecs import Zstd  # type: ignore[import-untyped]

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
    return fs.get_mapper(rest)  # type: ignore[return-value]


def _endpoint_url(conn_settings: IngestionSettings) -> str:
    """Build the S3 endpoint URL from MinIO settings."""
    scheme = "https" if conn_settings.MINIO_SECURE else "http"
    return f"{scheme}://{conn_settings.MINIO_ENDPOINT}"


def write_dataset(
    dataset: xr.Dataset,
    store: str | PathLike[str] | Mapping[str, bytes],
    *,
    chunks: Mapping[str, int] | None = None,
) -> str:
    """Write a normalized dataset to a Zarr store.

    Chunk sizes default to :data:`DEFAULT_CHUNKS` (falling back to the
    full extent of any dimension not covered) and every data variable is
    stored with ``Zstd`` compression. No dask backend is required; the
    chunk grid and compressor are recorded in the Zarr metadata.

    Args:
        dataset: Normalized dataset (e.g. from
            :func:`ingestion.providers.noaa.parser.parse_grib2`).
        store: Local path, ``s3://`` URL, or a mutable mapping.
        chunks: Optional per-dimension chunk sizes.

    Returns:
        The store target as a string (for reporting).

    Raises:
        ValueError: If the store target is unsupported.
    """
    resolved = _resolve_store(store)
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
    return os.fspath(store) if isinstance(store, PathLike) else str(store)


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
    return xr.open_zarr(resolved)


def store_exists(store: str | PathLike[str] | Mapping[str, bytes]) -> bool:
    """Return whether a readable Zarr store already exists at ``store``.

    Used by the ingestion orchestration to decide whether a write is a fresh
    ingest (the target may be written in place) or a re-ingest (the target is
    already served by a ``ready`` catalog row and must not be truncated until
    the new store is known-good). A store that exists but fails to open is
    treated as absent so it is never written over destructively.

    Args:
        store: A local path, ``s3://`` URL, or mapping.

    Returns:
        True when a store can be opened at ``store``, False otherwise.
    """
    resolved = _resolve_store(store)
    if isinstance(resolved, str):
        return os.path.exists(resolved)
    try:
        dataset = xr.open_zarr(resolved)
    except Exception:
        return False
    dataset.close()
    return True
