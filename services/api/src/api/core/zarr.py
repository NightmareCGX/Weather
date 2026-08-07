"""Zarr store reading for the API serving tier.

The API tier reads normalized forecast datasets from Zarr stores (local
directory or an ``s3://`` bucket) to serve point forecasts, probabilities, and
ensemble statistics. The read helper lives here so the API service does not
import the ingestion package at runtime, keeping the two services
independently deployable (``docs/ARCHITECTURE.md`` sections 3.1/3.5).

Only the read path is provided: the API never writes Zarr stores (ingestion
owns writes). Reading resolves the same ``s3://`` scheme the ingestion writer
uses so a store produced by the ingestion worker is readable here.
"""

from __future__ import annotations

import os
from os import PathLike
from typing import MutableMapping

import s3fs  # type: ignore[import-untyped]
import xarray as xr

from api.core.config import settings


def _resolve_store(
    store: str | PathLike[str] | MutableMapping[str, bytes],
) -> str | PathLike[str] | MutableMapping[str, bytes]:
    """Resolve a Zarr store target to an xarray-readable location.

    Args:
        store: A local directory path, an ``s3://`` URL, or an existing
            mutable mapping.

    Returns:
        An object accepted by :func:`xarray.open_zarr`.

    Raises:
        ValueError: If an ``s3://`` URL cannot be parsed.
    """
    if isinstance(store, MutableMapping):
        return store
    path = os.fspath(store)
    if path.startswith("s3://"):
        return _resolve_s3_store(path)
    if path.startswith("file://"):
        return path[len("file://") :]
    return path


def _resolve_s3_store(path: str) -> MutableMapping[str, bytes]:
    """Build an ``FSMap`` over an ``s3://`` URL using the API's S3 settings.

    Args:
        path: An ``s3://bucket/prefix`` URL.

    Returns:
        An ``s3fs`` mapping object accepted by xarray.

    Raises:
        ValueError: If the bucket/prefix cannot be derived from the URL.
    """
    rest = path[len("s3://") :].strip("/")
    if not rest:
        raise ValueError(f"Invalid S3 store URL: {path!r}")
    scheme = "https" if settings.MINIO_SECURE else "http"
    fs = s3fs.S3FileSystem(
        key=settings.MINIO_ACCESS_KEY,
        secret=settings.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": f"{scheme}://{settings.MINIO_ENDPOINT}"},
    )
    # ``get_mapper`` is untyped in the s3fs stub (import-untyped), so mypy
    # sees ``Any``. The declared ``MutableMapping[str, bytes]`` return is
    # enforced by assigning through a concrete-typed intermediate instead of
    # returning the ``Any`` directly.
    mapper: MutableMapping[str, bytes] = fs.get_mapper(rest)
    return mapper


def read_dataset(store: str | PathLike[str] | MutableMapping[str, bytes]) -> xr.Dataset:
    """Read a Zarr store back into a numpy-backed dataset.

    Args:
        store: A local path, ``s3://`` URL, or mapping to read.

    Returns:
        The dataset read back from the Zarr store.
    """
    resolved = _resolve_store(store)
    # ``xr.open_zarr`` is overloaded and returns ``Dataset`` only for some
    # overloads; with the ``Any``-typed store mypy infers ``Any``. Narrowing
    # through a ``Dataset``-typed intermediate enforces the declared return
    # type (the value is always a concrete ``Dataset`` at runtime).
    dataset: xr.Dataset = xr.open_zarr(resolved)
    return dataset
