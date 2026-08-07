"""Test-only Zarr writer for the API test fixtures.

The API service is independently deployable and must not import the ingestion
package at runtime (``docs/ARCHITECTURE.md`` sections 3.1/3.5). The API test
fixtures need to write deterministic Zarr stores locally, so this module
provides a minimal, test-only writer that mirrors the ingestion writer's
chunking and compression. It is used only by ``tests/fixtures/`` and is not
part of the production API serving path.
"""

from __future__ import annotations

import os
from os import PathLike
from typing import Mapping

import xarray as xr
from numcodecs import Zstd  # type: ignore[import-untyped]

#: Default chunks applied per dimension when none are provided.
DEFAULT_CHUNKS: Mapping[str, int] = {
    "time": 1,
    "lead_time_hours": 1,
    "isobaricInhPa": 1,
    "latitude": 100,
    "longitude": 100,
}


def write_dataset(
    dataset: xr.Dataset,
    store: str | PathLike[str],
    *,
    chunks: Mapping[str, int] | None = None,
) -> str:
    """Write a dataset to a local Zarr store with Zstd compression.

    Args:
        dataset: The dataset to persist.
        store: A local directory path (the only supported target).
        chunks: Optional per-dimension chunk sizes; defaults to
            :data:`DEFAULT_CHUNKS` with the full extent of any dimension not
            covered.

    Returns:
        The store path as a string.

    Raises:
        ValueError: If ``store`` is not a local path.
    """
    path = os.fspath(store)
    defaults = chunks or DEFAULT_CHUNKS

    def _chunk_sizes(name: object) -> tuple[int, ...]:
        def _size(dim: object) -> int:
            size = dataset.sizes[dim]
            assert size is not None
            return size

        def _default(dim: object) -> int:
            return defaults.get(str(dim), _size(dim))

        return tuple(
            min(_default(dim), _size(dim))
            for dim in dataset[name].dims  # type: ignore[index]
        )

    encoding = {
        name: {"chunks": _chunk_sizes(name), "compressor": Zstd(level=5)}
        for name in dataset.data_vars
    }
    dataset.to_zarr(path, mode="w", encoding=encoding)
    return path
