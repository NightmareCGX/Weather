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
import struct
import threading
from collections import OrderedDict
from collections.abc import Generator, MutableMapping
from contextlib import contextmanager
from os import PathLike
from typing import Any

import numpy as np
import s3fs  # type: ignore[import-untyped]
import xarray as xr
from numcodecs import Zstd  # type: ignore[import-untyped]

from api.core.config import settings

#: Canonical Weather Platform Sharded v1 (sharded_v1) binary layout constants
SHARD_MAGIC: int = 0x53484152  # 'SHAR' in little-endian
INDEX_ENTRY_SIZE: int = 16     # uint64 offset, uint64 length
TRAILER_SIZE: int = 12         # uint32 num_chunks, uint32 index_byte_size, uint32 magic


class ShardedV1Reader:
    """Production reader for Weather Platform Sharded v1 (sharded_v1) stores.

    Performs granular byte-range GETs to read inner compressed chunks without
    downloading entire shard files, backed by process-local bounded LRU index
    and chunk caches.
    """

    def __init__(
        self,
        store: str | PathLike[str] | MutableMapping[str, bytes],
        *,
        max_cached_indices: int = 4096,
        max_cached_chunks: int = 2048,
    ) -> None:
        self.store = store
        self.max_cached_indices = max_cached_indices
        self.max_cached_chunks = max_cached_chunks
        self._compressor = Zstd(level=5)
        self._index_cache: OrderedDict[str, list[tuple[int, int]]] = OrderedDict()
        self._chunk_cache: OrderedDict[str, np.ndarray[Any, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._fs: Any | None = None

    def _resolve_fs_and_root(self) -> tuple[Any, str]:
        path = os.fspath(self.store) if isinstance(self.store, (str, PathLike)) else ""
        if path.startswith("s3://"):
            rest = path[len("s3://") :].strip("/")
            if self._fs is None:
                scheme = "https" if settings.MINIO_SECURE else "http"
                self._fs = s3fs.S3FileSystem(
                    key=settings.MINIO_ACCESS_KEY,
                    secret=settings.MINIO_SECRET_KEY,
                    client_kwargs={"endpoint_url": f"{scheme}://{settings.MINIO_ENDPOINT}"},
                    use_listings_cache=False,
                )
            return self._fs, rest
        return None, path

    def _get_shard_key(self, variable: str, member: int | None, lead_time_hours: int) -> str:
        if member is not None:
            return f"{variable}/shard.mem{member:03d}_L{lead_time_hours:04d}.shard"
        return f"{variable}/shard.det_L{lead_time_hours:04d}.shard"

    def get_shard_index(
        self,
        shard_key: str,
        expected_num_chunks: int = 120,
        *,
        generation: str | None = None,
    ) -> list[tuple[int, int]]:
        """Retrieve shard index from LRU cache or fetch via tail Range GET."""
        store_path = os.fspath(self.store) if isinstance(self.store, (str, PathLike)) else ""
        cache_key = f"{store_path}::{generation or 'live'}::{shard_key}"

        with self._cache_lock:
            if cache_key in self._index_cache:
                self._index_cache.move_to_end(cache_key)
                return self._index_cache[cache_key]

        fs, root = self._resolve_fs_and_root()
        trailer_and_index_len = expected_num_chunks * INDEX_ENTRY_SIZE + TRAILER_SIZE
        if fs is not None:
            full = f"{root}/{shard_key}"
            try:
                tail_bytes = fs.cat_file(full, start=-trailer_and_index_len)
            except Exception:
                return []
        else:
            full = os.path.join(root, *shard_key.split("/"))
            if not os.path.isfile(full):
                return []
            try:
                with open(full, "rb") as fh:
                    fh.seek(-trailer_and_index_len, os.SEEK_END)
                    tail_data = fh.read(trailer_and_index_len)
                tail_bytes = tail_data
            except Exception:
                return []

        if len(tail_bytes) < trailer_and_index_len:
            return []

        index_size = expected_num_chunks * INDEX_ENTRY_SIZE
        index_bytes = tail_bytes[:index_size]
        entries: list[tuple[int, int]] = []
        for i in range(expected_num_chunks):
            off, length = struct.unpack_from("<QQ", index_bytes, i * INDEX_ENTRY_SIZE)
            entries.append((off, length))

        with self._cache_lock:
            self._index_cache[cache_key] = entries
            if len(self._index_cache) > self.max_cached_indices:
                self._index_cache.popitem(last=False)

        return entries

    def read_chunk(
        self,
        variable: str,
        *,
        member: int | None,
        lead_time_hours: int,
        chunk_row: int,
        chunk_col: int,
        generation: str | None = None,
    ) -> np.ndarray[Any, Any]:
        """Read and decompress a 100x100 float32 chunk from the target shard container."""
        if chunk_row < 0 or chunk_row >= 8 or chunk_col < 0 or chunk_col >= 15:
            return np.full((100, 100), np.nan, dtype=np.float32)

        chunk_idx = chunk_row * 15 + chunk_col
        shard_key = self._get_shard_key(variable, member, lead_time_hours)
        store_path = os.fspath(self.store) if isinstance(self.store, (str, PathLike)) else ""
        chunk_cache_key = f"{store_path}::{generation or 'live'}::{shard_key}::{chunk_idx}"

        with self._cache_lock:
            if chunk_cache_key in self._chunk_cache:
                self._chunk_cache.move_to_end(chunk_cache_key)
                return self._chunk_cache[chunk_cache_key]

        entries = self.get_shard_index(shard_key, generation=generation)
        if not entries or chunk_idx >= len(entries):
            return np.full((100, 100), np.nan, dtype=np.float32)

        off, length = entries[chunk_idx]
        if length == 0:
            arr = np.full((100, 100), np.nan, dtype=np.float32)
        else:
            fs, root = self._resolve_fs_and_root()
            if fs is not None:
                full = f"{root}/{shard_key}"
                chunk_comp = fs.cat_file(full, start=off, end=off + length)
            else:
                full = os.path.join(root, *shard_key.split("/"))
                with open(full, "rb") as fh:
                    fh.seek(off)
                    chunk_comp = fh.read(length)

            raw = self._compressor.decode(chunk_comp)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(100, 100).copy()

        with self._cache_lock:
            self._chunk_cache[chunk_cache_key] = arr
            if len(self._chunk_cache) > self.max_cached_chunks:
                self._chunk_cache.popitem(last=False)

        return arr

    def read_point_value(
        self,
        variable: str,
        *,
        member: int | None,
        lead_time_hours: int,
        lat_idx: int,
        lon_idx: int,
        generation: str | None = None,
    ) -> float:
        """Read a single cell value via granular byte-range GET."""
        chunk_row = lat_idx // 100
        chunk_col = lon_idx // 100
        arr = self.read_chunk(
            variable,
            member=member,
            lead_time_hours=lead_time_hours,
            chunk_row=chunk_row,
            chunk_col=chunk_col,
            generation=generation,
        )
        sub_lat = lat_idx % 100
        sub_lon = lon_idx % 100
        return float(arr[sub_lat, sub_lon])

    def interpolate_point(
        self,
        variable: str,
        *,
        member: int | None,
        lead_time_hours: int,
        lat_idx: list[int],
        lon_idx: list[int],
        t_row: float,
        t_col: float,
        generation: str | None = None,
    ) -> float:
        """Bilinearly interpolate a variable at 2x2 neighborhood coordinates."""
        lat0, lat1 = lat_idx[0], lat_idx[1]
        lon0, lon1 = lon_idx[0], lon_idx[1]

        r0, c0 = lat0 // 100, lon0 // 100
        r1, c1 = lat1 // 100, lon1 // 100

        if r0 == r1 and c0 == c1:
            arr = self.read_chunk(
                variable,
                member=member,
                lead_time_hours=lead_time_hours,
                chunk_row=r0,
                chunk_col=c0,
                generation=generation,
            )
            val_00 = float(arr[lat0 % 100, lon0 % 100])
            val_01 = float(arr[lat0 % 100, lon1 % 100])
            val_10 = float(arr[lat1 % 100, lon0 % 100])
            val_11 = float(arr[lat1 % 100, lon1 % 100])
        else:
            val_00 = self.read_point_value(
                variable, member=member, lead_time_hours=lead_time_hours, lat_idx=lat0, lon_idx=lon0, generation=generation
            )
            val_01 = self.read_point_value(
                variable, member=member, lead_time_hours=lead_time_hours, lat_idx=lat0, lon_idx=lon1, generation=generation
            )
            val_10 = self.read_point_value(
                variable, member=member, lead_time_hours=lead_time_hours, lat_idx=lat1, lon_idx=lon0, generation=generation
            )
            val_11 = self.read_point_value(
                variable, member=member, lead_time_hours=lead_time_hours, lat_idx=lat1, lon_idx=lon1, generation=generation
            )

        lower = val_00 + (val_01 - val_00) * t_col
        upper = val_10 + (val_11 - val_10) * t_col
        return float(lower + (upper - lower) * t_row)

    def read_window(
        self,
        variable: str,
        *,
        member: int | None,
        lead_time_hours: int,
        lat_min: int,
        lat_max: int,
        lon_min: int,
        lon_max: int,
        generation: str | None = None,
    ) -> np.ndarray[Any, Any]:
        """Read a bounded rectangular spatial window [lat_min..lat_max, lon_min..lon_max] (inclusive)."""
        lat_len = lat_max - lat_min + 1
        lon_len = lon_max - lon_min + 1
        window = np.full((lat_len, lon_len), np.nan, dtype=np.float32)

        r_start = lat_min // 100
        r_end = lat_max // 100
        c_start = lon_min // 100
        c_end = lon_max // 100

        for r_chunk in range(r_start, r_end + 1):
            chunk_lat_start = r_chunk * 100
            chunk_lat_end = min((r_chunk + 1) * 100, 721)

            sub_lat_start = max(0, lat_min - chunk_lat_start)
            sub_lat_end = min(chunk_lat_end - chunk_lat_start, lat_max - chunk_lat_start + 1)
            win_lat_start = max(0, chunk_lat_start - lat_min)
            win_lat_end = win_lat_start + (sub_lat_end - sub_lat_start)

            for c_chunk in range(c_start, c_end + 1):
                chunk_lon_start = c_chunk * 100
                chunk_lon_end = min((c_chunk + 1) * 100, 1440)

                sub_lon_start = max(0, lon_min - chunk_lon_start)
                sub_lon_end = min(chunk_lon_end - chunk_lon_start, lon_max - chunk_lon_start + 1)
                win_lon_start = max(0, chunk_lon_start - lon_min)
                win_lon_end = win_lon_start + (sub_lon_end - sub_lon_start)

                if sub_lat_end > sub_lat_start and sub_lon_end > sub_lon_start:
                    chunk_arr = self.read_chunk(
                        variable,
                        member=member,
                        lead_time_hours=lead_time_hours,
                        chunk_row=r_chunk,
                        chunk_col=c_chunk,
                        generation=generation,
                    )
                    window[win_lat_start:win_lat_end, win_lon_start:win_lon_end] = chunk_arr[
                        sub_lat_start:sub_lat_end, sub_lon_start:sub_lon_end
                    ]

        return window

    def read_ensemble_mean_window(
        self,
        variable: str,
        *,
        members: tuple[int, ...],
        lead_time_hours: int,
        lat_min: int,
        lat_max: int,
        lon_min: int,
        lon_max: int,
        expected_members: int = 30,
        generation: str | None = None,
    ) -> np.ndarray[Any, Any]:
        """Read and compute member-mean across ensemble members for a spatial window."""
        if not members:
            return np.full((lat_max - lat_min + 1, lon_max - lon_min + 1), np.nan, dtype=np.float32)

        stack: list[np.ndarray[Any, Any]] = []
        for m in members:
            w = self.read_window(
                variable,
                member=m,
                lead_time_hours=lead_time_hours,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                generation=generation,
            )
            stack.append(w)

        from domain.coverage import is_cell_statistically_valid

        arr_stack = np.stack(stack, axis=0)
        finite_counts = np.sum(np.isfinite(arr_stack), axis=0)
        valid_cells = is_cell_statistically_valid(finite_counts, expected_members)
        with np.errstate(all="ignore"):
            mean_vals = np.nanmean(arr_stack, axis=0)
            return np.where(valid_cells, mean_vals, np.nan).astype(np.float32)


_readers: dict[str, ShardedV1Reader] = {}
_readers_lock = threading.Lock()


def get_sharded_reader(
    store: str | PathLike[str] | MutableMapping[str, bytes],
) -> ShardedV1Reader:
    """Return a process-cached ShardedV1Reader for the store path."""
    path = os.fspath(store) if isinstance(store, (str, PathLike)) else id(store)
    path_key = str(path)
    with _readers_lock:
        if path_key not in _readers:
            _readers[path_key] = ShardedV1Reader(store)
        return _readers[path_key]


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
        use_listings_cache=False,
    )
    # ``get_mapper`` is untyped in the s3fs stub (import-untyped), so mypy
    # sees ``Any``. The declared ``MutableMapping[str, bytes]`` return is
    # enforced by assigning through a concrete-typed intermediate instead of
    # returning the ``Any`` directly.
    mapper: MutableMapping[str, bytes] = fs.get_mapper(rest)
    return mapper


def read_dataset(store: str | PathLike[str] | MutableMapping[str, bytes]) -> xr.Dataset:
    """Read a Zarr store back into a numpy-backed dataset (always fresh).

    This is the uncached open. The serving paths use
    :func:`open_serving_dataset` (or :func:`read_dataset_cached`), which reuses
    the lazily-opened dataset per ``(store_path, serving_generation)``; this
    function remains for callers that need an unconditional fresh open (and as
    the opener the cache invokes on miss).

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


@contextmanager
def open_serving_dataset(
    store: str | PathLike[str] | MutableMapping[str, bytes],
) -> Generator[xr.Dataset, None, None]:
    """Provide an opened xarray.Dataset with deterministic ownership and caching.

    * For in-memory / MutableMapping stores: opens fresh and closes on context exit.
    * For path-backed stores with a valid trusted generation (marker-v1):
      borrows the cached Dataset from StoreHandleCache; does NOT close it on context exit.
    * For path-backed stores with confirmed missing manifest (legacy):
      opens a fresh Dataset via read_dataset; deterministically closes it on context exit via finally.
    * For malformed manifests: raises ManifestReadError (fail closed).
    * For infrastructure/IO failures: propagates the error (fail closed).
    """
    if isinstance(store, MutableMapping):
        ds = read_dataset(store)
        try:
            yield ds
        finally:
            ds.close()
        return

    path = os.fspath(store)
    from api.core.manifest_reader import manifest_generation

    generation = manifest_generation(path)
    if generation is None:
        # Confirmed absent manifest (legacy compatibility path):
        # Open fresh, bypass StoreHandleCache, close deterministically on exit.
        ds = read_dataset(path)
        try:
            yield ds
        finally:
            ds.close()
    else:
        # Trusted generation: borrow from StoreHandleCache, do not close on exit.
        from api.core.store_cache import store_handle_cache

        def _open() -> xr.Dataset:
            return read_dataset(path)

        dataset, _hit = store_handle_cache.get_or_open((path, generation), _open)
        yield dataset


def read_dataset_cached(
    store: str | PathLike[str] | MutableMapping[str, bytes],
) -> xr.Dataset:
    """Return the lazily-opened dataset for a store, reusing per trusted generation.

    When a trusted committed generation exists (marker-v1), the lazy dataset
    is reused from StoreHandleCache. When no manifest exists (legacy store),
    caching is bypassed and a fresh dataset is returned.

    Note: Prefer :func:`open_serving_dataset` for request-bounded serving
    reads to ensure deterministic dataset cleanup when uncached.
    """
    if isinstance(store, MutableMapping):
        return read_dataset(store)
    path = os.fspath(store)
    from api.core.manifest_reader import manifest_generation

    generation = manifest_generation(path)
    if generation is None:
        return read_dataset(path)

    from api.core.store_cache import store_handle_cache

    def _open() -> xr.Dataset:
        return read_dataset(path)

    dataset, _hit = store_handle_cache.get_or_open((path, generation), _open)
    return dataset
