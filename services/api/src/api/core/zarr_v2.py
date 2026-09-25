"""Reader for Weather Platform Sharded v2 (``sharded_v2``) stores.

``sharded_v2`` keeps the frozen sharded_v1 container geometry (120 inner
100x100 chunks per shard, identical index/trailer/object keys, Zstd level 5)
but persists each variable in its authoritative little-endian payload dtype
from :func:`domain.storage_dtype.resolve_storage_dtype`:

* ``<f2`` continuous fields (temperature, RH, wind, gust, rate, cloud cover,
  snow depth, visibility);
* ``<f4`` semantic-compatibility exceptions (``precipitation_amount_3h``,
  ``cloud_ceiling``) and ensemble-mean probability flags;
* ``u1`` deterministic/member categorical flags.

This reader subclasses the frozen :class:`ShardedV1Reader` and overrides only
the chunk decode: index parsing, shard-key layout, placement math, the bilinear
helper, the caches, and the executors are inherited unchanged. The chunk cache
holds the **native persisted dtype** (f16 chunks stay f16, u8 stay u8) — the
API RAM saving comes exactly from that; promotion to the compute domain happens
only at the boundary (four corner scalars for points, the request-level f32
working window for maps, the existing float32/float64 statistics inputs for
ensembles).
"""

from __future__ import annotations

import os
from os import PathLike
from typing import Any

import numpy as np

from domain.reclamation import TARGET_KIND_DET, TARGET_KIND_MEAN, TARGET_KIND_MEM
from domain.storage_dtype import (
    SHARDED_V2_FORMAT_VERSION,
    resolve_storage_dtype,
)

from api.core.zarr import ShardedV1Reader

#: Chunks per shard axis: the canonical 8x15 = 120 chunk grid over 721x1440.
_CHUNK_ROWS = 8
_CHUNK_COLS = 15
_CHUNK_CELLS = 100 * 100


class ShardedV2Reader(ShardedV1Reader):
    """Chunk reader for sharded_v2 stores with native-dtype caching.

    Byte-level differences from v1 are exactly one thing: the payload dtype of
    each chunk, resolved per (variable, product role) from the frozen v2
    matrix. Missing/absent chunks fill with the dtype's own null value (NaN for
    floats, 0 for uint8 flags) so callers see the same "absent" semantics v1
    produces.
    """

    def _product_role(self, member: int | None, is_mean: bool) -> str:
        if is_mean:
            return TARGET_KIND_MEAN
        if member is not None:
            return TARGET_KIND_MEM
        return TARGET_KIND_DET

    def _empty_chunk(self, variable: str, member: int | None, is_mean: bool) -> np.ndarray[Any, Any]:
        dtype = resolve_storage_dtype(
            SHARDED_V2_FORMAT_VERSION, variable, self._product_role(member, is_mean)
        )
        fill: Any = np.nan if dtype.kind == "f" else 0
        return np.full((100, 100), fill, dtype=dtype)

    def read_chunk(
        self,
        variable: str,
        *,
        member: int | None,
        lead_time_hours: int,
        chunk_row: int,
        chunk_col: int,
        generation: str | None = None,
        is_mean: bool = False,
    ) -> np.ndarray[Any, Any]:
        """Read and decompress one 100x100 chunk in its native persisted dtype."""
        if chunk_row < 0 or chunk_row >= _CHUNK_ROWS or chunk_col < 0 or chunk_col >= _CHUNK_COLS:
            return self._empty_chunk(variable, member, is_mean)

        chunk_idx = chunk_row * _CHUNK_COLS + chunk_col
        shard_key = self._get_shard_key(variable, member, lead_time_hours, is_mean=is_mean)
        store_path = os.fspath(self.store) if isinstance(self.store, (str, PathLike)) else ""
        chunk_cache_key = f"{store_path}::{generation or 'live'}::{shard_key}::{chunk_idx}"

        with self._cache_lock:
            if chunk_cache_key in self._chunk_cache:
                self._chunk_cache.move_to_end(chunk_cache_key)
                return self._chunk_cache[chunk_cache_key]

        entries = self.get_shard_index(shard_key, generation=generation)
        # len() rather than truthiness: an ndarray's bool is ambiguous (and raises).
        if len(entries) == 0 or chunk_idx >= len(entries):
            return self._empty_chunk(variable, member, is_mean)

        # Explicit int conversion: numpy scalars must reach s3fs as plain ints
        # so the Range header is formatted as an integer byte range.
        off, length = int(entries[chunk_idx][0]), int(entries[chunk_idx][1])
        if length == 0:
            arr = self._empty_chunk(variable, member, is_mean)
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
            dtype = resolve_storage_dtype(
                SHARDED_V2_FORMAT_VERSION, variable, self._product_role(member, is_mean)
            )
            if len(raw) != _CHUNK_CELLS * dtype.itemsize:
                raise ValueError(
                    f"sharded_v2 chunk byte length {len(raw)} does not match "
                    f"dtype {dtype.str!r} ({_CHUNK_CELLS * dtype.itemsize} expected) "
                    f"for {shard_key!r} chunk {chunk_idx}: payload and dtype "
                    "matrix disagree."
                )
            arr = np.frombuffer(raw, dtype=dtype).reshape(100, 100).copy()

        with self._cache_lock:
            self._chunk_cache[chunk_cache_key] = arr
            if len(self._chunk_cache) > self.max_cached_chunks:
                self._chunk_cache.popitem(last=False)

        return arr
