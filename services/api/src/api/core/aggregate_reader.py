"""Aggregate (statistic) shard reader: the serving counterpart of the ingestion writer.

The container is written by ``ingestion.core.aggregate_writer``; this module reads it without
importing that package, because the API tier must stay independently deployable -- the same
constraint the member-shard reader carries. Both sides take the byte layout from
``domain.shard_format``, so the format itself cannot drift; what is duplicated here is only the
geometry and the access pattern, and a test asserts the two agree on a container written by
the real writer.

What it optimises
-----------------
The aggregate's consumers are the point endpoints -- ``/v1/ensembles``,
``/v1/probabilities`` and the ensemble branch of ``/v1/points`` -- which need **every stored
field at one latitude/longitude**. The container therefore lays a location's fields
contiguously, and :meth:`AggregateShardReader.read_location` fetches that whole group as a
**single byte range**. Measured at 4 ranges for a point's 2x2 corner window instead of 38-68.

The gate is deliberately conservative
-------------------------------------
:meth:`AggregateShardReader.open` returns ``None`` for anything it cannot interpret with
certainty: a v1 container, an unknown encoding id, a chunk count that does not divide into
whole field planes, a truncated tail. A caller falls back to the member shards. That is what
lets the read path be switched on per store while the write path is still being phased in --
an unreadable aggregate is a missing optimisation, never a wrong answer.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt
from domain.shard_format import (
    DESCRIPTOR_SIZE,
    ENCODING_F32,
    INDEX_ENTRY_SIZE,
    TRAILER_SIZE,
    ShardDescriptor,
    ShardFormatError,
    parse_index,
    parse_trailer,
    split_v2_tail,
)
from numcodecs import Zstd  # type: ignore[import-untyped]

#: Encodings this reader can decode. A container whose descriptor names an id absent here is
#: refused rather than guessed at, so a newer writer cannot be half-read by an older reader.
KNOWN_ENCODINGS: frozenset[int] = frozenset({ENCODING_F32})

#: Store-relative suffix of an aggregate shard object, matching the writer's key grammar.
AGGREGATE_SHARD_SUFFIX: str = "shard.agg"

#: A zstd frame is self-describing, so no write level is negotiated with the writer.
_DECODER = Zstd()


def aggregate_shard_key(variable: str, lead_time_hours: int) -> str:
    """Store-relative key of one ``(variable, lead)``'s aggregate object."""
    return f"{variable}/{AGGREGATE_SHARD_SUFFIX}_L{int(lead_time_hours):04d}.shard"


@dataclass(frozen=True)
class AggregateGeometry:
    """Chunk addressing for one aggregate container, recovered from its descriptor.

    Every field is derived rather than passed in: a reader opening a store it did not write
    must not assume the field count, the grid or the chunk shape, which is exactly what the
    descriptor exists to remove.

    Attributes:
        n_fields: Number of statistic planes.
        grid_lat: Grid latitude extent.
        grid_lon: Grid longitude extent.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.
        num_chunks: Total chunk count across every field.
    """

    n_fields: int
    grid_lat: int
    grid_lon: int
    chunk_lat: int
    chunk_lon: int
    num_chunks: int

    @property
    def lat_chunks(self) -> int:
        """Number of chunks along latitude."""
        return -(-self.grid_lat // self.chunk_lat)

    @property
    def lon_chunks(self) -> int:
        """Number of chunks along longitude."""
        return -(-self.grid_lon // self.chunk_lon)

    def in_bounds(self, chunk_row: int, chunk_col: int) -> bool:
        """Whether a chunk-grid location exists."""
        return 0 <= chunk_row < self.lat_chunks and 0 <= chunk_col < self.lon_chunks

    def ordinal(self, field: int, chunk_row: int, chunk_col: int) -> int:
        """Chunk ordinal for one ``(field, row, col)``, matching the writer's layout."""
        return (chunk_row * self.lon_chunks + chunk_col) * self.n_fields + field

    def spatial_group(self, chunk_row: int, chunk_col: int) -> list[int]:
        """Chunk ordinals for every field at one location, in field order.

        This is the group a point query needs whole, and the reason the container lays a
        location's fields contiguously.
        """
        start = (chunk_row * self.lon_chunks + chunk_col) * self.n_fields
        return list(range(start, start + self.n_fields))


def recover_geometry(descriptor: ShardDescriptor) -> AggregateGeometry | None:
    """Derive chunk addressing from a descriptor, or ``None`` if it is not an aggregate.

    Returns ``None`` rather than raising for a shape that cannot be an aggregate -- an unknown
    encoding, an empty container, or a chunk count that does not divide into whole field
    planes -- because the caller's response is to fall back to the member shards, not to fail
    the request.
    """
    if descriptor.encoding_id not in KNOWN_ENCODINGS:
        return None
    lat_chunks, lon_chunks = descriptor.expected_shape()
    chunks_per_field = lat_chunks * lon_chunks
    if chunks_per_field <= 0 or descriptor.num_chunks <= 0:
        return None
    if descriptor.num_chunks % chunks_per_field:
        return None
    return AggregateGeometry(
        n_fields=descriptor.num_chunks // chunks_per_field,
        grid_lat=descriptor.grid_lat,
        grid_lon=descriptor.grid_lon,
        chunk_lat=descriptor.chunk_lat,
        chunk_lon=descriptor.chunk_lon,
        num_chunks=descriptor.num_chunks,
    )


class AggregateShardReader:
    """Reads aggregate shards, caching indices and decoded field groups.

    Both caches are bounded LRU, mirroring the member-shard reader's discipline. A whole field
    *group* is cached rather than individual chunks, because the access pattern is "every
    field at one location": caching chunks separately would add bookkeeping to a fetch that
    always asks for the group.
    """

    def __init__(
        self,
        store: str | PathLike[str] | MutableMapping[str, bytes],
        *,
        max_cached_indices: int = 64,
        max_cached_groups: int = 128,
    ) -> None:
        self.store = store
        self.max_cached_indices = max_cached_indices
        self.max_cached_groups = max_cached_groups
        # Keyed by shard and generation: a bumped generation must not reuse a stale index.
        self._index_cache: OrderedDict[str, list[tuple[int, int]]] = OrderedDict()
        self._geometry_cache: OrderedDict[str, AggregateGeometry] = OrderedDict()
        self._group_cache: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._fs: Any | None = None

    # -- store access ------------------------------------------------------------

    def _resolve_fs_and_root(self) -> tuple[Any, str]:
        """Resolve an ``s3://`` store to a filesystem and bucket/prefix.

        Builds its own client rather than importing the ingestion package, because the API tier
        must remain independently deployable.
        """
        path = os.fspath(self.store) if isinstance(self.store, (str, PathLike)) else ""
        if path.startswith("s3://"):
            import s3fs  # type: ignore[import-untyped]

            from api.core.config import settings

            rest = path[len("s3://") :].strip("/")
            with self._cache_lock:
                if self._fs is None:
                    scheme = "https" if settings.MINIO_SECURE else "http"
                    self._fs = s3fs.S3FileSystem(
                        key=settings.MINIO_ACCESS_KEY,
                        secret=settings.MINIO_SECRET_KEY,
                        client_kwargs={
                            "endpoint_url": f"{scheme}://{settings.MINIO_ENDPOINT}"
                        },
                        config_kwargs={
                            "max_pool_connections": int(
                                settings.API_S3_MAX_POOL_CONNECTIONS
                            )
                        },
                        use_listings_cache=False,
                    )
            return self._fs, rest
        return None, path

    def _read_bytes(
        self, key: str, *, start: int | None = None, end: int | None = None
    ) -> bytes | None:
        """Read a whole object or a byte range. ``None`` when absent or unreadable.

        A read failure means "no aggregate here", not an error: the caller's fallback is the
        member shards, so a missing aggregate is a missing optimisation.
        """
        if isinstance(self.store, MutableMapping):
            # An in-memory store is a whole object or nothing: a range read would have to
            # slice, and the callers that use a mapping are tests.
            blob = self.store.get(key)
            if blob is None:
                return None
            data = bytes(blob)
            if start is None:
                return data
            if start < 0:
                return data[start:] if end is None else data[start:end]
            return data[start:] if end is None else data[start:end]

        fs, root = self._resolve_fs_and_root()
        try:
            if fs is not None:
                if start is None:
                    return bytes(fs.cat_file(f"{root}/{key}"))
                return bytes(fs.cat_file(f"{root}/{key}", start=start, end=end))
            full = os.path.join(root, *key.split("/"))
            if not os.path.isfile(full):
                return None
            with open(full, "rb") as handle:
                if start is None:
                    return handle.read()
                if start < 0:
                    # A tail read. The negative offset is what the trailer probe uses, and a
                    # forward seek would raise on it.
                    handle.seek(start, os.SEEK_END)
                    if end is None:
                        return handle.read()
                    return handle.read(end)
                handle.seek(start)
                if end is None:
                    return handle.read()
                return handle.read(end - start)
        except Exception:  # noqa: BLE001 - degraded to "absent" by design
            return None

    def _store_key(self) -> str:
        if isinstance(self.store, (str, PathLike)):
            return os.fspath(self.store)
        return str(id(self.store))

    # -- container parsing -------------------------------------------------------

    def _load_geometry(
        self, shard_key: str, generation: str | None
    ) -> tuple[AggregateGeometry, list[tuple[int, int]]] | None:
        """Fetch and validate a container's tail, yielding its geometry and index.

        ``None`` for anything that is not a well-formed, decodable aggregate container. Those
        checks are the point of this method: a v1 object, an unknown encoding, or a geometry
        that cannot be an aggregate all mean "no aggregate here".
        """
        cache_key = f"{self._store_key()}::{generation or 'live'}::{shard_key}"
        with self._cache_lock:
            if cache_key in self._geometry_cache:
                self._geometry_cache.move_to_end(cache_key)
                self._index_cache.move_to_end(cache_key)
                return self._geometry_cache[cache_key], self._index_cache[cache_key]

        # The descriptor's length depends on num_chunks, which only the trailer declares, so
        # the tail is read in two steps: 12 bytes for the trailer, then the rest.
        trailer_bytes = self._read_bytes(shard_key, start=-TRAILER_SIZE)
        if trailer_bytes is None or len(trailer_bytes) < TRAILER_SIZE:
            return None
        try:
            trailer = parse_trailer(trailer_bytes)
        except ShardFormatError:
            return None
        if not trailer.is_v2:
            return None

        tail_size = trailer.num_chunks * INDEX_ENTRY_SIZE + DESCRIPTOR_SIZE + TRAILER_SIZE
        tail = self._read_bytes(shard_key, start=-tail_size)
        if tail is None or len(tail) < tail_size:
            return None
        try:
            index_bytes, descriptor = split_v2_tail(tail, trailer.num_chunks)
            entries = parse_index(index_bytes, trailer.num_chunks)
        except ShardFormatError:
            return None

        geometry = recover_geometry(descriptor)
        if geometry is None or geometry.num_chunks != len(entries):
            return None

        with self._cache_lock:
            self._geometry_cache[cache_key] = geometry
            self._index_cache[cache_key] = entries
            while len(self._geometry_cache) > self.max_cached_indices:
                self._geometry_cache.popitem(last=False)
                self._index_cache.popitem(last=False)
        return geometry, entries

    # -- public API --------------------------------------------------------------

    def open(
        self, variable: str, lead_time_hours: int, *, generation: str | None = None
    ) -> AggregateGeometry | None:
        """Return a variable's aggregate geometry, or ``None`` when it is unusable.

        This is the capability probe: a caller tests for an aggregate and falls back to the
        member shards without excepting, which is what makes the phase-in incremental.
        """
        loaded = self._load_geometry(
            aggregate_shard_key(variable, lead_time_hours), generation
        )
        return None if loaded is None else loaded[0]

    def read_location(
        self,
        variable: str,
        *,
        lead_time_hours: int,
        chunk_row: int,
        chunk_col: int,
        generation: str | None = None,
    ) -> npt.NDArray[np.float32] | None:
        """Read every field's chunk at one location as one ``(n_fields, lat, lon)`` stack.

        A single byte range covers the whole group, because the container lays a location's
        fields contiguously. ``None`` when the aggregate is absent or unreadable, or the
        location is off-grid.

        Raises:
            ShardFormatError: if the container's index disagrees with its payload, or a chunk
                does not decode to the geometry's chunk size. Unlike an absent shard, that is
                corrupt state, and degrading it silently would hide the corruption.
        """
        shard_key = aggregate_shard_key(variable, lead_time_hours)
        loaded = self._load_geometry(shard_key, generation)
        if loaded is None:
            return None
        geometry, entries = loaded
        if not geometry.in_bounds(chunk_row, chunk_col):
            return None

        cache_key = (
            f"{self._store_key()}::{generation or 'live'}::{shard_key}::{chunk_row}.{chunk_col}"
        )
        with self._cache_lock:
            if cache_key in self._group_cache:
                self._group_cache.move_to_end(cache_key)
                return self._group_cache[cache_key]

        group = geometry.spatial_group(chunk_row, chunk_col)
        stack = np.full(
            (geometry.n_fields, geometry.chunk_lat, geometry.chunk_lon),
            np.nan,
            dtype=np.float32,
        )
        offsets = [entries[ordinal][0] for ordinal in group]
        lengths = [entries[ordinal][1] for ordinal in group]
        if all(length == 0 for length in lengths):
            # An all-fill location: NaN planes without a fetch, matching the member reader's
            # treatment of an intentionally omitted chunk.
            self._cache_group(cache_key, stack)
            return stack

        # One range covering the group. The chunks are contiguous by construction, but the span
        # is computed from the entries rather than assumed, so a container with a gap fails on
        # the length check below instead of returning shifted data.
        start = int(min(offsets))
        end = int(
            max(offset + length for offset, length in zip(offsets, lengths, strict=True))
        )
        blob = self._read_bytes(shard_key, start=start, end=end)
        if blob is None:
            return None

        expected_bytes = geometry.chunk_lat * geometry.chunk_lon * 4
        for index, ordinal in enumerate(group):
            offset, length = entries[ordinal]
            if length == 0:
                continue
            relative = int(offset) - start
            payload = blob[relative : relative + int(length)]
            if len(payload) != int(length):
                raise ShardFormatError(
                    f"{shard_key}: chunk {ordinal} declares {length} bytes but the fetched "
                    f"range yielded {len(payload)}; the index disagrees with the payload"
                )
            try:
                raw = _DECODER.decode(payload)
            except Exception as exc:  # noqa: BLE001 - a decode failure is corrupt state
                # A chunk that will not decode means the container is damaged, not that the
                # aggregate is absent. Wrapping keeps the reader's error contract uniform:
                # absent aggregates return None, damaged ones raise ShardFormatError.
                raise ShardFormatError(
                    f"{shard_key}: chunk {ordinal} failed to decode: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if len(raw) != expected_bytes:
                raise ShardFormatError(
                    f"{shard_key}: chunk {ordinal} decoded to {len(raw)} bytes, expected "
                    f"{expected_bytes}"
                )
            stack[index] = np.frombuffer(raw, dtype=np.float32).reshape(
                geometry.chunk_lat, geometry.chunk_lon
            )

        self._cache_group(cache_key, stack)
        return stack

    def _cache_group(self, cache_key: str, stack: npt.NDArray[np.float32]) -> None:
        with self._cache_lock:
            self._group_cache[cache_key] = stack
            while len(self._group_cache) > self.max_cached_groups:
                self._group_cache.popitem(last=False)

    def invalidate(self) -> None:
        """Drop every cached index, geometry and field group."""
        with self._cache_lock:
            self._index_cache.clear()
            self._geometry_cache.clear()
            self._group_cache.clear()


__all__ = [
    "AGGREGATE_SHARD_SUFFIX",
    "KNOWN_ENCODINGS",
    "AggregateGeometry",
    "AggregateShardReader",
    "aggregate_shard_key",
    "recover_geometry",
]
