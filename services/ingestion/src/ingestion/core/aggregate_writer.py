"""Encoding of aggregate (statistic) shards into ``sharded_v2`` containers.

An aggregate shard replaces the members of one ``(variable, lead)`` region with a small set
of statistic field planes computed from the whole member set. The planes are laid out
**spatially first**: chunk ordinal ``k`` addresses ``within, field = divmod(k, n_fields)`` and
``row, col = divmod(within, lon_chunks)``, so one spatial location's every field sits
contiguously.

The ordering is chosen for the aggregate's actual consumer, which is the point path:
``/v1/ensembles``, ``/v1/probabilities`` and the ensemble branch of ``/v1/points`` all resolve
one latitude/longitude and need every stored field's 2x2 corner window. Laying the fields of
one location together makes that window four contiguous ranges instead of 2 x n_fields of them
-- measured at 4 range GETs versus 38-68 depending on the encoding. Each fetch costs ~2.5 ms
on the deployed host, dominated by s3fs/Python overhead rather than bandwidth, so GET count is
the cost to minimise.

The bytes are identical under either ordering, so the non-consumer loses nothing. A
whole-field window read would prefer the other order, but no aggregate consumer does that:
map tiles and the wind vector field read the separate precomputed MEAN shard
(``has_mean_shard``) and never reduce members at serve time.

Nothing here chooses *which* statistics to store; that is
:mod:`domain.aggregate`'s job. This module is only the container plumbing.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numcodecs import Zstd  # type: ignore[import-untyped]

from domain.aggregate import (
    KIND_MEAN_STD_BINS,
    NAN_SENTINEL,
    AggregateError,
    AggregateSpec,
    dequantise_field,
    quantise_field,
)
from domain.shard_format import (
    DESCRIPTOR_SIZE,
    ENCODING_F32,
    ENCODING_I16,
    INDEX_ENTRY_SIZE,
    PER_FIELD_SCALE,
    SHARD_V2_MAGIC,
    TRAILER_SIZE,
    ShardDescriptor,
    build_container_v2,
    parse_index,
    split_v2_tail,
)

#: Inner chunk geometry. Matches the member-shard chunking so the two layouts share the
#: reader's geometry handling and the same read-amplification behaviour.
DEFAULT_CHUNK_LAT: int = 100
DEFAULT_CHUNK_LON: int = 100

#: Compression level used by :func:`encode_aggregate_shard` unless overridden.
#:
#: Level 6 is the measured optimum for this data, and the measurement is worth stating because
#: it is not what the earlier investigation concluded. On real GEFS members at 721x1440, a
#: 34-field aggregate container sized 12.87 MB at level 6, 13.89 MB at 5, **13.54 MB at 7**,
#: 12.75 MB at 9 and 11.79 MB at 12. Level 7 is 5% *worse* than level 6: a zstd level selects
#: a search strategy rather than a monotone effort, and 7 lands in a worse local choice here.
#: The cost of going further is the reason to stop: encoding one container takes 0.84 s at 5,
#: 1.07 s at 6, 2.18 s at 9 and 6.64 s at 12, which over a cycle's 14 variables x 81 leads is
#: ~23 minutes of one core at 6 versus ~2.5 hours at 12 for a further 8%.
#:
#: It is a writer-side knob only -- a zstd frame is self-describing, so the decode path does
#: not depend on it.
DEFAULT_ZSTD_LEVEL: int = 6

#: Suffix of an aggregate shard object, following the shard key grammar.
AGGREGATE_SHARD_SUFFIX: str = "shard.agg"


class AggregateWriterError(ValueError):
    """Raised when aggregate planes cannot be encoded into a container."""


@dataclass(frozen=True)
class AggregateShardLayout:
    """Geometry of one aggregate shard container, derivable from the descriptor.

    Attributes:
        n_fields: Number of statistic planes.
        grid_lat: Full grid latitude extent.
        grid_lon: Full grid longitude extent.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.
    """

    n_fields: int
    grid_lat: int
    grid_lon: int
    chunk_lat: int = DEFAULT_CHUNK_LAT
    chunk_lon: int = DEFAULT_CHUNK_LON

    def __post_init__(self) -> None:
        if self.n_fields < 1:
            raise AggregateWriterError(f"n_fields must be positive, got {self.n_fields}")
        for name in ("grid_lat", "grid_lon", "chunk_lat", "chunk_lon"):
            if getattr(self, name) < 1:
                raise AggregateWriterError(f"{name} must be positive, got {getattr(self, name)}")

    @property
    def lat_chunks(self) -> int:
        """Number of chunks along latitude."""
        return -(-self.grid_lat // self.chunk_lat)

    @property
    def lon_chunks(self) -> int:
        """Number of chunks along longitude."""
        return -(-self.grid_lon // self.chunk_lon)

    @property
    def chunks_per_field(self) -> int:
        """Number of chunks in one field plane."""
        return self.lat_chunks * self.lon_chunks

    @property
    def num_chunks(self) -> int:
        """Total chunk count across every field plane."""
        return self.n_fields * self.chunks_per_field

    def chunk_ordinal(self, field: int, row: int, col: int) -> int:
        """Container chunk ordinal for ``(field, row, col)``, or ``-1`` when off-grid.

        Spatially first: a location's fields are contiguous, so a reader needing the same
        location across every field fetches one range per location rather than one per
        (field, location).
        """
        if not (0 <= field < self.n_fields):
            return -1
        if not (0 <= row < self.lat_chunks and 0 <= col < self.lon_chunks):
            return -1
        return (row * self.lon_chunks + col) * self.n_fields + field

    def locate(self, ordinal: int) -> tuple[int, int, int]:
        """Inverse of :func:`chunk_ordinal`.

        Raises:
            AggregateWriterError: if ``ordinal`` is outside the container.
        """
        if not (0 <= ordinal < self.num_chunks):
            raise AggregateWriterError(
                f"chunk ordinal {ordinal} outside 0..{self.num_chunks - 1}"
            )
        within, field = divmod(ordinal, self.n_fields)
        row, col = divmod(within, self.lon_chunks)
        return field, row, col

    def spatial_group(self, row: int, col: int) -> range:
        """Chunk ordinals covering every field at one location, in field order.

        This is the group a point query needs whole, which is why the layout exists.

        Raises:
            AggregateWriterError: if the location is off-grid.
        """
        if not (0 <= row < self.lat_chunks and 0 <= col < self.lon_chunks):
            raise AggregateWriterError(
                f"location ({row}, {col}) outside the "
                f"{self.lat_chunks}x{self.lon_chunks} chunk grid"
            )
        start = (row * self.lon_chunks + col) * self.n_fields
        return range(start, start + self.n_fields)

    def to_descriptor(
        self,
        *,
        member_count: int = 0,
        encoding_id: int = ENCODING_F32,
        scale: float = 1.0,
    ) -> ShardDescriptor:
        """Build the container descriptor for this layout.

        Args:
            member_count: Ensemble members the planes were computed from. Recorded in the
                descriptor because the statistics an aggregate stores do not imply it, and the
                API reports it. ``0`` for a container that is not a member aggregate.
            encoding_id: How the payload encodes each value.
            scale: Value units per stored integer. Must be 1.0 for :data:`ENCODING_F32`.
        """
        return ShardDescriptor(
            encoding_id=encoding_id,
            scale=scale,
            chunk_lat=self.chunk_lat,
            chunk_lon=self.chunk_lon,
            grid_lat=self.grid_lat,
            grid_lon=self.grid_lon,
            num_chunks=self.num_chunks,
            index_byte_size=self.num_chunks * INDEX_ENTRY_SIZE,
            member_count=member_count,
        )


def layout_from_descriptor(descriptor: ShardDescriptor) -> AggregateShardLayout:
    """Recover the layout a reader needs from a container descriptor alone.

    Derived rather than passed in: a reader opening a store it did not write must not assume
    the field count or the grid, which is exactly what the descriptor exists to remove.
    """
    lat_chunks, lon_chunks = descriptor.expected_shape()
    chunks_per_field = lat_chunks * lon_chunks
    if chunks_per_field == 0:
        raise AggregateWriterError("descriptor geometry yields zero chunks per field")
    if descriptor.num_chunks % chunks_per_field:
        raise AggregateWriterError(
            f"descriptor num_chunks {descriptor.num_chunks} is not a multiple of "
            f"{chunks_per_field} chunks per field"
        )
    return AggregateShardLayout(
        n_fields=descriptor.num_chunks // chunks_per_field,
        grid_lat=descriptor.grid_lat,
        grid_lon=descriptor.grid_lon,
        chunk_lat=descriptor.chunk_lat,
        chunk_lon=descriptor.chunk_lon,
    )


def _chunk_buffer(
    plane: npt.NDArray[np.generic],
    row: int,
    col: int,
    layout: AggregateShardLayout,
) -> npt.NDArray[np.generic]:
    """Extract one padded chunk buffer from a plane, in the plane's own dtype.

    Edge chunks are padded to the full chunk extent, matching the member-shard writer, so every
    chunk in a container has the same byte length before compression. The pad value follows the
    dtype: NaN for float planes and :data:`NAN_SENTINEL` for fixed-point ones, since a stored
    int16 cannot hold NaN and a zero pad would decode as a real value.
    """
    if plane.dtype == np.int16:
        buffer = np.full((layout.chunk_lat, layout.chunk_lon), NAN_SENTINEL, dtype=np.int16)
    else:
        buffer = np.full((layout.chunk_lat, layout.chunk_lon), np.nan, dtype=np.float32)
    r0, c0 = row * layout.chunk_lat, col * layout.chunk_lon
    r1 = min(r0 + layout.chunk_lat, plane.shape[0])
    c1 = min(c0 + layout.chunk_lon, plane.shape[1])
    if r1 > r0 and c1 > c0:
        buffer[: r1 - r0, : c1 - c0] = plane[r0:r1, c0:c1]
    return buffer


def encode_aggregate_shard(
    planes: Sequence[npt.NDArray[np.float32]],
    layout: AggregateShardLayout,
    *,
    level: int = DEFAULT_ZSTD_LEVEL,
    member_count: int = 0,
    field_scales: tuple[float, ...] | None = None,
) -> bytes:
    """Encode statistic planes into a single ``sharded_v2`` container.

    Args:
        planes: One ``(lat, lon)`` float32 plane per field, in storage order.
        layout: Geometry; ``n_fields`` must equal ``len(planes)``.
        level: Zstd level for the inner chunks.
        member_count: Ensemble members the planes were computed from, recorded in the
            descriptor. ``0`` for a container that is not a member aggregate.
        field_scales: Fixed-point step per field. When given, each plane is quantised to int16
            at its own step before compression and the container declares the ``i16`` encoding
            with the per-field scale marker; when ``None`` the planes are stored as float32.
            Per field rather than per container because one aggregate holds fields of different
            magnitudes -- a 314 K mean at a bin probability's 0.001 step would need 314000 and
            overflow int16 -- and quantising is not optional for size: measured on real GEFS
            members, float32 17.73 MB becomes 13.54 MB as int16 at the spec's scales.

    Returns:
        The container bytes: payload, index, descriptor, trailer.

    Raises:
        AggregateWriterError: if the plane count or shapes disagree with the layout, or a
            field would exceed its scale's range. A field that clips is silently wrong *and*
            compresses better, so it is refused rather than clamped.
    """
    if len(planes) != layout.n_fields:
        raise AggregateWriterError(
            f"{len(planes)} planes but layout declares {layout.n_fields} fields"
        )
    for field, plane in enumerate(planes):
        if plane.shape != (layout.grid_lat, layout.grid_lon):
            raise AggregateWriterError(
                f"field {field} has shape {plane.shape}, expected "
                f"{(layout.grid_lat, layout.grid_lon)}"
            )

    if field_scales is not None:
        if len(field_scales) != len(planes):
            raise AggregateWriterError(
                f"{len(field_scales)} scales for {len(planes)} fields"
            )
        try:
            codes = quantise_planes(planes, field_scales)
        except AggregateError as exc:
            raise AggregateWriterError(str(exc)) from exc
        return _assemble(
            codes, layout, level=level, member_count=member_count,
            encoding_id=ENCODING_I16, scale=PER_FIELD_SCALE,
        )

    return _assemble(
        planes, layout, level=level, member_count=member_count,
        encoding_id=ENCODING_F32, scale=1.0,
    )


def quantise_planes(
    planes: Sequence[npt.NDArray[np.float32]],
    field_scales: tuple[float, ...],
) -> list[npt.NDArray[np.int16]]:
    """Quantise each plane at its own step, refusing any that would clip.

    Delegates to :func:`domain.aggregate.quantise_field` for the arithmetic so the writer and
    the domain's own round-trip tests agree by construction rather than by inspection, and adds
    the range check that ``encode_aggregate`` performs for a spec-driven call.
    """
    if len(planes) != len(field_scales):
        raise AggregateError(f"{len(planes)} fields but {len(field_scales)} scales")
    for index, (plane, scale) in enumerate(zip(planes, field_scales, strict=True)):
        finite = np.isfinite(plane)
        if finite.any() and np.any(np.abs(plane[finite] / scale) > np.iinfo(np.int16).max):
            raise AggregateError(
                f"field {index} exceeds its scale {scale} and would clip; widen the scale"
            )
    return [quantise_field(plane, scale) for plane, scale in zip(planes, field_scales, strict=True)]


def _assemble(
    planes: Sequence[npt.NDArray[np.generic]],
    layout: AggregateShardLayout,
    *,
    level: int,
    member_count: int,
    encoding_id: int,
    scale: float,
) -> bytes:
    """Compress and index a set of equally-shaped planes into a container.

    Shared by the float and fixed-point paths: the only difference between them is the buffer
    dtype and what the descriptor declares, so the chunking, the spatial-major emit order and
    the index assembly exist once.
    """
    compressor = Zstd(level=level)

    # Emit in spatial-major order so the payload matches chunk_ordinal: one location's fields
    # adjacent. Iterating field-major here would silently produce a container whose index
    # disagrees with its own geometry.
    payloads: list[bytes] = []
    for row in range(layout.lat_chunks):
        for col in range(layout.lon_chunks):
            for plane in planes:
                buffer = _chunk_buffer(plane, row, col, layout)
                payloads.append(compressor.encode(buffer.tobytes(order="C")))

    return build_container_v2(
        payloads,
        descriptor=layout.to_descriptor(
            member_count=member_count, encoding_id=encoding_id, scale=scale
        ),
    )


#: A decoder used only to call ``decode``. A zstd frame is self-describing, so the decode
#: path does not depend on the level the writer chose and no level has to be negotiated
#: between writer and reader.
_DECODER = Zstd()


def decode_aggregate_chunk(
    container: bytes,
    ordinal: int,
    *,
    field_scales: tuple[float, ...] | None = None,
) -> npt.NDArray[np.float32]:
    """Decode one chunk from an aggregate container by its ordinal, as float32.

    Applies the container's declared encoding. A fixed-point payload needs the per-field steps,
    which the descriptor cannot carry (one field cannot describe an aggregate whose fields
    differ: a 314 K mean needs 0.01 while a bin probability wants 0.001), so ``field_scales``
    supplies the stored-field table. Both tiers derive it from the variable -- the writer from
    the spec it was handed, the reader from ``domain.field_layout`` -- so it is passed in rather
    than guessed.

    Args:
        container: The full container bytes.
        ordinal: Row-major chunk ordinal across all field planes.
        field_scales: Step per stored field, in storage order (the member-count field first).
            Required for a fixed-point payload.

    Raises:
        AggregateWriterError: if the container is malformed, the ordinal is off-grid, or a
            fixed-point payload was handed no usable table. The last is a refusal rather than a
            default: there is no step a reader could safely assume.
    """
    if len(container) < TRAILER_SIZE + DESCRIPTOR_SIZE:
        raise AggregateWriterError(f"container too short: {len(container)} bytes")
    num_chunks, index_byte_size, magic = struct.unpack("<III", container[-TRAILER_SIZE:])
    if magic != SHARD_V2_MAGIC:
        raise AggregateWriterError(
            f"not a sharded_v2 container (magic 0x{magic:08x})"
        )
    tail_start = len(container) - (index_byte_size + DESCRIPTOR_SIZE + TRAILER_SIZE)
    if tail_start < 0:
        raise AggregateWriterError("declared index size exceeds the container length")
    index_bytes, descriptor = split_v2_tail(container[tail_start:], num_chunks)
    layout = layout_from_descriptor(descriptor)

    # Which field this chunk belongs to is what selects the scale, and the layout answers it.
    field, _row, _col = layout.locate(ordinal)
    entries = parse_index(index_bytes, num_chunks)
    offset, length = entries[ordinal]
    if length == 0:
        return np.full((layout.chunk_lat, layout.chunk_lon), np.nan, dtype=np.float32)

    raw = _DECODER.decode(container[offset : offset + length])
    if descriptor.encoding_id == ENCODING_F32:
        return (
            np.frombuffer(raw, dtype=np.float32)
            .reshape(layout.chunk_lat, layout.chunk_lon)
            .copy()
        )
    if descriptor.encoding_id != ENCODING_I16:  # pragma: no cover - parse rejects others
        raise AggregateWriterError(
            f"container declares unknown encoding {descriptor.encoding_id}"
        )
    scale = descriptor.scale
    if scale == PER_FIELD_SCALE:
        if field_scales is None or len(field_scales) != layout.n_fields:
            raise AggregateWriterError(
                "this container stores fixed-point codes at per-field scales, but "
                f"{'no' if field_scales is None else len(field_scales)} scales were supplied "
                f"for {layout.n_fields} fields; there is no step a reader could assume"
            )
        scale = float(field_scales[field])
    codes = np.frombuffer(raw, dtype=np.int16).reshape(
        layout.chunk_lat, layout.chunk_lon
    )
    return dequantise_field(codes.copy(), scale)


def chunk_ordinals_for_field(
    layout: AggregateShardLayout, field: int
) -> list[int]:
    """Chunk ordinals covering one field plane, in row-major order.

    Scattered under the spatial-major layout (the stride is ``n_fields``), so this returns a
    list rather than a range. A reader that wants a whole plane is better served by walking
    the spatial groups, but this exists for callers that inspect a single field.

    Raises:
        AggregateWriterError: if ``field`` is outside the layout.
    """
    if not (0 <= field < layout.n_fields):
        raise AggregateWriterError(
            f"field {field} outside 0..{layout.n_fields - 1}"
        )
    return [
        layout.chunk_ordinal(field, row, col)
        for row in range(layout.lat_chunks)
        for col in range(layout.lon_chunks)
    ]


def layout_for_spec(
    spec: AggregateSpec,
    *,
    grid_lat: int,
    grid_lon: int,
    chunk_lat: int = DEFAULT_CHUNK_LAT,
    chunk_lon: int = DEFAULT_CHUNK_LON,
    n_fields: int | None = None,
) -> AggregateShardLayout:
    """Build the layout for a given aggregate spec and grid.

    ``n_fields`` overrides the spec's own field count, which a caller needs when the container
    also carries fields the spec does not know about -- the per-cell member count leads every
    stored field vector and is not part of an ``AggregateSpec``, since a spec describes a
    distribution rather than how it is packaged.
    """
    return AggregateShardLayout(
        n_fields=spec.n_fields if n_fields is None else n_fields,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        chunk_lat=chunk_lat,
        chunk_lon=chunk_lon,
    )


def aggregate_store_relative_key(variable_code: str, lead_time_hours: int) -> str:
    """Store-relative key of an aggregate shard object.

    Shaped like the member keys so the lifecycle's key grammar and the store inventory need
    no special case; the kind token is ``agg``.
    """
    return f"{variable_code}/shard.agg_L{lead_time_hours:04d}.shard"


def field_metadata(
    spec: AggregateSpec,
    layout: AggregateShardLayout,
) -> Mapping[str, object]:
    """Reader-facing description of an aggregate shard's contents.

    Persisted alongside the container so a reader can interpret it without re-deriving the
    spec, and so a spec change is visible in the store rather than implied by the writer.

    Only the parameters that apply to the spec's kind are recorded. A bin parameter on a
    quantile shard would be a field the reader could mistake for meaningful.
    """
    metadata: dict[str, object] = {
        "kind": spec.kind,
        "n_fields": spec.n_fields,
        "field_names": list(spec.field_names),
        "field_scales": list(spec.field_scales),
        "chunk_lat": layout.chunk_lat,
        "chunk_lon": layout.chunk_lon,
        "grid_lat": layout.grid_lat,
        "grid_lon": layout.grid_lon,
        "chunks_per_field": layout.chunks_per_field,
        "container_format": "sharded_v2",
    }
    if spec.kind == KIND_MEAN_STD_BINS:
        metadata["n_bins"] = spec.n_bins
        metadata["sigma_range"] = spec.sigma_range
    else:
        metadata["levels"] = list(spec.levels)
    return metadata
