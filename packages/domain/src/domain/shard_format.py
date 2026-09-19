"""Binary container format for sharded forecast stores (shared by writer and reader).

Two container generations exist:

``sharded_v1``
    ``[payload][index][trailer]`` where the trailer is
    ``<uint32 num_chunks><uint32 index_byte_size><uint32 SHARD_V1_MAGIC>``.
    It is **not self-describing**: dtype, inner chunk shape, grid size, compressor and
    value scale are all implicit and were hardcoded independently in the writer and in
    every reader. A reader that assumed the wrong chunk count silently mis-sliced the
    index instead of failing, so the format was impossible to evolve safely.

``sharded_v2``
    ``[payload][index][descriptor][trailer]`` where the descriptor is 40 bytes carrying
    magic, format version, encoding id, value scale and the chunk/grid geometry, and the
    trailer repeats ``num_chunks``/``index_byte_size`` so a reader can decide whether an
    object is a container and how large its index is from the last 12 bytes alone.

    Putting the descriptor in the *tail* rather than a leading header is deliberate: the
    reader's hot path already does a tail range-GET to reach the index, so a single 52-byte
    tail read now yields both the trailer and the descriptor. A leading header would have
    cost a second round trip on every cold shard.

This module is pure: no I/O, no compressors. It defines the bytes and validates them.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

#: Container magics. Distinct per generation so a reader can tell them apart from the
#: last four bytes of an object, and so a v1 reader cannot silently accept a v2 object.
SHARD_V1_MAGIC: int = 0x53484152  # 'SHAR'
SHARD_V2_MAGIC: int = 0x53485632  # 'SHV2'

#: Index entry width: a little-endian ``(uint64 offset, uint64 length)`` pair.
INDEX_ENTRY_SIZE: int = 16

#: Trailer width: ``num_chunks``, ``index_byte_size``, ``magic``.
TRAILER_SIZE: int = 12

#: Descriptor width: see :data:`DESCRIPTOR_FMT`.
DESCRIPTOR_SIZE: int = 40

#: Smallest tail read that yields both the trailer and the descriptor.
TAIL_PROBE_SIZE: int = TRAILER_SIZE + DESCRIPTOR_SIZE  # 52

#: Container format version string, recorded in the commit manifest. Derived from
#: FORMAT_VERSION_BYTE below so the manifest and the descriptor byte cannot diverge.
FORMAT_VERSION_V2: str = "sharded_v2"

#: Value encoding identifiers carried in the descriptor.
#:
#: ``f32`` stores IEEE float32 values and declares ``scale = 1.0``.
#:
#: ``i16`` stores signed 16-bit fixed-point codes: value = code * scale. The descriptor's
#: single ``scale`` field cannot describe a multi-field aggregate whose fields carry different
#: steps (a 314 K mean needs 0.01 to fit int16 while a 0-1 bin probability wants 0.001), so
#: ``scale == 0.0`` is the explicit marker for "per field, consult the variable's spec". It is
#: a marker rather than a guess precisely so that a reader without the spec cannot silently
#: decode a mean with a probability's step: there is no value it could assume.
ENCODING_F32: int = 1
ENCODING_I16: int = 2
ENCODING_NAMES: dict[int, str] = {
    ENCODING_F32: "f32",
    ENCODING_I16: "i16",
}

#: Scale an encoding must declare, or ``None`` when any valid scale is accepted.
#:
#: A float container that declares a scale would silently rescale every element on read, so the
#: mismatch must be rejected. A fixed-point container accepts either a uniform step or the
#: per-field marker; a negative step is an error in both directions.
REQUIRED_SCALE_BY_ENCODING: dict[int, float | None] = {
    ENCODING_F32: 1.0,
    ENCODING_I16: None,
}

#: The ``i16`` scale meaning "each field's step comes from the variable's spec".
PER_FIELD_SCALE: float = 0.0

#: Scale a float container declares. Retained as the name callers already use for "no scaling".
REQUIRED_SCALE: float = 1.0

#: Serialized descriptor layout. All fields little-endian.
#:
#: ======  ===  ========================================
#: offset  fmt  field
#: ======  ===  ========================================
#: 0       4s   magic (``SHV2``)
#: 4       1B   format version (2)
#: 5       1B   encoding id
#: 6       2H   flags (reserved, must be 0)
#: 8       4f   scale (value units per stored integer; 1.0 for float encodings)
#: 12      2H   inner chunk latitude extent
#: 14      2H   inner chunk longitude extent
#: 16      4I   grid latitude extent
#: 20      4I   grid longitude extent
#: 24      4I   num_chunks
#: 28      4I   index_byte_size
#: 32      4I   member_count (0 when the container is not a member aggregate)
#: 36      4s   reserved (must be zero)
#: ======  ===  ========================================
#:
#: ``member_count`` was carved out of the reserved region rather than appended, so the
#: descriptor is still :data:`DESCRIPTOR_SIZE` bytes and every existing offset still holds.
#: It is stored rather than derived because an aggregate collapses the members it was computed
#: from, and the count the API reports cannot be recovered afterwards.
#:
#: A reader that predates this field rejects a non-zero reserved region outright, so an older
#: reader refuses a newer container instead of misreading it -- which is the safe direction,
#: and why the format version byte did not need to change.
DESCRIPTOR_FMT: str = "<4sBBHfHHIIIII4s"
_TRAILER_FMT: str = "<III"
_RESERVED = b"\x00" * 4

assert struct.calcsize(DESCRIPTOR_FMT) == DESCRIPTOR_SIZE, "descriptor layout drifted"
assert struct.calcsize(_TRAILER_FMT) == TRAILER_SIZE, "trailer layout drifted"


class ShardFormatError(ValueError):
    """Raised when a container's bytes do not satisfy the format contract.

    Raised instead of returning a degraded value: every failure this covers previously
    manifested as a silently mis-sliced index or a wrongly-shaped array.
    """


@dataclass(frozen=True)
class ShardDescriptor:
    """The self-describing part of a ``sharded_v2`` container.

    Attributes:
        encoding_id: Value encoding of the inner chunk payload.
        scale: Value units per stored integer step. ``1.0`` for float encodings.
        chunk_lat: Inner chunk latitude extent.
        chunk_lon: Inner chunk longitude extent.
        grid_lat: Full grid latitude extent.
        grid_lon: Full grid longitude extent.
        num_chunks: Number of inner chunks in the index.
        index_byte_size: ``num_chunks * INDEX_ENTRY_SIZE``.
        member_count: Ensemble members the aggregate was computed from; ``0`` for a container
            that is not a member aggregate (a member shard, a flag fraction, an unimplemented
            case).
        flags: Reserved; must be 0.
    """

    encoding_id: int
    scale: float
    chunk_lat: int
    chunk_lon: int
    grid_lat: int
    grid_lon: int
    num_chunks: int
    index_byte_size: int
    member_count: int = 0
    flags: int = 0

    @property
    def encoding_name(self) -> str:
        """Human-readable encoding identifier."""
        return ENCODING_NAMES.get(self.encoding_id, f"unknown({self.encoding_id})")

    def expected_shape(self) -> tuple[int, int]:
        """The ``(lat_chunks, lon_chunks)`` grid implied by geometry and chunk extents."""
        lat_chunks = -(-self.grid_lat // self.chunk_lat) if self.chunk_lat else 0
        lon_chunks = -(-self.grid_lon // self.chunk_lon) if self.chunk_lon else 0
        return lat_chunks, lon_chunks

    def chunk_index(self, chunk_row: int, chunk_col: int) -> int:
        """Row-major chunk ordinal, or ``-1`` when the position is off-grid."""
        lat_chunks, lon_chunks = self.expected_shape()
        if not (0 <= chunk_row < lat_chunks and 0 <= chunk_col < lon_chunks):
            return -1
        return chunk_row * lon_chunks + chunk_col


def container_member_count(descriptor: ShardDescriptor) -> int | None:
    """How many members this container was aggregated from, or ``None`` if it records none.

    This is the **container-wide** total, which is what the descriptor can hold. The count the
    API reports is per point -- one member can be missing at one cell and present at its
    neighbour -- and that one is stored as field 0 of the field vector (see
    ``domain.field_layout``), so a reader wanting the per-point answer reads the field, not this.

    The total is worth keeping anyway: it says whether a container was built from the complete
    member set, which is the first thing to check when a statistic looks wrong.
    """
    return descriptor.member_count if descriptor.member_count > 0 else None


#: Numeric format version recorded in the descriptor, and the only one parse_descriptor
#: will accept. Bumped only when the descriptor layout itself changes.
FORMAT_VERSION_BYTE: int = 2


def build_descriptor(descriptor: ShardDescriptor) -> bytes:
    """Serialize a descriptor. Returns exactly :data:`DESCRIPTOR_SIZE` bytes."""
    return struct.pack(
        DESCRIPTOR_FMT,
        struct.pack("<I", SHARD_V2_MAGIC),
        FORMAT_VERSION_BYTE,
        descriptor.encoding_id,
        descriptor.flags,
        descriptor.scale,
        descriptor.chunk_lat,
        descriptor.chunk_lon,
        descriptor.grid_lat,
        descriptor.grid_lon,
        descriptor.num_chunks,
        descriptor.index_byte_size,
        descriptor.member_count,
        _RESERVED,
    )


def parse_descriptor(raw: bytes) -> ShardDescriptor:
    """Parse and validate a descriptor.

    Raises:
        ShardFormatError: on short input, a wrong magic, an unknown format version, a
            non-zero reserved field, an unknown encoding id, non-positive geometry, a
            scale inconsistent with the encoding, or an ``index_byte_size`` inconsistent
            with ``num_chunks``.
    """
    if len(raw) < DESCRIPTOR_SIZE:
        raise ShardFormatError(
            f"descriptor truncated: {len(raw)} bytes, need {DESCRIPTOR_SIZE}"
        )
    (
        magic_bytes,
        version,
        encoding_id,
        flags,
        scale,
        chunk_lat,
        chunk_lon,
        grid_lat,
        grid_lon,
        num_chunks,
        index_byte_size,
        member_count,
        reserved,
    ) = struct.unpack(DESCRIPTOR_FMT, raw[:DESCRIPTOR_SIZE])

    magic = struct.unpack("<I", magic_bytes)[0]
    if magic != SHARD_V2_MAGIC:
        raise ShardFormatError(
            f"bad descriptor magic 0x{magic:08x}, expected 0x{SHARD_V2_MAGIC:08x}"
        )
    if version != FORMAT_VERSION_BYTE:
        raise ShardFormatError(
            f"unsupported container format version {version}, expected {FORMAT_VERSION_BYTE}"
        )
    if flags != 0:
        raise ShardFormatError(f"reserved flags must be 0, got {flags}")
    if reserved != _RESERVED:
        raise ShardFormatError("reserved descriptor bytes must be zero")
    if encoding_id not in ENCODING_NAMES:
        raise ShardFormatError(
            f"unknown encoding id {encoding_id}; known: {sorted(ENCODING_NAMES)}"
        )
    if chunk_lat <= 0 or chunk_lon <= 0:
        raise ShardFormatError(
            f"chunk extents must be positive, got {chunk_lat}x{chunk_lon}"
        )
    if grid_lat <= 0 or grid_lon <= 0:
        raise ShardFormatError(f"grid extents must be positive, got {grid_lat}x{grid_lon}")
    expected_index = num_chunks * INDEX_ENTRY_SIZE
    if index_byte_size != expected_index:
        raise ShardFormatError(
            f"index_byte_size {index_byte_size} disagrees with "
            f"num_chunks {num_chunks} * {INDEX_ENTRY_SIZE} = {expected_index}"
        )
    # A container that declares a scale while carrying float values would silently
    # rescale every element on read, so the mismatch must be rejected here -- and a
    # fixed-point container must name a step (uniform, or the per-field marker) rather
    # than leaving a reader to assume one.
    required_scale = REQUIRED_SCALE_BY_ENCODING[encoding_id]
    if required_scale is not None and scale != required_scale:
        raise ShardFormatError(
            f"encoding {ENCODING_NAMES[encoding_id]!r} must declare scale "
            f"{required_scale}, got {scale}"
        )
    if required_scale is None and (
        scale < 0.0 or (scale == 0.0 and encoding_id != ENCODING_I16)
    ):
        raise ShardFormatError(
            f"encoding {ENCODING_NAMES[encoding_id]!r} must declare a non-negative scale "
            f"(0.0 means per field), got {scale}"
        )

    return ShardDescriptor(
        encoding_id=encoding_id,
        scale=float(scale),
        chunk_lat=int(chunk_lat),
        chunk_lon=int(chunk_lon),
        grid_lat=int(grid_lat),
        grid_lon=int(grid_lon),
        num_chunks=int(num_chunks),
        index_byte_size=int(index_byte_size),
        member_count=int(member_count),
        flags=int(flags),
    )


def build_trailer(num_chunks: int, index_byte_size: int) -> bytes:
    """Serialize a trailer. Returns exactly :data:`TRAILER_SIZE` bytes."""
    return struct.pack(_TRAILER_FMT, num_chunks, index_byte_size, SHARD_V2_MAGIC)


@dataclass(frozen=True)
class TrailerInfo:
    """Trailer fields plus which container generation produced them."""

    num_chunks: int
    index_byte_size: int
    magic: int

    @property
    def is_v2(self) -> bool:
        """Whether the object declares itself a ``sharded_v2`` container."""
        return self.magic == SHARD_V2_MAGIC

    @property
    def is_v1(self) -> bool:
        """Whether the object declares itself a ``sharded_v1`` container."""
        return self.magic == SHARD_V1_MAGIC


def parse_trailer(raw: bytes) -> TrailerInfo:
    """Parse a container trailer, accepting either generation.

    Callers that require a specific generation must check :attr:`TrailerInfo.is_v1` /
    :attr:`TrailerInfo.is_v2`; an unrecognized magic is an error rather than an
    assumption, because the previous reader treated the magic as decorative and would
    happily slice a foreign object.

    Raises:
        ShardFormatError: on short input or an unrecognized magic.
    """
    if len(raw) < TRAILER_SIZE:
        raise ShardFormatError(f"trailer truncated: {len(raw)} bytes, need {TRAILER_SIZE}")
    num_chunks, index_byte_size, magic = struct.unpack(_TRAILER_FMT, raw[:TRAILER_SIZE])
    if magic not in (SHARD_V1_MAGIC, SHARD_V2_MAGIC):
        raise ShardFormatError(
            f"unrecognized container magic 0x{magic:08x}; expected "
            f"0x{SHARD_V1_MAGIC:08x} (v1) or 0x{SHARD_V2_MAGIC:08x} (v2)"
        )
    if index_byte_size != num_chunks * INDEX_ENTRY_SIZE:
        raise ShardFormatError(
            f"trailer index_byte_size {index_byte_size} disagrees with "
            f"num_chunks {num_chunks} * {INDEX_ENTRY_SIZE}"
        )
    return TrailerInfo(
        num_chunks=int(num_chunks), index_byte_size=int(index_byte_size), magic=int(magic)
    )


def container_tail_size(num_chunks: int) -> int:
    """Bytes to range-GET to obtain a v2 container's index, descriptor and trailer."""
    return num_chunks * INDEX_ENTRY_SIZE + DESCRIPTOR_SIZE + TRAILER_SIZE


def v1_container_tail_size(num_chunks: int) -> int:
    """Bytes to range-GET to obtain a v1 container's index and trailer."""
    return num_chunks * INDEX_ENTRY_SIZE + TRAILER_SIZE


def split_v2_tail(tail: bytes, num_chunks: int) -> tuple[bytes, ShardDescriptor]:
    """Split a v2 tail block into its index bytes and validated descriptor.

    Args:
        tail: The last :func:`container_tail_size` bytes of the object.
        num_chunks: Chunk count from the trailer, which the caller has already parsed.

    Raises:
        ShardFormatError: if the block is short, the descriptor is invalid, or the
            descriptor's ``num_chunks`` disagrees with the trailer's.
    """
    expected = container_tail_size(num_chunks)
    if len(tail) < expected:
        raise ShardFormatError(
            f"v2 tail truncated: {len(tail)} bytes, need {expected} for {num_chunks} chunks"
        )
    block = tail[-expected:]
    index_bytes = block[: num_chunks * INDEX_ENTRY_SIZE]
    descriptor = parse_descriptor(block[num_chunks * INDEX_ENTRY_SIZE :])
    if descriptor.num_chunks != num_chunks:
        raise ShardFormatError(
            f"descriptor num_chunks {descriptor.num_chunks} disagrees with trailer "
            f"num_chunks {num_chunks}"
        )
    return index_bytes, descriptor


def build_container_v2(
    chunks: list[bytes],
    *,
    descriptor: ShardDescriptor,
) -> bytes:
    """Assemble a ``sharded_v2`` container from already-compressed chunk payloads.

    Empty chunks are recorded as a zero-length entry, exactly as ``sharded_v1`` does, so
    an all-fill shard still costs its index and descriptor but no payload.

    Args:
        chunks: Compressed inner chunk payloads, in row-major chunk order.
        descriptor: Geometry and encoding; ``num_chunks`` must equal ``len(chunks)``.

    Raises:
        ShardFormatError: if ``len(chunks)`` disagrees with the descriptor.
    """
    if len(chunks) != descriptor.num_chunks:
        raise ShardFormatError(
            f"{len(chunks)} chunk payloads but descriptor declares "
            f"{descriptor.num_chunks}"
        )
    if descriptor.index_byte_size != descriptor.num_chunks * INDEX_ENTRY_SIZE:
        raise ShardFormatError(
            f"descriptor index_byte_size {descriptor.index_byte_size} disagrees with "
            f"num_chunks {descriptor.num_chunks} * {INDEX_ENTRY_SIZE}"
        )

    body = bytearray()
    entries: list[tuple[int, int]] = []
    offset = 0
    for payload in chunks:
        if payload:
            entries.append((offset, len(payload)))
            body += payload
            offset += len(payload)
        else:
            entries.append((0, 0))

    body += b"".join(struct.pack("<QQ", off, length) for off, length in entries)
    body += build_descriptor(descriptor)
    body += build_trailer(descriptor.num_chunks, descriptor.index_byte_size)
    return bytes(body)


def parse_index(index_bytes: bytes, num_chunks: int) -> list[tuple[int, int]]:
    """Parse an index block into ``(offset, length)`` pairs.

    Raises:
        ShardFormatError: if the block is shorter than ``num_chunks`` entries.
    """
    expected = num_chunks * INDEX_ENTRY_SIZE
    if len(index_bytes) < expected:
        raise ShardFormatError(
            f"index truncated: {len(index_bytes)} bytes, need {expected} for {num_chunks} chunks"
        )
    return [
        struct.unpack_from("<QQ", index_bytes, i * INDEX_ENTRY_SIZE)
        for i in range(num_chunks)
    ]


def extract_chunk(container: bytes, offset: int, length: int) -> bytes:
    """Return the payload slice for one index entry.

    A zero-length entry (an all-fill chunk) yields ``b""``. An entry that runs past the
    payload region is an error: it means the index and payload disagree, which previously
    produced a short or misaligned decompression input rather than a clear failure.
    """
    if length == 0:
        return b""
    payload_limit = len(container) - DESCRIPTOR_SIZE - TRAILER_SIZE
    if offset + length > payload_limit:
        raise ShardFormatError(
            f"index entry ({offset}, {length}) runs past the payload region "
            f"(limit {payload_limit})"
        )
    return container[offset : offset + length]
