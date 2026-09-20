"""Whether a variable's aggregate container exists and can be read back.

The predicate in :mod:`domain.supersession` decides whether members may be deleted; this module
supplies the one fact it cannot compute itself, because it needs the store. It answers a single
question per ``(variable, lead)``: **is there a readable aggregate container here, and is it this
variable's?**

Why the checks are what they are
--------------------------------
**A partial read is not evidence.** The container is ``sharded_v2``, whose descriptor and trailer
sit in its last 52 bytes, so a tail block is enough to decide whether one exists and what it
holds. A container whose trailer does not parse, or whose encoding id this reader does not know,
or whose geometry does not divide into whole field planes, is **not** evidence: it is either
another format version or a damaged object, and a reader given it would fall back to the members.
Deleting the members in either case would leave a store with neither representation.

**The field count is checked against the variable's declared layout.** A container whose plane
count disagrees with ``domain.field_layout.aggregate_fields_for`` was built from a different
layout, which is the silent mis-decode the layout module exists to prevent. It is refused here for
the same reason the API's reader refuses it: the sizes have to agree, or the bytes are not this
variable's statistics.

**Every failure is a "no", and one is recorded as a warning.** The caller is a bulk pass over a
cycle's units, so a variable whose aggregate is missing must not stop the pass; but a missing
aggregate on a variable the classification says has one is worth a log line, since it means the
switch would have deleted members that nothing replaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from domain.field_layout import FieldLayoutError, aggregate_fields_for
from domain.shard_format import (
    DESCRIPTOR_SIZE,
    ENCODING_F32,
    ENCODING_I16,
    PER_FIELD_SCALE,
    TRAILER_SIZE,
    ShardFormatError,
    container_member_count,
    container_tail_size,
    parse_trailer,
    split_v2_tail,
)

from ingestion.core.aggregate_writer import aggregate_store_relative_key
from ingestion.core.store_io import StoreAccessError, StoreIO, StoreRef

logger = logging.getLogger(__name__)

#: Encodings this probe treats as a readable aggregate. A container naming another id is a format
#: this reader does not know how to hand to a serving path, so it is not evidence that the
#: members have a replacement.
KNOWN_ENCODINGS: frozenset[int] = frozenset({ENCODING_F32, ENCODING_I16})

#: Bytes the tail probe reads: enough for the trailer and the fixed-size descriptor.
TAIL_PROBE_BYTES: int = TRAILER_SIZE + DESCRIPTOR_SIZE


@dataclass(frozen=True)
class AggregateEvidence:
    """What the store says about one ``(variable, lead)``'s aggregate container.

    Attributes:
        variable: Variable code asked about.
        lead_time_hours: Lead asked about.
        present: Whether an object exists at the aggregate key at all.
        readable: Whether it parses as a container this platform can serve from. False
            distinguishes "no aggregate" from "an aggregate I cannot interpret", which the
            caller reports differently: the first is the normal pre-aggregation state, the
            second is a store whose write path is ahead of its read path.
        member_count: Container-wide member total the descriptor records, or ``None``.
        detail: Why the answer is what it is, for a log line or a queue row's ``last_error``.
    """

    variable: str
    lead_time_hours: int
    present: bool
    readable: bool
    member_count: int | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether this container is evidence that the members have a replacement."""
        return self.present and self.readable


def _interprets_tail(tail: bytes, variable: str) -> tuple[bool, int | None, str]:
    """Whether a container's tail parses as this variable's aggregate.

    The tail is the container's last :func:`container_tail_size` bytes -- index, descriptor and
    trailer -- so everything the answer needs is in it and the payload is never read.

    Returns ``(readable, member_count, detail)``.
    """
    if len(tail) < TAIL_PROBE_BYTES:
        return False, None, f"object is {len(tail)} bytes, shorter than a container tail"
    try:
        trailer = parse_trailer(tail[-TRAILER_SIZE:])
    except ShardFormatError as exc:
        return False, None, f"trailer does not parse: {exc}"
    if not trailer.is_v2:
        return False, None, "not a sharded_v2 container"
    tail_size = container_tail_size(trailer.num_chunks)
    if len(tail) < tail_size:
        return False, None, "container is shorter than its own index declares"
    try:
        _index_bytes, descriptor = split_v2_tail(tail[-tail_size:], trailer.num_chunks)
    except ShardFormatError as exc:
        return False, None, f"tail does not parse: {exc}"
    if descriptor.encoding_id not in KNOWN_ENCODINGS:
        return False, None, f"unknown encoding id {descriptor.encoding_id}"
    if descriptor.encoding_id == ENCODING_I16 and descriptor.scale != PER_FIELD_SCALE:
        # A fixed-point container that does not declare per-field scales cannot be dequantised:
        # the reader would have to assume one step for fields of different magnitudes, and the
        # assumption is exactly what PER_FIELD_SCALE exists to forbid.
        return False, None, "fixed-point container without the per-field scale marker"
    try:
        layout = aggregate_fields_for(variable)
    except FieldLayoutError as exc:
        return False, None, f"no declared field layout: {exc}"
    lat_chunks, lon_chunks = descriptor.expected_shape()
    chunks_per_field = lat_chunks * lon_chunks
    if chunks_per_field <= 0 or descriptor.num_chunks % chunks_per_field:
        return False, None, "chunk count does not divide into whole field planes"
    if descriptor.num_chunks // chunks_per_field != layout.n_fields:
        return (
            False,
            None,
            f"field count {descriptor.num_chunks // chunks_per_field} does not match "
            f"{layout.n_fields} declared for {variable!r}",
        )
    return True, container_member_count(descriptor), ""


def probe_aggregate(
    store: StoreRef, variable: str, lead_time_hours: int
) -> AggregateEvidence:
    """Read one ``(variable, lead)``'s aggregate container and report what it is.

    **Two reads of the tail, never the payload.** The container's trailer declares how many chunks
    it has, which is what sizes the index between the payload and the descriptor; so the first
    read takes the last :data:`TAIL_PROBE_BYTES`, and the second takes the last
    ``container_tail_size(num_chunks)``. On S3 that is two range GETs against an object that is
    tens of megabytes, and the second is skipped entirely once the first shows the object is too
    short to be a container at all. Measured locally on a real 11.5 MB aggregate: 0.13 ms for the
    tail reads against 2.61 ms for the whole object, and on S3 the difference is the transfer.

    A store whose container the caller has already read is one this probe reads again; that is the
    price of keeping the predicate independent of any particular reader, and the reclamation
    planner probes each ``(variable, lead)`` once per pass.
    """
    key = aggregate_store_relative_key(variable, lead_time_hours)
    io = StoreIO(store)
    try:
        head = io.read_range(key, start=-TAIL_PROBE_BYTES)
    except StoreAccessError as exc:
        return AggregateEvidence(
            variable, lead_time_hours, False, False, None, f"cannot read {key}: {exc}"
        )
    if len(head) < TAIL_PROBE_BYTES:
        return AggregateEvidence(
            variable,
            lead_time_hours,
            True,
            False,
            None,
            f"object is {len(head)} bytes, shorter than a container tail",
        )
    try:
        trailer = parse_trailer(head[-TRAILER_SIZE:])
    except ShardFormatError as exc:
        # The trailer is the first thing read, so its failure is reported from here rather than
        # paying for the longer read that the parse would have asked for.
        return AggregateEvidence(
            variable, lead_time_hours, True, False, None, f"trailer does not parse: {exc}"
        )
    if not trailer.is_v2:
        return AggregateEvidence(
            variable, lead_time_hours, True, False, None, "not a sharded_v2 container"
        )
    tail_size = container_tail_size(trailer.num_chunks)
    try:
        tail = (
            head
            if tail_size <= TAIL_PROBE_BYTES
            else io.read_range(key, start=-tail_size)
        )
    except StoreAccessError as exc:
        return AggregateEvidence(
            variable, lead_time_hours, True, False, None, f"cannot read {key}: {exc}"
        )
    readable, member_count, detail = _interprets_tail(tail, variable)
    return AggregateEvidence(
        variable, lead_time_hours, True, readable, member_count, detail
    )


def aggregate_supersedes_members(
    store: StoreRef, variable: str, lead_time_hours: int
) -> bool:
    """Whether a readable aggregate container for this pair exists.

    The convenience form for callers that only need the predicate. A missing or unreadable
    container is a plain ``False``; use :func:`probe_aggregate` when the reason matters.
    """
    return probe_aggregate(store, variable, lead_time_hours).ok


__all__ = [
    "KNOWN_ENCODINGS",
    "TAIL_PROBE_BYTES",
    "AggregateEvidence",
    "aggregate_supersedes_members",
    "probe_aggregate",
]
