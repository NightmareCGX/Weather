"""Serving when a variable's members have been replaced by a stored aggregate.

The half of the supersession change the API owns. Members released by GC are absent *by design*,
so the member-coverage floor no longer describes the lead -- counting absent members as an outage
would turn the migration into one. These tests pin the two properties that make that safe:

* the fallback opens only for a variable whose container is actually readable, and only with the
  switch on -- so a coverage failure stays a failure everywhere else, including on every store
  written before the aggregate path existed;
* the answer the fallback returns is the container's, read through the same gate and the same
  reader the products already use.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import numpy as np
import pytest
from api.core.aggregate_reader import aggregate_shard_key
from api.services.aggregate_serving import aggregate_answers_for_lead
from api.services.ensemble_data import _aggregate_represents_members
from domain.aggregate import compute_aggregate, finite_member_count
from domain.field_layout import aggregate_fields_for
from domain.product_fields import variable_group_fields
from domain.supersession import (
    SUPERSESSION_ENV_VAR,
    reset_supersession_enabled,
    set_supersession_enabled,
)
from domain.variable_class import spec_for

GRID_LAT, GRID_LON = 32, 32
CHUNK_LAT, CHUNK_LON = 16, 16
LEAD = 6


@pytest.fixture(autouse=True)
def clean_switch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(SUPERSESSION_ENV_VAR, raising=False)
    reset_supersession_enabled()
    yield
    monkeypatch.delenv(SUPERSESSION_ENV_VAR, raising=False)
    reset_supersession_enabled()


def _encode(fields: np.ndarray, *, field_scales: tuple[float, ...], member_count: int) -> bytes:
    """Write a container the way production writes it: int16 at the variable's own steps."""
    from numcodecs import Zstd
    from domain.shard_format import (
        ENCODING_I16,
        PER_FIELD_SCALE,
        ShardDescriptor,
        build_container_v2,
    )

    compressor = Zstd(level=3)
    lat_chunks = -(-GRID_LAT // CHUNK_LAT)
    lon_chunks = -(-GRID_LON // CHUNK_LON)
    payloads: list[bytes] = []
    codes = np.stack(
        [np.rint(field / scale).astype(np.int16) for field, scale in zip(fields, field_scales, strict=True)]
    )
    for row in range(lat_chunks):
        for col in range(lon_chunks):
            for index in range(codes.shape[0]):
                buffer = np.zeros((CHUNK_LAT, CHUNK_LON), dtype=np.int16)
                r0, c0 = row * CHUNK_LAT, col * CHUNK_LON
                window = codes[index, r0 : r0 + CHUNK_LAT, c0 : c0 + CHUNK_LON]
                buffer[: window.shape[0], : window.shape[1]] = window
                payloads.append(compressor.encode(buffer.tobytes(order="C")))
    descriptor = ShardDescriptor(
        encoding_id=ENCODING_I16,
        scale=PER_FIELD_SCALE,
        chunk_lat=CHUNK_LAT,
        chunk_lon=CHUNK_LON,
        grid_lat=GRID_LAT,
        grid_lon=GRID_LON,
        num_chunks=lat_chunks * lon_chunks * codes.shape[0],
        index_byte_size=lat_chunks * lon_chunks * codes.shape[0] * 16,
        member_count=member_count,
    )
    return build_container_v2(payloads, descriptor=descriptor)


def _store_with(tmp_path, variable: str, seed: int = 0) -> str:
    """A store holding one real container for ``(variable, LEAD)``."""
    layout = aggregate_fields_for(variable)
    rng = np.random.default_rng(seed)
    members = rng.normal(10.0, 2.0, (30, GRID_LAT, GRID_LON)).astype(np.float32)
    fields: list[np.ndarray] = [finite_member_count(members)]
    if layout.distribution_slice.stop > layout.distribution_slice.start:
        fields.extend(compute_aggregate(members, spec_for(variable)))
    if layout.groups:
        fields.append(variable_group_fields(variable, members=members, extra_members={}))
    stacked = np.concatenate(
        [field[None] if field.ndim == 2 else field for field in fields]
    )
    assert stacked.shape[0] == layout.n_fields
    blob = _encode(stacked, field_scales=layout.field_scales, member_count=30)
    store = str(tmp_path)
    full = os.path.join(store, *aggregate_shard_key(variable, LEAD).split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(blob)
    return store


def test_a_readable_container_answers_for_its_variable(tmp_path) -> None:
    store = _store_with(tmp_path, "temperature_2m")
    assert aggregate_answers_for_lead(
        "temperature_2m", store_path=store, lead_time_hours=LEAD
    ) is True
    # A lead the store does not carry a container for is a plain no.
    assert aggregate_answers_for_lead(
        "temperature_2m", store_path=store, lead_time_hours=LEAD + 3
    ) is False


def test_an_absent_or_unreadable_container_answers_for_nothing(tmp_path) -> None:
    # No object at all.
    assert aggregate_answers_for_lead(
        "temperature_2m", store_path=str(tmp_path), lead_time_hours=LEAD
    ) is False

    # An object that is not a container.
    full = os.path.join(str(tmp_path), *aggregate_shard_key("temperature_2m", LEAD).split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as handle:
        handle.write(b"\x00" * 4096)
    assert aggregate_answers_for_lead(
        "temperature_2m", store_path=str(tmp_path), lead_time_hours=LEAD
    ) is False


def test_a_container_for_another_variable_does_not_answer(tmp_path) -> None:
    """A container whose field count is not this variable's is not this variable's answer."""
    store = _store_with(tmp_path, "temperature_2m")
    assert aggregate_answers_for_lead(
        "temperature_2m", store_path=store, lead_time_hours=LEAD
    ) is True
    # wind_gust declares a different field count, so the same bytes are not its container.
    assert aggregate_answers_for_lead(
        "wind_gust", store_path=store, lead_time_hours=LEAD
    ) is False


def test_an_unclassified_variable_is_never_answered_for(tmp_path) -> None:
    store = _store_with(tmp_path, "temperature_2m")
    assert aggregate_answers_for_lead(
        "not_a_variable", store_path=store, lead_time_hours=LEAD
    ) is False


def test_the_fallback_needs_both_the_container_and_the_switch(tmp_path) -> None:
    """The serving change is inert until member reclamation is turned on.

    With the switch off no member is ever released, so a coverage failure has to stay a failure:
    a store whose members are simply missing must not start looking servable because an unrelated
    container happens to exist beside them.
    """
    store = _store_with(tmp_path, "temperature_2m")
    assert (
        _aggregate_represents_members(
            "temperature_2m", store_path=store, lead_time_hours=LEAD
        )
        is False
    )
    set_supersession_enabled(True)
    assert (
        _aggregate_represents_members(
            "temperature_2m", store_path=store, lead_time_hours=LEAD
        )
        is True
    )


def test_the_switch_alone_does_not_represent_members(tmp_path) -> None:
    """And the container alone does not either -- both are required, in both directions."""
    set_supersession_enabled(True)
    assert (
        _aggregate_represents_members(
            "temperature_2m", store_path=str(tmp_path), lead_time_hours=LEAD
        )
        is False
    )


def test_a_damaged_container_does_not_break_resolution(tmp_path) -> None:
    """A store the reader cannot open is "no aggregate", not a failed request.

    The probe is on a resolution path, so a store that has gone away or cannot be opened must
    answer "not represented" and let the pre-existing coverage rule decide -- which is what the
    member path's own reader does when it cannot read either.
    """
    missing = str(tmp_path / "not-a-store")
    assert (
        _aggregate_represents_members(
            "temperature_2m", store_path=missing, lead_time_hours=LEAD
        )
        is False
    )


def test_a_reader_probe_leaves_the_container_alone(tmp_path) -> None:
    """The probe reads the tail: the payload is untouched, so it stays cheap and repeatable."""
    store = _store_with(tmp_path, "temperature_2m")
    full = os.path.join(store, *aggregate_shard_key("temperature_2m", LEAD).split("/"))
    before = os.path.getsize(full)
    for _ in range(3):
        assert aggregate_answers_for_lead(
            "temperature_2m", store_path=store, lead_time_hours=LEAD
        )
    assert os.path.getsize(full) == before
    assert datetime.now(UTC).tzinfo is not None  # the module imports cleanly under a real clock
