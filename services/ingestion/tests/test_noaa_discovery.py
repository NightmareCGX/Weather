"""Offline tests for NOAA upstream publication discovery (Phase 5B-2).

All tests run against deterministic in-test S3 ListObjectsV2 XML fixtures via
``httpx.MockTransport`` — no live upstream dependency (CI-safe). The fixtures
mirror the real bucket behavior probed in the Phase 5A investigation
(anonymous listing, Key/Size/ETag/LastModified, continuation-token pagination,
1000-key pages).
"""

from __future__ import annotations

from datetime import date, datetime

import httpx
import pytest

from ingestion.core.config import IngestionSettings
from ingestion.providers.noaa.discovery import (
    DiscoveryInvalidResponseError,
    DiscoveryPaginationError,
    DiscoveryUnavailableError,
    gfs_cycle_prefix,
    gefs_cycle_prefix,
    publication_changed,
    snapshot_gfs_cycle,
    snapshot_gefs_cycle,
)

CYCLE_DATE = date(2026, 7, 21)
CYCLE_HOUR = 0
LM = "2026-07-21T03:35:10.000Z"


def _fast_settings(**overrides) -> IngestionSettings:
    """Settings with bounded retries/backoff so error tests stay fast."""
    return IngestionSettings(
        DOWNLOAD_RETRIES=0, RETRY_BACKOFF_SECONDS=0.0, **overrides
    )


def _listing_xml(
    entries: list[tuple[str, int, str] | str], *, truncated: bool, next_token: str | None
) -> bytes:
    """Render one ListObjectsV2 page: entries are (key, size, lm) or bare keys."""

    def _parts(entry: tuple[str, int, str] | str) -> tuple[str, int, str]:
        return (entry, 1, LM) if isinstance(entry, str) else entry

    contents = "".join(
        "<Contents>"
        f"<Key>{key}</Key>"
        f"<LastModified>{lm}</LastModified>"
        "<ETag>&quot;abc123&quot;</ETag>"
        f"<Size>{size}</Size>"
        "<StorageClass>STANDARD</StorageClass>"
        "</Contents>"
        for key, size, lm in (_parts(entry) for entry in entries)
    )
    trunc = "true" if truncated else "false"
    token = f"<NextContinuationToken>{next_token}</NextContinuationToken>" if next_token else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        "<Name>bucket</Name>"
        f"<KeyCount>{len(entries)}</KeyCount>"
        f"<IsTruncated>{trunc}</IsTruncated>"
        f"{token}"
        f"{contents}"
        "</ListBucketResult>"
    ).encode("utf-8")


def _transport(pages: dict[str | None, bytes]) -> httpx.MockTransport:
    """A listing endpoint serving fixed XML bodies per continuation token.

    ``None`` maps to the first page (no continuation token); any other key is
    the token value seen in the request.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("continuation-token")
        body = pages.get(token)
        if body is None:
            return httpx.Response(400, content=b"<Unexpected/>")
        return httpx.Response(200, content=body)

    return httpx.MockTransport(handler)


def _gfs_key(lead: int, *, idx: bool = False, hour: int = CYCLE_HOUR) -> str:
    return (
        f"gfs.{CYCLE_DATE:%Y%m%d}/{hour:02d}/atmos/"
        f"gfs.t{hour:02d}z.pgrb2.0p25.f{lead:03d}{'.idx' if idx else ''}"
    )


def _gefs_key(member: int, lead: int, *, idx: bool = False, hour: int = CYCLE_HOUR) -> str:
    return (
        f"gefs.{CYCLE_DATE:%Y%m%d}/{hour:02d}/atmos/pgrb2sp25/"
        f"gep{member:02d}.t{hour:02d}z.pgrb2s.0p25.f{lead:03d}{'.idx' if idx else ''}"
    )


# ---------------------------------------------------------------------------
# GFS
# ---------------------------------------------------------------------------


async def test_gfs_data_plus_idx_is_available_and_canonical_filtered() -> None:
    """Data + idx → complete; upstream-only hourly leads stay out of the contract view."""
    entries = []
    for lead in (0, 1, 2, 3, 4, 6, 9):  # 1/2/4 are upstream hourly, not contract
        entries.append((_gfs_key(lead), 1000, LM))
        entries.append((_gfs_key(lead, idx=True), 50, LM))
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )

    assert snapshot.model == "gfs"
    assert snapshot.prefix == gfs_cycle_prefix(CYCLE_DATE, CYCLE_HOUR)
    assert snapshot.ignored_key_count == 0
    assert snapshot.is_artifact_complete(None, 0)
    assert snapshot.is_artifact_complete(None, 6)
    assert snapshot.is_artifact_complete(None, 9)
    # Canonical platform interpretation excludes the upstream hourly leads.
    assert snapshot.available_leads() == (0, 3, 6, 9)
    # Raw reality still observed them.
    assert snapshot.observed_leads() == (0, 1, 2, 3, 4, 6, 9)
    assert snapshot.highest_observed_lead() == 9
    assert snapshot.region(None, 6).data is not None
    assert snapshot.region(None, 6).data.size == 1000
    assert snapshot.region(None, 6).idx is not None


async def test_gfs_data_without_idx_is_incomplete() -> None:
    entries = [(_gfs_key(12), 1000, LM)]
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert not snapshot.is_artifact_complete(None, 12)
    assert snapshot.available_leads() == ()
    assert snapshot.highest_observed_lead() == 12  # observed but not complete


async def test_gfs_idx_without_data_is_incomplete() -> None:
    entries = [(_gfs_key(15, idx=True), 50, LM)]
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert not snapshot.is_artifact_complete(None, 15)
    assert snapshot.available_leads() == ()


async def test_gfs_explicit_sequence_overrides_default_filtering() -> None:
    """The sequence-aware helper accepts a caller-provided lead sequence."""
    entries = []
    for lead in (0, 3, 6, 12):
        entries.append((_gfs_key(lead), 1000, LM))
        entries.append((_gfs_key(lead, idx=True), 50, LM))
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.available_leads((6, 12)) == (6, 12)


async def test_gfs_pagination_spans_pages() -> None:
    page1 = [(_gfs_key(0), 1, LM), (_gfs_key(0, idx=True), 1, LM)]
    page2 = [(_gfs_key(3), 1, LM), (_gfs_key(3, idx=True), 1, LM)]
    pages = {
        None: _listing_xml(page1, truncated=True, next_token="token-1"),
        "token-1": _listing_xml(page2, truncated=False, next_token=None),
    }
    snapshot = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.available_leads() == (0, 3)


async def test_gfs_empty_prefix_is_a_valid_empty_snapshot() -> None:
    """A cycle that has not started publishing is an empty snapshot, not an error."""
    pages = {None: _listing_xml([], truncated=False, next_token=None)}
    snapshot = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.regions == {}
    assert snapshot.observed_leads() == ()
    assert snapshot.highest_observed_lead() is None
    assert snapshot.available_leads() == ()


async def test_gfs_unrelated_and_malformed_keys_are_ignored_with_diagnostics() -> None:
    """Keys outside the exact product grammar must not crash the snapshot."""
    entries = [
        (_gfs_key(6), 1000, LM),
        (_gfs_key(6, idx=True), 50, LM),
        # Unrelated products under the same prefix path:
        f"gfs.{CYCLE_DATE:%Y%m%d}/{CYCLE_HOUR:02d}/atmos/gfs.t{CYCLE_HOUR:02d}z.atmf000.nc",
        f"gfs.{CYCLE_DATE:%Y%m%d}/{CYCLE_HOUR:02d}/atmos/gfs.t{CYCLE_HOUR:02d}z.pgrb2.0p25.f006.old",
        # Wrong cycle hour (defensive hour check):
        _gfs_key(6, hour=6),
        # Malformed lead claiming the product pattern:
        f"gfs.{CYCLE_DATE:%Y%m%d}/{CYCLE_HOUR:02d}/atmos/gfs.t{CYCLE_HOUR:02d}z.pgrb2.0p25.f00X",
        f"gfs.{CYCLE_DATE:%Y%m%d}/{CYCLE_HOUR:02d}/atmos/gfs.t{CYCLE_HOUR:02d}z.pgrb2.0p25.f6",
    ]
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.available_leads() == (6,)
    assert snapshot.ignored_key_count == 5
    assert snapshot.ignored_key_samples  # diagnostics preserved


# ---------------------------------------------------------------------------
# GEFS
# ---------------------------------------------------------------------------


def _gefs_entries(complete_members: list[int], lead: int, *, idx_only: list[int] = ()) -> list[tuple[str, int, str]]:
    entries: list[tuple[str, int, str]] = []
    for member in complete_members:
        entries.append((_gefs_key(member, lead), 2000, LM))
        entries.append((_gefs_key(member, lead, idx=True), 60, LM))
    for member in idx_only:
        entries.append((_gefs_key(member, lead, idx=True), 60, LM))
    return entries


async def test_gefs_thirty_of_thirty_members_complete() -> None:
    entries = _gefs_entries(list(range(1, 31)), 0)
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gefs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.complete_member_count(0) == 30
    assert snapshot.available_members(0) == tuple(range(1, 31))
    assert snapshot.missing_members(0, expected=tuple(range(1, 31))) == ()
    assert snapshot.available_member_leads() == tuple(
        (m, 0) for m in range(1, 31)
    )


async def test_gefs_twenty_nine_of_thirty_members_complete() -> None:
    entries = _gefs_entries(list(range(1, 30)), 3)  # member 30 missing
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gefs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.complete_member_count(3) == 29
    assert snapshot.missing_members(3, expected=tuple(range(1, 31))) == (30,)
    assert not snapshot.is_artifact_complete(30, 3)


async def test_gefs_partial_publication_progression() -> None:
    """A publishing lead: some members complete, some data-only/idx-only."""
    entries = [
        *_gefs_entries(list(range(1, 9)), 15),  # 8 complete
        *[_gefs_key(m, 15) for m in (9, 10)],  # 2 data-only
        *[_gefs_key(m, 15, idx=True) for m in (11,)],  # 1 idx-only
    ]
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gefs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.complete_member_count(15) == 8
    assert snapshot.missing_members(15, expected=tuple(range(1, 31))) == tuple(
        range(9, 31)
    )
    assert snapshot.highest_observed_lead() == 15


async def test_gefs_ignores_gec00_geavg_gespr_and_other_products() -> None:
    base = f"gefs.{CYCLE_DATE:%Y%m%d}/{CYCLE_HOUR:02d}/atmos/pgrb2sp25/"
    entries = [
        *_gefs_entries(list(range(1, 31)), 0),
        f"{base}gec00.t{CYCLE_HOUR:02d}z.pgrb2s.0p25.f000",
        f"{base}gec00.t{CYCLE_HOUR:02d}z.pgrb2s.0p25.f000.idx",
        f"{base}geavg.t{CYCLE_HOUR:02d}z.pgrb2s.0p25.f000",
        f"{base}geavg.t{CYCLE_HOUR:02d}z.pgrb2s.0p25.f000.idx",
        f"{base}gespr.t{CYCLE_HOUR:02d}z.pgrb2s.0p25.f000",
        f"{base}gespr.t{CYCLE_HOUR:02d}z.pgrb2s.0p25.f000.idx",
        f"gefs.{CYCLE_DATE:%Y%m%d}/{CYCLE_HOUR:02d}/atmos/pgrb2/gep01.t{CYCLE_HOUR:02d}z.pgrb2.0p25.f000",
    ]
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    snapshot = await snapshot_gefs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.complete_member_count(0) == 30
    assert snapshot.ignored_key_count == 7
    observed_keys = {region.data.key for region in snapshot.regions.values() if region.data}
    assert all(
        "geavg" not in key and "gespr" not in key and "gec00" not in key
        for key in observed_keys
    )


async def test_gefs_pagination_across_many_objects() -> None:
    """Pagination handles a shared prefix spanning >1000 listed objects."""
    entries = _gefs_entries(list(range(1, 31)), 0)
    entries += _gefs_entries(list(range(1, 31)), 3)
    entries += _gefs_entries(list(range(1, 31)), 6)
    # 30 members x 3 leads x (data + idx) = 180 entries; pad with unrelated
    # geavg/gespr keys to push the fixture past the 1000-key page boundary.
    base = gefs_cycle_prefix(CYCLE_DATE, CYCLE_HOUR)
    filler = [
        (f"{base}geavg.t{CYCLE_HOUR:02d}z.pgrb2s.0p25.f{lead:03d}", 1, LM)
        for lead in range(0, 850)
    ]
    entries.extend(filler)
    assert len(entries) > 1000
    first, second = entries[:1000], entries[1000:]
    pages = {
        None: _listing_xml(first, truncated=True, next_token="page-2"),
        "page-2": _listing_xml(second, truncated=False, next_token=None),
    }
    snapshot = await snapshot_gefs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert snapshot.complete_member_count(0) == 30
    assert snapshot.complete_member_count(3) == 30
    assert snapshot.complete_member_count(6) == 30
    assert snapshot.ignored_key_count == len(filler)


# ---------------------------------------------------------------------------
# Snapshot diff (publication activity)
# ---------------------------------------------------------------------------


async def test_publication_changed_detects_member_growth() -> None:
    """f015 8/30 → 22/30 is publication activity, per the Phase 5A scenario."""
    before_entries = _gefs_entries(list(range(1, 9)), 15)
    after_entries = _gefs_entries(list(range(1, 23)), 15)
    before = await snapshot_gefs_cycle(
        CYCLE_DATE,
        CYCLE_HOUR,
        conn_settings=_fast_settings(),
        transport=_transport({None: _listing_xml(before_entries, truncated=False, next_token=None)}),
    )
    after = await snapshot_gefs_cycle(
        CYCLE_DATE,
        CYCLE_HOUR,
        conn_settings=_fast_settings(),
        transport=_transport({None: _listing_xml(after_entries, truncated=False, next_token=None)}),
    )
    assert publication_changed(before, after)
    assert before.complete_member_count(15) == 8
    assert after.complete_member_count(15) == 22


async def test_publication_unchanged_for_identical_snapshots() -> None:
    entries = _gefs_entries(list(range(1, 31)), 0)
    pages = {None: _listing_xml(entries, truncated=False, next_token=None)}
    first = await snapshot_gefs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    second = await snapshot_gefs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(), transport=_transport(pages)
    )
    assert first.fingerprint() == second.fingerprint()
    assert not publication_changed(first, second)


async def test_publication_changed_detects_size_update() -> None:
    """A same-key size change (upstream rewrite) also counts as activity."""
    entries_a = [(_gfs_key(0), 1000, LM), (_gfs_key(0, idx=True), 50, LM)]
    entries_b = [(_gfs_key(0), 1200, LM), (_gfs_key(0, idx=True), 50, LM)]
    first = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(),
        transport=_transport({None: _listing_xml(entries_a, truncated=False, next_token=None)}),
    )
    second = await snapshot_gfs_cycle(
        CYCLE_DATE, CYCLE_HOUR, conn_settings=_fast_settings(),
        transport=_transport({None: _listing_xml(entries_b, truncated=False, next_token=None)}),
    )
    assert publication_changed(first, second)


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


def _error_transport(
    status_codes: list[int] | None = None,
    *,
    body: bytes = b"<Listing/>",
    raise_transport: bool = False,
) -> httpx.MockTransport:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if raise_transport:
            raise httpx.ConnectError("connection refused", request=request)
        assert status_codes is not None
        status = status_codes[min(state["calls"] - 1, len(status_codes) - 1)]
        return httpx.Response(status, content=body)

    return httpx.MockTransport(handler)


async def test_transport_failure_raises_unavailable_not_empty() -> None:
    """Network failures must NOT degrade into 'nothing published'."""
    with pytest.raises(DiscoveryUnavailableError):
        await snapshot_gfs_cycle(
            CYCLE_DATE,
            CYCLE_HOUR,
            conn_settings=_fast_settings(),
            transport=_error_transport(raise_transport=True),
        )


async def test_server_error_raises_unavailable_after_retries() -> None:
    settings_retry = IngestionSettings(DOWNLOAD_RETRIES=2, RETRY_BACKOFF_SECONDS=0.0)
    transport = _error_transport(status_codes=[500, 503, 500])
    with pytest.raises(DiscoveryUnavailableError):
        await snapshot_gfs_cycle(
            CYCLE_DATE, CYCLE_HOUR, conn_settings=settings_retry, transport=transport
        )


async def test_access_denied_error_document_raises_invalid_response() -> None:
    error_doc = (
        '<?xml version="1.0" encoding="UTF-8"?><Error>'
        "<Code>AccessDenied</Code><Message>denied</Message></Error>"
    ).encode()
    with pytest.raises(DiscoveryInvalidResponseError, match="AccessDenied"):
        await snapshot_gfs_cycle(
            CYCLE_DATE,
            CYCLE_HOUR,
            conn_settings=_fast_settings(),
            transport=_error_transport(status_codes=[403], body=error_doc),
        )


async def test_malformed_xml_raises_invalid_response() -> None:
    with pytest.raises(DiscoveryInvalidResponseError):
        await snapshot_gfs_cycle(
            CYCLE_DATE,
            CYCLE_HOUR,
            conn_settings=_fast_settings(),
            transport=_error_transport(status_codes=[200], body=b"<not-xml"),
        )


async def test_truncated_page_without_token_raises_invalid_response() -> None:
    body = _listing_xml([(_gfs_key(0), 1, LM)], truncated=True, next_token=None)
    with pytest.raises(DiscoveryInvalidResponseError, match="continuation token"):
        await snapshot_gfs_cycle(
            CYCLE_DATE,
            CYCLE_HOUR,
            conn_settings=_fast_settings(),
            transport=_error_transport(status_codes=[200], body=body),
        )


async def test_runaway_pagination_raises_pagination_error() -> None:
    body = _listing_xml([(_gfs_key(0), 1, LM)], truncated=True, next_token="loop")
    pages = {None: body, "loop": body}
    with pytest.raises(DiscoveryPaginationError):
        await snapshot_gfs_cycle(
            CYCLE_DATE,
            CYCLE_HOUR,
            conn_settings=_fast_settings(),
            transport=_transport(pages),
        )


def test_last_modified_metadata_is_preserved() -> None:
    from datetime import timezone

    from ingestion.providers.noaa.discovery import _parse_last_modified

    assert _parse_last_modified(LM) == datetime(
        2026, 7, 21, 3, 35, 10, tzinfo=timezone.utc
    )
    assert _parse_last_modified(None) is None
    assert _parse_last_modified("not-a-date") is None
