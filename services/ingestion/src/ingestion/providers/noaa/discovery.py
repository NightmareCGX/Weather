"""Read-only NOAA upstream publication discovery (anonymous S3 ListObjectsV2).

This module answers "what has NOAA actually published for one forecast cycle?"
by listing the upstream AWS Open Data buckets — it never downloads data and
never touches durable platform state (no Zarr writes, no catalog writes, no
markers). It is the observation input for the future realtime lead-wave
scheduler (Phase 5C); the shared GFS+GEFS completeness barrier and frontier
planning deliberately live ABOVE this layer.

Mechanism (Phase 5A investigation, verified live 2026-09-02):

* Anonymous ``ListObjectsV2`` (``GET https://{bucket}.s3.amazonaws.com/?list-type=2&prefix=…``)
  is permitted on both buckets and returns every object's key, size, ETag, and
  Last-Modified with paginated results (1000 keys/page).
* Product-scoped prefixes keep one cycle snapshot cheap:
  - GFS  ``gfs.{date}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f``  → 1 page
    (209 data + 209 .idx objects for the full upstream horizon);
  - GEFS ``gefs.{date}/{hh}/atmos/pgrb2sp25/``              → ~6 pages
    (the shared product prefix spans gep01..gep30 plus geavg/gespr, whose keys
    are parsed out / ignored). This is preferred over 30 per-member listings.
* Artifact completeness predicate: a GRIB artifact is *available* only when
  the data object AND its ``.idx`` sidecar are both listed. S3 object
  visibility is atomic (an object appears only after its upload completes)
  and the ``.idx`` is generated from the completed GRIB2 file (observed to
  land seconds after it), so data + idx presence is a strong fully-published
  signal. Data-object presence alone must never be treated as complete.
* Discovery observes EVERY key matching the product grammar — including
  upstream-only GFS hourly leads (f001, f002, …) that are not platform
  targets. Interpretation/filtering against the canonical platform lead
  sequence (``domain.horizon``) is available via the snapshot's sequence-aware
  helpers; discovery itself never decides platform eligibility.

Error semantics (§Phase 5B): a successful EMPTY listing is a valid, empty
snapshot (the cycle has not started publishing). Network/transport failures
and 5xx responses raise :class:`DiscoveryUnavailableError` after bounded
retries; a non-200 response, an S3 error document, or unparseable XML raises
:class:`DiscoveryInvalidResponseError`; pagination that fails to terminate
raises :class:`DiscoveryPaginationError`. Errors are never silently mapped to
"nothing published" — Phase 5C backoff needs that distinction.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping

import httpx

from domain.horizon import canonical_lead_time_hours
from ingestion.core.base import IngestionError
from ingestion.core.config import IngestionSettings, settings

_S3_XML_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

#: S3 ListObjectsV2 page size (the protocol maximum).
_LIST_PAGE_SIZE = 1000

#: Hard cap on listing pages per snapshot so a broken continuation loop fails
#: instead of running unbounded (a full GEFS product prefix is ~6 pages).
_MAX_LIST_PAGES = 64

#: Full-key grammar for GFS 0.25° pgrb2 products (data + .idx sidecar).
_GFS_KEY_RE = re.compile(
    r"^gfs\.\d{8}/(?P<hour>\d{2})/atmos/gfs\.t(?P=hour)z\.pgrb2\.0p25\.f"
    r"(?P<lead>\d{3})(?P<idx>\.idx)?$"
)

#: Full-key grammar for GEFS 0.25° pgrb2sp25 perturbation-member products
#: (data + .idx sidecar). ``geavg``/``gespr``/``gec00`` keys do not match the
#: ``gepNN`` grammar and are ignored (counted as unrelated keys).
_GEFS_KEY_RE = re.compile(
    r"^gefs\.\d{8}/(?P<hour>\d{2})/atmos/pgrb2sp25/"
    r"gep(?P<member>\d{2})\.t(?P=hour)z\.pgrb2s\.0p25\.f"
    r"(?P<lead>\d{3})(?P<idx>\.idx)?$"
)


class DiscoveryError(IngestionError):
    """Base error for upstream discovery failures."""


class DiscoveryUnavailableError(DiscoveryError):
    """The upstream could not be reached (transport failure / 5xx after retries)."""


class DiscoveryInvalidResponseError(DiscoveryError):
    """The upstream responded, but the response is not a usable listing."""


class DiscoveryPaginationError(DiscoveryError):
    """Listing pagination failed to terminate or exceeded the page budget."""


@dataclass(frozen=True)
class ArtifactObservation:
    """Metadata of one listed upstream object (advisory, for activity diffs)."""

    key: str
    size: int
    etag: str | None = None
    last_modified: datetime | None = None


@dataclass(frozen=True)
class RegionArtifacts:
    """The listed publication state of one logical (member?, lead) artifact.

    ``member`` is ``None`` for deterministic products (GFS). An artifact is
    **complete** only when both the data object and its ``.idx`` sidecar are
    present in the same listing snapshot.
    """

    data: ArtifactObservation | None
    idx: ArtifactObservation | None

    @property
    def is_complete(self) -> bool:
        """True when both the GRIB2 data object and its .idx sidecar are listed."""
        return self.data is not None and self.idx is not None


@dataclass(frozen=True)
class CycleSnapshot:
    """Immutable per-model view of what upstream has published for one cycle.

    Regions are keyed by ``(member, lead)`` where ``member`` is ``None`` for
    deterministic products. The snapshot preserves the raw upstream reality
    (including non-contract leads such as GFS hourly files) plus a bounded
    sample of ignored keys for diagnostics; sequence-aware helpers apply the
    canonical platform lead sequence on demand.
    """

    model: str
    cycle_date: date
    cycle_hour: int
    prefix: str
    regions: Mapping[tuple[int | None, int], RegionArtifacts]
    ignored_key_count: int = 0
    ignored_key_samples: tuple[str, ...] = field(default_factory=tuple)

    # -- raw reality ---------------------------------------------------

    def observed_leads(self) -> tuple[int, ...]:
        """All leads with ANY listed artifact (complete or not), ascending.

        Includes non-platform leads (e.g. upstream GFS hourly f001/f002).
        """
        return tuple(sorted({lead for (_, lead) in self.regions}))

    def highest_observed_lead(self) -> int | None:
        """The highest lead with any listed artifact, or ``None`` when empty."""
        leads = self.observed_leads()
        return leads[-1] if leads else None

    def region(self, member: int | None, lead: int) -> RegionArtifacts:
        """The listed artifacts for one (member, lead); absent when unlisted."""
        return self.regions.get((member, lead), RegionArtifacts(data=None, idx=None))

    # -- availability (data + .idx predicate) --------------------------

    def is_artifact_complete(self, member: int | None, lead: int) -> bool:
        """Whether the (member, lead) artifact has both data and .idx listed."""
        return self.region(member, lead).is_complete

    def available_members(self, lead: int) -> tuple[int, ...]:
        """Members whose artifact for ``lead`` is complete (ascending)."""
        return tuple(
            sorted(
                member
                for (member, lead_val), region in self.regions.items()
                if lead_val == lead and member is not None and region.is_complete
            )
        )

    def complete_member_count(self, lead: int) -> int:
        """How many members' artifacts for ``lead`` are complete."""
        return len(self.available_members(lead))

    def missing_members(self, lead: int, expected: tuple[int, ...]) -> tuple[int, ...]:
        """Expected members whose artifact for ``lead`` is not complete."""
        present = set(self.available_members(lead))
        return tuple(m for m in expected if m not in present)

    def available_leads(
        self, sequence: tuple[int, ...] | None = None
    ) -> tuple[int, ...]:
        """Deterministic-product leads complete within an optional sequence.

        For GFS-style snapshots (``member=None`` regions), a lead is available
        when its data + .idx are both listed. When ``sequence`` is given
        (default: the model's canonical horizon), only sequence leads are
        returned — this is how upstream-only hourly GFS leads are excluded
        from platform interpretation.
        """
        seq = (
            sequence
            if sequence is not None
            else canonical_lead_time_hours(self.model)
        )
        allowed = set(seq)
        return tuple(
            sorted(
                lead
                for (member, lead), region in self.regions.items()
                if member is None and lead in allowed and region.is_complete
            )
        )

    def available_member_leads(
        self, sequence: tuple[int, ...] | None = None
    ) -> tuple[tuple[int, int], ...]:
        """Ensemble ``(member, lead)`` artifacts complete, lead-ordered.

        When ``sequence`` is given (default: the model's canonical horizon),
        only leads within the sequence are returned.
        """
        seq = (
            sequence
            if sequence is not None
            else canonical_lead_time_hours(self.model)
        )
        allowed = set(seq)
        return tuple(
            sorted(
                (member, lead)
                for (member, lead), region in self.regions.items()
                if member is not None and lead in allowed and region.is_complete
            )
        )

    # -- activity ------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable digest of the snapshot's publication state (for diffing)."""
        import hashlib

        hasher = hashlib.sha256()
        for (member, lead), region in sorted(
            self.regions.items(), key=lambda item: (item[0][0] or -1, item[0][1])
        ):
            state = (
                member if member is not None else -1,
                lead,
                region.data is not None,
                region.data.size if region.data is not None else -1,
                region.idx is not None,
                region.idx.size if region.idx is not None else -1,
            )
            hasher.update(repr(state).encode("utf-8"))
        return hasher.hexdigest()


def publication_changed(before: CycleSnapshot, after: CycleSnapshot) -> bool:
    """Whether upstream publication advanced between two snapshots.

    Any change counts — new keys, per-lead member-count growth, or size
    changes — not just complete-frontier growth, so a lead that is actively
    receiving members (e.g. 8/30 → 22/30) is recognizable as publication
    activity.
    """
    return before.fingerprint() != after.fingerprint()


def gfs_cycle_prefix(cycle_date: date, cycle_hour: int) -> str:
    """Product-scoped GFS listing prefix for one cycle (data + .idx)."""
    return f"gfs.{cycle_date:%Y%m%d}/{cycle_hour:02d}/atmos/gfs.t{cycle_hour:02d}z.pgrb2.0p25.f"


def gefs_cycle_prefix(cycle_date: date, cycle_hour: int) -> str:
    """Product-scoped GEFS listing prefix for one cycle (shared across members)."""
    return f"gefs.{cycle_date:%Y%m%d}/{cycle_hour:02d}/atmos/pgrb2sp25/"


async def snapshot_gfs_cycle(
    cycle_date: date,
    cycle_hour: int,
    *,
    conn_settings: IngestionSettings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CycleSnapshot:
    """List what GFS has published for one cycle (read-only).

    Args:
        cycle_date: UTC cycle date.
        cycle_hour: UTC cycle hour (0/6/12/18).
        conn_settings: Optional settings override (defaults to the global
            settings; the GFS bucket base URL comes from ``AWS_GFS_BASE_URL``).
        transport: Optional httpx transport injection for offline tests.

    Returns:
        The immutable GFS cycle snapshot (possibly empty).
    """
    return await _snapshot_cycle(
        model="gfs",
        cycle_date=cycle_date,
        cycle_hour=cycle_hour,
        prefix=gfs_cycle_prefix(cycle_date, cycle_hour),
        key_pattern=_GFS_KEY_RE,
        conn_settings=conn_settings,
        transport=transport,
        bucket_base=_bucket_base(conn_settings, gfs=True),
    )


async def snapshot_gefs_cycle(
    cycle_date: date,
    cycle_hour: int,
    *,
    conn_settings: IngestionSettings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CycleSnapshot:
    """List what GEFS has published for one cycle (read-only).

    Uses the shared ``pgrb2sp25`` product prefix (lowest request count);
    ``gep01..gepNN`` keys are parsed, ``geavg``/``gespr``/``gec00`` and any
    other keys are ignored with diagnostics.

    Args:
        cycle_date: UTC cycle date.
        cycle_hour: UTC cycle hour (0/6/12/18).
        conn_settings: Optional settings override (defaults to the global
            settings; the GEFS bucket base URL comes from ``AWS_GEFS_BASE_URL``).
        transport: Optional httpx transport injection for offline tests.

    Returns:
        The immutable GEFS cycle snapshot (possibly empty).
    """
    return await _snapshot_cycle(
        model="gefs",
        cycle_date=cycle_date,
        cycle_hour=cycle_hour,
        prefix=gefs_cycle_prefix(cycle_date, cycle_hour),
        key_pattern=_GEFS_KEY_RE,
        conn_settings=conn_settings,
        transport=transport,
        bucket_base=_bucket_base(conn_settings, gfs=False),
    )


def _bucket_base(
    conn_settings: IngestionSettings | None, *, gfs: bool
) -> str:
    """Return the bucket list endpoint (bucket root) for the model."""
    resolved = conn_settings or settings
    base = str(resolved.AWS_GFS_BASE_URL if gfs else resolved.AWS_GEFS_BASE_URL)
    return base.rstrip("/")


def _parse_last_modified(raw: str | None) -> datetime | None:
    """Parse an S3 ISO-8601 Last-Modified timestamp (advisory metadata)."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_listing_page(body: bytes) -> tuple[list[dict[str, Any]], str | None]:
    """Parse one ListObjectsV2 XML page into contents + continuation token.

    Returns:
        ``(contents, next_token)`` where each content entry carries ``key``,
        ``size``, ``etag``, and ``last_modified``.

    Raises:
        DiscoveryInvalidResponseError: If the body is not parseable XML or is
            an S3 error document rather than a listing.
    """
    import logging

    logger = logging.getLogger(__name__)
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise DiscoveryInvalidResponseError(
            f"upstream listing is not valid XML: {exc}"
        ) from exc
    if _local_tag(root) != "ListBucketResult":
        code = None
        if _local_tag(root) == "Error":
            for child in root:
                if _local_tag(child) == "Code":
                    code = child.text
                    break
        raise DiscoveryInvalidResponseError(
            f"upstream listing endpoint returned an error document "
            f"(code={code or root.tag})"
        )
    contents: list[dict[str, Any]] = []
    for node in root.findall(f"{_S3_XML_NS}Contents"):
        key_node = node.findtext(f"{_S3_XML_NS}Key")
        if key_node is None:
            # A Contents entry without a key cannot be interpreted; skip it
            # (counted as ignored by the caller's grammar filter).
            continue
        size_raw = node.findtext(f"{_S3_XML_NS}Size")
        try:
            size = int(size_raw) if size_raw is not None else 0
        except ValueError:
            size = 0
        etag = node.findtext(f"{_S3_XML_NS}ETag")
        last_modified = _parse_last_modified(
            node.findtext(f"{_S3_XML_NS}LastModified")
        )
        contents.append(
            {
                "key": key_node,
                "size": size,
                "etag": etag.strip('"') if etag else None,
                "last_modified": last_modified,
            }
        )
    truncated = (root.findtext(f"{_S3_XML_NS}IsTruncated") or "false").strip().lower()
    next_token: str | None = None
    if truncated == "true":
        next_token = root.findtext(f"{_S3_XML_NS}NextContinuationToken")
        if not next_token:
            raise DiscoveryInvalidResponseError(
                "upstream listing is truncated but carries no continuation token"
            )
    logger.debug("parsed listing page: contents=%d truncated=%s", len(contents), truncated)
    return contents, next_token


def _local_tag(element: ET.Element) -> str:
    """Return an element's tag without any XML namespace prefix."""
    return element.tag.rsplit("}", 1)[-1]


def _describe_error_response(response: httpx.Response, prefix: str) -> str:
    """Describe a non-200 listing response, extracting the S3 error code when present."""
    code: str | None = None
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        root = None
    if root is not None and _local_tag(root) == "Error":
        for child in root:
            if _local_tag(child) == "Code":
                code = child.text
                break
    detail = f" (code={code})" if code else ""
    return (
        f"upstream listing returned HTTP {response.status_code} for prefix "
        f"{prefix!r}{detail}"
    )


async def _list_all_pages(
    client: httpx.AsyncClient,
    bucket_base: str,
    prefix: str,
    *,
    conn_settings: IngestionSettings,
) -> list[dict[str, Any]]:
    """List every object under ``prefix`` (paginated, bounded, with retries).

    Transport errors and 5xx responses are retried with the connector's
    bounded backoff conventions; any other non-200 response, an S3 error
    document, or unparseable XML fails fast as an invalid response.

    Raises:
        DiscoveryUnavailableError: After retry exhaustion on transport/5xx.
        DiscoveryInvalidResponseError: On a non-200 status or bad document.
        DiscoveryPaginationError: If pagination exceeds the page budget or a
            truncated page carries no token.
    """
    import asyncio
    import logging

    logger = logging.getLogger(__name__)
    attempts = int(conn_settings.DOWNLOAD_RETRIES) + 1
    backoff = float(conn_settings.RETRY_BACKOFF_SECONDS)
    keys: list[dict[str, Any]] = []
    continuation: str | None = None
    pages = 0
    while True:
        params: dict[str, str] = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": str(_LIST_PAGE_SIZE),
        }
        if continuation:
            params["continuation-token"] = continuation
        last_error: BaseException | None = None
        page: tuple[list[dict[str, Any]], str | None] | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = await client.get(bucket_base, params=params)
            except httpx.TransportError as exc:
                last_error = exc
                if attempt < attempts:
                    await asyncio.sleep(backoff * attempt)
                continue
            if response.status_code >= 500:
                last_error = DiscoveryUnavailableError(
                    f"upstream listing returned HTTP {response.status_code} "
                    f"for prefix {prefix!r}"
                )
                if attempt < attempts:
                    await asyncio.sleep(backoff * attempt)
                continue
            if response.status_code != 200:
                raise DiscoveryInvalidResponseError(
                    _describe_error_response(response, prefix)
                )
            last_error = None
            page = _parse_listing_page(response.content)
            break
        if page is None:
            assert last_error is not None
            if isinstance(last_error, DiscoveryUnavailableError):
                raise last_error
            raise DiscoveryUnavailableError(
                f"upstream listing unreachable after {attempts} attempts for "
                f"prefix {prefix!r}"
            ) from last_error

        page_contents, continuation = page
        keys.extend(page_contents)
        pages += 1
        if pages > _MAX_LIST_PAGES:
            raise DiscoveryPaginationError(
                f"listing prefix {prefix!r} exceeded {_MAX_LIST_PAGES} pages; "
                "refusing to paginate unboundedly"
            )
        if continuation is None:
            return keys
        logger.debug("listing prefix %s: page %d complete, continuing", prefix, pages)


def _snapshot_from_keys(
    *,
    model: str,
    cycle_date: date,
    cycle_hour: int,
    prefix: str,
    key_pattern: re.Pattern[str],
    listed: list[dict[str, Any]],
) -> CycleSnapshot:
    """Build the immutable snapshot by applying the product grammar to keys."""
    import logging

    logger = logging.getLogger(__name__)
    regions: dict[tuple[int | None, int], RegionArtifacts] = {}
    ignored: list[str] = []
    for entry in listed:
        key = str(entry["key"])
        match = key_pattern.match(key)
        if match is None:
            # Anything outside the exact expected product grammar is unrelated
            # (geavg/gespr/gec00 on GEFS, .nc products, stray suffixes) —
            # ignore with diagnostics instead of failing the snapshot.
            ignored.append(key)
            continue
        member = int(match.group("member")) if "member" in match.groupdict() else None
        if int(match.group("hour")) != cycle_hour:
            # Defensive: an object from a different cycle hour under this
            # prefix is not part of this snapshot.
            ignored.append(key)
            continue
        lead = int(match.group("lead"))
        observation = ArtifactObservation(
            key=key,
            size=int(entry["size"]),
            etag=entry["etag"],
            last_modified=entry["last_modified"],
        )
        region_key = (member, lead)
        existing = regions.get(region_key, RegionArtifacts(data=None, idx=None))
        if match.group("idx"):
            regions[region_key] = RegionArtifacts(data=existing.data, idx=observation)
        else:
            regions[region_key] = RegionArtifacts(data=observation, idx=existing.idx)
    if ignored:
        logger.debug(
            "discovery ignored %d unrelated key(s) under %s (samples: %s)",
            len(ignored),
            prefix,
            ignored[:5],
        )
    return CycleSnapshot(
        model=model,
        cycle_date=cycle_date,
        cycle_hour=cycle_hour,
        prefix=prefix,
        regions=regions,
        ignored_key_count=len(ignored),
        ignored_key_samples=tuple(ignored[:20]),
    )


async def _snapshot_cycle(
    *,
    model: str,
    cycle_date: date,
    cycle_hour: int,
    prefix: str,
    key_pattern: re.Pattern[str],
    conn_settings: IngestionSettings | None,
    transport: httpx.AsyncBaseTransport | None,
    bucket_base: str,
) -> CycleSnapshot:
    """Snapshot one cycle: paginate the product prefix and parse the keys."""
    resolved = conn_settings or settings
    client_kwargs: dict[str, Any] = {
        "timeout": float(resolved.REQUEST_TIMEOUT_SECONDS),
        "headers": {"User-Agent": str(resolved.NOAA_USER_AGENT)},
    }
    if transport is not None:
        client_kwargs["transport"] = transport
    async with httpx.AsyncClient(**client_kwargs) as client:
        listed = await _list_all_pages(
            client, bucket_base, prefix, conn_settings=resolved
        )
    return _snapshot_from_keys(
        model=model,
        cycle_date=cycle_date,
        cycle_hour=cycle_hour,
        prefix=prefix,
        key_pattern=key_pattern,
        listed=listed,
    )
