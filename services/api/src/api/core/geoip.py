"""Local GeoLite2 IP geolocation for the ``/v1/locate`` coarse-region fallback.

A self-hosted origin has no CDN injecting the Cloudflare ``cf-*`` location
headers, so the visitor address is read from the request itself and resolved
against a local MaxMind database. Two properties of the reader matter for that
deployment:

* It is opened with ``MODE_MMAP``, so the kernel pages the binary search tree in
  on demand. Serving a handful of distinct visitor subnets keeps a few megabytes
  resident instead of the whole ~70 MB file, and because the mapping is
  file-backed those pages are shared across the uvicorn workers rather than
  duplicated per worker. ``MODE_MEMORY`` (the file read into each process heap)
  must never be used here: the API container runs near its memory limit.
* It is reopened when the database file changes on disk, so a GeoLite2 refresh
  is picked up without restarting the API.

Lookups are microseconds, which is faster than a Redis round trip, so results
are not cached: there is no stale entry to invalidate after a database refresh
and no per-IP key growth.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import threading
from dataclasses import dataclass

import maxminddb
from maxminddb.errors import InvalidDatabaseError
from maxminddb.types import Record
from starlette.requests import Request

from api.core.config import settings

logger = logging.getLogger(__name__)

#: ``settings.LOCATE_PROVIDER`` value that enables the local database.
PROVIDER_MAXMIND = "maxmind"

#: ``settings.LOCATE_PROXY_MODE`` values.
PROXY_MODE_AUTO = "auto"
PROXY_MODE_ALWAYS = "always"
PROXY_MODE_NEVER = "never"

_reader: maxminddb.Reader | None = None
#: ``(st_mtime_ns, st_size)`` of the file the open reader was built from.
_reader_signature: tuple[int, int] | None = None
_missing_database_warned = False
_reader_lock = threading.Lock()


@dataclass(frozen=True)
class GeoLocation:
    """Coarse visitor location resolved from an IP address."""

    latitude: float
    longitude: float
    city: str | None
    region: str | None
    country: str | None


def client_ip(request: Request) -> str | None:
    """Resolve the visitor address behind the gateway (see ``LOCATE_PROXY_MODE``).

    The gateway *overwrites* ``X-Real-IP`` with ``$remote_addr``, so it cannot be
    forged through nginx — but a caller that reaches the published API port
    directly can set it to anything. ``auto`` therefore trusts the header only
    when the socket peer is not a routable address, which is precisely the case
    for a request that came from the gateway container or the host. The
    ``X-Forwarded-For`` chain is deliberately not consulted: nginx appends to it,
    so its leftmost entry is caller-supplied.
    """
    peer = request.client.host if request.client is not None else None
    mode = str(settings.LOCATE_PROXY_MODE).strip().lower()
    if mode == PROXY_MODE_NEVER:
        return peer

    forwarded = (request.headers.get("x-real-ip") or "").strip()
    if not forwarded:
        return peer
    if mode == PROXY_MODE_ALWAYS:
        return forwarded
    if peer is not None and not is_public_ip(peer):
        return forwarded
    return peer


def is_public_ip(value: str) -> bool:
    """True when the address is a routable visitor address.

    ``is_global`` is False for loopback, private, link-local, CGNAT and
    documentation ranges. Those must not reach the database: a development or
    same-host request would otherwise resolve to whatever MaxMind records for
    that range instead of falling through to the default CONUS view.
    """
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        # Proxies that speak IPv6 to the origin may report IPv4 visitors in the
        # mapped form, which is not itself in the global registry.
        address = address.ipv4_mapped
    return address.is_global


def _database_signature(path: str) -> tuple[int, int] | None:
    """Identity of the database file, used to detect an in-place refresh."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _open_reader(path: str) -> maxminddb.Reader:
    """Open the database memory-mapped (split out so tests can inject a reader)."""
    return maxminddb.open_database(path, maxminddb.MODE_MMAP)


def _discard_reader() -> None:
    """Close and forget the open reader so the next lookup reopens the file."""
    global _reader, _reader_signature
    with _reader_lock:
        previous = _reader
        _reader = None
        _reader_signature = None
    if previous is not None:
        previous.close()


def _get_reader() -> maxminddb.Reader | None:
    """Return the open reader, reopening it when the database file is replaced.

    The fast path is a lock-free read of two module globals: a concurrent
    reopen can at worst cost one redundant open, never a torn reader.
    """
    global _reader, _reader_signature, _missing_database_warned

    path = settings.LOCATE_GEOIP_DB_PATH
    signature = _database_signature(path)
    if signature is None:
        if not _missing_database_warned:
            _missing_database_warned = True
            logger.warning(
                "GeoLite2 database not found at %s; /v1/locate returns 404 until "
                "it is provisioned (see docs/DEPLOYMENT.md)",
                path,
            )
        return None

    if _reader is not None and _reader_signature == signature:
        return _reader

    with _reader_lock:
        if _reader is not None and _reader_signature == signature:
            return _reader
        try:
            reader = _open_reader(path)
        except (OSError, ValueError, InvalidDatabaseError) as exc:
            logger.warning("Failed to open GeoLite2 database at %s: %s", path, exc)
            return None
        previous = _reader
        _reader = reader
        _reader_signature = signature
        _missing_database_warned = False
        if previous is not None:
            previous.close()
            logger.info("Reopened GeoLite2 database at %s after a refresh", path)
        return reader


def _node(record: Record, *keys: str) -> Record | None:
    """Walk a nested record, returning None as soon as the path is absent."""
    node: Record = record
    for key in keys:
        if not isinstance(node, dict):
            return None
        child = node.get(key)
        if child is None:
            return None
        node = child
    return node


def _string(record: Record, *keys: str) -> str | None:
    value = _node(record, *keys)
    return value if isinstance(value, str) and value else None


def _number(record: Record, *keys: str) -> float | None:
    value = _node(record, *keys)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _subdivision(record: Record) -> str | None:
    """Name of the most specific subdivision (GeoLite2 orders them broad to specific)."""
    subdivisions = _node(record, "subdivisions")
    if not isinstance(subdivisions, list) or not subdivisions:
        return None
    most_specific = subdivisions[0]
    return _string(most_specific, "names", "en") or _string(most_specific, "iso_code")


def lookup_location(ip: str) -> GeoLocation | None:
    """Resolve a public IP address to a coarse location, or None when unknown."""
    reader = _get_reader()
    if reader is None:
        return None

    try:
        record = reader.get(ip)
    except ValueError:
        # Malformed address: not a visitor this tier can place.
        return None
    except InvalidDatabaseError as exc:
        # A corrupt database cannot answer any query; drop the reader so the
        # next request reopens the file instead of failing until a restart.
        logger.warning("GeoLite2 lookup failed for %s: %s", ip, exc)
        _discard_reader()
        return None

    if record is None:
        return None

    latitude = _number(record, "location", "latitude")
    longitude = _number(record, "location", "longitude")
    if latitude is None or longitude is None:
        return None
    if latitude == 0.0 and longitude == 0.0:
        # The (0, 0) null fix is what unresolvable ranges carry; keeping it
        # would center the map in the Gulf of Guinea instead of the default view.
        return None

    return GeoLocation(
        latitude=latitude,
        longitude=longitude,
        city=_string(record, "city", "names", "en"),
        region=_subdivision(record),
        country=_string(record, "country", "iso_code"),
    )


def _reset_for_testing() -> None:
    """Drop the cached reader and the one-shot warning flag (tests only)."""
    global _missing_database_warned
    _discard_reader()
    _missing_database_warned = False
