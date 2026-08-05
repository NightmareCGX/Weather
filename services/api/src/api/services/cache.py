"""Redis-primary point-query cache with a PostgreSQL fallback audit ledger.

Point forecast responses are cached in Redis with a TTL aligned to the
model update cadence (``public, max-age=1800`` per API.md section 2.1).
Redis is a best-effort accelerator: when it is unavailable the response is
computed directly and a ``point_query_fallback_audit`` row records the
fallback (the table's purpose per DATABASE.md section 2). The cache key is a
deterministic canonical string derived from the normalized request, so
identical requests always map to the same key.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import redis as redis_lib
from pydantic import ValidationError
from sqlalchemy.orm import Session

from api.core.config import settings
from api.models.entities import PointQueryFallbackAudit
from api.schemas import PointForecastEnvelope

#: Cache TTL in seconds, aligned to the ``public, max-age=1800`` cache policy.
POINT_CACHE_TTL_SECONDS = 1800

#: Fallback audit reason: Redis was unreachable while reading the cache.
FALLBACK_REASON_REDIS_READ_UNAVAILABLE = "redis_read_unavailable"
#: Fallback audit reason: Redis was unreachable while writing the cache.
FALLBACK_REASON_REDIS_WRITE_UNAVAILABLE = "redis_write_unavailable"
#: Fallback audit reason: the cached payload was corrupt and treated as a miss.
FALLBACK_REASON_CORRUPT_CACHE_ENTRY = "corrupt_cache_entry"
#: Fallback audit reason: Redis was unreachable on both the read and write
#: paths for the same request. Because the audit row's ``cache_key`` is the
#: primary key (one row per request), both failures are combined into a
#: single reason so neither signal is lost.
FALLBACK_REASON_REDIS_READ_AND_WRITE_UNAVAILABLE = (
    "redis_read_and_write_unavailable"
)


@dataclass(frozen=True)
class _CacheRead:
    """Result of reading the point cache for a key.

    Attributes:
        hit: Whether a valid cached response was found.
        envelope: The cached response when ``hit`` is True, else ``None``.
        fallback_reason: Why the read fell back to computing, or ``None`` when
            the read was a clean miss (no entry) or a hit.
    """

    hit: bool
    envelope: PointForecastEnvelope | None
    fallback_reason: str | None = None


class PointCache:
    """Redis-primary cache for point forecasts with PostgreSQL fallback.

    Args:
        redis_url: Redis URL. Defaults to the configured ``REDIS_URL``.
        ttl_seconds: Cache TTL for stored responses.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = POINT_CACHE_TTL_SECONDS,
    ) -> None:
        self._redis_url = redis_url or settings.REDIS_URL
        self._ttl_seconds = ttl_seconds
        self._client = redis_lib.from_url(
            self._redis_url, decode_responses=True
        )

    def get(self, cache_key: str) -> _CacheRead:
        """Read the cache for a key, reporting how the read resolved.

        Redis errors and unparseable cached payloads (malformed JSON or a
        schema that fails validation) are treated as cache misses so a
        transient Redis failure or a corrupt entry never fails the request. A
        corrupt entry is deleted so it cannot keep breaking the endpoint. The
        returned :class:`_CacheRead` distinguishes these cases so the caller
        can record a best-effort fallback audit event.

        Args:
            cache_key: The deterministic cache key.

        Returns:
            A ``_CacheRead`` describing whether the read was a hit, a clean
            miss, or a fallback (Redis unavailable / corrupt entry).
        """
        try:
            raw = self._client.get(cache_key)
        except redis_lib.RedisError:
            return _CacheRead(
                hit=False,
                envelope=None,
                fallback_reason=FALLBACK_REASON_REDIS_READ_UNAVAILABLE,
            )
        if raw is None:
            return _CacheRead(hit=False, envelope=None)
        try:
            envelope = PointForecastEnvelope.model_validate_json(raw)
        except (ValueError, TypeError, json.JSONDecodeError, ValidationError):
            # Malformed JSON, a schema-incompatible payload, or a Pydantic
            # validation failure is a cache miss: delete the corrupt entry
            # (best-effort) so it cannot keep breaking the endpoint, and fall
            # back to computing the forecast. ValidationError is caught
            # explicitly (its ValueError inheritance is not guaranteed across
            # Pydantic versions).
            self._delete(cache_key)
            return _CacheRead(
                hit=False,
                envelope=None,
                fallback_reason=FALLBACK_REASON_CORRUPT_CACHE_ENTRY,
            )
        return _CacheRead(hit=True, envelope=envelope)

    def compute_or_retrieve(
        self,
        db: Session,
        cache_key: str,
        query_params: str,
        compute: Callable[[], PointForecastEnvelope],
    ) -> PointForecastEnvelope:
        """Return a cached response or compute, store, and return it.

        When Redis is unavailable on either the read or write path, or the
        cached payload is corrupt, the response is computed directly and a
        best-effort ``point_query_fallback_audit`` row records the fallback
        with a reason. No point forecast is ever lost because of a Redis
        outage.

        Args:
            db: Database session used to record fallback audit rows.
            cache_key: Deterministic cache key for this request.
            query_params: The normalized query parameter string recorded on
                fallback.
            compute: Callable producing the response on a cache miss.

        Returns:
            The point forecast response.
        """
        read = self.get(cache_key)
        if read.hit and read.envelope is not None:
            return read.envelope

        response = compute()

        write_fallback = False
        try:
            self._client.setex(
                cache_key, self._ttl_seconds, response.model_dump_json()
            )
        except redis_lib.RedisError:
            # Redis is unreachable on the write path; the forecast response
            # is still returned.
            write_fallback = True

        if write_fallback and read.fallback_reason is not None:
            # Redis was unavailable on both the read and write paths for this
            # request. The audit row's cache_key is the primary key (one row
            # per request), so both failures are combined into one reason.
            self._record_fallback(
                db,
                cache_key,
                query_params,
                FALLBACK_REASON_REDIS_READ_AND_WRITE_UNAVAILABLE,
            )
        elif write_fallback:
            self._record_fallback(
                db, cache_key, query_params, FALLBACK_REASON_REDIS_WRITE_UNAVAILABLE
            )
        elif read.fallback_reason is not None:
            # The read fell back (Redis unavailable or corrupt entry); record
            # it so the audit ledger reflects the outage.
            self._record_fallback(db, cache_key, query_params, read.fallback_reason)

        return response

    def _delete(self, cache_key: str) -> None:
        """Best-effort deletion of a corrupt cache entry."""
        try:
            self._client.delete(cache_key)
        except redis_lib.RedisError:
            pass

    def _record_fallback(
        self,
        db: Session,
        cache_key: str,
        query_params: str,
        fallback_reason: str,
    ) -> None:
        """Record a PostgreSQL fallback audit row for a Redis outage.

        The audit row is best-effort metadata and is never a hard dependency
        for serving the forecast: if writing it fails (e.g. a database error),
        the error is swallowed so the forecast response is still returned.

        Args:
            db: Database session used to record the audit row.
            cache_key: The deterministic cache key of the request.
            query_params: The normalized query parameter string.
            fallback_reason: Why the request fell back to computing the
                forecast (Redis read/write unavailable, or corrupt entry).
        """
        try:
            now = datetime.now(timezone.utc)
            db.add(
                PointQueryFallbackAudit(
                    cache_key=cache_key,
                    query_params=query_params,
                    created_at=now,
                    expires_at=now + timedelta(seconds=self._ttl_seconds),
                    fallback_reason=fallback_reason,
                )
            )
            db.commit()
        except Exception:  # noqa: BLE001 - best-effort audit write must not fail the forecast
            db.rollback()


def build_point_cache_key(
    *,
    model: str,
    latitude: float,
    longitude: float,
    resolved_via: str,
    location_id: str | None,
    variables: tuple[str, ...] | None,
    units: str,
    start_lead_time_hours: int | None,
    end_lead_time_hours: int | None,
) -> str:
    """Build a deterministic cache key for a point forecast request.

    The key is a SHA-256 digest of a canonical JSON payload, so identical
    normalized requests always produce the same key. ``resolved_via`` and
    ``location_id`` together are a stable location identity: distinct spatial
    specifiers can resolve to the same coordinates but produce different
    payloads (``resolved_via``, ``elevation_m``), and two same-type records
    at the same coordinates can differ in elevation. Including the resolved
    record's ``id`` (or ``None`` for coordinate resolution) guarantees no two
    distinct payloads share a cache key.
    """
    payload = {
        "model": model,
        "latitude": latitude,
        "longitude": longitude,
        "resolved_via": resolved_via,
        "location_id": location_id,
        "variables": sorted(variables) if variables else None,
        "units": units,
        "start_lead_time_hours": start_lead_time_hours,
        "end_lead_time_hours": end_lead_time_hours,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "point:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
