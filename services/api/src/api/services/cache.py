"""Redis-primary response cache with a PostgreSQL fallback audit ledger.

Forecast responses are cached in Redis with a TTL aligned to the model
update cadence and the cache policy of the caching endpoint (``public,
max-age=1800`` for point forecasts and ensemble statistics, ``public,
max-age=3600`` for probabilities, per API.md). Redis is a best-effort
accelerator: when it is unavailable the response is computed directly and a
``point_query_fallback_audit`` row records the fallback (the table's purpose
per DATABASE.md section 2). The cache key is a deterministic canonical string
derived from the normalized request, so identical requests always map to the
same key.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar, cast

import redis as redis_lib
from pydantic import BaseModel, ValidationError
from redis import Redis
from sqlalchemy.orm import Session

from api.core.config import settings
from api.models.entities import PointQueryFallbackAudit
from api.schemas import PointForecastEnvelope

#: Envelope model type stored in the cache.
TEnvelope = TypeVar("TEnvelope", bound=BaseModel)

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
FALLBACK_REASON_REDIS_READ_AND_WRITE_UNAVAILABLE = "redis_read_and_write_unavailable"

#: Socket connect timeout in seconds for the Redis client. A stalled
#: connection (e.g. a network partition where the connect is accepted but
#: never answered) must not block a request thread indefinitely: the
#: fallback path only triggers on ``redis.RedisError``, and redis-py raises
#: ``TimeoutError`` (a ``RedisError`` subclass) when these timeouts fire.
REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
#: Socket read/command timeout in seconds for the Redis client.
REDIS_SOCKET_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class _CacheRead(Generic[TEnvelope]):
    """Result of reading the cache for a key.

    Attributes:
        hit: Whether a valid cached response was found.
        envelope: The cached response when ``hit`` is True, else ``None``.
        fallback_reason: Why the read fell back to computing, or ``None`` when
            the read was a clean miss (no entry) or a hit.
    """

    hit: bool
    envelope: TEnvelope | None
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
        # ``redis_lib.from_url`` is untyped in the redis stubs (no ``no-untyped-
        # call``-free alternative exists at this public boundary), so the single
        # untyped call is allowed and the concrete ``redis.Redis`` type is
        # asserted on the attribute so the client's methods are type-checked.
        self._client: Redis = redis_lib.from_url(  # type: ignore[no-untyped-call]
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
        )

    def get(
        self,
        cache_key: str,
        *,
        # The default's static type is widened to ``type[BaseModel]`` so mypy
        # accepts the concrete default without a type-ignore; callers pass the
        # concrete ``model_type`` explicitly and its value type is preserved.
        model_type: type[BaseModel] = PointForecastEnvelope,
    ) -> _CacheRead[TEnvelope]:
        """Read the cache for a key, reporting how the read resolved.

        Redis errors and unparseable cached payloads (malformed JSON or a
        schema that fails validation) are treated as cache misses so a
        transient Redis failure or a corrupt entry never fails the request. A
        corrupt entry is deleted so it cannot keep breaking the endpoint. The
        returned :class:`_CacheRead` distinguishes these cases so the caller
        can record a best-effort fallback audit event.

        Args:
            cache_key: The deterministic cache key.
            model_type: The envelope model the cached payload is validated
                against. Defaults to :class:`PointForecastEnvelope`.

        Returns:
            A ``_CacheRead`` describing whether the read was a hit, a clean
            miss, or a fallback (Redis unavailable / corrupt entry).
        """
        try:
            # The redis stub types ``.get`` as ``Awaitable[Any] | Any`` even
            # for a synchronous client; with ``decode_responses=True`` the
            # value is a ``str | None`` at runtime. ``cast`` narrows the union
            # so the payload is validated as JSON text.
            raw = cast(str | None, self._client.get(cache_key))
        except redis_lib.RedisError:
            return _CacheRead[TEnvelope](
                hit=False,
                envelope=None,
                fallback_reason=FALLBACK_REASON_REDIS_READ_UNAVAILABLE,
            )
        if raw is None:
            return _CacheRead[TEnvelope](hit=False, envelope=None)
        try:
            # ``model_type`` is widened to ``type[BaseModel]`` for the default;
            # the caller-supplied concrete type is preserved by the cast.
            envelope = cast(TEnvelope, model_type.model_validate_json(raw))
        except (ValueError, TypeError, json.JSONDecodeError, ValidationError):
            # Malformed JSON, a schema-incompatible payload, or a Pydantic
            # validation failure is a cache miss: delete the corrupt entry
            # (best-effort) so it cannot keep breaking the endpoint, and fall
            # back to computing the forecast. ValidationError is caught
            # explicitly (its ValueError inheritance is not guaranteed across
            # Pydantic versions).
            self._delete(cache_key)
            return _CacheRead[TEnvelope](
                hit=False,
                envelope=None,
                fallback_reason=FALLBACK_REASON_CORRUPT_CACHE_ENTRY,
            )
        return _CacheRead[TEnvelope](hit=True, envelope=envelope)

    def compute_or_retrieve(
        self,
        db: Session,
        cache_key: str,
        query_params: str,
        compute: Callable[[], TEnvelope],
        *,
        # See the note on ``get`` for why the default's static type is widened
        # to ``type[BaseModel]``.
        model_type: type[BaseModel] = PointForecastEnvelope,
    ) -> TEnvelope:
        """Return a cached response or compute, store, and return it.

        When Redis is unavailable on either the read or write path, or the
        cached payload is corrupt, the response is computed directly and a
        best-effort ``point_query_fallback_audit`` row records the fallback
        with a reason. No forecast is ever lost because of a Redis outage.

        Args:
            db: Database session used to record fallback audit rows.
            cache_key: Deterministic cache key for this request.
            query_params: The normalized query parameter string recorded on
                fallback.
            compute: Callable producing the response on a cache miss.
            model_type: The envelope model the cached payload is validated
                against. Defaults to :class:`PointForecastEnvelope`.

        Returns:
            The response.
        """
        read: _CacheRead[TEnvelope] = self.get(cache_key, model_type=model_type)
        if read.hit and read.envelope is not None:
            return read.envelope

        response = compute()

        write_fallback = False
        try:
            self._client.setex(cache_key, self._ttl_seconds, response.model_dump_json())
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


def build_probability_cache_key(
    *,
    model: str,
    latitude: float,
    longitude: float,
    variable: str,
    threshold: float,
    operator: str,
    lead_time_hours: int,
    threshold_max: float | None,
) -> str:
    """Build a deterministic cache key for a probability forecast request.

    The key is a SHA-256 digest of a canonical JSON payload, following the
    same convention as :func:`build_point_cache_key`. ``operator`` and both
    thresholds are included because they change the computed probability.
    """
    payload = {
        "model": model,
        "latitude": latitude,
        "longitude": longitude,
        "variable": variable,
        "threshold": threshold,
        "operator": operator,
        "lead_time_hours": lead_time_hours,
        "threshold_max": threshold_max,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "probability:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_ensemble_cache_key(
    *,
    model: str,
    latitude: float,
    longitude: float,
    variable: str,
    lead_time_hours: int,
    include_members: bool = False,
) -> str:
    """Build a deterministic cache key for an ensemble statistics request.

    The key is a SHA-256 digest of a canonical JSON payload, following the
    same convention as :func:`build_point_cache_key`. ``include_members`` is
    part of the key so a statistics-only cached response (which omits the
    member array) can never satisfy a distribution request, and a member-heavy
    response can never satisfy a statistics-only request.
    """
    payload = {
        "model": model,
        "latitude": latitude,
        "longitude": longitude,
        "variable": variable,
        "lead_time_hours": lead_time_hours,
        "include_members": include_members,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "ensemble:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
