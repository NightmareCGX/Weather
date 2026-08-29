"""PostgreSQL advisory-lock coordinator for the ingestion worker.

This module owns the PostgreSQL session-level advisory-lock lifecycle used by
the region-write concurrency protocol. The pure key derivation lives in
``domain.locks``; this module adds the PG-specific acquire/release,
transaction, timeout, cancellation, and Connection-invalidation behavior.

Lock roles (global order, deadlock-free):

    admission turnstile
        → store gate (SHARED for writers/readers, EXCLUSIVE for init/finalize)
        → sorted unique physical-region locks

The store gate uses **native** PostgreSQL shared/exclusive advisory locks on
the SAME key: ``pg_advisory_lock_shared``/``pg_advisory_unlock_shared`` for
SHARED and ``pg_advisory_lock``/``pg_advisory_unlock`` for EXCLUSIVE. There is
no shared→exclusive upgrade; a holder of SHARED fully releases it before
requesting EXCLUSIVE.

Session-level advisory locks survive COMMIT and are released only by the
matching unlock or by the physical session terminating (connection close,
crash, or invalidation). Every successful acquisition must therefore have a
matching unlock, verified against the database, in the caller's ``finally``.
If an unlock cannot be confirmed the affected physical Connection is
invalidated and never returned normally to the pool.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Connection

from domain.locks import admission_key, region_key, store_gate_key

logger = logging.getLogger(__name__)


class LockTimeoutError(RuntimeError):
    """Raised when an advisory-lock acquisition exceeds its deadline."""


class NonPostgresCoordinatorError(RuntimeError):
    """Raised when the lock coordinator is used against a non-PostgreSQL URL.

    PostgreSQL advisory locks are a PostgreSQL-only feature; the concurrency
    protocol is not safe on SQLite or other dialects.
    """


@dataclass
class _LockSet:
    """Tracks session-level advisory locks acquired on one Connection."""

    held: set[int] = field(default_factory=set)
    shared_held: set[int] = field(default_factory=set)

    def record_exclusive(self, key: int) -> None:
        self.held.add(key)

    def record_shared(self, key: int) -> None:
        self.shared_held.add(key)

    def drop_exclusive(self, key: int) -> None:
        self.held.discard(key)

    def drop_shared(self, key: int) -> None:
        self.shared_held.discard(key)

    def all_keys(self) -> set[int]:
        return self.held | self.shared_held


class StoreLockCoordinator:
    """Owns advisory-lock acquisition/release for one worker scope.

    A single coordinator is bound to one physical ``Connection`` (the
    one-worker/one-Connection model). It tracks acquired locks so partial
    failures can release every already-acquired lock before propagating.
    """

    def __init__(
        self,
        connection: Connection,
        *,
        store_path: str,
        endpoint: str | None = None,
        secure: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._connection = connection
        self._store_path = store_path
        self._endpoint = endpoint
        self._secure = secure
        self._timeout_seconds = timeout_seconds
        self._locks = _LockSet()
        self._lock = threading.Lock()
        self._owns_connection = False

    # ------------------------------------------------------------------
    # Key derivation (delegates to the pure shared module)
    # ------------------------------------------------------------------
    def store_gate(self) -> int:
        return store_gate_key(
            self._store_path, endpoint=self._endpoint, secure=self._secure
        )

    def admission(self) -> int:
        return admission_key(
            self._store_path, endpoint=self._endpoint, secure=self._secure
        )

    def region(self, region_id: str) -> int:
        return region_key(
            self._store_path,
            region_id,
            endpoint=self._endpoint,
            secure=self._secure,
        )

    # ------------------------------------------------------------------
    # Public acquire/release
    # ------------------------------------------------------------------
    def acquire_admission(self) -> None:
        """Acquire the EXCLUSIVE admission turnstile key for this store."""
        self._acquire_exclusive(self.admission())

    def release_admission(self) -> None:
        self._release_exclusive(self.admission())

    def acquire_shared_admission(self) -> None:
        self._acquire_shared(self.admission())

    def release_shared_admission(self) -> None:
        self._release_shared(self.admission())

    def acquire_shared_gate(self) -> None:
        self._acquire_shared(self.store_gate())

    def release_shared_gate(self) -> None:
        self._release_shared(self.store_gate())

    def acquire_exclusive_gate(self) -> None:
        self._acquire_exclusive(self.store_gate())

    def release_exclusive_gate(self) -> None:
        self._release_exclusive(self.store_gate())

    def acquire_region_locks(self, region_ids: list[str]) -> None:
        """Acquire sorted unique region locks in a single batched round-trip."""
        keys = sorted({self.region(r) for r in region_ids})
        self._acquire_batch_exclusive(keys)

    def release_region_locks(self, region_ids: list[str]) -> None:
        """Release sorted unique region locks in a single batched round-trip."""
        keys = sorted({self.region(r) for r in region_ids}, reverse=True)
        with self._lock:
            held = [k for k in keys if k in self._locks.held]
        self._release_batch_exclusive(held)

    def release_all(self) -> None:
        """Release every lock this coordinator holds (descending, best-effort)."""
        with self._lock:
            exclusive = sorted(self._locks.held, reverse=True)
            shared = sorted(self._locks.shared_held, reverse=True)
        if exclusive:
            self._release_batch_exclusive(exclusive)
        for key in shared:
            self._release_shared(key)

    # ------------------------------------------------------------------
    # Internal PG helpers
    # ------------------------------------------------------------------
    def _acquire_batch_exclusive(self, keys: list[int]) -> None:
        """Acquire a sorted batch of exclusive advisory locks in bounded round-trips.

        Uses non-blocking pg_try_advisory_lock() over the candidate keys in a single
        query, checks for complete acquisition, unlocks partial acquisitions if any key
        was unavailable, and retries until deadline.

        On any unhandled query or connection error where lock acquisition state is indeterminate,
        the physical connection is invalidated immediately so the backend drops any locks.
        """
        if not keys:
            return
        with self._lock:
            for k in keys:
                if k in self._locks.held or k in self._locks.shared_held:
                    raise RuntimeError(f"advisory lock {k} already held by this worker")

        deadline = time.monotonic() + self._timeout_seconds
        sorted_keys = sorted(keys)

        while True:
            acquired_keys: list[int] = []
            try:
                stmt = text(
                    "SELECT k, pg_try_advisory_lock(k) AS acquired "
                    "FROM unnest(CAST(:keys AS bigint[])) AS k"
                )
                results = self._connection.execute(stmt, {"keys": sorted_keys}).fetchall()
                all_acquired = True
                for row in results:
                    k_val = int(row[0])
                    acq = bool(row[1])
                    if acq:
                        acquired_keys.append(k_val)
                    else:
                        all_acquired = False

                if all_acquired:
                    with self._lock:
                        for k in acquired_keys:
                            self._locks.record_exclusive(k)
                    return

                # Partial acquisition: at least one key was held by another worker.
                # Must release all keys acquired in this attempt before sleeping.
                if acquired_keys:
                    self._release_batch_exclusive(acquired_keys)

            except Exception:
                if acquired_keys:
                    try:
                        self._release_batch_exclusive(acquired_keys)
                    except Exception:
                        pass
                logger.error(
                    "Error during batched advisory lock acquisition on %s; invalidating "
                    "physical Connection to guarantee no lock leaks",
                    self._store_path,
                )
                self._invalidate_connection()
                raise

            if time.monotonic() >= deadline:
                raise LockTimeoutError(
                    f"timed out acquiring {len(keys)} advisory locks on {self._store_path} "
                    f"after {self._timeout_seconds}s"
                )
            time.sleep(0.02)

    def _release_batch_exclusive(self, keys: list[int]) -> None:
        """Release a batch of exclusive advisory locks in a single SQL round-trip.

        Verifies that every lock returns true from pg_advisory_unlock. If any unlock
        returns false or raises, invalidates the physical connection.
        """
        if not keys:
            return
        sorted_keys = sorted(keys, reverse=True)
        ok = True
        try:
            stmt = text(
                "SELECT k, pg_advisory_unlock(k) AS released "
                "FROM unnest(CAST(:keys AS bigint[])) AS k"
            )
            results = self._connection.execute(stmt, {"keys": sorted_keys}).fetchall()
            for row in results:
                if not bool(row[1]):
                    ok = False
        except Exception:
            ok = False

        if not ok:
            logger.error(
                "Batched advisory unlock returned false or raised on %s; invalidating "
                "the physical Connection so no lock leaks into the pool",
                self._store_path,
            )
            self._invalidate_connection()

        with self._lock:
            for k in keys:
                self._locks.held.discard(k)

    def _acquire_exclusive(self, key: int) -> None:
        self._acquire(
            key, "pg_advisory_lock", "pg_try_advisory_lock", exclusive=True
        )

    def _acquire_shared(self, key: int) -> None:
        self._acquire(
            key,
            "pg_advisory_lock_shared",
            "pg_try_advisory_lock_shared",
            exclusive=False,
        )

    def _acquire(self, key: int, blocking: str, try_fn: str, *, exclusive: bool) -> None:
        with self._lock:
            already = key in self._locks.held or key in self._locks.shared_held
        if already:
            # Reentrant acquisition is not supported: the protocol acquires
            # each key once. Refuse rather than double-acquire (which would
            # require a second unlock).
            raise RuntimeError(f"advisory lock {key} already held by this worker")
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                self._connection.execute(text("BEGIN"))
                self._connection.execute(
                    text("SET LOCAL lock_timeout = :ms"),
                    {"ms": int(max(1, self._timeout_seconds * 1000))},
                )
                # Try non-blocking first so a cancelled request can bail out
                # between attempts without a long blocking wait.
                acquired = self._connection.execute(
                    text(f"SELECT {try_fn}(:key)"), {"key": key}
                ).scalar()
                if acquired:
                    self._connection.execute(text("COMMIT"))
                    with self._lock:
                        if exclusive:
                            self._locks.record_exclusive(key)
                        else:
                            self._locks.record_shared(key)
                    return
                self._connection.execute(text("ROLLBACK"))
            except Exception:
                self._connection.execute(text("ROLLBACK"))
                raise
            if time.monotonic() >= deadline:
                raise LockTimeoutError(
                    f"timed out acquiring advisory lock {key} after "
                    f"{self._timeout_seconds}s"
                )
            # Poll briefly. This is a bounded non-abandoning wait.
            time.sleep(0.02)

    def _release_exclusive(self, key: int) -> None:
        self._release(key, "pg_advisory_unlock")

    def _release_shared(self, key: int) -> None:
        self._release(key, "pg_advisory_unlock_shared")

    def _release(self, key: int, unlock_fn: str) -> None:
        with self._lock:
            in_exclusive = key in self._locks.held
            in_shared = key in self._locks.shared_held
        if not in_exclusive and not in_shared:
            return  # not held by this worker; nothing to unlock
        try:
            self._connection.execute(text("BEGIN"))
            ok = self._connection.execute(
                text(f"SELECT {unlock_fn}(:key)"), {"key": key}
            ).scalar()
            self._connection.execute(text("COMMIT"))
        except Exception:
            ok = False
        if not ok:
            # Unlock could not be confirmed; the physical session must be
            # dropped so its remaining locks die with it. Never return this
            # Connection normally to the pool.
            logger.error(
                "advisory unlock of %s returned false/raised on %s; invalidating "
                "the physical Connection so no lock leaks into the pool",
                key,
                self._store_path,
            )
            self._invalidate_connection()
        with self._lock:
            self._locks.held.discard(key)
            self._locks.shared_held.discard(key)

    def _invalidate_connection(self) -> None:
        """Invalidate the affected physical Connection (not the whole Engine)."""
        try:
            self._connection.invalidate()
        except Exception:  # noqa: BLE001 - invalidation is best-effort
            pass

    def close_connection(self) -> None:
        """Release all locks and return the Connection to the pool.

        If any lock remains held after release (unlock failed), the Connection
        is invalidated so the physical session (and its locks) is dropped.
        """
        self.release_all()
        with self._lock:
            remaining = self._locks.all_keys()
        if remaining:
            logger.error(
                "Connection returned to pool with %d residual advisory locks; "
                "invalidating instead",
                len(remaining),
            )
            self._invalidate_connection()
        self._connection.close()

    @property
    def held_keys(self) -> set[int]:
        """Return the keys currently held (for tests/diagnostics)."""
        with self._lock:
            return set(self._locks.all_keys())
