"""PostgreSQL advisory-lock leadership for the GC orchestrator daemon.

One deployment-wide GC orchestrator may run at a time. Leadership is an
**optimization-only** guard: it prevents two GC workers from attempting the
same physical object deletions concurrently, and it must never be relied on
for data correctness — per-store physical deletions are serialized by the
coordinator's EXCLUSIVE store gate.

A session-level advisory lock on the shared leader key
(:func:`domain.locks.gc_leader_key`) is held on a dedicated connection for
the GC daemon's lifetime:
* process crash -> the physical session dies -> leadership is released naturally;
* a second instance fails ``try_lock`` and exits cleanly or remains passive;
* leadership is distinct and independent from realtime scheduler leadership.
"""

from __future__ import annotations

import logging
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from domain.locks import GC_LEADER_IDENTITY, gc_leader_key

logger = logging.getLogger(__name__)


class GcLeadershipUnavailableError(RuntimeError):
    """Raised when GC leadership is requested on a non-PostgreSQL catalog."""


class GcLeadership:
    """Session-level advisory-lock leadership on one dedicated PostgreSQL connection."""

    def __init__(self, engine: Engine, *, identity: str | None = None) -> None:
        self._engine = engine
        self._key = gc_leader_key(identity or GC_LEADER_IDENTITY)
        self._conn: Connection | None = None
        self._held = False

    @property
    def is_leader(self) -> bool:
        """Whether this instance currently holds leadership."""
        return self._held and self.check_leadership()

    def check_leadership(self) -> bool:
        """Verify that this instance still actively holds leadership on PostgreSQL."""
        if not self._held or self._conn is None:
            return False
        if self._conn.closed or self._conn.invalidated:
            self._held = False
            return False
        try:
            held = self._conn.execute(
                text(
                    "SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                    "AND pid = pg_backend_pid() "
                    "AND ((classid::bigint << 32) | (objid::bigint & 4294967295)) = :key"
                ),
                {"key": self._key},
            ).scalar()
            if not held:
                logger.warning("GC leadership advisory lock is no longer held in session")
                self._held = False
                return False
            return True
        except Exception as exc:
            logger.warning("GC leadership connection health check failed: %s", exc)
            self._held = False
            try:
                self._conn.invalidate()
            except Exception:
                pass
            self._conn = None
            return False

    def acquire(self) -> bool:
        """Try to become the GC leader (non-blocking).

        Returns:
            True when leadership was acquired; False when another instance holds it.

        Raises:
            GcLeadershipUnavailableError: On a non-PostgreSQL dialect.
        """
        if self._engine.dialect.name != "postgresql":
            raise GcLeadershipUnavailableError(
                "GC orchestrator leadership requires a PostgreSQL catalog"
            )
        conn = self._engine.connect()
        try:
            acquired = conn.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": self._key}
            ).scalar()
        except BaseException:
            conn.close()
            raise
        if acquired:
            self._conn = conn
            self._held = True
            logger.info("GC leadership acquired on key %s", self._key)
            return True
        conn.close()
        return False

    def release(self) -> None:
        """Release leadership and close the dedicated connection."""
        if not self._held or self._conn is None:
            return
        try:
            if not self._conn.closed and not self._conn.invalidated:
                self._conn.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": self._key}
                )
        except Exception as exc:
            logger.warning("Error releasing GC leadership lock: %s", exc)
        finally:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._held = False


class NoopGcLeadership:
    """Fake GC leadership for tests that always succeeds."""

    def __init__(self, *, is_leader: bool = True) -> None:
        self._is_leader = is_leader

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    def check_leadership(self) -> bool:
        return self._is_leader

    def acquire(self) -> bool:
        return self._is_leader

    def release(self) -> None:
        self._is_leader = False
