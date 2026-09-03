"""PostgreSQL advisory-lock leadership for the realtime scheduler (narrow).

One deployment-wide realtime scheduler may run at a time. Leadership is an
**optimization-only** guard: it prevents two schedulers from planning and
dispatching the same waves (duplicated downloads/writes), and it must never be
relied on for correctness — per-store region writes are already serialized by
the coordinator's own advisory-lock protocol, and every wave is reconciled
against durable state on the next poll.

A session-level advisory lock on the shared leader key
(:func:`domain.locks.scheduler_leader_key`) is held on a dedicated connection
for the scheduler's lifetime:

* process crash → the physical session dies → leadership is released
  naturally (no stale leader state, no manual cleanup);
* a second instance fails ``try_lock`` and clearly exits (or stays passive);
* no new locking model is introduced for big-batch execution — big-batch and
  realtime converge on the exact same per-store store gates.
"""

from __future__ import annotations

import logging
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from domain.locks import scheduler_leader_key

logger = logging.getLogger(__name__)


class LeadershipUnavailableError(RuntimeError):
    """Raised when leadership is requested on a non-PostgreSQL catalog.

    Session-level advisory locks are a PostgreSQL feature; the scheduler
    requires PostgreSQL for its catalog and its leadership alike (SQLite test
    harnesses inject a fake leadership instead).
    """


class SchedulerLeadership:
    """Session-level advisory-lock leadership on one dedicated connection."""

    def __init__(self, engine: Engine, *, identity: str | None = None) -> None:
        self._engine = engine
        self._key = scheduler_leader_key(identity or "")
        self._conn: Connection | None = None
        self._held = False

    @property
    def is_leader(self) -> bool:
        """Whether this instance currently holds leadership."""
        return self._held and self.check_leadership()

    def check_leadership(self) -> bool:
        """Verify that this instance still actively holds leadership on PostgreSQL.

        Executes a fast query on the dedicated connection against ``pg_locks``.
        If the connection died, was closed, or the lock is no longer held
        server-side, marks leadership as lost and returns False.
        """
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
                logger.warning(
                    "realtime leadership advisory lock is no longer held in session"
                )
                self._held = False
                return False
            return True
        except Exception as exc:
            logger.warning(
                "realtime leadership connection health check failed: %s", exc
            )
            self._held = False
            try:
                self._conn.invalidate()
            except Exception:
                pass
            self._conn = None
            return False

    def acquire(self) -> bool:
        """Try to become the leader (non-blocking).

        Returns:
            True when leadership was acquired; False when another instance
            holds it.

        Raises:
            LeadershipUnavailableError: On a non-PostgreSQL dialect.
        """
        if self._engine.dialect.name != "postgresql":
            raise LeadershipUnavailableError(
                "realtime scheduler leadership requires a PostgreSQL catalog"
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
            logger.info(
                "realtime leadership acquired (advisory key %s)", self._key
            )
            return True
        conn.close()
        return False

    def release(self) -> None:
        """Release leadership (best-effort; the session dying releases it too)."""
        if not self._held or self._conn is None:
            return
        try:
            released = self._conn.execute(
                text("SELECT pg_advisory_unlock(:key)"), {"key": self._key}
            ).scalar()
            if not released:
                logger.error(
                    "realtime leadership unlock returned false; invalidating "
                    "the connection so the lock dies with the session"
                )
                self._conn.invalidate()
            self._conn.close()
        except Exception:  # noqa: BLE001 - release is best-effort
            logger.exception("error while releasing realtime leadership")
        finally:
            self._conn = None
            self._held = False


class NoopLeadership:
    """Test/diagnostic stand-in that always "acquires" leadership."""

    def __init__(self) -> None:
        self.is_leader = False

    def acquire(self) -> bool:
        self.is_leader = True
        return True

    def release(self) -> None:
        self.is_leader = False

    def check_leadership(self) -> bool:
        return self.is_leader
