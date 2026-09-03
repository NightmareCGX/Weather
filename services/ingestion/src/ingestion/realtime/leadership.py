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
        return self._held

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
