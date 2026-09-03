"""API reader-gate: participate in the SHARED store gate for forecast reads.

The API serving tier reads forecast Zarr stores. To never observe a store
mid-re-ingest, each store-reading endpoint:

    select candidate
        -> acquire SHARED admission
        -> acquire SHARED store gate
        -> fresh Core revalidation (run status, path, availability)
        -> generation-aware lazy-dataset reuse (Phase 2) + bounded selection
        -> materialize ONLY the selected subset under the gate
        -> release SHARED store gate + admission

The gate uses the **same** PostgreSQL advisory-lock key derivation as the
ingestion writer (via ``domain.locks``), so compliant API readers and
compliant ingestion writers coordinate on the same SHARED/EXCLUSIVE store gate.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from domain.locks import (
    admission_key,
    store_gate_key,
)


logger = logging.getLogger(__name__)


T = TypeVar("T")


class ReaderGateClosing(Exception):
    """Raised when a gated operation is attempted after shutdown began."""


class ReaderGateTimeout(Exception):
    """Raised when the reader gate or pool checkout exceeds its deadline."""


class ReaderGateShutdownTimeout(Exception):
    """Raised when shutdown times out with active gated handlers."""


class ReaderGateLifecycle:
    """Tracks active gated handlers and the closing flag for shutdown."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._closing = False
        self._active_count = 0

    def enter(self) -> None:
        with self._condition:
            if self._closing:
                raise ReaderGateClosing("server is shutting down")
            self._active_count += 1

    def exit(self) -> None:
        with self._condition:
            self._active_count -= 1
            self._condition.notify_all()

    def begin_shutdown(self) -> None:
        with self._condition:
            self._closing = True

    def wait_drained(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._active_count > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ReaderGateShutdownTimeout(self._active_count)
                self._condition.wait(timeout=remaining)

    @property
    def active_count(self) -> int:
        with self._condition:
            return self._active_count


class ReaderLockPool:
    """Dedicated reader-lock Engine/pool for the SHARED gate.

    One Connection per gated read; the SHARED advisory lock is held on it for
    the full materialization. Returns connections to the pool on release; a
    failed unlock invalidates the physical Connection.
    """

    def __init__(
        self,
        url: str,
        *,
        pool_size: int,
        max_overflow: int,
        pool_timeout: float,
    ) -> None:
        self._engine = create_engine(
            url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True,
        )

    def connect(self) -> Connection:
        return self._engine.connect()

    def dispose(self) -> None:
        self._engine.dispose()


class _ReaderGateSession:
    """One gated read: holds a reader-lock Connection and the SHARED gate."""

    def __init__(self, pool: ReaderLockPool, store_path: str) -> None:
        self._pool = pool
        self._store_path = store_path
        self._conn: Connection | None = None
        self._gate_held = False
        self._gate_key = store_gate_key(store_path)
        self._admission_key = admission_key(store_path)

    def _is_lock_timeout_error(self, exc: BaseException) -> bool:
        """Check if an exception is a PostgreSQL lock_timeout cancellation (55P03)."""
        orig = getattr(exc, "orig", None)
        if orig is not None:
            pgcode = getattr(orig, "pgcode", None)
            if pgcode == "55P03":
                return True
            sqlstate = getattr(orig, "sqlstate", None)
            if sqlstate == "55P03":
                return True
        exc_str = str(exc).lower()
        return (
            "55p03" in exc_str
            or "canceling statement due to lock timeout" in exc_str
            or "lock_not_available" in exc_str
        )

    def acquire(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        if time.monotonic() >= deadline:
            raise ReaderGateTimeout("reader gate timed out before checkout")
        self._conn = self._pool.connect()
        try:
            rem_admission = max(0.001, deadline - time.monotonic())
            self._acquire_shared(self._admission_key, rem_admission)
            try:
                rem_gate = max(0.001, deadline - time.monotonic())
                self._acquire_shared(self._gate_key, rem_gate)
                self._gate_held = True
            finally:
                self._release_shared(self._admission_key)
        except BaseException:
            if self._gate_held:
                self._release_shared(self._gate_key)
                self._gate_held = False
            self._close_conn()
            raise

    def _acquire_shared(self, key: int, timeout_seconds: float) -> None:
        assert self._conn is not None
        timeout_ms = int(max(1, timeout_seconds * 1000))
        try:
            with self._conn.begin():
                self._conn.execute(
                    text("SET LOCAL lock_timeout = :ms"),
                    {"ms": timeout_ms},
                )
                self._conn.execute(
                    text("SELECT pg_advisory_lock_shared(:key)"),
                    {"key": key},
                )
        except Exception as exc:
            if self._is_lock_timeout_error(exc):
                raise ReaderGateTimeout(
                    f"reader gate timed out acquiring SHARED lock on {self._store_path} "
                    f"after {timeout_seconds}s"
                ) from exc
            logger.error(
                "reader SHARED lock acquisition error on %s (key=%s); invalidating: %s",
                self._store_path,
                key,
                exc,
            )
            self._invalidate_conn()
            raise

    def _release_shared(self, key: int) -> None:
        if self._conn is None:
            return
        # Defensive rollback if an unhandled exception left a stray open transaction
        if self._conn.in_transaction():
            try:
                self._conn.rollback()
            except Exception:
                pass
        ok = True
        try:
            with self._conn.begin():
                result = self._conn.execute(
                    text("SELECT pg_advisory_unlock_shared(:key)"), {"key": key}
                ).scalar()
                if not bool(result):
                    ok = False
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Error releasing reader SHARED lock on %s (key=%s): %s",
                self._store_path,
                key,
                exc,
            )
            ok = False
        if not ok:
            logger.error(
                "reader SHARED unlock of %s failed; invalidating Connection", key
            )
            self._invalidate_conn()

    def _invalidate_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.invalidate()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    def _close_conn(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def revalidate(self, db_url: str) -> tuple[bool, str | None]:
        """Fresh Core revalidation of run status + path on the lock Connection.

        Returns ``(ok, path)``; when ``ok`` is False the caller falls back to
        the next candidate. Valid statuses for reading are 'ready', 'processing',
        and 'partial'. Runs in 'failed' status are rejected.
        """
        assert self._conn is not None
        row = None
        with self._conn.begin():
            row = self._conn.execute(
                text(
                    "SELECT status, zarr_store_path FROM model_runs "
                    "WHERE zarr_store_path = :path LIMIT 1"
                ),
                {"path": self._store_path},
            ).first()
        if row is None:
            return False, None
        status, path = row
        return (
            status in ("ready", "processing", "partial") and path == self._store_path
        ), path

    def release(self) -> None:
        if self._gate_held:
            self._release_shared(self._gate_key)
            self._gate_held = False
        self._close_conn()

    @property
    def connection(self) -> Connection | None:
        return self._conn


def gated_read_dataset_with_selector(
    store_path: str,
    selector: Callable[[Any], T],
) -> T:
    """Run a request-bounded selection on a forecast Zarr store under the gate.

    This is the **performance-correct read path** (Phase 1 remediation): the
    reader gate acquires the SHARED lock, then invokes ``selector`` on the
    **lazy** xarray dataset. The selector applies the request's bounded
    selection (lead/member/spatial window/point) and materializes **only** that
    selection (via ``.values`` / ``np.asarray`` on the bounded array) *inside*
    the gate, so all Zarr/S3 chunk I/O happens under the SHARED lock. The gate
    is released only after the selection is fully materialized; the returned
    value is in-memory (numpy-backed) with no lazy references escaping.

    Contract on ``selector``:

    * It receives the lazily-opened ``xr.Dataset`` and returns an **already
      materialized** in-memory value (a ``np.ndarray``, a small raw region, a
      point series, a member vector, or a small named tuple of such values).
    * It must not return a lazy/full xarray object: it must invoke ``.values``
      (or an equivalent read) on a *bounded* selection before returning.
    * It should never index/read the full 2-D ``(lat, lon)`` grid for a tile
      or point request.

    Args:
        store_path: The forecast Zarr store path (``s3://`` or local).
        selector: ``lambda ds: <already-materialized bounded result>``.

    Returns:
        The in-memory result produced by ``selector``.

    Raises:
        FileNotFoundError: If the run is no longer READY (revalidation fails).
    """
    from api.core.config import settings
    from api.core.zarr import open_serving_dataset

    def materialize_selected() -> T:
        # open_serving_dataset provides the dataset under the authoritative
        # serving freshness and ownership contract (reusing StoreHandleCache
        # when a trusted generation exists, and opening fresh + closing on
        # exit when uncached). The selector materializes only its bounded
        # selection before returning — all chunk I/O happens HERE, under the
        # SHARED gate.
        with open_serving_dataset(store_path) as ds:
            return selector(ds)

    try:
        from api.main import reader_lifecycle, reader_pool
    except ImportError:
        reader_lifecycle = None  # type: ignore[assignment]
        reader_pool = None  # type: ignore[assignment]
    if reader_pool is None or reader_lifecycle is None:
        return materialize_selected()
    try:
        return gated_read(
            reader_pool,
            reader_lifecycle,
            store_path=store_path,
            revalidate_db_url=str(settings.DATABASE_URL),
            materialize=materialize_selected,
            timeout_seconds=float(settings.API_READER_GATE_TIMEOUT_SECONDS),
        )
    except (FileNotFoundError, ReaderGateTimeout):
        raise FileNotFoundError(f"run {store_path!r} is not a ready, readable run")


def gated_read(
    pool: ReaderLockPool,
    lifecycle: ReaderGateLifecycle,
    *,
    store_path: str,
    revalidate_db_url: str,
    materialize: Callable[[], T],
    timeout_seconds: float = 30.0,
) -> T:
    """Run a fully materialized forecast read under the SHARED store gate.

    Args:
        pool: The reader-lock Connection pool.
        lifecycle: The shutdown lifecycle (active-op tracking).
        store_path: The forecast Zarr store path.
        revalidate_db_url: The DB URL for fresh Core revalidation.
        materialize: A callable that performs the Zarr open + full
            materialization (returns a fully materialized value). It must not
            return a lazy xarray/Zarr object.
        timeout_seconds: Overall operation deadline.

    Returns:
        The fully materialized result.

    Raises:
        ReaderGateTimeout: If the gate/pool checkout exceeds the deadline.
    """
    lifecycle.enter()
    try:
        session = _ReaderGateSession(pool, store_path)
        session.acquire(timeout_seconds)
        try:
            ok, path = session.revalidate(revalidate_db_url)
            if not ok:
                raise FileNotFoundError(
                    f"run {store_path!r} is not a ready, readable run"
                )
            return materialize()
        finally:
            session.release()
    finally:
        lifecycle.exit()
