"""Process-isolated GRIB2 decode boundary for the ingestion CLI.

The native ecCodes C library (used by ``cfgrib``) is **not thread-safe**: no
``threading.Lock`` guards the decode path (verified in
``cfgrib.xarray_store``/``cfgrib.dataset``/``cfgrib.messages`` and the
``eccodes`` high-level binding), and concurrent decode calls from multiple
threads of one OS process can corrupt ecCodes native state. Observable
failures include ``fatal flex scanner internal error--end of buffer missed``
and ``ecCodes assertion failed`` — native aborts below normal Python exception
handling.

This module isolates the **native decode boundary** inside dedicated worker
processes. A persistent, fixed-size :class:`ProcessPoolExecutor` reuses a
bounded set of worker processes (no per-file spawn), each with its own
independent Python interpreter + cfgrib + ecCodes native state. The parent
process owns everything after decode: variable mapping, unit normalization,
lead-time validation, region-write concurrency (advisory locks, markers,
Zarr), catalog reconciliation, and finalization.

Process-boundary contract (Windows spawn-safe):

* the worker entrypoint :func:`decode_forecast_file` is module-top-level and
  importable — never a closure;
* the input contract is a small pickle-serializable argument list (file path
  and decode parameters);
* the result is a raw-normalized ``xarray.Dataset`` (decode + normalize only);
  no ``VariableSpec`` mapping or unit conversion runs inside the worker (those
  are pure numpy/xarray and stay in the parent);
* no SQLAlchemy engine, MinIO client, advisory-lock object, Zarr synchronizer,
  or session crosses the boundary.

Crash handling: a worker that dies (a native ecCodes abort) surfaces as
:class:`BrokenProcessPool` on the parent side. The parent survives, the
affected forecast file is reported failed, and the region is never committed
(no COMPLETE marker is ever written, so the run cannot be marked READY for it).

Hang handling (the 2026-10-28 incident): a worker that *stalls* — a wedged
native decode, or the inherited-lock fork deadlock this module's original
un-pinned ``fork`` start method was exposed to — used to freeze the parent
forever (``future.result()`` with no timeout). The pool now pins the
``spawn`` start method on every platform (fork of a multi-threaded process is
documented by CPython as deadlock-prone and is deprecated in 3.12) and bounds
every decode with :attr:`DecodePool.decode_timeout_seconds`: a timed-out
decode terminates the pool's workers and surfaces as
:class:`BrokenProcessPool`, exactly like a crash, so the existing per-file /
per-wave failure accounting absorbs it and the pool is rebuilt lazily on the
next submit. ``shutdown`` is bounded as well: stalled workers are terminated
instead of joined forever.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

from xarray import Dataset

logger = logging.getLogger(__name__)

#: Grace period (seconds) a shutdown waits for worker processes to exit on
#: their own before terminating the survivors.
_SHUTDOWN_GRACE_SECONDS = 5.0


def decode_forecast_file(file_path: str) -> Dataset:
    """Decode a GRIB2 file to a raw-normalized dataset (worker-process decode).

    The single entrypoint the persistent process pool executes. It must remain
    module-top-level, importable, and pickle-safe (no closure), so Windows
    ``spawn`` re-imports it in every worker process.

    Args:
        file_path: Absolute path to the staged GRIB2 file.

    Returns:
        The raw-normalized ``xarray.Dataset`` (decode + normalize; the platform
        variable mapping and unit conversion are applied by the parent).

    Raises:
        Exception: The parser's ``GribParsingError`` (or any ecCodes/native
            failure) propagates; a native abort kills the worker process and
            surfaces to the parent as ``BrokenProcessPool``.
    """
    # Test-only hook: spawned children inherit the parent's environment, so a
    # test can simulate a *hung* decode (the failure mode the pool timeout
    # exists for) without monkeypatching — patched state does NOT cross the
    # spawn boundary. Production leaves it unset and pays one ``os.environ``
    # lookup per decode.
    delay = float(os.environ.get("WEATHER_INGEST_TEST_DECODE_DELAY_SECONDS", "0") or 0.0)
    if delay > 0.0:
        time.sleep(delay)

    # Imported lazily so this module never forces cfgrib/ecCodes to load in the
    # PARENT process (the parent needs no native decode state).
    from ingestion.providers.noaa.parser import parse_grib2

    return parse_grib2(file_path)


@dataclass
class DecodePool:
    """A persistent fixed-size process pool for GRIB decode isolation.

    The pool holds up to ``max_workers`` reusable decode worker processes for
    its lifetime. Each worker process owns an independent Python interpreter +
    cfgrib + ecCodes native state, so submitting a decode task to the pool never
    re-spawns Python/ecCodes per file.

    The pool pins the ``spawn`` start method on every platform: the worker
    entrypoint and argument contract are spawn-safe by design, and ``fork``
    (the Linux default) copies the whole multi-threaded parent — locks,
    sockets, asyncio state — into the child, which is the documented recipe
    for the inherited-lock deadlock behind the 2026-10-28 outage.

    ``max_workers`` aligns with the CLI's ``--concurrency``: the same bound
    that limits in-flight forecast-file operations also sizes the decode pool.

    Attributes:
        max_workers: The fixed pool size (decode worker process count).
        decode_timeout_seconds: Wall-clock budget for one decode. ``None``
            (the default) disables the bound; the production wave runner
            always passes ``settings.DECODE_TIMEOUT_SECONDS``. A timed-out
            decode terminates the pool and surfaces as ``BrokenProcessPool``.
    """

    max_workers: int = 4
    decode_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        self._executor: ProcessPoolExecutor | None = None
        self._mp_context = mp.get_context("spawn")

    @property
    def start_method(self) -> str:
        """The pinned multiprocessing start method (always ``spawn``)."""
        return self._mp_context.get_start_method()

    @property
    def _pool(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=self._mp_context,
            )
        return self._executor

    def submit(self, path: str | Path) -> Future[Dataset]:
        """Submit one decode task to the persistent pool.

        The returned Future, when its ``result()`` is awaited, produces the
        decoded dataset or raises the worker's exception. A worker death
        (native abort) raises :class:`BrokenProcessPool` on the awaiting side.

        Prefer :meth:`result` for awaiting: it adds the bounded-timeout hang
        protection on top of the raw future.

        Args:
            path: Path to the GRIB2 file to decode.

        Returns:
            A ``concurrent.futures.Future`` producing the decoded dataset.
        """
        resolved = str(Path(path).resolve())
        return self._pool.submit(decode_forecast_file, resolved)

    def result(self, future: Future[Dataset]) -> Dataset:
        """Await one decode future under the pool's bounded timeout.

        A decode that exceeds :attr:`decode_timeout_seconds` cannot be
        interrupted cooperatively (the worker is wedged inside native code by
        definition), so the whole pool is terminated and rebuilt lazily on the
        next :meth:`submit`, and this call raises :class:`BrokenProcessPool` —
        the exact signal a worker *crash* produces, so the callers' existing
        failure accounting (per-file on the pipeline path, wave-fatal on the
        seed path) absorbs the timeout without new semantics.

        Blast radius: terminating the pool also kills any other in-flight
        decode in the same pool. In the wave runner the decode semaphore caps
        in-flight decodes at ``decode_concurrency`` (== ``max_workers``), so at
        most the sibling workers' current files are affected; they fail as
        ``BrokenProcessPool`` and are retried by reconciliation.

        Args:
            future: A future returned by :meth:`submit`.

        Returns:
            The decoded dataset.

        Raises:
            BrokenProcessPool: When the worker died, or when the decode
                exceeded the configured timeout.
            Exception: The worker's own exception, propagated unchanged.
        """
        timeout = self.decode_timeout_seconds
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.error(
                "decode worker stalled for %.1fs (timeout=%.1fs); terminating "
                "the decode pool (max_workers=%d) so the wave can continue",
                timeout if timeout is not None else float("inf"),
                timeout if timeout is not None else float("inf"),
                self.max_workers,
            )
            self._terminate_pool()
            raise BrokenProcessPool(
                f"decode exceeded decode_timeout_seconds={timeout}; decode "
                "pool workers were terminated and the pool will be rebuilt "
                "on the next submit"
            ) from None

    def _terminate_pool(self) -> None:
        """Terminate every worker immediately; the pool is rebuilt lazily."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        # Snapshot the worker handles BEFORE shutdown: the executor's manager
        # thread clears (and can None out) ``_processes`` concurrently, and a
        # terminated/killed Process object stays joinable afterwards.
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=False)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except (ValueError, OSError):  # already reaped / pid recycled
                continue

    def shutdown(self) -> None:
        """Shut down the persistent pool with a *bounded* worker join.

        The caller must have already drained every in-flight future (a worker
        death is surfaced to the caller as ``BrokenProcessPool`` before
        shutdown). Workers are given :data:`_SHUTDOWN_GRACE_SECONDS` to exit
        on their own; survivors (stalled decodes, wedged native state) are
        terminated so a hung worker can never make ``shutdown`` — and with it
        the wave's ``finally`` block, and with that the whole scheduler —
        block forever.
        """
        executor, self._executor = self._executor, None
        if executor is None:
            return
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=False)
        deadline = time.monotonic() + _SHUTDOWN_GRACE_SECONDS
        for process in processes:
            try:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=_SHUTDOWN_GRACE_SECONDS)
            except (ValueError, OSError):  # already reaped / pid recycled
                continue