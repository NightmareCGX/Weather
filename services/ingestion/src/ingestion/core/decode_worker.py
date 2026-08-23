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
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path

from xarray import Dataset


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

    ``max_workers`` aligns with the CLI's ``--concurrency``: the same bound
    that limits in-flight forecast-file operations also sizes the decode pool.

    Attributes:
        max_workers: The fixed pool size (decode worker process count).
    """

    max_workers: int = 4

    def __post_init__(self) -> None:
        self._executor: ProcessPoolExecutor | None = None

    @property
    def _pool(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def submit(self, path: str | Path) -> Future[Dataset]:
        """Submit one decode task to the persistent pool.

        The returned Future, when its ``result()`` is awaited, produces the
        decoded dataset or raises the worker's exception. A worker death
        (native abort) raises :class:`concurrent.futures.process.BrokenProcessPool`
        on the awaiting side.

        Args:
            path: Path to the GRIB2 file to decode.

        Returns:
            A ``concurrent.futures.Future`` producing the decoded dataset.
        """
        resolved = str(Path(path).resolve())
        return self._pool.submit(decode_forecast_file, resolved)

    def shutdown(self) -> None:
        """Shut down the persistent pool, joining worker processes.

        The caller must have already drained every in-flight future (a worker
        death is surfaced to the caller as ``BrokenProcessPool`` before
        shutdown).
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None