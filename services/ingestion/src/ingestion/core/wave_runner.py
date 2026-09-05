"""Reusable ingestion wave executor (extracted from the CLI orchestration).

This module owns the *execution* of one ingestion wave — the download → decode
→ region-write → settled-lead publication → coalesced-finalization pipeline —
so that both the ``weather-ingest`` CLI (big-batch mode) and the future
Phase 5C realtime scheduler call the same code path. It is a mechanical
extraction of the former ``ingestion.cli`` private orchestration; the
coordinator protocol, concurrency stages, cancellation behavior, progress
reporting, and error propagation are unchanged.

The module also pins the Phase 5B wave-target / cycle-horizon split:

* **Wave targets** — :attr:`RunSpec.target_lead_time_hours` (and
  :attr:`RunSpec.members` for GEFS) are the leads/members *this invocation
  ingests*. They drive work-item planning, staging, progress reporting, and
  the settled-lead publication set.
* **Canonical cycle horizon** — :attr:`RunCatalogSpec.expected_lead_time_hours`
  (and ``expected_members``) is the complete lead/member set the cycle's store
  is expected to serve when fully ingested (the model/product contract from
  :mod:`domain.horizon`). It drives store pre-allocation and final run-status
  readiness, and is deliberately independent of any single invocation's
  targets. Repeated disjoint-target invocations accumulate into one cycle
  store; the run stays ``partial`` until the horizon is complete.

The wave runner refuses targets outside the canonical horizon (they could
never be committed into a horizon-pre-allocated store) and requires a
non-empty horizon, so the split cannot be silently lost inside the ingestion
path.

The catalog-session plumbing (``_catalog_session_factory``) stays an
injectable module global: tests replace it with an in-memory SQLite engine
instead of the configured PostgreSQL engine.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from domain.horizon import canonical_lead_time_hours
from ingestion.core.base import PredecessorState
from ingestion.core.catalog import RunCatalogSpec, VariableSpec
from ingestion.core.pipeline import (
    _apply_variable_mapping,
    _normalize_canonical_units,
    _normalize_cloud_cover_intervals,
    _normalize_precipitation_increments,
    _validate_requested_lead,
    _validate_requested_member,
)
from ingestion.providers.noaa.connector import NOAAConnector

#: Default platform surface-variable mapping for NOAA GFS/GEFS files. Each
#: entry maps the cfgrib-emitted variable name (the GRIB ``cfVarName``, which
#: for 2-metre temperature is ``t2m`` — not the GRIB ``shortName`` ``2t``) to
#: the platform ``code`` recorded in ``forecast_variables``. ``source_code``
#: must equal the *emitted* data-variable name so the pipeline's
#: ``_apply_variable_mapping`` can match it.
DEFAULT_VARIABLES: tuple[VariableSpec, ...] = (
    VariableSpec(
        code="temperature_2m",
        name="2-Meter Temperature",
        unit="°C",
        source_code="t2m",
    ),
    VariableSpec(
        code="precipitation_rate",
        name="Precipitation Rate",
        unit="mm/h",
        source_code="prate",
    ),
    VariableSpec(
        code="precipitation_amount_3h",
        name="3-Hour Precipitation Amount",
        unit="mm",
        source_code="tp",
    ),
    VariableSpec(
        code="crain",
        name="Categorical Rain Flag",
        unit="flag",
        source_code="crain",
    ),
    VariableSpec(
        code="csnow",
        name="Categorical Snow Flag",
        unit="flag",
        source_code="csnow",
    ),
    VariableSpec(
        code="cfrzr",
        name="Categorical Freezing Rain Flag",
        unit="flag",
        source_code="cfrzr",
    ),
    VariableSpec(
        code="cicep",
        name="Categorical Ice Pellets Flag",
        unit="flag",
        source_code="cicep",
    ),
    VariableSpec(
        code="relative_humidity_2m",
        name="2-Meter Relative Humidity",
        unit="%",
        source_code="r2",
    ),
    VariableSpec(
        code="wind_gust",
        name="Wind Gust",
        unit="km/h",
        source_code="gust",
    ),
    VariableSpec(
        code="visibility",
        name="Visibility",
        unit="m",
        source_code="vis",
    ),
    VariableSpec(
        code="snow_depth",
        name="Snow Depth",
        unit="m",
        source_code="sde",
    ),
    VariableSpec(
        code="wind_u_10m",
        name="10-Meter U Wind Component",
        unit="m/s",
        source_code="u10",
    ),
    VariableSpec(
        code="wind_v_10m",
        name="10-Meter V Wind Component",
        unit="m/s",
        source_code="v10",
    ),
    VariableSpec(
        code="cloud_cover_3h",
        name="3-Hour Cloud Cover",
        unit="%",
        source_code="tcc",
    ),
    VariableSpec(
        code="cloud_ceiling",
        name="Cloud Ceiling Height",
        unit="m",
        source_code="gh",
    ),
)

#: Center metadata keyed by ``center_id``.
_CENTER_METADATA: dict[str, tuple[str, str]] = {
    "noaa": ("National Oceanic and Atmospheric Administration", "USA"),
}

#: Model display metadata keyed by ``model_id``.
_MODEL_METADATA: dict[str, tuple[str, bool]] = {
    "gfs": ("Global Forecast System", False),
    "gefs": ("Global Ensemble Forecast System", True),
}

#: Grid metadata keyed by ``grid_id``.
_GRID_METADATA: dict[str, tuple[str, float]] = {
    "global_025deg": ("Global 0.25 Degree Grid", 25.0),
}


@dataclass(frozen=True)
class RunSpec:
    """One forecast-run specification: a model + cycle + its wave targets.

    A run spec describes *what this invocation ingests* (the wave targets),
    never the cycle's expected horizon — the horizon is the model/product
    contract (``domain.horizon``) carried by the derived
    :class:`RunCatalogSpec` (``expected_lead_time_hours`` /
    ``expected_members``). Keeping the two apart is what makes repeated
    disjoint-target invocations accumulate safely into one cycle store.

    Attributes:
        model: A model identifier (``gfs`` or ``gefs``).
        cycle_date: UTC date of the model run.
        cycle_hour: UTC cycle hour.
        target_lead_time_hours: The leads this invocation ingests (the wave
            targets). Must be a subset of the model's canonical horizon.
        members: GEFS perturbation member identities (``1..30``) to ingest in
            this invocation. Empty for deterministic models (GFS ingests all
            leads of the cycle store). Member identity is the real upstream
            number, never a positional completion index. The store's member
            axis is still pre-allocated with the full contract set gep01..gep30.
        store: Optional explicit store path (must match the identity unless
            ``allow_custom_store``).
        allow_custom_store: Whether a non-canonical ``store`` is accepted.
    """

    model: str
    cycle_date: date
    cycle_hour: int
    target_lead_time_hours: tuple[int, ...]
    members: tuple[int, ...] = ()
    store: str | None = None
    allow_custom_store: bool = False

    @property
    def cycle_time(self) -> datetime:
        """The UTC cycle time of this run."""
        return datetime(
            self.cycle_date.year,
            self.cycle_date.month,
            self.cycle_date.day,
            self.cycle_hour,
            tzinfo=timezone.utc,
        )


@dataclass(frozen=True)
class ConcurrencyPlan:
    """Effective concurrency and staging bounds for an ingestion wave.

    Decouples requested CLI concurrency into independently bounded resource stages:
    * ``download_concurrency``: Network I/O ceiling (NOMADS HTTP range GETs).
    * ``decode_concurrency``: CPU compute ceiling (ProcessPool ecCodes decoding).
    * ``write_concurrency``: Database & Storage I/O ceiling (PostgreSQL advisory
      locks + Zarr chunk writes + COMPLETE marker PUTs).
    * ``staging_concurrency``: Maximum total in-flight active/queued items in the
      pipeline, bounding peak resident decoded datasets in memory.
    """

    requested: int
    download_concurrency: int
    decode_concurrency: int
    write_concurrency: int
    staging_concurrency: int


def _detect_effective_cpus() -> int:
    """Affinity/cpuset-aware conservative CPU detection for decode worker sizing.

    Respects Linux process affinity (e.g. Docker/cgroup cpuset pinning via
    ``sched_getaffinity``) with fallback to ``os.cpu_count()``. Note: CFS
    bandwidth/quota limits may differ from cpuset affinity; ``MAX_DECODE_CONCURRENCY``
    provides an explicit safety ceiling. On Windows, enforces a hard ceiling of 61
    workers to stay within the 64-handle limit of ``_winapi.WaitForMultipleObjects``
    used by Python's ``ProcessPoolExecutor``.
    """
    import os
    import sys

    cpus: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            cpus = len(os.sched_getaffinity(0))
        except (NotImplementedError, OSError, AttributeError):
            pass
    if cpus is None or cpus < 1:
        cpus = os.cpu_count() or 1
    if sys.platform == "win32":
        cpus = min(cpus, 61)
    return max(1, cpus)


def _resolve_concurrency_plan(
    requested: int, settings: Any | None = None
) -> ConcurrencyPlan:
    """Derive decoupled stage capacities from requested CLI concurrency.

    Args:
        requested: Requested concurrency integer (from ``--concurrency``).
        settings: Optional ``IngestionSettings`` instance. Defaults to the
            global settings object.

    Returns:
        The resolved :class:`ConcurrencyPlan`.
    """
    if settings is None:
        from ingestion.core.config import settings as default_settings

        settings = default_settings

    req = max(1, requested)
    eff_cpus = _detect_effective_cpus()
    max_download = max(1, int(settings.MAX_DOWNLOAD_CONCURRENCY))
    max_decode = max(1, int(settings.MAX_DECODE_CONCURRENCY))
    max_write = max(1, int(settings.MAX_WRITE_CONCURRENCY))

    download = min(req, max_download)
    decode = min(req, eff_cpus, max_decode)
    write = min(req, max_write)
    staging = download + decode + write

    return ConcurrencyPlan(
        requested=req,
        download_concurrency=download,
        decode_concurrency=decode,
        write_concurrency=write,
        staging_concurrency=staging,
    )


def _destination_for(
    spec: RunSpec, staging_dir: Path, *, lead: int, member: int | None = None
) -> Path:
    """Return the staged download path for a (member,) lead file.

    The path encodes the model, cycle date, cycle hour, lead (and member) so
    distinct forecast runs never collide in the staging directory.

    Args:
        spec: The run spec.
        staging_dir: The run-scoped staging directory.
        lead: Forecast lead time.
        member: GEFS member identity, or ``None`` for deterministic.

    Returns:
        The staging path.
    """
    date_str = f"{spec.cycle_date:%Y%m%d}"
    if member is not None:
        name = (
            f"gep{member:02d}.{date_str}.t{spec.cycle_hour:02d}z.pgrb2s.0p25."
            f"f{lead:03d}"
        )
    else:
        name = f"{spec.model}.{date_str}.t{spec.cycle_hour:02d}z.pgrb2.0p25.f{lead:03d}"
    return staging_dir / f"{name}.grib2"


def _cleanup_sources(staging_dir: Path, destinations: list[Path] | set[Path]) -> None:
    """Delete successfully-ingested source files and their associated .idx cache files.

    Performs direct O(1) unlinks for primary source files and direct index files,
    followed by at most one single directory scan to unlink hash-based cfgrib index files
    matching the committed artifacts. Eliminates repeated O(N^2) directory globs.

    Deletion is best-effort post-commit resource reclamation. Failure to delete
    an already-committed artifact logs a warning and does not invalidate
    committed forecast data. Only filesystem errors (OSError) are caught.

    Args:
        staging_dir: Parent directory containing the staged files.
        destinations: Collection of staged GRIB2 file paths to remove.
    """
    import logging

    logger = logging.getLogger(__name__)
    if not destinations:
        return

    dest_set = set(destinations)
    committed_names = {d.name for d in dest_set}

    # 1. Direct O(1) unlinks for primary files and exact .idx files
    for destination in dest_set:
        try:
            destination.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to delete committed source artifact %s: %s; data is safe.",
                destination,
                exc,
            )

        direct_idx = Path(f"{destination}.idx")
        if direct_idx.name != destination.name:
            try:
                direct_idx.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Failed to delete direct index artifact %s: %s; data is safe.",
                    direct_idx,
                    exc,
                )

    # 2. Single-pass directory scan for cfgrib hash index files: <filename>.<hash>.idx
    try:
        if staging_dir.exists():
            for entry in staging_dir.iterdir():
                try:
                    if entry.name.endswith(".idx"):
                        no_ext = entry.name.removesuffix(".idx")
                        candidate = (
                            no_ext.rpartition(".")[0] if "." in no_ext else no_ext
                        )
                        if candidate in committed_names or no_ext in committed_names:
                            entry.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Failed to delete index artifact %s: %s; data is safe.",
                        entry,
                        exc,
                    )
    except OSError as exc:
        logger.warning(
            "Error scanning for index artifacts in %s: %s; data is safe.",
            staging_dir,
            exc,
        )


def _cleanup_source(destination: Path) -> None:
    """Delete a successfully-ingested source file and its associated .idx cache files.

    Deletion is best-effort post-commit resource reclamation. Failure to delete
    an already-committed artifact logs a warning and does not invalidate
    committed forecast data. Only filesystem errors (OSError) are caught.

    Args:
        destination: Path to the staged GRIB2 file to remove.
    """
    _cleanup_sources(destination.parent, [destination])


async def _run_wave(
    spec: RunSpec,
    args: Any,
    catalog_spec: RunCatalogSpec,
    store_path: str,
    concurrency: int,
    failures: list[str],
    cancel_event: threading.Event | None = None,
) -> str:
    """Download and ingest every lead/member target of a single forecast run.

    Outer resource-guarded entrypoint ensuring that all active S3 data-plane
    filesystems are deterministically drained on wave completion or any early
    failure (e.g. initialization, download, or decode exceptions).
    """
    from ingestion.core.s3 import close_wave_data_s3_fs

    try:
        return await _run_wave_impl(
            spec=spec,
            args=args,
            catalog_spec=catalog_spec,
            store_path=store_path,
            concurrency=concurrency,
            failures=failures,
            cancel_event=cancel_event,
        )
    finally:
        close_wave_data_s3_fs()


async def _run_wave_impl(
    spec: RunSpec,
    args: Any,
    catalog_spec: RunCatalogSpec,
    store_path: str,
    concurrency: int,
    failures: list[str],
    cancel_event: threading.Event | None = None,
) -> str:
    """Internal wave execution pipeline.

    Implements the approved decoupled pipeline architecture (Phase 1):
    - retained-seed fresh-store initialization;
    - one wave-level EXCLUSIVE pre-update (run -> partial + UPDATING markers);
    - seed worker starts immediately in write stage;
    - non-seed items:
        1. Staging envelope admission (staging_sem: bounds in-flight items / memory)
        2. Bounded network download (download_sem)
        3. Bounded process-isolated decode & parent normalization (decode_sem + DecodePool)
        4. Application-level write admission (write_sem: bounds DB pool & Zarr concurrency)
        5. Deferred DB connection checkout inside write executor critical section;
    - non-abandoning drain of all worker futures before finalization gate;
    - one coalesced finalization (EXCLUSIVE store gate) after all workers drain.

    Wave targets vs cycle horizon: the wave's work items come from
    ``spec.target_lead_time_hours`` (and ``spec.members``), while the store
    pre-allocation and the finalizer's readiness expectation come from
    ``catalog_spec.expected_lead_time_hours`` / ``expected_members`` (the
    canonical cycle horizon). Repeated disjoint-target invocations therefore
    accumulate into one cycle store and the run converges to ``ready`` only
    when the canonical horizon is complete.

    Args:
        spec: The forecast-run specification (wave targets).
        args: The CLI argument namespace (only ``download_dir``,
            ``no_progress``, ``keep_downloads``, and ``lock_timeout`` are read,
            via ``getattr`` with defaults, so non-CLI callers may pass any
            compatible namespace).
        catalog_spec: The run's catalog metadata; its ``expected_*`` fields
            carry the canonical cycle horizon.
        store_path: The cycle's Zarr store path.
        concurrency: Requested concurrency (resolved into stage capacities).
        failures: Mutable list collecting per-file failure descriptions.
        cancel_event: Optional external cancellation event (Phase 5C). When
            provided, the wave uses it instead of creating its own, so a
            caller (e.g. the realtime scheduler on shutdown) can trigger the
            existing non-abandoning drain. Defaults to ``None`` (the CLI
            behavior is unchanged: the wave owns its event).

    Returns:
        The finalizer's derived run status ('ready', 'partial', or 'processing').

    Raises:
        ValueError: If the catalog spec carries no canonical horizon, or a
            wave target lies outside the canonical horizon.
        CycleStoreMismatchError: If a lead's cycle mismatches the store.
        LeadTimeMismatchError: If a downloaded file's lead disagrees with the
            requested lead.
    """
    import logging
    import uuid
    from concurrent.futures import ThreadPoolExecutor

    from ingestion.core.cancel import await_all_workers_non_abandoning
    from ingestion.core.config import settings
    from ingestion.core.coordinator import (
        RunCoordinator,
        WaveRegion,
    )
    from ingestion.core.decode_worker import DecodePool
    from ingestion.core.observability import (
        PipelineProgressTracker,
        create_progress_renderer,
    )

    logger = logging.getLogger(__name__)

    # Wave-target / cycle-horizon split (Phase 5B). The horizon comes from the
    # catalog spec (the model/product contract); the targets come from the run
    # spec (this invocation). Refuse to run when either is inconsistent so the
    # distinction cannot be silently lost.
    horizon_leads = tuple(catalog_spec.expected_lead_time_hours)
    horizon_members = tuple(catalog_spec.expected_members)
    if not horizon_leads:
        raise ValueError(
            "catalog_spec.expected_lead_time_hours is empty: the wave runner "
            "requires the canonical cycle horizon for store pre-allocation and "
            "final readiness; wave targets are RunSpec.target_lead_time_hours."
        )
    outside_leads = sorted(set(spec.target_lead_time_hours) - set(horizon_leads))
    if outside_leads:
        raise ValueError(
            f"Wave target lead(s) {outside_leads} are outside the canonical "
            f"cycle horizon {min(horizon_leads)}..{max(horizon_leads)} for "
            f"model={spec.model!r}; they could never be committed into the "
            "horizon-pre-allocated cycle store."
        )
    outside_members = sorted(set(spec.members) - set(horizon_members))
    if outside_members:
        raise ValueError(
            f"Wave target member(s) {outside_members} are outside the required "
            f"member contract {horizon_members} for model={spec.model!r}."
        )

    # Phase 6D: Early ingestion admission guard against claimed/deleted cycles.
    from ingestion.core.base import CycleTombstonedError
    from ingestion.core.catalog import is_cycle_fenced_or_deleted

    with _catalog_session() as session:
        if is_cycle_fenced_or_deleted(session, spec.cycle_time):
            raise CycleTombstonedError(
                f"Refusing ingestion for cycle {spec.cycle_time.isoformat()}: "
                "cycle is claimed for deletion or already tombstoned."
            )

    run_tag = (
        f"staging_{spec.model}_{spec.cycle_date:%Y%m%d}_"
        f"{spec.cycle_hour:02d}z_{uuid.uuid4().hex}"
    )
    staging_dir = Path(args.download_dir) / run_tag
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Each (member, lead) work item, or just (lead) for deterministic.
    # Lead-major ordering for ensemble models enables early progressive publication per settled lead.
    if spec.model != "gefs":
        items: list[tuple[int | None, int]] = [
            (None, lead) for lead in sorted(spec.target_lead_time_hours)
        ]
    else:
        items = [
            (member, lead)
            for lead in sorted(spec.target_lead_time_hours)
            for member in sorted(spec.members)
        ]

    seed_item = items[0]
    seed_member, seed_lead = seed_item

    # Observability: tracker and live UI renderer
    no_progress = getattr(args, "no_progress", False)
    tracker = PipelineProgressTracker(
        model=spec.model,
        cycle_str=spec.cycle_time.strftime("%Y-%m-%d %H:%MZ"),
        total_items=len(items),
    )
    renderer = create_progress_renderer(tracker, no_progress=no_progress)
    renderer.start()
    tracker.record_milestone("run_start")

    ui_stop_event = asyncio.Event()

    async def _ui_update_loop() -> None:
        while not ui_stop_event.is_set():
            try:
                renderer.update()
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception:
                break

    ui_task = asyncio.create_task(_ui_update_loop())

    # Resolve decoupled stage capacities
    plan = _resolve_concurrency_plan(concurrency, settings)
    logger.info(
        "Starting wave: model=%s cycle=%s items=%d requested_concurrency=%d "
        "effective_concurrency=(download=%d, decode=%d, write=%d, staging=%d) "
        "db_pool=(size=%d, max_overflow=%d, timeout=%.1fs)",
        spec.model,
        spec.cycle_time,
        len(items),
        concurrency,
        plan.download_concurrency,
        plan.decode_concurrency,
        plan.write_concurrency,
        plan.staging_concurrency,
        int(settings.DB_POOL_SIZE),
        int(settings.DB_MAX_OVERFLOW),
        float(settings.DB_POOL_TIMEOUT_SECONDS),
    )

    coordinator = RunCoordinator(
        catalog_spec,
        store_path,
        timeout_seconds=float(getattr(args, "lock_timeout", 30.0)),
    )
    wave_cancel_event: threading.Event = (
        cancel_event if cancel_event is not None else threading.Event()
    )
    write_completed_events: dict[tuple[int | None, int], asyncio.Event] = {
        item: asyncio.Event() for item in items
    }
    decode_completed_events: dict[tuple[int | None, int], asyncio.Event] = {
        item: asyncio.Event() for item in items
    }
    predecessor_states: dict[tuple[int | None, int], PredecessorState] = {}
    predecessor_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=plan.write_concurrency)
    # The persistent decode pool: up to ``plan.decode_concurrency`` reusable worker
    # processes each holding independent cfgrib/ecCodes native state.
    decode_pool = DecodePool(
        max_workers=min(len(items), max(1, plan.decode_concurrency))
    )
    engine = _catalog_session_factory()

    var_codes = tuple(v.code for v in catalog_spec.variables)

    try:
        async with NOAAConnector() as connector:
            # 1. Retained seed. Download the seed first, then decode it in a
            #    worker process (the native ecCodes boundary).
            seed_dest = _destination_for(
                spec, staging_dir, lead=seed_lead, member=seed_member
            )
            Path(seed_dest).parent.mkdir(parents=True, exist_ok=True)
            tracker.set_init_phase("seed_download")
            tracker.record_milestone("seed_download_start")
            tracker.on_download_start(seed_member, seed_lead, is_seed=True)
            t_dl_start = time.monotonic()
            try:
                await connector.download(
                    spec.model,
                    spec.cycle_date,
                    spec.cycle_hour,
                    seed_lead,
                    seed_dest,
                    member=seed_member,
                    variables=var_codes,
                )
                tracker.record_milestone("seed_download_complete")
                tracker.on_download_complete(
                    seed_member,
                    seed_lead,
                    duration_ms=(time.monotonic() - t_dl_start) * 1000.0,
                )
            except Exception:
                tracker.on_download_failed(
                    seed_member,
                    seed_lead,
                    duration_ms=(time.monotonic() - t_dl_start) * 1000.0,
                )
                tracker.set_init_phase("failed")
                raise

            tracker.set_init_phase("seed_decode")
            tracker.record_milestone("seed_decode_start")
            tracker.on_decode_start(seed_member, seed_lead)
            t_dec_start = time.monotonic()
            try:
                seed_future = decode_pool.submit(seed_dest)
                seed_dataset = _decode_and_normalize(
                    seed_future, catalog_spec, store_path=store_path, member=seed_member
                )

                raw_precip_for_future = None
                if "tp" in seed_dataset.data_vars:
                    raw_precip_for_future = np.copy(seed_dataset["tp"].values)
                elif "precipitation_amount_3h" in seed_dataset.data_vars:
                    raw_precip_for_future = np.copy(seed_dataset["precipitation_amount_3h"].values)

                raw_cloud_for_future = None
                if "tcc" in seed_dataset.data_vars:
                    raw_cloud_for_future = np.copy(seed_dataset["tcc"].values)
                elif "cloud_cover_3h" in seed_dataset.data_vars:
                    raw_cloud_for_future = np.copy(seed_dataset["cloud_cover_3h"].values)

                _validate_requested_lead(seed_dataset, seed_lead)
                _validate_requested_member(seed_dataset, seed_member)

                if raw_precip_for_future is not None or raw_cloud_for_future is not None:
                    with predecessor_lock:
                        predecessor_states[seed_item] = PredecessorState(
                            precip_raw=raw_precip_for_future,
                            cloud_raw=raw_cloud_for_future,
                        )
                decode_completed_events[seed_item].set()

                tracker.record_milestone("seed_decode_complete")
                tracker.on_decode_complete(
                    seed_member,
                    seed_lead,
                    duration_ms=(time.monotonic() - t_dec_start) * 1000.0,
                )
            except Exception:
                decode_completed_events[seed_item].set()
                tracker.on_decode_failed(
                    seed_member,
                    seed_lead,
                    duration_ms=(time.monotonic() - t_dec_start) * 1000.0,
                )
                tracker.set_init_phase("failed")
                raise

            # 2. Determine run id + same-cycle.
            run_id: str | None = None
            is_same_cycle = False
            tracker.set_init_phase("catalog_init")
            tracker.record_milestone("catalog_init_start")
            with _catalog_session() as db:
                from ingestion.core.catalog import ModelRunRecord
                from sqlalchemy import select

                row = (
                    db.execute(
                        select(ModelRunRecord).where(
                            ModelRunRecord.zarr_store_path == store_path
                        )
                    )
                    .scalars()
                    .first()
                )
                if row is not None:
                    run_id = str(row.id)
                    is_same_cycle = True
            tracker.record_milestone("catalog_init_complete")

            # 3. Wave-level EXCLUSIVE pre-update (init + UPDATING markers).
            #    The store is pre-allocated with the canonical cycle horizon (not
            #    this wave's targets) so repeated disjoint-target waves always
            #    find their leads inside the pre-allocated coordinate axis.
            pre_conn = engine.connect()
            try:
                tracker.set_init_phase("initialize_run_store")
                coordinator.initialize_run_store(
                    pre_conn,
                    seed_dataset=seed_dataset,
                    expected_leads=horizon_leads,
                    expected_members=horizon_members,
                    run_id=run_id,
                    is_same_cycle=is_same_cycle,
                    observer=tracker,
                )
                regions = [
                    WaveRegion(
                        lead_time_hours=lead,
                        member=member,
                        generation=_new_generation(),
                    )
                    for member, lead in items
                ]
                coordinator.pre_update_wave(
                    pre_conn,
                    regions=regions,
                    run_id=run_id,
                    is_same_cycle=is_same_cycle,
                    executor=executor,
                    cancel_event=wave_cancel_event,
                    observer=tracker,
                )
                tracker.set_init_phase("store_ready")
                tracker.record_milestone("store_ready")
            except Exception:
                tracker.set_init_phase("failed")
                raise
            finally:
                pre_conn.close()

            # 4. Decoupled stage semaphores (Phase 1):
            # - download_sem: at most `plan.download_concurrency` active HTTP downloads;
            # - decode_sem: at most `plan.decode_concurrency` active decode jobs;
            # - write_sem: at most `plan.write_concurrency` active DB/Zarr writes;
            # - staging_sem: at most `plan.staging_concurrency` in-flight pipeline admissions
            #   (bounding peak resident decoded datasets in memory).
            download_sem = asyncio.Semaphore(plan.download_concurrency)
            decode_sem = asyncio.Semaphore(plan.decode_concurrency)
            write_sem = asyncio.Semaphore(plan.write_concurrency)
            staging_sem = asyncio.Semaphore(plan.staging_concurrency)
            futures_lock = threading.Lock()
            registered_worker_futures: list[asyncio.Future[Any]] = []
            pipeline_tasks: list[asyncio.Task[Any]] = []

            generation_by_region = {r.region_id: r.generation for r in regions}

            # Synchronous write execution: checks out DB connection only for the
            # coordinated critical section (advisory locks + Zarr write + COMPLETE marker).
            def _run_region_write(
                dataset: xr.Dataset, member: int | None, lead: int, generation: str
            ) -> None:
                worker_conn = engine.connect()
                try:
                    coordinator.write_region_worker(
                        worker_conn,
                        dataset=dataset,
                        member=member,
                        generation=generation,
                        expected_leads=horizon_leads,
                        expected_members=horizon_members,
                    )
                finally:
                    worker_conn.close()

            # Retained-seed writer uses the retained dataset (no re-parse).
            def _run_seed_region() -> None:
                region_id = _region_id_for(seed_lead, seed_member)
                generation = generation_by_region.get(region_id)
                if generation is None:
                    raise RuntimeError(f"no generation for region {region_id}")
                _run_region_write(seed_dataset, seed_member, seed_lead, generation)

            loop = asyncio.get_event_loop()

            # Track pending tasks per lead for intermediate settled-lead publication
            expected_members_for_lead = spec.members if spec.members else (None,)
            lead_pending: dict[int, set[int | None]] = {
                lead_val: set(expected_members_for_lead)
                for lead_val in spec.target_lead_time_hours
            }
            lead_settle_lock = threading.Lock()
            published_leads: set[int] = set()
            run_id_for_pub = _resolve_run_id(catalog_spec, store_path)

            def _check_and_publish_lead(lead_val: int) -> None:
                if lead_val in published_leads:
                    return
                published_leads.add(lead_val)
                pub_conn = engine.connect()
                try:
                    coordinator.publish_settled_lead(
                        pub_conn,
                        run_id=run_id_for_pub,
                        spec=catalog_spec,
                        lead_time_hours=lead_val,
                        expected_members=spec.members,
                    )
                except Exception as exc:
                    logger.warning("Settled-lead publication failed for lead %d: %s", lead_val, exc)
                finally:
                    pub_conn.close()

            def _on_item_settled(member_val: int | None, lead_val: int) -> None:
                with lead_settle_lock:
                    if lead_val in lead_pending:
                        lead_pending[lead_val].discard(member_val)
                        if not lead_pending[lead_val]:
                            _check_and_publish_lead(lead_val)

            # Seed task: starts immediately after pre-update under write_sem admission
            async def _run_seed_task() -> None:
                if wave_cancel_event.is_set():
                    return
                async with write_sem:
                    if wave_cancel_event.is_set():
                        return
                    tracker.record_milestone("seed_write_start")
                    tracker.on_write_start(seed_member, seed_lead, is_seed=True)
                    t_wr_start = time.monotonic()
                    fut = loop.run_in_executor(executor, _run_seed_region)
                    with futures_lock:
                        registered_worker_futures.append(fut)

                    cancel_requested = False
                    while not fut.done():
                        try:
                            await asyncio.shield(fut)
                        except asyncio.CancelledError:
                            cancel_requested = True
                            wave_cancel_event.set()
                            continue
                        except Exception:
                            break

                    try:
                        fut.result()
                        wr_dur = (time.monotonic() - t_wr_start) * 1000.0
                        tracker.record_milestone("seed_write_complete")
                        tracker.on_write_complete(
                            seed_member, seed_lead, duration_ms=wr_dur
                        )
                        write_completed_events[seed_item].set()
                        _on_item_settled(seed_member, seed_lead)
                    except Exception as exc:  # noqa: BLE001 - report failure
                        wr_dur = (time.monotonic() - t_wr_start) * 1000.0
                        tracker.on_write_failed(
                            seed_member, seed_lead, duration_ms=wr_dur
                        )
                        failures.append(
                            f"{spec.model} member={seed_member} lead={seed_lead}: {exc}"
                        )
                        write_completed_events[seed_item].set()
                        _on_item_settled(seed_member, seed_lead)

                    if cancel_requested:
                        raise asyncio.CancelledError

            pipeline_tasks.append(asyncio.create_task(_run_seed_task()))

            # Non-seed pipeline tasks:
            # bounded download -> bounded decode & parent normalize -> bounded write admission -> write
            async def _pipeline_item(member: int | None, lead: int) -> None:
                if wave_cancel_event.is_set():
                    return
                dest = _destination_for(spec, staging_dir, lead=lead, member=member)

                # Stage 1: Pipeline admission (bounds total in-flight active/queued work)
                async with staging_sem:
                    if wave_cancel_event.is_set():
                        return

                    # Stage 2: Bounded download
                    async with download_sem:
                        if wave_cancel_event.is_set():
                            return
                        tracker.on_download_start(member, lead)
                        t_dl_start = time.monotonic()
                        try:
                            await connector.download(
                                spec.model,
                                spec.cycle_date,
                                spec.cycle_hour,
                                lead,
                                dest,
                                member=member,
                                variables=var_codes,
                            )
                            dl_dur = (time.monotonic() - t_dl_start) * 1000.0
                            tracker.on_download_complete(
                                member, lead, duration_ms=dl_dur
                            )
                        except Exception as exc:  # noqa: BLE001 - report download failure
                            dl_dur = (time.monotonic() - t_dl_start) * 1000.0
                            tracker.on_download_failed(
                                member, lead, duration_ms=dl_dur
                            )
                            failures.append(
                                f"{spec.model} member={member} lead={lead} download: {exc}"
                            )
                            decode_completed_events[(member, lead)].set()
                            write_completed_events[(member, lead)].set()
                            _on_item_settled(member, lead)
                            return

                    # Predecessor coordination for 6h-reset leads requiring de-accumulation / reconstruction.
                    # Waiting occurs OUTSIDE and BEFORE acquiring decode_sem to prevent semaphore inversion.
                    if (
                        lead % 6 == 0
                        and lead > 0
                        and any(
                            v.code in ("precipitation_amount_3h", "cloud_cover_3h")
                            for v in catalog_spec.variables
                        )
                    ):
                        pred_item = (member, lead - 3)
                        if pred_item in decode_completed_events:
                            await decode_completed_events[pred_item].wait()
                            if wave_cancel_event.is_set():
                                return

                    # Stage 3: Bounded decode (ProcessPool execution + parent normalization)
                    # ZERO DB connections checked out during this compute-intensive phase.
                    if wave_cancel_event.is_set():
                        return
                    ds: xr.Dataset | None = None
                    async with decode_sem:
                        if wave_cancel_event.is_set():
                            return
                        tracker.on_decode_start(member, lead)
                        t_dec_start = time.monotonic()
                        decode_fut = decode_pool.submit(dest)
                        try:
                            # Retrieve and consume predecessor raw state if this is a 6h reset lead
                            pred_precip = None
                            pred_cloud = None
                            if lead % 6 == 0 and lead > 0:
                                pred_item = (member, lead - 3)
                                with predecessor_lock:
                                    pred_state = predecessor_states.pop(pred_item, None)
                                if pred_state is not None:
                                    pred_precip = pred_state.precip_raw
                                    pred_cloud = pred_state.cloud_raw

                            ds = _decode_and_normalize(
                                decode_fut,
                                catalog_spec,
                                store_path=store_path,
                                predecessor_array=pred_precip,
                                predecessor_cloud_array=pred_cloud,
                                member=member,
                            )
                            _validate_requested_lead(ds, lead)
                            _validate_requested_member(ds, member)

                            # Store raw arrays for future dependent leads
                            raw_precip_for_future = None
                            if "tp" in ds.data_vars:
                                raw_precip_for_future = np.copy(ds["tp"].values)
                            elif "precipitation_amount_3h" in ds.data_vars:
                                raw_precip_for_future = np.copy(ds["precipitation_amount_3h"].values)

                            raw_cloud_for_future = None
                            if "tcc" in ds.data_vars:
                                raw_cloud_for_future = np.copy(ds["tcc"].values)
                            elif "cloud_cover_3h" in ds.data_vars:
                                raw_cloud_for_future = np.copy(ds["cloud_cover_3h"].values)

                            if raw_precip_for_future is not None or raw_cloud_for_future is not None:
                                with predecessor_lock:
                                    predecessor_states[(member, lead)] = PredecessorState(
                                        precip_raw=raw_precip_for_future,
                                        cloud_raw=raw_cloud_for_future,
                                    )

                            dec_dur = (time.monotonic() - t_dec_start) * 1000.0
                            tracker.on_decode_complete(
                                member, lead, duration_ms=dec_dur
                            )
                            decode_completed_events[(member, lead)].set()
                        except Exception as exc:  # noqa: BLE001 - report decode failure
                            dec_dur = (time.monotonic() - t_dec_start) * 1000.0
                            tracker.on_decode_failed(
                                member, lead, duration_ms=dec_dur
                            )
                            failures.append(
                                f"{spec.model} member={member} lead={lead} decode: {exc}"
                            )
                            decode_completed_events[(member, lead)].set()
                            write_completed_events[(member, lead)].set()
                            _on_item_settled(member, lead)
                            return

                    # Stage 4: Bounded write admission (application-level backpressure BEFORE thread submission)
                    if wave_cancel_event.is_set():
                        return
                    region_id = _region_id_for(lead, member)
                    generation = generation_by_region.get(region_id)
                    if generation is None:
                        failures.append(
                            f"{spec.model} member={member} lead={lead}: no generation for region {region_id}"
                        )
                        return

                    async with write_sem:
                        if wave_cancel_event.is_set():
                            return
                        tracker.on_write_start(member, lead)
                        t_wr_start = time.monotonic()
                        assert ds is not None
                        worker_fut = loop.run_in_executor(
                            executor, _run_region_write, ds, member, lead, generation
                        )
                        with futures_lock:
                            registered_worker_futures.append(worker_fut)

                        # Stage 5: Non-abandoning worker wait
                        cancel_requested = False
                        while not worker_fut.done():
                            try:
                                await asyncio.shield(worker_fut)
                            except asyncio.CancelledError:
                                cancel_requested = True
                                wave_cancel_event.set()
                                continue
                            except Exception:
                                break

                        try:
                            worker_fut.result()
                            wr_dur = (time.monotonic() - t_wr_start) * 1000.0
                            tracker.on_write_complete(
                                member, lead, duration_ms=wr_dur
                            )
                            write_completed_events[(member, lead)].set()
                            _on_item_settled(member, lead)
                        except Exception as exc:  # noqa: BLE001 - report write failure
                            wr_dur = (time.monotonic() - t_wr_start) * 1000.0
                            tracker.on_write_failed(
                                member, lead, duration_ms=wr_dur
                            )
                            failures.append(
                                f"{spec.model} member={member} lead={lead} write: {exc}"
                            )
                            write_completed_events[(member, lead)].set()
                            _on_item_settled(member, lead)
                        finally:
                            # Drop local dataset reference so memory is freed promptly
                            ds = None

                        if cancel_requested:
                            raise asyncio.CancelledError

            tracker.record_milestone("wave_tasks_created")
            for member, lead in items:
                if (member, lead) != seed_item:
                    pipeline_tasks.append(
                        asyncio.create_task(_pipeline_item(member, lead))
                    )

            # 5. Aggregate drain: wait for all outer pipeline tasks
            tracker.record_milestone("post_write_task_gather_start")
            results, cancelled = await await_all_workers_non_abandoning(
                pipeline_tasks, wave_cancel_event
            )
            tracker.record_milestone("post_write_task_gather_complete")
            for res in results:
                if isinstance(res, BaseException) and not isinstance(
                    res, asyncio.CancelledError
                ):
                    msg = str(res)
                    if not any(msg in f for f in failures):
                        failures.append(msg)

            # 6. Finalization gate: Verify that 100% of underlying worker futures are genuinely settled
            for fut in registered_worker_futures:
                if not fut.done():
                    raise RuntimeError(
                        "Finalization gate invariant violated: active executor worker detected"
                    )

                tracker.record_milestone("download_client_close_start")
        tracker.record_milestone("download_client_close_complete")

        # 7. Coalesced finalization (after all worker Futures drained).
        #    Readiness is evaluated against the canonical cycle horizon, not this
        #    wave's targets, so the run converges to 'ready' only when the whole
        #    horizon is committed.
        tracker.on_finalize_start()
        t_fin_start = time.monotonic()
        fin_conn = engine.connect()
        try:
            run_id = _resolve_run_id(catalog_spec, store_path)
            finalize_result = coordinator.finalize_run(
                fin_conn,
                run_id=run_id,
                spec=catalog_spec,
                expected_leads=horizon_leads,
                expected_members=horizon_members,
                observer=tracker,
            )
            status = finalize_result.status
            fin_dur = (time.monotonic() - t_fin_start) * 1000.0
            tracker.on_finalize_complete(duration_ms=fin_dur)

            # Post-finalization cleanup: clean up only regions proven committed
            # by THIS wave's generation, unless --keep-downloads is set.
            if not getattr(args, "keep_downloads", False):
                committed_dests: list[Path] = []
                for r in regions:
                    committed_gen = finalize_result.committed_regions.get(r.region_id)
                    if committed_gen is not None and committed_gen == r.generation:
                        dest = _destination_for(
                            spec, staging_dir, lead=r.lead_time_hours, member=r.member
                        )
                        committed_dests.append(dest)
                if committed_dests:
                    _cleanup_sources(staging_dir, committed_dests)
                # Best-effort removal of the owned staging directory.
                try:
                    staging_dir.rmdir()
                except OSError as exc:
                    logger.warning(
                        "Failed to remove staging directory %s: %s; data is safe.",
                        staging_dir,
                        exc,
                    )
        except Exception:
            fin_dur = (time.monotonic() - t_fin_start) * 1000.0
            tracker.on_finalize_failed(duration_ms=fin_dur)
            raise
        finally:
            fin_conn.close()
    finally:
        engine.dispose()
        executor.shutdown(wait=True)
        decode_pool.shutdown()
        ui_stop_event.set()
        ui_task.cancel()
        renderer.stop()

    # Emit and print final startup timeline report
    report = tracker.timeline.format_report(
        model=spec.model,
        cycle_str=spec.cycle_time.strftime("%Y-%m-%d %H:%MZ"),
        total_items=len(items),
    )
    if not no_progress:
        print("\n" + report)
    logger.info("Startup timeline breakdown:\n%s", report)

    if cancelled:
        raise asyncio.CancelledError

    return status


def _decode_and_normalize(
    future: "concurrent.futures.Future[xr.Dataset]",
    catalog_spec: RunCatalogSpec,
    *,
    store_path: str | None = None,
    predecessor_array: Any | None = None,
    predecessor_cloud_array: Any | None = None,
    member: int | None = None,
) -> xr.Dataset:
    """Await a decode worker result and normalize it in the parent process.

    The GRIB decode itself happened inside an isolated decode worker process
    (the native ecCodes boundary). Here the parent receives the raw-normalized
    dataset — transported via pickling — and applies the pure-numpy platform
    normalization that must stay in the orchestrator: precipitation accumulation
    de-accumulation, variable-name mapping to the platform vocabulary,
    canonical-unit conversion, and the model-id attribute. A worker process that
    died during decode (a native ecCodes abort) surfaces here as
    ``concurrent.futures.process.BrokenProcessPool`` (its ``result()`` raises),
    which the caller records as a per-file failure — the parent stays alive and
    the region is never committed.

    Args:
        future: The decode-pool future for the staged GRIB2 file.
        catalog_spec: The run's catalog metadata (variable specs + model id).
        store_path: Optional store path for predecessor lookup at 6h leads.
        predecessor_array: Optional explicit predecessor 2D array for precipitation.
        predecessor_cloud_array: Optional explicit predecessor 2D array for cloud cover.
        member: Optional member identity for ensemble predecessor lookup.

    Returns:
        The mapped, canonical-unit, model-tagged dataset.

    Raises:
        BaseException: The decode worker's exception (or ``BrokenProcessPool``
            when a worker process died), propagated to the caller's failure
            accounting.
    """
    ds = future.result()
    ds = _normalize_precipitation_increments(
        ds,
        catalog_spec.variables,
        store_path=store_path,
        predecessor_array=predecessor_array,
        member=member,
    )
    cloud_pred = (
        predecessor_cloud_array
        if predecessor_cloud_array is not None
        else predecessor_array
    )
    ds = _normalize_cloud_cover_intervals(
        ds,
        catalog_spec.variables,
        store_path=store_path,
        predecessor_array=cloud_pred,
        member=member,
    )
    ds = _apply_variable_mapping(ds, catalog_spec.variables)
    ds = _normalize_canonical_units(ds, catalog_spec.variables)
    ds.attrs["model_id"] = catalog_spec.model_id
    return ds


def _region_id_for(lead: int, member: int | None) -> str:
    from domain.locks import logical_region_encoding

    return logical_region_encoding(lead_time_hours=lead, member=member)


def _new_generation() -> str:
    import uuid

    return uuid.uuid4().hex


def _catalog_session() -> Session:
    """Open a catalog session using the injectable session factory.

    Tests monkeypatch ``_catalog_session_factory`` to route catalog writes to
    an in-memory SQLite database instead of the configured PostgreSQL engine.
    """
    return Session(bind=_catalog_session_factory())


def _default_catalog_engine() -> "Engine":
    """Return the configured ingestion catalog engine."""
    from ingestion.core.db import engine

    return engine


#: Injectable engine factory for the wave runner's catalog access. Production
#: returns the configured ingestion engine; tests replace this with an
#: in-memory SQLite engine so the coordinator path can be exercised without PG.
_catalog_session_factory = _default_catalog_engine


def _resolve_run_id(spec: RunCatalogSpec, store_path: str) -> str:
    """Resolve the run id for a store path (create it if absent)."""
    from sqlalchemy import select

    from ingestion.core.catalog import ModelRunRecord, record_run

    with _catalog_session() as db:
        row = (
            db.execute(
                select(ModelRunRecord).where(
                    ModelRunRecord.zarr_store_path == store_path
                )
            )
            .scalars()
            .first()
        )
        if row is not None:
            return str(row.id)
        # Fresh run: create the catalog rows (processing status).
        ds = _synthetic_spec_dataset(spec)
        run = record_run(db, spec, ds)
        return str(run.id)


def _synthetic_spec_dataset(spec: RunCatalogSpec) -> "xr.Dataset":
    """Build a minimal dataset for catalog row creation when no file is retained."""
    import numpy as np

    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    lead = spec.expected_lead_time_hours[0] if spec.expected_lead_time_hours else 6
    return xr.Dataset(
        data_vars={
            v.code: (
                ("lead_time_hours", "latitude", "longitude"),
                np.full((1, 4, 4), np.nan, dtype=np.float32),
            )
            for v in spec.variables
        },
        coords={
            "lead_time_hours": [lead],
            "latitude": lat,
            "longitude": lon,
        },
        attrs={"model_id": spec.model_id, "cycle_time": spec.cycle_time.isoformat()},
    )


def _build_spec(
    spec: RunSpec,
    args: Any,
    store_path: str,
) -> RunCatalogSpec:
    """Build the run's catalog metadata from a run spec + CLI defaults.

    The returned spec carries the **canonical cycle horizon** (the model's
    complete lead sequence from ``domain.horizon`` plus, for GEFS, the full
    gep01..gep30 member contract) as its ``expected_*`` expectation — never
    the invocation's wave targets. Store pre-allocation and final run-status
    readiness are evaluated against this horizon; the invocation's targets
    remain on ``RunSpec.target_lead_time_hours`` / ``RunSpec.members``.
    """
    center_name, center_country = _CENTER_METADATA[args.center_id]
    model_name, is_ensemble = _MODEL_METADATA[spec.model]
    grid_name, grid_resolution_km = _GRID_METADATA.get(
        args.grid_id, (args.grid_id, 0.0)
    )
    variables = tuple(args.variable) if args.variable is not None else DEFAULT_VARIABLES
    # Canonical cycle horizon (model/product contract): the complete lead
    # sequence the cycle store is expected to serve when fully ingested, and
    # for GEFS the full required perturbation set gep01..gep30. Wave targets
    # stay on the RunSpec; see the wave-target / cycle-horizon split.
    expected_lead_time_hours = canonical_lead_time_hours(spec.model)
    expected_members = tuple(range(1, 31)) if spec.model == "gefs" else ()
    return RunCatalogSpec(
        center_id=args.center_id,
        center_name=center_name,
        center_country=center_country,
        model_id=spec.model,
        model_name=model_name,
        is_ensemble=is_ensemble,
        resolution_km=grid_resolution_km,
        version_string=args.version_string,
        cycle_time=spec.cycle_time,
        grid_id=args.grid_id,
        grid_name=grid_name,
        grid_resolution_km=grid_resolution_km,
        product_type="surface",
        zarr_store_path=store_path,
        variables=variables,
        expected_lead_time_hours=expected_lead_time_hours,
        expected_members=expected_members,
    )
