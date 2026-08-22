"""Command-line production entrypoint for the ingestion worker.

Runs the full ingestion flow for one or more NOAA forecast runs:

    download GRIB2 (NOMADS)  ->  parse GRIB2  ->  write Zarr  ->  record catalog (ready)

A *forecast run* is a ``(model, cycle)`` pair — one ``model_runs`` row
(``UNIQUE(model_version_id, cycle_time)`` per DATABASE.md) whose Zarr store
accumulates every lead. NOMADS serves one GRIB2 file per lead, so each lead of
a cycle is downloaded and merged into that cycle's store.

Batch semantics (ACCEPTANCE_REMEDIATION_PLAN §7): an invocation describes a
*set of forecast-run specifications*, never a Cartesian product.

* A single run spec is ``--model X --cycle-date D --cycle-hour H`` plus one or
  more ``--lead-time-hours``.
* ``--model``, ``--cycle-date``, ``--cycle-hour``, and ``--lead-time-hours``
  are repeatable. When more than one model/date/hour is given, they are zipped
  into aligned run specs (never broadcast across each other).
* ``--manifest`` supplies an explicit list of run specs for complex jobs.
* ``--dry-run`` prints the resolved run specs without downloading/writing.

The store path is derived from the forecast identity
(``s3://weather-data/{model}/{cycle_date}/{cycle_hour}/cycle.zarr``); a
supplied ``--store`` may not silently contradict it.

The requested ``--lead-time-hours`` is used to build the download URL. After
the file is parsed, ingestion fails fast if the file's decoded lead differs
from the requested lead (the file is the source of truth and is never
relabeled), so a stale or mislabeled upstream file aborts instead of silently
ingesting an unexpected lead.

Usage:

    # Single run, many leads (store derived):
    python -m ingestion.cli ingest --model gfs --cycle-date 2026-07-21 \\
        --cycle-hour 0 --lead-time-hours 0 6 12 18

    # Two cycles of one model (store derived per cycle):
    python -m ingestion.cli ingest --model gfs --cycle-date 2026-07-21 \\
        --cycle-hour 0 12 --lead-time-hours 0 6 12 18

    # Explicit multi-run manifest:
    python -m ingestion.cli ingest --manifest ingest-manifest.json

Or via the installed console script (see ``services/ingestion/pyproject.toml``):

    weather-ingest ingest --model gefs --cycle-date 2026-07-21 \\
        --cycle-hour 0 --lead-time-hours 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import xarray as xr
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from ingestion.core.base import CycleStoreMismatchError, LeadTimeMismatchError
from ingestion.core.catalog import RunCatalogSpec, VariableSpec
from ingestion.core.pipeline import (
    _apply_variable_mapping,
    _normalize_canonical_units,
    _validate_requested_lead,
)
from ingestion.providers.noaa.connector import NOAAConnector

#: Supported NOAA model identifiers (NOMADS GFS/GEFS).
SUPPORTED_MODELS = ("gfs", "gefs")

#: Zarr store path template for a forecast cycle. One ``model_runs`` row is one
#: cycle; the store path is a pure function of the forecast identity
#: (model + cycle date + cycle hour), so the same cycle always maps to the same
#: store and different cycles can never collide (ACCEPTANCE_REMEDIATION_PLAN §5).
STORE_PATH_TEMPLATE = (
    "s3://weather-data/{model}/{cycle_date:%Y-%m-%d}/{cycle_hour:02d}/cycle.zarr"
)


def derive_store_path(model: str, cycle_date: date, cycle_hour: int) -> str:
    """Return the canonical Zarr store path for a forecast cycle.

    The path separates model, cycle date, and cycle hour, matching the
    documented convention (``s3://weather-data/gfs/2026-07-21/00/cycle.zarr``).
    This is the single source of truth for store-path construction; callers
    must not duplicate the layout logic.

    Args:
        model: A model identifier (``gfs`` or ``gefs``).
        cycle_date: UTC date of the model run.
        cycle_hour: UTC cycle hour.

    Returns:
        The canonical ``s3://`` store path for the cycle.
    """
    return STORE_PATH_TEMPLATE.format(
        model=model,
        cycle_date=cycle_date,
        cycle_hour=cycle_hour,
    )


def validate_store_path(
    store: str | None,
    model: str,
    cycle_date: date,
    cycle_hour: int,
    *,
    allow_custom_store: bool = False,
) -> str:
    """Validate (or derive) the store path for a forecast cycle.

    The approved storage layout derives the store path from the forecast
    identity. An explicitly supplied ``--store`` may not silently contradict
    ``model`` / ``cycle date`` / ``cycle hour``:

    * When ``store`` is ``None``, the canonical path is derived.
    * When ``store`` is supplied and differs from the canonical path, it is
      rejected unless ``allow_custom_store`` is true (an explicit override).

    This is a fail-fast path-level check; the Zarr identity guard in
    ``_merge_lead`` remains the authoritative cross-cycle protection.

    Args:
        store: The ``--store`` value, or ``None`` to derive.
        model: A model identifier.
        cycle_date: UTC date of the model run.
        cycle_hour: UTC cycle hour.
        allow_custom_store: Whether a supplied path that differs from the
            canonical layout is accepted.

    Returns:
        The store path to use for the cycle.

    Raises:
        ValueError: If a supplied store contradicts the forecast identity and
            ``allow_custom_store`` is false.
    """
    canonical = derive_store_path(model, cycle_date, cycle_hour)
    if store is None:
        return canonical
    if store == canonical:
        return store
    if allow_custom_store:
        return store
    raise ValueError(
        f"Store path {store!r} does not match the forecast identity "
        f"(model={model}, cycle={cycle_date}T{cycle_hour:02d}Z). Expected "
        f"{canonical!r}. Pass --allow-custom-store to override."
    )

@dataclass(frozen=True)
class RunSpec:
    """One forecast-run specification: a model + cycle + its leads/members.

    Attributes:
        model: A model identifier (``gfs`` or ``gefs``).
        cycle_date: UTC date of the model run.
        cycle_hour: UTC cycle hour.
        lead_time_hours: The leads to ingest for this run.
        members: GEFS perturbation member identities (``1..30``) to ingest.
            Empty for deterministic models (GFS ingests all leads of the cycle
            store). Member identity is the real upstream number, never a
            positional completion index.
        store: Optional explicit store path (must match the identity unless
            ``allow_custom_store``).
        allow_custom_store: Whether a non-canonical ``store`` is accepted.
    """

    model: str
    cycle_date: date
    cycle_hour: int
    lead_time_hours: tuple[int, ...]
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


def _as_list(value: Any) -> list[Any]:
    """Coerce a CLI nargs value to a flat list.

    With ``nargs="+"`` each flag occurrence is a list; a single flag can carry
    multiple values (``--model gfs gefs``). ``None`` (a flag not supplied)
    becomes an empty list so the batch expansion can distinguish "not given"
    from "given".
    """
    if value is None:
        return []
    if isinstance(value, list):
        flat: list[Any] = []
        for item in value:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [value]


def expand_run_specs(
    args: argparse.Namespace,
) -> list[RunSpec]:
    """Expand CLI arguments into a list of run specifications.

    This is the anti-Cartesian core: model/date/hour lists are zipped into
    aligned run specs, and each run carries the full lead list. If the caller
    provides a manifest, it is the authoritative source and the flag lists are
    ignored. A single model/date/hour with multiple leads is "all leads of that
    one cycle" — not a product across models.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The resolved list of run specs (one per model×cycle).

    Raises:
        ValueError: If no run is derivable from the flags, or the aligned
            model/date/hour lists have unequal lengths, or a manifest is
            malformed.
    """
    if getattr(args, "manifest", None) is not None:
        return _parse_manifest(args.manifest)
    models = _as_list(args.model)
    dates = _as_list(args.cycle_date)
    hours = _as_list(args.cycle_hour)
    leads = tuple(_as_list(args.lead_time_hours))
    members = tuple(_as_list(getattr(args, "member", None)))
    if not models or not dates or not hours or not leads:
        raise ValueError(
            "At least one --model, --cycle-date, --cycle-hour, and "
            "--lead-time-hours is required."
        )
    # A GEFS run without an explicit member list defaults to the full
    # perturbation set gep01..gep30 (member identity preserved per file).
    if "gefs" in models and not members:
        members = tuple(range(1, 31))
    # Aligned-zipped expansion: the number of (model, date, hour) triples must
    # match unless a single value is broadcast across the others. A single
    # model/date/hour broadcasts; multiple values must align 1:1.
    triples = _align_triples(models, dates, hours)
    return [
        RunSpec(
            model=model,
            cycle_date=cycle_date,
            cycle_hour=cycle_hour,
            lead_time_hours=leads,
            members=members,
            store=args.store,
            allow_custom_store=args.allow_custom_store,
        )
        for model, cycle_date, cycle_hour in triples
    ]


def _align_triples(
    models: list[Any],
    dates: list[Any],
    hours: list[Any],
) -> list[tuple[str, date, int]]:
    """Zip model/date/hour lists into aligned triples.

    A list of length 1 is broadcast; multiple values must have matching
    lengths (or be a mix of 1 and N). Anything else is ambiguous and rejected
    to prevent an accidental Cartesian product.

    Raises:
        ValueError: If the lists cannot be aligned.
    """
    lengths = {len(models), len(dates), len(hours)} - {1}
    if len(lengths) > 1:
        raise ValueError(
            "Multiple --model/--cycle-date/--cycle-hour values must align "
            "1:1 (or be a mix of 1 and N); refusing to guess a Cartesian "
            "product."
        )
    n = max(len(models), len(dates), len(hours))

    def _pick(values: list[Any]) -> list[Any]:
        return values * n if len(values) == 1 else values

    return list(zip(_pick(models), _pick(dates), _pick(hours)))


def _parse_manifest(path: str) -> list[RunSpec]:
    """Parse a manifest JSON file into run specifications.

    The manifest schema is an explicit ``runs`` list:

    .. code-block:: json

        {
          "runs": [
            {"model": "gfs", "cycle_date": "2026-08-13",
             "cycle_hour": "00", "lead_time_hours": [0, 6, 12, 18]}
          ]
        }

    Args:
        path: Path to the manifest file.

    Returns:
        The parsed run specs.

    Raises:
        ValueError: If the manifest is missing, malformed, or contains an
            invalid spec.
    """
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read manifest {path!r}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("runs"), list):
        raise ValueError(
            "Manifest must be a JSON object with a 'runs' list of "
            "{model, cycle_date, cycle_hour, lead_time_hours} objects."
        )
    specs: list[RunSpec] = []
    for entry in raw["runs"]:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest run must be a JSON object.")
        required = ("model", "cycle_date", "cycle_hour", "lead_time_hours")
        missing = [key for key in required if key not in entry]
        if missing:
            raise ValueError(
                f"Manifest run is missing required key(s): {', '.join(missing)}."
            )
        if entry["model"] not in SUPPORTED_MODELS:
            raise ValueError(
                f"Manifest model {entry['model']!r} is not supported."
            )
        try:
            cycle_date = date.fromisoformat(str(entry["cycle_date"]))
            cycle_hour = int(entry["cycle_hour"])
            leads = tuple(int(lead) for lead in entry["lead_time_hours"])
            raw_members = entry.get("members", [])
            members = tuple(int(m) for m in raw_members)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid manifest run: {entry!r}") from exc
        if cycle_hour not in (0, 6, 12, 18) or not leads:
            raise ValueError(f"Invalid manifest run: {entry!r}")
        # GEFS defaults to the full perturbation set when members is omitted.
        if entry["model"] == "gefs" and not members:
            members = tuple(range(1, 31))
        specs.append(
            RunSpec(
                model=entry["model"],
                cycle_date=cycle_date,
                cycle_hour=cycle_hour,
                lead_time_hours=leads,
                members=members,
            )
        )
    if not specs:
        raise ValueError("Manifest 'runs' list is empty.")
    return specs


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


def _parse_variable(spec: str) -> VariableSpec:
    """Parse a ``CODE:NAME:UNIT[:SOURCE]`` CLI variable spec."""
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) not in (3, 4) or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "variable must be CODE:NAME:UNIT[:SOURCE], e.g. "
            "temperature_2m:2-Meter Temperature:°C:t2m"
        )
    code, name, unit = parts[0], parts[1], parts[2]
    source = parts[3] if len(parts) == 4 else None
    return VariableSpec(code=code, name=name, unit=unit, source_code=source)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weather-ingest",
        description="Ingest a NOAA GFS/GEFS forecast file into the platform.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest", help="download, parse, and record one or more forecast runs"
    )
    ingest.add_argument(
        "--model", nargs="+", choices=SUPPORTED_MODELS,
        help="NOAA model identifier(s) (gfs or gefs); one or more. Required "
             "unless --manifest is given.",
    )
    ingest.add_argument(
        "--cycle-date", nargs="+", type=date.fromisoformat,
        help="UTC run date(s) (ISO format, e.g. 2026-07-21); one or more. "
             "Required unless --manifest is given.",
    )
    ingest.add_argument(
        "--cycle-hour", nargs="+", type=int, choices=(0, 6, 12, 18),
        help="UTC run cycle hour(s); one or more. Required unless --manifest "
             "is given.",
    )
    ingest.add_argument(
        "--lead-time-hours", nargs="+", type=int,
        help="Forecast lead time offset(s) from cycle time (0-384); one or "
             "more. Required unless --manifest is given.",
    )
    ingest.add_argument(
        "--member", nargs="+", type=int,
        help="GEFS perturbation member identity/identities (1..30). For GEFS "
             "each member is downloaded as its own gepNN file and ingested "
             "independently; member identity is preserved regardless of "
             "completion order. Ignored for deterministic models.",
    )
    ingest.add_argument(
        "--manifest",
        default=None,
        help="Path to a JSON manifest describing an explicit list of runs "
             "({model, cycle_date, cycle_hour, lead_time_hours}); the "
             "authoritative source for complex batch jobs.",
    )
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run specifications without downloading or "
             "writing anything.",
    )
    ingest.add_argument(
        "--max-runs",
        type=int,
        default=16,
        help="Maximum number of run specifications a batch may expand to "
             "(default 16); exceeding it aborts to prevent an accidental "
             "huge job.",
    )
    ingest.add_argument(
        "--store",
        default=None,
        help="Zarr store path/URL of the run's cycle. When omitted it is "
             "derived from --model/--cycle-date/--cycle-hour "
             "(s3://weather-data/{model}/{date}/{hour}/cycle.zarr). All leads "
             "of a cycle are merged into this store, so pass the same --store "
             "for every lead. A supplied path that contradicts the forecast "
             "identity is rejected unless --allow-custom-store is set.",
    )
    ingest.add_argument(
        "--allow-custom-store",
        action="store_true",
        help="Accept an explicit --store that differs from the derived "
             "s3://weather-data/{model}/{date}/{hour}/cycle.zarr layout.",
    )
    ingest.add_argument(
        "--download-dir", default="downloads",
        help="Local directory the GRIB2 file is downloaded to (created if "
             "missing).",
    )
    ingest.add_argument(
        "--concurrency", type=int, default=4,
        help="Maximum number of forecast files fetched/ingested concurrently "
             "per run (default 4). Bounded so NOMADS is not flooded and disk "
             "staging stays bounded.",
    )
    ingest.add_argument("--center-id", default="noaa")
    ingest.add_argument("--version-string", default="v1.0")
    ingest.add_argument("--grid-id", default="global_025deg")
    ingest.add_argument(
        "--variable", action="append", default=None,
        type=_parse_variable, metavar="CODE:NAME:UNIT[:SOURCE]",
        help="Catalog variable metadata; repeatable. Defaults to the "
             "documented platform surface vocabulary (temperature_2m, "
             "precipitation_rate).",
    )
    return parser


def _run_ingest(args: argparse.Namespace) -> int:
    """Download and ingest the resolved forecast-run specifications.

    Each run spec is processed independently: a failure in one run (a bad
    lead file, a cross-cycle store, an upstream outage) does not abort the
    others. The overall exit status is non-zero if any run failed, so failures
    are never silently lost. ``--dry-run`` prints the resolved specs without
    downloading or writing.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0 on full success, non-zero if any run failed).
    """
    run_specs = expand_run_specs(args)
    if len(run_specs) > args.max_runs:
        raise SystemExit(
            f"Batch expands to {len(run_specs)} runs, exceeding --max-runs "
            f"({args.max_runs}). Refusing to run an accidental huge job; use "
            "a --manifest or raise --max-runs."
        )
    if args.dry_run:
        for spec in run_specs:
            store_path = validate_store_path(
                spec.store,
                spec.model,
                spec.cycle_date,
                spec.cycle_hour,
                allow_custom_store=spec.allow_custom_store,
            )
            print(
                f"dry-run: model={spec.model} "
                f"cycle={spec.cycle_time.strftime('%Y-%m-%dT%H:%MZ')} "
                f"leads={sorted(spec.lead_time_hours)} -> {store_path}"
            )
        return 0

    failed = 0
    for spec in run_specs:
        try:
            _ingest_one_run(spec, args)
        except (CycleStoreMismatchError, LeadTimeMismatchError, Exception) as exc:  # noqa: BLE001 - report every run failure
            failed += 1
            print(f"run FAILED: model={spec.model} cycle={spec.cycle_time}: {exc}")
    if failed:
        print(f"{failed}/{len(run_specs)} run(s) failed.")
        return 1
    print(f"Ingested {len(run_specs)} run(s) successfully.")
    return 0


def _destination_for(
    spec: RunSpec, args: argparse.Namespace, *, lead: int, member: int | None = None
) -> Path:
    """Return the staged download path for a (member,) lead file.

    The path encodes the model/cycle/lead (and member) so distinct files never
    collide in the staging directory, and re-ingestion of the same file
    overwrites its own staged copy.

    Args:
        spec: The run spec.
        args: Parsed CLI arguments (download dir).
        lead: Forecast lead time.
        member: GEFS member identity, or ``None`` for deterministic.

    Returns:
        The staging path.
    """
    if member is not None:
        name = (
            f"gep{member:02d}.t{spec.cycle_hour:02d}z.pgrb2s.0p25."
            f"f{lead:03d}"
        )
    else:
        name = (
            f"{spec.model}.t{spec.cycle_hour:02d}z.pgrb2.0p25.f{lead:03d}"
        )
    return Path(args.download_dir) / f"{name}.grib2"


def _cleanup_source(destination: Path) -> None:
    """Delete a successfully-ingested source file and its ``.idx``.

    Deletion is best-effort source-artifact cleanup. A failure to delete an
    already-successfully-ingested file must NOT invalidate the ingested data;
    the file is left in place for a later cleanup/retry and the failure logged.

    Args:
        destination: The staged GRIB2 path (the ``.idx`` sibling is also
            removed when present).
    """
    import logging

    logger = logging.getLogger(__name__)
    for path in (destination, Path(str(destination) + ".idx")):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # noqa: BLE001 - cleanup must never fail ingest
            logger.warning(
                "Failed to delete ingested source artifact %s: %s; data is "
                "safe, cleanup will be retried.",
                path,
                exc,
            )


def _ingest_one_run(spec: RunSpec, args: argparse.Namespace) -> None:
    """Download and ingest every lead/member of a single forecast run.

    Implements the approved region-write concurrency protocol:

    * retained-seed fresh-store initialization;
    * one wave-level EXCLUSIVE pre-update (run -> partial + UPDATING markers);
    * bounded region workers (SHARED gate + region locks + generation check);
    * one coalesced finalization after the wave drains.

    A failure in one file does not abort the others; the wave finalizer still
    runs (the run stays partial if any target is incomplete).

    Args:
        spec: The forecast-run specification.
        args: Parsed CLI arguments (download dir, catalog defaults, max
            concurrent files).

    Raises:
        CycleStoreMismatchError: If a lead's cycle mismatches the store.
        LeadTimeMismatchError: If a downloaded file's lead disagrees with the
            requested lead.
    """
    store_path = validate_store_path(
        spec.store,
        spec.model,
        spec.cycle_date,
        spec.cycle_hour,
        allow_custom_store=spec.allow_custom_store,
    )
    catalog_spec = _build_spec(spec, args, store_path)
    concurrency = max(1, int(getattr(args, "concurrency", 4)))
    failures: list[str] = []

    # Each (member, lead) work item, or just (lead) for deterministic.
    if spec.model != "gefs":
        items: list[tuple[int | None, int]] = [
            (None, lead) for lead in sorted(spec.lead_time_hours)
        ]
    else:
        items = [
            (member, lead)
            for member in sorted(spec.members)
            for lead in sorted(spec.lead_time_hours)
        ]

    # Retained seed: deterministically select the lowest (member, lead) item,
    # download + parse it exactly once before dispatch.
    seed_item = items[0]
    seed_member, seed_lead = seed_item

    async def _prepare_seed(connector: NOAAConnector) -> xr.Dataset:
        dest = _destination_for(spec, args, lead=seed_lead, member=seed_member)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        await connector.download(
            spec.model, spec.cycle_date, spec.cycle_hour, seed_lead, dest, member=seed_member
        )
        ds = _parse_for_ingest(dest)
        ds = _apply_variable_mapping(ds, catalog_spec.variables)
        ds = _normalize_canonical_units(ds, catalog_spec.variables)
        ds.attrs["model_id"] = catalog_spec.model_id
        _validate_requested_lead(ds, seed_lead)
        return ds

    async def _run_wave() -> str:
        from concurrent.futures import ThreadPoolExecutor

        from ingestion.core.cancel import await_all_workers_non_abandoning
        from ingestion.core.coordinator import (
            RunCoordinator,
            WaveRegion,
        )

        coordinator = RunCoordinator(
            catalog_spec,
            store_path,
            timeout_seconds=float(getattr(args, "lock_timeout", 30.0)),
        )
        cancel_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=concurrency)
        worker_futures = []
        engine = _catalog_session_factory()

        async with NOAAConnector() as connector:
            # 1. Retained seed.
            seed_dataset = await _prepare_seed(connector)

            # 2. Determine run id + same-cycle.
            run_id: str | None = None
            is_same_cycle = False
            with _catalog_session() as db:
                from ingestion.core.catalog import ModelRunRecord
                from sqlalchemy import select

                row = db.execute(
                    select(ModelRunRecord).where(
                        ModelRunRecord.zarr_store_path == store_path
                    )
                ).scalars().first()
                if row is not None:
                    run_id = str(row.id)
                    is_same_cycle = True

            # 3. Wave-level EXCLUSIVE pre-update (init + UPDATING markers).
            pre_conn = engine.connect()
            try:
                coordinator.initialize_run_store(
                    pre_conn,
                    seed_dataset=seed_dataset,
                    expected_leads=spec.lead_time_hours,
                    expected_members=spec.members,
                    run_id=run_id,
                    is_same_cycle=is_same_cycle,
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
                    cancel_event=cancel_event,
                )
            finally:
                pre_conn.close()

            # 4. Download every non-seed file on the event loop (bounded by a
            #    semaphore) before dispatching the sync workers.
            download_sem = asyncio.Semaphore(concurrency)

            async def _download_non_seed(member: int | None, lead: int) -> None:
                if (member, lead) == seed_item:
                    return
                dest = _destination_for(spec, args, lead=lead, member=member)
                async with download_sem:
                    try:
                        await connector.download(
                            spec.model, spec.cycle_date, spec.cycle_hour,
                            lead, dest, member=member,
                        )
                    except Exception as exc:  # noqa: BLE001 - report this file's failure
                        failures.append(f"{spec.model} member={member} lead={lead}: {exc}")

            await asyncio.gather(
                *[_download_non_seed(m, lead) for m, lead in items if (m, lead) != seed_item]
            )

            # 5. Dispatch region workers (sync, in the executor).
            generation_by_region = {r.region_id: r.generation for r in regions}

            def _run_region_worker(member: int | None, lead: int) -> None:
                worker_conn = engine.connect()
                try:
                    ds = _parse_for_ingest(
                        _destination_for(spec, args, lead=lead, member=member)
                    )
                    ds = _apply_variable_mapping(ds, catalog_spec.variables)
                    ds = _normalize_canonical_units(ds, catalog_spec.variables)
                    ds.attrs["model_id"] = catalog_spec.model_id
                    _validate_requested_lead(ds, lead)
                    region_id = _region_id_for(lead, member)
                    generation = generation_by_region.get(region_id)
                    if generation is None:
                        raise RuntimeError(f"no generation for region {region_id}")
                    coordinator.write_region_worker(
                        worker_conn,
                        dataset=ds,
                        member=member,
                        generation=generation,
                        expected_leads=spec.lead_time_hours,
                        expected_members=spec.members,
                    )
                finally:
                    worker_conn.close()

            # 6. Retained-seed writer uses the retained dataset (no re-parse).
            def _run_seed_region() -> None:
                worker_conn = engine.connect()
                try:
                    region_id = _region_id_for(seed_lead, seed_member)
                    generation = generation_by_region.get(region_id)
                    if generation is None:
                        raise RuntimeError(f"no generation for region {region_id}")
                    coordinator.write_region_worker(
                        worker_conn,
                        dataset=seed_dataset,
                        member=seed_member,
                        generation=generation,
                        expected_leads=spec.lead_time_hours,
                        expected_members=spec.members,
                    )
                finally:
                    worker_conn.close()

            # 7. Aggregate drain.
            loop = asyncio.get_event_loop()
            for member, lead in items:
                if (member, lead) == seed_item:
                    fut = loop.run_in_executor(executor, _run_seed_region)
                else:
                    fut = loop.run_in_executor(
                        executor, _run_region_worker, member, lead
                    )
                worker_futures.append(fut)
            results, cancelled = await await_all_workers_non_abandoning(
                worker_futures, cancel_event
            )
            # Record per-file failures.
            for idx, res in enumerate(results):
                if isinstance(res, BaseException):
                    m, lead = items[idx]
                    failures.append(f"{spec.model} member={m} lead={lead}: {res}")

        # 7. Coalesced finalization (after all worker Futures drained).
        fin_conn = engine.connect()
        try:
            run_id = _resolve_run_id(catalog_spec, store_path)
            status = coordinator.finalize_run(
                fin_conn,
                run_id=run_id,
                spec=catalog_spec,
                expected_leads=spec.lead_time_hours,
                expected_members=spec.members,
            )
        finally:
            fin_conn.close()
            engine.dispose()
            executor.shutdown(wait=False)
        return status

    status = asyncio.run(_run_wave())

    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(items)} file(s) failed for "
            f"model={spec.model} cycle={spec.cycle_time}: "
            + "; ".join(failures[:5])
            + ("; ..." if len(failures) > 5 else "")
        )
    print(
        f"Ingested {len(items)} region(s) for model={spec.model} "
        f"cycle={spec.cycle_time} ({status}) -> {store_path}"
    )


def _parse_for_ingest(destination: str | Path) -> xr.Dataset:
    """Parse a staged GRIB2 file (used by the CLI's region workers)."""
    from ingestion.providers.noaa.parser import parse_grib2

    return parse_grib2(os.fspath(destination))


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


#: Injectable engine factory for the CLI's catalog access. Production returns
#: the configured ingestion engine; tests replace this with an in-memory
#: SQLite engine so the CLI coordinator path can be exercised without PG.
_catalog_session_factory = _default_catalog_engine


def _resolve_run_id(spec: RunCatalogSpec, store_path: str) -> str:
    """Resolve the run id for a store path (create it if absent)."""
    from sqlalchemy import select

    from ingestion.core.catalog import ModelRunRecord, record_run

    with _catalog_session() as db:
        row = db.execute(
            select(ModelRunRecord).where(ModelRunRecord.zarr_store_path == store_path)
        ).scalars().first()
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
    args: argparse.Namespace,
    store_path: str,
) -> RunCatalogSpec:
    """Build the run's catalog metadata from a run spec + CLI defaults."""
    center_name, center_country = _CENTER_METADATA[args.center_id]
    model_name, is_ensemble = _MODEL_METADATA[spec.model]
    grid_name, grid_resolution_km = _GRID_METADATA.get(
        args.grid_id, (args.grid_id, 0.0)
    )
    variables = (
        tuple(args.variable) if args.variable is not None else DEFAULT_VARIABLES
    )
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
        # The expected lead set is the run's full requested lead list; run-level
        # readiness compares committed leads against it. For GEFS the expected
        # member set is the full perturbation set gep01..gep30.
        expected_lead_time_hours=tuple(spec.lead_time_hours),
        expected_members=tuple(spec.members),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        return _run_ingest(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
